"""Actual final validation receipts, immutable-production gates and restored-arm probes."""

import datetime
import json
import subprocess
import time

import torch
from context_ab_core import Arm, trees_equal
from context_distribution_study import ROOT, Store, consume, parent_inputs, read, sha, tensor_hash
from context_study_metrics import require
from context_training_ab import AB, ARMS, CHECKPOINTS, TOOLS, verify_protected
from training_progress import publish_progress

from unified_edge.training.checkpoint import load_checkpoint
from unified_edge.training.config import canonical_hash
from unified_edge.training.device import configure_device

FILES = (
    *TOOLS,
    "scripts/finish_context_training_ab.py",
    "scripts/context_phase_summary.py",
    "scripts/finish_context_distribution_study.py",
    "scripts/validate_context_ab.py",
    "tests/test_context_ab_report.py",
    "tests/test_training.py",
    "tests/test_context_distribution_study.py",
    "tests/test_context_study_phases.py",
)


def measurement_hashes():
    return {
        p.relative_to(AB).as_posix(): sha(p)
        for p in sorted(AB.rglob("*.json"))
        if p.name != "progress.json" and "validation-closeout" not in p.parts
    }


def verify_receipt(path):
    envelope = read(path)
    value = envelope["value"]
    require(envelope["sha256"] == canonical_hash(value), "validation receipt corrupt")
    require(value["schema"] == "context-ab-validation-1", "validation schema mismatch")
    require(value["integrity"]["status"] == "PASS", "final integrity failed")
    require(
        value["measurement_hashes"] == measurement_hashes(), "validation measurement evidence stale"
    )
    require(
        value["tested_files"] == {p: sha(ROOT / p) for p in FILES}, "validation code evidence stale"
    )
    required = {
        "pytest",
        "ruff",
        "format",
        "compileall",
        "cpu_dependencies",
        "cuda_dependencies",
        "diff",
    }
    require(
        {v["name"] for v in value["commands"]} == required
        and len(value["commands"]) == len(required),
        "missing final validation command",
    )
    require(all(v["returncode"] == 0 for v in value["commands"]), "final validation failed")
    return dict(value, receipt_sha256=sha(path))


@torch.inference_mode()
def integrity():
    binding = read(AB / "binding.json")
    root = Store(AB, binding)
    require((root.get("integrity_final") or {}).get("status") == "PASS", "experiment not complete")
    parent, resolved, config, _, _, _, full = parent_inputs()
    verify_protected(binding, full)
    configure_device("cuda:0")
    torch.set_num_threads(config.cpu_threads)
    torch.use_deterministic_algorithms(True)
    evidence = {}
    for name, length in ARMS:
        arm = Arm(parent, resolved, config, length, canonical_hash(binding), name, "cuda:0")
        identities = {}
        for step in CHECKPOINTS:
            path = AB / name / f"checkpoints/update_{step:06d}"
            arm.restore(path, AB)
            saved = load_checkpoint(path)
            require(
                trees_equal(arm.optimizer.state_dict(), saved["optimizer"]),
                "optimizer restore mismatch",
            )
            require(
                trees_equal(arm.scheduler.state_dict(), saved["scheduler"]),
                "scheduler restore mismatch",
            )
            require(trees_equal(arm.rng, saved["rng"]), "RNG restore mismatch")
            if step == 0:
                require(
                    trees_equal(saved["model"], parent["model"])
                    and trees_equal(saved["optimizer"], parent["optimizer"]),
                    "initial parent clone mismatch",
                )
            identities[str(step)] = {
                "state_sha256": sha(path / "state.pt"),
                "manifest_sha256": sha(path / "manifest.json"),
                "parameter_sha256": tensor_hash(arm.model.state_dict()),
            }
        evaluation = AB / "evaluation" / name / "update_000512"
        store = Store(evaluation, read(evaluation / "binding.json"))
        require(
            identities["512"]["parameter_sha256"] == store.get("complete")["parameter_sha256"],
            "saved trained model differs from evaluated model",
        )
        arm.model.eval()
        for i in range(6):
            expected = store.get(f"generation_{i}")
            state = consume(arm.model, bytes.fromhex(expected["prefix_hex"]))
            output = []
            for _ in range(64):
                byte = arm.model.predict(state)[0, :256].argmax().item()
                output.append(byte)
                state = arm.model.consume(torch.tensor([byte], device="cuda:0"), state)
            require(bytes(output).hex() == expected["output_hex"], "restored generation differs")
        evidence[name] = {
            "checkpoints": identities,
            "all_six_restored_generations_equal": True,
            "optimizer_scheduler_rng_restore": "EXACT",
            "finite_loadable_generatable": True,
        }
        del arm
    verify_protected(binding, full)
    return {
        "status": "PASS",
        "arms": evidence,
        "protected_inventory": binding["protected_inventory"],
        "source_sha256": binding["code"]["source_sha256"],
        "test": "SEALED; streaming integrity hashes only",
    }


def run():
    before = integrity()
    hashes = measurement_hashes()
    files = {p: sha(ROOT / p) for p in FILES}
    tests = [p for p in FILES if p.startswith("tests/")]
    commands = [
        (
            "pytest",
            [".venv/Scripts/python.exe", "-m", "pytest", *tests, "-q", "-p", "no:cacheprovider"],
        ),
        ("ruff", [".venv/Scripts/python.exe", "-m", "ruff", "check", *FILES]),
        ("format", [".venv/Scripts/python.exe", "-m", "ruff", "format", "--check", *FILES]),
        ("compileall", [".venv/Scripts/python.exe", "-m", "compileall", "-q", *FILES]),
        ("cpu_dependencies", [".venv/Scripts/python.exe", "-m", "pip", "check"]),
        ("cuda_dependencies", [".venv-cuda/Scripts/python.exe", "-m", "pip", "check"]),
        ("diff", ["git", "-c", "safe.directory=E:/Theai", "diff", "--check"]),
    ]
    results = []
    for name, args in commands:
        start = time.perf_counter()
        stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        p = subprocess.run(
            args,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        results.append(
            {
                "name": name,
                "args": args,
                "returncode": p.returncode,
                "stdout": p.stdout,
                "stderr": p.stderr,
                "seconds": time.perf_counter() - start,
                "started_utc": stamp,
            }
        )
        print(f"CHECK {name}: {p.returncode}", flush=True)
    # No second inference pass; verify source/files/production after tests without repeating probes.
    binding = read(AB / "binding.json")
    _, _, _, _, _, _, full = parent_inputs()
    verify_protected(binding, full)
    require(
        hashes == measurement_hashes() and files == {p: sha(ROOT / p) for p in FILES},
        "evidence changed during validation",
    )
    value = {
        "schema": "context-ab-validation-1",
        "integrity": before,
        "measurement_hashes": hashes,
        "tested_files": files,
        "commands": results,
    }
    envelope = {"value": value, "sha256": canonical_hash(value)}
    root = AB / "validation-closeout"
    root.mkdir(exist_ok=True)
    path = root / f"attempt_{time.time_ns()}.json"
    path.write_text(json.dumps(envelope, indent=2) + "\n", encoding="utf-8")
    publish_progress(root / "commands.json", envelope)
    require(all(r["returncode"] == 0 for r in results), "final checks failed; receipt retained")


if __name__ == "__main__":
    run()
