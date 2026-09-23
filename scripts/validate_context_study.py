"""Run and bind actual lightweight closeout commands after completed study measurements."""

import datetime
import json
import subprocess
import time

from context_distribution_study import (
    PRODUCTION,
    ROOT,
    STUDY,
    Store,
    parent_inputs,
    read,
    sha,
    verify_inventory,
)
from context_study_metrics import require
from finish_context_distribution_study import VALIDATION_FILES, measurement_sets
from training_progress import publish_progress

from unified_edge.training.config import canonical_hash


def integrity():
    saved, _, _, _, _, before, _ = parent_inputs()
    require(sum(v.numel() for v in saved["model"].values()) == 1929579, "parameter count changed")
    original = read(STUDY / "binding.json")
    require(before == original["parent_before"], "parent inventory mismatch")
    counts = {}
    for relative in ("", "position-distributions", "phase-matched"):
        root = STUDY / relative
        store = Store(root, read(root / "binding.json"))
        require(store.get("integrity_final")["status"] == "PASS", "measurement incomplete")
        counts[relative or "main"] = len(read(root / "progress.json")["completed"])
    recovery = read(STUDY / "recovery-v1/reconciliation.json")
    for name, digest in recovery["preserved_measurements"].items():
        require(sha(STUDY / name) == digest, "recovered measurement changed")
    for name, digest in original["tools"].items():
        require(sha(ROOT / "scripts" / name) == digest, "original producer changed")
    checkpoints = read(ROOT / "reports/full_training_2m_stage_a.json")["checkpoint_identities"]
    actual = {
        p.relative_to(PRODUCTION).as_posix()
        for p in PRODUCTION.glob("attempt-*/checkpoints/step_*")
    }
    require(actual == {v["path"] for v in checkpoints}, "production checkpoint set changed")
    for row in checkpoints:
        for name, key in (("state.pt", "state_sha256"), ("manifest.json", "manifest_sha256")):
            require(
                sha(PRODUCTION / row["path"] / name) == row[key], "production checkpoint changed"
            )
    verify_inventory(before)
    return {
        "status": "PASS",
        "parent_inventory": before,
        "measurement_counts": counts,
        "measurement_set_sha256": measurement_sets(),
        "preserved_recovery_files": len(recovery["preserved_measurements"]),
        "production_checkpoint_count": len(checkpoints),
        "test": "SEALED; streaming hash-only integrity",
    }


def run():
    initial = integrity()
    cpu = ".venv/Scripts/python.exe"
    cuda = ".venv-cuda/Scripts/python.exe"
    tests = [p for p in VALIDATION_FILES if p.startswith("tests/")]
    commands = [
        ("pytest", [cpu, "-m", "pytest", *tests, "-q", "-p", "no:cacheprovider"]),
        ("ruff", [cpu, "-m", "ruff", "check", *VALIDATION_FILES]),
        ("format", [cpu, "-m", "ruff", "format", "--check", *VALIDATION_FILES]),
        ("compileall", [cpu, "-m", "compileall", "-q", *VALIDATION_FILES]),
        ("cpu_dependencies", [cpu, "-m", "pip", "check"]),
        ("cuda_dependencies", [cuda, "-m", "pip", "check"]),
        ("diff", ["git", "-c", "safe.directory=E:/Theai", "diff", "--check"]),
    ]
    hashes = {p: sha(ROOT / p) for p in VALIDATION_FILES}
    rows = []
    for name, args in commands:
        started = datetime.datetime.now(datetime.timezone.utc).isoformat()
        start = time.perf_counter()
        process = subprocess.run(
            args,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        rows.append(
            {
                "name": name,
                "args": args,
                "started_utc": started,
                "seconds": time.perf_counter() - start,
                "returncode": process.returncode,
                "stdout": process.stdout,
                "stderr": process.stderr,
            }
        )
        print(f"CHECK {name}: exit {process.returncode}", flush=True)
    require(
        hashes == {p: sha(ROOT / p) for p in VALIDATION_FILES}, "tested files changed during checks"
    )
    final = integrity()
    require(initial == final, "integrity changed during validation")
    value = {
        "schema": "study-validation-1",
        "final_integrity": final,
        "commands": rows,
        "files_sha256": hashes,
        "measurement_integrity": {
            "main": sha(STUDY / "integrity_final.json"),
            "positions": sha(STUDY / "position-distributions/integrity_final.json"),
            "phases": sha(STUDY / "phase-matched/integrity_final.json"),
            "architecture_sanity": sha(STUDY / "phase-matched/architecture_sanity.json"),
        },
    }
    root = STUDY / "validation-closeout"
    root.mkdir(exist_ok=True)
    # Preserve every actual attempt, including failures; the finalizer rejects failures.
    envelope = {"value": value, "sha256": canonical_hash(value)}
    receipt = root / (
        "attempt_"
        + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        + ".json"
    )
    receipt.write_text(json.dumps(envelope, indent=2) + "\n", encoding="utf-8")
    publish_progress(root / "commands.json", envelope)
    require(
        all(v["returncode"] == 0 for v in rows), "closeout command failed; inspect saved receipt"
    )


if __name__ == "__main__":
    run()
