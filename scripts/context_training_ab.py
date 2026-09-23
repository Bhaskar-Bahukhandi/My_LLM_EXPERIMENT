"""Prepare/run a fixed, disposable 32-versus-64 context-training A/B; never production."""

import argparse
import copy
import hashlib
import json
import subprocess
import time
from types import SimpleNamespace

import torch
from context_ab_core import QUOTAS, Arm, independent, matched_targets, select_blocks, trees_equal
from context_ab_evaluation import (
    distribution_panel,
    generation_panel,
    matched_panel,
    phase_panel,
    segmented_panel,
)
from context_distribution_study import (
    DATA,
    PARAMETER_SHA,
    PRODUCTION,
    ROOT,
    STUDY,
    Store,
    parent_inputs,
    read,
    sha,
    tensor_hash,
    verify_corpus,
    verify_inventory,
)
from context_study_metrics import PROTOCOL, require, validation_documents
from context_study_phases import selections

from unified_edge.training.config import canonical_hash
from unified_edge.training.device import configure_device
from unified_edge.training.trainer import code_identity, environment

AB = ROOT / "data/context-training-ab-2m-v1"
TOOLS = (
    "scripts/context_ab_core.py",
    "scripts/context_ab_evaluation.py",
    "scripts/context_training_ab.py",
    "tests/test_context_training_ab.py",
    "scripts/context_distribution_study.py",
    "scripts/context_study_metrics.py",
    "scripts/context_study_phases.py",
    "scripts/training_progress.py",
)
CHECKPOINTS = (0, 128, 256, 512)
ARMS = (("arm-a-32", 32), ("arm-b-64", 64))


def protected_inventory(before):
    result = dict(before)
    for suffix in ("md", "json"):
        path = ROOT / f"reports/context_distribution_study_2m.{suffix}"
        result[path.relative_to(ROOT).as_posix()] = sha(path)
    checkpoints = read(ROOT / "reports/full_training_2m_stage_a.json")["checkpoint_identities"]
    require(len(checkpoints) == 10, "accepted checkpoint count changed")
    actual = {
        p.relative_to(PRODUCTION).as_posix()
        for p in PRODUCTION.glob("attempt-*/checkpoints/step_*")
    }
    require(actual == {v["path"] for v in checkpoints}, "production checkpoint set changed")
    for row in checkpoints:
        for name, key in (("state.pt", "state_sha256"), ("manifest.json", "manifest_sha256")):
            p = PRODUCTION / row["path"] / name
            require(sha(p) == row[key], "accepted production checkpoint changed")
            result[p.relative_to(ROOT).as_posix()] = sha(p)
    return result


def verify_protected(binding, full):
    verify_inventory(binding["parent_before"])
    require(
        protected_inventory(binding["parent_before"]) == binding["protected_inventory"],
        "protected study/production artifact changed",
    )
    require(verify_corpus(full) == binding["split_hashes"], "corpus changed")
    require(
        code_identity()["source_sha256"] == binding["code"]["source_sha256"], "model source changed"
    )
    for name, digest in binding["tools"].items():
        require(sha(ROOT / name) == digest, "bound experiment tool changed: " + name)


def prepare():
    require(not AB.exists(), "study already exists; use --run to recover verified artifacts")
    saved, resolved, config, manifest, _, before, full = parent_inputs()
    accepted = read(ROOT / "reports/context_distribution_study_2m.json")
    require(
        accepted["recommendation"] == "MORE_EVIDENCE_REQUIRED",
        "unexpected accepted context decision",
    )
    docs = [d for d in manifest.documents if d.split == "train"]
    schedule = select_blocks(docs, lambda d: (DATA / d.path).read_bytes())
    require(len(schedule) == 512, "schedule quota mismatch")
    anchors = selections(read(STUDY / "binding.json")["anchors"])
    matched = [a for i, a in enumerate(anchors) if i % 32 < 8]
    small = [
        {"path": a["path"], "domain": a["domain"], "start": a["offset"] - 128, "end": a["offset"]}
        for i, a in enumerate(anchors)
        if i % 32 < 4
    ]
    positions = read(STUDY / "position-distributions/binding.json")["selections"]
    receipt = []
    for name, args in (
        (
            "pytest",
            [
                ".venv/Scripts/python.exe",
                "-m",
                "pytest",
                "tests/test_context_training_ab.py",
                "-q",
                "-p",
                "no:cacheprovider",
            ],
        ),
        ("ruff", [".venv/Scripts/python.exe", "-m", "ruff", "check", *TOOLS]),
        ("format", [".venv/Scripts/python.exe", "-m", "ruff", "format", "--check", *TOOLS]),
    ):
        start = time.perf_counter()
        completed = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=False)
        receipt.append(
            {
                "name": name,
                "args": args,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "seconds": time.perf_counter() - start,
            }
        )
    # Keep failed preflight evidence outside the not-yet-bound experiment root.
    preflight = ROOT / "data/context-training-ab-preflight-v1"
    preflight.mkdir(exist_ok=True)
    attempt = preflight / f"attempt_{time.time_ns()}.json"
    attempt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    require(all(r["returncode"] == 0 for r in receipt), "pretraining tests/static checks failed")
    binding = {
        "schema": "context-training-ab-1",
        "classification": "DISPOSABLE_AB_ONLY",
        "parent_before": before,
        "protected_inventory": protected_inventory(before),
        "parameter_sha256": PARAMETER_SHA,
        "code": code_identity(),
        "tools": {p: sha(ROOT / p) for p in TOOLS},
        "model_config": resolved.to_dict(),
        "training_config": config.to_dict(),
        "parent_scheduler": saved["scheduler"],
        "parent_optimizer_counts": saved["optimizer_update_counts"],
        "parent_checkpoint_state_sha256": before[
            "data/full-training-2m-v1/attempt-003/checkpoints/step_005000/state.pt"
        ],
        "split_hashes": full["split_sha256"],
        "schedule_sha256": canonical_hash(schedule),
        "schedule_policy": (
            "nonoverlapping document-aligned 128-byte blocks; lowest domain hash "
            "keys; independent order hash"
        ),
        "domain_quotas": QUOTAS,
        "arm_order": [name for name, _ in ARMS],
        "updates": 512,
        "bytes_per_update": 128,
        "total_target_bytes_per_arm": 65536,
        "checkpoints": list(CHECKPOINTS),
        "normalization": "summed NLL over exactly128 targets divided by128; batches4x32 versus2x64",
        "evaluation": {
            "phase_anchors": anchors,
            "matched_anchors": matched,
            "small_validation": small,
            "position_selections": positions,
            "intermediate_positions": [p for p in positions if p["length"] == 32],
            "phase_grid": {str(r): [n + r for n in (32, 64, 128, 248)] for r in range(8)},
            "phase_batch_size": 8,
            "forensics_at": [0, 256, 512],
            "small_panel_at": [128, 256, 512],
            "full_validation_at": 512,
            "full_lengths": [32, 64],
            "prompts_hex": PROTOCOL["generation_prompts_hex"],
            "generation_bytes": 64,
            "baseline": (
                "reuse accepted completed parent measurements only after exact "
                "clone/hash/source equivalence"
            ),
            "bootstrap": (
                "2000 paired percentile draws seed1730; aggregate resamples phase-averaged anchors"
            ),
            "unchanged_nll_tolerance": 1e-5,
        },
        "decision_policy": {
            "scope": (
                "Project-local exploratory decision margins, not universal "
                "quality/SOTA thresholds; single seed, anchors clustered within "
                "documents."
            ),
            "review_64": (
                "ALL gates: B aggregate64+r minus32+r CI upper<0 and mean<-1e-5; "
                "B-A history-benefit difference CI upper<0; B32 validation no "
                "more than0.01 NLL worse than A or parent; no domain B-A "
                "regression>0.03; severe-context fraction no more than0.05 above "
                "A/parent; control mass increase<=0.001; B training<=2x A time "
                "and <=1800seconds."
            ),
            "continue_32": (
                "A equals/exceeds B held-out quality, or B lacks beneficial "
                "history use, or B regressions/cost lack quality gain, unless "
                "both gains<0.01 NLL."
            ),
            "freeze_2m": (
                "Both full32 improvements versus parent<0.01 NLL and no "
                "demonstrated beneficial history; infrastructure proof already "
                "achieved; scaling requires separate review."
            ),
            "ambiguous": (
                "MORE_EVIDENCE_REQUIRED when gates/effects conflict or "
                "uncertainty prevents a clear interpretation."
            ),
            "promotion": "NEVER AUTOMATIC; neither branch enters production",
        },
        "recovery_policy": (
            "Restore latest completed experiment checkpoint; compare semantic "
            "metrics for any durable updates replayed; append separate physical "
            "replay accounting. Never count replay as new logical exposure."
        ),
        "timing_policy": (
            "Sequential A then B; quality primary; wall time secondary with "
            "cache/thermal/order confounding; evaluation excluded from training "
            "timing"
        ),
        "pretraining_receipt_sha256": canonical_hash(receipt),
    }
    store = Store(AB, binding)
    store.put("schedule", {"blocks": schedule})
    store.put(
        "pretraining_validation", {"commands": receipt, "tested_files_sha256": binding["tools"]}
    )
    print("PREPARED: 512 TRAIN-only matched blocks; tests pass; immutable policy bound", flush=True)


def evaluation(arm, checkpoint, binding, payloads, manifest):
    root = AB / "evaluation" / arm.arm / f"update_{arm.step:06d}"
    store = Store(
        root,
        {
            "study_binding_sha256": canonical_hash(binding),
            "arm": arm.arm,
            "update": arm.step,
            "checkpoint_state_sha256": sha(checkpoint / "state.pt"),
        },
    )
    if store.get("complete") is not None:
        return
    before = tensor_hash(arm.model.state_dict())
    arm.model.eval()
    panel = binding["evaluation"]
    small = [
        (
            SimpleNamespace(path=r["path"], domain=r["domain"]),
            payloads[r["path"]][r["start"] : r["end"]],
        )
        for r in panel["small_validation"]
    ]
    for length in (32, 64):
        with torch.inference_mode():
            segmented_panel(arm.model, small, length, store, "small")
    phase_panel(arm.model, panel["phase_anchors"], payloads, store, arm.step in (256, 512))
    matched_panel(arm.model, panel["matched_anchors"], payloads, store)
    generation_panel(arm.model, store)
    distribution_panel(
        arm.model,
        panel["position_selections"] if arm.step == 512 else panel["intermediate_positions"],
        payloads,
        store,
    )
    if arm.step == 512:
        docs = validation_documents(manifest, payloads)
        for length in (32, 64):
            value = segmented_panel(arm.model, docs, length, store, "full")
            require(value["count"] == 500000, "full validation target count mismatch")
    require(before == tensor_hash(arm.model.state_dict()), "evaluation changed branch weights")
    store.put("complete", {"status": "PASS", "parameter_sha256": before, "test": "SEALED"})
    print(f"MILESTONE: {arm.arm} evaluation {arm.step} complete", flush=True)


def run():
    binding = read(AB / "binding.json")
    store = Store(AB, binding)
    saved, resolved, config, manifest, payloads, _, full = parent_inputs()
    verify_protected(binding, full)
    require(
        store.get("pretraining_validation")["tested_files_sha256"] == binding["tools"],
        "pretraining evidence mismatch",
    )
    schedule = store.get("schedule")["blocks"]
    require(canonical_hash(schedule) == binding["schedule_sha256"], "schedule mutation")
    configure_device("cuda:0")
    torch.set_num_threads(config.cpu_threads)
    torch.use_deterministic_algorithms(True)
    require(
        environment(config) == read(STUDY / "binding.json")["environment"],
        "GPU environment changed",
    )
    arms = [
        Arm(saved, resolved, config, length, canonical_hash(binding), name, "cuda:0")
        for name, length in ARMS
    ]
    independent(saved, *(a.state() for a in arms))
    require(
        all(sum(p.numel() for p in a.model.parameters()) == 1929579 for a in arms),
        "parameter count changed",
    )
    require(
        trees_equal(arms[0].rng, arms[1].rng)
        and trees_equal(arms[0].rng, {k: saved[k] for k in arms[0].rng}),
        "initial RNG mismatch",
    )
    require(
        all(tensor_hash(a.model.state_dict()) == PARAMETER_SHA for a in arms),
        "initial parameter mismatch",
    )
    require(
        trees_equal(arms[0].optimizer.state_dict(), arms[1].optimizer.state_dict())
        and trees_equal(arms[0].optimizer.state_dict(), saved["optimizer"]),
        "initial optimizer mismatch",
    )
    require(
        all(trees_equal(a.scheduler.state_dict(), saved["scheduler"]) for a in arms),
        "scheduler mismatch",
    )
    if store.get("initial_equivalence") is None:
        # Cheap equality probe; completed full parent validation is reused by exact identity.
        target = torch.tensor([list(next(iter(payloads.values()))[:32])], device="cuda:0")
        with torch.inference_mode():
            require(
                torch.equal(arms[0].model(target), arms[1].model(target)),
                "baseline clone outputs differ",
            )
        store.put(
            "initial_equivalence",
            {
                "status": "PASS",
                "independent_storage": True,
                "model_optimizer_scheduler_rng_equal": True,
                "accepted_parent_full32_nll": 2.362495391641617,
                "baseline_evidence_sha256": sha(
                    ROOT / "reports/context_distribution_study_2m.json"
                ),
            },
        )
    train_payloads = {}
    doc_map = {d.path: d for d in manifest.documents if d.split == "train"}

    def block_bytes(row):
        require(row["document"] in doc_map, "training block references non-TRAIN document")
        if row["document"] not in train_payloads:
            raw = (DATA / row["document"]).read_bytes()
            require(
                hashlib.sha256(raw).hexdigest() == doc_map[row["document"]].sha256,
                "TRAIN file changed",
            )
            train_payloads[row["document"]] = raw
        raw = train_payloads[row["document"]][row["start"] : row["end"]]
        require(
            len(raw) == 128 and hashlib.sha256(raw).hexdigest() == row["sha256"],
            "training block changed",
        )
        require(
            torch.equal(matched_targets(raw, 32).flatten(), matched_targets(raw, 64).flatten()),
            "same-byte gate failed",
        )
        return raw

    # Both no-update GPU forms pass before either arm trains.
    for arm in arms:
        key = f"preflight_{arm.length}"
        if store.get(key) is None:
            initial = copy.deepcopy(arm.state())
            evidence = arm.update(block_bytes(schedule[0]), advance=False)
            require(trees_equal(arm.state(), initial), "no-update preflight changed clone state")
            require(
                evidence["memory"]["cuda"]["peak_allocated_bytes"] < 1024**3,
                "preflight memory gate",
            )
            store.put(key, evidence)
    for arm in arms:
        Store(AB / arm.arm, {"study_binding_sha256": canonical_hash(binding), "arm": arm.arm})
        zero = AB / arm.arm / "checkpoints/update_000000"
        if not zero.exists():
            arm.save(zero, AB)
    for arm in arms:
        armstore = Store(
            AB / arm.arm, {"study_binding_sha256": canonical_hash(binding), "arm": arm.arm}
        )
        observed = sorted(
            int(p.stem.split("_")[-1]) for p in (AB / arm.arm).glob("observation_*.json")
        )
        require(observed == list(range(1, len(observed) + 1)), "observation journal has gaps")
        for milestone in (128, 256, 512):
            checkpoint = AB / arm.arm / f"checkpoints/update_{milestone:06d}"
            if checkpoint.exists():
                arm.restore(checkpoint, AB)
            else:
                available = sorted((AB / arm.arm / "checkpoints").glob("update_*"))
                arm.restore(available[-1], AB)
                durable = sorted((AB / arm.arm).glob("observation_*.json"))
                max_observed = max((int(p.stem.split("_")[-1]) for p in durable), default=0)
                recovery = sum(1 for _ in (AB / arm.arm).glob("recovery_*.json"))
                if max_observed > arm.step:
                    armstore.put(
                        f"recovery_{recovery:03d}",
                        {
                            "checkpoint_update": arm.step,
                            "durable_updates": max_observed,
                            "planned_replay": max_observed - arm.step,
                            "inflight_unjournaled_update": (
                                "unknown, at most one; excluded from logical counts"
                            ),
                        },
                    )
                while arm.step < milestone:
                    record = arm.update(block_bytes(schedule[arm.step]))
                    name = f"observation_{arm.step:06d}"
                    previous = armstore.get(name)
                    if previous is None:
                        armstore.put(name, record)
                    else:
                        require(
                            previous["semantic"] == record["semantic"],
                            "deterministic replay mismatch",
                        )
                        armstore.put(f"replay_{recovery:03d}_{arm.step:06d}", record)
                    if arm.step % 64 == 0:
                        print(
                            f"MILESTONE: {arm.arm} {arm.step}/512 updates; "
                            f"{arm.step * 128} target bytes",
                            flush=True,
                        )
                arm.save(checkpoint, AB)
            evaluation(arm, checkpoint, binding, payloads, manifest)
            verify_protected(binding, full)
    require(
        all(a.step == 512 and a.scheduler.completed == 5512 for a in arms), "final update mismatch"
    )
    require(
        trees_equal(arms[0].scheduler.state_dict(), arms[1].scheduler.state_dict()),
        "final schedule mismatch",
    )
    verify_protected(binding, full)
    if store.get("integrity_final") is None:
        store.put(
            "integrity_final",
            {
                "status": "PASS",
                "parent_unchanged": True,
                "production_updates": 0,
                "test": "SEALED; streaming integrity hashes only",
            },
        )
    print("MILESTONE: disposable A/B complete; no promotion", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    require(args.prepare != args.run, "choose exactly one of --prepare/--run")
    prepare() if args.prepare else run()
