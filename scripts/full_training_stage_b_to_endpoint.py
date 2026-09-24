"""Additive continuation of the frozen production trajectory; never an A/B promotion."""

import argparse
import gc
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path

import torch
from context_ab_core import trees_equal
from context_ab_evaluation import distribution_panel, generation_panel, segmented_panel
from context_distribution_study import (
    DATA,
    PARENT,
    PRODUCTION,
    ROOT,
    STUDY,
    Store,
    parent_inputs,
    read,
    sha,
    tensor_hash,
    verify_corpus,
)
from context_study_metrics import distribution_summary, require, validation_documents
from finish_context_distribution_study import whitespace_summary
from full_training_stage_a import cpu
from training_progress import compare_replay, journal, publish_progress, semantic_row

from unified_edge.training.checkpoint import load_checkpoint
from unified_edge.training.config import TrainingConfig, canonical_hash
from unified_edge.training.data import BatchStream, WindowDataset
from unified_edge.training.device import cuda_rng_state
from unified_edge.training.experiment import semantic_metrics
from unified_edge.training.optimization import finite_parameters
from unified_edge.training.trainer import Trainer, code_identity

CONT = PRODUCTION / "continuation-v1"
EVAL = ROOT / "data/full-training-2m-endpoint-evaluation-v1"
END = 78167
VALIDATIONS = (10000, 20000, 30000, 40000, 50000, 60000, 70000, END)
CHECKPOINTS = tuple(sorted(set(VALIDATIONS) | {15000}))
PROOFS = (20000, 40000, 60000, END)
TOOLS = (
    "scripts/full_training_stage_b_to_endpoint.py",
    "scripts/full_training_stage_a.py",
    "scripts/training_progress.py",
    "scripts/context_ab_core.py",
    "scripts/context_ab_evaluation.py",
    "scripts/context_distribution_study.py",
    "scripts/context_study_metrics.py",
    "scripts/context_study_phases.py",
    "scripts/finish_context_distribution_study.py",
    "tests/test_production_continuation.py",
    "scripts/validate_production_continuation.py",
)
DOMAIN_POLICY = {
    "version": "parent-relative-consecutive-1",
    "material_nll": 0.03,
    "consecutive_major_points": 2,
    "rule": "Both consecutive domain NLLs exceed step5000 by >0.03; current domain "
    "NLL exceeds previous; aggregate NLL at both points is below step5000. "
    "Stop at the already-saved current major checkpoint. Threshold reuses the "
    "accepted A/B domain margin, frozen before production continuation.",
}


def write_new(path, value):
    path = Path(path)
    require(not path.exists(), f"immutable artifact already exists: {path.name}")
    publish_progress(path, value)


def append(path, value):
    with Path(path).open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def domain_watch(parent, previous, current, policy=DOMAIN_POLICY):
    require(policy == DOMAIN_POLICY, "domain policy mismatch")
    if previous is None:
        return []
    return [
        d
        for d, p in parent["domains"].items()
        if all(
            v["domains"][d]["nll"] - p["nll"] > policy["material_nll"] and v["nll"] < parent["nll"]
            for v in (previous, current)
        )
        and current["domains"][d]["nll"] > previous["domains"][d]["nll"]
    ]


def expected_cursor(step, windows=312666):
    draws = step * 4
    return ((draws - 1) // windows, (draws - 1) % windows + 1) if draws else (0, 0)


def audit_rows(rows, start=5000, initial_bytes=639613, windows=312666):
    total = initial_bytes
    for step, row in enumerate(rows, start + 1):
        semantic_row(row)
        require(row["step"] == step, "continuation journal gap/duplicate")
        require(
            type(row["valid_target_count"]) is int
            and 4 <= row["valid_target_count"] <= 128
            and step <= END,
            "invalid production target count/endpoint",
        )
        total += row["valid_target_count"]
        epoch, offset = expected_cursor(step, windows)
        require(
            row["micro_step"] == 2 * step and row["examples_seen"] == 4 * step,
            "continuation counter mismatch",
        )
        require(row["bytes_seen"] == total, "continuation byte gap")
        require(
            row["cursor"]["epoch"] == epoch and row["cursor"]["offset"] == offset,
            "continuation cursor mismatch",
        )
        require(
            row["optimizer_update_counts"] == [step] * 56 and row["scheduler_completed"] == step,
            "optimizer/scheduler discontinuity",
        )
        require(row["finite_gradients"] is True, "nonfinite journal update")
        require(
            row["summed_nll"] == row["loss"] * row["valid_target_count"], "loss accounting mismatch"
        )
    return total


def read_rows(path):
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    require(not text or text.endswith("\n"), "partial journal tail; preserve and review")
    return [json.loads(line) for line in text.splitlines()]


def compare_row(actual, expected):
    compare_replay(actual, expected)
    for key in (
        "cursor",
        "optimizer_update_counts",
        "scheduler_completed",
        "finite_gradients",
        "summed_nll",
    ):
        require(actual[key] == expected[key], f"replay mismatch: {key}")


def verify_binding(binding, full):
    for name, digest in binding["protected"].items():
        require(sha(ROOT / name) == digest, f"protected artifact changed: {name}")
    for name, digest in binding["tools"].items():
        require(sha(ROOT / name) == digest, f"continuation tool changed: {name}")
    require(code_identity()["source_sha256"] == binding["source_sha256"], "model source changed")
    require(verify_corpus(full) == binding["split_sha256"], "corpus changed")
    require(binding["domain_policy"] == DOMAIN_POLICY, "domain policy changed")


def prepare():
    require(not CONT.exists(), "continuation already prepared; use --run to recover")
    parent, resolved, config, manifest, _, before, full = parent_inputs()
    require(
        config == TrainingConfig(device="cuda:0", total_steps=END, warmup_steps=5),
        "frozen configuration mismatch",
    )
    progress = read(PRODUCTION / "progress.json")
    require(
        progress["status"] == "STAGE A COMPLETE; REVIEW PENDING"
        and progress["step"] == 5000
        and progress["final_restore"] == "EXACT PASS"
        and (PRODUCTION / progress["checkpoint"]).resolve() == PARENT.resolve(),
        "unexpected production progress; no automatic repair",
    )
    accepted = read(ROOT / "reports/full_training_2m_stage_a.json")
    for item in accepted["checkpoint_identities"]:
        path = PRODUCTION / item["path"]
        require(
            sha(path / "state.pt") == item["state_sha256"]
            and sha(path / "manifest.json") == item["manifest_sha256"],
            "accepted checkpoint mismatch",
        )
    rows = journal(sorted(PRODUCTION.glob("attempt-*/observations.jsonl")))
    require(len(rows) == 5000 and rows[-1]["bytes_seen"] == 639613, "Stage-A journal mismatch")
    for actual, expected in zip(parent["metrics"], rows, strict=True):
        compare_replay(actual, expected)
    data = WindowDataset(manifest, DATA, "train", 32)
    require(len(data) == 312666, "frozen window count mismatch")
    total = math.ceil(len(data) / 4)
    extras = total * 4 - len(data)
    stream = BatchStream(data, config.seed + 1, 2)
    extra_bytes = sum(len(w.payload) for w in stream.next_batch()[:extras])
    require(
        total == END and extras == 2 and 10000000 + extra_bytes == 10000064,
        "one-pass accounting mismatch",
    )
    protected = dict(before)
    for p in PRODUCTION.rglob("*"):
        if p.is_file():
            protected[p.relative_to(ROOT).as_posix()] = sha(p)
    for stem in (
        "context_distribution_study_2m",
        "context_training_ab_2m",
        "real_training_diagnostic_2m",
    ):
        for suffix in ("md", "json"):
            p = ROOT / f"reports/{stem}.{suffix}"
            protected[p.relative_to(ROOT).as_posix()] = sha(p)
    positions = read(STUDY / "position-distributions/binding.json")["selections"]
    binding = {
        "schema": "production-continuation-1",
        "kind": "SAME_RUN_CONTINUATION_AMENDMENT",
        "production_binding_sha256": sha(PRODUCTION / "binding.json"),
        "parent": PARENT.relative_to(ROOT).as_posix(),
        "parent_step": 5000,
        "parent_state_sha256": sha(PARENT / "state.pt"),
        "training_config": config.to_dict(),
        "resolved_sha256": resolved.sha256,
        "source_sha256": code_identity()["source_sha256"],
        "code": code_identity(),
        "tools": {p: sha(ROOT / p) for p in TOOLS},
        "protected": protected,
        "split_sha256": full["split_sha256"],
        "manifest_sha256": manifest.sha256,
        "schedule": {
            "windows": len(data),
            "windows_per_update": 4,
            "total_steps": END,
            "extra_epoch1_windows": extras,
            "total_bytes": 10000064,
            "parent_bytes": 639613,
            "new_bytes": 9360451,
        },
        "checkpoint_steps": list(CHECKPOINTS),
        "validation_steps": list(VALIDATIONS),
        "restore_proof_steps": list(PROOFS),
        "domain_policy": DOMAIN_POLICY,
        "position_selections": [p for p in positions if p["length"] == 32],
        "test": "SEALED; streaming integrity hashes only",
        "progress_policy": "Original Stage-A progress preserved; continuation-v1/progress.json "
        "is the amendment index; fsynced append-only observations are authoritative.",
        "recovery_policy": "Restore latest verified checkpoint; compare every durable semantic "
        "row exactly; separately journal physical replay; never overwrite checkpoints.",
        "ab": "DISPOSABLE; neither branch promoted",
    }
    from validate_production_continuation import verify_receipt

    receipt_path = ROOT / "data/full-training-2m-continuation-validation-v1/preflight.json"
    verify_receipt(receipt_path)
    binding["preflight_validation_sha256"] = sha(receipt_path)
    CONT.mkdir()
    write_new(CONT / "binding.json", binding)
    publish_progress(CONT / "progress.json", {"status": "PREPARED", "step": 5000})
    print("Continuation identities and one-pass accounting PASS; zero updates", flush=True)


def state(trainer):
    return cpu(
        {
            "model": trainer.model.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "scheduler": trainer.scheduler.state_dict(),
            "cursor": trainer.stream.state_dict(),
            "metrics": semantic_metrics(trainer.metrics),
            "python_rng": random.getstate(),
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": cuda_rng_state(trainer.device),
            "counters": [
                trainer.global_step,
                trainer.micro_step,
                trainer.examples_seen,
                trainer.bytes_seen,
            ],
        }
    )


def check_restored(trainer, saved):
    actual = state(trainer)
    for key, value in (
        ("model", saved["model"]),
        ("optimizer", saved["optimizer"]),
        ("scheduler", saved["scheduler"]),
        ("cursor", saved["data_cursor"]),
        ("python_rng", saved["python_rng"]),
        ("cpu_rng", saved["torch_cpu_rng"]),
        ("cuda_rng", saved["cuda_rng"]),
    ):
        require(trees_equal(actual[key], value), f"exact restore mismatch: {key}")
    require(
        actual["counters"]
        == [saved[k] for k in ("global_step", "micro_step", "examples_seen", "bytes_seen")],
        "restore counters",
    )
    require(len(trainer.metrics) == saved["global_step"], "restore metric length")
    for a, b in zip(trainer.metrics, saved["metrics"], strict=True):
        compare_replay(a, b)


def step(trainer, order_hashes):
    start = time.perf_counter()
    row = trainer.step()  # Accepted implementation: unchanged corpus checks and update semantics.
    require(
        all(p.is_cuda and p.dtype == torch.float32 for p in trainer.model.parameters()),
        "production execution profile changed",
    )
    counts = []
    for group in trainer.optimizer.param_groups:
        for p in group["params"]:
            values = trainer.optimizer.state[p]
            require(
                all(
                    not isinstance(v, torch.Tensor) or bool(torch.isfinite(v).all())
                    for v in values.values()
                ),
                "nonfinite optimizer state",
            )
            counts.append(int(values["step"].item()))
    epoch, offset = trainer.stream.epoch, trainer.stream.offset
    require((epoch, offset) == expected_cursor(trainer.global_step), "cursor discontinuity")
    require(
        counts == [trainer.global_step] * 56 and trainer.scheduler.completed == trainer.global_step,
        "optimizer discontinuity",
    )
    if epoch not in order_hashes:
        order_hashes[epoch] = canonical_hash(trainer.stream.order)
    memory = row["memory"]["cuda"]
    require(
        memory["reserved_bytes"] <= 1024**3 and memory["free_device_bytes"] >= 1024**3,
        "production CUDA memory budget exceeded",
    )
    row.update(
        summed_nll=row["loss"] * row["valid_target_count"],
        finite_gradients=True,
        optimizer_update_counts=counts,
        scheduler_completed=trainer.scheduler.completed,
        cursor={"epoch": epoch, "offset": offset, "order_sha256": order_hashes[epoch]},
        update_seconds=time.perf_counter() - start,
    )
    return row


def checkpoint(trainer, binding):
    path = trainer.save()
    saved = load_checkpoint(path)
    check_restored(trainer, saved)
    write_new(
        path / "continuation_binding.json",
        {
            "amendment_sha256": canonical_hash(binding),
            "production_binding_sha256": binding["production_binding_sha256"],
            "checkpoint_manifest_sha256": sha(path / "manifest.json"),
            "step": trainer.global_step,
        },
    )
    return path


def latest_checkpoint(binding, rows):
    latest, step_number = PARENT, 5000
    for path in sorted(CONT.glob("attempt-*/checkpoints/step_*")):
        saved = load_checkpoint(path)
        side_path = path / "continuation_binding.json"
        require(side_path.exists(), "incomplete checkpoint publication; preserve for review")
        side = read(side_path)
        require(
            side["amendment_sha256"] == canonical_hash(binding)
            and side["production_binding_sha256"] == binding["production_binding_sha256"]
            and side["checkpoint_manifest_sha256"] == sha(path / "manifest.json")
            and side["step"] == saved["global_step"],
            "continuation checkpoint mismatch",
        )
        n = saved["global_step"]
        require(
            5000 < n <= 5000 + len(rows) and path.name == f"step_{n:06d}",
            "checkpoint/journal ordering mismatch",
        )
        for a, b in zip(saved["metrics"][5000:], rows[: n - 5000], strict=True):
            compare_replay(a, b)
        if n > step_number:
            latest, step_number = path, n
    return latest


def evaluate(trainer, saved, binding, manifest, payloads):
    root = EVAL / f"step_{trainer.global_step:06d}"
    store = Store(
        root,
        {
            "amendment_sha256": canonical_hash(binding),
            "step": trainer.global_step,
            "checkpoint_sha256": sha(saved / "state.pt"),
        },
    )
    if (complete := store.get("complete")) is not None:
        return complete
    before = state(trainer)
    trainer.model.eval()
    try:
        full = segmented_panel(
            trainer.model, validation_documents(manifest, payloads), 32, store, "full"
        )
        require(full["count"] == 500000, "validation incomplete")
        generation_panel(trainer.model, store)
        distribution_panel(trainer.model, binding["position_selections"], payloads, store)
    finally:
        trainer.model.train()
    require(trees_equal(state(trainer), before), "evaluation altered production state")
    generations = [store.get(f"generation_{i}") for i in range(6)]
    result = {
        "step": trainer.global_step,
        "validation": full,
        "parameter_sha256": tensor_hash(trainer.model.state_dict()),
        "generations": [
            dict(
                {k: g[k] for k in ("prefix_hex", "output_hex", "escaped")},
                whitespace=whitespace_summary(g["steps"]),
            )
            for g in generations
        ],
        "position_distribution": distribution_summary(
            [
                store.get(f"position_{i:03d}")["distribution"]
                for i in range(len(binding["position_selections"]))
            ]
        ),
        "generation_distribution": distribution_summary(
            [s for g in generations for s in g["steps"]]
        ),
        "test": "SEALED",
    }
    store.put("complete", result)
    return result


def restore_proof(trainer, saved, binding, resolved, config, manifest, result):
    proof_file = CONT / f"restore_{trainer.global_step:06d}.json"
    if proof_file.exists():
        proof = read(proof_file)
        require(
            proof["checkpoint_sha256"] == sha(saved / "state.pt")
            and proof["status"] == "EXACT PASS",
            "restore proof mismatch",
        )
        return
    expected = state(trainer)
    root = CONT / f"restore-{trainer.global_step:06d}-{time.time_ns()}"
    other = Trainer(resolved, config, manifest, DATA, root, resume_from=saved)
    check_restored(other, load_checkpoint(saved))
    require(trees_equal(state(other), expected), "independent recreated state differs")
    # Reuse the accepted finite/clock-checked greedy path in a separate proof store.
    store = Store(
        EVAL / root.name,
        {"amendment_sha256": canonical_hash(binding), "checkpoint_sha256": sha(saved / "state.pt")},
    )
    other.model.eval()
    generation_panel(other.model, store)
    for i, g in enumerate(result["generations"]):
        require(
            store.get(f"generation_{i}")["output_hex"] == g["output_hex"],
            "restored generation differs",
        )
    del other
    gc.collect()
    torch.cuda.empty_cache()
    require(trees_equal(state(trainer), expected), "restore proof changed live state/RNG")
    write_new(
        proof_file,
        {
            "status": "EXACT PASS",
            "step": trainer.global_step,
            "checkpoint_sha256": sha(saved / "state.pt"),
            "model_optimizer_scheduler_counters_cursor_rng": "EXACT",
            "six_generations": "EXACT",
            "optimizer_updates": 0,
        },
    )


def run_locked():
    binding = read(CONT / "binding.json")
    parent, resolved, config, manifest, payloads, _, full = parent_inputs()
    verify_binding(binding, full)
    require(
        str(torch.__version__) == "2.6.0+cu118" and torch.version.cuda == "11.8",
        "CUDA environment mismatch",
    )
    driver = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version,name,memory.total", "--format=csv,noheader"],
        text=True,
    ).strip()
    require(driver == read(PRODUCTION / "binding.json")["driver"], "GPU/driver changed")
    progress_path = CONT / "progress.json"
    progress = read(progress_path)
    require(
        progress["status"] not in ("DOMAIN_REGRESSION_REVIEW_REQUIRED", "ENDPOINT_COMPLETE"),
        "continuation stopped or completed; no automatic override",
    )
    rows = read_rows(CONT / "observations.jsonl")
    audit_rows(rows)
    require(5000 <= progress["step"] <= 5000 + len(rows), "published progress exceeds journal")
    saved = latest_checkpoint(binding, rows)
    attempt = CONT / f"attempt-{time.time_ns()}"
    trainer = Trainer(resolved, config, manifest, DATA, attempt, resume_from=saved)
    require(trainer.env == read(PRODUCTION / "binding.json")["environment"], "environment changed")
    require(trainer.run_manifest["parameter_count"] == 1929579, "parameter count changed")
    check_restored(trainer, load_checkpoint(saved))
    order_hashes = {}
    replay = rows[trainer.global_step - 5000 :]
    write_new(
        attempt / "recovery.json",
        {
            "checkpoint": saved.relative_to(ROOT).as_posix(),
            "checkpoint_step": trainer.global_step,
            "published_step": progress["step"],
            "durable_step": 5000 + len(rows),
            "replay_updates_required": len(replay),
            "replay_bytes_required": sum(r["valid_target_count"] for r in replay),
        },
    )
    for expected in replay:
        row = step(trainer, order_hashes)
        append(attempt / "physical_replay.jsonl", row)
        compare_row(row, expected)
    progress.update(
        status="RUNNING", step=trainer.global_step, checkpoint=saved.relative_to(ROOT).as_posix()
    )
    publish_progress(progress_path, progress)
    if not (CONT / "preflight.json").exists():
        require(trainer.global_step == 5000 and not replay, "missing initial preflight")
        check_restored(trainer, parent)
        write_new(
            CONT / "preflight.json",
            {
                "status": "PASS",
                "step": 5000,
                "optimizer_updates": 0,
                "parameter_sha256": tensor_hash(trainer.model.state_dict()),
                "exact_model_optimizer_scheduler_rng_cursor": True,
                "environment": trainer.env,
                "amendment_sha256": canonical_hash(binding),
            },
        )
        print("MILESTONE: continuation preflight PASS", flush=True)
    parent_full = read(ROOT / "reports/context_distribution_study_2m.json")["segmented"]["32"]
    evaluations = []
    for n in VALIDATIONS:
        path = EVAL / f"step_{n:06d}/complete.json"
        if path.exists():
            require(n <= trainer.global_step, "evaluation ahead of trajectory")
            store = Store(path.parent, read(path.parent / "binding.json"))
            evaluations.append(store.get("complete"))

    def boundary():
        nonlocal saved
        n = trainer.global_step
        if n in CHECKPOINTS:
            # Reuse an already complete checkpoint when recovering at the same boundary.
            if load_checkpoint(saved)["global_step"] != n:
                saved = checkpoint(trainer, binding)
            verify_binding(binding, full)
            progress.update(step=n, checkpoint=saved.relative_to(ROOT).as_posix())
            publish_progress(progress_path, progress)
        if n in VALIDATIONS:
            result = evaluate(trainer, saved, binding, manifest, payloads)
            previous = next((e["validation"] for e in reversed(evaluations) if e["step"] < n), None)
            if not any(e["step"] == n for e in evaluations):
                evaluations.append(result)
            if n in PROOFS:
                restore_proof(trainer, saved, binding, resolved, config, manifest, result)
            bad = domain_watch(parent_full, previous, result["validation"])
            progress.update(
                last_validation=n, validation_nll=result["validation"]["nll"], domain_regression=bad
            )
            if bad:
                progress["status"] = "DOMAIN_REGRESSION_REVIEW_REQUIRED"
            publish_progress(progress_path, progress)
            print(
                f"MILESTONE: {n}; validation NLL {result['validation']['nll']}; "
                f"domain review {bad}",
                flush=True,
            )
            if bad:
                return False
        return True

    if not boundary():
        return
    while trainer.global_step < END:
        row = step(trainer, order_hashes)
        append(CONT / "observations.jsonl", row)
        if trainer.global_step % 100 == 0:
            progress["step"] = trainer.global_step
            publish_progress(progress_path, progress)
        if trainer.global_step in CHECKPOINTS and not boundary():
            return
    require(
        trainer.global_step == END
        and trainer.bytes_seen == 10000064
        and trainer.examples_seen == 312668
        and trainer.micro_step == 156334
        and (trainer.stream.epoch, trainer.stream.offset) == (1, 2),
        "endpoint accounting",
    )
    finite_parameters(trainer.model, "endpoint")
    verify_binding(binding, full)
    progress.update(
        status="ENDPOINT_COMPLETE",
        step=END,
        bytes_seen=trainer.bytes_seen,
        parameter_sha256=tensor_hash(trainer.model.state_dict()),
        test="SEALED",
    )
    publish_progress(progress_path, progress)
    print("MILESTONE: endpoint complete; finalization required", flush=True)


def run():
    # OS lock survives no process; stale file is harmless after infrastructure interruption.
    import msvcrt

    with (CONT / "runner.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            run_locked()
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--run", action="store_true")
    args = parser.parse_args()
    prepare() if args.prepare else run()
