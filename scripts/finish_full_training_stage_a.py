"""Verify saved Stage-A evidence and summarize it without training or evaluation."""

import hashlib
import json
import math
import statistics

import torch
from corpus_acquisition import ROOT, digest, read
from full_training_stage_a import RUN, inputs
from training_progress import compare_replay, journal

from unified_edge.training.checkpoint import load_checkpoint


def finish():
    frozen, source, manifest, resolved, schedule = inputs()
    binding = read(RUN / "binding.json")
    progress = read(RUN / "progress.json")
    amendment = read(RUN / "recovery-v1/amendment.json")
    assert progress["status"] == "STAGE A COMPLETE; REVIEW PENDING"
    assert progress["step"] == 5000 and progress["final_restore"] == "EXACT PASS"
    assert binding["frozen_inputs"] == frozen and binding["schedule"] == schedule
    assert binding["resolved_sha256"] == resolved.sha256
    assert binding["runner_sha256"] == amendment["original_runner_sha256"]
    protected = dict(binding["preserved_sha256"])
    protected.update(
        {
            "scripts/full_training_stage_a.py": amendment["corrected_runner_sha256"],
            "scripts/training_progress.py": amendment["publisher_sha256"],
            "scripts/recover_stage_a.py": amendment["recovery_script_sha256"],
        }
    )
    for path, checksum in protected.items():
        assert digest((ROOT / path).read_bytes()) == checksum, path
    assert digest((RUN / "binding.json").read_bytes()) == amendment["binding_sha256"]
    inventory = read(RUN / "recovery-v1/inventory.json")
    assert (
        digest((RUN / "recovery-v1/inventory.json").read_bytes())
        == amendment["interrupted_inventory_sha256"]
    )
    assert (
        digest((RUN / "recovery-v1/original_runner.py").read_bytes())
        == amendment["original_runner_sha256"]
    )
    for path, record in inventory["files"].items():
        snapshot = RUN / "recovery-v1/snapshot" / path
        target = snapshot if snapshot.exists() else RUN / path
        assert digest(target.read_bytes()) == record["sha256"], path
        assert target.stat().st_size == record["bytes"], path
    paths = sorted(RUN.glob("attempt-*/observations.jsonl"))
    rows = journal(paths)
    assert len(rows) == 5000
    checkpoints = []
    final = None
    for path in sorted(RUN.glob("attempt-*/checkpoints/step_*")):
        saved = load_checkpoint(path)
        sidecar = read(path / "production_binding.json")
        assert sidecar["binding_sha256"] == amendment["binding_sha256"]
        assert sidecar["checkpoint_manifest_sha256"] == digest(
            (path / "manifest.json").read_bytes()
        )
        assert sidecar["split_sha256"] == frozen["split_sha256"]
        assert sidecar["full_total_steps"] == 78167
        assert sidecar["manifest_sha256"] == manifest.sha256
        assert saved["model_config"] == resolved.to_dict()
        assert saved["code"]["source_sha256"] == source["source_sha256"]
        assert saved["training_config"] == binding["training_config"]
        assert saved["dataset_sha256"] == manifest.sha256
        checkpoints.append(
            {
                "path": path.relative_to(RUN).as_posix(),
                "step": saved["global_step"],
                "state_sha256": read(path / "manifest.json")["sha256"],
                "manifest_sha256": sidecar["checkpoint_manifest_sha256"],
            }
        )
        if path.relative_to(RUN).as_posix() == progress["checkpoint"]:
            final = saved
    assert final is not None and final["global_step"] == 5000
    assert set(binding["checkpoint_steps"]) <= {c["step"] for c in checkpoints}
    assert len(final["metrics"]) == len(rows)
    for actual, expected in zip(final["metrics"], rows, strict=True):
        compare_replay(actual, expected)
    checksum = hashlib.sha256()
    for name, value in final["model"].items():
        assert torch.isfinite(value).all(), name
        checksum.update(name.encode())
        checksum.update(bytes(value.contiguous().view(torch.uint8).flatten().tolist()))
    assert checksum.hexdigest() == progress["parameter_sha256"]
    parameter_count = sum(
        math.prod(p["shape"]) for p in final["parameter_structure"] if p["requires_grad"]
    )
    assert parameter_count == 1929579 and len(final["parameter_structure"]) == 56
    assert final["optimizer_update_counts"] == [5000] * 56
    assert final["scheduler"]["completed"] == 5000
    assert final["data_cursor"]["epoch"] == 0 and final["data_cursor"]["offset"] == 20000
    assert final["accumulation_position"] == 0
    for field in ("micro_step", "examples_seen", "bytes_seen"):
        assert final[field] == rows[-1][field]
    for values in final["optimizer"]["state"].values():
        assert all(
            not isinstance(v, torch.Tensor) or torch.isfinite(v).all() for v in values.values()
        )
    assert [v["step"] for v in progress["validation"]] == [0, 250, 1000, 2500, 5000]
    assert all(v["valid_target_count"] == 500000 for v in progress["validation"])
    assert [v["step"] for v in progress["generation"]] == [0, 250, 1000, 2500, 5000]
    for group in progress["generation"]:
        assert [s["prefix_hex"] for s in group["samples"]] == binding["prefixes_hex"]
        assert all(len(bytes.fromhex(s["hex"])) == 64 for s in group["samples"])
    recovery = progress["recovery"]
    replay_path = RUN / "recovery-v1/replay_observations.jsonl"
    replay = [json.loads(line) for line in replay_path.read_text(encoding="utf-8").splitlines()]
    assert len(replay) == recovery["physical_replay_updates"] == 180
    for actual, expected in zip(replay, rows[252:432], strict=True):
        compare_replay(actual, expected)
    assert sum(r["valid_target_count"] for r in replay) == recovery["physical_replay_bytes"]
    restored = read(RUN / "stage-a-restored/run.json")
    assert restored["resumed_from"]["step"] == 5000
    assert restored["resumed_from"]["checkpoint_manifest"] == read(
        RUN / progress["checkpoint"] / "manifest.json"
    )
    assert restored["resolved_sha256"] == resolved.sha256
    assert restored["training_sha256"] == binding["training_sha256"]
    assert restored["dataset_sha256"] == manifest.sha256
    assert restored["code"]["source_sha256"] == source["source_sha256"]
    overhead = {
        "updates": recovery["physical_replay_updates"] + progress["resume_proof"]["extra_updates"],
        "windows": recovery["physical_replay_updates"] * 4
        + progress["resume_proof"]["extra_windows"],
        "target_bytes": recovery["physical_replay_bytes"] + progress["resume_proof"]["extra_bytes"],
    }
    seconds = sum(row["update_seconds"] for row in rows)
    cuda = [row["memory"]["cuda"] for row in rows] + [v["memory"] for v in progress["validation"]]
    train = {
        "logical_updates": 5000,
        "micro_steps": final["micro_step"],
        "windows": final["examples_seen"],
        "target_bytes": final["bytes_seen"],
        "first_100_mean_nll": statistics.mean(r["loss"] for r in rows[:100]),
        "last_100_mean_nll": statistics.mean(r["loss"] for r in rows[-100:]),
        "minimum_update_nll": min(r["loss"] for r in rows),
        "final_update_nll": rows[-1]["loss"],
        "final_lr": rows[-1]["learning_rate"],
        "review_points": [
            {
                k: rows[step - 1][k]
                for k in (
                    "step",
                    "moving_mean_100_nll",
                    "learning_rate",
                    "gradient_norm",
                    "clipped",
                    "bytes_seen",
                    "examples_seen",
                    "update_seconds",
                    "memory",
                )
            }
            for step in (250, 1000, 2500, 5000)
        ],
        "maximum_preclip_gradient_norm": max(r["gradient_norm"] for r in rows),
        "clipped_updates": sum(r["clipped"] for r in rows),
        "logical_update_seconds": seconds,
        "target_bytes_per_second_excluding_validation_replay": final["bytes_seen"] / seconds,
        "validation_seconds": sum(v["seconds"] for v in progress["validation"]),
        "remaining_training_only_projected_hours": (78167 - 5000) * seconds / 5000 / 3600,
        "elapsed_note": (
            "Summed measured update durations; not uninterrupted wall time. Validation, "
            "replay, downtime and generation excluded."
        ),
        "trajectory_every_100": [
            {k: r[k] for k in ("step", "moving_mean_100_nll", "learning_rate", "bytes_seen")}
            for r in rows
            if r["step"] % 100 == 0
        ],
    }
    memory = {
        "maximum_allocated_bytes": max(m["allocated_bytes"] for m in cuda),
        "maximum_peak_allocated_bytes": max(m["peak_allocated_bytes"] for m in cuda),
        "maximum_peak_reserved_bytes": max(m["peak_reserved_bytes"] for m in cuda),
        "minimum_sampled_free_device_bytes": min(m["free_device_bytes"] for m in cuda),
        "maximum_sampled_process_rss_bytes": max(r["memory"]["process_rss_bytes"] for r in rows),
        "maximum_process_peak_working_set_bytes": max(
            r["memory"]["process_peak_working_set_bytes"] for r in rows
        ),
        "final_tensor_accounting": rows[-1]["memory"],
    }
    evidence = {
        "schema": "1",
        "status": "READY FOR CONTINUATION REVIEW",
        "production_binding": binding,
        "current_source_identity": source,
        "recovery_commit": "5879c142a295e8beac38e0839eec91eb9dbfd5ab",
        "recovery_amendment": amendment,
        "recovery": progress["recovery"],
        "resume_proof": progress["resume_proof"],
        "physical_overhead": overhead,
        "restored_run_manifest": restored,
        "train": train,
        "validation": progress["validation"],
        "validation_changes": [
            {
                "from_step": a["step"],
                "to_step": b["step"],
                "nll_change": b["raw_byte_nll"] - a["raw_byte_nll"],
                "assessment": "improved"
                if b["raw_byte_nll"] < a["raw_byte_nll"]
                else "not_improved",
            }
            for a, b in zip(progress["validation"], progress["validation"][1:])
        ],
        "generation": progress["generation"],
        "memory": memory,
        "parameter_count": parameter_count,
        "parameter_sha256": checksum.hexdigest(),
        "checkpoint_identities": checkpoints,
        "final_restoration": {
            "status": progress["final_restore"],
            "evidence": (
                "Completed runner destroyed/recreated Trainer and compared model, optimizer, "
                "scheduler, semantic metrics, counters, cursor, Python/CPU/CUDA RNG and all "
                "six generation probes exactly before publishing completion."
            ),
            "finalization": (
                "Reused completed restore; independently rehashed every checkpoint and "
                "compared all 5000 checkpoint semantic rows with original journals. No "
                "optimizer update or validation rerun."
            ),
        },
        "integrity": {
            "frozen_inputs": "PASS",
            "diagnostic_artifacts": "PASS",
            "interruption_inventory": "PASS",
            "test": "SEALED; integrity hashes only",
        },
        "artifact_sha256": {
            p.relative_to(ROOT).as_posix(): digest(p.read_bytes())
            for p in paths
            + [RUN / "binding.json", RUN / "progress.json", RUN / "recovery-v1/result.json"]
        },
        "recorded_finalization_checks_2026_09_22": {
            "cpu_targeted": "22 passed",
            "cuda_placement_resume": "2 passed; 5 deselected",
            "ruff_format_compile": "PASS",
            "cpu_cuda_pip_check": "PASS",
            "warning": "Optional unused NumPy bridge unavailable; no dependency change",
        },
        "limitations": [
            (
                "32-byte windows expose approximately four completed 8-byte payload patches. "
                "No long-context retention evidence; no 4K/32K/1M claim."
            ),
            (
                "Greedy byte-mode degeneracy persists: empty prefix emits 64 spaces, "
                "conditioned prefixes emit short fragments then spaces. Healthy validation "
                "does not establish useful language quality."
            ),
            (
                "Autograd activation bytes are not measured separately; memory headroom "
                "applies only to this bounded profile."
            ),
            (
                "No TEST evaluation, new training, architecture change or continuation beyond "
                "5000 during finalization."
            ),
        ],
        "recommendation": (
            "Human review chooses completion of the existing 78167-update 32-byte "
            "schedule or a separately reviewed longer-window curriculum. Neither is "
            "automatically launched."
        ),
    }
    output = ROOT / "reports/full_training_2m_stage_a.json"
    output.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Stage A integrity PASS: {len(rows)} logical updates, {len(checkpoints)} checkpoints")


if __name__ == "__main__":
    finish()
