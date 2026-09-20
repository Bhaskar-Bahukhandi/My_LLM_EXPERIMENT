"""Reconstruct only logged unsaved production updates under an exact replay gate."""

import json
from pathlib import Path

from corpus_acquisition import PILOT, ROOT, digest, read, write_new
from full_training_stage_a import RUN, inputs
from real_training_diagnostic import tensor_hash
from training_progress import compare_replay, journal, publish_progress, recovery_range

from unified_edge.training.checkpoint import load_checkpoint
from unified_edge.training.config import TrainingConfig
from unified_edge.training.data import BatchStream
from unified_edge.training.trainer import Trainer


def main():
    recovery = RUN / "recovery-v1"
    inventory = read(recovery / "inventory.json")
    for path, value in inventory["files"].items():
        assert digest((RUN / path).read_bytes()) == value["sha256"], path
    frozen, source, manifest, resolved, schedule = inputs()
    binding = read(RUN / "binding.json")
    assert binding["frozen_inputs"] == frozen
    assert binding["schedule"] == schedule
    config = TrainingConfig.from_dict(binding["training_config"])
    assert config.total_steps == 78167
    checkpoints = []
    for path in sorted(RUN.glob("attempt-*/checkpoints/*")):
        saved = load_checkpoint(path)
        sidecar = read(path / "production_binding.json")
        assert sidecar["binding_sha256"] == digest((RUN / "binding.json").read_bytes())
        assert sidecar["checkpoint_manifest_sha256"] == digest(
            (path / "manifest.json").read_bytes()
        )
        assert sidecar["manifest_sha256"] == manifest.sha256
        assert sidecar["split_sha256"] == frozen["split_sha256"]
        assert saved["model_config"] == resolved.to_dict()
        assert saved["training_config"] == config.to_dict()
        assert saved["dataset_sha256"] == manifest.sha256
        assert saved["scheduler"]["completed"] == saved["global_step"]
        assert saved["code"]["source_sha256"] == source["source_sha256"]
        checkpoints.append((saved["global_step"], path, saved))
    durable, checkpoint, saved = max(checkpoints, key=lambda item: (item[0], str(item[1])))
    paths = sorted(RUN.glob("attempt-*/observations.jsonl"))
    rows = journal(paths)
    previous = read(RUN / "progress.json")
    temporary = read(RUN / "progress.tmp")
    assert temporary["step"] == len(rows)
    assert temporary["validation"] == previous["validation"]
    assert temporary["generation"] == previous["generation"]
    replay = recovery_range(rows, durable, previous["step"])
    assert durable == 252 and len(rows) == 432 and len(replay) == 180
    amendment = {
        "version": 1,
        "production_id": binding["production_id"],
        "binding_sha256": digest((RUN / "binding.json").read_bytes()),
        "original_runner_sha256": inventory["original_runner_sha256"],
        "corrected_runner_sha256": digest((ROOT / "scripts/full_training_stage_a.py").read_bytes()),
        "publisher_sha256": digest((ROOT / "scripts/training_progress.py").read_bytes()),
        "recovery_script_sha256": digest(Path(__file__).read_bytes()),
        "reason": "Windows progress publication robustness only; "
        "model/training semantics unchanged",
        "cadence": "observation every update; progress every 100 plus checkpoint/evaluation/status",
        "publication": "unique temporary JSON; flush/fsync; six bounded attempts, "
        "backoff 0.05,0.1,0.2,0.4,0.8 seconds; prior valid progress survives exhaustion",
        "interrupted_inventory_sha256": digest((recovery / "inventory.json").read_bytes()),
    }
    write_new(recovery / "amendment.json", amendment)
    trainer = Trainer(
        resolved, config, manifest, PILOT, RUN / "attempt-002", resume_from=checkpoint
    )
    assert trainer.env == binding["environment"]
    reference = BatchStream(trainer.train_data, config.seed, config.batch_size)
    reference.load_state_dict(saved["data_cursor"])
    durable_identity = {
        "path": checkpoint.relative_to(RUN).as_posix(),
        "step": trainer.global_step,
        "micro_step": trainer.micro_step,
        "windows": trainer.examples_seen,
        "bytes": trainer.bytes_seen,
        "scheduler_step": trainer.scheduler.completed,
        "cursor_epoch": reference.epoch,
        "cursor_offset": reference.offset,
        "cursor_order_sha256": digest(json.dumps(reference.order).encode()),
        "parameter_sha256": tensor_hash(trainer.model),
    }
    for expected in replay:
        windows = [w for _ in range(config.accumulation_steps) for w in reference.next_batch()]
        assert sum(len(w.payload) for w in windows) == expected["valid_target_count"]
        actual = trainer.step()
        compare_replay(actual, expected)
        assert trainer.stream.state_dict() == reference.state_dict()
        with (recovery / "replay_observations.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(actual, allow_nan=False) + "\n")
    path = trainer.save()
    write_new(
        path / "production_binding.json",
        {
            "binding_sha256": digest((RUN / "binding.json").read_bytes()),
            "checkpoint_manifest_sha256": digest((path / "manifest.json").read_bytes()),
            "full_total_steps": config.total_steps,
            "manifest_sha256": manifest.sha256,
            "split_sha256": frozen["split_sha256"],
            "amendment_sha256": digest((recovery / "amendment.json").read_bytes()),
        },
    )
    result = {
        "status": "INTERRUPTED TRAJECTORY RECOVERY: EXACT PASS",
        "durable": durable_identity,
        "last_original_completed_step": len(rows),
        "original_progress_step": previous["step"],
        "original_temporary_step": temporary["step"],
        "replay_first": durable + 1,
        "replay_last": len(rows),
        "physical_replay_updates": len(replay),
        "physical_replay_bytes": sum(row["valid_target_count"] for row in replay),
        "logical_updates_added": 0,
        "semantic_max_error": 0.0,
        "cursor_ordering": "every replay update checked against frozen BatchStream permutation",
        "recovery_checkpoint": path.relative_to(RUN).as_posix(),
        "recovered_parameter_sha256": tensor_hash(trainer.model),
        "locking_process": "UNKNOWN; WinError 5 was observed, not attributed to a specific process",
        "hash_verified_candidate_paths": [p.relative_to(RUN).as_posix() for _, p, _ in checkpoints],
    }
    write_new(recovery / "result.json", result)
    temporary.update(
        step=trainer.global_step, checkpoint=path.relative_to(RUN).as_posix(), recovery=result
    )
    publish_progress(RUN / "progress.json", temporary)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
