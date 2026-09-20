"""Fixed Stage A of the final one-pass run; resumable checkpoints, sealed test."""

import argparse
import gc
import json
import math
import random
import subprocess
import time
from pathlib import Path

import torch
from benchmark_fingerprints import canonical
from corpus_acquisition import PILOT, ROOT, digest, read, write_new
from corpus_freeze import verify_manifest
from real_training_diagnostic import tensor_hash

from unified_edge.resolve import ResolvedConfig
from unified_edge.training.checkpoint import load_checkpoint
from unified_edge.training.config import TrainingConfig
from unified_edge.training.data import BatchStream, DatasetManifest, WindowDataset
from unified_edge.training.experiment import generate_bytes, semantic_metrics, tree_error
from unified_edge.training.memory import cuda_memory
from unified_edge.training.trainer import Trainer, code_identity

RUN = ROOT / "data/full-training-2m-v1"
POINTS = (0, 250, 1000, 2500, 5000)
CHECKPOINTS = (250, 1000, 2000, 2500, 3000, 4000, 5000)
PREFIXES = (b"", b"The ", b"def ", b"import ", b"class ", b"x = ")


@torch.inference_mode()
def prompted_generation(model, prefix, length=64):
    if prefix not in PREFIXES or length != 64:
        raise ValueError("Stage-A qualitative probe policy is frozen")
    if not prefix:
        return dict(generate_bytes(model, length), prefix_hex="", prefix_escaped="b''")
    was_training = model.training
    model.eval()
    state = model.start()
    output = []
    try:
        for byte in prefix:
            state = model.consume(
                torch.tensor([byte], device=next(model.parameters()).device), state
            )
        for _ in range(length):
            logits = model.predict(state)
            assert torch.isfinite(logits[:, :256]).all()
            symbol = logits[:, :256].argmax(-1)
            output.append(symbol.item())
            state = model.consume(symbol, state)
        assert state.shared.steps == 1 + (len(prefix) + length) // 8
        assert state.hierarchy.pending.shape[1] == (len(prefix) + length) % 8
    finally:
        model.train(was_training)
    payload = bytes(output)
    return {
        "prefix_hex": prefix.hex(),
        "prefix_escaped": repr(prefix),
        "policy": "greedy raw-byte classes 0..255",
        "length": length,
        "hex": payload.hex(),
        "escaped": repr(payload),
        "shared_steps": state.shared.steps,
        "completed_patches": state.hierarchy.completed_patches,
    }


def inputs():
    accepted = read(ROOT / "reports/real_training_diagnostic_2m.json")
    frozen = accepted["diagnostic"]["binding"]
    source = code_identity()
    assert source["source_sha256"] == frozen["code"]["source_sha256"]
    full_path = PILOT / "final-corpus-v1/manifest.json"
    full = read(full_path)
    assert digest(full_path.read_bytes()) == frozen["frozen_manifest"]["raw_sha256"]
    assert digest(canonical(full)) == frozen["frozen_manifest"]["canonical_sha256"]
    assert verify_manifest(full, PILOT) == frozen["split_bytes"]
    assert full["split_sha256"] == frozen["split_sha256"]
    manifest = DatasetManifest.from_dict(read(PILOT / "final-corpus-v1/training_manifest.json"))
    assert manifest.sha256 == frozen["native_manifest_sha256"]
    resolved = ResolvedConfig.from_dict(read(ROOT / "reports/edge_2m_resolved.json"))
    assert resolved.sha256 == frozen["resolved_sha256"]
    data = WindowDataset(manifest, PILOT, "train", 32)
    per_update = 4
    total = math.ceil(len(data) / per_update)
    extra = total * per_update - len(data)
    stream = BatchStream(data, 18, 2)
    extra_bytes = sum(len(w.payload) for w in stream.next_batch()[:extra]) if extra else 0
    schedule = {
        "train_windows": len(data),
        "windows_per_update": per_update,
        "one_pass_total_steps": total,
        "extra_epoch1_windows": extra,
        "one_pass_target_bytes_including_extra": 10000000 + extra_bytes,
        "stage_stop": 5000,
    }
    assert schedule["train_windows"] == 312666 and total == 78167 and extra == 2
    return frozen, source, manifest, resolved, schedule


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu(v) for v in value)
    return value


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
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "counters": [
                trainer.global_step,
                trainer.micro_step,
                trainer.examples_seen,
                trainer.bytes_seen,
            ],
        }
    )


def run(resume):
    frozen, source, manifest, resolved, schedule = inputs()
    config = TrainingConfig(
        device="cuda:0", total_steps=schedule["one_pass_total_steps"], warmup_steps=5
    )
    assert str(torch.__version__) == "2.6.0+cu118" and torch.version.cuda == "11.8"
    driver = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version,name,memory.total", "--format=csv,noheader"],
        text=True,
    ).strip()
    assert driver == frozen["driver"]
    if resume:
        binding = read(RUN / "binding.json")
        progress = read(RUN / "progress.json")
        assert binding["training_config"] == config.to_dict()
        assert binding["schedule"] == schedule
        assert binding["frozen_inputs"] == frozen
        saved = RUN / progress["checkpoint"]
        saved_state = load_checkpoint(saved)
        if saved_state["global_step"] != progress["step"]:
            raise RuntimeError("Unsaved updates need explicit recovery; no automatic replay")
        attempt = RUN / f"attempt-{len(list(RUN.glob('attempt-*'))):03d}"
        trainer = Trainer(resolved, config, manifest, PILOT, attempt, resume_from=saved)
        assert trainer.env == binding["environment"]
    else:
        RUN.mkdir(exist_ok=False)
        trainer = Trainer(resolved, config, manifest, PILOT, RUN / "attempt-000")
        assert tensor_hash(trainer.model) == frozen["initial_parameter_sha256"]
        assert trainer.run_manifest["parameter_count"] == 1929579
        assert trainer.env == frozen["environment"]
        preserved_paths = [
            ROOT / "reports/real_training_diagnostic_2m.md",
            ROOT / "reports/real_training_diagnostic_2m.json",
        ]
        preserved_paths += list((ROOT / "data/real-diagnostic-2m-v1").rglob("state.pt"))
        binding = {
            "production_id": "edge-2m-one-pass-v1",
            "driver": driver,
            "code": source,
            "runner_sha256": digest(Path(__file__).read_bytes()),
            "frozen_inputs": frozen,
            "schedule": schedule,
            "training_config": config.to_dict(),
            "training_sha256": config.sha256,
            "resolved_sha256": resolved.sha256,
            "environment": trainer.env,
            "initial_parameter_sha256": tensor_hash(trainer.model),
            "warmup_rationale": "Preserve validated five-update warmup; only cosine horizon "
            "changes to the exact full one-pass schedule, never the Stage-A stop.",
            "validation_steps": list(POINTS),
            "checkpoint_steps": list(CHECKPOINTS),
            "prefixes_hex": [p.hex() for p in PREFIXES],
            "preserved_sha256": {
                p.relative_to(ROOT).as_posix(): digest(p.read_bytes()) for p in preserved_paths
            },
        }
        write_new(RUN / "binding.json", binding)
        progress = {
            "status": "RUNNING",
            "step": 0,
            "checkpoint": None,
            "validation": [],
            "generation": [],
            "resume_proof": None,
        }

    def publish():
        path = RUN / "progress.tmp"
        path.write_text(json.dumps(progress, indent=2, allow_nan=False), encoding="utf-8")
        path.replace(RUN / "progress.json")

    def save():
        path = trainer.save()
        write_new(
            path / "production_binding.json",
            {
                "binding_sha256": digest((RUN / "binding.json").read_bytes()),
                "checkpoint_manifest_sha256": digest((path / "manifest.json").read_bytes()),
                "full_total_steps": config.total_steps,
                "manifest_sha256": manifest.sha256,
                "split_sha256": frozen["split_sha256"],
            },
        )
        progress["checkpoint"] = path.relative_to(RUN).as_posix()
        publish()
        return path

    def evaluate():
        step = trainer.global_step
        if not any(row["step"] == step for row in progress["generation"]):
            progress["generation"].append(
                {"step": step, "samples": [prompted_generation(trainer.model, p) for p in PREFIXES]}
            )
            publish()
        if any(row["step"] == step for row in progress["validation"]):
            return
        print("Full validation begins", step, flush=True)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        result = trainer.validate()
        assert result["valid_target_count"] == 500000
        result.update(seconds=time.perf_counter() - started, memory=cuda_memory(trainer.device))
        progress["validation"].append(result)
        publish()
        print("Validation complete", step, result["raw_byte_nll"], flush=True)

    def step():
        start = time.perf_counter()
        row = trainer.step()
        assert all(p.is_cuda and p.dtype == torch.float32 for p in trainer.model.parameters())
        for values in trainer.optimizer.state.values():
            for value in values.values():
                assert not isinstance(value, torch.Tensor) or torch.isfinite(value).all()
        memory = row["memory"]["cuda"]
        assert memory["reserved_bytes"] <= 1024**3 and memory["free_device_bytes"] >= 1024**3
        row["update_seconds"] = time.perf_counter() - start
        row["target_bytes_per_second"] = row["valid_target_count"] / row["update_seconds"]
        recent = trainer.metrics[-100:]
        row["moving_mean_100_nll"] = sum(x["loss"] for x in recent) / len(recent)
        with (trainer.run_dir / "observations.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        progress["step"] = trainer.global_step
        publish()
        if trainer.global_step % 100 == 0:
            print("Update", trainer.global_step, row["moving_mean_100_nll"], flush=True)

    def prove_resume():
        nonlocal trainer
        if progress["resume_proof"] is not None:
            return
        if trainer.global_step != 250:
            raise RuntimeError("Resume proof must start from the saved update-250 boundary")
        saved = RUN / progress["checkpoint"]
        step()
        step()
        expected = state(trainer)
        expected_generation = prompted_generation(trainer.model, b"def ")
        save()
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        attempt = RUN / f"attempt-{len(list(RUN.glob('attempt-*'))):03d}"
        trainer = Trainer(resolved, config, manifest, PILOT, attempt, resume_from=saved)
        trainer.step()
        trainer.step()
        actual = state(trainer)
        assert tree_error(actual, expected) == 0.0
        assert prompted_generation(trainer.model, b"def ") == expected_generation
        progress["resume_proof"] = {
            "status": "EXACT PASS",
            "checkpoint_step": 250,
            "compare_step": 252,
            "extra_updates": 2,
            "extra_windows": 8,
            "extra_bytes": sum(r["valid_target_count"] for r in trainer.metrics[-2:]),
        }
        save()
        print("Full-schedule exact resume PASS", flush=True)

    publish()
    if trainer.global_step in POINTS:
        evaluate()
    if trainer.global_step == 250:
        prove_resume()
    elif trainer.global_step > 250 and progress["resume_proof"] is None:
        raise RuntimeError("Interrupted proof needs explicit reconciliation; no automatic replay")
    while trainer.global_step < 5000:
        step()
        if trainer.global_step in CHECKPOINTS:
            save()
        if trainer.global_step in POINTS:
            evaluate()
        if trainer.global_step == 250:
            prove_resume()
    expected_hash = tensor_hash(trainer.model)
    final = RUN / progress["checkpoint"]
    expected = state(trainer)
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    trainer = Trainer(
        resolved, config, manifest, PILOT, RUN / "stage-a-restored", resume_from=final
    )
    assert tree_error(state(trainer), expected) == 0.0
    assert tensor_hash(trainer.model) == expected_hash
    assert [prompted_generation(trainer.model, p) for p in PREFIXES] == progress["generation"][-1][
        "samples"
    ]
    inputs()
    for path, checksum in binding["preserved_sha256"].items():
        assert digest((ROOT / path).read_bytes()) == checksum
    progress.update(
        status="STAGE A COMPLETE; REVIEW PENDING",
        parameter_sha256=expected_hash,
        final_restore="EXACT PASS",
        corpus_unchanged=True,
    )
    publish()
    print(progress["status"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args().resume)
