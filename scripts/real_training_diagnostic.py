"""One fixed real-data diagnostic using the accepted trainer, without test evaluation."""

import gc
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

import torch
from corpus_acquisition import PILOT, ROOT, digest, read, write_new
from corpus_freeze import verify_manifest

from unified_edge.resolve import ResolvedConfig
from unified_edge.training.checkpoint import CheckpointError, load_checkpoint
from unified_edge.training.config import TrainingConfig
from unified_edge.training.data import DatasetManifest
from unified_edge.training.experiment import generate_bytes, semantic_metrics, tree_error
from unified_edge.training.memory import cuda_memory
from unified_edge.training.trainer import Trainer, code_identity

RUN = ROOT / "data/real-diagnostic-2m-v1"


def tensor_hash(model):
    checksum = hashlib.sha256()
    for name, value in model.state_dict().items():
        checksum.update(name.encode())
        checksum.update(
            bytes(value.detach().cpu().contiguous().view(torch.uint8).flatten().tolist())
        )
    return checksum.hexdigest()


def verify_inputs():
    evidence = read(ROOT / "reports/real_corpus_final_evidence.json")
    manifest_path = PILOT / "final-corpus-v1/manifest.json"
    assert digest(manifest_path.read_bytes()) == evidence["manifest"]["raw_sha256"]
    manifest = read(manifest_path)
    assert manifest["split_sha256"] == evidence["split_sha256"]
    assert manifest["split_bytes"] == {"train": 10000000, "validation": 500000, "test": 500000}
    # Test bytes are read only for sealed integrity verification, never evaluated or displayed.
    verify_manifest(manifest, PILOT)
    native = PILOT / "final-corpus-v1/training_manifest.json"
    assert digest(native.read_bytes()) == evidence["training_manifest_sha256"]
    code = code_identity()
    assert code["source_sha256"] == evidence["model_source_sha256"]
    assert code["commit"] == evidence["rollback_commit"]
    return evidence, DatasetManifest.from_dict(read(native)), code


def snapshot(trainer):
    return {
        "model": {k: v.detach().cpu().clone() for k, v in trainer.model.state_dict().items()},
        "optimizer": load_checkpoint(trainer.save())["optimizer"],
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


def main():
    evidence, manifest, code = verify_inputs()
    config = TrainingConfig(device="cuda:0", total_steps=250, warmup_steps=5)
    resolved = ResolvedConfig.from_dict(read(ROOT / "reports/edge_2m_resolved.json"))
    assert str(torch.__version__) == "2.6.0+cu118" and torch.version.cuda == "11.8"
    recovery = RUN.exists()
    if recovery:
        previous = read(RUN / "progress.json")
        if previous["validation"] or previous["updates"] or list(RUN.rglob("state.pt")):
            raise RuntimeError("Pre-update recovery refuses completed work")
    else:
        RUN.mkdir(exist_ok=False)
    trainer = Trainer(
        resolved, config, manifest, PILOT, RUN / ("initial-recovered" if recovery else "initial")
    )
    assert trainer.run_manifest["parameter_count"] == 1929579
    assert torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 2050"
    assert torch.cuda.get_device_capability(0) == (8, 6)
    binding = {
        "code": code,
        "runner_sha256": digest(Path(__file__).read_bytes()),
        "resolved_config": resolved.to_dict(),
        "resolved_sha256": resolved.sha256,
        "training_config": config.to_dict(),
        "native_manifest_sha256": manifest.sha256,
        "frozen_manifest": evidence["manifest"],
        "split_sha256": evidence["split_sha256"],
        "split_bytes": evidence["split_bytes"],
        "environment": trainer.env,
        "parameter_count": 1929579,
        "initial_parameter_sha256": tensor_hash(trainer.model),
        "expected_working_tree": subprocess.check_output(
            ["git", "-c", "safe.directory=E:/Theai", "status", "--porcelain"], text=True
        ).splitlines(),
        "driver": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version,name,memory.total", "--format=csv,noheader"],
            text=True,
        ).strip(),
        "optimizer_groups": [
            {
                "weight_decay": g["weight_decay"],
                "tensors": len(g["params"]),
                "parameters": sum(p.numel() for p in g["params"]),
            }
            for g in trainer.optimizer.param_groups
        ],
        "validation_updates": [0, 5, 125, 250],
        "checkpoint_updates": [3, 5, 125, 250],
        "maximum_effective_bytes": 32000,
        "sampling": "existing deterministic BatchStream; no full epoch",
        "test_policy": "sealed; hash-only integrity reads",
    }
    if recovery:
        frozen = read(RUN / "input_binding.json")
        for key in binding.keys() - {"runner_sha256", "expected_working_tree"}:
            if binding[key] != frozen[key]:
                raise RuntimeError(f"Recovered initialization differs: {key}")
        write_new(
            RUN / "pre_update_recovery.json",
            {
                "original_progress": previous,
                "recovery_runner_sha256": binding["runner_sha256"],
                "reason": "Prior process absent; no completed validation/update/checkpoint",
                "initialization_checksum_equal": True,
            },
        )
        binding = frozen
        report = previous
    else:
        write_new(RUN / "input_binding.json", binding)
        report = {
            "binding": binding,
            "validation": [],
            "updates": [],
            "memory": [],
            "status": "RUNNING",
            "generation_before": generate_bytes(trainer.model),
            "generation_prompt": "empty prompt / existing BOS; greedy 64 raw bytes",
        }

    def publish():
        temporary = RUN / "progress.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(RUN / "progress.json")

    def memory(stage):
        row = {"stage": stage, **cuda_memory(trainer.device)}
        assert row["reserved_bytes"] <= 1024**3
        assert row["free_device_bytes"] >= 1024**3
        report["memory"].append(row)
        return row

    def validate():
        print("Full validation begins", trainer.global_step, flush=True)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        row = trainer.validate()
        assert row["valid_target_count"] == 500000
        row.update(seconds=time.perf_counter() - started, memory=memory("validation"))
        report["validation"].append(row)
        publish()
        print("Validation", row["step"], row["raw_byte_nll"], row["seconds"], flush=True)

    def step():
        started = time.perf_counter()
        row = trainer.step()
        for state in trainer.optimizer.state.values():
            for value in state.values():
                assert not isinstance(value, torch.Tensor) or torch.isfinite(value).all()
        assert all(p.is_cuda and p.dtype == torch.float32 for p in trainer.model.parameters())
        row["update_seconds"] = time.perf_counter() - started
        row["target_bytes_per_second"] = row["valid_target_count"] / row["update_seconds"]
        row["moving_average_nll"] = sum(r["raw_byte_nll"] for r in trainer.metrics[-20:]) / len(
            trainer.metrics[-20:]
        )
        row["optimizer_status"] = "finite update completed"
        report["updates"].append(row)
        memory("update_" + str(trainer.global_step))
        publish()
        if trainer.global_step % 25 == 0:
            print("Update", trainer.global_step, row["moving_average_nll"], flush=True)

    def checkpoint():
        path = trainer.save()
        write_new(
            path / "corpus_binding.json",
            {
                "input_binding_sha256": digest((RUN / "input_binding.json").read_bytes()),
                "native_dataset_sha256": manifest.sha256,
                "frozen_manifest": evidence["manifest"],
                "split_sha256": evidence["split_sha256"],
                "checkpoint_manifest_sha256": digest((path / "manifest.json").read_bytes()),
            },
        )
        return path

    memory("model_loaded")
    publish()
    validate()
    torch.cuda.reset_peak_memory_stats()

    def forward_hook(module, inputs, output):
        assert output.is_cuda and torch.isfinite(output[..., :256]).all()
        memory("after_forward")

    def before_optimizer(optimizer, args, kwargs):
        rows = []
        for name, p in trainer.model.named_parameters():
            g = p.grad
            rows.append(
                {
                    "name": name,
                    "missing": g is None,
                    "finite": g is not None and bool(torch.isfinite(g).all()),
                    "norm_after_clip": None if g is None else g.norm().item(),
                }
            )
        assert all(not r["missing"] and r["finite"] and r["norm_after_clip"] > 0 for r in rows)
        report["gradient_audit"] = rows
        report["post_clip_global_norm"] = sum(r["norm_after_clip"] ** 2 for r in rows) ** 0.5
        assert report["post_clip_global_norm"] <= config.clip_norm + 1e-5
        memory("after_backward_and_clip")

    fhandle = trainer.model.register_forward_hook(forward_hook)
    ohandle = trainer.optimizer.register_step_pre_hook(before_optimizer)
    step()
    fhandle.remove()
    ohandle.remove()
    step()
    step()
    saved = checkpoint()
    step()
    step()
    expected = snapshot(trainer)
    expected_generation = generate_bytes(trainer.model)
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    trainer = Trainer(resolved, config, manifest, PILOT, RUN / "resumed", resume_from=saved)
    # Wrong corpus identity must fail before any continuation update.
    invalid = load_checkpoint(saved)
    invalid["dataset_sha256"] = "0" * 64
    try:
        trainer._restore(invalid)
    except CheckpointError:
        report["wrong_corpus_rejected"] = True
    else:
        raise AssertionError("Wrong corpus checkpoint accepted")
    trainer.step()
    trainer.step()
    actual = snapshot(trainer)
    assert tree_error(actual, expected) == 0.0
    assert generate_bytes(trainer.model) == expected_generation
    report["resume"] = {
        "status": "EXACT PASS",
        "checkpoint_update": 3,
        "continuation_updates": 2,
        "max_tensor_error": 0.0,
        "semantic_loss_error": 0.0,
        "generation_equal": True,
        "compared": list(actual),
        "extra_replay_updates": 2,
    }
    print("Real-data smoke and exact resume PASS", flush=True)
    del actual, expected, invalid
    validate()
    while trainer.global_step < 250:
        step()
        if trainer.global_step in (125, 250):
            checkpoint()
            validate()
    report["generation_after"] = generate_bytes(trainer.model)
    report["final_parameter_sha256"] = tensor_hash(trainer.model)
    final_checkpoint = trainer.run_dir / "checkpoints/step_000250"
    final_generation = report["generation_after"]
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    trainer = Trainer(
        resolved, config, manifest, PILOT, RUN / "final-restored", resume_from=final_checkpoint
    )
    assert tensor_hash(trainer.model) == report["final_parameter_sha256"]
    assert generate_bytes(trainer.model) == final_generation
    checkpoint()
    verify_inputs()
    report["corpus_unchanged"] = True
    report["trained_checkpoint_restore"] = "weights and generation exact; re-checkpoint PASS"
    report["status"] = "MEASUREMENTS COMPLETE; REVIEW PENDING"
    publish()
    print(report["status"], flush=True)


if __name__ == "__main__":
    main()
