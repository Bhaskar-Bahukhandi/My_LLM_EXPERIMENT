"""Fixed tiny-overfit and restart evidence; no external data or GPU path."""

import copy
import json
import random
import time
from dataclasses import replace
from functools import partial
from pathlib import Path

import torch

from unified_edge.training.checkpoint import load_checkpoint
from unified_edge.training.config import TrainingConfig
from unified_edge.training.data import create_tiny_fixture
from unified_edge.training.memory import process_memory
from unified_edge.training.trainer import Trainer


def tree_error(actual, expected) -> float:
    """Compare all semantic state fields exactly and report the largest tensor error."""
    if isinstance(actual, torch.Tensor):
        if (
            not isinstance(expected, torch.Tensor)
            or actual.dtype != expected.dtype
            or actual.shape != expected.shape
        ):
            raise AssertionError("state tensor structure mismatch")
        error = (actual.double() - expected.double()).abs().max().item() if actual.numel() else 0.0
        if not torch.equal(actual, expected):
            raise AssertionError(f"resume tensor mismatch: max error {error}")
        return error
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        return max((tree_error(actual[k], expected[k]) for k in actual), default=0.0)
    if isinstance(actual, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        return max((tree_error(a, b) for a, b in zip(actual, expected, strict=True)), default=0.0)
    assert actual == expected
    return 0.0


def semantic_metrics(rows):
    return [{k: v for k, v in row.items() if k != "elapsed_seconds"} for row in rows]


@torch.inference_mode()
def generate_bytes(model, length: int = 64) -> dict:
    if type(length) is not int or not 1 <= length <= 256:
        raise ValueError("bounded generation length must be 1..256")
    was_training = model.training
    model.eval()
    state = model.start()
    output, control_argmax = [], 0
    try:
        for _ in range(length):
            logits = model.predict(state)
            assert torch.isfinite(logits[:, :256]).all()
            control_argmax += int(logits.argmax(-1).item() >= 256)
            symbol = logits[:, :256].argmax(-1)
            output.append(symbol.item())
            state = model.consume(symbol, state)
        assert state.shared.steps == 1 + length // 8
        assert state.hierarchy.pending.shape[1] == length % 8
    finally:
        model.train(was_training)
    payload = bytes(output)
    return {
        "policy": "greedy over raw-byte classes 0..255",
        "length": length,
        "hex": payload.hex(),
        "escaped": repr(payload),
        "unrestricted_head_control_argmax_count": control_argmax,
        "shared_steps": state.shared.steps,
        "completed_patches": state.hierarchy.completed_patches,
    }


def run_resume_gate(root: Path, model_config, manifest, data_root: Path) -> dict:
    config = replace(TrainingConfig(), total_steps=6, warmup_steps=2)
    continuous = Trainer(model_config, config, manifest, data_root, root / "continuous")
    continuous.train_until(3)
    random.random()
    torch.rand(3)
    continuous.train_until(6)
    expected = {
        "model": copy.deepcopy(continuous.model.state_dict()),
        "optimizer": copy.deepcopy(continuous.optimizer.state_dict()),
        "scheduler": continuous.scheduler.state_dict(),
        "cursor": continuous.stream.state_dict(),
        "metrics": semantic_metrics(continuous.metrics),
        "counters": [
            continuous.global_step,
            continuous.micro_step,
            continuous.examples_seen,
            continuous.bytes_seen,
        ],
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
    }
    next_windows = continuous.stream.next_batch()
    del continuous
    interrupted = Trainer(model_config, config, manifest, data_root, root / "interrupted")
    interrupted.train_until(3)
    random.random()
    torch.rand(3)
    path = interrupted.save()
    del interrupted
    resumed = Trainer(model_config, config, manifest, data_root, root / "resumed", resume_from=path)
    resumed.train_until(6)
    actual = {
        "model": resumed.model.state_dict(),
        "optimizer": resumed.optimizer.state_dict(),
        "scheduler": resumed.scheduler.state_dict(),
        "cursor": resumed.stream.state_dict(),
        "metrics": semantic_metrics(resumed.metrics),
        "counters": [
            resumed.global_step,
            resumed.micro_step,
            resumed.examples_seen,
            resumed.bytes_seen,
        ],
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
    }
    error = tree_error(actual, expected)
    assert resumed.stream.next_batch() == next_windows
    return {
        "total_updates": 6,
        "split_update": 3,
        "max_parameter_optimizer_rng_tensor_error": error,
        "scheduler_cursor_counters_semantic_metrics_exact": True,
        "next_batch_exact": True,
        "comparison_excludes": ["wall-clock measurements", "run ID and creation time"],
        "training_config": config.to_dict(),
    }


@torch.inference_mode()
def trained_causality(model, payload: bytes) -> dict:
    data = torch.tensor([list(payload[:137])])
    full = model(data)
    allowed = torch.ones(267, dtype=torch.bool)
    allowed[256:258] = False
    assert torch.isfinite(full[..., allowed]).all()
    maximum = 0.0
    for cut in (0, 7, 8, 9, 31, 32, 63, 64, 65, 127, 128, 129):
        altered = data.clone()
        altered[:, cut:] = (altered[:, cut:] + 71) % 256
        result = model(altered)
        torch.testing.assert_close(result[:, : cut + 1], full[:, : cut + 1], atol=1e-5, rtol=1e-4)
        maximum = max(
            maximum,
            (result[:, : cut + 1, allowed] - full[:, : cut + 1, allowed]).abs().max().item(),
        )
    state, stepped = model.start(), []
    for token in data[0]:
        stepped.append(model.predict(state))
        state = model.consume(token.reshape(1), state)
    stepped = torch.stack(stepped, 1)
    torch.testing.assert_close(stepped, full, atol=1e-5, rtol=1e-4)
    return {
        "future_suffix_max_abs": maximum,
        "full_step_max_abs": (stepped[..., allowed] - full[..., allowed]).abs().max().item(),
    }


def run_overfit(root: Path, model_config) -> dict:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    before = process_memory()
    manifest = create_tiny_fixture(root / "data")
    config = TrainingConfig()
    trainer = Trainer(model_config, config, manifest, root / "data", root / "overfit")
    names = {id(p): name for name, p in trainer.model.named_parameters()}
    groups = [
        {
            "weight_decay": group["weight_decay"],
            "parameter_count": sum(p.numel() for p in group["params"]),
            "tensor_names": [names[id(p)] for p in group["params"]],
        }
        for group in trainer.optimizer.param_groups
    ]
    initial_train, initial_valid = trainer.validate("train"), trainer.validate()
    snapshots, updates = [{"step": 0, "train": initial_train, "validation": initial_valid}], []
    seen, maximum_gradient = set(), 0.0

    def observe(name, gradient):
        nonlocal maximum_gradient
        assert torch.isfinite(gradient).all(), name
        seen.add(name)
        maximum_gradient = max(maximum_gradient, gradient.abs().max().item())
        return gradient

    handles = [
        p.register_hook(partial(observe, name)) for name, p in trainer.model.named_parameters()
    ]
    started = time.perf_counter()
    try:
        for _ in range(config.total_steps):
            row = trainer.step()
            updates.append(row)
            if row["step"] % 10 == 0:
                train, valid = trainer.validate("train"), trainer.validate()
                snapshots.append({"step": row["step"], "train": train, "validation": valid})
                print(
                    f"step={row['step']} train_nll={train['raw_byte_nll']:.6f} "
                    f"validation_nll={valid['raw_byte_nll']:.6f}",
                    flush=True,
                )
    finally:
        for handle in handles:
            handle.remove()
    duration = time.perf_counter() - started
    assert len(seen) == len(names) == 56
    final_train, final_valid = snapshots[-1]["train"], snapshots[-1]["validation"]
    checkpoint = trainer.save()
    generation = generate_bytes(trainer.model)
    regression = trained_causality(trainer.model, (root / "data/train.raw").read_bytes())
    output_targets = torch.tensor([list((root / "data/train.raw").read_bytes()[:137])])
    with torch.inference_mode():
        expected_output = trainer.model(output_targets).clone()
    saved = load_checkpoint(checkpoint)
    original_run_manifest = trainer.run_manifest
    del trainer
    restored = Trainer(
        model_config, config, manifest, root / "data", root / "restored", resume_from=checkpoint
    )
    assert tree_error(restored.model.state_dict(), saved["model"]) == 0
    assert generate_bytes(restored.model) == generation
    with torch.inference_mode():
        assert torch.equal(restored.model(output_targets), expected_output)
    assert restored.validate("train")["raw_byte_nll"] == final_train["raw_byte_nll"]
    restored_checkpoint = restored.save()
    assert load_checkpoint(restored_checkpoint)["global_step"] == config.total_steps
    reduction = 1 - final_train["raw_byte_nll"] / initial_train["raw_byte_nll"]
    result = {
        "training_config": config.to_dict(),
        "dataset_manifest": manifest.to_dict(),
        "dataset_manifest_sha256": manifest.sha256,
        "run_manifest": original_run_manifest,
        "restored_run_manifest": restored.run_manifest,
        "optimizer_groups": groups,
        "initial_train": initial_train,
        "initial_validation": initial_valid,
        "final_train": final_train,
        "final_validation": final_valid,
        "minimum_evaluated_train_nll": min(row["train"]["raw_byte_nll"] for row in snapshots),
        "training_nll_reduction_fraction": reduction,
        "passed_reduction_gate": reduction >= 0.5,
        "optimizer_phase_wall_seconds": duration,
        "optimizer_steps": restored.global_step,
        "micro_steps": restored.micro_step,
        "examples_seen": restored.examples_seen,
        "bytes_seen": restored.bytes_seen,
        "validation_snapshots": snapshots,
        "updates": updates,
        "generation": generation,
        "trained_causality": regression,
        "checkpoint_restore_weights_and_generation_exact": True,
        "checkpoint_restore_full_logits_exact": True,
        "checkpoint_restore_output_sequence_length": output_targets.shape[1],
        "gradient_tensors_observed_finite": len(seen),
        "maximum_observed_gradient_element": maximum_gradient,
        "process_memory_before_model": before,
        "process_memory_after_restore": process_memory(),
        "checkpoint": checkpoint.relative_to(root).as_posix(),
    }
    with (root / "overfit_result.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    assert result["passed_reduction_gate"], "fixed tiny-overfit training reduction gate failed"
    return result
