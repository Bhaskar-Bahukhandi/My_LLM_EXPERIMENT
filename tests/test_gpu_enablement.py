"""Bounded real-CUDA gates; unavailable CUDA is an explicit skip, never a pass."""

import copy
import gc
import json
import random
import uuid
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from test_training import model_config

from unified_edge.config import AuthoredConfig, Hardware, ModelRequest, SearchPolicy
from unified_edge.dense_model import DenseByteModel
from unified_edge.parameters import audit_parameters
from unified_edge.resolve import resolve_config
from unified_edge.training.checkpoint import load_checkpoint, save_checkpoint
from unified_edge.training.config import TrainingConfig
from unified_edge.training.data import create_tiny_fixture
from unified_edge.training.device import configure_device
from unified_edge.training.experiment import generate_bytes, semantic_metrics, tree_error
from unified_edge.training.gpu_probe import probe_candidate
from unified_edge.training.memory import cuda_memory
from unified_edge.training.optimization import build_optimizer, finite_parameters, raw_byte_nll
from unified_edge.training.trainer import Trainer, code_identity

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA unavailable: real GPU gates not executed"
    ),
]
LENGTHS = (1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 63, 64, 65, 127, 128, 129)
CUTS = (0, 1, 7, 8, 9, 15, 16, 17, 63, 64, 65, 127, 128, 129)
ATOL, RTOL = 1e-5, 1e-4


@pytest.fixture(scope="module")
def gpu_run():
    device = configure_device("cuda:0")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    root = (
        Path(__file__).resolve().parents[1]
        / "evidence/gpu_enablement"
        / ("run-" + uuid.uuid4().hex[:12])
    )
    root.mkdir(parents=True, exist_ok=False)
    (root / "source.json").write_text(json.dumps(code_identity(), indent=2), encoding="utf-8")
    print("CUDA evidence:", root, flush=True)
    yield root, device
    gc.collect()
    torch.cuda.empty_cache()


def record(root, name, value):
    with (root / (name + ".json")).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
    print(name, "PASS", flush=True)


def compare(actual, reference):
    actual, reference = actual.detach().to("cpu"), reference.detach().to("cpu")
    assert actual.shape == reference.shape and actual.dtype == reference.dtype
    assert torch.equal(torch.isfinite(actual), torch.isfinite(reference))
    torch.testing.assert_close(actual, reference, atol=ATOL, rtol=RTOL)
    mask = torch.isfinite(reference)
    if not mask.any():
        return {"max_absolute": 0.0, "max_relative": 0.0}
    delta = (actual[mask].double() - reference[mask].double()).abs()
    return {
        "max_absolute": delta.max().item(),
        "max_relative": (delta / reference[mask].double().abs().clamp_min(1e-12)).max().item(),
    }


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu").clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def test_cuda_availability_placement_parameters_and_decay(gpu_run):
    root, device = gpu_run
    cpu = DenseByteModel(model_config())
    tagged = {name for name, p in cpu.named_parameters() if getattr(p, "_no_weight_decay", False)}
    gpu = DenseByteModel(model_config()).to(device)
    gpu.load_state_dict(cpu.state_dict())
    assert [(n, p.shape) for n, p in cpu.named_parameters()] == [
        (n, p.shape) for n, p in gpu.named_parameters()
    ]
    assert (
        audit_parameters(gpu)["unique_trainable_parameters"]
        == audit_parameters(cpu)["unique_trainable_parameters"]
        == 1929579
    )
    assert all(p.device == device and p.dtype == torch.float32 for p in gpu.parameters())
    assert tagged == {
        name for name, p in gpu.named_parameters() if getattr(p, "_no_weight_decay", False)
    }
    optimizer = build_optimizer(gpu, TrainingConfig(device="cuda:0"))
    grouped = [p for g in optimizer.param_groups for p in g["params"]]
    assert len(grouped) == len({id(p) for p in grouped}) == 56
    for group in optimizer.param_groups:
        assert all(
            group["weight_decay"] == 0
            for p in group["params"]
            if getattr(p, "_no_weight_decay", False)
        )
    with torch.inference_mode():
        state = gpu.start(2)
        assert state.hierarchy.pending.device == state.hierarchy.hidden.device == device
        assert all(layer.conv.device == layer.ssm.device == device for layer in state.shared.layers)
        restored = gpu.restore_state(gpu.export_state(state))
        assert torch.equal(gpu.predict(state), gpu.predict(restored))
        with pytest.raises(ValueError, match="device"):
            gpu.restore_state(cpu.export_state(cpu.start(2)))
    props = torch.cuda.get_device_properties(device)
    record(
        root,
        "availability",
        {
            "torch": str(torch.__version__),
            "runtime": torch.version.cuda,
            "gpu": props.name,
            "compute_capability": [props.major, props.minor],
            "total_vram": props.total_memory,
            "parameters": 1929579,
            "tensors": 56,
            "memory": cuda_memory(device),
        },
    )


def test_cpu_cuda_full_incremental_and_state_parity(gpu_run):
    root, device = gpu_run
    results = []
    with torch.inference_mode():
        for seed, batch in ((17, 1), (29, 2)):
            torch.manual_seed(seed)
            cpu = DenseByteModel(model_config()).eval()
            gpu = DenseByteModel(model_config()).to(device).eval()
            gpu.load_state_dict(cpu.state_dict())
            data = torch.randint(0, 256, (batch, 129))
            gpu_data = data.to(device)
            a, b = cpu.start(batch), gpu.start(batch)
            results.append(
                {
                    "seed": seed,
                    "batch": batch,
                    "length": 0,
                    "initial_logits": compare(gpu.predict(b), cpu.predict(a)),
                }
            )
            assert gpu(gpu_data[:, :0]).shape == (batch, 0, 267)
            stepped = []
            for index in range(129):
                stepped.append(gpu.predict(b))
                a, b = cpu.consume(data[:, index], a), gpu.consume(gpu_data[:, index], b)
                length = index + 1
                if length in LENGTHS:
                    full = gpu(gpu_data[:, :length])
                    row = {
                        "seed": seed,
                        "batch": batch,
                        "length": length,
                        "cpu_gpu_logits": compare(full, cpu(data[:, :length])),
                        "gpu_full_step": compare(full, torch.stack(stepped, 1)),
                        "hidden": compare(b.hierarchy.hidden, a.hierarchy.hidden),
                        "layers": [
                            {"conv": compare(y.conv, x.conv), "ssm": compare(y.ssm, x.ssm)}
                            for x, y in zip(a.shared.layers, b.shared.layers, strict=True)
                        ],
                    }
                    assert a.shared.steps == b.shared.steps == 1 + length // 8
                    assert (
                        a.hierarchy.completed_patches
                        == b.hierarchy.completed_patches
                        == length // 8
                    )
                    assert torch.equal(b.hierarchy.pending.to("cpu"), a.hierarchy.pending)
                    if length in (7, 16, 65):
                        b = gpu.restore_state(gpu.export_state(b))
                    results.append(row)
            inputs = torch.randn(batch, 17, cpu.config.shape.d_model)
            full_cpu, end_cpu = cpu.shared(inputs)
            full_gpu, end_gpu = gpu.shared(inputs.to(device))
            first, incoming = gpu.shared(inputs[:, :5].to(device))
            rest, continued = gpu.shared(inputs[:, 5:].to(device), incoming)
            chunked = torch.cat((first, rest), dim=1)
            step_state = gpu.shared.initialize_state(batch)
            step_outputs = []
            for token in inputs.to(device).unbind(1):
                output, step_state = gpu.shared.step(token, step_state)
                step_outputs.append(output)
            results.append(
                {
                    "seed": seed,
                    "batch": batch,
                    "shared_sequence_length": 17,
                    "split_point": 5,
                    "shared_cpu_gpu": compare(full_gpu, full_cpu),
                    "shared_full_step": compare(full_gpu, torch.stack(step_outputs, 1)),
                    "shared_nonzero_state_continuation": compare(chunked, full_gpu),
                    "full_state_cpu_gpu": [
                        {"conv": compare(y.conv, x.conv), "ssm": compare(y.ssm, x.ssm)}
                        for x, y in zip(end_cpu.layers, end_gpu.layers, strict=True)
                    ],
                    "full_state_continuation": [
                        {"conv": compare(y.conv, x.conv), "ssm": compare(y.ssm, x.ssm)}
                        for x, y in zip(end_gpu.layers, continued.layers, strict=True)
                    ],
                    "full_state_step": [
                        {"conv": compare(y.conv, x.conv), "ssm": compare(y.ssm, x.ssm)}
                        for x, y in zip(end_gpu.layers, step_state.layers, strict=True)
                    ],
                }
            )
            assert end_gpu.steps == end_cpu.steps == continued.steps == step_state.steps == 17
    record(
        root,
        "parity",
        {"atol": ATOL, "rtol": RTOL, "relative_denominator_floor": 1e-12, "cases": results},
    )


def test_cuda_adversarial_causality(gpu_run):
    root, device = gpu_run
    torch.manual_seed(17)
    model = DenseByteModel(model_config()).to(device).eval()
    rows = []
    with torch.inference_mode():
        data = torch.randint(0, 256, (2, 137), device=device)
        reference = model(data)
        for cut in CUTS:
            altered = data.clone()
            altered[:, cut:] = (altered[:, cut:] + 71) % 256
            rows.append(
                {"cut": cut, **compare(model(altered)[:, : cut + 1], reference[:, : cut + 1])}
            )
    record(root, "causality", {"batch": 2, "length": 137, "cases": rows})


def test_cuda_gradients_update_and_memory(gpu_run):
    root, device = gpu_run
    result = probe_candidate(model_config(), batch=2, length=32)
    assert result["classification"] == "FITS_TRAINING_MECHANICS"
    assert result["recommended_headroom"]
    torch.manual_seed(17)
    model = DenseByteModel(model_config()).to(device)
    targets = torch.arange(64, device=device).reshape(2, 32)
    nll, count = raw_byte_nll(model(targets), targets)
    (nll / count).backward()
    norms = {
        name: p.grad.norm().item() for name, p in model.named_parameters() if p.grad is not None
    }
    assert len(norms) == 56 and all(0 < v < float("inf") for v in norms.values())
    result["gradient_norms"] = norms
    record(root, "gradients_memory", result)


def snapshot(trainer):
    return cpu_tree(
        {
            "model": trainer.model.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "scheduler": trainer.scheduler.state_dict(),
            "cursor": trainer.stream.state_dict(),
            "counts": [
                trainer.global_step,
                trainer.micro_step,
                trainer.examples_seen,
                trainer.bytes_seen,
            ],
            "metrics": semantic_metrics(trainer.metrics),
            "python_rng": random.getstate(),
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
        }
    )


def test_cuda_exact_resume_rng_and_trained_generation(gpu_run):
    root, device = gpu_run
    data_root = root / "resume-data"
    manifest = create_tiny_fixture(data_root)
    config = replace(TrainingConfig(), device="cuda:0", total_steps=6, warmup_steps=2)
    continuous = Trainer(model_config(), config, manifest, data_root, root / "continuous")
    continuous.train_until(3)
    random.random()
    torch.rand(3)
    torch.rand(4, device=device)
    continuous.train_until(6)
    expected = snapshot(continuous)
    next_batch = continuous.stream.next_batch()
    generation = generate_bytes(continuous.model)
    del continuous
    gc.collect()
    interrupted = Trainer(model_config(), config, manifest, data_root, root / "interrupted")
    interrupted.train_until(3)
    random.random()
    torch.rand(3)
    torch.rand(4, device=device)
    path = interrupted.save()
    saved = load_checkpoint(path)
    assert len(saved["cuda_rng"]) == 1
    bad = copy.deepcopy(saved)
    bad["cuda_rng"] = [torch.zeros(4, dtype=torch.uint8)]
    corrupt_path = save_checkpoint(root / "invalid-rng-checkpoint", bad)
    with pytest.raises(ValueError, match="CUDA RNG"):
        Trainer(
            model_config(),
            config,
            manifest,
            data_root,
            root / "invalid-rng-run",
            resume_from=corrupt_path,
        )
    assert not (root / "invalid-rng-run").exists()
    del interrupted, saved, bad
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    resumed = Trainer(
        model_config(), config, manifest, data_root, root / "resumed", resume_from=path
    )
    restore_memory = cuda_memory(device)
    resumed.train_until(6)
    difference = tree_error(snapshot(resumed), expected)
    assert resumed.stream.next_batch() == next_batch
    assert generate_bytes(resumed.model) == generation
    finite_parameters(resumed.model, "resumed CUDA")
    record(
        root,
        "resume",
        {
            "updates": 6,
            "split": 3,
            "max_tensor_error": difference,
            "rng_cursor_counters_scheduler_metrics_exact": True,
            "generation_exact": True,
            "generation": generation,
            "memory_after_restore": restore_memory,
            "comparison_excludes": ["elapsed time", "run identity and creation time"],
        },
    )


def test_cuda_short_learning_and_checkpoint(gpu_run):
    root, device = gpu_run
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    data_root = root / "learning-data"
    manifest = create_tiny_fixture(data_root)
    config = replace(TrainingConfig(), device="cuda:0", total_steps=10, warmup_steps=2)
    trainer = Trainer(model_config(), config, manifest, data_root, root / "learning")
    initial = trainer.validate("train")
    updates = [trainer.step() for _ in range(10)]
    final, validation = trainer.validate("train"), trainer.validate()
    assert final["raw_byte_nll"] < initial["raw_byte_nll"]
    generation = generate_bytes(trainer.model)
    path = trainer.save()
    expected = cpu_tree(trainer.model.state_dict())
    del trainer
    gc.collect()
    restored = Trainer(
        model_config(), config, manifest, data_root, root / "learning-restored", resume_from=path
    )
    assert tree_error(cpu_tree(restored.model.state_dict()), expected) == 0
    assert restored.validate("train") == final
    assert generate_bytes(restored.model) == generation
    restored.save()
    record(
        root,
        "learning",
        {
            "config": config.to_dict(),
            "manifest": manifest.to_dict(),
            "initial_train": initial,
            "final_train": final,
            "held_out": validation,
            "updates": updates,
            "generation": generation,
            "checkpoint_restore_exact": True,
            "memory_after_restore": cuda_memory(device),
        },
    )


def test_cuda_fit_matrix(gpu_run):
    root, _ = gpu_run
    configurations = [model_config()]
    for target in (6_000_000, 20_000_000):
        result = resolve_config(
            AuthoredConfig(
                model=ModelRequest(target_parameters=target),
                search=SearchPolicy(widths=(256, 384, 512), depths=(4, 6, 8, 12)),
                hardware=Hardware(memory_budget_bytes=1024**3),
            )
        )
        assert result.selected is not None
        configurations.append(result.selected)
    rows = [probe_candidate(config, batch=1, length=32) for config in configurations]
    record(root, "fit_matrix", rows)
    assert rows[0]["classification"] == "FITS_TRAINING_MECHANICS"
