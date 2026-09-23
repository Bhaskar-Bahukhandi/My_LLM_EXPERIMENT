"""Controls required before any disposable GPU A/B update."""

import copy
import hashlib
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from context_ab_core import (
    QUOTAS,
    Arm,
    independent,
    matched_targets,
    safe_destination,
    select_blocks,
    trees_equal,
)  # noqa: E402
from context_distribution_study import PRODUCTION  # noqa: E402
from test_mamba import accepted_config  # noqa: E402

from unified_edge.dense_model import DenseByteModel  # noqa: E402
from unified_edge.training.config import TrainingConfig  # noqa: E402
from unified_edge.training.optimization import (  # noqa: E402
    WarmupCosine,
    build_optimizer,
    raw_byte_nll,
)


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        yield
    torch.set_num_threads(previous)


def documents():
    payload = bytes(range(256)) * 160
    return [
        SimpleNamespace(
            path=d,
            split="train",
            domain=d,
            byte_count=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
        for d in QUOTAS
    ], payload


def test_selection_quota_boundaries_nonoverlap_and_content_identity():
    docs, raw = documents()
    selected = select_blocks(docs, lambda d: raw)
    assert selected == select_blocks(list(reversed(docs)), lambda d: raw)
    assert len(selected) == 512
    assert {d: sum(r["domain"] == d for r in selected) for d in QUOTAS} == QUOTAS
    for d in docs:
        intervals = sorted((r["start"], r["end"]) for r in selected if r["document"] == d.path)
        assert all(0 <= a < b <= d.byte_count and b - a == 128 for a, b in intervals)
        assert all(b <= c for (_, b), (c, _) in zip(intervals, intervals[1:]))
    for row in selected:
        assert hashlib.sha256(raw[row["start"] : row["end"]]).hexdigest() == row["sha256"]
    with pytest.raises(ValueError, match="capacity"):
        select_blocks(docs, lambda d: raw, dict(QUOTAS, general_text=999999))


@pytest.mark.parametrize("split", ["validation", "test"])
def test_training_builder_rejects_holdouts_before_read(split):
    docs, _ = documents()
    docs[0].split = split

    def forbidden(_):
        raise AssertionError("holdout content read")

    with pytest.raises(ValueError, match="TRAIN only"):
        select_blocks(docs, forbidden)


def test_same_bytes_and_exact_loss_normalization():
    raw = bytes(range(128))
    a, b = matched_targets(raw, 32), matched_targets(raw, 64)
    assert a.shape == (4, 32) and b.shape == (2, 64)
    assert bytes(a.flatten().tolist()) == bytes(b.flatten().tolist()) == raw
    torch.manual_seed(43)
    logits = torch.randn(128, 267, requires_grad=True)
    loss_a, count_a = raw_byte_nll(logits.reshape(4, 32, 267), a)
    grad_a = torch.autograd.grad(loss_a / 128, logits, retain_graph=True)[0]
    loss_b, count_b = raw_byte_nll(logits.reshape(2, 64, 267), b)
    grad_b = torch.autograd.grad(loss_b / 128, logits, retain_graph=True)[0]
    assert count_a == count_b == 128
    assert torch.equal(grad_a, grad_b)
    reference = torch.nn.functional.cross_entropy(logits, torch.arange(128), reduction="mean")
    assert torch.equal(grad_a, torch.autograd.grad(reference, logits)[0])


def parent_fixture():
    torch.manual_seed(43)
    config = TrainingConfig(total_steps=600)
    resolved = accepted_config()
    model = DenseByteModel(resolved)
    optimizer = build_optimizer(model, config)
    scheduler = WarmupCosine(optimizer, config)
    target = torch.arange(128).reshape(4, 32)
    (raw_byte_nll(model(target), target)[0] / 128).backward()
    optimizer.step()
    scheduler.advance()
    parent = {
        "global_step": 1,
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "scheduler": scheduler.state_dict(),
        "optimizer_update_counts": [1] * len(list(model.parameters())),
        "python_rng": random.getstate(),
        "torch_cpu_rng": torch.get_rng_state(),
        "cuda_rng": None,
    }
    return parent, resolved, config


@pytest.mark.parametrize("length", [32, 64])
def test_clone_preflight_checkpoint_resume_and_wrong_binding(tmp_path, length):
    parent, resolved, config = parent_fixture()
    original = copy.deepcopy(parent)
    a = Arm(parent, resolved, config, length, "bound", "a", "cpu")
    b = Arm(parent, resolved, config, 64, "bound", "b", "cpu")
    assert trees_equal(a.model.state_dict(), b.model.state_dict())
    assert trees_equal(a.optimizer.state_dict(), b.optimizer.state_dict())
    assert trees_equal(a.scheduler.state_dict(), parent["scheduler"])
    independent(parent, a.state(), b.state())
    with pytest.raises(ValueError, match="aliases"):
        independent(a.state(), a.state())
    initial = copy.deepcopy(a.state())
    preflight = a.update(bytes(range(128)), advance=False)
    assert preflight["semantic"]["gradient_tensors"] == 56
    assert trees_equal(a.state(), initial)
    first = a.update(bytes(range(128)))
    assert first["semantic"]["target_bytes"] == 128
    checkpoint = a.save(tmp_path / "a" / "update_000001", tmp_path)
    second = a.update(bytes(reversed(range(128))))
    final = copy.deepcopy(a.state())
    del a
    restored = Arm(parent, resolved, config, length, "bound", "a", "cpu")
    restored.restore(checkpoint, tmp_path)
    repeated = restored.update(bytes(reversed(range(128))))
    assert repeated["semantic"] == second["semantic"]
    assert trees_equal(restored.state(), final)
    restored.restore(checkpoint, tmp_path)
    assert random.getstate() == restored.rng["python_rng"]
    assert torch.equal(torch.get_rng_state(), restored.rng["torch_cpu_rng"])
    wrong = Arm(parent, resolved, config, length, "wrong", "a", "cpu")
    with pytest.raises(ValueError, match="binding"):
        wrong.restore(checkpoint, tmp_path)
    assert trees_equal(parent, original)


def test_experiment_paths_cannot_write_production_or_escape(tmp_path):
    with pytest.raises(ValueError, match="production"):
        safe_destination(PRODUCTION / "step_005001", PRODUCTION)
    with pytest.raises(ValueError, match="outside"):
        safe_destination(tmp_path.parent / "escape", tmp_path)
    assert safe_destination(tmp_path / "arm" / "checkpoint", tmp_path).is_relative_to(tmp_path)


def test_branch_validation_weights_tails_and_reuses_completed_blocks(tmp_path, monkeypatch):
    import math

    import context_ab_evaluation as evaluation
    from context_distribution_study import Store

    original_tensor = torch.tensor

    def cpu_tensor(*args, **kwargs):
        if kwargs.get("device") == "cuda:0":
            kwargs["device"] = "cpu"
        return original_tensor(*args, **kwargs)

    monkeypatch.setattr(torch, "tensor", cpu_tensor)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(
        evaluation,
        "memory",
        lambda: {
            "peak_allocated_bytes": 0,
            "peak_reserved_bytes": 0,
            "free_device_bytes": 0,
            "process_rss_status": "UNVERIFIED",
        },
    )

    class Model:
        calls = 0

        def __call__(self, target):
            self.calls += 1
            logits = torch.zeros(*target.shape, 267)
            logits[..., [256, 257]] = -torch.inf
            return logits

    model = Model()
    docs = [
        (SimpleNamespace(path="a", domain="general_text"), bytes(129)),
        (SimpleNamespace(path="b", domain="code"), bytes(65)),
    ]
    store = Store(tmp_path, {"test": "isolated evaluation"})
    result = evaluation.segmented_panel(model, docs, 64, store, "small")
    assert result["count"] == 194
    assert result["domains"]["general_text"]["count"] == 129
    assert result["domains"]["code"]["count"] == 65
    assert sum(v["count"] for v in result["buckets"].values()) == 194
    assert abs(result["nll"] - math.log(265)) < 1e-6
    calls = model.calls
    assert evaluation.segmented_panel(model, docs, 64, store, "small") == result
    assert model.calls == calls
