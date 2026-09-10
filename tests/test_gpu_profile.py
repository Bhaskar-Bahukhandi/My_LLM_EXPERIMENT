"""CPU-runnable negative tests for CUDA boundaries and explicit failure classification."""

import pytest
import torch
from test_training import model_config

import unified_edge.training.gpu_probe as probe
from unified_edge.training.config import TrainingConfig
from unified_edge.training.device import configure_device, validate_cuda_rng
from unified_edge.training.memory import cuda_memory


def test_cuda_profile_rejects_unsupported_and_unavailable(monkeypatch):
    for device in ("cuda", "cuda:1", "mps", "auto"):
        with pytest.raises(ValueError):
            TrainingConfig(device=device)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="available"):
        configure_device("cuda:0")
    with pytest.raises(ValueError, match="CUDA memory"):
        cuda_memory(torch.device("cpu"))
    with pytest.raises(ValueError, match="CPU checkpoint"):
        validate_cuda_rng([torch.zeros(4, dtype=torch.uint8)], torch.device("cpu"))


def test_oom_is_recorded_at_requested_stage_without_resizing(monkeypatch):
    monkeypatch.setattr(probe, "configure_device", lambda _: torch.device("cuda:0"))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(probe, "cuda_memory", lambda _: {})

    def allocation_failure(_):
        raise torch.cuda.OutOfMemoryError("simulated allocation failure")

    monkeypatch.setattr(probe, "DenseByteModel", allocation_failure)
    row = probe.probe_candidate(model_config(), batch=2, length=64)
    assert row["classification"] == "DOES_NOT_FIT"
    assert row["failure_stage"] == "model_load"
    assert row["batch_size"] == 2 and row["sequence_length"] == 64
    assert (
        row["forward_status"] == row["backward_status"] == row["optimizer_step_status"] == "NOT_RUN"
    )
