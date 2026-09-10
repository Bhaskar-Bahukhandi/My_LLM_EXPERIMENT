"""Explicit deterministic execution boundary for the CPU and single-GPU FP32 profiles."""

import os

import torch

CUDA_ALLOCATOR_BUDGET_BYTES = 1024**3
CUDA_FREE_HEADROOM_BYTES = 1024**3


def configure_device(name: str) -> torch.device:
    if name not in ("cpu", "cuda:0"):
        raise ValueError("supported execution devices are cpu and cuda:0")
    device = torch.device(name)
    if device.type == "cuda":
        if torch.cuda.is_initialized() and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
            raise RuntimeError("configure deterministic cuBLAS before CUDA initialization")
        workspace = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if workspace != ":4096:8":
            raise ValueError("CUDA profile requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("CUDA profile requires exactly one available CUDA device")
        torch.cuda.set_device(device)
        free, total = torch.cuda.mem_get_info(device)
        if free < CUDA_ALLOCATOR_BUDGET_BYTES + CUDA_FREE_HEADROOM_BYTES:
            raise RuntimeError("CUDA profile lacks allocator budget plus 1 GiB free headroom")
        torch.cuda.set_per_process_memory_fraction(CUDA_ALLOCATOR_BUDGET_BYTES / total, device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return device


def cuda_rng_state(device: torch.device):
    return torch.cuda.get_rng_state_all() if device.type == "cuda" else None


def validate_cuda_rng(state, device: torch.device):
    if device.type == "cpu":
        if state is not None:
            raise ValueError("CPU checkpoint must not contain CUDA RNG state")
        return
    expected = torch.cuda.get_rng_state_all()
    if not isinstance(state, list) or len(state) != len(expected):
        raise ValueError("CUDA RNG device-count mismatch")
    for index, (value, reference) in enumerate(zip(state, expected, strict=True)):
        if (
            not isinstance(value, torch.Tensor)
            or value.device.type != "cpu"
            or value.dtype != torch.uint8
            or value.shape != reference.shape
        ):
            raise ValueError("invalid CUDA RNG state")
        torch.Generator(device=f"cuda:{index}").set_state(value)
