"""Explicit deterministic execution boundary for the CPU and single-GPU FP32 profiles."""

import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass

import torch

CUDA_ALLOCATOR_BUDGET_BYTES = 1024**3
CUDA_FREE_HEADROOM_BYTES = 1024**3


class DeviceAdmissionError(RuntimeError):
    """An explicitly selected device fails the declared execution policy."""


@dataclass(frozen=True)
class DevicePolicy:
    profile: str
    selected_index: int
    schema: str = "1"

    def __post_init__(self):
        if self.schema != "1" or self.profile not in (
            "20m_t4_single_fp32_pilot",
            "20m_rtx2050_fp32_fallback",
        ):
            raise ValueError("unsupported GPU admission profile/version")
        if type(self.selected_index) is not int or self.selected_index < 0:
            raise ValueError("GPU policy requires an explicit nonnegative selected_index")

    @property
    def allocator_cap_bytes(self):
        return 8 * 1024**3 if self.profile == "20m_t4_single_fp32_pilot" else 5 * 1024**3 // 2

    @property
    def free_headroom_bytes(self):
        return 2 * 1024**3 if self.profile == "20m_t4_single_fp32_pilot" else 1024**3 // 2

    def to_dict(self):
        return asdict(self)


def discover_devices() -> dict:
    """Discovery does not choose a device, allocate model tensors or configure execution."""
    return {"cuda_available": torch.cuda.is_available(), "device_count": torch.cuda.device_count()}


def device_index(name: str) -> int | None:
    if name == "cpu":
        return None
    if not isinstance(name, str) or not re.fullmatch(r"cuda:(0|[1-9][0-9]*)", name):
        raise ValueError("device must be cpu or explicit cuda:N; no automatic selection")
    return int(name.split(":")[1])


def admit_memory(free: int, total: int, cap: int, headroom: int, *, startup: bool):
    required = cap + headroom if startup else headroom
    if total < cap or free < required:
        raise DeviceAdmissionError("CUDA profile lacks allocator budget/free headroom")


def _driver_version() -> tuple[str | None, str]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None, "NVIDIA_SMI_UNAVAILABLE"
    try:
        result = subprocess.run(
            [executable, "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None, "NVIDIA_SMI_QUERY_FAILED"
    versions = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    if len(versions) != 1:
        return None, "NVIDIA_SMI_VERSION_AMBIGUOUS"
    return versions.pop(), "NVIDIA_SMI_READ_ONLY_QUERY"


def hardware_diagnostic(policy: DevicePolicy | None = None) -> dict:
    """Read-only discovery/telemetry; never admits a run or changes CUDA settings.

    Querying a selected CUDA device may initialize its driver context. No tensors,
    training, shell wrapper, device masking or memory-policy mutation is performed.
    An optional fixed nvidia-smi query records the driver version when available.
    """
    if policy is not None and not isinstance(policy, DevicePolicy):
        raise TypeError("diagnostic requires a typed DevicePolicy")
    discovery = discover_devices()
    result = {
        **discovery,
        "selected_device": None,
        "name": None,
        "compute_capability": None,
        "total_vram_bytes": None,
        "free_vram_bytes": None,
        "torch_version": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "driver": None,
        "driver_status": "UNAVAILABLE_FROM_SUPPORTED_TORCH_API",
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "profile": policy.to_dict() if policy else None,
        "allocator_policy": {
            "cap_bytes": policy.allocator_cap_bytes,
            "free_headroom_bytes": policy.free_headroom_bytes,
            "status": "ANALYTICALLY_PLANNED",
            "applied_by_diagnostic": False,
        }
        if policy
        else None,
        "execution_authorized": False,
        "status": "NO_CUDA" if not discovery["cuda_available"] else "SELECTION_REQUIRED",
    }
    if policy is not None and discovery["cuda_available"]:
        if policy.selected_index >= discovery["device_count"]:
            raise DeviceAdmissionError("selected device is not visible")
        device = torch.device(f"cuda:{policy.selected_index}")
        props = torch.cuda.get_device_properties(device)
        free, total = torch.cuda.mem_get_info(device)
        driver, driver_status = _driver_version()
        result.update(
            selected_device=str(device),
            name=props.name,
            compute_capability=[props.major, props.minor],
            total_vram_bytes=total,
            free_vram_bytes=free,
            status="DIAGNOSTIC_ONLY_NOT_EXECUTION_APPROVAL",
            driver=driver,
            driver_status=driver_status,
        )
    return result


def configure_device(name: str, policy: DevicePolicy | None = None) -> torch.device:
    index = device_index(name)
    if policy is not None and (
        not isinstance(policy, DevicePolicy) or index != policy.selected_index
    ):
        raise ValueError("execution device must match explicit GPU admission policy selection")
    device = torch.device(name)
    if device.type == "cuda":
        discovery = discover_devices()
        if not discovery["cuda_available"] or index >= discovery["device_count"]:
            raise DeviceAdmissionError("selected CUDA device is not available")
        if policy is None and (index != 0 or discovery["device_count"] != 1):
            raise DeviceAdmissionError(
                "legacy CUDA profile requires exactly one device; select a named policy"
            )
        if torch.cuda.is_initialized() and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
            raise DeviceAdmissionError("configure deterministic cuBLAS before CUDA initialization")
        workspace = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if workspace != ":4096:8":
            raise ValueError("CUDA profile requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
        torch.cuda.set_device(device)
        free, total = torch.cuda.mem_get_info(device)
        cap = policy.allocator_cap_bytes if policy else CUDA_ALLOCATOR_BUDGET_BYTES
        headroom = policy.free_headroom_bytes if policy else CUDA_FREE_HEADROOM_BYTES
        admit_memory(free, total, cap, headroom, startup=True)
        torch.cuda.set_per_process_memory_fraction(cap / total, device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return device


def cuda_rng_state(device: torch.device):
    return [torch.cuda.get_rng_state(device)] if device.type == "cuda" else None


def validate_cuda_rng(state, device: torch.device):
    if device.type == "cpu":
        if state is not None:
            raise ValueError("CPU checkpoint must not contain CUDA RNG state")
        return
    expected = cuda_rng_state(device)
    if not isinstance(state, list) or len(state) != len(expected):
        raise ValueError("CUDA RNG device-count mismatch")
    for value, reference in zip(state, expected, strict=True):
        if (
            not isinstance(value, torch.Tensor)
            or value.device.type != "cpu"
            or value.dtype != torch.uint8
            or value.shape != reference.shape
        ):
            raise ValueError("invalid CUDA RNG state")
        torch.Generator(device=device).set_state(value)
