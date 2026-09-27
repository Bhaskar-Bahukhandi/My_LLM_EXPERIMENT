"""Tensor payloads and OS-reported host resident memory; distinct from GPU/activations."""

import ctypes
import sys
from ctypes import wintypes
from pathlib import Path

import torch


def linux_rss(status: str) -> dict:
    """Parse Linux /proc/self/status kB units (1024 bytes), not decimal kilobytes."""
    fields = {}
    for line in status.splitlines():
        key, _, value = line.partition(":")
        if key in ("VmRSS", "VmHWM"):
            parts = value.split()
            if key in fields or len(parts) != 2 or parts[1] != "kB" or not parts[0].isdigit():
                raise ValueError("invalid Linux RSS field/units")
            fields[key] = int(parts[0]) * 1024
    if "VmRSS" not in fields:
        raise ValueError("Linux status has no VmRSS")
    return {"rss_bytes": fields["VmRSS"], "peak_rss_bytes": fields.get("VmHWM")}


def process_memory() -> dict:
    method = (
        "GetProcessMemoryInfo WorkingSetSize" if sys.platform == "win32" else "/proc/self/status"
    )
    try:
        if sys.platform == "win32":
            measured = _windows_memory()
        elif sys.platform.startswith("linux"):
            measured = linux_rss(Path("/proc/self/status").read_text(encoding="ascii"))
        else:
            return {
                "rss_bytes": None,
                "source": None,
                "method": None,
                "supported": False,
                "process_rss_status": "UNSUPPORTED",
                "reason": "unsupported host platform",
            }
        return {
            **measured,
            "source": method,
            "method": method,
            "supported": True,
            "process_rss_status": "MEASURED",
            "process_rss_bytes": measured["rss_bytes"],
            "process_peak_working_set_bytes": measured.get("peak_rss_bytes"),
        }
    except (OSError, ValueError) as error:
        return {
            "rss_bytes": None,
            "source": method,
            "method": method,
            "supported": True,
            "process_rss_status": "UNVERIFIED",
            "reason": str(error),
        }


def _windows_memory() -> dict:
    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    try:
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        data = Counters()
        data.cb = ctypes.sizeof(data)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(data), data.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return {"rss_bytes": data.WorkingSetSize, "peak_rss_bytes": data.PeakWorkingSetSize}
    except OSError as error:
        raise OSError(f"Windows working set unavailable: {error}") from error


def training_memory(model, optimizer, batch: int) -> dict:
    parameters = list(model.parameters())
    optimizer_tensors = [
        v
        for state in optimizer.state.values()
        for v in state.values()
        if isinstance(v, torch.Tensor)
    ]
    shape = model.config.shape
    # Identical canonical state axes without allocating and zeroing scratch tensors.
    recurrent_bytes = (
        4
        * batch
        * shape.shared_layers
        * ((shape.d_inner + 2 * shape.d_state) * shape.d_conv + shape.d_inner * shape.d_state)
    )

    def payload(tensors):
        unique = {id(t): t for t in tensors}
        return sum(t.numel() * t.element_size() for t in unique.values())

    return {
        "parameter_bytes": payload(parameters),
        "gradient_bytes": payload([p.grad for p in parameters if p.grad is not None]),
        "optimizer_state_bytes": payload(optimizer_tensors),
        "canonical_recurrent_state_bytes": recurrent_bytes,
        "recurrent_note": "analytic canonical FP32 axes; no allocation or persistent trainer cache",
        "activation_autograd_bytes": "UNMEASURED_SEPARATELY",
        **process_memory(),
        **({"cuda": cuda_memory(parameters[0].device)} if parameters[0].is_cuda else {}),
    }


def cuda_memory(device: torch.device) -> dict:
    """Synchronized allocator metrics and driver free memory; neither is process RSS."""
    if device.type != "cuda":
        raise ValueError("CUDA memory measurement requires a CUDA device")
    torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    return {
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "free_device_bytes": free,
        "total_device_bytes": total,
    }
