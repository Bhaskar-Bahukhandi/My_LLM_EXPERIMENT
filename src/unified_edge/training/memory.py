"""Tensor payloads and Windows process working set, kept as distinct measurements."""

import ctypes
import os
from ctypes import wintypes

import torch


def process_memory() -> dict:
    if os.name != "nt":
        return {"process_rss_status": "UNVERIFIED", "reason": "Windows measurement only"}

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
        return {
            "process_rss_status": "MEASURED",
            "method": "GetProcessMemoryInfo WorkingSetSize",
            "process_rss_bytes": data.WorkingSetSize,
            "process_peak_working_set_bytes": data.PeakWorkingSetSize,
        }
    except OSError as error:
        return {"process_rss_status": "UNVERIFIED", "reason": str(error)}


def training_memory(model, optimizer, batch: int) -> dict:
    parameters = list(model.parameters())
    optimizer_tensors = [
        v
        for state in optimizer.state.values()
        for v in state.values()
        if isinstance(v, torch.Tensor)
    ]
    state = model.shared.initialize_state(batch)
    recurrent = [t for layer in state.layers for t in (layer.conv, layer.ssm)]

    def payload(tensors):
        unique = {id(t): t for t in tensors}
        return sum(t.numel() * t.element_size() for t in unique.values())

    return {
        "parameter_bytes": payload(parameters),
        "gradient_bytes": payload([p.grad for p in parameters if p.grad is not None]),
        "optimizer_state_bytes": payload(optimizer_tensors),
        "canonical_recurrent_state_bytes": payload(recurrent),
        "recurrent_note": "standalone canonical allocation; not persistent trainer cache",
        "activation_autograd_bytes": "UNMEASURED_SEPARATELY",
        **process_memory(),
    }
