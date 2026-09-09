"""Binary-safe payload conversion and controls from the canonical configuration registry."""

from collections.abc import Iterable, Sequence
from types import MappingProxyType

import torch

from unified_edge.config import CONTROL_IDS, NUM_SYMBOLS

CONTROL_BY_NAME = MappingProxyType(dict(CONTROL_IDS))
NAME_BY_CONTROL = MappingProxyType({value: name for name, value in CONTROL_IDS})
PAD_ID = CONTROL_BY_NAME["pad"]
BOS_ID = CONTROL_BY_NAME["bos"]
EOS_ID = CONTROL_BY_NAME["eos"]


def bytes_to_symbols(payload: bytes | bytearray | memoryview) -> list[int]:
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("payload must be bytes, bytearray or memoryview; text is not normalized")
    return list(bytes(payload))


def symbols_to_bytes(symbols: Iterable[int]) -> bytes:
    values = []
    for value in symbols:
        if type(value) is not int or not 0 <= value <= 255:
            raise ValueError(f"payload byte ID must be an integer in 0..255, got {value!r}")
        values.append(value)
    return bytes(values)


def control_id(name: str) -> int:
    if not isinstance(name, str) or name not in CONTROL_BY_NAME:
        raise ValueError(f"unknown control name {name!r}")
    return CONTROL_BY_NAME[name]


def control_name(value: int) -> str:
    if type(value) is not int or value not in NAME_BY_CONTROL:
        raise ValueError(f"unknown control ID {value!r}")
    return NAME_BY_CONTROL[value]


def validate_symbols(
    ids: torch.Tensor, *, allow_padding: bool = True, allow_bos: bool = True
) -> None:
    if not isinstance(ids, torch.Tensor) or ids.dtype != torch.long:
        raise TypeError("symbol IDs must be a torch.int64 tensor")
    invalid = (ids < 0) | (ids >= NUM_SYMBOLS)
    if invalid.any():
        raise ValueError(
            f"unknown symbol ID {ids[invalid].flatten()[0].item()}; expected 0..{NUM_SYMBOLS - 1}"
        )
    if not allow_padding and (ids == PAD_ID).any():
        raise ValueError("PAD is storage only and cannot be consumed or complete a patch")
    if not allow_bos and (ids == BOS_ID).any():
        raise ValueError("BOS is out-of-band bootstrap conditioning, not a payload slot")


def pad_symbol_sequences(
    sequences: Sequence[Sequence[int]], patch_size: int = 8
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return right-padded storage and real lengths; padding is not an observed symbol."""
    if type(patch_size) is not int or patch_size != 8:
        raise ValueError("symbol contract v1 requires patch_size=8")
    if not sequences:
        raise ValueError("at least one sequence is required")
    for row in sequences:
        if any(type(value) is not int for value in row):
            raise TypeError("sequence IDs must be Python integers")
        validate_symbols(torch.tensor(row, dtype=torch.long), allow_padding=False, allow_bos=False)
    lengths = torch.tensor([len(row) for row in sequences], dtype=torch.long)
    width = ((int(lengths.max()) + patch_size - 1) // patch_size) * patch_size
    storage = torch.full((len(sequences), width), PAD_ID, dtype=torch.long)
    for index, row in enumerate(sequences):
        storage[index, : len(row)] = torch.tensor(row, dtype=torch.long)
    return storage, lengths


def mask_non_emittable(logits: torch.Tensor) -> torch.Tensor:
    """PAD/BOS are intentionally impossible outputs; the other controls remain model symbols."""
    if logits.shape[-1] != NUM_SYMBOLS:
        raise ValueError(f"expected {NUM_SYMBOLS} output classes")
    indices = torch.tensor([PAD_ID, BOS_ID], device=logits.device)
    return logits.index_fill(-1, indices, -torch.inf)
