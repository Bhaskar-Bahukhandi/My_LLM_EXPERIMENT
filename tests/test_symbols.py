import random

import pytest
import torch

from unified_edge.config import CONTROL_IDS, NUM_SYMBOLS
from unified_edge.symbols import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    bytes_to_symbols,
    control_id,
    control_name,
    pad_symbol_sequences,
    symbols_to_bytes,
    validate_symbols,
)


def test_all_bytes_binary_and_no_text_normalization():
    rng = random.Random(41)
    payloads = [
        bytes(range(256)),
        b"\x00" * 17,
        b"\xff\xc0\x80\x00",
        bytes(range(128, 256)),
        bytes(rng.randrange(256) for _ in range(1025)),
        b"",
    ]
    for payload in payloads:
        assert symbols_to_bytes(bytes_to_symbols(payload)) == payload
        assert bytes_to_symbols(bytearray(payload)) == list(payload)
        assert bytes_to_symbols(memoryview(payload)) == list(payload)
    with pytest.raises(TypeError, match="text is not normalized"):
        bytes_to_symbols("é\r\n")


def test_control_registry_has_one_source_and_no_collisions():
    assert NUM_SYMBOLS == 267
    assert {value for _, value in CONTROL_IDS} == set(range(256, 267))
    assert len({0, PAD_ID, EOS_ID, BOS_ID}) == 4
    for name, value in CONTROL_IDS:
        assert control_id(name) == value
        assert control_name(value) == name
        with pytest.raises(ValueError, match="payload byte ID"):
            symbols_to_bytes([value])
    for invalid in ["new_control", "EOS", 258, None]:
        with pytest.raises(ValueError, match="unknown control name"):
            control_id(invalid)


@pytest.mark.parametrize("value", [-1, 267, 999, True, 1.5])
def test_invalid_payload_and_control_values(value):
    with pytest.raises(ValueError):
        symbols_to_bytes([value])
    with pytest.raises(ValueError):
        control_name(value)


@pytest.mark.parametrize(
    "ids,error",
    [
        (torch.tensor([267]), ValueError),
        (torch.tensor([-1]), ValueError),
        (torch.tensor([2.0]), TypeError),
        (torch.tensor([True]), TypeError),
    ],
)
def test_tensor_ids_fail_specifically(ids, error):
    with pytest.raises(error):
        validate_symbols(ids)


def test_padding_is_storage_not_observed_input():
    lengths = [0, 1, 2, 7, 8, 9, 15, 16, 17, 127, 128, 129]
    rows = [[i % 256 for i in range(length)] for length in lengths]
    storage, sizes = pad_symbol_sequences(rows)
    assert storage.shape == (len(lengths), 136)
    assert sizes.tolist() == lengths
    for i, length in enumerate(lengths):
        assert storage[i, :length].tolist() == rows[i]
        assert (storage[i, length:] == PAD_ID).all()
        assert symbols_to_bytes(storage[i, :length].tolist()) == bytes(rows[i])
    empty, sizes = pad_symbol_sequences([[], []])
    assert empty.shape == (2, 0) and sizes.tolist() == [0, 0]
    for invalid in [PAD_ID, BOS_ID]:
        with pytest.raises(ValueError):
            pad_symbol_sequences([[invalid]])
    with pytest.raises(TypeError):
        pad_symbol_sequences([[True]])
    with pytest.raises(ValueError):
        pad_symbol_sequences([])
