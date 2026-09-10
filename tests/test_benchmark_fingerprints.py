"""Regression checks for evaluation-only fingerprint semantics."""

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "benchmark_fingerprints", Path(__file__).parents[1] / "scripts/benchmark_fingerprints.py"
)
fp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fp)


def test_detection_normalization_does_not_modify_input():
    value = "Ａlpha\tBeta\nGAMMA delta epsilon"
    original = value
    assert fp.fingerprint(value) == fp.fingerprint("alpha beta gamma delta epsilon")
    assert value == original
    assert fp.fingerprint(value) != fp.fingerprint("alpha beta gamma delta changed")


def test_near_signature_is_ordered_bounded_and_detects_small_edit():
    text = " ".join(f"word{i}" for i in range(300))
    left = fp.fingerprint(text)
    right = fp.fingerprint(text.replace("word150", "replacement"))
    assert left["bottom64"] == sorted(set(left["bottom64"]))
    assert len(left["bottom64"]) == 64
    assert len(set(left["bottom64"]) & set(right["bottom64"])) >= 55
    assert fp.fingerprint("1 + 2")["bottom64"] == []


def test_mbpp_original_split_boundaries_are_not_randomized():
    expected = {
        1: "prompt",
        10: "prompt",
        11: "test",
        510: "test",
        511: "validation",
        600: "validation",
        601: "train",
        974: "train",
    }
    assert {key: fp.mbpp_split(key) for key in expected} == expected
    with pytest.raises(ValueError):
        fp.mbpp_split(975)


def test_raw_mutation_is_rejected_before_parsing(tmp_path):
    path = tmp_path / "evaluation.jsonl"
    data = b'{"question":"example"}\n'
    path.write_bytes(data)
    item = {"local_path": str(path), "raw_sha256": fp.sha(data)}
    assert fp.load_rows(item) == [{"question": "example"}]
    path.write_bytes(data + b" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        fp.load_rows(item)


def test_nested_fields_keep_problem_solution_and_tests_separate():
    fields = dict(fp.strings({"problem": "p", "solution": "s", "tests": ["t1", "t2"]}))
    assert fields == {"/problem": "p", "/solution": "s", "/tests/0": "t1", "/tests/1": "t2"}
