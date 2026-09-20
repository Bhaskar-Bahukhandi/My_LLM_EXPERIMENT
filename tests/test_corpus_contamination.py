"""Benchmark matching verifies candidates and preserves short-field boundaries."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from corpus_contamination import ExclusionIndex  # noqa: E402


def test_normalized_embedded_problem_and_short_boundary():
    index = ExclusionIndex(
        [
            {
                "identity": "benchmark/test/1",
                "text": "Calculate the volume of a sphere of radius seventeen.",
            },
            {"identity": "benchmark/test/2", "text": "tiny answer"},
        ]
    )
    assert index.match("Prefix CALCULATE  THE VOLUME OF A SPHERE OF RADIUS SEVENTEEN. suffix")
    assert index.match("prefix\ntiny answer\nsuffix")["method"] == "SHORT_EXACT_DOCUMENT_OR_LINE"
    assert index.match("prefix tiny answer suffix") is None


def test_near_match_is_verified_and_fingerprint_binding_is_strict():
    problem = " ".join(f"word{i}" for i in range(100))
    index = ExclusionIndex([{"identity": "problem/1", "text": problem}])
    match = index.match(problem.replace("word50", "changed"))
    assert match["method"] == "VERIFIED_PARAGRAPH_OR_WINDOW_JACCARD_0.85"
    assert index.match("Unrelated explanation without shared problem material.") is None
    with pytest.raises(ValueError, match="fingerprint/content mismatch"):
        ExclusionIndex(
            [{"identity": "bad", "text": problem, "fingerprint": {"normalized_sha256": "invalid"}}]
        )
