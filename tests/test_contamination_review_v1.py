"""Adversarial tests for an unadopted support-field exception."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from corpus_matcher_v2_proposal import ProposedIndex, isolated_import  # noqa: E402


def field(text, path, identity="example/test/1"):
    return {"text": text, "field_path": path, "identity": identity + path}


@pytest.mark.parametrize("path", ["/problem", "/canonical_solution", "/answer"])
def test_full_problem_and_answer_remain_protected_with_normalization(path):
    text = "Find the distinctive triangular lattice boundary with seventeen missing vertices."
    index = ProposedIndex([field(text, path)])
    assert index.match(text)
    assert index.match("prefix " + text.upper().replace(" ", "  ") + " suffix")


def test_import_exception_does_not_mask_real_answer_or_near_match():
    answer = " ".join(f"term{i}" for i in range(100))
    fields = [field("import math", "/test_imports/0"), field(answer, "/canonical_solution")]
    index = ProposedIndex(fields)
    assert index.match("ordinary prefix\nimport math\nordinary suffix") is None
    payload = "import math\n\n" + answer + "\n\nimport math"
    assert index.match(payload)
    near = index.match(answer.replace("term50", "changed"))
    assert near["method"] == "VERIFIED_PARAGRAPH_OR_WINDOW_JACCARD_0.85"
    assert ProposedIndex(reversed(fields)).match(payload) == index.match(payload)


def test_complete_short_answers_override_support_exception_and_frequency():
    index = ProposedIndex(
        [field("import math", "/test_imports/0"), field("import math", "/answer")]
    )
    assert index.match("import math")
    for text in ("return 1", "25", "return x + y"):
        assert ProposedIndex([field(text, "/canonical_solution")]).match(text)


def test_metadata_and_non_import_support_are_distinguished():
    metadata = "Distinctive metadata is not an actual benchmark problem or answer."
    assert ProposedIndex([field(metadata, "/source_file")]).match(metadata) is None
    code = "import math\nassert 7 == 7"
    assert not isolated_import(code)
    assert ProposedIndex([field(code, "/test_imports/0")]).match(code)
    assert isolated_import("from collections import deque")
    assert not isolated_import("not valid Python !")
