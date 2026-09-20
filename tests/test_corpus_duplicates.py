"""Duplicate decisions must not depend on quotas, input order or normalization bytes."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from corpus_duplicates import deduplicate  # noqa: E402


def document(identity, split, text):
    return {"identity": identity, "split": split, "payload": text.encode("utf-8")}


def test_exact_normalized_cross_split_cluster_is_completely_excluded():
    docs = [document("b", "test", "Public   TEXT"), document("a", "train", "public text")]
    before = [d["payload"] for d in docs]
    result = deduplicate(docs)
    assert all(not d["keep"] for d in result["decisions"].values())
    assert all(d["reason"] == "CROSS_SPLIT_DUPLICATE" for d in result["decisions"].values())
    assert [d["payload"] for d in docs] == before
    assert result == deduplicate(list(reversed(docs)))


def test_verified_near_duplicate_and_different_content():
    text = " ".join(f"token{i}" for i in range(200))
    docs = [
        document("a", "train", text),
        document("b", "train", text + " extra"),
        document("c", "train", "Entirely different mathematical discussion of integration"),
    ]
    result = deduplicate(docs)
    assert result["decisions"]["a"]["keep"]
    assert not result["decisions"]["b"]["keep"]
    assert result["decisions"]["c"]["keep"]
    assert any(e["method"] == "pilot-near-v1" for e in result["verified_edges"])
