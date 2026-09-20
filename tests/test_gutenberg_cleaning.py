"""Book cleaning must preserve every retained byte and reject stale reviews."""

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from gutenberg_cleaning import reviewed_payload  # noqa: E402


def test_exact_book_removals_and_stale_or_overlapping_reviews():
    body = b"An original novel paragraph, with its punctuation and line endings.\r\n" * 8
    raw = b"Wrapper\r\n" + body + b"Supplement\r\n"
    review = {
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "removed_ranges": [[0, 9, "WRAPPER"], [9 + len(body), len(raw), "SUPPLEMENT"]],
    }
    assert reviewed_payload(raw, review) == body
    with pytest.raises(ValueError, match="does not match"):
        reviewed_payload(raw + b"changed", review)
    review["removed_ranges"].insert(1, [8, 10, "OVERLAP"])
    with pytest.raises(ValueError, match="overlapping"):
        reviewed_payload(raw, review)
