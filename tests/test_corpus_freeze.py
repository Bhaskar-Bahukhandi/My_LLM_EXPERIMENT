"""Final selection preserves bytes, quotas, split boundaries and hash verification."""

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from corpus_freeze import loader_check, select_documents, verify_manifest  # noqa: E402


def document(identity, split, raw):
    return {
        "identity": identity,
        "source": "source",
        "domain": "code",
        "split": split,
        "source_unit": identity,
        "payload": raw,
    }


def test_selection_is_deterministic_and_does_not_borrow_or_reencode():
    docs = [document("b", "train", b"second"), document("a", "train", b"\x00\xff\x80abc")]
    quotas = [{"source": "source", "domain": "code", "split": "train", "required_bytes": 4}]
    selected, cells = select_documents(docs, quotas)
    assert selected[0]["selected_payload"] == b"\x00\xff\x80a"
    assert (selected, cells) == select_documents(list(reversed(docs)), quotas)
    with pytest.raises(ValueError, match="no borrowing"):
        select_documents(docs, [dict(quotas[0], split="test", required_bytes=1)])
    overlapping = document("a", "test", b"different")
    with pytest.raises(ValueError, match="crosses splits"):
        select_documents(docs + [overlapping], quotas)


def test_three_split_loader_round_trip_and_mutation_detection(tmp_path):
    records, totals, hashes = [], {}, {}
    for split in ("train", "validation", "test"):
        raw = b"\x00\x80\xff\xc3" + split.encode() * 20
        path = split + ".raw"
        (tmp_path / path).write_bytes(raw)
        checksum = hashlib.sha256(raw).hexdigest()
        records.append(
            {
                "path": path,
                "split": split,
                "domain": "code",
                "sha256": checksum,
                "byte_count": len(raw),
            }
        )
        totals[split], hashes[split] = len(raw), checksum
    manifest = {
        "corpus_id": "byte-contract-test",
        "documents": records,
        "split_bytes": totals,
        "split_sha256": hashes,
    }
    training, checks = loader_check(manifest, tmp_path)
    assert {d.split for d in training.documents} == {"train", "validation"}
    assert len(checks) == 6
    assert all(c["bytes_round_tripped"] == totals[c["split"]] for c in checks)
    path = tmp_path / "test.raw"
    changed = bytearray(path.read_bytes())
    changed[0] ^= 1
    path.write_bytes(changed)
    with pytest.raises(ValueError, match="mutation"):
        verify_manifest(manifest, tmp_path)
