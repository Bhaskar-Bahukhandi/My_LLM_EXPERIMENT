"""Gutenberg identity gates never silently substitute an ebook or changed edition."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import gutenberg_acquisition as g  # noqa: E402


def test_approved_book_preservation_resume_and_changed_header(monkeypatch, tmp_path):
    raw = b"Approved original edition header\nNovel bytes.\n"
    work = {
        "id": 1400,
        "decision": "APPROVE_ORIGINAL_TEXT_ONLY",
        "files": [{"bytes": len(raw)}],
        "header": {"header_bytes": 10, "header_sha256": hashlib.sha256(raw[:10]).hexdigest()},
    }
    auth = tmp_path / "auth/docs"
    auth.mkdir(parents=True)
    (auth / "gutenberg_work_allowlist_2m.json").write_text(
        json.dumps({"works": [work]}), encoding="utf-8"
    )
    monkeypatch.setattr(g, "AUTH", tmp_path / "auth")
    monkeypatch.setattr(g, "PILOT", tmp_path / "pilot")
    monkeypatch.setattr(g, "require_space", lambda additional: 0)
    monkeypatch.setattr(g, "download", lambda work: raw)
    first = g.acquire(1400)
    assert (g.PILOT / first["raw_path"]).read_bytes() == raw
    monkeypatch.setattr(g, "download", lambda work: pytest.fail("redownload"))
    assert g.acquire(1400) == first
    with pytest.raises(ValueError, match="not an approved"):
        g.acquire(1260)
    monkeypatch.setattr(g, "PILOT", tmp_path / "changed")
    monkeypatch.setattr(g, "download", lambda work: b"Changed!" + raw[8:])
    with pytest.raises(ValueError, match="quarantined"):
        g.acquire(1400)
    assert not (g.PILOT / "raw/gutenberg/pg1400.txt").exists()
    assert (g.PILOT / "quarantine/gutenberg-1400.json").exists()
