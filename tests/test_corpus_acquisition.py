"""Pinned acquisition preserves bytes and fails before admitting mismatches."""

import hashlib
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "acquisition", Path(__file__).parents[1] / "scripts/corpus_acquisition.py"
)
a = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(a)


def setup(monkeypatch, tmp_path, payload):
    monkeypatch.setattr(a, "PILOT", tmp_path / "pilot")
    monkeypatch.setattr(a, "ROOT", tmp_path)
    monkeypatch.setattr(a, "fetch", lambda url, limit: payload)
    return {
        "source": "source",
        "path": "docs/example.py",
        "commit": "a" * 40,
        "repository": "https://github.com/owner/repo",
        "domain": "code",
        "source_unit": "example",
        "reserved_split": "train",
        "raw_bytes_from_tree": len(payload),
        "git_blob_sha1": hashlib.sha1(
            b"blob " + str(len(payload)).encode() + b"\0" + payload
        ).hexdigest(),
    }


def test_preserves_arbitrary_bytes_and_resumes_without_fetch(monkeypatch, tmp_path):
    payload = b"\x00\xff\x80not executable\n"
    file = setup(monkeypatch, tmp_path, payload)
    first = a.acquire_file(file)
    monkeypatch.setattr(a, "fetch", lambda *args: pytest.fail("unexpected redownload"))
    assert a.acquire_file(file) == first
    raw = a.PILOT / first["raw_path"]
    assert raw.read_bytes() == payload
    raw.write_bytes(b"changed")
    with pytest.raises(ValueError, match="mutated"):
        a.acquire_file(file)


def test_mismatch_is_quarantined_without_admission(monkeypatch, tmp_path):
    file = setup(monkeypatch, tmp_path, b"expected")
    monkeypatch.setattr(a, "fetch", lambda *args: b"wrong revision")
    with pytest.raises(ValueError, match="quarantined"):
        a.acquire_file(file)
    assert not list((a.PILOT / "raw").rglob("*"))
    assert list((a.PILOT / "quarantine").glob("*.json"))


def test_storage_gate_precedes_fetch(monkeypatch, tmp_path):
    file = setup(monkeypatch, tmp_path, b"data")
    monkeypatch.setattr(a, "DATA_CAP", 1)
    monkeypatch.setattr(a, "fetch", lambda *args: pytest.fail("download before gate"))
    with pytest.raises(RuntimeError, match="storage cap"):
        a.acquire_file(file)


def test_response_size_bound_is_enforced(monkeypatch):
    import io

    reads = []

    class Response(io.BytesIO):
        def read(self, size=-1):
            reads.append(size)
            return super().read(size)

    monkeypatch.setattr(a.urllib.request, "urlopen", lambda *args, **kwargs: Response(b"x" * 20))
    with pytest.raises(ValueError, match="declared byte bound"):
        a.fetch("https://raw.githubusercontent.com/owner/repo/pin/file", 8)
    assert reads == [9]


def test_disk_accounting_matches_the_original_full_scan(monkeypatch, tmp_path):
    monkeypatch.setattr(a, "ROOT", tmp_path)
    monkeypatch.setattr(a, "PILOT", tmp_path / "pilot")
    roots = [a.PILOT] + [
        tmp_path / "evidence" / name
        for name in ("download_authorization", "authorization_r2", "authorization_r3")
    ]
    for index, root in enumerate(roots):
        (root / "nested").mkdir(parents=True)
        (root / "nested" / "bytes").write_bytes(b"x" * (index + 1))
    (tmp_path / "unrelated").write_bytes(b"excluded")
    expected = sum(p.stat().st_size for root in roots for p in root.rglob("*") if p.is_file())
    assert a.disk_bytes() == expected == 10
