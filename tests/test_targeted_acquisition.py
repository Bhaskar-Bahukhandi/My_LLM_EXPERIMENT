"""Ineligible metadata must fail before any targeted source download."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from corpus_targeted_acquisition import acquire_targeted, validate_tree_entry  # noqa: E402


def test_whole_batch_mode_gate_precedes_first_download():
    files = [
        {"source": "source", "path": p, "git_blob_sha1": p, "raw_bytes_from_tree": 10}
        for p in ("eligible.py", "executable.py")
    ]
    tree = {
        f["path"]: {
            "path": f["path"],
            "type": "blob",
            "sha": f["git_blob_sha1"],
            "size": 10,
            "mode": "100644",
        }
        for f in files
    }
    tree["executable.py"]["mode"] = "100755"
    calls = []
    with pytest.raises(ValueError, match="mode-100644"):
        acquire_targeted(files, {"source": tree}, acquire=calls.append)
    assert calls == []
    tree["executable.py"]["mode"] = "120000"
    with pytest.raises(ValueError, match="mode-100644"):
        acquire_targeted(files, {"source": tree}, acquire=calls.append)
    assert calls == []
    tree["executable.py"]["mode"] = "100644"
    acquire_targeted(files, {"source": tree}, acquire=calls.append)
    assert calls == files


def test_tree_size_and_blob_binding_fail_closed():
    file = {"path": "source.py", "git_blob_sha1": "expected", "raw_bytes_from_tree": 10}
    entry = {"path": "source.py", "type": "blob", "mode": "100644", "sha": "changed", "size": 10}
    with pytest.raises(ValueError, match="pinned tree"):
        validate_tree_entry(file, entry)
    entry.update(sha="expected", size=11)
    with pytest.raises(ValueError, match="pinned tree"):
        validate_tree_entry(file, entry)
