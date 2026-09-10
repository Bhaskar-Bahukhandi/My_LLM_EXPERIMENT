"""Checksummed, readable-before-publish checkpoint directories; no overwrite."""

import hashlib
import json
import os
import pickle
import tempfile
from pathlib import Path

import torch


class CheckpointError(ValueError):
    """A checkpoint is corrupt or incompatible with the requested run."""


def load_checkpoint(path: Path) -> dict:
    path = Path(path)
    try:
        record = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if (
            not isinstance(record, dict)
            or set(record) != {"schema", "file", "sha256"}
            or record["schema"] != "1"
            or record["file"] != "state.pt"
        ):
            raise CheckpointError("invalid checkpoint integrity manifest")
        target = path / "state.pt"
        if hashlib.sha256(target.read_bytes()).hexdigest() != record["sha256"]:
            raise CheckpointError("checkpoint SHA-256 mismatch")
        state = torch.load(target, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or state.get("schema") != "2":
            raise CheckpointError("unsupported checkpoint schema")
        return state
    except (OSError, EOFError, RuntimeError, pickle.UnpicklingError, json.JSONDecodeError) as error:
        raise CheckpointError(f"checkpoint unreadable: {error}") from error


def save_checkpoint(path: Path, state: dict) -> Path:
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(f"checkpoint already exists: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".checkpoint-", dir=path.parent) as temporary:
        scratch = Path(temporary).resolve()
        if scratch.parent != path.parent or path == path.parent:
            raise ValueError("checkpoint publication must stay within its parent directory")
        target = scratch / "state.pt"
        with target.open("xb") as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        with (scratch / "manifest.json").open("x", encoding="utf-8") as handle:
            json.dump({"schema": "1", "file": "state.pt", "sha256": digest}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        load_checkpoint(scratch)
        # On this Windows-local profile rename fails if the destination already exists.
        if path.exists():
            raise FileExistsError(f"checkpoint already exists: {path.name}")
        os.rename(scratch, path)
    return path
