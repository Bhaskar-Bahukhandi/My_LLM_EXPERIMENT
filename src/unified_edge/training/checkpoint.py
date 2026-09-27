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


def parameter_inventory(model, optimizer, resolved_sha256: str) -> dict:
    """Bind canonical names and optimizer order, counting aliases exactly once."""
    names, tensors, aliases = {}, [], {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if not parameter.requires_grad:
            continue
        if id(parameter) in names:
            aliases[name] = names[id(parameter)]
            continue
        names[id(parameter)] = name
        tensors.append(
            {"name": name, "shape": list(parameter.shape), "dtype": str(parameter.dtype)}
        )
    grouped = [p for group in optimizer.param_groups for p in group["params"]]
    if len(grouped) != len({id(p) for p in grouped}) or {id(p) for p in grouped} != set(names):
        raise CheckpointError("optimizer must cover every unique trainable parameter exactly once")
    return {
        "schema": "1",
        "resolved_sha256": resolved_sha256,
        "tensors": tensors,
        "aliases": aliases,
        "unique_trainable_tensors": len(tensors),
        "optimizer_parameter_names": [names[id(p)] for p in grouped],
        "optimizer_state_semantics": (
            "Lazy AdamW state exists exactly for positive per-parameter update counts."
        ),
    }


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
        if not isinstance(state, dict) or state.get("schema") not in ("2", "3"):
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
        # Same-parent rename publishes a complete directory on supported local filesystems.
        # Run ownership is exclusive; neither API intentionally overwrites a checkpoint.
        if path.exists():
            raise FileExistsError(f"checkpoint already exists: {path.name}")
        os.rename(scratch, path)
    return path
