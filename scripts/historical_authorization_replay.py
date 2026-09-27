"""Verify and replay historical authorization without mutating its frozen snapshot."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_RELATIVE = Path("data/pilot-2m-r1/provenance/authorization")
HISTORICAL_COMMIT = "720af6e63ee8759654b287848173bbd3e74867c4"
MANIFEST_SHA256 = "8461a8d3aaad724a6880627a91520257ab743dae3713c95ad592b5467395ad8d"
HISTORICAL_REPORT_SHA256 = "6658bb3c6f1e973870c744ad26288880626132dbc6aefb96a0b32c5433ec52e9"
# These supplemental replay files are outside the original 57-entry manifest.
# Each was independently matched to LF or CRLF bytes at HISTORICAL_COMMIT.
REPLAY_FILES = {
    "scripts/benchmark_fingerprints.py": (
        "b0b772942cf971caf43b807ec96d4a888cfb9af9bb1d2b59aa7ed82a568a15c5"
    ),
    "tests/test_authorization_capacity.py": (
        "71bb617d95fba6def28a5999891990185d73182edeb23adcc74471cee69b0b40"
    ),
    "tests/test_authorization_r3.py": (
        "8f64b921e847eea2475c2029cad1e13cc15f299d55fa72e7e2c8c3922e5efa33"
    ),
    "tests/test_benchmark_fingerprints.py": (
        "777ea36cb30694b19f121dd0b11423e12b06dce076e646db69d88884d1bbae97"
    ),
}
HISTORICAL_TESTS = tuple(name for name in REPLAY_FILES if name.startswith("tests/"))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def checked_bytes(root, relative, expected):
    logical = PurePosixPath(relative)
    if logical.is_absolute() or ".." in logical.parts or logical.as_posix() != relative:
        raise ValueError(f"noncanonical snapshot path: {relative}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"snapshot path escapes root: {relative}")
    data = path.read_bytes()
    if digest(data) != expected:
        raise ValueError(f"snapshot hash mismatch: {relative}")
    return data


def verify_snapshot(root=ROOT):
    snapshot = Path(root) / SNAPSHOT_RELATIVE
    raw = (snapshot.parent / "authorization_snapshot.json").read_bytes()
    if digest(raw) != MANIFEST_SHA256:
        raise ValueError("SNAPSHOT_IDENTITY_MISMATCH: original manifest-file SHA-256")
    manifest = json.loads(raw)
    if manifest["commit"] != HISTORICAL_COMMIT or len(manifest["files"]) != 57:
        raise ValueError("snapshot commit/file-count mismatch")
    if manifest["files"]["reports/real_corpus_2m_evidence.json"] != HISTORICAL_REPORT_SHA256:
        raise ValueError("historical report binding mismatch")
    entries = {**manifest["files"], **REPLAY_FILES}
    payloads = {name: checked_bytes(snapshot, name, checksum) for name, checksum in entries.items()}
    return snapshot, payloads


def replay(root=ROOT):
    snapshot, payloads = verify_snapshot(root)
    # Never run pytest in the immutable directory (cache/bytecode could modify it).
    # No corpus files are copied; inputs are hash-bound reports, contracts and test code.
    with tempfile.TemporaryDirectory(prefix="edge-authorization-replay-") as temporary:
        replay_root = Path(temporary)
        for name, data in payloads.items():
            destination = replay_root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        command = [
            sys.executable,
            "-B",
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "--rootdir",
            str(replay_root),
            *HISTORICAL_TESTS,
        ]
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
        # The child is an independent historical suite, not a recursive current-suite run.
        env.pop("PYTEST_ADDOPTS", None)
        result = subprocess.run(command, cwd=replay_root, env=env, capture_output=True, text=True)
    # Detect mutation during replay; output is not accepted without post-verification.
    _, after = verify_snapshot(root)
    if after != payloads:
        raise ValueError("snapshot changed during replay")
    return {
        "snapshot_root": str(snapshot),
        "manifest_sha256": MANIFEST_SHA256,
        "manifest_entries_verified": 57,
        "supplemental_replay_files_verified": len(REPLAY_FILES),
        "command": command,
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "snapshot_post_verification": "PASS",
        "training_executed": False,
        "test_payload_accessed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    result = replay(args.root)
    print(json.dumps(result, indent=2))
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
