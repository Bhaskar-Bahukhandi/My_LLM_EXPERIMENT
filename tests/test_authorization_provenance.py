"""Current evidence context and replay fail-closed checks; no model or corpus reads."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "authorization_replay_checks", ROOT / "scripts/historical_authorization_replay.py"
)
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


def test_current_report_is_bound_by_final_corpus_evidence():
    current_path = "reports/real_corpus_2m_evidence.json"
    current = json.loads((ROOT / current_path).read_bytes())
    final = json.loads((ROOT / "reports/real_corpus_final_evidence.json").read_bytes())
    validation = json.loads((ROOT / "reports/real_corpus_final_validation.json").read_bytes())
    assert (
        replay.digest((ROOT / current_path).read_bytes()) == final["preserved_sha256"][current_path]
    )
    assert (
        replay.digest((ROOT / "reports/real_corpus_final_evidence.json").read_bytes())
        == (validation["evidence_sha256"])
    )
    assert current["schema_version"] == 2
    assert current["historical_authorization"]["snapshot_sha256"] == replay.MANIFEST_SHA256
    assert current["historical_authorization"]["root"] == replay.SNAPSHOT_RELATIVE.as_posix()
    assert (
        "Current-root historical tests would reject"
        in current["validation"]["historical_test_context"]
    )


def test_snapshot_is_complete_and_historical_tests_remain_unchanged():
    _, payloads = replay.verify_snapshot()
    for name in (
        *replay.HISTORICAL_TESTS,
        "scripts/authorization_capacity.py",
        "scripts/benchmark_fingerprints.py",
    ):
        assert (ROOT / name).read_bytes() == payloads[name]


def test_missing_snapshot_fails_closed(tmp_path):
    with pytest.raises(FileNotFoundError):
        replay.verify_snapshot(tmp_path)


def test_manifest_mutation_fails_before_replay(tmp_path):
    metadata = tmp_path / replay.SNAPSHOT_RELATIVE.parent / "authorization_snapshot.json"
    metadata.parent.mkdir(parents=True)
    original = (ROOT / replay.SNAPSHOT_RELATIVE.parent / metadata.name).read_bytes()
    metadata.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="SNAPSHOT_IDENTITY_MISMATCH"):
        replay.verify_snapshot(tmp_path)


def test_artifact_mutation_and_path_escape_fail_closed(tmp_path):
    path = tmp_path / "report.json"
    original = b'{"status":"historical"}\n'
    path.write_bytes(original)
    expected = replay.digest(original)
    assert replay.checked_bytes(tmp_path, path.name, expected) == original
    path.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="snapshot hash mismatch"):
        replay.checked_bytes(tmp_path, path.name, expected)
    with pytest.raises(ValueError, match="noncanonical"):
        replay.checked_bytes(tmp_path, "../outside.json", expected)
