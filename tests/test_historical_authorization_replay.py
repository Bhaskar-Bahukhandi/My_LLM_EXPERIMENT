"""Required historical suite, separate from current-artifact checks."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "historical_replay", Path(__file__).parents[1] / "scripts/historical_authorization_replay.py"
)
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


@pytest.mark.historical_replay
def test_immutable_historical_authorization_suite():
    result = replay.replay()
    print(result["stdout"])
    assert result["exit_code"] == 0, result["stdout"] + result["stderr"]
    assert result["snapshot_post_verification"] == "PASS"
