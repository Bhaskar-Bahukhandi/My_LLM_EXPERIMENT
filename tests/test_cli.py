import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/models/edge_2m.yaml"


def run_cli(*args, seed="1"):
    env = {**os.environ, "PYTHONHASHSEED": seed}
    return subprocess.run(
        [sys.executable, "-m", "unified_edge.cli", *map(str, args)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_cli_reproducible_across_process_hash_seeds_and_does_not_overwrite(tmp_path):
    output = tmp_path / "audit.json"
    first = run_cli(CONFIG, "--output", output, seed="1")
    second = run_cli(CONFIG, seed="789")
    assert first.returncode == second.returncode == 2
    assert first.stdout == second.stdout
    assert json.loads(output.read_text()) == json.loads(first.stdout)
    original = output.read_bytes()
    overwrite = run_cli(CONFIG, "--output", output)
    assert overwrite.returncode == 1
    assert output.read_bytes() == original
    assert "configuration error" in overwrite.stderr


@pytest.mark.parametrize(
    "model,exit_code,status",
    [
        ({"parameter_tolerance": 0.05}, 0, "WITHIN_TARGET"),
        ({"d_model": 130}, 3, "NO_LEGAL_CANDIDATE"),
    ],
)
def test_cli_other_result_classes(tmp_path, model, exit_code, status):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"schema_version": "1", "model": model}), encoding="utf-8")
    result = run_cli(config)
    assert result.returncode == exit_code
    assert json.loads(result.stdout)["status"] == status


def test_cli_invalid_config_fails_without_report(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text('schema_version: "1"\nunknown: 1\n', encoding="utf-8")
    output = tmp_path / "must_not_exist.json"
    result = run_cli(config, "--output", output)
    assert result.returncode == 1
    assert "unknown" in result.stderr
    assert not output.exists()
