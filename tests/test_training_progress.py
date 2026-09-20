"""Windows publication failures and strict replay of logged unsaved updates."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from test_training import make_data, model_config

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from training_progress import (  # noqa: E402
    compare_replay,
    journal,
    publish_progress,
    recovery_range,
)

from unified_edge.training.checkpoint import CheckpointError  # noqa: E402
from unified_edge.training.config import TrainingConfig  # noqa: E402
from unified_edge.training.trainer import Trainer  # noqa: E402


def test_transient_replace_retries_only_complete_json(tmp_path):
    path = tmp_path / "progress.json"
    path.write_text('{"step": 1}')
    calls, pauses = [], []

    def replace_file(source, destination):
        assert json.loads(Path(source).read_text()) == {"step": 2}
        assert json.loads(path.read_text()) == {"step": 1}
        calls.append(source)
        if len(calls) < 3:
            raise PermissionError("sharing violation")
        Path(source).replace(destination)

    publish_progress(path, {"step": 2}, replace=replace_file, sleep=pauses.append)
    assert json.loads(path.read_text()) == {"step": 2}
    assert pauses == [0.05, 0.1]
    assert all(Path(p).name != "progress.tmp" for p in calls)


def test_failed_or_incomplete_publication_preserves_authoritative_file(tmp_path):
    path = tmp_path / "progress.json"
    original = b'{"step": 1}'
    path.write_bytes(original)

    def denied(*args):
        raise PermissionError("held open")

    with pytest.raises(PermissionError):
        publish_progress(path, {"step": 2}, replace=denied, sleep=lambda _: None, attempts=2)
    with pytest.raises(ValueError):
        publish_progress(path, {"step": float("nan")}, replace=denied)
    assert path.read_bytes() == original


def test_real_checkpoint_replay_and_journal_ahead_of_progress(tmp_path):
    root = tmp_path / "data"
    manifest = make_data(root)
    config = TrainingConfig(total_steps=5, warmup_steps=1)
    trainer = Trainer(model_config(), config, manifest, root, tmp_path / "original")
    rows = [trainer.step()]
    checkpoint = trainer.save()
    rows += [trainer.step(), trainer.step()]
    log = tmp_path / "observations.jsonl"

    def write(values):
        log.write_text("".join(json.dumps(row) + "\n" for row in values))

    write(rows)
    parsed = journal([log])
    assert recovery_range(parsed, 1, 2) == rows[1:]
    assert recovery_range(parsed, 1, 1) == rows[1:]
    resumed = Trainer(
        model_config(), config, manifest, root, tmp_path / "restored", resume_from=checkpoint
    )
    for expected in recovery_range(parsed, 1, 2):
        actual = resumed.step()
        compare_replay(actual, expected)
        with pytest.raises(ValueError, match="replay mismatch"):
            compare_replay(dict(actual, raw_byte_nll=actual["raw_byte_nll"] + 0.1), expected)
    for invalid in (rows + [rows[-1]], [rows[1], rows[0], rows[2]], [rows[0], rows[2]]):
        write(invalid)
        with pytest.raises(ValueError, match="duplicate, reordered or missing"):
            journal([log])
    with pytest.raises(CheckpointError):
        Trainer(
            model_config(),
            replace(config, learning_rate=0.001),
            manifest,
            root,
            tmp_path / "wrong-config",
            resume_from=checkpoint,
        )
    with pytest.raises(CheckpointError):
        Trainer(
            model_config(),
            config,
            replace(manifest, corpus_id="another-corpus"),
            root,
            tmp_path / "wrong-corpus",
            resume_from=checkpoint,
        )
