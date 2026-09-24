"""Continuation gates, immutable evidence and exact production-state restoration."""

import ast
import copy
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import full_training_stage_b_to_endpoint as continuation  # noqa: E402
from context_ab_core import trees_equal  # noqa: E402

from unified_edge.resolve import ResolvedConfig  # noqa: E402
from unified_edge.training.checkpoint import load_checkpoint  # noqa: E402
from unified_edge.training.config import TrainingConfig  # noqa: E402
from unified_edge.training.data import create_tiny_fixture  # noqa: E402
from unified_edge.training.trainer import Trainer  # noqa: E402


def test_domain_watch_is_consecutive_material_and_not_posthoc():
    def result(domain, aggregate=2.0):
        return {"nll": aggregate, "domains": {"code": {"nll": domain}}}

    parent = result(3.0, 2.5)
    assert continuation.domain_watch(parent, None, result(3.2)) == []
    assert continuation.domain_watch(parent, result(3.04), result(3.05)) == ["code"]
    assert continuation.domain_watch(parent, result(3.04), result(3.035)) == []
    assert continuation.domain_watch(parent, result(3.01), result(3.2)) == []
    assert continuation.domain_watch(parent, result(3.04, 2.6), result(3.2)) == []
    with pytest.raises(ValueError, match="policy"):
        continuation.domain_watch(parent, None, parent, {"material_nll": 1.0})


def row(step=5001, total=639741):
    return dict(
        step=step,
        micro_step=step * 2,
        examples_seen=step * 4,
        bytes_seen=total,
        loss=2.0,
        raw_byte_nll=2.0,
        raw_byte_perplexity=7.389,
        valid_target_count=128,
        learning_rate=0.003,
        gradient_norm=1.2,
        clip_threshold=1.0,
        clipped=True,
        summed_nll=256.0,
        finite_gradients=True,
        optimizer_update_counts=[step] * 56,
        scheduler_completed=step,
        cursor={"epoch": 0, "offset": step * 4, "order_sha256": "frozen"},
    )


def test_journal_replay_counter_order_and_finite_gates():
    original = row()
    assert continuation.audit_rows([original]) == 639741
    continuation.compare_row(original, copy.deepcopy(original))
    for key, value in (
        ("step", 5002),
        ("bytes_seen", 639742),
        ("finite_gradients", False),
        ("optimizer_update_counts", [5000] * 56),
        ("summed_nll", 255.0),
    ):
        changed = dict(original, **{key: value})
        with pytest.raises(ValueError):
            continuation.audit_rows([changed])
    changed = copy.deepcopy(original)
    changed["cursor"]["order_sha256"] = "different"
    with pytest.raises(ValueError, match="replay mismatch"):
        continuation.compare_row(changed, original)
    assert continuation.expected_cursor(78167) == (1, 2)
    assert continuation.expected_cursor(5000) == (0, 20000)


def test_publication_never_overwrites_and_partial_journal_rejected(tmp_path):
    path = tmp_path / "receipt.json"
    continuation.write_new(path, {"status": "PASS"})
    before = path.read_bytes()
    with pytest.raises(ValueError, match="immutable"):
        continuation.write_new(path, {"status": "CHANGED"})
    assert path.read_bytes() == before
    journal = tmp_path / "rows.jsonl"
    continuation.append(journal, row())
    assert continuation.read_rows(journal) == [row()]
    with journal.open("ab") as handle:
        handle.write(b"{")
    with pytest.raises(ValueError, match="partial journal"):
        continuation.read_rows(journal)


def test_new_runner_critical_checks_survive_python_optimization():
    tree = ast.parse(Path(continuation.__file__).read_text(encoding="utf-8-sig"))
    assert not any(isinstance(node, ast.Assert) for node in ast.walk(tree))
    assert continuation.END == 78167
    assert continuation.VALIDATIONS == (10000, 20000, 30000, 40000, 50000, 60000, 70000, 78167)
    assert continuation.PROOFS == (20000, 40000, 60000, 78167)


def test_exact_restore_and_next_update_on_real_model(tmp_path):
    manifest = create_tiny_fixture(tmp_path / "data")
    resolved = ResolvedConfig.from_dict(
        json.loads((Path(__file__).parents[1] / "reports/edge_2m_resolved.json").read_text())
    )
    config = TrainingConfig(total_steps=4, warmup_steps=1)
    original = Trainer(resolved, config, manifest, tmp_path / "data", tmp_path / "a")
    original.step()
    saved = original.save()
    snapshot = load_checkpoint(saved)
    continuation.check_restored(original, snapshot)
    expected = continuation.state(original)
    original.step()
    after = continuation.state(original)
    restored = Trainer(
        resolved, config, manifest, tmp_path / "data", tmp_path / "b", resume_from=saved
    )
    continuation.check_restored(restored, snapshot)
    assert trees_equal(continuation.state(restored), expected)
    restored.step()
    assert trees_equal(continuation.state(restored), after)
    wrong = load_checkpoint(restored.save())
    wrong["scheduler"]["completed"] = 0
    with pytest.raises(ValueError, match="scheduler"):
        continuation.check_restored(restored, wrong)
    torch.set_num_threads(1)
