"""Small deterministic fixtures for the isolated read-only evaluation contract."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from context_distribution_study import PRODUCTION, Store, verify_inventory  # noqa: E402
from context_study_metrics import (  # noqa: E402
    LENGTHS,
    add_losses,
    anchor_history,
    check_baseline,
    distribution,
    paired_summary,
    select_anchors,
    summarize_losses,
    validation_documents,
)

from unified_edge.training.data import DatasetManifest  # noqa: E402


def test_baseline_gate():
    check_baseline(2.362495391641617, 500000)
    with pytest.raises(ValueError, match="did not reproduce"):
        check_baseline(2.363, 500000)
    with pytest.raises(ValueError, match="500000"):
        check_baseline(2.3624953916, 499999)


def test_anchors_preserve_target_boundaries_domains_and_determinism(tmp_path):
    (tmp_path / "a").write_bytes(bytes(range(256)) * 2)
    (tmp_path / "b").write_bytes(b"different train")
    manifest = DatasetManifest.create(
        tmp_path, "test", [("a", "validation", "math"), ("b", "train", "code")]
    )
    payloads = manifest.verify(tmp_path)
    docs = validation_documents(manifest, payloads)
    anchors = select_anchors(docs, 8)
    assert anchors == select_anchors(docs, 8)
    assert len(anchors) == 8 and {a["domain"] for a in anchors} == {"math"}
    for anchor in anchors:
        for length in LENGTHS:
            assert len(anchor_history(anchor, payloads, length)) == length
            assert payloads[anchor["path"]][anchor["offset"]] == anchor["target"]
    with pytest.raises(ValueError, match="boundary"):
        anchor_history(dict(anchors[0], offset=255), payloads, 256)
    with pytest.raises(ValueError, match="identity"):
        anchor_history(dict(anchors[0], target=-1), payloads, 32)
    with pytest.raises(ValueError, match="rejects TEST"):
        validation_documents(manifest, payloads, "test")


def test_original_and_conditional_probabilities_and_entropy():
    row = distribution(torch.zeros(267))
    assert row["byte_mass_p"] == pytest.approx(256 / 267)
    assert row["control_mass_p"] == pytest.approx(11 / 267)
    assert row["space_p"] == pytest.approx(1 / 267)
    assert row["space_q"] == pytest.approx(1 / 256)
    assert row["entropy_nats"] == pytest.approx(5.545177444479562)
    assert (row["top1_byte"], row["top2_byte"]) == (0, 1)
    assert row["margin_q"] == 0
    values = torch.zeros(267)
    values[256:258] = -torch.inf
    values[32] = 0.1
    row = distribution(values)
    assert row["masked_symbol_ids"] == [256, 257]
    assert row["classification"] == "ARGMAX_SPACE_DOMINANCE_WITH_BROAD_DISTRIBUTION"
    assert sum(row["control_probabilities_p"]) + row["byte_mass_p"] == pytest.approx(1)
    values[32] = 20
    assert distribution(values)["classification"] == "SEVERE_BYTE_MODE_CONCENTRATION"
    values[0] = torch.nan
    with pytest.raises(ValueError):
        distribution(values)


def test_bucket_domain_weighting_and_paired_deltas():
    groups = {}
    add_losses(groups, "math", [1, 2, 3])
    add_losses(groups, "math", [10])
    result = summarize_losses(groups)["math"]
    assert result["count"] == 4 and result["nll"] == 4
    rows = [{"scores": {str(n): {"nll": 3 if n == 32 else 2} for n in LENGTHS}}] * 4
    result = paired_summary(rows)
    assert result["64"]["paired_delta"] == -1
    assert result["64"]["paired_delta_ci95"] == [-1, -1]
    assert result["32"]["unchanged_fraction"] == 1


def test_progress_recovery_checksums_and_no_overwrite(tmp_path):
    store = Store(tmp_path, {"protocol": 1})
    store.put("block_0", {"count": 10})
    (tmp_path / "progress.json").write_text("interrupted")
    recovered = Store(tmp_path, {"protocol": 1})
    assert recovered.get("block_0") == {"count": 10}
    with pytest.raises(ValueError, match="overwrite"):
        recovered.put("block_0", {})
    with pytest.raises(ValueError, match="binding mismatch"):
        Store(tmp_path, {"protocol": 2})
    path = tmp_path / "block_0.json"
    path.write_text(path.read_text().replace('"count": 10', '"count": 11'))
    with pytest.raises(ValueError, match="integrity"):
        recovered.get("block_0")


def test_parent_write_guard_and_inventory_rejection(monkeypatch):
    with pytest.raises(ValueError, match="cannot write production"):
        Store(PRODUCTION / "forbidden-study", {})
    monkeypatch.setattr("context_distribution_study.inventory", lambda: {"parent": "changed"})
    with pytest.raises(ValueError, match="immutable"):
        verify_inventory({"parent": "original"})


def test_disposable_clone_storage_isolation():
    import copy

    model = torch.nn.Linear(2, 2)
    clone = copy.deepcopy(model)
    before = model.weight.detach().clone()
    clone(torch.ones(1, 2)).sum().backward()
    with torch.no_grad():
        clone.weight.add_(1)
    assert torch.equal(model.weight, before)
    assert model.weight.grad is None
    assert clone.weight.data_ptr() != model.weight.data_ptr()


def test_disposable_probe_exercises_real_helper_without_parent_gradients(monkeypatch):
    from context_distribution_study import disposable_probe

    class Fixture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(256, 267)

        def forward(self, target):
            return self.embedding(target)

    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(
        "context_distribution_study.memory",
        lambda: {
            "peak_allocated_bytes": 0,
            "peak_reserved_bytes": 0,
            "free_device_bytes": 0,
            "process_rss_status": "UNVERIFIED",
        },
    )
    model = Fixture()
    before = model.embedding.weight.detach().clone()
    result = disposable_probe(model, torch.tensor([[0, 128, 255]]))
    assert result["status"] == "PASS" and result["optimizer_updates"] == 0
    assert result["target_count"] == 3 and result["gradient_norm"] > 0
    assert model.embedding.weight.grad is None
    assert torch.equal(before, model.embedding.weight)


def test_corpus_canonicalizer_matches_existing_utf8_contract():
    import hashlib

    from benchmark_fingerprints import canonical

    from unified_edge.training.config import canonical_hash

    value = {"title": "caf\u00e9"}
    assert b"\\u00e9" not in canonical(value)
    assert hashlib.sha256(canonical(value)).hexdigest() != canonical_hash(value)


def test_segmented_path_matches_objective_boundaries_and_recovers(tmp_path, monkeypatch):
    from context_distribution_study import segmented

    from unified_edge.training.data import Document
    from unified_edge.training.optimization import raw_byte_nll

    class Fixture(torch.nn.Module):
        calls = 0

        def forward(self, target):
            self.calls += 1
            return torch.zeros(*target.shape, 267)

    original_to = torch.Tensor.to

    def cpu_to(tensor, *args, **kwargs):
        if args == ("cuda:0",):
            return tensor
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", cpu_to)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(
        "context_distribution_study.memory",
        lambda: {
            "peak_allocated_bytes": 0,
            "peak_reserved_bytes": 0,
            "free_device_bytes": 0,
            "process_rss_status": "UNVERIFIED",
        },
    )
    gates = []
    monkeypatch.setattr(
        "context_distribution_study.check_baseline", lambda n, c: gates.append((n, c))
    )
    docs = [
        (Document("a", "0" * 64, 35, 1, "validation", "general_text"), b"a" * 35),
        (Document("b", "1" * 64, 70, 1, "validation", "code"), b"b" * 70),
    ]
    store = Store(tmp_path, {"small_fixture": True})
    model = Fixture()
    result = segmented(model, docs, 32, store)
    assert result["valid_target_count"] == 105 and result["windows"] == 5
    assert result["domains"]["general_text"]["count"] == 35
    assert result["domains"]["code"]["count"] == 70
    assert gates == [(result["nll"], 105)]
    expected = 0
    for count in (32, 3, 64, 6):
        targets = torch.zeros(1, count, dtype=torch.long)
        expected += raw_byte_nll(model(targets), targets)[0].item()
    assert result["sum_nll"] == pytest.approx(expected, abs=1e-4)
    calls = model.calls
    assert segmented(model, docs, 32, store) == result
    assert model.calls == calls
    longer = segmented(model, docs, 64, store)
    assert longer["buckets"]["0"]["count"] == 70
    assert longer["buckets"]["32"]["count"] == 35


def test_position_survey_strata_are_deterministic_and_bounded():
    from context_study_positions import selections

    anchors = [
        dict(domain=d, offset=256 + i, target=i % 256, key=f"{i:04d}", path=d)
        for d in ("code", "text")
        for i in range(256)
    ]
    selected = selections(anchors, per_stratum=2)
    assert selected == selections(list(reversed(anchors)), per_stratum=2)
    assert len(selected) == 2 * 2 * (1 + 2 + 4 + 8)
    for row in selected:
        assert 0 <= row["position"] < row["length"]
        assert row["anchor"]["offset"] - row["position"] >= 0
        assert row["anchor"]["offset"] % row["length"] == row["position"]


def test_resume_commit_metadata_does_not_accept_changed_model_source():
    from resume_context_distribution_study import bound_source

    original = {"commit": "parent", "source_sha256": "same"}
    current = {"commit": "published", "source_sha256": "same", "dirty": True}
    assert bound_source(current, original) == dict(current, commit="parent")
    with pytest.raises(ValueError, match="model source changed"):
        bound_source(dict(current, source_sha256="different"), original)


def test_finalizer_requires_actual_complete_and_current_validation_evidence(tmp_path, monkeypatch):
    import json

    import finish_context_distribution_study as finalizer

    from unified_edge.training.config import canonical_hash

    monkeypatch.setattr(finalizer, "ROOT", tmp_path)
    monkeypatch.setattr(finalizer, "STUDY", tmp_path / "study")
    for name in finalizer.VALIDATION_FILES:
        p = tmp_path / name
        p.parent.mkdir(exist_ok=True)
        p.write_text("tested content")
    for name in ("integrity_final.json", "position-distributions/integrity_final.json"):
        p = tmp_path / "study" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("completed measurements")
    value = {
        "schema": "study-validation-1",
        "final_integrity": {"status": "PASS"},
        "commands": [
            {"name": n, "returncode": 0}
            for n in (
                "pytest",
                "ruff",
                "format",
                "compileall",
                "cpu_dependencies",
                "cuda_dependencies",
                "diff",
            )
        ],
        "files_sha256": {n: finalizer.sha(tmp_path / n) for n in finalizer.VALIDATION_FILES},
        "measurement_integrity": {
            "main": finalizer.sha(tmp_path / "study/integrity_final.json"),
            "positions": finalizer.sha(
                tmp_path / "study/position-distributions/integrity_final.json"
            ),
        },
    }
    path = tmp_path / "validation.json"
    path.write_text(json.dumps({"value": value, "sha256": canonical_hash(value)}))
    assert finalizer.validation_closeout(path)["commands"] == value["commands"]
    (tmp_path / finalizer.VALIDATION_FILES[0]).write_text("changed since tests")
    with pytest.raises(ValueError, match="stale"):
        finalizer.validation_closeout(path)
    value["commands"] = value["commands"][:-1]
    path.write_text(json.dumps({"value": value, "sha256": canonical_hash(value)}))
    with pytest.raises(ValueError, match="missing final validation"):
        finalizer.validation_closeout(path)
