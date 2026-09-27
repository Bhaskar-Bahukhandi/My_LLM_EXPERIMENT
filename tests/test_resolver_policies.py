"""Construction-only policy, compatibility and truthful status regressions."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from unified_edge.config import (
    AuthoredConfig,
    ConfigError,
    ModelRequest,
    ResolverPolicy,
    SearchPolicy,
    load_config,
)
from unified_edge.parameters import audit_executable_parameters
from unified_edge.resolve import ResolvedConfig, resolve_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def pilot():
    return load_config(ROOT / "configs/models/edge_20m_candidate.yaml")


def test_pinned_yaml_exact_real_meta_inventory_formula_and_component_identity(pilot):
    resolved = resolve_config(pilot).selected
    assert resolved is not None
    audit = audit_executable_parameters(resolved)
    retained = json.loads((ROOT / "reports/20m_prebuild_mechanical.json").read_text())
    reviewed = retained["resumed_accounting"]["selected"]
    assert resolved.shape.to_dict() == reviewed["shape"]
    assert pilot.model.target_parameters == 20_000_000
    assert pilot.model.parameter_tolerance == 0.05
    assert audit["unique_trainable_parameters"] == audit["formula_parameters"] == 20_387_531
    assert audit["unique_trainable_tensors"] == audit["named_trainable_tensors"] == 128
    assert audit["components"] == reviewed["components"]
    assert sum(audit["components"].values()) == 20_387_531
    assert audit["tensor_ledger"] == reviewed["tensor_ledger"]  # no padding or extra tensors
    assert audit["inventory_audit"]["unique_trainable_parameters"] == 20_387_531
    assert not audit["forward_executed"] and not audit["parameter_values_materialized"]


def test_yaml_and_resolved_round_trips_include_policy_identity(pilot, tmp_path):
    path = tmp_path / "roundtrip.yaml"
    path.write_text(yaml.safe_dump(pilot.to_dict()), encoding="utf-8")
    assert load_config(path) == pilot
    selected = resolve_config(pilot).selected
    again = resolve_config(load_config(path)).selected
    assert again.to_dict() == selected.to_dict()
    assert ResolvedConfig.from_dict(selected.to_dict()) == selected
    assert selected.to_dict()["resolver"] == {"policy": "edge_dense_v2_20m", "mode": "explicit"}
    legacy = replace(pilot, resolver=None)
    assert resolve_config(legacy).selected.shape == selected.shape
    assert resolve_config(legacy).selected.sha256 != selected.sha256
    assert resolve_config(legacy).selected.authored_sha256 != selected.authored_sha256


def test_legacy_2m_serialization_and_inventory_remain_exact():
    authored = load_config(ROOT / "configs/models/edge_2m.yaml")
    old = json.loads((ROOT / "reports/edge_2m_resolved.json").read_text())
    resolved = resolve_config(authored).selected
    assert resolved.to_dict() == old
    assert ResolvedConfig.from_dict(old).to_dict() == old
    assert resolved.sha256 == "7fe06b639741003e4c5ad9d57d32c2fd2b86f6805b2048ad1efd6aaa75e3a6f1"
    assert "resolver" not in authored.to_dict() and "resolver" not in resolved.to_dict()
    assert AuthoredConfig.from_dict(authored.to_dict()) == authored
    assert resolve_config(AuthoredConfig.from_dict({"schema_version": "1"})).selected == resolved
    audit = audit_executable_parameters(resolved)
    old_inventory = json.loads((ROOT / "reports/20m_prebuild_mechanical.json").read_text())[
        "baseline_2m_parameters"
    ]
    assert audit["unique_trainable_parameters"] == 1_929_579
    assert audit["tensor_ledger"] == old_inventory["tensor_ledger"]
    assert audit["components"] == old_inventory["components"]


def test_target_magnitude_does_not_select_policy():
    large_legacy = AuthoredConfig(model=ModelRequest(target_parameters=20_000_000))
    result = resolve_config(large_legacy)
    assert large_legacy.resolver is None
    assert {c.d_model for c in result.candidates} == {128, 192, 256}
    assert max(c.parameters for c in result.candidates) == 5_385_771
    assert ResolverPolicy("edge_dense_v1_2m", "search").search_space == SearchPolicy()


def test_diagnostic_search_never_selects_or_overrides_pilot(pilot):
    search = load_config(ROOT / "configs/models/edge_20m_search.yaml")
    a, b = resolve_config(search).to_dict(), resolve_config(search).to_dict()
    assert a == b
    assert a["status"] == "SEARCH_COMPLETE" and a["resolved_config"] is None
    pairs = {(c["d_model"], c["shared_layers"]) for c in a["candidates"]}
    assert {(256, 46), (320, 30), (384, 21), (512, 12), (640, 8), (896, 4), (1024, 3)} <= pairs
    ranks = [(abs(c["delta"]), c["shared_layers"], c["d_model"]) for c in a["candidate_ranking"]]
    assert ranks == sorted(ranks)
    explicit = resolve_config(pilot)
    assert len(explicit.candidates) == 1
    assert explicit.selected.shape.d_model == 512 and explicit.selected.shape.shared_layers == 12
    assert any(abs(c["delta"]) < abs(explicit.candidates[0].delta) for c in a["candidates"])
    assert not a["readiness"]["pilot_execution_ready"]
    assert a["readiness"]["construction_status"] == "NOT_CONSTRUCTED"
    decoder_control = replace(search, model=replace(search.model, decoder_dim=128))
    controls = resolve_config(decoder_control).candidates
    assert (
        next(c for c in controls if (c.d_model, c.shared_layers) == (512, 12)).parameters
        == 20_114_763
    )
    assert (
        next(c for c in controls if (c.d_model, c.shared_layers) == (384, 21)).parameters
        == 20_061_951
    )
    for candidate in resolve_config(search).candidates:
        if candidate.status == "VALID_GENERIC_PATH":
            assert candidate.d_model % search.model.headdim == 0
            assert candidate.parameters > 0
    assert search.model.patch_size == 8 and not search.model.moe_enabled


@pytest.mark.parametrize("policy", ["edge_dense_v3_40m", "unknown", "", None, 20, {}])
def test_unsupported_policy_does_not_fall_through(policy):
    with pytest.raises(ConfigError, match="unsupported resolver.policy"):
        AuthoredConfig.from_dict(
            {"schema_version": "1", "resolver": {"policy": policy, "mode": "search"}}
        )


@pytest.mark.parametrize(
    "resolver",
    [
        None,
        {},
        {"policy": "edge_dense_v2_20m"},
        {"mode": "search"},
        {"policy": "edge_dense_v2_20m", "mode": "auto"},
    ],
)
def test_missing_or_ambiguous_policy_rejected(resolver):
    with pytest.raises(ConfigError):
        AuthoredConfig.from_dict({"schema_version": "1", "resolver": resolver})


def test_mode_conflicts_and_mutable_policy_lists_fail(pilot):
    with pytest.raises(ConfigError, match="cannot override a pinned shape"):
        replace(pilot, resolver=replace(pilot.resolver, mode="search"))
    with pytest.raises(ConfigError, match="pinned"):
        replace(pilot, model=replace(pilot.model, d_model="auto"))
    with pytest.raises(ConfigError, match="conflicts"):
        replace(pilot, search=SearchPolicy())
    with pytest.raises(ConfigError, match="both dimensions auto"):
        replace(
            pilot,
            model=replace(pilot.model, d_model="auto"),
            resolver=replace(pilot.resolver, mode="search"),
        )


@pytest.mark.parametrize(
    "change",
    [
        {"d_model": 130},
        {"patch_size": 4},
        {"chunk_patches": 8},
        {"shared_layers": 1025},
    ],
)
def test_invalid_versioned_shape_has_no_fallback(pilot, change):
    result = resolve_config(replace(pilot, model=replace(pilot.model, **change)))
    assert result.status == "NO_LEGAL_CANDIDATE" and result.selected is None
    assert len(result.candidates) == 1 and result.candidates[0].reason


def test_explicit_tolerance_rejection_does_not_publish_a_resolved_model(pilot):
    result = resolve_config(replace(pilot, model=replace(pilot.model, parameter_tolerance=0)))
    assert result.status == "OUTSIDE_TARGET" and result.selected is None
    assert not result.candidates[0].within_tolerance
    assert result.to_dict()["readiness"]["training_execution_status"] == "NOT_RUN"


def test_moe_and_resolved_policy_tampering_fail(pilot):
    with pytest.raises(ConfigError, match="MoE"):
        replace(pilot.model, moe_enabled=True)
    data = resolve_config(pilot).selected.to_dict()
    data["resolver"]["policy"] = "unknown"
    with pytest.raises(ConfigError, match="unsupported"):
        ResolvedConfig.from_dict(data)
    data["resolver"] = {"policy": "edge_dense_v2_20m", "mode": "search"}
    with pytest.raises(ConfigError, match="explicit"):
        ResolvedConfig.from_dict(data)


def test_status_is_scoped_and_does_not_execute_model_or_change_rng(pilot, monkeypatch):
    from unified_edge.dense_model import DenseByteModel

    def forbidden(*args, **kwargs):
        raise AssertionError("construction audit must not execute a forward")

    monkeypatch.setattr(DenseByteModel, "forward", forbidden)
    before = torch.get_rng_state().clone()
    report = resolve_config(pilot).to_dict()
    assert torch.equal(before, torch.get_rng_state())
    status = report["readiness"]
    assert status["architecture_status"] == "RECOMMENDED_FOR_PILOT"
    assert status["accounting_status"] == "META_VERIFIED"
    assert status["training_execution_status"] == "NOT_RUN"
    assert not status["pilot_execution_ready"] and not status["training_authorized"]
    assert status["next_gate"] == "IMPLEMENT_R9_THEN_RUN_APPROVED_20M_EXECUTION_GATES"
    assert status["scale_stage"] == "20M_PILOT_CANDIDATE"
    assert status["construction_status"] == "META_CONSTRUCTED"
    assert status["quality_status"] == "UNVERIFIED"
    assert "R1-R8 infrastructure implemented and reviewed" in status["evidence_scope"]
    assert "R9 remains unimplemented" in status["evidence_scope"]
    assert "Hardware execution gates remain unpassed" in status["evidence_scope"]
    assert "No pilot or training authorization" in status["evidence_scope"]
    assert status["evidence_reference"].split("; ") == [
        "reports/20m_prebuild_audit.json",
        "reports/20m_i1_resolver_config_implementation.json",
        "reports/20m_i2_training_infrastructure.json",
    ]
    assert all((ROOT / path).is_file() for path in status["evidence_reference"].split("; "))
    assert "training_ready" not in report
    assert "not implemented" not in json.dumps(report)
    legacy = resolve_config(load_config(ROOT / "configs/models/edge_2m.yaml")).to_dict()
    assert legacy["readiness"]["training_execution_status"] == "COMPLETED_FROZEN_LINEAGE"
    assert legacy["readiness"]["quality_status"] == "ENGINEERING_BASELINE_ONLY"
    assert legacy["readiness"]["next_gate"] == "PRESERVE_FROZEN_2M_COMPLETE_20M_R9"
    assert (
        "does not train, load weights or authorize current-source resume"
        in (legacy["readiness"]["evidence_scope"])
    )
    assert not legacy["readiness"]["pilot_execution_ready"]
    assert not legacy["readiness"]["training_authorized"]
    unreviewed = replace(pilot, model=replace(pilot.model, decoder_dim=128))
    assert (
        resolve_config(unreviewed).to_dict()["readiness"]["training_execution_status"] == "NOT_RUN"
    )
    assert (
        resolve_config(unreviewed).to_dict()["readiness"]["architecture_status"]
        != "RECOMMENDED_FOR_PILOT"
    )
    assert not report["memory_preflight"]["runtime_fit_verified"]
