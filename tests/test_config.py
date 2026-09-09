from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from unified_edge.config import (
    AuthoredConfig,
    ConfigError,
    Hardware,
    ModelRequest,
    SearchPolicy,
    load_config,
)
from unified_edge.resolve import ResolvedConfig, resolve_config

ROOT = Path(__file__).resolve().parents[1]


def test_authored_and_resolved_round_trip_and_immutability():
    authored = load_config(ROOT / "configs/models/edge_2m.yaml")
    assert AuthoredConfig.from_dict(authored.to_dict()) == authored
    result = resolve_config(authored)
    assert result.selected is not None
    resolved = result.selected
    assert ResolvedConfig.from_dict(resolved.to_dict()) == resolved
    assert ResolvedConfig.from_dict(resolved.to_dict()).sha256 == resolved.sha256
    with pytest.raises(FrozenInstanceError):
        resolved.shape.d_model = 999
    with pytest.raises(FrozenInstanceError):
        authored.model.d_model = 999
    assert "auto" not in str(resolved.to_dict())
    assert (
        resolved.to_dict()["derived"]["d_inner"] == resolved.shape.expand * resolved.shape.d_model
    )


@pytest.mark.parametrize(
    "document",
    [
        {},
        {"schema_version": 1},
        {"schema_version": "99"},
        {"schema_version": "1", "mdoel": {}},
        {"schema_version": "1", "model": {"d_modle": 128}},
        {"schema_version": "1", "hardware": {"gpu": 0}},
        {"schema_version": "1", "search": {"seed": 3}},
        {"schema_version": "1", "model": []},
        {"schema_version": "1", "search": {"widths": "128"}},
        {"schema_version": "1", "model": {"moe_enabled": True}},
        {"schema_version": "1", "model": {"moe_enabled": 0}},
    ],
)
def test_unknown_and_unsupported_schema_rejected(document):
    with pytest.raises(ConfigError):
        AuthoredConfig.from_dict(document)


@pytest.mark.parametrize(
    "field,value",
    [
        ("d_model", True),
        ("shared_layers", 1.5),
        ("target_parameters", 0),
        ("byte_dim", -1),
        ("decoder_dim", "64"),
        ("d_state", None),
        ("d_conv", 0),
        ("expand", False),
        ("headdim", 1.2),
        ("patch_size", []),
        ("parameter_tolerance", float("nan")),
        ("parameter_tolerance", float("inf")),
        ("parameter_tolerance", True),
        ("parameter_tolerance", -0.1),
        ("parameter_tolerance", 1),
        ("d_model", "AUTO"),
    ],
)
def test_bad_model_field_types_fail_before_instantiation(field, value):
    with pytest.raises(ConfigError):
        ModelRequest(**{field: value})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"widths": ()},
        {"widths": (128, 128)},
        {"depths": (True,)},
        {"widths": [128]},
        {"width_multiple": 0},
        {"depths": (-1,)},
        {"widths": tuple(range(1, 66)), "depths": tuple(range(1, 66))},
    ],
)
def test_bad_search_policy(kwargs):
    with pytest.raises(ConfigError):
        SearchPolicy(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"backend": "cuda_triton_safe"},
        {"device": "cuda"},
        {"precision": "bf16"},
        {"batch_size": True},
        {"memory_budget_bytes": -1},
    ],
)
def test_no_silent_hardware_fallback(kwargs):
    with pytest.raises(ConfigError):
        Hardware(**kwargs)


@pytest.mark.parametrize(
    "text",
    [
        'schema_version: "1"\nmodel:\n  d_model: 128\n  d_model: 256\n',
        'schema_version: "1"\nschema_version: "1"\n',
        'schema_version: "1"\n1: bad\n',
        'schema_version: "1"\nmodel: !!python/object:object {}\n',
        'schema_version: "1"\nmodel: [\n',
    ],
)
def test_yaml_duplicate_unsafe_and_malformed_input(tmp_path, text):
    path = tmp_path / "invalid.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


@pytest.mark.parametrize(
    "section,key,value",
    [
        (None, "semantic_reference_commit", "wrong"),
        (None, "schema_version", "2"),
        (None, "unknown", 1),
        ("derived", "nheads", 999),
        ("control_ids", "eos", 256),
        ("model", "unused_width", 256),
        ("hardware", "precision", "fp16"),
    ],
)
def test_resolved_metadata_and_derived_values_cannot_drift(section, key, value):
    data = resolve_config(AuthoredConfig()).selected.to_dict()
    target = data if section is None else data[section]
    target[key] = value
    with pytest.raises(ConfigError):
        ResolvedConfig.from_dict(data)


def test_resolved_contract_rejects_omitted_fields():
    data = resolve_config(AuthoredConfig()).selected.to_dict()
    del data["model"]["shared_layers"]
    with pytest.raises(ConfigError, match="every dimension"):
        ResolvedConfig.from_dict(data)


def test_resolution_is_order_independent_and_preserves_request():
    authored = AuthoredConfig()
    reversed_search = replace(
        authored.search,
        widths=tuple(reversed(authored.search.widths)),
        depths=tuple(reversed(authored.search.depths)),
    )
    a = resolve_config(authored)
    b = resolve_config(replace(authored, search=reversed_search))
    assert a.selected.shape == b.selected.shape
    assert a.candidates == b.candidates
    assert a.to_dict() == resolve_config(authored).to_dict()
    # Authored order is provenance, even though it does not affect candidate selection.
    assert a.selected.authored_sha256 != b.selected.authored_sha256
    assert authored.model.d_model == "auto"


def test_outside_target_is_reported_without_changing_tolerance():
    authored = AuthoredConfig()
    result = resolve_config(authored)
    assert result.status == "OUTSIDE_TARGET"
    assert result.selected.parameter_tolerance == 0.02
    counts = [c.parameters for c in result.candidates if c.parameters is not None]
    selected_count = result.to_dict()["actual_inventory_parameters"]
    assert abs(selected_count - 2_000_000) == min(abs(c - 2_000_000) for c in counts)
    assert not result.to_dict()["training_ready"]


def test_legal_candidate_can_meet_a_declared_band():
    authored = AuthoredConfig(model=ModelRequest(parameter_tolerance=0.05))
    result = resolve_config(authored)
    assert result.status == "WITHIN_TARGET"
    assert result.selected.parameter_tolerance == 0.05


def test_explicit_dimensions_are_not_silently_replaced():
    authored = AuthoredConfig(model=ModelRequest(d_model=128, shared_layers=2))
    result = resolve_config(authored)
    assert len(result.candidates) == 1
    assert result.selected.shape.d_model == 128
    assert result.selected.shape.shared_layers == 2


@pytest.mark.parametrize(
    "model,reason",
    [
        (ModelRequest(d_model=130, headdim=64), "divisible"),
        (ModelRequest(d_model=160, headdim=32), "clean search policy"),
        (ModelRequest(patch_size=4), "patch_size=8"),
        (ModelRequest(chunk_patches=8), "chunk_patches=16"),
        (ModelRequest(shared_layers=1025), "safety limit"),
    ],
)
def test_illegal_candidates_are_explained(model, reason):
    result = resolve_config(AuthoredConfig(model=model))
    assert result.status == "NO_LEGAL_CANDIDATE"
    assert result.selected is None
    assert all(c.status == "INVALID" and reason in c.reason for c in result.candidates)
