from dataclasses import replace
from io import BytesIO

import pytest
import torch
from torch import nn

from unified_edge.config import AuthoredConfig, Hardware, ModelShape, SearchPolicy
from unified_edge.parameters import (
    audit_parameters,
    formula_parameter_estimate,
    instantiate_inventory,
    parameter_specs,
)
from unified_edge.preflight import memory_preflight
from unified_edge.resolve import resolve_config


def shape(**kwargs):
    return replace(ModelShape(64, 2, 16, 32, 8, 4, 2, 16, 8, 16), **kwargs)


def test_audit_distinguishes_real_ties_frozen_buffers_and_serialization():
    module = nn.Module()
    module.a = nn.Linear(3, 2, bias=False)
    module.b = nn.Linear(3, 2, bias=False)
    module.b.weight = module.a.weight
    module.register_parameter("frozen", nn.Parameter(torch.ones(5), requires_grad=False))
    module.register_buffer("running", torch.ones(4))
    module.register_buffer("running_alias", module.running)
    module.register_buffer("scratch", torch.ones(7), persistent=False)
    audit = audit_parameters(module)
    assert audit["unique_trainable_parameters"] == 6
    assert audit["unique_frozen_parameters"] == 5
    assert audit["unique_buffer_elements"] == 11
    assert audit["unique_parameter_bytes"] == 44
    assert audit["unique_buffer_bytes"] == 44
    assert audit["aliases"] == {"b.weight": "a.weight", "running_alias": "running"}
    assert sum(audit["components"].values()) == 6
    assert audit["state_dict_tensor_entries"] == 5
    assert audit["state_dict_entry_elements"] == 25
    assert audit["serialized_file_bytes"] is None
    buffer = BytesIO()
    torch.save(module.state_dict(), buffer)
    assert len(buffer.getvalue()) > audit["state_dict_logical_payload_bytes"]
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=True)
    assert restored["a.weight"].data_ptr() == restored["b.weight"].data_ptr()


@pytest.mark.parametrize(
    "width,layers,state,head",
    [
        (64, 1, 8, 16),
        (128, 2, 32, 32),
        (192, 3, 64, 64),
        (256, 4, 64, 64),
    ],
)
def test_instantiated_inventory_formula_and_independent_primitives(width, layers, state, head):
    config = shape(d_model=width, shared_layers=layers, d_state=state, headdim=head)
    inventory = instantiate_inventory(config)
    audit = audit_parameters(inventory)
    assert all(p.is_meta for p in inventory.parameters())
    assert audit["unique_trainable_parameters"] == formula_parameter_estimate(config)
    assert sum(audit["components"].values()) == audit["unique_trainable_parameters"]
    assert audit["unique_frozen_parameters"] == audit["unique_buffer_elements"] == 0
    assert len(parameter_specs(config)) == audit["state_dict_tensor_entries"]
    # Independently ask ordinary PyTorch primitives for the reference block's parameter sizes.
    d, i, n, h = width, config.d_inner, state, config.nheads
    primitives = nn.ModuleList(
        [
            nn.RMSNorm(d, device="meta"),
            nn.Linear(d, 2 * i + 2 * n + h, bias=False, device="meta"),
            nn.Conv1d(i + 2 * n, i + 2 * n, 4, groups=i + 2 * n, device="meta"),
            nn.RMSNorm(i, device="meta"),
            nn.Linear(i, d, bias=False, device="meta"),
        ]
    )
    expected_shared = layers * (sum(p.numel() for p in primitives.parameters()) + 3 * h)
    assert audit["components"]["shared"] == expected_shared
    decoder = nn.ModuleList(
        [
            nn.Linear(d, config.decoder_dim, device="meta"),
            nn.GRUCell(config.byte_dim, config.decoder_dim, device="meta"),
            nn.RMSNorm(config.decoder_dim, device="meta"),
        ]
    )
    assert audit["components"]["decoder"] == sum(p.numel() for p in decoder.parameters())


def test_inventory_does_not_change_rng():
    torch.manual_seed(42)
    before = torch.get_rng_state().clone()
    resolve_config(AuthoredConfig())
    torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)


@pytest.mark.parametrize("batch,head,state", [(1, 16, 8), (2, 32, 64), (3, 64, 32)])
def test_memory_matches_tensor_axes_and_scales_with_batch(batch, head, state):
    config = shape(headdim=head, d_state=state)
    audit = audit_parameters(instantiate_inventory(config))
    hardware = Hardware(batch_size=batch)
    memory = memory_preflight(config, audit, hardware)
    ssm = torch.empty(batch, config.shared_layers, config.nheads, head, state, device="meta")
    convolution = torch.empty(
        batch, config.shared_layers, config.d_inner + 2 * state, config.d_conv, device="meta"
    )
    assert memory.ssm_state_bytes == ssm.numel() * ssm.element_size()
    assert memory.convolution_state_bytes == convolution.numel() * convolution.element_size()
    assert memory.adam_moment_bytes == 2 * memory.gradient_bytes
    assert memory.known_training_payload_bytes == (
        memory.known_inference_payload_bytes + memory.gradient_bytes + memory.adam_moment_bytes
    )
    assert memory.status == "UNKNOWN_REQUIRES_MEASUREMENT"
    assert memory.activation_bytes is memory.peak_process_rss_bytes is None
    assert not memory.measured_runtime


def test_memory_budget_is_a_rejection_gate_not_a_fit_claim():
    config = shape()
    audit = audit_parameters(instantiate_inventory(config))
    base = memory_preflight(config, audit, Hardware())
    exact = memory_preflight(
        config, audit, Hardware(memory_budget_bytes=base.known_training_payload_bytes)
    )
    assert exact.status == "UNKNOWN_REQUIRES_MEASUREMENT"
    too_small = memory_preflight(
        config, audit, Hardware(memory_budget_bytes=base.known_training_payload_bytes - 1)
    )
    assert too_small.status == "DOES_NOT_FIT"


def test_memory_gates_precede_parameter_distance():
    base = AuthoredConfig(search=SearchPolicy(widths=(192, 256), depths=(4,)))
    unconstrained = resolve_config(base)
    assert unconstrained.selected.shape.d_model == 256
    candidates = unconstrained.candidates
    smaller_budget = next(c.known_training_payload_bytes for c in candidates if c.d_model == 192)
    constrained = resolve_config(
        replace(base, hardware=Hardware(memory_budget_bytes=smaller_budget))
    )
    assert constrained.selected.shape.d_model == 192
    rejection = next(c for c in constrained.candidates if c.d_model == 256)
    assert rejection.status == "REJECTED_MEMORY"
    assert "exceeds budget" in rejection.reason


def test_all_memory_rejections_are_retained():
    result = resolve_config(AuthoredConfig(hardware=Hardware(memory_budget_bytes=1)))
    assert result.status == "NO_LEGAL_CANDIDATE"
    assert result.selected is None
    assert all(c.status == "REJECTED_MEMORY" for c in result.candidates)


def test_non_tensor_extra_state_is_reported_separately():
    class WithMetadata(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(3))

        def get_extra_state(self):
            return {"format_version": 1}

    report = audit_parameters(WithMetadata())
    assert report["state_dict_tensor_entries"] == 1
    assert report["state_dict_entry_elements"] == 3
    assert report["state_dict_non_tensor_entries"] == ["_extra_state"]


def test_inventory_formula_disagreement_fails_loudly(monkeypatch):
    import unified_edge.resolve as resolver

    monkeypatch.setattr(resolver, "formula_parameter_estimate", lambda shape: -1)
    with pytest.raises(RuntimeError, match="inventory/formula conflict"):
        resolver.resolve_config(AuthoredConfig())


def test_extreme_tensor_dimensions_reject_before_metadata_allocation():
    from unified_edge.config import ModelRequest

    config = AuthoredConfig(model=ModelRequest(d_model=2**63, shared_layers=1))
    result = resolve_config(config)
    assert result.status == "NO_LEGAL_CANDIDATE"
    assert "int64 tensor indexing" in result.candidates[0].reason
