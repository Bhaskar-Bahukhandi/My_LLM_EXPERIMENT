import copy
import io
import json
from dataclasses import replace
from math import prod
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from unified_edge.mamba import LayerState, Mamba2Block, SharedMambaTrunk
from unified_edge.parameters import audit_parameters, parameter_specs
from unified_edge.resolve import ResolvedConfig

ATOL, RTOL = 1e-5, 1e-4
LENGTHS = (0, 1, 2, 7, 8, 9, 15, 16, 17, 31, 32)


@pytest.fixture(scope="module", autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def accepted_config():
    return ResolvedConfig.from_dict(
        json.loads(
            (Path(__file__).resolve().parents[1] / "reports/edge_2m_resolved.json").read_text()
        )
    )


def errors(actual, expected):
    delta = (actual - expected).abs()
    if not delta.numel():
        return {"max_abs": 0.0, "max_relative": 0.0}
    return {
        "max_abs": delta.max().item(),
        "max_relative": (delta / expected.abs().clamp_min(1e-8)).max().item(),
    }


def assert_states(actual, expected):
    assert actual.steps == expected.steps
    assert len(actual.layers) == len(expected.layers) == 4
    for a, b in zip(actual.layers, expected.layers, strict=True):
        torch.testing.assert_close(a.conv, b.conv, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(a.ssm, b.ssm, atol=ATOL, rtol=RTOL)


def equation_step(block, inputs, state):
    """FP64 equation oracle; never calls production projection, normalization or recurrence."""
    dtype = torch.float64
    x = inputs.to(dtype)
    normed = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5)
    projected = F.linear(normed * block.pre_norm.weight.to(dtype), block.in_proj.weight.to(dtype))
    s = block.shape
    z, raw, time = projected.split((s.d_inner, s.d_inner + 2 * s.d_state, s.nheads), -1)
    memory = torch.cat((state.conv.to(dtype)[:, :, 1:], raw.unsqueeze(-1)), -1)
    mixed = block.conv.bias.to(dtype).expand(x.shape[0], -1)
    for tap in range(s.d_conv):
        mixed = mixed + memory[:, :, tap] * block.conv.weight[:, 0, tap].to(dtype)
    mixed = mixed * mixed.sigmoid()
    values, b, c = mixed.split((s.d_inner, s.d_state, s.d_state), -1)
    values = values.reshape(x.shape[0], s.nheads, s.headdim)
    dt = torch.logaddexp(torch.zeros_like(time), time + block.dt_bias.to(dtype))
    a = -block.A_log.to(dtype).exp()
    next_ssm = state.ssm.to(dtype) * (dt * a).exp()[:, :, None, None]
    next_ssm = next_ssm + dt[:, :, None, None] * values[:, :, :, None] * b[:, None, None, :]
    readout = (next_ssm * c[:, None, None, :]).sum(-1)
    readout = readout + block.D.to(dtype)[None, :, None] * values
    gated = readout.flatten(-2) * z * z.sigmoid()
    normalized = gated * torch.rsqrt(gated.square().mean(-1, keepdim=True) + 1e-5)
    output = x + F.linear(
        normalized * block.gated_norm.weight.to(dtype), block.out_proj.weight.to(dtype)
    )
    return output, LayerState(memory, next_ssm)


def tensor_reconciliation(trunk):
    mapping = {
        "pre_norm": "pre_norm.weight",
        "in_proj": "in_proj.weight",
        "conv_weight": "conv.weight",
        "conv_bias": "conv.bias",
        "dt_bias": "dt_bias",
        "A_log": "A_log",
        "D": "D",
        "gated_norm": "gated_norm.weight",
        "out_proj": "out_proj.weight",
    }
    actual = dict(trunk.named_parameters())
    rows = []
    for spec in parameter_specs(trunk.config.shape):
        if spec.component != "shared":
            continue
        _, layer, name = spec.name.split("_", 2)
        tensor_name = f"layers.{layer}.{mapping[name]}"
        parameter = actual.pop(tensor_name)
        rows.append(
            {
                "layer": int(layer),
                "tensor_name": tensor_name,
                "expected_shape": list(spec.shape),
                "inventory_count": prod(spec.shape),
                "actual_shape": list(parameter.shape),
                "actual_count": parameter.numel(),
                "difference": parameter.numel() - prod(spec.shape),
            }
        )
    assert not actual, "unexpected executable shared tensors"
    return rows


def state_accounting(state):
    result = {}
    for kind in ("ssm", "conv"):
        tensors = [getattr(layer, kind) for layer in state.layers]
        assert len({t.untyped_storage().data_ptr() for t in tensors}) == len(tensors)
        result[kind] = {
            "elements": sum(t.numel() for t in tensors),
            "tensor_bytes": sum(t.numel() * t.element_size() for t in tensors),
            "storage_bytes": sum(t.untyped_storage().nbytes() for t in tensors),
            "per_layer_shape": list(tensors[0].shape),
        }
    result["combined_bytes"] = result["ssm"]["tensor_bytes"] + result["conv"]["tensor_bytes"]
    return result


def test_shared_every_tensor_and_initialization_contract():
    torch.manual_seed(23)
    trunk = SharedMambaTrunk(accepted_config())
    rows = tensor_reconciliation(trunk)
    assert len(rows) == 36
    assert all(r["difference"] == 0 and r["expected_shape"] == r["actual_shape"] for r in rows)
    audit = audit_parameters(trunk)
    assert audit["unique_trainable_parameters"] == 1_728_096
    assert audit["unique_frozen_parameters"] == audit["unique_buffer_elements"] == 0
    assert audit["aliases"] == {}
    for block in trunk.layers:
        dt = F.softplus(block.dt_bias)
        assert ((dt >= 0.001) & (dt <= 0.1)).all()
        rates = block.A_log.exp()
        assert ((rates >= 1) & (rates <= 16)).all()
        assert torch.equal(block.D, torch.ones_like(block.D))
        assert all(p._no_weight_decay for p in (block.dt_bias, block.A_log, block.D))
        assert block.in_proj.bias is block.out_proj.bias is None


@pytest.mark.parametrize("batch", (1, 2, 4))
def test_actual_state_bytes_initialization_and_reset(batch):
    trunk = SharedMambaTrunk(accepted_config())
    state = trunk.initialize_state(batch)
    count = state_accounting(state)
    assert count["ssm"]["elements"] == batch * 4 * 8 * 64 * 64
    assert count["conv"]["elements"] == batch * 4 * (512 + 2 * 64) * 4
    assert count["combined_bytes"] == 565_248 * batch
    assert count["ssm"]["storage_bytes"] == count["ssm"]["tensor_bytes"]
    assert count["conv"]["storage_bytes"] == count["conv"]["tensor_bytes"]
    assert state.steps == 0
    for layer in state.layers:
        assert torch.count_nonzero(layer.conv) == torch.count_nonzero(layer.ssm) == 0
    with torch.no_grad():
        _, advanced = trunk.step(torch.randn(batch, 256), state)
        reset = trunk.reset(advanced)
    assert_states(reset, state)
    assert all(v.ssm.abs().sum() > 0 for v in advanced.layers)
    assert all(v.conv.abs().sum() > 0 for v in advanced.layers)
    assert state_accounting(advanced)["combined_bytes"] == count["combined_bytes"]


@pytest.mark.parametrize("seed,batch", ((7, 1), (29, 2), (101, 4)))
@torch.no_grad()
def test_dense_ssd_step_equation_and_chunk_continuity(seed, batch):
    torch.manual_seed(seed)
    trunk = SharedMambaTrunk(accepted_config()).eval()
    for length in LENGTHS:
        data = torch.randn(batch, length, 256)
        expected, final = trunk(data)
        stepped, state = [], trunk.initialize_state(batch)
        for t in range(length):
            out, state = trunk.step(data[:, t], state)
            stepped.append(out)
        actual = torch.stack(stepped, 1) if length else data.clone()
        torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)
        assert_states(state, final)
        for split in sorted({0, length // 2, min(1, length), max(0, length - 1), length}):
            left, middle = trunk(data[:, :split])
            right, resumed = trunk(data[:, split:], middle)
            torch.testing.assert_close(torch.cat((left, right), 1), expected, atol=ATOL, rtol=RTOL)
            assert_states(resumed, final)
    # Exercise the independent high-precision oracle on all four layers with nonzero state.
    state = trunk.initialize_state(batch)
    for _ in range(4):
        inputs = torch.randn(batch, 256)
        oracle = inputs
        reference_layers = []
        for block, layer_state in zip(trunk.layers, state.layers, strict=True):
            oracle, reference_state = equation_step(block, oracle, layer_state)
            reference_layers.append(reference_state)
        out, actual_state = trunk.step(inputs, state)
        torch.testing.assert_close(out.double(), oracle, atol=ATOL, rtol=RTOL)
        for a, b in zip(actual_state.layers, reference_layers, strict=True):
            torch.testing.assert_close(a.ssm.double(), b.ssm, atol=ATOL, rtol=RTOL)
            torch.testing.assert_close(a.conv.double(), b.conv, atol=ATOL, rtol=RTOL)
        state = actual_state


@torch.no_grad()
def test_serialized_multilayer_resume_and_snapshot_ownership():
    torch.manual_seed(83)
    trunk = SharedMambaTrunk(accepted_config())
    data = torch.randn(2, 32, 256)
    uninterrupted, final = trunk(data)
    for split in (0, 1, 7, 8, 9, 16, 17, 31, 32):
        prefix, state = trunk(data[:, :split])
        snapshot = trunk.export_state(state)
        assert all(not item["ssm"].requires_grad for item in snapshot["layers"])
        buffer = io.BytesIO()
        torch.save({"weights": trunk.state_dict(), "state": snapshot}, buffer)
        buffer.seek(0)
        payload = torch.load(buffer, weights_only=True)
        restored_model = SharedMambaTrunk(trunk.config)
        restored_model.load_state_dict(payload["weights"])
        restored = restored_model.restore_state(payload["state"])
        suffix, resumed = restored_model(data[:, split:], restored)
        torch.testing.assert_close(
            torch.cat((prefix, suffix), 1), uninterrupted, atol=ATOL, rtol=RTOL
        )
        assert_states(resumed, final)
        payload["state"]["layers"][0]["ssm"].fill_(99)
        assert not torch.equal(restored.layers[0].ssm, payload["state"]["layers"][0]["ssm"])
        snapshot["layers"][0]["conv"].fill_(99)
        assert not torch.equal(state.layers[0].conv, snapshot["layers"][0]["conv"])


def test_reject_corrupt_states_and_unsupported_precision():
    trunk = SharedMambaTrunk(accepted_config())
    good = trunk.export_state(trunk.initialize_state(2))
    mutations = [
        lambda s: s.update(extra=1),
        lambda s: s.update(schema="2"),
        lambda s: s.update(resolved_sha256="wrong"),
        lambda s: s.update(batch=True),
        lambda s: s.update(batch=1),
        lambda s: s.update(steps=-1),
        lambda s: s.update(steps=True),
        lambda s: s.update(layers=s["layers"][:-1]),
        lambda s: s["layers"][0].update(extra=1),
        lambda s: s["layers"][0].update(conv=torch.zeros(2, 640, 3)),
        lambda s: s["layers"][0].update(ssm=torch.zeros(2, 8, 64, 63)),
        lambda s: s["layers"][0].update(ssm=s["layers"][0]["ssm"].double()),
        lambda s: s["layers"][0].update(ssm=torch.zeros(2, 8, 64, 64, device="meta")),
        lambda s: s["layers"][2]["ssm"].fill_(float("nan")),
        lambda s: s["layers"][3]["conv"].fill_(float("inf")),
    ]
    for mutate in mutations:
        snapshot = copy.deepcopy(good)
        mutate(snapshot)
        with pytest.raises(ValueError):
            trunk.restore_state(snapshot)
    wrong = replace(trunk.config, authored_sha256="0" * 64)
    with pytest.raises(ValueError, match="identity"):
        SharedMambaTrunk(wrong).restore_state(good)
    for invalid in (0, -1, True):
        with pytest.raises(ValueError, match="batch"):
            trunk.initialize_state(invalid)
    state = trunk.initialize_state(2)
    with pytest.raises(ValueError, match="shape"):
        trunk.step(torch.randn(1, 256), state)
    with pytest.raises(ValueError, match="non-finite"):
        trunk.step(torch.full((2, 256), float("nan")), state)
    with pytest.raises(ValueError, match="CPU FP32"):
        trunk.double().initialize_state(1)


def test_nonfinite_math_fails_with_layer_context_without_clamping():
    trunk = SharedMambaTrunk(accepted_config())
    state = trunk.initialize_state(1)
    with torch.no_grad():
        trunk.layers[2].A_log.fill_(1000)
    with pytest.raises(ValueError, match=r"layer 2:.*exp\(A_log\)"):
        trunk.step(torch.ones(1, 256), state)
    with pytest.raises(ValueError, match=r"layer 2:.*exp\(A_log\)"):
        trunk(torch.ones(1, 2, 256), state)


def test_ssd_and_step_gradient_equivalence():
    torch.manual_seed(91)
    full = SharedMambaTrunk(accepted_config())
    step = copy.deepcopy(full)
    x = torch.randn(2, 9, 256, requires_grad=True)
    other = x.detach().clone().requires_grad_()
    y, _ = full(x)
    y.square().mean().backward()
    state, outputs = step.initialize_state(2), []
    for t in range(9):
        output, state = step.step(other[:, t], state)
        outputs.append(output)
    torch.stack(outputs, 1).square().mean().backward()
    torch.testing.assert_close(x.grad, other.grad, atol=ATOL, rtol=RTOL)
    for (name, a), (_, b) in zip(full.named_parameters(), step.named_parameters(), strict=True):
        assert a.grad is not None and b.grad is not None, name
        assert torch.isfinite(a.grad).all() and a.grad.abs().sum() > 0, name
        torch.testing.assert_close(a.grad, b.grad, atol=ATOL, rtol=RTOL)


def test_block_explicit_incoming_state_is_not_mutated():
    block = Mamba2Block(accepted_config().shape)
    state = block.initialize_state(1)
    state.ssm.uniform_(-0.1, 0.1)
    state.conv.uniform_(-0.1, 0.1)
    original = LayerState(state.conv.clone(), state.ssm.clone())
    data = torch.randn(1, 2, 256)
    full, final = block(data, state)
    left, middle = block.step(data[:, 0], state)
    right, end = block.step(data[:, 1], middle)
    torch.testing.assert_close(full, torch.stack((left, right), 1), atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(final.ssm, end.ssm, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(state.conv, original.conv, atol=0, rtol=0)
    torch.testing.assert_close(state.ssm, original.ssm, atol=0, rtol=0)


def test_finite_dt_and_rate_cannot_hide_overflow_in_discretization():
    trunk = SharedMambaTrunk(accepted_config())
    state = trunk.initialize_state(1)
    with torch.no_grad():
        trunk.layers[1].A_log.fill_(80)
        trunk.layers[1].dt_bias.fill_(1e10)
    assert torch.isfinite(trunk.layers[1].A_log.exp()).all()
    assert torch.isfinite(F.softplus(trunk.layers[1].dt_bias)).all()
    with pytest.raises(ValueError, match="layer 1:.*log decay"):
        trunk.step(torch.ones(1, 256), state)
    with pytest.raises(ValueError, match="layer 1:.*log decay"):
        trunk(torch.ones(1, 2, 256), state)
