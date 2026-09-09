import copy
import io
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F
from test_mamba import accepted_config, assert_states

from unified_edge.dense_model import DenseByteModel, DenseState
from unified_edge.parameters import audit_parameters
from unified_edge.symbols import BOS_ID, PAD_ID

ATOL, RTOL = 1e-5, 1e-4
CUTS = (0, 1, 2, 6, 7, 8, 9, 15, 16, 17, 62, 63, 64, 65, 126, 127, 128, 129)


@pytest.fixture(scope="module", autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def model():
    torch.manual_seed(43)
    return DenseByteModel(accepted_config()).eval()


@torch.no_grad()
def stream(model, symbols, state=None):
    state = model.start(symbols.shape[0]) if state is None else state
    outputs = []
    for t in range(symbols.shape[1]):
        outputs.append(model.predict(state))
        state = model.consume(symbols[:, t], state)
    empty = model.hierarchy.output.weight.new_empty(symbols.shape[0], 0, 267)
    return (torch.stack(outputs, 1) if outputs else empty), state


def finite_logits(value):
    assert torch.isneginf(value[..., [PAD_ID, BOS_ID]]).all()
    mask = torch.ones(267, dtype=torch.bool)
    mask[[PAD_ID, BOS_ID]] = False
    assert torch.isfinite(value[..., mask]).all()
    return value[..., mask]


def test_full_executable_reconciliation(model):
    actual = audit_parameters(model)
    assert actual["unique_trainable_parameters"] == 1_929_579
    assert actual["components"] == {"hierarchy": 201_483, "shared": 1_728_096}
    assert actual["aliases"] == {}
    assert actual["unique_buffer_elements"] == actual["unique_frozen_parameters"] == 0
    assert actual["state_dict_entry_elements"] == actual["unique_trainable_parameters"]
    expected = {
        "embedding": 17600,
        "encoder": 25280,
        "decoder": 107520,
        "output": 34443,
        "bootstrap": 16640,
    }
    assert audit_parameters(model.hierarchy)["components"] == expected


def test_bos_goes_through_all_shared_layers_in_production(model, monkeypatch):
    calls = []
    original = model.shared.step

    def observe(inputs, state):
        calls.append((inputs.detach().clone(), state.steps))
        return original(inputs, state)

    monkeypatch.setattr(model.shared, "step", observe)
    state = model.start(2)
    assert len(calls) == 1 and calls[0][1] == 0
    bos = model.hierarchy.initial_conditioning(2)
    torch.testing.assert_close(calls[0][0], bos, atol=0, rtol=0)
    context, expected_shared = original(bos, model.shared.initialize_state(2))
    expected_local = model.hierarchy.start(2, conditioning=context)
    torch.testing.assert_close(state.hierarchy.hidden, expected_local.hidden, atol=0, rtol=0)
    assert_states(state.shared, expected_shared)
    assert not torch.allclose(state.hierarchy.hidden, model.hierarchy.start(2).hidden)
    assert all(
        layer.conv.abs().sum() > 0 and layer.ssm.abs().sum() > 0 for layer in state.shared.layers
    )
    logits = model(torch.tensor([[10], [20]]))
    torch.testing.assert_close(logits[:, 0], model.predict(state), atol=ATOL, rtol=RTOL)
    finite_logits(logits).square().mean().backward()
    for layer in model.shared.layers:
        assert layer.in_proj.weight.grad.abs().sum() > 0
        assert layer.out_proj.weight.grad.abs().sum() > 0


@torch.no_grad()
def test_each_completed_event_updates_once_and_only_after_eight_symbols(model, monkeypatch):
    steps = []
    original = model.shared.step

    def observe(inputs, state):
        steps.append(state.steps)
        return original(inputs, state)

    monkeypatch.setattr(model.shared, "step", observe)
    state = model.start()
    for consumed in range(1, 138):
        old_state = state
        state = model.consume(torch.tensor([consumed % 256]), state)
        assert state.shared.steps == 1 + consumed // 8
        assert state.hierarchy.completed_patches == consumed // 8
        assert state.hierarchy.pending.shape[1] == consumed % 8
        assert not state.hierarchy.awaiting_context
        if consumed % 8:
            assert state.shared is old_state.shared
        assert len(steps) == 1 + consumed // 8
    assert steps == list(range(18))
    broken = replace(state, shared=replace(state.shared, steps=state.shared.steps + 1))
    with pytest.raises(ValueError, match="clock"):
        model.predict(broken)
    with pytest.raises(ValueError, match="clock"):
        model.consume(torch.tensor([3]), broken)
    waiting = replace(state.hierarchy, hidden=None, pending=state.hierarchy.pending[:, :0])
    with pytest.raises(ValueError, match="unprocessed"):
        model.predict(DenseState(waiting, state.shared))


@pytest.mark.parametrize("seed,batch", ((3, 1), (37, 2), (79, 1)))
@torch.no_grad()
def test_integrated_full_step_and_adversarial_causality(seed, batch):
    torch.manual_seed(seed)
    model = DenseByteModel(accepted_config()).eval()
    data = torch.randint(0, 256, (batch, 137))
    full = model(data)
    incremental, _ = stream(model, data)
    finite_logits(full)
    torch.testing.assert_close(full, incremental, atol=ATOL, rtol=RTOL)
    for cut in CUTS:
        changed = data.clone()
        changed[:, cut:] = (changed[:, cut:] + 113) % 256
        output = model(changed)
        torch.testing.assert_close(output[:, : cut + 1], full[:, : cut + 1], atol=ATOL, rtol=RTOL)
        assert not torch.equal(output[:, cut + 1 :, :256], full[:, cut + 1 :, :256])
    for cut in (0, 7, 8, 9, 63, 64, 65, 127, 128, 129):
        prefix, state = stream(model, data[:, :cut])
        torch.testing.assert_close(prefix, full[:, :cut], atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(model.predict(state), full[:, cut], atol=ATOL, rtol=RTOL)


@torch.no_grad()
def test_integrated_saved_state_continues_at_arbitrary_byte_boundaries(model):
    data = torch.randint(0, 256, (2, 137))
    reference, final = stream(model, data)
    for split in (0, 1, 7, 8, 9, 15, 16, 17, 63, 64, 65, 127, 128, 129):
        prefix, state = stream(model, data[:, :split])
        buffer = io.BytesIO()
        torch.save({"weights": model.state_dict(), "state": model.export_state(state)}, buffer)
        buffer.seek(0)
        saved = torch.load(buffer, weights_only=True)
        restored = DenseByteModel(model.config)
        restored.load_state_dict(saved["weights"])
        restored_state = restored.restore_state(saved["state"])
        torch.testing.assert_close(
            restored.predict(restored_state), model.predict(state), atol=0, rtol=0
        )
        altered_future = (data[:, split] + 17) % 256
        restored.consume(altered_future, restored_state)
        torch.testing.assert_close(
            restored.predict(restored_state), model.predict(state), atol=0, rtol=0
        )
        suffix, resumed = stream(restored, data[:, split:], restored_state)
        torch.testing.assert_close(torch.cat((prefix, suffix), 1), reference, atol=0, rtol=0)
        assert_states(resumed.shared, final.shared)
        torch.testing.assert_close(resumed.hierarchy.hidden, final.hierarchy.hidden, atol=0, rtol=0)


@torch.no_grad()
def test_prefix_memory_through_mamba_and_reset(model):
    a = torch.zeros(1, 8, dtype=torch.long)
    b = torch.full_like(a, 255)
    _, state_a = stream(model, a)
    _, state_b = stream(model, b)
    for left, right in zip(state_a.shared.layers, state_b.shared.layers, strict=True):
        assert not torch.equal(left.ssm, right.ssm)
    common = torch.tensor([[1, 2, 3, 4, 5]])
    logits_a, state_a = stream(model, common, state_a)
    logits_b, state_b = stream(model, common, state_b)
    assert (finite_logits(logits_a) - finite_logits(logits_b)).abs().max() > 1e-5
    reset_a, reset_b = model.reset(state_a), model.reset(state_b)
    initial = model.start()
    torch.testing.assert_close(model.predict(reset_a), model.predict(initial), atol=0, rtol=0)
    torch.testing.assert_close(model.predict(reset_b), model.predict(initial), atol=0, rtol=0)
    assert_states(reset_a.shared, initial.shared)
    assert reset_a.shared.steps == 1
    assert reset_a.hierarchy.completed_patches == 0


@pytest.mark.parametrize("seed", (13, 59, 97))
def test_integrated_finite_nonzero_gradients_for_every_parameter(seed):
    torch.manual_seed(seed)
    model = DenseByteModel(accepted_config())
    data = torch.randint(0, 256, (2, 25))
    logits = model(data)
    loss = F.cross_entropy(logits.flatten(0, 1), data.flatten())
    assert torch.isfinite(loss)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    assert model.hierarchy.embedding.symbols.weight.grad[PAD_ID].count_nonzero() == 0


@torch.no_grad()
def test_empty_lengths_and_invalid_integrated_contracts(model):
    for length in (0, 1, 2, 7, 8, 9, 15, 16, 17, 127, 128, 129):
        data = torch.arange(2 * length).reshape(2, length) % 256
        full = model(data)
        stepped, _ = stream(model, data)
        assert full.shape == (2, length, 267)
        torch.testing.assert_close(full, stepped, atol=ATOL, rtol=RTOL)
    state = model.start()
    for invalid in (PAD_ID, BOS_ID, -1, 267):
        with pytest.raises(ValueError):
            model(torch.tensor([[invalid]]))
        with pytest.raises(ValueError):
            model.consume(torch.tensor([invalid]), state)
    snapshot = model.export_state(state)
    for mutate in (
        lambda s: s.update(extra=1),
        lambda s: s.update(schema="99"),
        lambda s: s.update(resolved_sha256="wrong"),
        lambda s: s["shared"].update(steps=0),
        lambda s: s["shared"].update(batch=2),
        lambda s: s["hierarchy"].update(hidden=torch.full((1, 128), float("inf"))),
    ):
        broken = copy.deepcopy(snapshot)
        mutate(broken)
        with pytest.raises(ValueError):
            model.restore_state(broken)
