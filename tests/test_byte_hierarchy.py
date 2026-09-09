import io
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from unified_edge.byte_hierarchy import ByteHierarchy, HierarchyState
from unified_edge.config import AuthoredConfig, ModelRequest
from unified_edge.parameters import audit_parameters, parameter_specs
from unified_edge.resolve import resolve_config
from unified_edge.symbols import BOS_ID, EOS_ID, PAD_ID

ATOL, RTOL = 1e-6, 1e-5


@pytest.fixture
def hierarchy():
    torch.manual_seed(19)
    config = resolve_config(AuthoredConfig()).selected
    model = ByteHierarchy(config).eval()
    return model


def test_bootstrap_is_learned_bos_conditioning(hierarchy):
    state = hierarchy.start(2)
    bos = hierarchy.embedding.symbols.weight[BOS_ID].expand(2, -1)
    context = F.linear(bos, hierarchy.bootstrap.weight, hierarchy.bootstrap.bias)
    expected = torch.tanh(
        F.linear(context, hierarchy.decoder.context.weight, hierarchy.decoder.context.bias)
    )
    torch.testing.assert_close(state.hidden, expected, atol=ATOL, rtol=RTOL)
    assert not torch.equal(state.hidden, torch.zeros_like(state.hidden))
    assert state.pending.shape == (2, 0)
    assert state.completed_patches == 0 and not state.awaiting_context
    assert hierarchy.embedding.symbols.weight[PAD_ID].count_nonzero() == 0
    assert hierarchy.embedding.symbols.weight[0].count_nonzero() > 0
    logits = hierarchy.predict(state)
    assert torch.isneginf(logits[:, [PAD_ID, BOS_ID]]).all()
    assert torch.isfinite(logits[:, :256]).all()
    assert torch.isfinite(logits[:, EOS_ID:]).all()


def test_patch_encoder_must_depend_on_all_eight_positions(hierarchy):
    embedded = torch.randn(1, 1, 8, hierarchy.config.shape.byte_dim, requires_grad=True)
    encoded = hierarchy.encoder(embedded)
    derivative = torch.autograd.grad(encoded.square().sum(), embedded)[0]
    for position in range(8):
        assert derivative[:, :, position].abs().sum() > 0, f"ignored position {position}"


def test_patch_encoding_is_nonoverlapping_and_matches_independent_local_equations(hierarchy):
    ids = torch.arange(32, dtype=torch.long).reshape(1, 4, 8)
    whole = hierarchy.encode_completed(ids)
    singles = torch.cat([hierarchy.encode_completed(ids[:, j : j + 1]) for j in range(4)], dim=1)
    torch.testing.assert_close(whole, singles, atol=ATOL, rtol=RTOL)
    changed = ids.clone()
    changed[:, 2] = 255
    result = hierarchy.encode_completed(changed)
    torch.testing.assert_close(result[:, [0, 1, 3]], whole[:, [0, 1, 3]], atol=0, rtol=0)
    assert not torch.allclose(result[:, 2], whole[:, 2])
    embedded = F.embedding(ids, hierarchy.embedding.symbols.weight)
    embedded = embedded + hierarchy.embedding.positions.weight
    encoder = hierarchy.encoder
    mixed = []
    for t in range(8):
        conv = encoder.conv.bias.expand(1, 4, -1)
        for k in range(3):
            source = t - 2 + k
            if source >= 0:
                conv = conv + embedded[:, :, source] * encoder.conv.weight[:, 0, k]
        value, gate = F.linear(conv, encoder.gate.weight, encoder.gate.bias).chunk(2, -1)
        y = value * gate * torch.sigmoid(gate)
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + encoder.norm.eps)
        mixed.append(y * encoder.norm.weight)
    expected = F.linear(
        torch.stack(mixed, 2).mean(2), encoder.projection.weight, encoder.projection.bias
    )
    torch.testing.assert_close(whole, expected, atol=ATOL, rtol=RTOL)
    assert hierarchy.encode_completed(torch.empty(2, 0, 8, dtype=torch.long)).shape == (2, 0, 256)


@pytest.mark.parametrize("length", [0, 1, 2, 7, 8, 9, 15, 16, 17, 127, 128, 129])
def test_streaming_complete_and_partial_patch_geometry(hierarchy, length):
    ids = (torch.arange(2 * length).reshape(2, length) % 256).long()
    state = hierarchy.start(2)
    events = []
    for t in range(length):
        if state.awaiting_context:
            # Supplied test context only; there is deliberately no production global trunk.
            state = hierarchy.condition(events[-1].representation, state)
        old_pending = state.pending.clone()
        old_hidden = state.hidden.clone()
        old_state = state
        state, event = hierarchy.consume(ids[:, t], state)
        torch.testing.assert_close(old_state.pending, old_pending, atol=0, rtol=0)
        torch.testing.assert_close(old_state.hidden, old_hidden, atol=0, rtol=0)
        assert state.completed_patches == (t + 1) // 8
        assert state.pending.shape[1] == (t + 1) % 8
        assert (event is not None) == ((t + 1) % 8 == 0)
        if event is not None:
            assert event.index == len(events)
            expected = hierarchy.encode_completed(ids[:, t - 7 : t + 1, None].transpose(1, 2))[:, 0]
            torch.testing.assert_close(event.representation, expected, atol=ATOL, rtol=RTOL)
            events.append(event)
            with pytest.raises(ValueError, match="awaits external"):
                hierarchy.predict(state)
            with pytest.raises(ValueError, match="awaits external"):
                hierarchy.consume(ids[:, t], state)
    assert len(events) == length // 8
    assert state.pending.tolist() == ids[:, length - length % 8 :].tolist()
    assert state.awaiting_context == (length > 0 and length % 8 == 0)


def independent_logits(model, targets):
    """Manual GRU gates and RMSNorm, independent of production decoder/step wrappers."""
    batch, length = targets.shape
    bos = model.embedding.symbols.weight[BOS_ID].expand(batch, -1)
    context = F.linear(bos, model.bootstrap.weight, model.bootstrap.bias)
    hidden = torch.tanh(F.linear(context, model.decoder.context.weight, model.decoder.context.bias))
    outputs = []
    for j in range(length):
        normalized = (
            hidden
            * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + model.decoder.norm.eps)
            * model.decoder.norm.weight
        )
        logits = F.linear(normalized, model.output.weight, model.output.bias)
        logits[:, [PAD_ID, BOS_ID]] = -torch.inf
        outputs.append(logits)
        embedded = F.embedding(targets[:, j], model.embedding.symbols.weight)
        embedded = embedded + model.embedding.positions.weight[j]
        ir, iz, inn = F.linear(
            embedded, model.decoder.gru.weight_ih, model.decoder.gru.bias_ih
        ).chunk(3, -1)
        hr, hz, hn = F.linear(hidden, model.decoder.gru.weight_hh, model.decoder.gru.bias_hh).chunk(
            3, -1
        )
        reset = torch.sigmoid(ir + hr)
        update = torch.sigmoid(iz + hz)
        candidate = torch.tanh(inn + reset * hn)
        hidden = candidate + update * (hidden - candidate)
    return torch.stack(outputs, 1)


def test_local_teacher_forcing_matches_manual_equations_and_incremental(hierarchy):
    targets = torch.randint(0, 256, (2, 8))
    full = hierarchy(targets)
    independent = independent_logits(hierarchy, targets)
    torch.testing.assert_close(full, independent, atol=ATOL, rtol=RTOL)
    state = hierarchy.start(2)
    incremental = []
    for j in range(8):
        incremental.append(hierarchy.predict(state))
        state, _ = hierarchy.consume(targets[:, j], state)
    torch.testing.assert_close(full, torch.stack(incremental, 1), atol=ATOL, rtol=RTOL)
    assert hierarchy(torch.empty(2, 0, dtype=torch.long)).shape == (2, 0, 267)


def test_teacher_forcing_future_inputs_have_zero_gradient(hierarchy):
    context = hierarchy.initial_conditioning(1).detach()
    embedded = torch.randn(1, 8, hierarchy.config.shape.byte_dim, requires_grad=True)
    features = hierarchy.decoder(context, embedded)
    for t in (1, 3, 7):
        gradient = torch.autograd.grad(features[:, t].square().sum(), embedded, retain_graph=True)[
            0
        ]
        assert torch.count_nonzero(gradient[:, t:]) == 0
        assert gradient[:, :t].abs().sum() > 0


@torch.no_grad()
def causal_test_trace(model, symbols):
    """Test-only causal accumulator of emitted patches; no Mamba semantics claimed."""
    batch, length = symbols.shape
    context = model.initial_conditioning(batch)
    state = model.start(batch)
    logits, states, events = [], [], []
    for t in range(length):
        if state.awaiting_context:
            context = context + events[-1].representation
            state = model.condition(context, state)
        logits.append(model.predict(state))
        states.append(model.export_state(state))
        state, event = model.consume(symbols[:, t], state)
        if event is not None:
            events.append(event)
    if length == 0:
        return torch.empty(batch, 0, 267), states, events
    return torch.stack(logits, 1), states, events


def test_adversarial_prefix_invariance_across_first_patch_and_128_boundary(hierarchy):
    generator = torch.Generator().manual_seed(819)
    original = torch.randint(0, 256, (1, 137), generator=generator)
    reference, reference_states, reference_events = causal_test_trace(hierarchy, original)
    for cut in (0, 1, 2, 6, 7, 8, 9, 15, 16, 17, 126, 127, 128, 129):
        changed = original.clone()
        changed[:, cut:] = (changed[:, cut:] + 127) % 256
        predictions, states, events = causal_test_trace(hierarchy, changed)
        torch.testing.assert_close(
            predictions[:, : cut + 1], reference[:, : cut + 1], atol=0, rtol=0
        )
        torch.testing.assert_close(
            states[cut]["pending"], reference_states[cut]["pending"], atol=0, rtol=0
        )
        torch.testing.assert_close(
            states[cut]["hidden"], reference_states[cut]["hidden"], atol=0, rtol=0
        )
        for index in range(cut // 8):
            torch.testing.assert_close(
                events[index].representation, reference_events[index].representation, atol=0, rtol=0
            )
        assert not torch.equal(predictions[:, cut + 1 :, :256], reference[:, cut + 1 :, :256])


def test_direct_first_patch_adversarial_causality(hierarchy):
    original = torch.tensor([[0, 255, 128, 192, 1, 2, 3, 4]])
    expected = hierarchy(original)
    for cut in range(8):
        changed = original.clone()
        changed[:, cut:] = (changed[:, cut:] + 1) % 256
        actual = hierarchy(changed)
        torch.testing.assert_close(actual[:, : cut + 1], expected[:, : cut + 1], atol=0, rtol=0)


def test_partial_and_awaiting_state_serialization_and_future_suffix(hierarchy):
    generator = torch.Generator().manual_seed(771)
    source = torch.randint(0, 256, (1, 137), generator=generator)
    context = hierarchy.initial_conditioning(1)
    state = hierarchy.start()
    last_event = None
    for consumed in range(130):
        if consumed in (0, 1, 7, 8, 9, 15, 16, 17, 127, 128, 129):
            buffer = io.BytesIO()
            # The caller owns external context and any outbound patch awaiting trunk processing.
            torch.save(
                {
                    "hierarchy": hierarchy.export_state(state),
                    "context": context,
                    "outbound_patch": None if last_event is None else last_event.representation,
                    "weights": hierarchy.state_dict(),
                },
                buffer,
            )
            buffer.seek(0)
            saved = torch.load(buffer, weights_only=True)
            restored_model = ByteHierarchy(hierarchy.config).eval()
            restored_model.load_state_dict(saved["weights"])
            restored = restored_model.restore_state(saved["hierarchy"])
            live = state
            if live.awaiting_context:
                with pytest.raises(ValueError, match="awaits external"):
                    restored_model.predict(restored)
                next_context = context + last_event.representation
                live = hierarchy.condition(next_context, live)
                restored = restored_model.condition(
                    saved["context"] + saved["outbound_patch"], restored
                )
            a = hierarchy.predict(live)
            b = restored_model.predict(restored)
            torch.testing.assert_close(a, b, atol=0, rtol=0)
            a, _ = hierarchy.consume(source[:, consumed], live)
            b, _ = restored_model.consume(source[:, consumed], restored)
            torch.testing.assert_close(a.pending, b.pending, atol=0, rtol=0)
            if a.hidden is not None:
                torch.testing.assert_close(a.hidden, b.hidden, atol=0, rtol=0)
            # An independently changed future still cannot affect this restored prefix's prediction.
            opposite = (source[:, consumed] + 17) % 256
            fork, _ = restored_model.consume(opposite, restored)
            if fork.hidden is not None:
                assert not torch.equal(fork.hidden, b.hidden)
        if state.awaiting_context:
            context = context + last_event.representation
            state = hierarchy.condition(context, state)
        state, event = hierarchy.consume(source[:, consumed], state)
        if event is not None:
            last_event = event


def test_corrupt_state_and_wrong_context_fail_closed(hierarchy):
    state = hierarchy.start()
    for mutate in (
        lambda s: s.update(extra=True),
        lambda s: s.update(state_schema="99"),
        lambda s: s.update(resolved_sha256="wrong"),
        lambda s: s.update(completed_patches=True),
        lambda s: s.update(pending=torch.tensor([[PAD_ID]])),
        lambda s: s.update(hidden=None),
        lambda s: s.update(hidden=torch.full((1, 128), float("nan"))),
        lambda s: s.update(hidden=torch.zeros(1, 129)),
    ):
        snapshot = hierarchy.export_state(state)
        mutate(snapshot)
        with pytest.raises((ValueError, TypeError)):
            hierarchy.restore_state(snapshot)
    for token in (PAD_ID, BOS_ID, -1, 267):
        with pytest.raises(ValueError):
            hierarchy.consume(torch.tensor([token]), state)
    with pytest.raises(ValueError):
        hierarchy.condition(torch.zeros(1, 256), state)
    with pytest.raises(ValueError):
        hierarchy.start(1, torch.zeros(2, 256))
    with pytest.raises(ValueError):
        hierarchy.start(1, torch.full((1, 256), float("inf")))
    with pytest.raises(ValueError):
        hierarchy(torch.zeros(1, 9, dtype=torch.long))
    for count in (0, -1, True):
        with pytest.raises(ValueError):
            hierarchy.start(count)


def test_random_lengths_controls_and_shape_contracts(hierarchy):
    generator = torch.Generator().manual_seed(921)
    for length in torch.randint(1, 160, (7,), generator=generator).tolist():
        data = torch.randint(0, 256, (2, length), generator=generator)
        data[:, length // 2] = EOS_ID
        logits, _, events = causal_test_trace(hierarchy, data)
        assert logits.shape == (2, length, 267)
        assert len(events) == length // 8
        assert torch.isfinite(logits[:, :, :256]).all()
        assert torch.isfinite(logits[:, :, EOS_ID:]).all()
    for shape in ((1, 7), (1, 1, 7), (1, 2, 9)):
        with pytest.raises(ValueError):
            hierarchy.encode_completed(torch.zeros(shape, dtype=torch.long))
    with pytest.raises(ValueError, match="PAD"):
        hierarchy.encode_completed(torch.full((1, 1, 8), PAD_ID))
    with pytest.raises(ValueError):
        hierarchy.embedding(torch.tensor([1, 2]), torch.tensor([8, 8]))
    with pytest.raises(ValueError):
        hierarchy.consume(torch.tensor([[1]]), hierarchy.start())
    with pytest.raises(ValueError):
        hierarchy.restore_state(
            hierarchy.export_state(
                HierarchyState(torch.zeros(1, 8, dtype=torch.long), torch.zeros(1, 128))
            )
        )


def test_partial_parameter_reconciliation_and_gradients(hierarchy):
    actual = audit_parameters(hierarchy)
    expected = {}
    for spec in parameter_specs(hierarchy.config.shape):
        if spec.component == "shared":
            continue
        expected.setdefault(spec.component, []).append(spec.shape)
    assert set(actual["components"]) == set(expected)
    assert actual["unique_trainable_parameters"] == 201_483
    assert actual["unique_buffer_elements"] == 0
    assert actual["aliases"] == {}
    for component, shapes in expected.items():
        parameters = list(getattr(hierarchy, component).parameters())
        assert sorted(tuple(p.shape) for p in parameters) == sorted(shapes)
        assert all(not p.is_meta and p.requires_grad for p in parameters)
    data = torch.arange(16).reshape(2, 8).long()
    local_loss = F.cross_entropy(hierarchy(data).reshape(-1, 267), data.reshape(-1))
    encoder_loss = hierarchy.encode_completed(data[:, None]).square().mean()
    (local_loss + encoder_loss).backward()
    for name, parameter in hierarchy.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert hierarchy.embedding.symbols.weight.grad[PAD_ID].count_nonzero() == 0


def test_same_implementation_accepts_another_legal_width():
    torch.manual_seed(17)
    config = resolve_config(
        AuthoredConfig(model=ModelRequest(d_model=128, shared_layers=2))
    ).selected
    model = ByteHierarchy(config)
    assert model(torch.ones(3, 7, dtype=torch.long)).shape == (3, 7, 267)
    assert model.encode_completed(torch.ones(3, 2, 8, dtype=torch.long)).shape == (3, 2, 128)
    wrong_config = replace(config, authored_sha256="0" * 64)
    with pytest.raises(ValueError, match="identity mismatch"):
        ByteHierarchy(wrong_config).restore_state(model.export_state(model.start()))


def test_explicit_initial_context_still_requires_a_valid_batch(hierarchy):
    with pytest.raises(ValueError, match="batch_size"):
        hierarchy.start(True, torch.zeros(1, 256))
