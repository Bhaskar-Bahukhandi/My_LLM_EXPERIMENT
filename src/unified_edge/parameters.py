"""Instantiated parameter accounting; no executable neural model is defined here."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

import torch
from torch import nn

from unified_edge.config import NUM_SYMBOLS, ConfigError, ModelShape, validate_architecture


@dataclass(frozen=True)
class TensorSpec:
    component: str
    name: str
    shape: tuple[int, ...]


def parameter_specs(shape: ModelShape) -> tuple[TensorSpec, ...]:
    validate_architecture(shape)
    d, b, q = shape.d_model, shape.byte_dim, shape.decoder_dim
    inner, n, h = shape.d_inner, shape.d_state, shape.nheads
    specs = []

    def add(component, name, *dims):
        specs.append(TensorSpec(component, name, dims))

    add("embedding", "symbols", NUM_SYMBOLS, b)
    add("embedding", "positions", shape.patch_size, b)
    add("encoder", "conv_weight", b, 1, 3)
    add("encoder", "conv_bias", b)
    add("encoder", "gate_weight", 2 * b, b)
    add("encoder", "gate_bias", 2 * b)
    add("encoder", "norm", b)
    add("encoder", "projection_weight", d, b)
    add("encoder", "projection_bias", d)
    for layer in range(shape.shared_layers):
        prefix = f"layer_{layer}_"
        for name, dims in (
            ("pre_norm", (d,)),
            ("in_proj", (2 * inner + 2 * n + h, d)),
            ("conv_weight", (inner + 2 * n, 1, shape.d_conv)),
            ("conv_bias", (inner + 2 * n,)),
            ("dt_bias", (h,)),
            ("A_log", (h,)),
            ("D", (h,)),
            ("gated_norm", (inner,)),
            ("out_proj", (d, inner)),
        ):
            add("shared", prefix + name, *dims)
    add("decoder", "context_weight", q, d)
    add("decoder", "context_bias", q)
    add("decoder", "gru_weight_ih", 3 * q, b)
    add("decoder", "gru_weight_hh", 3 * q, q)
    add("decoder", "gru_bias_ih", 3 * q)
    add("decoder", "gru_bias_hh", 3 * q)
    add("decoder", "norm", q)
    add("output", "weight", NUM_SYMBOLS, q)
    add("output", "bias", NUM_SYMBOLS)
    add("bootstrap", "weight", d, b)
    add("bootstrap", "bias", d)
    for spec in specs:
        if prod(spec.shape) > 2**63 - 1:
            raise ConfigError(f"{spec.component}.{spec.name} exceeds int64 tensor indexing")
    return tuple(specs)


def instantiate_inventory(shape: ModelShape) -> nn.Module:
    """Allocate declared Parameter metadata only; this container has no forward path."""
    inventory = nn.Module()
    for spec in parameter_specs(shape):
        if spec.component not in inventory._modules:
            inventory.add_module(spec.component, nn.Module())
        inventory.get_submodule(spec.component).register_parameter(
            spec.name, nn.Parameter(torch.empty(spec.shape, device="meta", dtype=torch.float32))
        )
    return inventory


def audit_parameters(module: nn.Module) -> dict:
    """Count unique tensor identities, while exposing serialization aliases separately."""
    trainable = frozen = buffer_elements = weight_bytes = buffer_bytes = 0
    seen_parameters, seen_buffers = {}, {}
    aliases, components = {}, {}
    for name, parameter in module.named_parameters(remove_duplicate=False):
        if id(parameter) in seen_parameters:
            aliases[name] = seen_parameters[id(parameter)]
            continue
        seen_parameters[id(parameter)] = name
        count = parameter.numel()
        weight_bytes += count * parameter.element_size()
        if parameter.requires_grad:
            trainable += count
            component = name.split(".")[0]
            components[component] = components.get(component, 0) + count
        else:
            frozen += count
    for name, buffer in module.named_buffers(remove_duplicate=False):
        if id(buffer) in seen_buffers:
            aliases[name] = seen_buffers[id(buffer)]
            continue
        seen_buffers[id(buffer)] = name
        buffer_elements += buffer.numel()
        buffer_bytes += buffer.numel() * buffer.element_size()
    serialized = module.state_dict()
    tensor_entries = {name: t for name, t in serialized.items() if isinstance(t, torch.Tensor)}
    return {
        "unique_trainable_parameters": trainable,
        "unique_frozen_parameters": frozen,
        "unique_buffer_elements": buffer_elements,
        "unique_parameter_bytes": weight_bytes,
        "unique_buffer_bytes": buffer_bytes,
        "components": dict(sorted(components.items())),
        "aliases": dict(sorted(aliases.items())),
        "state_dict_tensor_entries": len(tensor_entries),
        "state_dict_non_tensor_entries": sorted(set(serialized) - set(tensor_entries)),
        "state_dict_entry_elements": sum(t.numel() for t in tensor_entries.values()),
        "state_dict_logical_payload_bytes": sum(
            t.numel() * t.element_size() for t in tensor_entries.values()
        ),
        "serialized_file_bytes": None,
        "payload_note": "logical entries; aliases may share storage; not serialized file size",
    }


def formula_parameter_estimate(shape: ModelShape) -> int:
    """Independent arithmetic cross-check of inventory v1, never the audit authority."""
    d, b, q = shape.d_model, shape.byte_dim, shape.decoder_dim
    i, n, h = shape.d_inner, shape.d_state, shape.nheads
    embedding = (NUM_SYMBOLS + shape.patch_size) * b
    encoder = 3 * b + b + 2 * b * b + 2 * b + b + d * b + d
    block = d + (2 * i + 2 * n + h) * d + (i + 2 * n) * (shape.d_conv + 1)
    block += 3 * h + i + d * i
    decoder = q * d + q + 3 * q * b + 3 * q * q + 6 * q + q
    output = NUM_SYMBOLS * (q + 1)
    bootstrap = d * b + d
    return embedding + encoder + shape.shared_layers * block + decoder + output + bootstrap
