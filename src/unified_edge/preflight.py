"""Lower-bound memory screening; an inventory cannot prove runtime memory fit."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from unified_edge.config import Hardware, ModelShape, validate_architecture


@dataclass(frozen=True)
class MemoryPreflight:
    status: str
    parameter_bytes: int
    gradient_bytes: int
    adam_moment_bytes: int
    ssm_state_bytes: int
    convolution_state_bytes: int
    local_state_bytes: int
    known_training_payload_bytes: int
    known_inference_payload_bytes: int
    memory_budget_bytes: int | None
    activation_bytes: None = None
    allocator_overhead_bytes: None = None
    temporary_buffer_bytes: None = None
    peak_process_rss_bytes: None = None
    measured_runtime: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def memory_preflight(shape: ModelShape, audit: dict, hardware: Hardware) -> MemoryPreflight:
    validate_architecture(shape)
    batch = hardware.batch_size
    parameters = audit["unique_parameter_bytes"]
    gradients = audit["unique_trainable_parameters"] * 4
    moments = 2 * gradients
    ssm = batch * shape.shared_layers * shape.nheads * shape.headdim * shape.d_state * 4
    convolution = (
        batch * shape.shared_layers * (shape.d_inner + 2 * shape.d_state) * shape.d_conv * 4
    )
    # Planned persistent context, local hidden state and at most P-1 pending int64 symbols.
    local = batch * ((shape.d_model + shape.decoder_dim) * 4 + (shape.patch_size - 1) * 8)
    inference = parameters + ssm + convolution + local
    training = inference + gradients + moments
    status = "UNKNOWN_REQUIRES_MEASUREMENT"
    if hardware.memory_budget_bytes is not None and training > hardware.memory_budget_bytes:
        status = "DOES_NOT_FIT"
    return MemoryPreflight(
        status,
        parameters,
        gradients,
        moments,
        ssm,
        convolution,
        local,
        training,
        inference,
        hardware.memory_budget_bytes,
    )
