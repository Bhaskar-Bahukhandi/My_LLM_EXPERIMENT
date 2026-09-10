"""Generic CPU/CUDA FP32 Mamba-2: explicit recurrence and independent dense SSD execution."""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from unified_edge.config import ModelShape, validate_architecture
from unified_edge.resolve import ResolvedConfig


@dataclass(frozen=True)
class LayerState:
    conv: torch.Tensor
    ssm: torch.Tensor


@dataclass(frozen=True)
class TrunkState:
    layers: tuple[LayerState, ...]
    steps: int = 0


def _finite(value: torch.Tensor, label: str) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"{label} contains non-finite values")


class Mamba2Block(nn.Module):
    """Pre-normalized residual block for the accepted full-width, single-group subset."""

    def __init__(self, shape: ModelShape):
        super().__init__()
        validate_architecture(shape)
        self.shape = shape
        i, n, h = shape.d_inner, shape.d_state, shape.nheads
        self.pre_norm = nn.RMSNorm(shape.d_model, eps=1e-5)
        self.in_proj = nn.Linear(shape.d_model, 2 * i + 2 * n + h, bias=False)
        self.conv = nn.Conv1d(i + 2 * n, i + 2 * n, shape.d_conv, groups=i + 2 * n)
        dt = torch.exp(torch.rand(h) * math.log(0.1 / 0.001) + math.log(0.001))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.empty(h).uniform_(1, 16).log())
        self.D = nn.Parameter(torch.ones(h))
        for parameter in (self.dt_bias, self.A_log, self.D):
            parameter._no_weight_decay = True
        self.gated_norm = nn.RMSNorm(i, eps=1e-5)
        self.out_proj = nn.Linear(i, shape.d_model, bias=False)

    def _validate_profile(self) -> None:
        device = self.in_proj.weight.device
        if device.type not in ("cpu", "cuda") or any(
            p.dtype != torch.float32 or p.device != device for p in self.parameters()
        ):
            raise ValueError("generic Mamba profile requires CPU FP32 or CUDA FP32 on one device")

    def initialize_state(self, batch: int) -> LayerState:
        self._validate_profile()
        if type(batch) is not int or batch < 1:
            raise ValueError("batch must be a positive integer")
        s = self.shape
        return LayerState(
            self.in_proj.weight.new_zeros(batch, s.d_inner + 2 * s.d_state, s.d_conv),
            self.in_proj.weight.new_zeros(batch, s.nheads, s.headdim, s.d_state),
        )

    def validate_state(self, state: LayerState, batch: int) -> None:
        self._validate_profile()
        if type(batch) is not int or batch < 1:
            raise ValueError("batch must be a positive integer")
        if not isinstance(state, LayerState):
            raise TypeError("expected LayerState")
        s = self.shape
        for name, value, shape in (
            ("conv", state.conv, (batch, s.d_inner + 2 * s.d_state, s.d_conv)),
            ("ssm", state.ssm, (batch, s.nheads, s.headdim, s.d_state)),
        ):
            if not isinstance(value, torch.Tensor) or value.shape != shape:
                raise ValueError(f"{name} state must have shape {shape}")
            if value.dtype != torch.float32 or value.device != self.in_proj.weight.device:
                raise ValueError(f"{name} state must be FP32 on the parameter device")
            _finite(value, name + " state")

    def _validate_input(self, inputs: torch.Tensor, ndim: int) -> None:
        if (
            not isinstance(inputs, torch.Tensor)
            or inputs.ndim != ndim
            or inputs.shape[0] < 1
            or inputs.shape[-1] != self.shape.d_model
        ):
            raise ValueError(
                f"Mamba input must have {ndim} axes, positive batch, last axis d_model"
            )
        if inputs.dtype != torch.float32 or inputs.device != self.in_proj.weight.device:
            raise ValueError("Mamba input must be FP32 on the parameter device")
        _finite(inputs, "Mamba input")

    def _project(self, inputs: torch.Tensor):
        s = self.shape
        z, xbc, dt_raw = self.in_proj(self.pre_norm(inputs)).split(
            (s.d_inner, s.d_inner + 2 * s.d_state, s.nheads), dim=-1
        )
        dt = F.softplus(dt_raw + self.dt_bias)
        rate = self.A_log.exp()
        _finite(dt, "Mamba dt")
        _finite(rate, "Mamba exp(A_log)")
        if (dt <= 0).any() or (rate <= 0).any():
            raise ValueError("Mamba dt and exp(A_log) must be strictly positive")
        log_decay = dt * -rate
        _finite(log_decay, "Mamba log decay")
        return z, xbc, dt, log_decay

    def _output(self, inputs: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        gated = y * F.silu(z)
        _finite(gated, "Mamba gated readout")
        output = inputs + self.out_proj(self.gated_norm(gated))
        _finite(output, "Mamba residual output")
        return output

    def step(self, inputs: torch.Tensor, state: LayerState) -> tuple[torch.Tensor, LayerState]:
        self._validate_input(inputs, 2)
        self.validate_state(state, inputs.shape[0])
        s = self.shape
        z, xbc, dt, log_decay = self._project(inputs)
        conv = torch.cat((state.conv[:, :, 1:], xbc.unsqueeze(-1)), dim=-1)
        mixed = F.silu((conv * self.conv.weight[:, 0]).sum(-1) + self.conv.bias)
        _finite(mixed, "Mamba convolution")
        x, b, c = mixed.split((s.d_inner, s.d_state, s.d_state), dim=-1)
        x = x.reshape(inputs.shape[0], s.nheads, s.headdim)
        # S[B,H,P,N]: per-head scalar decay, with B/C shared across all heads.
        decay = torch.exp(log_decay)
        ssm = state.ssm * decay[:, :, None, None]
        ssm = ssm + torch.einsum("bh,bhp,bn->bhpn", dt, x, b)
        _finite(ssm, "Mamba updated SSM")
        y = torch.einsum("bhpn,bn->bhp", ssm, c) + self.D[None, :, None] * x
        return self._output(inputs, y.flatten(-2), z), LayerState(conv, ssm)

    def forward(
        self, inputs: torch.Tensor, state: LayerState | None = None
    ) -> tuple[torch.Tensor, LayerState]:
        """Independent O(T^2) SSD reference; supports a nonzero incoming recurrent state."""
        self._validate_input(inputs, 3)
        batch, length, _ = inputs.shape
        state = self.initialize_state(batch) if state is None else state
        self.validate_state(state, batch)
        if length == 0:
            return inputs.clone(), LayerState(state.conv.clone(), state.ssm.clone())
        s = self.shape
        z, xbc, dt, log_decay = self._project(inputs)
        history = torch.cat((state.conv[:, :, 1:], xbc.transpose(1, 2)), dim=-1)
        mixed = F.silu(F.conv1d(history, self.conv.weight, self.conv.bias, groups=history.shape[1]))
        _finite(mixed, "Mamba sequence convolution")
        x, b, c = mixed.transpose(1, 2).split((s.d_inner, s.d_state, s.d_state), dim=-1)
        x = x.reshape(batch, length, s.nheads, s.headdim)
        log_decay = log_decay.transpose(1, 2)  # [B,H,T]
        causal = torch.ones(length, length, dtype=torch.bool, device=inputs.device).tril()
        lower = causal.tril(-1)
        # Segment[t,s] sums log decays at s+1..t without subtracting prefix sums.
        segment = log_decay.unsqueeze(-1).expand(-1, -1, -1, length).masked_fill(~lower, 0)
        segment = segment.cumsum(dim=-2)
        _finite(segment, "Mamba segment log decay")
        transition = segment.masked_fill(~causal, -torch.inf).exp()
        weighted_x = x * dt.unsqueeze(-1)
        bc = torch.einsum("btn,bsn->bts", c, b)
        y = torch.einsum("bhts,bts,bshp->bthp", transition, bc, weighted_x)
        initial_log_decay = log_decay.cumsum(-1)
        _finite(initial_log_decay, "Mamba initial-state log decay")
        initial_decay = initial_log_decay.exp()
        y = y + torch.einsum("bhpn,btn,bht->bthp", state.ssm, c, initial_decay)
        y = y + x * self.D[None, None, :, None]
        final_ssm = state.ssm * initial_decay[:, :, -1, None, None]
        final_ssm = final_ssm + torch.einsum(
            "bhs,bshp,bsn->bhpn", transition[:, :, -1], weighted_x, b
        )
        final_conv = torch.cat((state.conv, xbc.transpose(1, 2)), dim=-1)[:, :, -s.d_conv :].clone()
        _finite(final_ssm, "Mamba final SSM")
        return self._output(inputs, y.flatten(-2), z), LayerState(final_conv, final_ssm)


class SharedMambaTrunk(nn.Module):
    def __init__(self, config: ResolvedConfig):
        super().__init__()
        if not isinstance(config, ResolvedConfig):
            raise TypeError("SharedMambaTrunk requires a validated ResolvedConfig")
        self.config = config
        self.layers = nn.ModuleList(
            Mamba2Block(config.shape) for _ in range(config.shape.shared_layers)
        )

    def initialize_state(self, batch: int) -> TrunkState:
        return TrunkState(tuple(layer.initialize_state(batch) for layer in self.layers))

    def validate_state(self, state: TrunkState, batch: int | None = None) -> None:
        if not isinstance(state, TrunkState):
            raise TypeError("expected TrunkState")
        if not isinstance(state.layers, tuple) or len(state.layers) != len(self.layers):
            raise ValueError("trunk state must contain exactly one LayerState per layer")
        if type(state.steps) is not int or state.steps < 0:
            raise ValueError("trunk steps must be a nonnegative integer")
        if batch is None:
            first = state.layers[0]
            if (
                not isinstance(first, LayerState)
                or not isinstance(first.conv, torch.Tensor)
                or first.conv.ndim != 3
            ):
                raise ValueError("first layer convolution state must have three axes")
            batch = first.conv.shape[0]
        for index, (layer, value) in enumerate(zip(self.layers, state.layers, strict=True)):
            try:
                layer.validate_state(value, batch)
            except (ValueError, TypeError) as error:
                raise ValueError(f"layer {index}: {error}") from error

    def reset(self, state: TrunkState) -> TrunkState:
        self.validate_state(state)
        return self.initialize_state(state.layers[0].conv.shape[0])

    def step(self, inputs: torch.Tensor, state: TrunkState) -> tuple[torch.Tensor, TrunkState]:
        self.layers[0]._validate_input(inputs, 2)
        self.validate_state(state, inputs.shape[0])
        next_layers = []
        for index, (layer, value) in enumerate(zip(self.layers, state.layers, strict=True)):
            try:
                inputs, next_state = layer.step(inputs, value)
            except ValueError as error:
                raise ValueError(f"layer {index}: {error}") from error
            next_layers.append(next_state)
        return inputs, TrunkState(tuple(next_layers), state.steps + 1)

    def forward(
        self, inputs: torch.Tensor, state: TrunkState | None = None
    ) -> tuple[torch.Tensor, TrunkState]:
        self.layers[0]._validate_input(inputs, 3)
        state = self.initialize_state(inputs.shape[0]) if state is None else state
        self.validate_state(state, inputs.shape[0])
        length = inputs.shape[1]
        next_layers = []
        for index, (layer, value) in enumerate(zip(self.layers, state.layers, strict=True)):
            try:
                inputs, next_state = layer(inputs, value)
            except ValueError as error:
                raise ValueError(f"layer {index}: {error}") from error
            next_layers.append(next_state)
        return inputs, TrunkState(tuple(next_layers), state.steps + length)

    def export_state(self, state: TrunkState) -> dict:
        self.validate_state(state)
        return {
            "schema": "1",
            "resolved_sha256": self.config.sha256,
            "batch": state.layers[0].conv.shape[0],
            "steps": state.steps,
            "layers": [
                {"conv": value.conv.detach().clone(), "ssm": value.ssm.detach().clone()}
                for value in state.layers
            ],
        }

    def restore_state(self, snapshot: dict) -> TrunkState:
        keys = {"schema", "resolved_sha256", "batch", "steps", "layers"}
        if not isinstance(snapshot, dict) or set(snapshot) != keys:
            raise ValueError("trunk snapshot has missing or unknown fields")
        if snapshot["schema"] != "1" or snapshot["resolved_sha256"] != self.config.sha256:
            raise ValueError("trunk snapshot schema or config identity mismatch")
        if type(snapshot["batch"]) is not int or snapshot["batch"] < 1:
            raise ValueError("snapshot batch must be a positive integer")
        if not isinstance(snapshot["layers"], list):
            raise ValueError("snapshot layers must be a list")
        layers = []
        for value in snapshot["layers"]:
            if not isinstance(value, dict) or set(value) != {"conv", "ssm"}:
                raise ValueError("layer snapshot requires exactly conv and ssm")
            layers.append(LayerState(value["conv"], value["ssm"]))
        state = TrunkState(tuple(layers), snapshot["steps"])
        self.validate_state(state, snapshot["batch"])
        return TrunkState(
            tuple(
                LayerState(v.conv.detach().clone(), v.ssm.detach().clone()) for v in state.layers
            ),
            state.steps,
        )
