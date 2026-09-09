"""Byte hierarchy with an explicit stop at every future shared-trunk boundary."""

from dataclasses import dataclass

import torch
from torch import nn

from unified_edge.byte_layers import ByteEmbedding, CompletedPatchEncoder, LocalDecoder
from unified_edge.config import NUM_SYMBOLS
from unified_edge.resolve import ResolvedConfig
from unified_edge.symbols import BOS_ID, mask_non_emittable, validate_symbols


@dataclass(frozen=True)
class HierarchyState:
    pending: torch.Tensor
    hidden: torch.Tensor | None
    completed_patches: int = 0

    @property
    def awaiting_context(self) -> bool:
        return self.hidden is None


@dataclass(frozen=True)
class CompletedPatch:
    index: int
    representation: torch.Tensor


class ByteHierarchy(nn.Module):
    def __init__(self, config: ResolvedConfig):
        super().__init__()
        if not isinstance(config, ResolvedConfig):
            raise TypeError("ByteHierarchy requires a validated ResolvedConfig")
        self.config = config
        shape = config.shape
        self.embedding = ByteEmbedding(shape)
        self.encoder = CompletedPatchEncoder(shape)
        self.decoder = LocalDecoder(shape)
        self.output = nn.Linear(shape.decoder_dim, NUM_SYMBOLS)
        self.bootstrap = nn.Linear(shape.byte_dim, shape.d_model)

    def _validate_context(self, context: torch.Tensor, batch: int) -> None:
        if type(batch) is not int or batch < 1:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(context, torch.Tensor) or context.shape != (
            batch,
            self.config.shape.d_model,
        ):
            raise ValueError("conditioning must have shape [B,d_model]")
        if context.device != self.output.weight.device or context.dtype != self.output.weight.dtype:
            raise ValueError("conditioning dtype/device must match the hierarchy")
        if not torch.isfinite(context).all():
            raise ValueError("conditioning contains non-finite values")

    def _validate_state(self, state: HierarchyState) -> None:
        if not isinstance(state, HierarchyState):
            raise TypeError("expected HierarchyState")
        pending = state.pending
        validate_symbols(pending, allow_padding=False, allow_bos=False)
        if (
            pending.ndim != 2
            or pending.shape[0] < 1
            or pending.shape[1] >= self.config.shape.patch_size
        ):
            raise ValueError("pending symbols must be [B,0..7] with positive batch")
        if pending.device != self.output.weight.device:
            raise ValueError("state device must match the hierarchy")
        if type(state.completed_patches) is not int or state.completed_patches < 0:
            raise ValueError("completed_patches must be a nonnegative integer")
        if state.hidden is None:
            if pending.shape[1] != 0 or state.completed_patches == 0:
                raise ValueError(
                    "awaiting state requires an emitted patch and an empty pending prefix"
                )
        else:
            expected = (pending.shape[0], self.config.shape.decoder_dim)
            if not isinstance(state.hidden, torch.Tensor) or state.hidden.shape != expected:
                raise ValueError("hidden state must be [B,decoder_dim]")
            if (
                state.hidden.device != self.output.weight.device
                or state.hidden.dtype != self.output.weight.dtype
            ):
                raise ValueError("hidden dtype/device must match the hierarchy")
            if not torch.isfinite(state.hidden).all():
                raise ValueError("hidden state contains non-finite values")

    def initial_conditioning(self, batch_size: int) -> torch.Tensor:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        bos = torch.full((batch_size,), BOS_ID, device=self.output.weight.device, dtype=torch.long)
        return self.bootstrap(self.embedding.symbols(bos))

    def start(
        self, batch_size: int = 1, conditioning: torch.Tensor | None = None
    ) -> HierarchyState:
        bos_context = (
            self.initial_conditioning(batch_size) if conditioning is None else conditioning
        )
        self._validate_context(bos_context, batch_size)
        pending = torch.empty(batch_size, 0, dtype=torch.long, device=self.output.weight.device)
        return HierarchyState(pending, self.decoder.initial_hidden(bos_context))

    def condition(self, context: torch.Tensor, state: HierarchyState) -> HierarchyState:
        self._validate_state(state)
        if not state.awaiting_context:
            raise ValueError("new trunk conditioning is allowed only after a completed patch event")
        self._validate_context(context, state.pending.shape[0])
        return HierarchyState(
            state.pending, self.decoder.initial_hidden(context), state.completed_patches
        )

    def predict(self, state: HierarchyState) -> torch.Tensor:
        self._validate_state(state)
        if state.awaiting_context:
            raise ValueError("completed patch awaits external trunk conditioning before prediction")
        return mask_non_emittable(self.output(self.decoder.norm(state.hidden)))

    def encode_completed(self, symbols: torch.Tensor) -> torch.Tensor:
        validate_symbols(symbols, allow_padding=False, allow_bos=False)
        if symbols.ndim != 3 or symbols.shape[2] != self.config.shape.patch_size:
            raise ValueError("completed symbols must have shape [B,N,8]")
        if symbols.device != self.output.weight.device:
            raise ValueError("symbols device must match the hierarchy")
        positions = torch.arange(self.config.shape.patch_size, device=symbols.device)
        return self.encoder(self.embedding(symbols, positions))

    def consume(
        self, symbol: torch.Tensor, state: HierarchyState
    ) -> tuple[HierarchyState, CompletedPatch | None]:
        self._validate_state(state)
        if state.awaiting_context:
            raise ValueError(
                "completed patch awaits external trunk conditioning before consumption"
            )
        validate_symbols(symbol, allow_padding=False, allow_bos=False)
        if symbol.shape != (state.pending.shape[0],) or symbol.device != state.pending.device:
            raise ValueError("consume requires one symbol per batch row on the state device")
        position = state.pending.shape[1]
        pending = torch.cat((state.pending, symbol[:, None]), dim=1)
        if pending.shape[1] == self.config.shape.patch_size:
            encoded = self.encode_completed(pending[:, None])[:, 0]
            event = CompletedPatch(state.completed_patches, encoded)
            empty = pending[:, :0].clone()
            return HierarchyState(empty, None, state.completed_patches + 1), event
        hidden = self.decoder.step(self.embedding(symbol, position), state.hidden)
        return HierarchyState(pending, hidden, state.completed_patches), None

    def forward(
        self, targets: torch.Tensor, conditioning: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Teacher-forced predictions within one patch, never across a missing global trunk."""
        validate_symbols(targets, allow_padding=False, allow_bos=False)
        if (
            targets.ndim != 2
            or targets.shape[0] < 1
            or targets.shape[1] > self.config.shape.patch_size
        ):
            raise ValueError("local teacher forcing expects [B,T<=8]")
        if targets.device != self.output.weight.device:
            raise ValueError("targets device must match the hierarchy")
        context = (
            self.initial_conditioning(targets.shape[0]) if conditioning is None else conditioning
        )
        self._validate_context(context, targets.shape[0])
        positions = torch.arange(targets.shape[1], device=targets.device)
        features = self.decoder(context, self.embedding(targets, positions))
        return mask_non_emittable(self.output(features))

    def export_state(self, state: HierarchyState) -> dict:
        self._validate_state(state)
        return {
            "state_schema": "1",
            "resolved_sha256": self.config.sha256,
            "pending": state.pending.detach().clone(),
            "hidden": None if state.hidden is None else state.hidden.detach().clone(),
            "completed_patches": state.completed_patches,
        }

    def restore_state(self, snapshot: dict) -> HierarchyState:
        expected = {"state_schema", "resolved_sha256", "pending", "hidden", "completed_patches"}
        if not isinstance(snapshot, dict) or set(snapshot) != expected:
            raise ValueError("hierarchy state snapshot has missing or unknown fields")
        if snapshot["state_schema"] != "1" or snapshot["resolved_sha256"] != self.config.sha256:
            raise ValueError("hierarchy state schema or resolved-config identity mismatch")
        state = HierarchyState(
            snapshot["pending"], snapshot["hidden"], snapshot["completed_patches"]
        )
        self._validate_state(state)
        return HierarchyState(
            state.pending.clone(),
            None if state.hidden is None else state.hidden.clone(),
            state.completed_patches,
        )
