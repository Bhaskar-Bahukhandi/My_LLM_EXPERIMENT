"""Dense byte/Mamba ownership and causal lifecycle, without training infrastructure."""

from dataclasses import dataclass

import torch
from torch import nn

from unified_edge.byte_hierarchy import ByteHierarchy, HierarchyState
from unified_edge.config import NUM_SYMBOLS
from unified_edge.mamba import SharedMambaTrunk, TrunkState
from unified_edge.resolve import ResolvedConfig
from unified_edge.symbols import validate_symbols


@dataclass(frozen=True)
class DenseState:
    hierarchy: HierarchyState
    shared: TrunkState


class DenseByteModel(nn.Module):
    def __init__(self, config: ResolvedConfig):
        super().__init__()
        if not isinstance(config, ResolvedConfig):
            raise TypeError("DenseByteModel requires a validated ResolvedConfig")
        self.config = config
        self.hierarchy = ByteHierarchy(config)
        self.shared = SharedMambaTrunk(config)

    def start(self, batch: int = 1) -> DenseState:
        shared = self.shared.initialize_state(batch)
        context, shared = self.shared.step(self.hierarchy.initial_conditioning(batch), shared)
        return DenseState(self.hierarchy.start(batch, conditioning=context), shared)

    def validate_state(self, state: DenseState) -> None:
        if not isinstance(state, DenseState):
            raise TypeError("expected DenseState")
        self.hierarchy._validate_state(state.hierarchy)
        self.shared.validate_state(state.shared, state.hierarchy.pending.shape[0])
        if state.hierarchy.awaiting_context:
            raise ValueError("integrated state cannot contain an unprocessed completed patch")
        if state.shared.steps != state.hierarchy.completed_patches + 1:
            raise ValueError("shared clock must equal one BOS plus completed payload patches")

    def reset(self, state: DenseState) -> DenseState:
        self.validate_state(state)
        return self.start(state.hierarchy.pending.shape[0])

    def predict(self, state: DenseState) -> torch.Tensor:
        self.validate_state(state)
        return self.hierarchy.predict(state.hierarchy)

    def consume(self, symbol: torch.Tensor, state: DenseState) -> DenseState:
        self.validate_state(state)
        local, event = self.hierarchy.consume(symbol, state.hierarchy)
        shared = state.shared
        if event is not None:
            if event.index + 1 != shared.steps:
                raise ValueError("completed patch index disagrees with the shared clock")
            context, shared = self.shared.step(event.representation, shared)
            local = self.hierarchy.condition(context, local)
        result = DenseState(local, shared)
        self.validate_state(result)
        return result

    def forward(self, targets: torch.Tensor) -> torch.Tensor:
        """Bounded differentiable SSD forward: context j sees BOS and patches strictly before j."""
        validate_symbols(targets, allow_padding=False, allow_bos=False)
        if targets.ndim != 2 or targets.shape[0] < 1:
            raise ValueError("dense targets must have shape [B,T] with positive batch")
        if targets.device != self.hierarchy.output.weight.device:
            raise ValueError("targets device must match model")
        batch, length = targets.shape
        if length == 0:
            self.shared.initialize_state(batch)
            return self.hierarchy.output.weight.new_empty(batch, 0, NUM_SYMBOLS)
        patch = self.config.shape.patch_size
        previous_patches = (length - 1) // patch
        complete = targets[:, : previous_patches * patch].reshape(batch, previous_patches, patch)
        encoded = self.hierarchy.encode_completed(complete)
        bos = self.hierarchy.initial_conditioning(batch).unsqueeze(1)
        contexts, _ = self.shared(torch.cat((bos, encoded), dim=1))
        logits = [
            self.hierarchy(
                targets[:, start : start + patch], conditioning=contexts[:, start // patch]
            )
            for start in range(0, length, patch)
        ]
        return torch.cat(logits, dim=1)

    def export_state(self, state: DenseState) -> dict:
        self.validate_state(state)
        return {
            "schema": "1",
            "resolved_sha256": self.config.sha256,
            "hierarchy": self.hierarchy.export_state(state.hierarchy),
            "shared": self.shared.export_state(state.shared),
        }

    def restore_state(self, snapshot: dict) -> DenseState:
        if not isinstance(snapshot, dict) or set(snapshot) != {
            "schema",
            "resolved_sha256",
            "hierarchy",
            "shared",
        }:
            raise ValueError("dense snapshot has missing or unknown fields")
        if snapshot["schema"] != "1" or snapshot["resolved_sha256"] != self.config.sha256:
            raise ValueError("dense snapshot schema or config identity mismatch")
        state = DenseState(
            self.hierarchy.restore_state(snapshot["hierarchy"]),
            self.shared.restore_state(snapshot["shared"]),
        )
        self.validate_state(state)
        return state
