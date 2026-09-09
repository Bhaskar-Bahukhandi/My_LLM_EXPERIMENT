"""Executable local layers; all global recurrent processing belongs outside this module."""

import torch
import torch.nn.functional as F
from torch import nn

from unified_edge.config import NUM_SYMBOLS, ModelShape, validate_architecture
from unified_edge.symbols import PAD_ID, validate_symbols


class ByteEmbedding(nn.Module):
    def __init__(self, shape: ModelShape):
        super().__init__()
        validate_architecture(shape)
        self.patch_size = shape.patch_size
        self.symbols = nn.Embedding(NUM_SYMBOLS, shape.byte_dim, padding_idx=PAD_ID)
        self.positions = nn.Embedding(shape.patch_size, shape.byte_dim)

    def forward(self, ids: torch.Tensor, positions: torch.Tensor | int) -> torch.Tensor:
        validate_symbols(ids)
        if type(positions) is int:
            positions = torch.full_like(ids, positions)
        if not isinstance(positions, torch.Tensor) or positions.dtype != torch.long:
            raise TypeError("positions must be int64 tensors or an integer")
        if positions.device != ids.device:
            raise ValueError("positions and symbols must be on the same device")
        if positions.shape != ids.shape and (ids.ndim == 0 or positions.shape != (ids.shape[-1],)):
            raise ValueError("positions must match symbols or their final sequence axis")
        if ((positions < 0) | (positions >= self.patch_size)).any():
            raise ValueError(f"positions must be in 0..{self.patch_size - 1}")
        return self.symbols(ids) + self.positions(positions)


class CompletedPatchEncoder(nn.Module):
    def __init__(self, shape: ModelShape):
        super().__init__()
        validate_architecture(shape)
        self.patch_size = shape.patch_size
        self.byte_dim = shape.byte_dim
        self.d_model = shape.d_model
        self.conv = nn.Conv1d(shape.byte_dim, shape.byte_dim, 3, groups=shape.byte_dim)
        self.gate = nn.Linear(shape.byte_dim, 2 * shape.byte_dim)
        self.norm = nn.RMSNorm(shape.byte_dim, eps=1e-5)
        self.projection = nn.Linear(shape.byte_dim, shape.d_model)

    def forward(self, embedded: torch.Tensor) -> torch.Tensor:
        if embedded.ndim != 4 or embedded.shape[2:] != (self.patch_size, self.byte_dim):
            raise ValueError("encoder expects completed embedded patches [B,N,8,byte_dim]")
        batch, patches = embedded.shape[:2]
        if batch < 1:
            raise ValueError("encoder batch must be positive")
        if patches == 0:
            return embedded.new_empty(batch, 0, self.d_model)
        x = embedded.reshape(batch * patches, self.patch_size, self.byte_dim).transpose(1, 2)
        x = self.conv(F.pad(x, (2, 0))).transpose(1, 2)
        value, gate = self.gate(x).chunk(2, dim=-1)
        mixed = self.norm(value * F.silu(gate))
        # A single kernel-3 endpoint cannot summarize the first five symbols of an 8-symbol patch.
        return self.projection(mixed.mean(dim=1)).reshape(batch, patches, self.d_model)


class LocalDecoder(nn.Module):
    def __init__(self, shape: ModelShape):
        super().__init__()
        validate_architecture(shape)
        self.d_model = shape.d_model
        self.byte_dim = shape.byte_dim
        self.decoder_dim = shape.decoder_dim
        self.patch_size = shape.patch_size
        self.context = nn.Linear(shape.d_model, shape.decoder_dim)
        self.gru = nn.GRUCell(shape.byte_dim, shape.decoder_dim)
        self.norm = nn.RMSNorm(shape.decoder_dim, eps=1e-5)

    def initial_hidden(self, context: torch.Tensor) -> torch.Tensor:
        if context.ndim != 2 or context.shape[-1] != self.d_model or context.shape[0] < 1:
            raise ValueError("decoder conditioning must be [B,d_model] with positive batch")
        return torch.tanh(self.context(context))

    def step(self, embedded: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        if embedded.ndim != 2 or embedded.shape[1] != self.byte_dim:
            raise ValueError("decoder symbol embedding must be [B,byte_dim]")
        if hidden.shape != (embedded.shape[0], self.decoder_dim):
            raise ValueError("decoder hidden state must be [B,decoder_dim]")
        return self.gru(embedded, hidden)

    def forward(self, context: torch.Tensor, targets_embedded: torch.Tensor) -> torch.Tensor:
        """Predict T targets using only their prefixes; target j is read after prediction j."""
        if (
            targets_embedded.ndim != 3
            or targets_embedded.shape[0] != context.shape[0]
            or targets_embedded.shape[-1] != self.byte_dim
            or targets_embedded.shape[1] > self.patch_size
        ):
            raise ValueError("local decoder targets must be [B,T<=8,byte_dim]")
        hidden = self.initial_hidden(context)
        length = targets_embedded.shape[1]
        if length == 0:
            return context.new_empty(context.shape[0], 0, self.decoder_dim)
        predictions = [self.norm(hidden)]
        for index in range(length - 1):
            hidden = self.step(targets_embedded[:, index], hidden)
            predictions.append(self.norm(hidden))
        return torch.stack(predictions, dim=1)
