"""Raw-byte objective, identity-based AdamW groups and update-indexed LR schedule."""

import math

import torch
import torch.nn.functional as F

from unified_edge.training.config import TrainingConfig


def raw_byte_nll(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, int]:
    if targets.dtype != torch.long or targets.ndim != 2 or targets.numel() == 0:
        raise ValueError("raw-byte targets must be nonempty int64 [B,T]")
    if ((targets < 0) | (targets > 255)).any():
        raise ValueError("raw-byte objective rejects controls and padding")
    if logits.shape != (*targets.shape, 267):
        raise ValueError("logits must match targets with 267 classes")
    loss_sum = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="sum")
    if not torch.isfinite(loss_sum):
        raise ValueError("non-finite raw-byte loss")
    return loss_sum, targets.numel()


def build_optimizer(model, config: TrainingConfig):
    decay, no_decay, seen = [], [], set()
    for parameter in model.parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        target = (
            no_decay
            if getattr(parameter, "_no_weight_decay", False) or parameter.ndim < 2
            else decay
        )
        target.append(parameter)
    groups = [
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.epsilon,
        foreach=False,
        fused=False,
    )


def finite_parameters(model, context: str):
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            raise ValueError(f"{context}: non-finite parameter {name}")


def finite_gradients(model, context: str):
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise ValueError(f"{context}: non-finite gradient {name}")


class WarmupCosine:
    def __init__(self, optimizer, config: TrainingConfig):
        self.optimizer, self.config = optimizer, config
        self.completed = 0
        self._set_lr()

    def lr_at(self, index: int) -> float:
        c = self.config
        if index < c.warmup_steps:
            factor = (index + 1) / c.warmup_steps
        else:
            progress = (index - c.warmup_steps) / max(1, c.total_steps - c.warmup_steps - 1)
            factor = (
                c.minimum_lr_ratio
                + (1 - c.minimum_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2
            )
        return c.learning_rate * factor

    def _set_lr(self):
        value = self.lr_at(min(self.completed, self.config.total_steps - 1))
        for group in self.optimizer.param_groups:
            group["lr"] = value

    def advance(self):
        if self.completed >= self.config.total_steps:
            raise ValueError("scheduler exhausted")
        self.completed += 1
        self._set_lr()

    def state_dict(self):
        return {
            "schema": "1",
            "type": "warmup_cosine",
            "completed": self.completed,
            "training_sha256": self.config.sha256,
        }

    def load_state_dict(self, state):
        if not isinstance(state, dict) or set(state) != {
            "schema",
            "type",
            "completed",
            "training_sha256",
        }:
            raise ValueError("invalid scheduler state fields")
        if (
            state["schema"] != "1"
            or state["type"] != "warmup_cosine"
            or state["training_sha256"] != self.config.sha256
        ):
            raise ValueError("scheduler compatibility mismatch")
        count = state["completed"]
        if type(count) is not int or not 0 <= count <= self.config.total_steps:
            raise ValueError("invalid scheduler update count")
        expected_lr = self.lr_at(min(count, self.config.total_steps - 1))
        if any(group["lr"] != expected_lr for group in self.optimizer.param_groups):
            raise ValueError("optimizer LR disagrees with restored scheduler")
        self.completed = count
