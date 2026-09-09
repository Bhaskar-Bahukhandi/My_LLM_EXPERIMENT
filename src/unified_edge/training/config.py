"""Strict CPU training configuration and canonical identities."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass


def canonical_hash(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class TrainingConfig:
    schema: str = "1"
    seed: int = 17
    sequence_length: int = 32
    batch_size: int = 2
    accumulation_steps: int = 2
    total_steps: int = 60
    learning_rate: float = 0.003
    warmup_steps: int = 5
    minimum_lr_ratio: float = 0.1
    weight_decay: float = 0.01
    clip_norm: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    cpu_threads: int = 1

    def __post_init__(self):
        if self.schema != "1":
            raise ValueError("unsupported training schema")
        for name in (
            "seed",
            "sequence_length",
            "batch_size",
            "accumulation_steps",
            "total_steps",
            "warmup_steps",
            "cpu_threads",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name in ("seed", "warmup_steps") else 1):
                raise ValueError(f"invalid integer training field {name}")
        if not 8 <= self.sequence_length <= 256:
            raise ValueError("bounded CPU sequence_length must be 8..256")
        if self.warmup_steps >= self.total_steps:
            raise ValueError("warmup_steps must be less than total_steps")
        for name in (
            "learning_rate",
            "minimum_lr_ratio",
            "weight_decay",
            "clip_norm",
            "beta1",
            "beta2",
            "epsilon",
        ):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"invalid finite training field {name}")
        if min(self.learning_rate, self.clip_norm, self.epsilon) <= 0 or self.weight_decay < 0:
            raise ValueError("LR, clipping and epsilon must be positive; decay nonnegative")
        if (
            not 0 <= self.minimum_lr_ratio <= 1
            or not 0 <= self.beta1 < 1
            or not 0 <= self.beta2 < 1
        ):
            raise ValueError("invalid LR ratio or AdamW betas")

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return canonical_hash(self.to_dict())

    @classmethod
    def from_dict(cls, value: dict):
        if not isinstance(value, dict) or set(value) - set(cls.__dataclass_fields__):
            raise ValueError("training config has unknown fields or is not an object")
        return cls(**value)
