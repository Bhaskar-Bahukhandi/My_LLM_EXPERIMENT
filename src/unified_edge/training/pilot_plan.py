"""Immutable R9 contracts. Parsing and planning never authorize execution."""

import math
from dataclasses import asdict, dataclass, fields
from decimal import Decimal
from enum import Enum

from unified_edge.training.config import TrainingConfig, canonical_hash

MODEL_SHA = "6e7ffdc3c3448f7f19fa4466a9780c369c05cc572030c1b026d4eef1503b9359"
DOMAINS = ("general_text", "code", "documentation", "structured_math")
PROMPTS = (
    b"",
    b"The ",
    b"Once ",
    b"def ",
    b"import ",
    b"class ",
    b"x = ",
    b"Let x = ",
    b"Parameters\n----------\n",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    require(
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
        "invalid SHA-256",
    )


def number(value):
    require(type(value) in (int, float) and math.isfinite(value), "nonfinite number")
    return Decimal(str(value))


class Contract:
    def to_dict(self):
        # JSON canonical form (including tuple -> array) shared by all identities.
        import json

        return json.loads(json.dumps(asdict(self), allow_nan=False))

    @property
    def sha256(self):
        return canonical_hash(self.to_dict())

    @classmethod
    def from_dict(cls, value):
        require(
            isinstance(value, dict) and set(value) == {f.name for f in fields(cls)},
            f"{cls.__name__}: missing/unknown fields",
        )
        defaults = cls()
        parsed = {}
        for field in fields(cls):
            item, default = value[field.name], getattr(defaults, field.name)
            if isinstance(default, Contract):
                item = type(default).from_dict(item)
            elif isinstance(default, tuple):
                require(isinstance(item, list), "expected JSON array")
                item = tuple(item)
            parsed[field.name] = item
        return cls(**parsed)

    def fixed(self):
        def equal_type(left, right):
            return (
                type(left) is type(right)
                and left == right
                and (
                    not isinstance(left, tuple)
                    or all(equal_type(a, b) for a, b in zip(left, right, strict=True))
                )
            )

        for field in fields(self):
            value, expected = getattr(self, field.name), field.default
            require(
                equal_type(value, expected),
                f"{type(self).__name__}.{field.name}: reviewed contract is fixed",
            )


@dataclass(frozen=True)
class LRProbePlan(Contract):
    schema: str = "edge-lr-probe-1"
    rates: tuple = (0.0003, 0.0006, 0.001)
    seed: int = 17
    updates: int = 256
    warmup: int = 32
    scheduler: str = "warmup_constant_v1"
    gpu_seconds: int = 10800
    nominal_bytes_per_arm: int = 2097152
    minimum_improvement: float = 0.1
    tie_distance: float = 0.01
    max_clipped_fraction: float = 0.5

    def __post_init__(self):
        self.fixed()


@dataclass(frozen=True)
class EvaluationPlan(Contract):
    schema: str = "edge-evaluation-1"
    lr_selection_bytes: int = 262144
    monitor_bytes: int = 262144
    confirmation_bytes: int = 3571712
    checkpoint_every: int = 250
    monitor_every: int = 250
    confirmation_every: int = 1000
    generation_every: int = 500
    context_every: int = 1000

    def __post_init__(self):
        self.fixed()
        require(
            self.lr_selection_bytes + self.monitor_bytes + self.confirmation_bytes == 4096000,
            "validation partition arithmetic",
        )

    def events(self, step, endpoint=8000, *, stopping=False):
        require(type(step) is int and 0 <= step <= endpoint, "invalid cadence step")
        events = []
        for name, period, zero, end in (
            ("monitor", self.monitor_every, True, False),
            ("confirmation", self.confirmation_every, False, True),
            ("generation", self.generation_every, True, True),
            ("context", self.context_every, True, True),
            ("checkpoint", self.checkpoint_every, True, True),
        ):
            if (
                (step == 0 and zero)
                or (step > 0 and step % period == 0)
                or (end and step == endpoint)
                or (name == "checkpoint" and stopping)
            ):
                events.append(name)
        return tuple(events)


@dataclass(frozen=True)
class GenerationDiagnosticPlan(Contract):
    schema: str = "edge-generation-1"
    prompts_hex: tuple = tuple(p.hex() for p in PROMPTS)
    seeds: tuple = (101, 202, 303, 404)
    length: int = 256
    temperature: float = 1.0
    policy: str = "greedy_and_unmodified_byte_sampling"

    def __post_init__(self):
        self.fixed()


@dataclass(frozen=True)
class ContextDiagnosticPlan(Contract):
    schema: str = "edge-context-1"
    anchors: int = 128
    anchors_per_domain: int = 32
    documents_per_domain: int = 16
    short_history: int = 32
    long_history: int = 128
    phases: tuple = tuple(range(8))
    bootstrap_resamples: int = 2000
    bootstrap_seed: int = 43

    def __post_init__(self):
        self.fixed()


@dataclass(frozen=True)
class PromotionGatePlan(Contract):
    schema: str = "edge-promotion-1"
    endpoint_only: bool = True
    nll_improvement: float = 0.1
    domain_improvement: float = 0.02
    control_mass: float = 0.01
    median_whitespace: float = 0.65
    maximum_whitespace: float = 0.9
    minimum_entropy_bits: float = 2.0
    maximum_run: int = 32
    median_late_space: float = 0.35
    sustained_space: float = 0.8
    printable: float = 0.9
    utf8_samples: int = 33
    diverse_prompts: int = 7
    distinct_samples: int = 3
    greedy_suffixes: int = 7
    sustained_margin: float = 0.95
    sustained_positions: int = 32
    jsd_nats: float = 0.005
    history_delta: float = 0.01
    shuffle_advantage: float = 0.005
    anchor_tv: float = 0.00001
    responsive_fraction: float = 0.75
    maximum_domain_degradation: float = 0.02
    bridge_improvement: float = 0.01
    replication_seed: int = 29

    def __post_init__(self):
        self.fixed()


@dataclass(frozen=True)
class PilotPlan(Contract):
    schema: str = "edge-pilot-1"
    seed: int = 17
    sequence_length: int = 64
    patch_size: int = 8
    batch_size: int = 8
    accumulation_steps: int = 16
    updates: int = 8000
    warmup: int = 400
    scheduler: str = "warmup_cosine"
    minimum_lr_ratio: float = 0.1
    optimizer: str = "AdamW"
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    weight_decay: float = 0.01
    clip_norm: float = 1.0
    precision: str = "FP32"
    tf32: bool = False
    monitoring: str = "PRODUCTION_FAST"
    gpu_seconds: int = 86400
    train_bytes: int = 65536000
    validation_bytes: int = 4096000
    future_sealed_test_reserve_bytes: int = 4096000
    train_quotas: tuple = (32768000, 16384000, 9830400, 6553600)
    validation_quotas: tuple = (2048000, 1024000, 614400, 409600)
    lr_probe: LRProbePlan = LRProbePlan()
    evaluation: EvaluationPlan = EvaluationPlan()
    generation: GenerationDiagnosticPlan = GenerationDiagnosticPlan()
    context: ContextDiagnosticPlan = ContextDiagnosticPlan()
    promotion: PromotionGatePlan = PromotionGatePlan()

    def __post_init__(self):
        self.fixed()

    @property
    def nominal_bytes_per_update(self):
        return self.sequence_length * self.batch_size * self.accumulation_steps

    def training_config(self, lr, device, *, probe=False):
        require(lr in self.lr_probe.rates, "LR must come from declared probe candidates")
        return TrainingConfig(
            schema="2",
            seed=self.seed,
            sequence_length=self.sequence_length,
            batch_size=self.batch_size,
            accumulation_steps=self.accumulation_steps,
            total_steps=self.lr_probe.updates if probe else self.updates,
            learning_rate=lr,
            warmup_steps=self.lr_probe.warmup if probe else self.warmup,
            minimum_lr_ratio=self.minimum_lr_ratio,
            device=device,
            beta1=self.beta1,
            beta2=self.beta2,
            epsilon=self.epsilon,
            weight_decay=self.weight_decay,
            clip_norm=self.clip_norm,
        )


class Failure(str, Enum):
    PLAN_INVALID = "PLAN_INVALID"
    PREREQUISITE_MISSING = "PREREQUISITE_MISSING"
    DATA_INTEGRITY_FAILURE = "DATA_INTEGRITY_FAILURE"
    LR_PROBE_FAILED = "LR_PROBE_FAILED"
    DEVICE_GATE_FAILED = "DEVICE_GATE_FAILED"
    NUMERICAL_FAILURE = "NUMERICAL_FAILURE"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    CHECKPOINT_FAILURE = "CHECKPOINT_FAILURE"
    DIAGNOSTIC_FAILURE = "DIAGNOSTIC_FAILURE"
    PROMOTION_GATE_FAILED = "PROMOTION_GATE_FAILED"
    USER_STOP = "USER_STOP"
