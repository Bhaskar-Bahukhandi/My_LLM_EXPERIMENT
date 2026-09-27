"""Deterministic candidate selection after structural, backend and memory gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

from unified_edge.config import (
    BACKEND,
    CONTROL_IDS,
    INVENTORY_VERSION,
    NUM_SYMBOLS,
    REFERENCE_COMMIT,
    SCHEMA_VERSION,
    AuthoredConfig,
    ConfigError,
    Hardware,
    ModelRequest,
    ModelShape,
    ResolverPolicy,
    SearchPolicy,
    digest,
    mapping,
    positive_int,
    validate_architecture,
)
from unified_edge.parameters import (
    audit_executable_parameters,
    audit_parameters,
    formula_parameter_estimate,
    instantiate_inventory,
)
from unified_edge.preflight import memory_preflight
from unified_edge.readiness import configuration_readiness


@dataclass(frozen=True)
class ResolvedConfig:
    shape: ModelShape
    hardware: Hardware
    target_parameters: int
    parameter_tolerance: float
    authored_sha256: str
    resolver: ResolverPolicy | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.shape, ModelShape) or not isinstance(self.hardware, Hardware):
            raise ConfigError("resolved config requires typed shape and hardware")
        validate_architecture(self.shape)
        positive_int(self.target_parameters, "resolved.target_parameters")
        # Reuse the authored tolerance contract rather than maintaining a second rule.
        ModelRequest(parameter_tolerance=self.parameter_tolerance)
        if self.resolver is not None:
            if not isinstance(self.resolver, ResolverPolicy) or self.resolver.mode != "explicit":
                raise ConfigError("resolved resolver must be an explicit ResolverPolicy")
        if (
            not isinstance(self.authored_sha256, str)
            or len(self.authored_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.authored_sha256)
        ):
            raise ConfigError("resolved.authored_sha256 must be a lowercase SHA-256 digest")

    def to_dict(self) -> dict:
        result = {
            "schema_version": SCHEMA_VERSION,
            "inventory_version": INVENTORY_VERSION,
            "architecture_family": "edge_mamba_dense_inventory",
            "semantic_reference_commit": REFERENCE_COMMIT,
            "backend_classification": "VALID_GENERIC_PATH",
            "model": self.shape.to_dict(),
            "derived": {
                "d_inner": self.shape.d_inner,
                "nheads": self.shape.nheads,
                "num_symbols": NUM_SYMBOLS,
                "bytes_per_chunk": self.shape.patch_size * self.shape.chunk_patches,
            },
            "control_version": "1",
            "control_ids": dict(CONTROL_IDS),
            "hardware": asdict(self.hardware),
            "target_parameters": self.target_parameters,
            "parameter_tolerance": self.parameter_tolerance,
            "authored_sha256": self.authored_sha256,
        }
        if self.resolver is not None:
            result["resolver"] = asdict(self.resolver)
        return result

    @property
    def sha256(self) -> str:
        return digest(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> ResolvedConfig:
        required = {
            "schema_version",
            "inventory_version",
            "architecture_family",
            "semantic_reference_commit",
            "backend_classification",
            "model",
            "derived",
            "control_version",
            "control_ids",
            "hardware",
            "target_parameters",
            "parameter_tolerance",
            "authored_sha256",
        }
        data = mapping(value, required | {"resolver"}, "resolved")
        if not required.issubset(data):
            raise ConfigError(f"resolved missing fields {sorted(required - set(data))}")
        model_data = mapping(data["model"], {f.name for f in fields(ModelShape)}, "resolved.model")
        if set(model_data) != {f.name for f in fields(ModelShape)}:
            raise ConfigError("resolved.model must specify every dimension")
        hardware_data = mapping(
            data["hardware"], {f.name for f in fields(Hardware)}, "resolved.hardware"
        )
        if set(hardware_data) != {f.name for f in fields(Hardware)}:
            raise ConfigError("resolved.hardware must specify every field")
        result = cls(
            ModelShape(**model_data),
            Hardware(**hardware_data),
            data["target_parameters"],
            data["parameter_tolerance"],
            data["authored_sha256"],
            ResolverPolicy.from_dict(data["resolver"]) if "resolver" in data else None,
        )
        # This also rejects tampered derived dimensions, controls, schema and reference pins.
        if digest(result.to_dict()) != digest(data):
            raise ConfigError("resolved metadata or derived values do not match inventory v1")
        return result


@dataclass(frozen=True)
class Candidate:
    d_model: int
    shared_layers: int
    status: str
    reason: str
    parameters: int | None = None
    delta: int | None = None
    within_tolerance: bool | None = None
    known_training_payload_bytes: int | None = None


@dataclass(frozen=True)
class Resolution:
    status: str
    selected: ResolvedConfig | None
    candidates: tuple[Candidate, ...]
    resolver: ResolverPolicy | None = None
    search_space: SearchPolicy = SearchPolicy()

    def to_dict(self) -> dict:
        result = {
            "status": self.status,
            "scope": "configuration validation and meta construction; no training execution",
            "backend": BACKEND,
            "candidates": [asdict(c) for c in self.candidates],
            "resolved_config": None,
            "resolver": asdict(self.resolver)
            if self.resolver
            else {"policy": "legacy_schema1_compatibility", "mode": "legacy_resolution"},
            "search_space": asdict(self.search_space),
            "ranking_purpose": "PARAMETER_DISTANCE_DIAGNOSTIC_NOT_ARCHITECTURE_RECOMMENDATION",
            "candidate_ranking": [
                asdict(c)
                for c in sorted(
                    (c for c in self.candidates if c.status == "VALID_GENERIC_PATH"),
                    key=lambda c: (abs(c.delta), c.shared_layers, c.d_model),
                )
            ],
            "readiness": configuration_readiness(self.selected, self.resolver).to_dict(),
        }
        if self.selected is not None:
            selected = self.selected
            executable = audit_executable_parameters(selected)
            audit = executable["inventory_audit"]
            actual = audit["unique_trainable_parameters"]
            result.update(
                {
                    "resolved_config": selected.to_dict(),
                    "resolved_sha256": selected.sha256,
                    "parameter_audit": audit,
                    "executable_accounting": executable,
                    "target_parameters": selected.target_parameters,
                    "actual_inventory_parameters": actual,
                    "active_inventory_parameters_dense": actual,
                    "delta": actual - selected.target_parameters,
                    "delta_percent": 100
                    * (actual - selected.target_parameters)
                    / selected.target_parameters,
                    "memory_preflight": memory_preflight(
                        selected.shape, audit, selected.hardware
                    ).to_dict(),
                    "next_gate": result["readiness"]["next_gate"],
                }
            )
        return result


def resolve_config(config: AuthoredConfig) -> Resolution:
    request = config.model
    widths = config.search.widths if request.d_model == "auto" else (request.d_model,)
    depths = config.search.depths if request.shared_layers == "auto" else (request.shared_layers,)
    fixed = asdict(request)
    for key in ("target_parameters", "parameter_tolerance", "moe_enabled"):
        del fixed[key]
    candidates, viable = [], []
    for width in sorted(widths):
        for depth in sorted(depths):
            values = {**fixed, "d_model": width, "shared_layers": depth}
            shape = ModelShape(**values)
            try:
                validate_architecture(shape)
                if config.resolver is not None and width % shape.headdim:
                    raise ConfigError("versioned policy requires d_model divisible by headdim")
                if width % config.search.width_multiple:
                    raise ConfigError(
                        f"d_model={width} violates clean search policy multiple "
                        f"{config.search.width_multiple}; not a universal backend restriction"
                    )
                inventory = instantiate_inventory(shape)
            except ConfigError as exc:
                candidates.append(Candidate(width, depth, "INVALID", str(exc)))
                continue
            audit = audit_parameters(inventory)
            actual = audit["unique_trainable_parameters"]
            estimate = formula_parameter_estimate(shape)
            if actual != estimate:
                raise RuntimeError(
                    f"inventory/formula conflict: instantiated={actual}, formula={estimate}"
                )
            memory = memory_preflight(shape, audit, config.hardware)
            delta = actual - request.target_parameters
            within = abs(delta) <= request.target_parameters * request.parameter_tolerance
            status = "VALID_GENERIC_PATH"
            reason = "legal structural inventory; execution and runtime fit unverified"
            if memory.status == "DOES_NOT_FIT":
                status = "REJECTED_MEMORY"
                reason = (
                    f"known training payload {memory.known_training_payload_bytes} exceeds "
                    f"budget {config.hardware.memory_budget_bytes}; unknown overhead only adds cost"
                )
            candidates.append(
                Candidate(
                    width,
                    depth,
                    status,
                    reason,
                    actual,
                    delta,
                    within,
                    memory.known_training_payload_bytes,
                )
            )
            if status == "VALID_GENERIC_PATH":
                viable.append((abs(delta), depth, width, shape, within))

    def result(status, selected=None):
        return Resolution(status, selected, tuple(candidates), config.resolver, config.search)

    if not viable:
        return result("NO_LEGAL_CANDIDATE")
    if config.resolver is not None and config.resolver.mode == "search":
        return result("SEARCH_COMPLETE")
    _, _, _, chosen, within = min(viable, key=lambda item: item[:3])
    if config.resolver is not None and not within:
        # Legacy configs retain their historical OUTSIDE_TARGET selection/identity.
        # Versioned explicit configs cannot publish a resolved candidate outside their gate.
        return result("OUTSIDE_TARGET")
    selected = ResolvedConfig(
        chosen,
        config.hardware,
        request.target_parameters,
        request.parameter_tolerance,
        digest(config.to_dict()),
        config.resolver,
    )
    return result("WITHIN_TARGET" if within else "OUTSIDE_TARGET", selected)
