"""Strict authored contracts and immutable, explicit model dimensions."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml

SCHEMA_VERSION = "1"
INVENTORY_VERSION = "1"
BACKEND = "torch_reference_inventory_v1"
REFERENCE_COMMIT = "d7b1ceb3c367ec9022925e812f507bcf706937c6"
CONTROL_IDS = (
    ("pad", 256),
    ("bos", 257),
    ("eos", 258),
    ("tool_call", 259),
    ("tool_result", 260),
    ("rag_begin", 261),
    ("rag_end", 262),
    ("code_begin", 263),
    ("code_end", 264),
    ("error", 265),
    ("retry", 266),
)
NUM_SYMBOLS = 256 + len(CONTROL_IDS)


class ConfigError(ValueError):
    """An invalid configuration, with the offending field in the message."""


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def mapping(value: object, allowed: set[str], path: str) -> dict:
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise ConfigError(f"{path} must be a mapping with string keys")
    unknown = set(value) - allowed
    if unknown:
        raise ConfigError(f"{path}: unknown fields {sorted(unknown)}")
    return value


def positive_int(value: object, path: str) -> None:
    if type(value) is not int or value <= 0:
        raise ConfigError(f"{path} must be a positive integer, got {value!r}")


@dataclass(frozen=True)
class ModelRequest:
    target_parameters: int = 2_000_000
    parameter_tolerance: float = 0.02
    d_model: int | str = "auto"
    shared_layers: int | str = "auto"
    byte_dim: int = 64
    decoder_dim: int = 128
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    headdim: int = 64
    patch_size: int = 8
    chunk_patches: int = 16
    moe_enabled: bool = False

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name in {"d_model", "shared_layers"}:
                if value != "auto":
                    positive_int(value, f"model.{f.name}")
            elif f.name == "parameter_tolerance":
                if (
                    type(value) not in (int, float)
                    or not math.isfinite(value)
                    or not 0 <= value < 1
                ):
                    raise ConfigError("model.parameter_tolerance must be finite in [0,1)")
            elif f.name == "moe_enabled":
                if type(value) is not bool or value:
                    raise ConfigError("model.moe_enabled must be false; MoE is not implemented")
            else:
                positive_int(value, f"model.{f.name}")


@dataclass(frozen=True)
class SearchPolicy:
    widths: tuple[int, ...] = (128, 192, 256)
    depths: tuple[int, ...] = (2, 3, 4, 6, 8, 12)
    width_multiple: int = 64

    def __post_init__(self) -> None:
        positive_int(self.width_multiple, "search.width_multiple")
        for name in ("widths", "depths"):
            values = getattr(self, name)
            if type(values) is not tuple or not values:
                raise ConfigError(f"search.{name} must be a nonempty tuple")
            for value in values:
                positive_int(value, f"search.{name}")
            if len(set(values)) != len(values):
                raise ConfigError(f"search.{name} has duplicate candidates")
        if len(self.widths) * len(self.depths) > 4096:
            raise ConfigError("search may enumerate at most 4096 candidates")


@dataclass(frozen=True)
class Hardware:
    backend: str = BACKEND
    device: str = "cpu"
    precision: str = "fp32"
    batch_size: int = 1
    memory_budget_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.backend != BACKEND:
            raise ConfigError(f"hardware.backend unsupported: {self.backend!r}")
        if self.device != "cpu" or self.precision != "fp32":
            raise ConfigError("inventory profile supports explicit cpu/fp32 only")
        positive_int(self.batch_size, "hardware.batch_size")
        if self.memory_budget_bytes is not None:
            positive_int(self.memory_budget_bytes, "hardware.memory_budget_bytes")


@dataclass(frozen=True)
class AuthoredConfig:
    schema_version: str = SCHEMA_VERSION
    model: ModelRequest = ModelRequest()
    search: SearchPolicy = SearchPolicy()
    hardware: Hardware = Hardware()

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ConfigError(f"unsupported schema_version {self.schema_version!r}")
        for name, cls in (
            ("model", ModelRequest),
            ("search", SearchPolicy),
            ("hardware", Hardware),
        ):
            if not isinstance(getattr(self, name), cls):
                raise ConfigError(f"{name} must be {cls.__name__}")

    @classmethod
    def from_dict(cls, value: object) -> AuthoredConfig:
        data = mapping(value, {"schema_version", "model", "search", "hardware"}, "config")
        if "schema_version" not in data:
            raise ConfigError("config.schema_version is required")
        parts = {}
        for name, contract in (
            ("model", ModelRequest),
            ("search", SearchPolicy),
            ("hardware", Hardware),
        ):
            part = dict(mapping(data.get(name, {}), {f.name for f in fields(contract)}, name))
            if name == "search":
                for key in ("widths", "depths"):
                    if key in part:
                        if not isinstance(part[key], list):
                            raise ConfigError(f"search.{key} must be a YAML/JSON list")
                        part[key] = tuple(part[key])
            parts[name] = contract(**part)
        return cls(schema_version=data["schema_version"], **parts)

    def to_dict(self) -> dict:
        return json.loads(canonical_json(asdict(self)))


@dataclass(frozen=True)
class ModelShape:
    d_model: int
    shared_layers: int
    byte_dim: int
    decoder_dim: int
    d_state: int
    d_conv: int
    expand: int
    headdim: int
    patch_size: int
    chunk_patches: int

    def __post_init__(self) -> None:
        for f in fields(self):
            positive_int(getattr(self, f.name), f"shape.{f.name}")

    @property
    def d_inner(self) -> int:
        return self.expand * self.d_model

    @property
    def nheads(self) -> int:
        return self.d_inner // self.headdim

    def to_dict(self) -> dict:
        return asdict(self)


def validate_architecture(shape: ModelShape) -> None:
    if shape.shared_layers > 1024:
        raise ConfigError("inventory audit safety limit: shared_layers must be <=1024")
    if shape.patch_size != 8 or shape.chunk_patches != 16:
        raise ConfigError("inventory v1 requires patch_size=8 and chunk_patches=16")
    if shape.d_inner % shape.headdim:
        raise ConfigError(
            f"d_inner={shape.d_inner} must be divisible by headdim={shape.headdim}; "
            f"backend={BACKEND}"
        )


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML with duplicate mapping keys rejected instead of overwritten."""


def _construct_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ConfigError("YAML keys must be strings")
        if key in result:
            raise ConfigError(f"duplicate YAML key {key!r} at line {key_node.start_mark.line + 1}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def load_config(path: str | Path) -> AuthoredConfig:
    try:
        data = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    return AuthoredConfig.from_dict(data)
