"""Versioned observational cadences; never part of optimizer or data semantics."""

from dataclasses import asdict, dataclass
from enum import Enum


class MonitoringMode(str, Enum):
    REFERENCE_PARANOID = "REFERENCE_PARANOID"
    PRODUCTION_FAST = "PRODUCTION_FAST"


@dataclass(frozen=True)
class MonitoringPolicy:
    mode: MonitoringMode = MonitoringMode.REFERENCE_PARANOID
    schema: str = "1"

    def __post_init__(self):
        if self.schema != "1" or not isinstance(self.mode, MonitoringMode):
            raise ValueError("monitoring requires schema 1 and a typed MonitoringMode")

    @property
    def verify_each_update(self) -> bool:
        return self.mode is MonitoringMode.REFERENCE_PARANOID

    def detailed(self, step: int) -> bool:
        return self.verify_each_update or step <= 10 or step % 100 == 0

    def to_dict(self) -> dict:
        return {**asdict(self), "mode": self.mode.value}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {"mode", "schema"}:
            raise ValueError("invalid monitoring policy fields")
        return cls(MonitoringMode(value["mode"]), value["schema"])
