"""Evidence-scoped status, outside immutable model/checkpoint serialization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from unified_edge.config import ModelShape, ResolverPolicy

if TYPE_CHECKING:
    from unified_edge.resolve import ResolvedConfig

# Identifies the accepted historical configuration, not arbitrary models near 2M.
FROZEN_2M_RESOLVED_SHA256 = "7fe06b639741003e4c5ad9d57d32c2fd2b86f6805b2048ad1efd6aaa75e3a6f1"
REVIEWED_20M_SHAPE = ModelShape(512, 12, 64, 256, 64, 4, 2, 64, 8, 16)


@dataclass(frozen=True)
class Readiness:
    scale_stage: str
    architecture_status: str
    accounting_status: str
    construction_status: str
    training_execution_status: str
    quality_status: str
    next_gate: str
    evidence_scope: str
    evidence_reference: str | None = None
    pilot_execution_ready: bool = False
    training_authorized: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def configuration_readiness(
    selected: ResolvedConfig | None, policy: ResolverPolicy | None
) -> Readiness:
    if selected is None:
        return Readiness(
            "20M_SEARCH" if policy and policy.policy == "edge_dense_v2_20m" else "UNSELECTED",
            "NO_SELECTED_ARCHITECTURE",
            "CANDIDATE_INVENTORY_ONLY",
            "NOT_CONSTRUCTED",
            "NOT_RUN",
            "UNVERIFIED",
            "REVIEW_CANDIDATES_OR_FIX_CONFIGURATION",
            "No executable configuration selected; candidate ranking is diagnostic only.",
        )
    if selected.sha256 == FROZEN_2M_RESOLVED_SHA256:
        return Readiness(
            "2M_GEN0_FROZEN",
            "IMPLEMENTED",
            "META_VERIFIED",
            "META_CONSTRUCTED",
            "COMPLETED_FROZEN_LINEAGE",
            "ENGINEERING_BASELINE_ONLY",
            "PRESERVE_FROZEN_2M_COMPLETE_20M_R9",
            "Training completion refers only to the accepted historical Gen-0 run; "
            "this resolver does not train, load weights or authorize current-source resume.",
            "reports/full_training_2m_final.json at 3d2698b42404ec67834ead162db0bd10240d270b",
        )
    if (
        policy is not None
        and policy.policy == "edge_dense_v2_20m"
        and selected.shape == REVIEWED_20M_SHAPE
        and selected.target_parameters == 20_000_000
        and selected.parameter_tolerance == 0.05
    ):
        return Readiness(
            "20M_PILOT_CANDIDATE",
            "RECOMMENDED_FOR_PILOT",
            "META_VERIFIED",
            "META_CONSTRUCTED",
            "NOT_RUN",
            "UNVERIFIED",
            "IMPLEMENT_R9_THEN_RUN_APPROVED_20M_EXECUTION_GATES",
            "R1-R8 infrastructure implemented and reviewed; R9 remains unimplemented. "
            "Hardware execution gates remain unpassed. No pilot or training authorization.",
            "reports/20m_prebuild_audit.json; reports/20m_i1_resolver_config_implementation.json; "
            "reports/20m_i2_training_infrastructure.json",
        )
    return Readiness(
        "UNREVIEWED_CONFIGURATION",
        "IMPLEMENTED_REFERENCE",
        "META_VERIFIED",
        "META_CONSTRUCTED",
        "NOT_RUN",
        "UNVERIFIED",
        "ARCHITECTURE_AND_EXECUTION_REVIEW",
        "Construction/accounting evidence only; no training or quality evidence for this identity.",
    )
