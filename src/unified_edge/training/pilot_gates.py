"""Receipt identities, prerequisite admission, LR endpoint selection and promotion."""

import json
from dataclasses import dataclass
from statistics import median

from unified_edge.training.config import canonical_hash
from unified_edge.training.pilot_plan import DOMAINS, Contract, LRProbePlan, digest, number, require

GATES = (
    "20M-0",
    "20M-1",
    "20M-2",
    "20M-3",
    "LR_PROBE",
    "APPROVED_DATA",
    "HUMAN_COMPUTE_APPROVAL",
    "EXECUTION_CONTEXT",
)
REQUIREMENTS = {
    "20M-0": ("parameters", "tensors", "tensor_signature", "config_roundtrip", "unclassified"),
    "20M-1": (
        "finite_loss",
        "finite_gradients",
        "parameter_participation",
        "causality",
        "batch_stream_reference_state_parity",
    ),
    "20M-2": (
        "hardware_receipt_sha256",
        "phase_memory_sha256",
        "allocator_headroom_pass",
        "target_bytes_per_second",
        "projected_seconds",
        "reserve_multiplier",
        "same_backend_exact_resume",
    ),
    "20M-3": (
        "fixture_hex",
        "repetitions",
        "updates",
        "batch",
        "accumulation",
        "length",
        "lr",
        "warmup",
        "train_reduction",
        "distinct_validation_logged",
        "resume_step",
        "exact_resume",
        "real_smoke_updates",
        "real_smoke_finite",
    ),
    "LR_PROBE": ("selection_receipt_sha256", "selected_lr"),
    "APPROVED_DATA": (
        "manifest_sha256",
        "subsets_sha256",
        "anchors_sha256",
        "review_receipt_sha256",
    ),
    "HUMAN_COMPUTE_APPROVAL": ("approval_reference", "gpu_seconds", "purpose"),
    "EXECUTION_CONTEXT": (
        "context_reference",
        "device",
        "profile",
        "single_visible_gpu",
        "host_boundary",
        "run_registry",
    ),
}


@dataclass(frozen=True)
class GateReceipt(Contract):
    schema: str
    gate: str
    scope: str
    subject_sha256: str
    evidence_sha256: str
    evidence_json: str

    def __post_init__(self):
        require(
            self.schema == "edge-gate-1"
            and self.gate in GATES
            and self.scope in ("REAL", "TEST_ONLY"),
            "invalid gate receipt",
        )
        digest(self.subject_sha256)
        digest(self.evidence_sha256)
        payload = json.loads(self.evidence_json)
        require(
            isinstance(payload, dict) and set(payload) == set(REQUIREMENTS[self.gate]),
            "gate receipt missing/unknown evidence fields",
        )
        require(canonical_hash(payload) == self.evidence_sha256, "evidence digest mismatch")

    @classmethod
    def from_dict(cls, value):
        require(
            isinstance(value, dict) and set(value) == set(cls.__dataclass_fields__),
            "unknown/missing gate fields",
        )
        return cls(**value)

    def validate(self):
        p = json.loads(self.evidence_json)
        if self.gate == "20M-0":
            require(
                p
                == {
                    "parameters": 20387531,
                    "tensors": 128,
                    "tensor_signature": (
                        "3f242a2332fa41d4737f5d184b1593051bf8384a7cb5e954f457c136848dfd61"
                    ),
                    "config_roundtrip": True,
                    "unclassified": 0,
                },
                "construction gate failed",
            )
        elif self.gate == "20M-1":
            require(all(v is True for v in p.values()), "forward/backward gate failed")
        elif self.gate == "20M-2":
            digest(p["hardware_receipt_sha256"])
            digest(p["phase_memory_sha256"])
            require(
                p["allocator_headroom_pass"] is True
                and p["same_backend_exact_resume"] is True
                and number(p["target_bytes_per_second"]) > 0
                and number(p["reserve_multiplier"]) == number(1.3)
                and 0 < number(p["projected_seconds"]) * number(1.3) <= number(86400),
                "hardware budget/mechanics gate failed",
            )
            require(
                number(p["projected_seconds"]) * number(p["target_bytes_per_second"])
                >= number(65536000),
                "projection contradicts measured useful throughput",
            )
        elif self.gate == "20M-3":
            exact = {
                "fixture_hex": b"edge400: abcdef\n".hex(),
                "repetitions": 64,
                "updates": 60,
                "batch": 2,
                "accumulation": 2,
                "length": 32,
                "lr": 0.003,
                "warmup": 5,
                "resume_step": 30,
            }
            require(
                all(p[k] == v and type(p[k]) is type(v) for k, v in exact.items())
                and number(p["train_reduction"]) >= number(0.5)
                and p["distinct_validation_logged"] is True
                and p["exact_resume"] is True
                and type(p["real_smoke_updates"]) is int
                and 1 <= p["real_smoke_updates"] <= 128
                and p["real_smoke_finite"] is True,
                "mechanics gate failed",
            )
        elif self.gate == "LR_PROBE":
            digest(p["selection_receipt_sha256"])
            require(p["selected_lr"] in LRProbePlan().rates, "undeclared selected LR")
        elif self.gate == "APPROVED_DATA":
            for value in p.values():
                digest(value)
        elif self.gate == "HUMAN_COMPUTE_APPROVAL":
            require(
                isinstance(p["approval_reference"], str)
                and bool(p["approval_reference"])
                and p["purpose"] in ("pilot", "lr_probe")
                and type(p["gpu_seconds"]) is int
                and 0 < p["gpu_seconds"] <= (86400 if p["purpose"] == "pilot" else 10800),
                "invalid bounded human approval",
            )
        else:
            require(
                p["device"] == "cuda:0"
                and p["profile"] == "20m_t4_single_fp32_pilot"
                and p["single_visible_gpu"] is True
                and p["host_boundary"] == "APPROVED_HOST"
                and isinstance(p["context_reference"], str)
                and bool(p["context_reference"])
                and isinstance(p["run_registry"], str)
                and bool(p["run_registry"]),
                "execution context must preserve approved single-T4 host boundary",
            )


def validate_prerequisites(
    receipts, subject_sha256, *, trusted_receipt_shas=(), scope="REAL", purpose="pilot"
):
    require(
        type(receipts) is tuple and all(type(r) is GateReceipt for r in receipts),
        "typed immutable gate receipts required",
    )
    require(len({r.gate for r in receipts}) == len(receipts), "duplicate prerequisite")
    required = set(GATES) - ({"LR_PROBE"} if purpose == "lr_probe" else set())
    require(
        purpose in ("pilot", "lr_probe") and scope in ("REAL", "TEST_ONLY"), "invalid gate scope"
    )
    missing = sorted(required - {r.gate for r in receipts})
    for receipt in receipts:
        require(
            receipt.scope == scope and receipt.subject_sha256 == subject_sha256,
            "gate scope/subject mismatch; TEST_ONLY cannot authorize REAL",
        )
        require(receipt.sha256 in trusted_receipt_shas, "receipt lacks explicit trusted approval")
        receipt.validate()
        if receipt.gate == "HUMAN_COMPUTE_APPROVAL":
            require(
                json.loads(receipt.evidence_json)["purpose"] == purpose, "approval purpose mismatch"
            )
    return {
        "status": "PASS" if not missing else "INCOMPLETE",
        "missing": missing,
        "execution_started": False,
        "scope": scope,
    }


def unstable_losses(losses):
    streak = 0
    for index, loss in enumerate(losses):
        number(loss)
        if index >= 64:
            high = number(loss) > 2 * number(median(losses[index - 32 : index]))
            streak = streak + 1 if high else 0
            if streak >= 3:
                return True
    return False


def select_lr(arms, elapsed_gpu_seconds, plan=LRProbePlan()):
    require(number(elapsed_gpu_seconds) >= 0, "invalid probe time")
    if number(elapsed_gpu_seconds) > number(plan.gpu_seconds):
        return {"status": "STOP", "reason": "BUDGET_EXCEEDED", "selected_lr": None}
    require(
        len(arms) == 3 and {a["lr"] for a in arms} == set(plan.rates), "all declared arms required"
    )
    for key in ("initial_parameters_sha256", "data_order_sha256", "subset_sha256"):
        require(len({a[key] for a in arms}) == 1, f"probe arms must share {key}")
        digest(arms[0][key])
    stable = []
    for a in arms:
        require(
            a["step"] == plan.updates
            and a["seed"] == 17
            and len(a["losses"]) == 256
            and len(a["clipped"]) == 256
            and all(type(v) is bool for v in a["clipped"])
            and a["scheduler"] == plan.scheduler,
            "probe requires complete fixed endpoints",
        )
        good = (
            a["finite_states"] is True
            and a["finite_gradients"] is True
            and a["finite_parameters"] is True
            and number(a["initial_nll"]) - number(a["endpoint_nll"]) >= number(0.1)
            and not unstable_losses(a["losses"])
            and sum(a["clipped"][-128:]) <= 64
        )
        if good:
            stable.append(a)
    if not stable:
        return {"status": "STOP", "reason": "LR_PROBE_FAILED", "selected_lr": None}
    best = min(number(a["endpoint_nll"]) for a in stable)
    chosen = min(
        (a for a in stable if number(a["endpoint_nll"]) - best <= number(0.01)),
        key=lambda a: a["lr"],
    )
    return {
        "status": "PASS",
        "selected_lr": chosen["lr"],
        "step": 256,
        "plan_sha256": plan.sha256,
        "arms_sha256": canonical_hash({"arms": arms}),
        "gpu_seconds": elapsed_gpu_seconds,
        "selection": "endpoint_only",
    }


def evaluate_promotion(evidence):
    required = {
        "implemented",
        "lr_status",
        "updates",
        "data_valid",
        "exposure_valid",
        "initial_nll",
        "endpoint_nll",
        "domain_improvements",
        "checkpoint_export_consistent",
        "stop_reason",
        "endpoint_only",
    }
    if not isinstance(evidence, dict) or set(evidence) != required:
        return {"status": "INCOMPLETE", "reason": "missing/unknown endpoint evidence"}
    e = evidence
    checks = {
        "implementation": e["implemented"] == [f"R{i}" for i in range(1, 10)],
        "lr": e["lr_status"] == "PASS",
        "bounded_completion": type(e["updates"]) is int and e["updates"] == 8000,
        "data": e["data_valid"] is True,
        "exposure": e["exposure_valid"] is True,
        "nll": number(e["initial_nll"]) - number(e["endpoint_nll"]) >= number(0.1),
        "domains": set(e["domain_improvements"]) == set(DOMAINS)
        and all(number(v) >= number(0.02) for v in e["domain_improvements"].values()),
        "durability": e["checkpoint_export_consistent"] is True,
        "no_stop": e["stop_reason"] is None,
        "endpoint_only": e["endpoint_only"] is True,
    }
    return {"status": "PASS" if all(checks.values()) else "STOP", "checks": checks}
