"""Fresh-arm bounded LR execution, callable only with explicit prerequisite receipts."""

import ctypes
import hashlib
import json
from pathlib import Path

import torch

from unified_edge.training.config import canonical_hash
from unified_edge.training.device import DevicePolicy
from unified_edge.training.monitoring import MonitoringMode, MonitoringPolicy
from unified_edge.training.pilot_data import evaluate_subset, validate_capacity
from unified_edge.training.pilot_gates import select_lr, validate_prerequisites
from unified_edge.training.pilot_plan import MODEL_SHA, PilotPlan, require
from unified_edge.training.pilot_runner import Budget, subject_identity, write_once
from unified_edge.training.trainer import Trainer


def parameter_hash(model):
    result = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        result.update(name.encode())
        result.update(str((tuple(value.shape), value.dtype)).encode())
        result.update(ctypes.string_at(value.data_ptr(), value.numel() * value.element_size()))
    return result.hexdigest()


def run_lr_probe(
    root,
    *,
    model,
    data,
    data_root,
    subsets,
    anchors,
    receipts,
    trusted_receipt_shas,
    scope="REAL",
    clock=None,
):
    plan = PilotPlan()
    subject = subject_identity(plan, model, data, subsets, anchors)
    check = validate_prerequisites(
        receipts,
        canonical_hash(subject),
        trusted_receipt_shas=trusted_receipt_shas,
        scope=scope,
        purpose="lr_probe",
    )
    require(check["status"] == "PASS", "PREREQUISITE_MISSING")
    subsets.validate(data, exact_capacity=scope == "REAL")
    anchors.validate(data)
    gates = {r.gate: json.loads(r.evidence_json) for r in receipts}
    approved = gates["APPROVED_DATA"]
    require(
        approved["manifest_sha256"] == data.sha256
        and approved["subsets_sha256"] == subsets.sha256
        and approved["anchors_sha256"] == anchors.sha256,
        "approved probe data mismatch",
    )
    if scope == "REAL":
        require(
            Path(root).resolve() == Path(gates["EXECUTION_CONTEXT"]["run_registry"]).resolve(),
            "probe must use the approved durable run registry",
        )
        validate_capacity(data)
        require(model.sha256 == MODEL_SHA and torch.cuda.device_count() == 1, "DEVICE_GATE_FAILED")
        device, policy = "cuda:0", DevicePolicy("20m_t4_single_fp32_pilot", 0)
    else:
        require(
            scope == "TEST_ONLY" and model.shape.d_model <= 64 and model.shape.shared_layers == 1,
            "TEST_ONLY probe requires tiny model",
        )
        device, policy = "cpu", None
    identity = {
        **subject,
        "purpose": "lr_probe",
        "scope": scope,
        "authorization": sorted(r.sha256 for r in receipts),
    }
    root = Path(root) / canonical_hash(identity)
    root.mkdir(parents=True, exist_ok=False)
    write_once(root / "identity.json", identity)
    kwargs = {} if clock is None else {"clock": clock}
    budget = Budget(min(10800, gates["HUMAN_COMPUTE_APPROVAL"]["gpu_seconds"]), **kwargs)
    arms = []
    try:
        for index, lr in enumerate(plan.lr_probe.rates):
            if budget.exhausted:
                raise TimeoutError("BUDGET_EXCEEDED")
            with budget.account():
                trainer = Trainer(
                    model,
                    plan.training_config(lr, device, probe=True),
                    data,
                    data_root,
                    root / f"arm-{index}",
                    scheduler_type="warmup_constant_v1",
                    monitoring=MonitoringPolicy(MonitoringMode.PRODUCTION_FAST),
                    device_policy=policy,
                )
                initial_parameters = parameter_hash(trainer.model)
                initial_order = canonical_hash(trainer.stream.state_dict())
                initial = evaluate_subset(trainer, subsets.selection)
            while trainer.global_step < 256:
                if budget.exhausted:
                    raise TimeoutError("BUDGET_EXCEEDED")
                with budget.account(trainer.device):
                    trainer.step()
            with budget.account(trainer.device):
                endpoint = evaluate_subset(trainer, subsets.selection)
                trainer.save()
            arms.append(
                {
                    "lr": lr,
                    "step": trainer.global_step,
                    "seed": 17,
                    "initial_parameters_sha256": initial_parameters,
                    "data_order_sha256": initial_order,
                    "subset_sha256": subsets.sha256,
                    "scheduler": "warmup_constant_v1",
                    "initial_nll": initial["nll"],
                    "endpoint_nll": endpoint["nll"],
                    "finite_states": True,
                    "finite_gradients": True,
                    "finite_parameters": True,
                    "losses": [r["loss"] for r in trainer.metrics],
                    "clipped": [r["clipped"] for r in trainer.metrics],
                    "actual_target_bytes": trainer.bytes_seen,
                }
            )
            write_once(root / f"arm-{index}.json", arms[-1])
            del trainer
        result = select_lr(arms, budget.used)
    except Exception as error:
        result = {
            "status": "STOP",
            "reason": str(error),
            "selected_lr": None,
            "gpu_seconds": budget.used,
        }
    result.update(scope=scope, identity=identity, real_probe_executed=scope == "REAL")
    write_once(root / "selection.json", result)
    return result
