"""Approval-bound R9 coordinator over Trainer; no cloud/network execution facility."""

import hashlib
import json
import os
import shutil
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import torch

from unified_edge.training.checkpoint import load_checkpoint
from unified_edge.training.config import canonical_hash
from unified_edge.training.device import DevicePolicy
from unified_edge.training.monitoring import MonitoringMode, MonitoringPolicy
from unified_edge.training.pilot_context import context_panel
from unified_edge.training.pilot_data import evaluate_subset, validate_capacity
from unified_edge.training.pilot_diagnostics import (
    evaluate_quality,
    generate_panel,
    isolated_diagnostics,
)
from unified_edge.training.pilot_gates import (
    evaluate_promotion,
    unstable_losses,
    validate_prerequisites,
)
from unified_edge.training.pilot_plan import MODEL_SHA, Failure, PilotPlan, digest, number, require
from unified_edge.training.trainer import Trainer, code_identity


def write_once(path, value):
    """Checksummed JSON atomically published on the same local filesystem."""
    path = Path(path)
    require(not path.exists(), "refuse evidence overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.pending")
    data = {"value": value, "sha256": canonical_hash(value)}
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, sort_keys=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    require(not path.exists(), "refuse evidence overwrite")
    os.rename(temporary, path)


def read_record(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    require(
        isinstance(data, dict)
        and set(data) == {"value", "sha256"}
        and canonical_hash(data["value"]) == data["sha256"],
        "evidence integrity failure",
    )
    return data["value"]


def sha_file(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class Budget:
    def __init__(self, limit, *, clock=time.monotonic, used=0.0):
        require(number(limit) > 0 and number(used) >= 0, "invalid budget")
        self.limit, self.used, self.clock = limit, used, clock

    @contextmanager
    def account(self, device=None):
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize(device)
        start = self.clock()
        try:
            yield
        finally:
            if device is not None and device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = self.clock() - start
            require(number(elapsed) >= 0, "clock moved backwards")
            self.used += elapsed

    @property
    def exhausted(self):
        return number(self.used) >= number(self.limit)


def subject_identity(plan, model, data, subsets, anchors):
    return {
        "schema": "edge-run-subject-1",
        "plan_sha256": plan.sha256,
        "model_sha256": model.sha256,
        "data_sha256": data.sha256,
        "subsets_sha256": subsets.sha256,
        "anchors_sha256": anchors.sha256,
        "seed": plan.seed,
        "source_sha256": code_identity()["source_sha256"],
    }


def exposure(trainer, physical=None):
    rows = trainer.metrics
    return {
        "logical_updates": trainer.global_step,
        "microsteps": trainer.micro_step,
        "sequences": trainer.examples_seen,
        "examples": trainer.examples_seen,
        "valid_target_bytes": trainer.bytes_seen,
        "tail_windows": sum(r["tail_windows"] for r in rows),
        "tail_bytes": sum(r["tail_bytes"] for r in rows),
        "physical_retry_replay": physical or {"updates": 0, "bytes": 0, "status": "EXACT"},
        "padding_bos_counted": False,
        "epoch_promise": False,
    }


class PilotRunner:
    """Explicit launch/resume API. Callers supply approved immutable receipt identities.

    TEST_ONLY scope is restricted to small CPU fixtures. A validated plan by itself
    cannot construct this coordinator. Session ownership is an exclusive lock.
    """

    def __init__(
        self,
        root,
        *,
        plan,
        model,
        config,
        data,
        data_root,
        subsets,
        anchors,
        receipts,
        trusted_receipt_shas,
        selected_lr_receipt_sha256,
        scope="REAL",
        resume=False,
        clock=time.monotonic,
        hooks=None,
    ):
        require(type(plan) is PilotPlan, "typed pilot plan required")
        digest(selected_lr_receipt_sha256)
        subsets.validate(data, exact_capacity=scope == "REAL")
        anchors.validate(data)
        subject = subject_identity(plan, model, data, subsets, anchors)
        gate_result = validate_prerequisites(
            receipts,
            canonical_hash(subject),
            trusted_receipt_shas=trusted_receipt_shas,
            scope=scope,
        )
        require(gate_result["status"] == "PASS", "PREREQUISITE_MISSING")
        gates = {r.gate: json.loads(r.evidence_json) for r in receipts}
        require(
            gates["LR_PROBE"]["selection_receipt_sha256"] == selected_lr_receipt_sha256
            and gates["LR_PROBE"]["selected_lr"] == config.learning_rate,
            "selected LR receipt mismatch",
        )
        approved = gates["APPROVED_DATA"]
        require(
            approved["manifest_sha256"] == data.sha256
            and approved["subsets_sha256"] == subsets.sha256
            and approved["anchors_sha256"] == anchors.sha256,
            "approved data identities mismatch",
        )
        policy = None
        if scope == "REAL":
            require(
                Path(root).resolve() == Path(gates["EXECUTION_CONTEXT"]["run_registry"]).resolve(),
                "launch must use the approved durable run registry",
            )
            validate_capacity(data)
            require(
                model.sha256 == MODEL_SHA
                and config == plan.training_config(config.learning_rate, "cuda:0"),
                "real pilot model/config mismatch",
            )
            require(torch.cuda.device_count() == 1, "DEVICE_GATE_FAILED: mask one T4 before launch")
            policy = DevicePolicy("20m_t4_single_fp32_pilot", 0)
        else:
            require(
                scope == "TEST_ONLY"
                and config.device == "cpu"
                and model.shape.d_model <= 64
                and model.shape.shared_layers == 1
                and config.total_steps <= 12,
                "TEST_ONLY limited to tiny CPU fixtures",
            )
        self.identity = {
            **subject,
            "scope": scope,
            "authorization_sha256": canonical_hash(
                {"receipts": sorted(r.sha256 for r in receipts)}
            ),
            "selected_lr_receipt_sha256": selected_lr_receipt_sha256,
            "training_sha256": config.sha256,
            "endpoint": config.total_steps,
        }
        self.run_id = canonical_hash(self.identity)
        self.root = Path(root).resolve() / self.run_id
        self.plan, self.subsets, self.anchors, self.hooks = plan, subsets, anchors, hooks or {}
        self.state = "PLANNED"
        if not resume:
            self.root.mkdir(parents=True, exist_ok=False)  # duplicate launch always refused
            write_once(self.root / "identity.json", self.identity)
        require(read_record(self.root / "identity.json") == self.identity, "run identity mismatch")
        self.lock = self.root / "session.lock"
        # OS advisory lock releases on process death; a second live owner is refused.
        self._lock_handle = self.lock.open("a+b")
        self._lock_handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                if self.lock.stat().st_size == 0:
                    self._lock_handle.write(b"0")
                    self._lock_handle.flush()
                    self._lock_handle.seek(0)
                msvcrt.locking(self._lock_handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError):
            self._lock_handle.close()
            raise RuntimeError("run already has a live owner") from None
        self.budget = Budget(
            min(plan.gpu_seconds, gates["HUMAN_COMPUTE_APPROVAL"]["gpu_seconds"]), clock=clock
        )
        self.completed, self.stop_reason = {}, None
        self.failure_category = None
        self.last_durable_exposure = None
        checkpoint = None
        try:
            self._transition("LR_SELECTED")
            self._transition("READY_FOR_HARDWARE_GATES")
            if resume:
                checkpoint, saved = self._recover()
                self.budget.used = saved["gpu_seconds"]
                self.stop_reason = saved["stop_reason"]
                self.last_durable_exposure = saved["exposure"]
            attempt = self.root / "sessions" / uuid.uuid4().hex
            with self.budget.account():
                self.trainer = Trainer(
                    model,
                    config,
                    data,
                    data_root,
                    attempt,
                    resume_from=checkpoint,
                    monitoring=MonitoringPolicy(MonitoringMode.PRODUCTION_FAST),
                    device_policy=policy,
                )
            self._transition("RUNNING")
            if not resume:
                self._snapshot()
        except BaseException:
            self.close()
            raise

    def close(self):
        if not self._lock_handle.closed:
            self._lock_handle.close()

    def _transition(self, target):
        edges = {
            "PLANNED": {"LR_SELECTED"},
            "LR_SELECTED": {"READY_FOR_HARDWARE_GATES"},
            "READY_FOR_HARDWARE_GATES": {"RUNNING"},
            "RUNNING": {"GRACEFUL_STOPPING", "COMPLETED_UNREVIEWED", "FAILED", "BLOCKED"},
            "GRACEFUL_STOPPING": {"COMPLETED_UNREVIEWED", "FAILED", "BLOCKED"},
        }
        require(target in edges.get(self.state, set()), "illegal run state transition")
        self.state = target

    def _hook(self, name):
        if name in self.hooks:
            self.hooks[name](self)

    def _snapshot(self):
        step = self.trainer.global_step
        path = self.root / "steps" / f"{step:06d}"
        if path.exists():
            return path
        pending = path.with_name(f".pending_{step:06d}")
        pending.mkdir(parents=True, exist_ok=False)
        with self.budget.account(self.trainer.device):
            cp = self.trainer.save(checkpoint_path=pending / "checkpoint")
        write_once(
            pending / "boundary.json",
            {
                "schema": "edge-boundary-1",
                "identity": self.identity,
                "step": step,
                "checkpoint_sha256": sha_file(cp / "state.pt"),
                "gpu_seconds": self.budget.used,
                "completed_events": dict(self.completed),
                "stop_reason": self.stop_reason,
                "exposure": exposure(self.trainer),
            },
        )
        self._hook("before_checkpoint_publication")
        os.rename(pending, path)
        self.last_durable_exposure = exposure(self.trainer)
        self._hook("after_checkpoint_publication")
        self._prune_recovery()
        return path

    def _prune_recovery(self):
        paths = sorted((self.root / "steps").glob("[0-9]*"))
        for old in paths[:-2]:
            step = int(old.name)
            # Keep declared cadence checkpoints; remove only superseded private recovery snapshots.
            if "checkpoint" not in self.plan.evaluation.events(
                step, self.trainer.config.total_steps
            ):
                require(
                    old.resolve().parent == (self.root / "steps").resolve()
                    and old.resolve().is_relative_to(self.root),
                    "unsafe recovery path",
                )
                shutil.rmtree(old)

    def _recover(self):
        steps = self.root / "steps"
        for pending in steps.glob(".pending_*"):
            saved = read_record(pending / "boundary.json")
            require(
                saved["identity"] == self.identity
                and sha_file(pending / "checkpoint/state.pt") == saved["checkpoint_sha256"],
                "pending checkpoint integrity mismatch",
            )
            cp = load_checkpoint(pending / "checkpoint")
            require(cp["global_step"] == saved["step"], "pending step mismatch")
            destination = steps / f"{saved['step']:06d}"
            require(not destination.exists(), "duplicate pending boundary")
            os.rename(pending, destination)
        boundaries = sorted(steps.glob("[0-9]*"))
        require(bool(boundaries), "no durable initial boundary")
        path = boundaries[-1]
        saved = read_record(path / "boundary.json")
        require(
            saved["identity"] == self.identity
            and sha_file(path / "checkpoint/state.pt") == saved["checkpoint_sha256"],
            "checkpoint/run identity mismatch",
        )
        for intent in (self.root / "updates").glob("*.json"):
            value = read_record(intent)
            require(
                value["step"] <= saved["step"],
                "ambiguous unfinished update: BLOCKED; no automatic optimizer replay",
            )
        self.completed = dict(saved["completed_events"])
        for key, expected in self.completed.items():
            require(
                canonical_hash(read_record(self.root / "events" / f"{key}.json")) == expected,
                "missing/changed completed event; refuse to skip or replay it",
            )
        event_seconds = 0.0
        for event in sorted((self.root / "events").glob("*.json")):
            value = read_record(event)
            require(
                value["run_id"] == self.run_id and value["step"] <= saved["step"],
                "event ahead of durable Trainer boundary",
            )
            require(
                value["name"]
                in self.plan.evaluation.events(value["step"], self.identity["endpoint"])
                or value["name"] == "checkpoint"
                or (value["name"] == "initial_confirmation" and value["step"] == 0),
                "unexpected persisted event",
            )
            require(event.stem == f"{value['step']:06d}_{value['name']}", "event filename mismatch")
            if event.stem in self.completed:
                require(
                    self.completed[event.stem] == canonical_hash(value), "event identity mismatch"
                )
            self.completed[event.stem] = canonical_hash(value)
            event_seconds = max(event_seconds, value["gpu_seconds"])
        for intent in (self.root / "event_intents").glob("*.json"):
            require(
                intent.stem in self.completed,
                "unfinished diagnostic: elapsed budget unknown; explicit review required",
            )
        for step in range(saved["step"]):
            for name in self.plan.evaluation.events(step, self.identity["endpoint"]):
                require(f"{step:06d}_{name}" in self.completed, "missing prior cadence event")
        if saved["step"] > 0:
            require("000000_initial_confirmation" in self.completed, "missing initial confirmation")
        saved["gpu_seconds"] = max(saved["gpu_seconds"], event_seconds)
        return path / "checkpoint", saved

    def _event(self, name):
        step = self.trainer.global_step
        key = f"{step:06d}_{name}"
        if key in self.completed:
            return
        self._hook("before_event")
        write_once(self.root / "event_intents" / f"{key}.json", {"event": key})
        with self.budget.account(self.trainer.device), isolated_diagnostics(self.trainer.model):
            if name == "checkpoint":
                result = {
                    "boundary_sha256": sha_file(
                        self.root / "steps" / f"{step:06d}" / "boundary.json"
                    )
                }
            elif name in ("monitor", "confirmation", "initial_confirmation"):
                subset = "confirmation" if name == "initial_confirmation" else name
                result = evaluate_subset(self.trainer, getattr(self.subsets, subset))
            elif name == "generation":
                result = {"traces": generate_panel(self.trainer.model, self.plan.generation)}
            else:
                result = context_panel(self.trainer, self.anchors)
        value = {
            "schema": "edge-event-1",
            "run_id": self.run_id,
            "step": step,
            "name": name,
            "result": result,
            "gpu_seconds": self.budget.used,
        }
        write_once(self.root / "events" / f"{key}.json", value)
        self.completed[key] = canonical_hash(value)
        self._hook("after_event")

    def request_stop(self, reason=Failure.USER_STOP):
        require(isinstance(reason, Failure), "typed stop reason required")
        self.stop_reason = reason.value
        self.failure_category = reason.value

    def run(self):
        require(self.state == "RUNNING", "run is not active")
        terminal = self.root / "terminal.json"
        if terminal.exists():
            return read_record(terminal)
        try:
            while True:
                if self.budget.exhausted:
                    self.request_stop(Failure.BUDGET_EXCEEDED)
                if self.stop_reason:
                    self._transition("GRACEFUL_STOPPING")
                    self._event("checkpoint")
                    break
                if self.trainer.global_step == 0:
                    self._event("initial_confirmation")
                    if self.budget.exhausted:
                        self.request_stop(Failure.BUDGET_EXCEEDED)
                    if self.stop_reason:
                        continue
                for name in self.plan.evaluation.events(
                    self.trainer.global_step, self.trainer.config.total_steps
                ):
                    self._event(name)
                    if self.budget.exhausted:
                        self.request_stop(Failure.BUDGET_EXCEEDED)
                    if self.stop_reason:
                        break
                if self.stop_reason:
                    continue
                if self.trainer.global_step == self.trainer.config.total_steps:
                    break
                step = self.trainer.global_step + 1
                write_once(self.root / "updates" / f"{step:06d}.json", {"step": step})
                try:
                    with self.budget.account(self.trainer.device):
                        self.trainer.step()
                except Exception as error:
                    self.failure_category = self._classify(error, inside_update=True)
                    if (
                        not self.trainer._failed
                        and self.failure_category == Failure.DATA_INTEGRITY_FAILURE.value
                    ):
                        self.request_stop(Failure.DATA_INTEGRITY_FAILURE)
                        self._transition("GRACEFUL_STOPPING")
                        self._event("checkpoint")
                        self._transition("COMPLETED_UNREVIEWED")
                        return self._terminal("GRACEFUL_STOP", str(error))
                    self._transition("FAILED")
                    return self._terminal("FAILED_UPDATE", str(error))
                if unstable_losses([r["loss"] for r in self.trainer.metrics]):
                    self.request_stop(Failure.NUMERICAL_FAILURE)
                self._snapshot()
            self._transition("COMPLETED_UNREVIEWED")
            return self._terminal("GRACEFUL_STOP" if self.stop_reason else "COMPLETED_UNREVIEWED")
        except Exception as error:
            self.failure_category = self._classify(error)
            if self.state in ("RUNNING", "GRACEFUL_STOPPING"):
                self._transition("FAILED")
            return self._terminal("FAILED_BOUNDARY", str(error))

    def _terminal(self, outcome, error=None):
        counts = exposure(self.trainer)
        if outcome == "FAILED_UPDATE":
            counts = dict(self.last_durable_exposure)
            counts["physical_retry_replay"] = {
                "status": "FAILED_ATTEMPT_NOT_LOGICAL_EXPOSURE",
                "attempts": 1,
                "completed_microsteps_observed": max(
                    0, self.trainer.micro_step - counts["microsteps"]
                ),
                "target_bytes": None,
                "reason": "Failure may occur inside a microstep; exact consumed bytes unverified.",
            }
        result = {
            "schema": "edge-run-result-1",
            "run_id": self.run_id,
            "identity": self.identity,
            "state": self.state,
            "outcome": outcome,
            "stop_reason": self.stop_reason,
            "error": error,
            "failure_category": self.failure_category,
            "gpu_seconds": self.budget.used,
            "budget_seconds": self.budget.limit,
            "exposure": counts,
            "completed_events": dict(self.completed),
            "training_authorized_by_implementation": False,
            "quality_status": "UNVERIFIED",
        }
        if outcome.startswith("FAILED"):
            try:
                result["failure_memory"] = self.trainer._memory()
            except Exception as memory_error:
                result["failure_memory"] = {"status": "UNVERIFIED", "error": str(memory_error)}
        result["endpoint_evidence"] = self.endpoint_evidence(outcome)
        write_once(self.root / "terminal.json", result)
        return result

    def endpoint_evidence(self, outcome):
        """Bind endpoint artifacts; never choose a favorable intermediate checkpoint."""
        endpoint = self.identity["endpoint"]
        names = {
            "initial": "000000_initial_confirmation",
            "confirmation": f"{endpoint:06d}_confirmation",
            "generation": f"{endpoint:06d}_generation",
            "context": f"{endpoint:06d}_context",
            "checkpoint": f"{endpoint:06d}_checkpoint",
        }
        if outcome != "COMPLETED_UNREVIEWED" or any(
            key not in self.completed for key in names.values()
        ):
            return {"status": "INCOMPLETE", "reason": "completed endpoint artifacts required"}
        events = {
            name: read_record(self.root / "events" / f"{key}.json")["result"]
            for name, key in names.items()
        }
        initial, final = events["initial"], events["confirmation"]
        domains = {
            d: float(number(initial["domains"][d]["nll"]) - number(row["nll"]))
            for d, row in final["domains"].items()
            if d in initial["domains"]
        }
        evidence = {
            "implemented": [f"R{i}" for i in range(1, 10)],
            "lr_status": "PASS",
            "updates": endpoint,
            "data_valid": True,
            "exposure_valid": self.trainer.bytes_seen
            == sum(r["valid_target_count"] for r in self.trainer.metrics),
            "initial_nll": initial["nll"],
            "endpoint_nll": final["nll"],
            "domain_improvements": domains,
            "checkpoint_export_consistent": False,
            "stop_reason": self.stop_reason,
            "endpoint_only": True,
        }
        # Local export is deliberately a separate checked operation. Until its manifest
        # exists, a complete pilot still cannot PASS 20M-4.
        return {
            "status": "COLLECTED_UNREVIEWED",
            "promotion_inputs": evidence,
            "promotion": evaluate_promotion(evidence),
            "generation": evaluate_quality(events["generation"]["traces"], step=endpoint),
            "context_status": events["context"].get("status", "INCOMPLETE"),
            "artifact_sha256": {name: self.completed[key] for name, key in names.items()},
            "scope": self.identity["scope"],
        }

    @staticmethod
    def _classify(error, *, inside_update=False):
        from unified_edge.training.checkpoint import CheckpointError
        from unified_edge.training.data import DataIntegrityError
        from unified_edge.training.device import DeviceAdmissionError
        from unified_edge.training.optimization import NumericalTrainingError

        if isinstance(error, DataIntegrityError):
            return Failure.DATA_INTEGRITY_FAILURE.value
        if isinstance(error, DeviceAdmissionError):
            return Failure.DEVICE_GATE_FAILED.value
        if isinstance(error, CheckpointError) or isinstance(error, OSError):
            return Failure.CHECKPOINT_FAILURE.value
        if isinstance(error, NumericalTrainingError) or inside_update:
            return Failure.NUMERICAL_FAILURE.value
        return Failure.DIAGNOSTIC_FAILURE.value


def export_manifest(
    *,
    identity,
    checkpoint_sha256,
    metrics_sha256,
    generation_sha256,
    context_sha256,
    environment_hardware_sha256,
    exposure_record,
    outcome,
):
    required_identity = {
        "schema",
        "plan_sha256",
        "model_sha256",
        "data_sha256",
        "subsets_sha256",
        "anchors_sha256",
        "seed",
        "source_sha256",
        "scope",
        "authorization_sha256",
        "selected_lr_receipt_sha256",
        "training_sha256",
        "endpoint",
    }
    require(
        isinstance(identity, dict) and set(identity) == required_identity,
        "export requires complete immutable run identity",
    )
    for key, value in identity.items():
        if key.endswith("sha256"):
            digest(value)
    for key in (
        "logical_updates",
        "microsteps",
        "sequences",
        "examples",
        "valid_target_bytes",
        "tail_windows",
        "tail_bytes",
    ):
        require(
            type(exposure_record.get(key)) is int and exposure_record[key] >= 0,
            "invalid export exposure",
        )
    require(
        exposure_record["examples"] == exposure_record["sequences"]
        and exposure_record["tail_bytes"] <= exposure_record["valid_target_bytes"],
        "inconsistent export exposure",
    )
    for value in (
        checkpoint_sha256,
        metrics_sha256,
        generation_sha256,
        context_sha256,
        environment_hardware_sha256,
    ):
        digest(value)
    require(
        outcome in ("COMPLETED_UNREVIEWED", "GRACEFUL_STOP", "FAILED_UPDATE", "FAILED_BOUNDARY"),
        "invalid export outcome",
    )
    return {
        "schema": "edge-export-manifest-1",
        "identity": identity,
        "checkpoint_sha256": checkpoint_sha256,
        "metrics_sha256": metrics_sha256,
        "generation_sha256": generation_sha256,
        "context_sha256": context_sha256,
        "environment_hardware_sha256": environment_hardware_sha256,
        "exposure": exposure_record,
        "outcome": outcome,
        "upload_performed": False,
    }
