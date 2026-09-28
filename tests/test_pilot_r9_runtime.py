"""Tiny deterministic R9 orchestration, durability, and diagnostic isolation."""

import copy
import hashlib
import json
import math
import random
import subprocess
import sys
from dataclasses import replace

import pytest
import torch
from test_training_infrastructure_i2 import tiny_model

from unified_edge.dense_model import DenseByteModel
from unified_edge.training.config import TrainingConfig, canonical_hash
from unified_edge.training.data import DatasetManifest, Document
from unified_edge.training.experiment import tree_error
from unified_edge.training.optimization import WarmupConstant, WarmupCosine, build_optimizer
from unified_edge.training.pilot_context import (
    Anchor,
    AnchorManifest,
    evaluate_context,
    measure_anchor,
)
from unified_edge.training.pilot_data import SubsetSpan, ValidationSubsets, validate_capacity
from unified_edge.training.pilot_diagnostics import generate_panel
from unified_edge.training.pilot_gates import GATES, GateReceipt, validate_prerequisites
from unified_edge.training.pilot_plan import DOMAINS, PilotPlan
from unified_edge.training.pilot_runner import Budget, PilotRunner, exposure, subject_identity
from unified_edge.training.trainer import Trainer


@pytest.fixture(autouse=True)
def settings():
    saved = random.getstate(), torch.get_rng_state(), torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    yield
    random.setstate(saved[0])
    torch.set_rng_state(saved[1])
    torch.set_num_threads(saved[2])
    torch.use_deterministic_algorithms(deterministic)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def fixture_data(root):
    root.mkdir()
    (root / "train").write_bytes(bytes(range(39)))
    files = [("train", "train", "general_text")]
    for domain in DOMAINS:
        for i in range(16):
            name = f"{domain}-{i}"
            (root / name).write_bytes((name.encode() + bytes(range(180)))[:180])
            files.append((name, "validation", domain))
    data = DatasetManifest.create(root, "TEST_ONLY-r9", files)
    docs = [d for d in data.documents if d.split == "validation"]
    anchors = tuple(
        Anchor(
            d.path,
            d.sha256,
            d.domain,
            offset,
            1,
            sha((root / d.path).read_bytes()[offset : offset + 1]),
        )
        for d in docs
        for offset in (140, 150)
    )
    anchor_manifest = AnchorManifest("edge-anchors-1", data.sha256, anchors)
    doc = docs[0]
    raw = (root / doc.path).read_bytes()
    spans = [SubsetSpan(doc.path, doc.sha256, i, i + 40, sha(raw[i : i + 40])) for i in (0, 40, 80)]
    subsets = ValidationSubsets(
        "edge-validation-subsets-1", data.sha256, (spans[0],), (spans[1],), (spans[2],)
    )
    return data, subsets, anchor_manifest


def synthetic_receipts(subject, data, subsets, anchors, budget=86400):
    values = {
        "20M-0": {
            "parameters": 20387531,
            "tensors": 128,
            "tensor_signature": "3f242a2332fa41d4737f5d184b1593051bf8384a7cb5e954f457c136848dfd61",
            "config_roundtrip": True,
            "unclassified": 0,
        },
        "20M-1": dict.fromkeys(
            (
                "finite_loss",
                "finite_gradients",
                "parameter_participation",
                "causality",
                "batch_stream_reference_state_parity",
            ),
            True,
        ),
        "20M-2": {
            "hardware_receipt_sha256": "1" * 64,
            "phase_memory_sha256": "2" * 64,
            "allocator_headroom_pass": True,
            "target_bytes_per_second": 2000,
            "projected_seconds": 40000,
            "reserve_multiplier": 1.3,
            "same_backend_exact_resume": True,
        },
        "20M-3": {
            "fixture_hex": b"edge400: abcdef\n".hex(),
            "repetitions": 64,
            "updates": 60,
            "batch": 2,
            "accumulation": 2,
            "length": 32,
            "lr": 0.003,
            "warmup": 5,
            "train_reduction": 0.5,
            "distinct_validation_logged": True,
            "resume_step": 30,
            "exact_resume": True,
            "real_smoke_updates": 1,
            "real_smoke_finite": True,
        },
        "LR_PROBE": {"selection_receipt_sha256": "3" * 64, "selected_lr": 0.0003},
        "APPROVED_DATA": {
            "manifest_sha256": data.sha256,
            "subsets_sha256": subsets.sha256,
            "anchors_sha256": anchors.sha256,
            "review_receipt_sha256": "4" * 64,
        },
        "HUMAN_COMPUTE_APPROVAL": {
            "approval_reference": "TEST_ONLY synthetic authority",
            "gpu_seconds": budget,
            "purpose": "pilot",
        },
        "EXECUTION_CONTEXT": {
            "context_reference": "TEST_ONLY mocked context",
            "device": "cuda:0",
            "profile": "20m_t4_single_fp32_pilot",
            "single_visible_gpu": True,
            "host_boundary": "APPROVED_HOST",
            "run_registry": "TEST_ONLY",
        },
    }
    return tuple(
        GateReceipt(
            "edge-gate-1",
            gate,
            "TEST_ONLY",
            subject,
            canonical_hash(values[gate]),
            json.dumps(values[gate]),
        )
        for gate in GATES
    )


def runner_inputs(tmp_path, budget=86400):
    root = tmp_path / "data"
    data, subsets, anchors = fixture_data(root)
    plan, model = PilotPlan(), tiny_model()
    config = replace(
        TrainingConfig(), sequence_length=16, total_steps=3, warmup_steps=1, learning_rate=0.0003
    )
    subject = canonical_hash(subject_identity(plan, model, data, subsets, anchors))
    receipts = synthetic_receipts(subject, data, subsets, anchors, budget)
    return dict(
        plan=plan,
        model=model,
        config=config,
        data=data,
        data_root=root,
        subsets=subsets,
        anchors=anchors,
        receipts=receipts,
        trusted_receipt_shas=tuple(r.sha256 for r in receipts),
        selected_lr_receipt_sha256="3" * 64,
        scope="TEST_ONLY",
    )


def test_constant_probe_schedule_and_restore():
    model = DenseByteModel(tiny_model())
    cfg = PilotPlan().training_config(0.0003, "cpu", probe=True)
    optimizer = build_optimizer(model, cfg)
    schedule = WarmupConstant(optimizer, cfg)
    assert schedule.lr_at(0) == cfg.learning_rate / 32
    assert schedule.lr_at(31) == schedule.lr_at(32) == schedule.lr_at(255) == cfg.learning_rate
    for _ in range(37):
        schedule.advance()
    saved = schedule.state_dict()
    schedule.load_state_dict(saved)
    with pytest.raises(ValueError, match="compatibility"):
        WarmupCosine(optimizer, cfg).load_state_dict(saved)
    assert saved["type"] == "warmup_constant_v1"


def test_subset_identity_overlap_capacity_and_no_test(tmp_path):
    data, subsets, anchors = fixture_data(tmp_path / "data")
    assert subsets.validate(data, exact_capacity=False) == [40, 40, 40]
    assert ValidationSubsets.from_dict(subsets.to_dict()) == subsets
    anchors.validate(data)
    with pytest.raises(ValueError, match="immutable document"):
        replace(anchors.anchors[0], document=[])
    assert AnchorManifest.from_dict(anchors.to_dict()) == anchors
    with pytest.raises(ValueError, match="overlap"):
        replace(subsets, monitor=subsets.selection)
    with pytest.raises(ValueError, match="capacities"):
        subsets.validate(data)
    with pytest.raises(ValueError, match="VALIDATION"):
        replace(anchors.anchors[0], split="test")
    with pytest.raises(ValueError, match="128"):
        replace(anchors, anchors=anchors.anchors[:-1])
    docs = tuple(
        Document(f"{split}-{domain}", sha(f"{split}-{domain}".encode()), n, 1, split, domain)
        for split, quota in (
            ("train", PilotPlan().train_quotas),
            ("validation", PilotPlan().validation_quotas),
        )
        for domain, n in zip(DOMAINS, quota, strict=True)
    )
    assert validate_capacity(DatasetManifest("1", "metadata-only", docs))["train"] == 65536000
    with pytest.raises(ValueError):
        validate_capacity(replace(data, corpus_id="wrong-capacity"))


def test_gates_test_only_cannot_authorize_real(tmp_path):
    kwargs = runner_inputs(tmp_path)
    receipts = kwargs["receipts"]
    subject = receipts[0].subject_sha256
    trusted = kwargs["trusted_receipt_shas"]
    assert (
        validate_prerequisites(receipts, subject, trusted_receipt_shas=trusted, scope="TEST_ONLY")[
            "status"
        ]
        == "PASS"
    )
    with pytest.raises(ValueError, match="TEST_ONLY"):
        validate_prerequisites(receipts, subject, trusted_receipt_shas=trusted)
    with pytest.raises(ValueError, match="trusted"):
        validate_prerequisites(receipts, subject, scope="TEST_ONLY")
    assert (
        validate_prerequisites(
            receipts[:-1], subject, trusted_receipt_shas=trusted, scope="TEST_ONLY"
        )["status"]
        == "INCOMPLETE"
    )


def cheap_panels(monkeypatch):
    import unified_edge.training.pilot_runner as module

    calls = []

    def generation(model, plan):
        random.random()
        torch.rand(2)
        calls.append("generation")
        return [{"scope": "TEST_ONLY"}]

    def context(trainer, manifest):
        random.random()
        torch.rand(2)
        calls.append("context")
        return {"scope": "TEST_ONLY"}

    monkeypatch.setattr(module, "generate_panel", generation)
    monkeypatch.setattr(module, "context_panel", context)
    return calls


def semantic(trainer):
    return {
        "weights": copy.deepcopy(trainer.model.state_dict()),
        "optimizer": copy.deepcopy(trainer.optimizer.state_dict()),
        "scheduler": trainer.scheduler.state_dict(),
        "cursor": trainer.stream.state_dict(),
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "metrics": [
            {k: v for k, v in r.items() if k != "elapsed_seconds"} for r in trainer.metrics
        ],
    }


def test_cadence_rng_next_update_duplicate_and_exposure(tmp_path, monkeypatch):
    kwargs = runner_inputs(tmp_path)
    calls = cheap_panels(monkeypatch)
    direct = Trainer(
        kwargs["model"], kwargs["config"], kwargs["data"], kwargs["data_root"], tmp_path / "direct"
    )
    direct.train_until(3)
    expected = semantic(direct)
    runner = PilotRunner(tmp_path / "runs", **kwargs)
    try:
        result = runner.run()
        assert result["outcome"] == "COMPLETED_UNREVIEWED", result
        assert tree_error(semantic(runner.trainer), expected) == 0
        e = exposure(runner.trainer)
        assert e["valid_target_bytes"] < 3 * 4 * 16
        assert e["tail_windows"] > 0 and e["tail_bytes"] == e["tail_windows"] * 7
        assert e["microsteps"] == 6 and e["sequences"] == 12
        assert calls == ["generation", "context", "generation", "context"]
        with pytest.raises(FileExistsError):
            PilotRunner(tmp_path / "runs", **kwargs)
    finally:
        runner.close()


class Interrupted(BaseException):
    pass


@pytest.mark.parametrize(
    "point",
    [
        "before_event",
        "after_event",
        "before_checkpoint_publication",
        "after_checkpoint_publication",
    ],
)
def test_interruption_exact_resume_events(tmp_path, monkeypatch, point):
    kwargs = runner_inputs(tmp_path)
    cheap_panels(monkeypatch)
    control = PilotRunner(tmp_path / "control", **kwargs)
    control_result = control.run()
    expected = semantic(control.trainer)
    control.close()
    fired = []

    def interrupt(run):
        if run.trainer.global_step > 0 and not fired:
            fired.append(True)
            raise Interrupted()

    runner = PilotRunner(tmp_path / "interrupt", **kwargs, hooks={point: interrupt})
    with pytest.raises(Interrupted):
        try:
            runner.run()
        finally:
            runner.close()
    resumed = PilotRunner(tmp_path / "interrupt", **kwargs, resume=True)
    try:
        result = resumed.run()
        assert result["outcome"] == "COMPLETED_UNREVIEWED", result
        assert tree_error(semantic(resumed.trainer), expected) == 0
        assert set(result["completed_events"]) == set(control_result["completed_events"])
        assert len(list((resumed.root / "updates").glob("*.json"))) == 3
    finally:
        resumed.close()


def test_mocked_budget_and_safe_user_stop(tmp_path, monkeypatch):
    kwargs = runner_inputs(tmp_path, budget=1)
    cheap_panels(monkeypatch)
    ticks = iter(range(100))
    runner = PilotRunner(tmp_path / "runs", **kwargs, clock=lambda: next(ticks))
    try:
        result = runner.run()
        assert result["outcome"] == "GRACEFUL_STOP"
        assert result["stop_reason"] == "BUDGET_EXCEEDED"
        assert result["exposure"]["logical_updates"] == 0
    finally:
        runner.close()
    budget = Budget(3, clock=iter((0, 3)).__next__)
    with budget.account():
        pass
    assert budget.exhausted and budget.used == 3


def test_dry_run_cli_never_constructs_trainer():
    result = subprocess.run(
        [sys.executable, "scripts/edge_pilot.py", "plan"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    data = json.loads(result.stdout)
    assert not data["execution_started"] and not data["execution_authorized"]
    assert data["plan_sha256"] == PilotPlan().sha256


def test_real_tiny_context_measurement_preserves_rng_weights_gradients(tmp_path):
    _, _, anchors = fixture_data(tmp_path / "data")
    model = DenseByteModel(tiny_model())
    before = copy.deepcopy(model.state_dict())
    py, cpu = random.getstate(), torch.get_rng_state()
    anchor = anchors.anchors[0]
    rows = measure_anchor(model, anchor, (tmp_path / "data" / anchor.document).read_bytes())
    assert len(rows) == 8 and all(math.isfinite(r["context_gradient_norm"]) for r in rows)
    assert tree_error(model.state_dict(), before) == 0
    assert all(p.grad is None for p in model.parameters())
    assert random.getstate() == py and torch.equal(torch.get_rng_state(), cpu)


def test_context_complete_panel_uses_document_clusters(tmp_path):
    _, _, anchors = fixture_data(tmp_path / "data")
    rows = []
    for index, a in enumerate(anchors.anchors):
        for phase in range(8):
            rows.append(
                dict(
                    anchor_sha256=a.sha256,
                    phase=phase,
                    short_nll=1.01,
                    long_nll=1.0,
                    shuffled_nll=1.005,
                    byte_tv=0.000011 if index < 96 else 0.0,
                    shared_delta=1.0,
                    conv_delta=1.0,
                    ssm_delta=1.0,
                    pre_tanh_delta=1.0,
                    hidden_delta=1.0,
                    logit_delta=1.0,
                    tanh_saturation=0.0,
                    context_gradient_norm=1.0,
                )
            )
    result = evaluate_context(rows, anchors)
    assert result["status"] == "PASS"
    assert result["summary"]["responsive_fraction"] == 0.75
    assert result["summary"]["history"]["documents"] == 64


def test_actual_diagnostics_preserve_next_tiny_update(tmp_path):
    kwargs = runner_inputs(tmp_path)
    args = (kwargs["model"], kwargs["config"], kwargs["data"], kwargs["data_root"])
    direct = Trainer(*args, tmp_path / "direct")
    direct.step()
    direct.step()
    expected = semantic(direct)
    observed = Trainer(*args, tmp_path / "observed")
    observed.step()
    observed.validate()
    traces = generate_panel(observed.model)
    assert len(traces) == 45
    anchor = kwargs["anchors"].anchors[0]
    measure_anchor(observed.model, anchor, observed.active_data.buffers[anchor.document])
    observed.step()
    assert tree_error(semantic(observed), expected) == 0


def test_failed_update_does_not_publish_partial_logical_exposure(tmp_path, monkeypatch):
    from unified_edge.training.optimization import NumericalTrainingError

    kwargs = runner_inputs(tmp_path)
    cheap_panels(monkeypatch)
    runner = PilotRunner(tmp_path / "runs", **kwargs)

    def fail(*args):
        raise NumericalTrainingError("injected loss failure")

    monkeypatch.setattr(runner.trainer.model, "forward", fail)
    # Skip scheduled diagnostics solely to inject the fault inside Trainer.step.
    monkeypatch.setattr(runner, "_event", lambda name: None)
    try:
        result = runner.run()
        assert result["outcome"] == "FAILED_UPDATE"
        assert result["exposure"]["logical_updates"] == 0
        assert result["exposure"]["microsteps"] == 0
        assert result["exposure"]["physical_retry_replay"]["target_bytes"] is None
        assert runner.trainer._failed
        with pytest.raises(RuntimeError):
            runner.trainer.step()
    finally:
        runner.close()
    with pytest.raises(ValueError, match="ambiguous unfinished update"):
        PilotRunner(tmp_path / "runs", **kwargs, resume=True)


def test_missing_completed_event_fails_closed(tmp_path, monkeypatch):
    kwargs = runner_inputs(tmp_path)
    cheap_panels(monkeypatch)
    runner = PilotRunner(tmp_path / "runs", **kwargs)

    def interrupt(run):
        if run.trainer.global_step == 1:
            raise Interrupted()

    runner.hooks["after_checkpoint_publication"] = interrupt
    try:
        with pytest.raises(Interrupted):
            runner.run()
    finally:
        runner.close()
    (runner.root / "events/000000_monitor.json").unlink()
    with pytest.raises(FileNotFoundError):
        PilotRunner(tmp_path / "runs", **kwargs, resume=True)


def test_probe_runner_fresh_arms_constant_schedule_and_budget(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import unified_edge.training.pilot_probe as probe

    kwargs = runner_inputs(tmp_path)
    receipts = []
    for r in kwargs["receipts"]:
        if r.gate == "LR_PROBE":
            continue
        if r.gate == "HUMAN_COMPUTE_APPROVAL":
            p = json.loads(r.evidence_json)
            p.update(purpose="lr_probe", gpu_seconds=10800)
            r = replace(r, evidence_json=json.dumps(p), evidence_sha256=canonical_hash(p))
        receipts.append(r)
    created = []

    class FakeTrainer:
        def __init__(self, model, config, *args, **options):
            assert options["scheduler_type"] == "warmup_constant_v1"
            assert config.total_steps == 256 and config.warmup_steps == 32 and config.seed == 17
            torch.manual_seed(config.seed)
            self.model = torch.nn.Linear(1, 1)
            self.stream = SimpleNamespace(state_dict=lambda: {"seed": 17, "order": [0, 1, 2]})
            self.global_step = 0
            self.metrics = []
            self.bytes_seen = 0
            self.device = torch.device("cpu")
            created.append(self)

        def step(self):
            self.global_step += 1
            self.bytes_seen += 8192
            self.metrics.append({"loss": 1.0, "clipped": False})

        def save(self):
            pass

    monkeypatch.setattr(probe, "Trainer", FakeTrainer)
    monkeypatch.setattr(
        probe,
        "evaluate_subset",
        lambda trainer, spans: {"nll": 1.0 if trainer.global_step == 0 else 0.9},
    )
    args = {k: kwargs[k] for k in ("model", "data", "data_root", "subsets", "anchors", "scope")}
    args.update(receipts=tuple(receipts), trusted_receipt_shas=tuple(r.sha256 for r in receipts))
    result = probe.run_lr_probe(tmp_path / "probes", **args, clock=lambda: 0.0)
    assert result["status"] == "PASS" and result["selected_lr"] == 0.0003
    assert len(created) == 3 and all(t.global_step == 256 for t in created)
    assert len({probe.parameter_hash(t.model) for t in created}) == 1
    with pytest.raises(FileExistsError):
        probe.run_lr_probe(tmp_path / "probes", **args, clock=lambda: 0.0)
    ticks = iter(range(0, 100000, 10801))
    failed = probe.run_lr_probe(tmp_path / "budget-probes", **args, clock=lambda: next(ticks))
    assert failed["status"] == "STOP" and failed["reason"] == "BUDGET_EXCEEDED"


def test_cooperative_user_stop_and_export_contract(tmp_path, monkeypatch):
    import importlib.util
    import signal
    from pathlib import Path

    from unified_edge.training.pilot_runner import export_manifest

    spec = importlib.util.spec_from_file_location(
        "edge_pilot", Path(__file__).parents[1] / "scripts/edge_pilot.py"
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    kwargs = runner_inputs(tmp_path)
    cheap_panels(monkeypatch)
    runner = PilotRunner(tmp_path / "runs", **kwargs)
    previous = signal.getsignal(signal.SIGINT)
    try:
        with cli.cooperative_stops(runner):
            signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
            result = runner.run()
        assert signal.getsignal(signal.SIGINT) == previous
        assert result["outcome"] == "GRACEFUL_STOP" and result["stop_reason"] == "USER_STOP"
        assert result["exposure"]["logical_updates"] == 0
        manifest = export_manifest(
            identity=runner.identity,
            checkpoint_sha256="a" * 64,
            metrics_sha256="b" * 64,
            generation_sha256="c" * 64,
            context_sha256="d" * 64,
            environment_hardware_sha256="e" * 64,
            exposure_record=result["exposure"],
            outcome=result["outcome"],
        )
        assert manifest["identity"]["plan_sha256"] == PilotPlan().sha256
        assert not manifest["upload_performed"]
    finally:
        runner.close()
