"""Bounded synthetic R4-R8 gates. No production corpus or 20M execution."""

import copy
import json
import random
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_training import assert_tree_equal, make_data

from unified_edge.config import Hardware, ModelShape, load_config
from unified_edge.dense_model import DenseByteModel
from unified_edge.resolve import ResolvedConfig, resolve_config
from unified_edge.training.checkpoint import CheckpointError, load_checkpoint, parameter_inventory
from unified_edge.training.config import TrainingConfig
from unified_edge.training.data import DataIntegrityError, DatasetManifest, VerifiedActiveData
from unified_edge.training.device import (
    DeviceAdmissionError,
    DevicePolicy,
    admit_memory,
    configure_device,
    discover_devices,
    hardware_diagnostic,
)
from unified_edge.training.experiment import EvidenceIntegrityError, tree_error
from unified_edge.training.memory import linux_rss, process_memory, training_memory
from unified_edge.training.monitoring import MonitoringMode, MonitoringPolicy
from unified_edge.training.optimization import NumericalTrainingError, build_optimizer
from unified_edge.training.trainer import Trainer

ROOT = Path(__file__).resolve().parents[1]
FAST = MonitoringPolicy(MonitoringMode.PRODUCTION_FAST)
PARANOID = MonitoringPolicy()


def tiny_model():
    return ResolvedConfig(
        ModelShape(64, 1, 8, 16, 8, 4, 2, 16, 8, 16), Hardware(), 100_000, 0.99, "0" * 64
    )


def trainer(tmp_path, mode=PARANOID, *, train=bytes(range(39)), name="run", **kwargs):
    root = tmp_path / "data"
    manifest = make_data(root, train=train)
    config = replace(TrainingConfig(), total_steps=12, warmup_steps=2, sequence_length=16)
    return Trainer(tiny_model(), config, manifest, root, tmp_path / name, monitoring=mode, **kwargs)


@pytest.fixture(autouse=True)
def process_settings():
    flags = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
    )
    state = (
        random.getstate(),
        torch.get_rng_state(),
        torch.get_num_threads(),
        torch.are_deterministic_algorithms_enabled(),
    )
    yield
    random.setstate(state[0])
    torch.set_rng_state(state[1])
    torch.set_num_threads(state[2])
    torch.use_deterministic_algorithms(state[3])
    (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
    ) = flags


@pytest.mark.parametrize("mode", [PARANOID, FAST])
def test_active_manifest_no_test_startup_resume_and_cadence(tmp_path, mode):
    run = trainer(tmp_path, mode)
    assert {d.split for d in run.manifest.documents} == {"train", "validation"}
    assert not (tmp_path / "data/test.raw").exists()
    assert run.train_data.snapshot is run.validation_data.snapshot is run.active_data
    assert run.observations["full_data_verifications"] == 1
    run.step()
    assert run.observations["full_data_verifications"] == 1 + int(mode.verify_each_update)
    path = run.save()
    assert run.observations["full_data_verifications"] == 2 + int(mode.verify_each_update)
    resumed = Trainer(
        tiny_model(),
        run.config,
        run.manifest,
        tmp_path / "data",
        tmp_path / "resume",
        monitoring=mode,
        resume_from=path,
    )
    assert resumed.observations["full_data_verifications"] == 2  # startup and restore boundary
    (tmp_path / "data/valid.raw").write_bytes(b"mutated")
    with pytest.raises(DataIntegrityError, match="mutation"):
        Trainer(
            tiny_model(),
            run.config,
            run.manifest,
            tmp_path / "data",
            tmp_path / "bad",
            monitoring=mode,
            resume_from=path,
        )


def test_test_split_is_rejected_before_any_payload_open(tmp_path, monkeypatch):
    import unified_edge.training.data as data

    def forbidden(*args):
        raise AssertionError("TEST must not be opened")

    monkeypatch.setattr(data, "file_bytes", forbidden)
    with pytest.raises(DataIntegrityError, match="excludes TEST"):
        DatasetManifest.create(tmp_path, "sealed", [("absent-test.raw", "test", "synthetic")])


@pytest.mark.parametrize("mode", [PARANOID, FAST])
def test_mutation_cannot_substitute_verified_bytes_or_publish(tmp_path, mode):
    run = trainer(tmp_path, mode)
    expected_windows = run.train_data.windows
    original = run.active_data.buffers["train.raw"]
    with pytest.raises(TypeError):
        run.active_data.buffers["train.raw"] = b"substitution"
    with pytest.raises(FrozenInstanceError):
        run.train_data.windows = ()
    (tmp_path / "data/train.raw").write_bytes(b"X" * len(original))
    if mode.verify_each_update:
        before = run.stream.state_dict()
        with pytest.raises(DataIntegrityError, match="mutation"):
            run.step()
        assert run.stream.state_dict() == before
    else:
        run.step()
        assert run.train_data.windows == expected_windows
        assert run.active_data.buffers["train.raw"] == original
    with pytest.raises(DataIntegrityError, match="mutation"):
        run.save()
    assert not (run.run_dir / "checkpoints").exists()
    run.manifest = replace(run.manifest, corpus_id="substituted")
    with pytest.raises(DataIntegrityError, match="manifest identity"):
        run.step()


def semantic_checkpoint(state):
    result = copy.deepcopy(state)
    for key in ("run_id", "elapsed_seconds", "monitoring"):
        del result[key]
    for row in result["metrics"]:
        del row["elapsed_seconds"]
    return result


def test_ten_updates_modes_gradients_checkpoints_rng_and_resume_bitwise(tmp_path):
    root = tmp_path / "data"
    manifest = make_data(root, train=bytes(range(39)))
    # A second TRAIN document exercises document boundaries and unequal tails.
    (root / "second.raw").write_bytes(b"abcdefghi")
    manifest = DatasetManifest.create(
        root,
        "tails",
        [(d.path, d.split, d.domain) for d in manifest.documents]
        + [("second.raw", "train", "synthetic")],
    )
    config = replace(TrainingConfig(), total_steps=12, warmup_steps=2, sequence_length=16)
    trajectories, checkpoints = [], []
    for mode_index, (mode, restore) in enumerate(((PARANOID, True), (FAST, True), (FAST, False))):
        run = Trainer(
            tiny_model(), config, manifest, root, tmp_path / f"mode-{mode_index}", monitoring=mode
        )
        rows, saved = [], []
        for step in range(1, 11):
            gradients = {}
            original_step = run.optimizer.step

            def observed_step():
                gradients.update(
                    {
                        n: p.grad.clone() if p.grad is not None else None
                        for n, p in run.model.named_parameters()
                    }
                )
                original_step()

            run.optimizer.step = observed_step
            row = run.step()
            run.optimizer.step = original_step
            rows.append(
                copy.deepcopy(
                    {
                        "loss": row["loss"],
                        "parameters": run.model.state_dict(),
                        "gradients": gradients,
                        "optimizer": run.optimizer.state_dict(),
                        "scheduler": run.scheduler.state_dict(),
                        "cursor": run.stream.state_dict(),
                        "python_rng": random.getstate(),
                        "torch_rng": torch.get_rng_state(),
                    }
                )
            )
            if step in (5, 10):
                path = run.save()
                saved.append(semantic_checkpoint(load_checkpoint(path)))
                if step == 5 and restore:
                    run = Trainer(
                        tiny_model(),
                        config,
                        manifest,
                        root,
                        tmp_path / f"resume-{mode_index}",
                        monitoring=mode,
                        resume_from=path,
                    )
        trajectories.append(rows)
        checkpoints.append(saved)
    assert_tree_equal(trajectories[0], trajectories[1])
    assert_tree_equal(trajectories[1], trajectories[2])
    assert_tree_equal(checkpoints[0], checkpoints[1])
    assert_tree_equal(checkpoints[1], checkpoints[2])


def test_fast_cadence_and_memory_do_not_allocate_recurrent_scratch(tmp_path, monkeypatch):
    run = trainer(tmp_path, FAST)

    def forbidden(*args):
        raise AssertionError("monitoring must not allocate recurrent scratch state")

    monkeypatch.setattr(run.model.shared, "initialize_state", forbidden)
    memory = training_memory(run.model, run.optimizer, 2)
    shape = run.model_config.shape
    assert memory["canonical_recurrent_state_bytes"] == 4 * 2 * shape.shared_layers * (
        shape.d_inner * shape.d_state + (shape.d_inner + 2 * shape.d_state) * shape.d_conv
    )
    monkeypatch.undo()
    run.train_until(11)
    assert run.observations["full_data_verifications"] == 1
    assert run.observations["memory_samples"] == run.observations["inventory_checks"] == 10
    assert FAST.detailed(100) and not FAST.detailed(99)


@pytest.mark.parametrize("mode", [PARANOID, FAST])
@pytest.mark.parametrize("fault", ["loss", "gradient", "updated_parameter"])
def test_critical_nonfinite_guards_block_checkpoint(tmp_path, mode, fault, monkeypatch):
    run = trainer(tmp_path, mode)
    if fault == "gradient":
        next(run.model.parameters()).register_hook(lambda g: torch.full_like(g, float("nan")))
    elif fault == "loss":
        original = run.model.forward
        monkeypatch.setattr(run.model, "forward", lambda x: original(x) * float("nan"))
    else:
        original = run.optimizer.step

        def bad_step():
            original()
            with torch.no_grad():
                next(run.model.parameters()).fill_(float("inf"))

        monkeypatch.setattr(run.optimizer, "step", bad_step)
    with pytest.raises(NumericalTrainingError, match="non-finite"):
        run.step()
    with pytest.raises(RuntimeError, match="failed"):
        run.save()
    assert not (run.run_dir / "checkpoints").exists()


def test_optimizer_corruption_cannot_be_published(tmp_path):
    run = trainer(tmp_path, FAST)
    run.step()
    next(iter(run.optimizer.state.values()))["exp_avg"].fill_(float("nan"))
    with pytest.raises(CheckpointError, match="AdamW"):
        run.save()
    assert not (run.run_dir / "checkpoints").exists()


def test_validation_preserves_rng_and_model_mode(tmp_path, monkeypatch):
    run = trainer(tmp_path, FAST)
    original = run.model.forward

    def noisy_diagnostic(target):
        random.random()
        torch.rand(1)
        return original(target)

    monkeypatch.setattr(run.model, "forward", noisy_diagnostic)
    before = random.getstate(), torch.get_rng_state()
    run.validate()
    assert run.model.training
    assert random.getstate() == before[0]
    assert torch.equal(torch.get_rng_state(), before[1])


def test_rss_platform_interfaces_and_units(monkeypatch):
    import unified_edge.training.memory as memory

    assert linux_rss("VmRSS: 123 kB\nVmHWM: 456 kB\n") == {
        "rss_bytes": 123 * 1024,
        "peak_rss_bytes": 456 * 1024,
    }
    for invalid in ("", "VmRSS: -1 kB", "VmRSS: 3 MB", "VmRSS: 3 kB\nVmRSS: 4 kB"):
        with pytest.raises(ValueError):
            linux_rss(invalid)
    monkeypatch.setattr(memory.sys, "platform", "win32")
    monkeypatch.setattr(
        memory, "_windows_memory", lambda: {"rss_bytes": 1024, "peak_rss_bytes": 2048}
    )
    measured = process_memory()
    assert (
        measured["supported"]
        and measured["rss_bytes"] == 1024
        and "WorkingSetSize" in measured["source"]
    )
    monkeypatch.setattr(memory.sys, "platform", "linux")
    monkeypatch.setattr(memory.Path, "read_text", lambda *a, **k: "VmRSS: 5 kB")
    assert process_memory()["rss_bytes"] == 5120
    monkeypatch.setattr(memory.sys, "platform", "unknown")
    unsupported = process_memory()
    assert not unsupported["supported"] and unsupported["rss_bytes"] is None


@pytest.mark.parametrize("count", [0, 1, 2, 4])
def test_discovery_has_no_implicit_selection(count, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: bool(count))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    monkeypatch.setattr(
        torch.cuda, "set_device", lambda *a: pytest.fail("discovery selected device")
    )
    assert discover_devices() == {"device_count": count, "cuda_available": bool(count)}
    result = hardware_diagnostic()
    assert result["selected_device"] is None and not result["execution_authorized"]
    assert result["status"] == ("NO_CUDA" if not count else "SELECTION_REQUIRED")


def test_named_profiles_selection_admission_and_diagnostic(monkeypatch):
    import unified_edge.training.device as devices

    monkeypatch.setattr(devices, "_driver_version", lambda: ("mock-driver", "MOCK"))
    t4 = DevicePolicy("20m_t4_single_fp32_pilot", 1)
    rtx = DevicePolicy("20m_rtx2050_fp32_fallback", 0)
    assert (t4.allocator_cap_bytes, t4.free_headroom_bytes) == (8 * 1024**3, 2 * 1024**3)
    assert (rtx.allocator_cap_bytes, rtx.free_headroom_bytes) == (5 * 1024**3 // 2, 1024**3 // 2)
    for p in (t4, rtx):
        admit_memory(
            p.allocator_cap_bytes + p.free_headroom_bytes,
            16 * 1024**3,
            p.allocator_cap_bytes,
            p.free_headroom_bytes,
            startup=True,
        )
        with pytest.raises(DeviceAdmissionError):
            admit_memory(
                p.allocator_cap_bytes + p.free_headroom_bytes - 1,
                16 * 1024**3,
                p.allocator_cap_bytes,
                p.free_headroom_bytes,
                startup=True,
            )
        with pytest.raises(DeviceAdmissionError):
            admit_memory(
                p.free_headroom_bytes - 1,
                16 * 1024**3,
                p.allocator_cap_bytes,
                p.free_headroom_bytes,
                startup=False,
            )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: (12 * 1024**3, 16 * 1024**3))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(name="mock T4", major=7, minor=5),
    )
    events = []
    monkeypatch.setattr(torch.cuda, "set_device", lambda d: events.append(("select", str(d))))
    monkeypatch.setattr(
        torch.cuda,
        "set_per_process_memory_fraction",
        lambda f, d: events.append(("cap", f, str(d))),
    )
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    with pytest.raises(DeviceAdmissionError, match="exactly one"):
        configure_device("cuda:0")
    with pytest.raises(ValueError, match="explicit"):
        configure_device("cuda")
    with pytest.raises(ValueError, match="match"):
        configure_device("cuda:0", t4)
    assert not events
    assert str(configure_device("cuda:1", t4)) == "cuda:1"
    assert events == [("select", "cuda:1"), ("cap", 0.5, "cuda:1")]
    before = list(events)
    diagnostic = hardware_diagnostic(t4)
    assert events == before and diagnostic["selected_device"] == "cuda:1"
    assert diagnostic["compute_capability"] == [7, 5] and diagnostic["driver"] == "mock-driver"
    assert TrainingConfig(schema="2", device="cuda:1").device == "cuda:1"
    with pytest.raises(ValueError):
        TrainingConfig(device="cuda:1")


def test_named_inventory_meta_counts_and_aliases():
    for filename, expected in (("edge_2m.yaml", 56), ("edge_20m_candidate.yaml", 128)):
        config = resolve_config(load_config(ROOT / "configs/models" / filename)).selected
        with torch.device("meta"):
            model = DenseByteModel(config)
        optimizer = build_optimizer(model, TrainingConfig())
        inventory = parameter_inventory(model, optimizer, config.sha256)
        assert inventory["unique_trainable_tensors"] == expected == len(list(model.parameters()))
        assert not optimizer.state
    tied = torch.nn.Module()
    tied.a = torch.nn.Parameter(torch.ones(2))
    tied.b = tied.a
    optimizer = build_optimizer(tied, TrainingConfig())
    assert parameter_inventory(tied, optimizer, "0" * 64)["aliases"] == {"b": "a"}
    optimizer.param_groups[0]["params"].append(tied.a)
    with pytest.raises(CheckpointError, match="exactly once"):
        parameter_inventory(tied, optimizer, "0" * 64)


@pytest.mark.parametrize(
    "mutation", ["missing", "unexpected", "architecture", "inventory", "legacy"]
)
def test_checkpoint_inventory_and_schema_fail_closed(tmp_path, mutation):
    run = trainer(tmp_path, FAST, train=b"short")
    run.step()
    path = run.save()
    state = load_checkpoint(path)
    assert len(state["optimizer"]["state"]) < len(state["parameter_inventory"]["tensors"])
    if mutation == "missing":
        state["model"].pop(next(iter(state["model"])))
    elif mutation == "unexpected":
        state["model"]["unexpected"] = torch.ones(1)
    elif mutation == "architecture":
        state["model_config"]["model"]["decoder_dim"] *= 2
    elif mutation == "inventory":
        state["parameter_inventory"]["optimizer_parameter_names"].reverse()
    else:
        state["schema"] = "2"
    with pytest.raises(CheckpointError):
        run._restore(state)


def test_critical_errors_survive_optimized_python(tmp_path):
    script = """
from unified_edge.training.experiment import tree_error, EvidenceIntegrityError
from unified_edge.training.data import DatasetManifest, DataIntegrityError
from unified_edge.training.device import DevicePolicy
from unified_edge.training.monitoring import MonitoringPolicy
from unified_edge.training.checkpoint import parameter_inventory, CheckpointError
import torch
from pathlib import Path
model = torch.nn.Linear(1, 1)
optimizer = torch.optim.AdamW([model.weight])
checks = [(lambda: tree_error({'a': 1}, {'b': 1}), EvidenceIntegrityError),
          (lambda: tree_error([1], [2]), EvidenceIntegrityError),
          (lambda: DatasetManifest.create(Path('.'), 'x',
                   [('NEVER_OPEN_TEST', 'test', 'x')]), DataIntegrityError),
          (lambda: DevicePolicy('unknown', 0), ValueError),
          (lambda: MonitoringPolicy.from_dict({'schema':'1','mode':'unsafe'}), ValueError),
          (lambda: parameter_inventory(model, optimizer, '0'*64), CheckpointError)]
for action, error in checks:
    try:
        action()
    except error:
        continue
    raise RuntimeError('optimized Python bypassed a critical gate')
print('optimized gates PASS')
"""
    result = subprocess.run(
        [sys.executable, "-B", "-O", "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "optimized gates PASS" in result.stdout
    with pytest.raises(EvidenceIntegrityError):
        tree_error({"a": 1}, {"b": 1})


def test_portable_paths_and_policy_roundtrip(tmp_path):
    run = trainer(tmp_path)
    assert str(tmp_path) not in json.dumps(run.run_manifest)
    for invalid in ("C:/data.raw", "folder\\data.raw", "../escape", "/absolute"):
        bad = replace(
            run.manifest,
            documents=(
                replace(run.manifest.documents[0], path=invalid),
                *run.manifest.documents[1:],
            ),
        )
        with pytest.raises(ValueError):
            VerifiedActiveData(bad, tmp_path / "data")
    for mode in (PARANOID, FAST):
        assert MonitoringPolicy.from_dict(mode.to_dict()) == mode
    assert not any(
        "assert " in p.read_text(encoding="utf-8")
        for p in (ROOT / "src/unified_edge/training").glob("*.py")
    )
