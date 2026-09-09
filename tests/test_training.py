import copy
import hashlib
import json
import math
import random
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from unified_edge.resolve import ResolvedConfig
from unified_edge.training.checkpoint import CheckpointError, load_checkpoint, save_checkpoint
from unified_edge.training.config import TrainingConfig
from unified_edge.training.data import (
    BatchStream,
    DatasetManifest,
    WindowDataset,
    create_tiny_fixture,
)
from unified_edge.training.optimization import WarmupCosine, build_optimizer, raw_byte_nll
from unified_edge.training.trainer import Trainer


@pytest.fixture(scope="module", autouse=True)
def restore_process_settings():
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    py_rng, torch_rng = random.getstate(), torch.get_rng_state()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(deterministic)
    random.setstate(py_rng)
    torch.set_rng_state(torch_rng)


def model_config():
    return ResolvedConfig.from_dict(
        json.loads(
            (Path(__file__).resolve().parents[1] / "reports/edge_2m_resolved.json").read_text()
        )
    )


def make_data(root, train=bytes(range(132)), valid=b"held out\xff\x80\x00\xc0" * 5):
    root.mkdir()
    (root / "train.raw").write_bytes(train)
    (root / "valid.raw").write_bytes(valid)
    return DatasetManifest.create(
        root,
        "test-binary-v1",
        [
            ("train.raw", "train", "binary"),
            ("valid.raw", "validation", "binary"),
        ],
    )


def small_config(**kwargs):
    return replace(TrainingConfig(), total_steps=4, warmup_steps=1, **kwargs)


def assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor)
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_tree_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            assert_tree_equal(x, y)
    else:
        assert a == b


def semantic_metrics(records):
    return [{k: v for k, v in row.items() if k != "elapsed_seconds"} for row in records]


def test_manifest_binary_roundtrip_paths_mutation_and_splits(tmp_path):
    root = tmp_path / "data"
    manifest = make_data(root, bytes(range(256)) + b"\x00\xff\xc0\x80" * 7)
    encoded = json.loads(json.dumps(manifest.to_dict()))
    restored = DatasetManifest.from_dict(encoded)
    assert restored.sha256 == manifest.sha256
    assert str(tmp_path) not in json.dumps(encoded)
    for size in (32, 64, 128, 256):
        dataset = WindowDataset(restored, root, "train", size)
        assert b"".join(w.payload for w in dataset.windows) == (root / "train.raw").read_bytes()
        assert all(0 < len(w.payload) <= size for w in dataset.windows)
    (root / "train.raw").write_bytes(b"changed")
    with pytest.raises(ValueError, match="mutation"):
        manifest.verify(root)
    with pytest.raises(ValueError, match="outside|relative|inside"):
        DatasetManifest.create(root, "bad", [("../outside", "train", "binary")])
    with pytest.raises(ValueError, match="identical"):
        identical_manifest(root)
    encoded["extra"] = True
    with pytest.raises(ValueError, match="fields"):
        DatasetManifest.from_dict(encoded)


def identical_manifest(root):
    (root / "valid-copy.raw").write_bytes((root / "valid.raw").read_bytes())
    return DatasetManifest.create(
        root,
        "bad",
        [
            ("valid.raw", "train", "binary"),
            ("valid-copy.raw", "validation", "binary"),
        ],
    )


def test_deterministic_order_cursor_and_epoch_tail(tmp_path):
    root = tmp_path / "data"
    manifest = make_data(root)
    dataset = WindowDataset(manifest, root, "train", 32)
    a, b = BatchStream(dataset, 37, 2), BatchStream(dataset, 37, 2)
    for _ in range(7):
        assert a.next_batch() == b.next_batch()
    state = a.state_dict()
    restored = BatchStream(dataset, 37, 2)
    restored.load_state_dict(state)
    assert restored.next_batch() == a.next_batch()
    state["order"] = [0] * len(dataset)
    with pytest.raises(ValueError, match="ordering"):
        restored.load_state_dict(state)


def test_training_config_and_objective_alignment():
    config = TrainingConfig()
    assert TrainingConfig.from_dict(config.to_dict()) == config
    for mutation in (
        {"seed": True},
        {"sequence_length": 4096},
        {"learning_rate": float("nan")},
        {"warmup_steps": 60},
        {"unknown": 1},
        {"minimum_lr_ratio": 2},
    ):
        with pytest.raises(ValueError):
            TrainingConfig.from_dict({**config.to_dict(), **mutation})
    targets = torch.tensor([[0, 1, 255, 128]])
    logits = torch.full((1, 4, 267), -8.0)
    logits.scatter_(-1, targets.unsqueeze(-1), 8.0)
    logits[:, :, 256:258] = -torch.inf
    loss, count = raw_byte_nll(logits, targets)
    assert count == 4 and loss.item() < 0.001
    expected = -F.log_softmax(logits, -1).gather(-1, targets.unsqueeze(-1)).sum()
    torch.testing.assert_close(loss, expected)
    for value in (256, 257, 258, -1):
        with pytest.raises(ValueError, match="rejects"):
            raw_byte_nll(logits, torch.full_like(targets, value))


def test_optimizer_identity_coverage_and_schedule(tmp_path):
    root = tmp_path / "data"
    manifest = make_data(root)
    trainer = Trainer(model_config(), small_config(), manifest, root, tmp_path / "run")
    optimizer = trainer.optimizer
    actual = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(actual) == len({id(p) for p in actual}) == 56
    assert {id(p) for p in actual} == {id(p) for p in trainer.model.parameters() if p.requires_grad}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            no_decay = getattr(parameter, "_no_weight_decay", False) or parameter.ndim < 2
            assert group["weight_decay"] == (0.0 if no_decay else trainer.config.weight_decay)
    config = replace(TrainingConfig(), total_steps=6, warmup_steps=2)
    schedule = WarmupCosine(build_optimizer(trainer.model, config), config)
    values = []
    for _ in range(6):
        values.append(schedule.optimizer.param_groups[0]["lr"])
        schedule.advance()
    assert values[:3] == [config.learning_rate / 2, config.learning_rate, config.learning_rate]
    assert values[-1] == pytest.approx(config.learning_rate * config.minimum_lr_ratio)
    assert values[2:] == sorted(values[2:], reverse=True)


def test_accumulation_matches_byte_weighted_effective_batch(tmp_path):
    root = tmp_path / "data"
    manifest = make_data(root, train=bytes(range(39)))
    a = Trainer(model_config(), small_config(batch_size=1), manifest, root, tmp_path / "a")
    b = Trainer(model_config(), small_config(accumulation_steps=1), manifest, root, tmp_path / "b")
    row_a, row_b = a.step(), b.step()
    assert row_a["micro_step"] == 2 and row_b["micro_step"] == 1
    assert row_a["step"] == row_b["step"] == 1
    assert row_a["valid_target_count"] == row_b["valid_target_count"] == 39
    assert row_a["raw_byte_nll"] == pytest.approx(row_b["raw_byte_nll"], abs=1e-6)
    assert row_a["clipped"] and row_a["gradient_norm"] > a.config.clip_norm
    for x, y in zip(a.model.parameters(), b.model.parameters(), strict=True):
        torch.testing.assert_close(x, y, atol=1e-5, rtol=1e-4)
    assert all(p.grad is None for p in a.model.parameters())


def test_validation_weighting_mode_and_optimizer_immutability(tmp_path):
    root = tmp_path / "data"
    manifest = make_data(root)
    trainer = Trainer(model_config(), small_config(), manifest, root, tmp_path / "run")
    trainer.step()
    before = copy.deepcopy(trainer.optimizer.state_dict())
    cursor = trainer.stream.state_dict()
    expected_sum, count = 0.0, 0
    with torch.inference_mode():
        for window in trainer.validation_data.windows:
            targets = torch.tensor([list(window.payload)])
            expected_sum += F.cross_entropy(
                trainer.model(targets).flatten(0, 1), targets.flatten(), reduction="sum"
            ).item()
            count += targets.numel()
    measured = trainer.validate()
    assert measured["raw_byte_nll"] == pytest.approx(expected_sum / count, abs=1e-6)
    assert measured["raw_byte_perplexity"] == math.exp(measured["raw_byte_nll"])
    assert measured["valid_target_count"] == count
    assert trainer.model.training
    assert trainer.validate() == measured
    assert_tree_equal(before, trainer.optimizer.state_dict())
    assert cursor == trainer.stream.state_dict()


def test_continuous_and_destroyed_restored_training_are_exact(tmp_path):
    root = tmp_path / "data"
    manifest = make_data(root)
    config = small_config()
    continuous = Trainer(model_config(), config, manifest, root, tmp_path / "continuous")
    continuous.train_until(2)
    random.random()
    torch.rand(3)
    random.random()
    torch.rand(4)
    continuous.train_until(4)
    expected_parameters = copy.deepcopy(continuous.model.state_dict())
    expected_optimizer = copy.deepcopy(continuous.optimizer.state_dict())
    expected_schedule = continuous.scheduler.state_dict()
    expected_cursor = continuous.stream.state_dict()
    expected_metrics = semantic_metrics(continuous.metrics)
    expected_counts = continuous.micro_step, continuous.examples_seen, continuous.bytes_seen
    expected_final_rng = random.getstate(), torch.get_rng_state()
    expected_next = continuous.stream.next_batch()
    del continuous
    interrupted = Trainer(model_config(), config, manifest, root, tmp_path / "interrupted")
    interrupted.train_until(2)
    random.random()
    torch.rand(3)
    path = interrupted.save()
    expected_rng = random.random(), torch.rand(4)
    del interrupted
    resumed = Trainer(
        model_config(), config, manifest, root, tmp_path / "resumed", resume_from=path
    )
    assert random.random() == expected_rng[0]
    assert torch.equal(torch.rand(4), expected_rng[1])
    resumed.train_until(4)
    assert random.getstate() == expected_final_rng[0]
    assert torch.equal(torch.get_rng_state(), expected_final_rng[1])
    assert_tree_equal(resumed.model.state_dict(), expected_parameters)
    assert_tree_equal(resumed.optimizer.state_dict(), expected_optimizer)
    assert resumed.scheduler.state_dict() == expected_schedule
    assert resumed.stream.state_dict() == expected_cursor
    assert semantic_metrics(resumed.metrics) == expected_metrics
    assert (resumed.micro_step, resumed.examples_seen, resumed.bytes_seen) == expected_counts
    assert resumed.stream.next_batch() == expected_next


def test_atomic_checkpoint_and_corruption_and_compatibility(tmp_path, monkeypatch):
    root = tmp_path / "data"
    manifest = make_data(root)
    trainer = Trainer(model_config(), small_config(), manifest, root, tmp_path / "run")
    trainer.step()
    path = trainer.save()
    state = load_checkpoint(path)
    with pytest.raises(FileExistsError):
        trainer.save()
    original_hash = hashlib.sha256((path / "state.pt").read_bytes()).hexdigest()
    import unified_edge.training.checkpoint as checkpoint_module

    def reject_rename(*args):
        raise OSError("simulated interrupted publication")

    monkeypatch.setattr(checkpoint_module.os, "rename", reject_rename)
    with pytest.raises(OSError, match="publication"):
        save_checkpoint(path.parent / "new-checkpoint", state)
    assert not (path.parent / "new-checkpoint").exists()
    assert not list(path.parent.glob(".checkpoint-*"))
    assert hashlib.sha256((path / "state.pt").read_bytes()).hexdigest() == original_hash
    with pytest.raises(CheckpointError, match="training_config"):
        Trainer(
            model_config(),
            replace(trainer.config, learning_rate=0.001),
            manifest,
            root,
            tmp_path / "bad-lr",
            resume_from=path,
        )
    with pytest.raises(CheckpointError, match="model_config"):
        Trainer(
            replace(model_config(), authored_sha256="0" * 64),
            trainer.config,
            manifest,
            root,
            tmp_path / "bad-model",
            resume_from=path,
        )
    (path / "state.pt").write_bytes(b"corrupted")
    with pytest.raises(CheckpointError, match="SHA-256"):
        load_checkpoint(path)


def test_strict_resume_counters_optimizer_rng_and_schema(tmp_path, monkeypatch):
    root = tmp_path / "data"
    manifest = make_data(root)
    trainer = Trainer(model_config(), small_config(), manifest, root, tmp_path / "run")
    trainer.step()
    path = trainer.save()
    state = load_checkpoint(path)
    mutations = [
        lambda s: s.update(schema="99"),
        lambda s: s.update(dataset_sha256="wrong"),
        lambda s: s.update(accumulation_position=1),
        lambda s: s.update(optimizer_type="SGD"),
        lambda s: s["data_cursor"].update(offset=0),
        lambda s: s.update(torch_cpu_rng=torch.zeros(4)),
        lambda s: s["optimizer"]["param_groups"][0].update(weight_decay=0.123),
        lambda s: s["scheduler"].update(completed=0),
        lambda s: s.update(elapsed_seconds=float("nan")),
        lambda s: s["metrics"][0].update(step=999),
        lambda s: s["optimizer"]["state"][0]["exp_avg_sq"].fill_(-1),
        lambda s: s["optimizer"]["state"].pop(0),
        lambda s: s["optimizer_update_counts"].__setitem__(0, 0),
        lambda s: s["optimizer_update_counts"].__setitem__(0, s["global_step"] + 1),
        lambda s: s["optimizer_update_counts"].pop(),
    ]
    import unified_edge.training.trainer as trainer_module

    for index, mutate in enumerate(mutations):
        bad = copy.deepcopy(state)
        mutate(bad)
        monkeypatch.setattr(trainer_module, "load_checkpoint", lambda _: bad)
        with pytest.raises((CheckpointError, ValueError)):
            Trainer(
                model_config(),
                trainer.config,
                manifest,
                root,
                tmp_path / f"invalid-{index}",
                resume_from=path,
            )


def test_run_identity_and_mutated_corpus_and_failed_gradient(tmp_path):
    root = tmp_path / "data"
    manifest = make_data(root)
    trainer = Trainer(model_config(), small_config(), manifest, root, tmp_path / "run")
    identity_bytes = (trainer.run_dir / "run.json").read_bytes()
    identity = json.loads(identity_bytes)
    assert identity["parameter_count"] == 1929579
    assert identity["dataset_sha256"] == manifest.sha256
    assert str(tmp_path) not in identity_bytes.decode()
    with pytest.raises(FileExistsError):
        Trainer(model_config(), trainer.config, manifest, root, trainer.run_dir)
    parameter = next(trainer.model.parameters())
    hook = parameter.register_hook(lambda grad: torch.full_like(grad, float("nan")))
    with pytest.raises(ValueError, match="non-finite gradient"):
        trainer.step()
    hook.remove()
    with pytest.raises(RuntimeError, match="failed"):
        trainer.save()
    assert (trainer.run_dir / "run.json").read_bytes() == identity_bytes
    (root / "train.raw").write_bytes(b"mutated")
    with pytest.raises(ValueError, match="mutation"):
        Trainer(model_config(), trainer.config, manifest, root, tmp_path / "bad-data")


def test_fixed_fixture_has_no_shared_windows(tmp_path):
    manifest = create_tiny_fixture(tmp_path / "tiny")
    train = WindowDataset(manifest, tmp_path / "tiny", "train", 32)
    valid = WindowDataset(manifest, tmp_path / "tiny", "validation", 32)
    assert sum(len(w.payload) for w in train.windows) == 1024
    assert sum(len(w.payload) for w in valid.windows) == 256
    assert not {w.payload for w in train.windows} & {w.payload for w in valid.windows}


def test_nonfinite_parameters_poison_update_before_data_is_consumed(tmp_path):
    root = tmp_path / "data"
    manifest = make_data(root)
    trainer = Trainer(model_config(), small_config(), manifest, root, tmp_path / "run")
    cursor = trainer.stream.state_dict()
    with torch.no_grad():
        next(trainer.model.parameters()).fill_(float("nan"))
    with pytest.raises(ValueError, match="before update.*non-finite parameter"):
        trainer.step()
    assert trainer.stream.state_dict() == cursor
    with pytest.raises(RuntimeError, match="failed"):
        trainer.step()


def test_nonfinite_loss_and_post_update_parameters_are_rejected(tmp_path, monkeypatch):
    root = tmp_path / "data"
    manifest = make_data(root)
    trainer = Trainer(model_config(), small_config(), manifest, root, tmp_path / "run")
    with pytest.raises(ValueError, match="non-finite raw-byte loss"):
        raw_byte_nll(torch.full((1, 1, 267), float("nan")), torch.zeros(1, 1, dtype=torch.long))
    original = trainer.optimizer.step

    def corrupt_update():
        original()
        with torch.no_grad():
            next(trainer.model.parameters()).fill_(float("inf"))

    monkeypatch.setattr(trainer.optimizer, "step", corrupt_update)
    with pytest.raises(ValueError, match="after update.*non-finite parameter"):
        trainer.step()
    with pytest.raises(RuntimeError, match="failed"):
        trainer.save()


@pytest.mark.parametrize(
    "payload", [b"short", bytes(range(39))], ids=["lazy-state", "lagging-count"]
)
def test_resume_supports_lazy_adamw_state_and_per_parameter_update_counts(tmp_path, payload):
    root = tmp_path / "data"
    manifest = make_data(root, train=payload)
    config = small_config(batch_size=1, accumulation_steps=1)
    original = Trainer(model_config(), config, manifest, root, tmp_path / "original")
    original.train_until(2)
    optimizer_state = original.optimizer.state_dict()["state"]
    if len(payload) < 8:
        assert len(optimizer_state) < len(list(original.model.parameters()))
    else:
        assert {v["step"].item() for v in optimizer_state.values()} == {1, 2}
    checkpoint = original.save()
    resumed = Trainer(
        model_config(), config, manifest, root, tmp_path / "resumed", resume_from=checkpoint
    )
    original.step()
    resumed.step()
    assert_tree_equal(original.model.state_dict(), resumed.model.state_dict())
    assert_tree_equal(original.optimizer.state_dict(), resumed.optimizer.state_dict())
    assert_tree_equal(original.scheduler.state_dict(), resumed.scheduler.state_dict())
    assert original.stream.state_dict() == resumed.stream.state_dict()
    assert semantic_metrics(original.metrics) == semantic_metrics(resumed.metrics)
