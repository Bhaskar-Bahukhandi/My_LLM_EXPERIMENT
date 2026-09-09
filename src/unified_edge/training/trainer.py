"""Small explicit trainer with update-boundary checkpoint and deterministic resume."""

import json
import math
import platform
import random
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch

from unified_edge.dense_model import DenseByteModel
from unified_edge.parameters import audit_parameters
from unified_edge.resolve import ResolvedConfig
from unified_edge.training.checkpoint import CheckpointError, load_checkpoint, save_checkpoint
from unified_edge.training.config import TrainingConfig, canonical_hash
from unified_edge.training.data import (
    BatchStream,
    DatasetManifest,
    WindowDataset,
    batches_by_length,
)
from unified_edge.training.memory import training_memory
from unified_edge.training.optimization import (
    WarmupCosine,
    build_optimizer,
    finite_gradients,
    finite_parameters,
    raw_byte_nll,
)


def code_identity() -> dict:
    root = Path(__file__).resolve().parents[3]
    command = ["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root)]
    commit = subprocess.run(
        [*command, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    dirty = subprocess.run(
        [*command, "status", "--porcelain"], capture_output=True, text=True, check=True
    )
    sources = {
        p.relative_to(root).as_posix(): p.read_text(encoding="utf-8")
        for p in sorted((root / "src").rglob("*.py"))
    }
    return {
        "commit": commit.stdout.strip(),
        "dirty": bool(dirty.stdout),
        "source_sha256": canonical_hash(sources),
    }


def environment(config: TrainingConfig) -> dict:
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "device": "cpu",
        "dtype": "float32",
        "threads": config.cpu_threads,
        "deterministic_algorithms": True,
    }


class Trainer:
    def __init__(
        self,
        model_config: ResolvedConfig,
        config: TrainingConfig,
        manifest: DatasetManifest,
        data_root: Path,
        run_dir: Path,
        *,
        resume_from: Path | None = None,
    ):
        if not isinstance(model_config, ResolvedConfig) or not isinstance(config, TrainingConfig):
            raise TypeError("trainer requires validated model and training configurations")
        self.config, self.model_config, self.manifest = config, model_config, manifest
        self.run_dir = Path(run_dir)
        if self.run_dir.exists():
            raise FileExistsError(f"run already exists: {self.run_dir.name}")
        self.train_data = WindowDataset(manifest, data_root, "train", config.sequence_length)
        self.validation_data = WindowDataset(
            manifest, data_root, "validation", config.sequence_length
        )
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.set_num_threads(config.cpu_threads)
        torch.use_deterministic_algorithms(True)
        self.model = DenseByteModel(model_config)
        self.optimizer = build_optimizer(self.model, config)
        self.scheduler = WarmupCosine(self.optimizer, config)
        self.stream = BatchStream(self.train_data, config.seed, config.batch_size)
        self.global_step = self.micro_step = self.examples_seen = self.bytes_seen = 0
        self.metrics = []
        self._updating = self._failed = False
        self._elapsed_offset = 0.0
        self.code, self.env = code_identity(), environment(config)
        self._started = time.perf_counter()
        parent = None
        if resume_from is not None:
            saved = load_checkpoint(resume_from)
            self._restore(saved)
            parent = {
                "run_id": saved["run_id"],
                "step": saved["global_step"],
                "checkpoint_manifest": json.loads(
                    (Path(resume_from) / "manifest.json").read_text()
                ),
            }
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.run_manifest = {
            "schema": "1",
            "run_id": f"{self.run_dir.name}-{uuid.uuid4().hex}",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "code": self.code,
            "resolved_sha256": model_config.sha256,
            "training_sha256": config.sha256,
            "dataset_sha256": manifest.sha256,
            "environment": self.env,
            "seed": config.seed,
            "parameter_count": audit_parameters(self.model)["unique_trainable_parameters"],
            "resumed_from": parent,
        }
        with (self.run_dir / "run.json").open("x", encoding="utf-8") as handle:
            json.dump(self.run_manifest, handle, indent=2, allow_nan=False)
        self._started = time.perf_counter()

    def _structure(self):
        return [
            {
                "name": name,
                "shape": list(p.shape),
                "dtype": str(p.dtype),
                "requires_grad": p.requires_grad,
            }
            for name, p in self.model.named_parameters()
        ]

    def _log(self, event):
        with (self.run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")

    def _usable(self):
        if self._failed or self._updating:
            raise RuntimeError(
                "trainer is failed or inside an update; resume the last complete checkpoint"
            )

    def step(self) -> dict:
        self._usable()
        if self.global_step >= self.config.total_steps:
            raise ValueError("configured optimizer updates exhausted")
        self.train_data.verify()
        self._updating, success = True, False
        try:
            finite_parameters(self.model, f"before update {self.global_step + 1}")
            microbatches = [self.stream.next_batch() for _ in range(self.config.accumulation_steps)]
            count = sum(len(w.payload) for batch in microbatches for w in batch)
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            nll_sum = 0.0
            for index, windows in enumerate(microbatches):
                try:
                    loss = sum(
                        raw_byte_nll(self.model(target), target)[0]
                        for target in batches_by_length(windows)
                    )
                    (loss / count).backward()
                except ValueError as error:
                    raise ValueError(
                        f"update {self.global_step + 1}, microbatch {index + 1}: {error}"
                    ) from error
                nll_sum += loss.detach().item()
                self.micro_step += 1
            finite_gradients(self.model, f"update {self.global_step + 1}")
            norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.clip_norm,
                error_if_nonfinite=True,
                foreach=False,
            ).item()
            lr = self.optimizer.param_groups[0]["lr"]
            self.optimizer.step()
            finite_parameters(self.model, f"after update {self.global_step + 1}")
            self.scheduler.advance()
            self.global_step += 1
            self.examples_seen += sum(len(batch) for batch in microbatches)
            self.bytes_seen += count
            memory = training_memory(self.model, self.optimizer, self.config.batch_size)
            self.optimizer.zero_grad(set_to_none=True)
            nll = nll_sum / count
            record = {
                "event": "train",
                "step": self.global_step,
                "micro_step": self.micro_step,
                "loss": nll,
                "raw_byte_nll": nll,
                "raw_byte_perplexity": math.exp(nll),
                "valid_target_count": count,
                "learning_rate": lr,
                "gradient_norm": norm,
                "clip_threshold": self.config.clip_norm,
                "clipped": norm > self.config.clip_norm,
                "examples_seen": self.examples_seen,
                "bytes_seen": self.bytes_seen,
                "elapsed_seconds": self._elapsed_offset + time.perf_counter() - self._started,
            }
            self.metrics.append(record)
            self._log({**record, "memory": memory})
            success = True
            return {**record, "memory": memory}
        finally:
            self._updating = False
            if not success:
                self._failed = True

    def train_until(self, step: int):
        if type(step) is not int or not self.global_step <= step <= self.config.total_steps:
            raise ValueError("train_until must be within the unchanged configured schedule")
        while self.global_step < step:
            self.step()

    def validate(self, split: str = "validation") -> dict:
        self._usable()
        if split not in ("train", "validation"):
            raise ValueError("invalid validation split")
        data = self.train_data if split == "train" else self.validation_data
        data.verify()
        was_training = self.model.training
        total, count = 0.0, 0
        try:
            self.model.eval()
            with torch.inference_mode():
                for start in range(0, len(data), self.config.batch_size):
                    for target in batches_by_length(
                        list(data.windows[start : start + self.config.batch_size])
                    ):
                        loss, valid = raw_byte_nll(self.model(target), target)
                        total += loss.item()
                        count += valid
        finally:
            self.model.train(was_training)
        nll = total / count
        record = {
            "event": "validation",
            "split": split,
            "step": self.global_step,
            "loss": nll,
            "raw_byte_nll": nll,
            "raw_byte_perplexity": math.exp(nll),
            "valid_target_count": count,
        }
        self._log(record)
        return record

    def save(self) -> Path:
        self._usable()
        self.train_data.verify()
        finite_parameters(self.model, "checkpoint")
        if any(p.grad is not None for p in self.model.parameters()):
            raise RuntimeError("checkpoint requires cleared gradients at an optimizer boundary")
        state = {
            "schema": "1",
            "run_id": self.run_manifest["run_id"],
            "model_config": self.model_config.to_dict(),
            "training_config": self.config.to_dict(),
            "dataset_sha256": self.manifest.sha256,
            "parameter_structure": self._structure(),
            "model": self.model.state_dict(),
            "optimizer_type": "AdamW",
            "optimizer": self.optimizer.state_dict(),
            "optimizer_update_counts": [
                int(self.optimizer.state[p]["step"].item()) if p in self.optimizer.state else 0
                for group in self.optimizer.param_groups
                for p in group["params"]
            ],
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "micro_step": self.micro_step,
            "accumulation_position": 0,
            "examples_seen": self.examples_seen,
            "bytes_seen": self.bytes_seen,
            "python_rng": random.getstate(),
            "torch_cpu_rng": torch.get_rng_state(),
            "data_cursor": self.stream.state_dict(),
            "code": self.code,
            "environment": self.env,
            "metrics": self.metrics,
            "elapsed_seconds": self._elapsed_offset + time.perf_counter() - self._started,
        }
        return save_checkpoint(self.run_dir / "checkpoints" / f"step_{self.global_step:06d}", state)

    def _restore(self, state):
        required = {
            "schema",
            "run_id",
            "model_config",
            "training_config",
            "dataset_sha256",
            "parameter_structure",
            "model",
            "optimizer_type",
            "optimizer",
            "optimizer_update_counts",
            "scheduler",
            "global_step",
            "micro_step",
            "accumulation_position",
            "examples_seen",
            "bytes_seen",
            "python_rng",
            "torch_cpu_rng",
            "data_cursor",
            "code",
            "environment",
            "metrics",
            "elapsed_seconds",
        }
        if set(state) != required or state["schema"] != "1":
            raise CheckpointError("checkpoint schema/fields mismatch")
        for key, expected in (
            ("model_config", self.model_config.to_dict()),
            ("training_config", self.config.to_dict()),
            ("dataset_sha256", self.manifest.sha256),
            ("parameter_structure", self._structure()),
            ("optimizer_type", "AdamW"),
            ("environment", self.env),
        ):
            if state[key] != expected:
                raise CheckpointError(f"resume compatibility mismatch: {key}")
        if state["code"]["source_sha256"] != self.code["source_sha256"]:
            raise CheckpointError("resume source code identity mismatch")
        step = state["global_step"]
        if type(step) is not int or not 0 <= step <= self.config.total_steps:
            raise CheckpointError("invalid checkpoint global step")
        for name in ("micro_step", "examples_seen", "bytes_seen", "accumulation_position"):
            if type(state[name]) is not int or state[name] < 0:
                raise CheckpointError(f"invalid checkpoint counter {name}")
        if (
            state["micro_step"] != step * self.config.accumulation_steps
            or state["accumulation_position"] != 0
        ):
            raise CheckpointError("checkpoint is not a complete accumulation boundary")
        if state["examples_seen"] != state["micro_step"] * self.config.batch_size:
            raise CheckpointError("checkpoint example counter mismatch")
        if (
            not state["examples_seen"]
            <= state["bytes_seen"]
            <= state["examples_seen"] * self.config.sequence_length
        ):
            raise CheckpointError("checkpoint byte counter mismatch")
        if not isinstance(state["metrics"], list) or len(state["metrics"]) != step:
            raise CheckpointError("checkpoint metric history mismatch")
        elapsed = state["elapsed_seconds"]
        if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
            raise CheckpointError("invalid checkpoint elapsed time")
        recorded_bytes = 0
        for index, row in enumerate(state["metrics"], 1):
            if not isinstance(row, dict) or row.get("step") != index:
                raise CheckpointError("invalid checkpoint metric step")
            count = row.get("valid_target_count")
            if type(count) is not int or count < 1:
                raise CheckpointError("invalid checkpoint metric byte count")
            recorded_bytes += count
            if row.get("bytes_seen") != recorded_bytes:
                raise CheckpointError("checkpoint metric/counter mismatch")
            for name in (
                "loss",
                "raw_byte_nll",
                "raw_byte_perplexity",
                "learning_rate",
                "gradient_norm",
                "elapsed_seconds",
            ):
                value = row.get(name)
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise CheckpointError(f"invalid checkpoint metric {name}")
        if recorded_bytes != state["bytes_seen"]:
            raise CheckpointError("checkpoint metric total byte mismatch")
        cursor = state["data_cursor"]
        if not isinstance(cursor, dict):
            raise CheckpointError("invalid checkpoint cursor")
        draws = state["examples_seen"]
        epoch, offset = (
            ((draws - 1) // len(self.train_data), (draws - 1) % len(self.train_data) + 1)
            if draws
            else (0, 0)
        )
        if cursor.get("epoch") != epoch or cursor.get("offset") != offset:
            raise CheckpointError("checkpoint cursor disagrees with example counter")
        self.stream.load_state_dict(cursor)
        current = self.model.state_dict()
        if set(state["model"]) != set(current):
            raise CheckpointError("checkpoint weight names mismatch")
        for name, value in state["model"].items():
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != current[name].shape
                or value.dtype != current[name].dtype
                or not torch.isfinite(value).all()
            ):
                raise CheckpointError(f"invalid checkpoint weight: {name}")
        self._validate_optimizer(state["optimizer"], step, state["optimizer_update_counts"])
        random.Random().setstate(state["python_rng"])
        rng = state["torch_cpu_rng"]
        if (
            not isinstance(rng, torch.Tensor)
            or rng.dtype != torch.uint8
            or rng.shape != torch.get_rng_state().shape
        ):
            raise CheckpointError("invalid Torch CPU RNG state")
        self.model.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        if self.scheduler.completed != step:
            raise CheckpointError("scheduler/global step mismatch")
        self.global_step, self.micro_step = step, state["micro_step"]
        self.examples_seen, self.bytes_seen = state["examples_seen"], state["bytes_seen"]
        self.metrics = state["metrics"]
        self._elapsed_offset = state["elapsed_seconds"]
        random.setstate(state["python_rng"])
        torch.set_rng_state(rng)

    def _validate_optimizer(self, state, step, update_counts):
        if not isinstance(state, dict) or set(state) != {"state", "param_groups"}:
            raise CheckpointError("invalid AdamW state fields")
        expected = self.optimizer.state_dict()["param_groups"]
        if len(state["param_groups"]) != len(expected):
            raise CheckpointError("optimizer group count mismatch")
        for a, b in zip(state["param_groups"], expected, strict=True):
            if set(a) != set(b) or any(a[k] != b[k] for k in b if k != "lr"):
                raise CheckpointError("optimizer group/hyperparameter mismatch")
        parameters = [p for group in self.optimizer.param_groups for p in group["params"]]
        # AdamW creates state only when a parameter receives its first gradient.
        # A short byte tail can leave the patch encoder unused for an update.
        if (
            not isinstance(update_counts, list)
            or len(update_counts) != len(parameters)
            or any(type(count) is not int or not 0 <= count <= step for count in update_counts)
        ):
            raise CheckpointError("invalid AdamW per-parameter update counts")
        active = {index for index, count in enumerate(update_counts) if count > 0}
        if set(state["state"]) != active:
            raise CheckpointError("optimizer parameter-state coverage mismatch")
        for index, value in state["state"].items():
            if set(value) != {"step", "exp_avg", "exp_avg_sq"}:
                raise CheckpointError("invalid AdamW tensor fields")
            for name in ("exp_avg", "exp_avg_sq"):
                tensor = value[name]
                if (
                    not isinstance(tensor, torch.Tensor)
                    or tensor.shape != parameters[index].shape
                    or tensor.dtype != torch.float32
                    or not torch.isfinite(tensor).all()
                ):
                    raise CheckpointError(f"invalid AdamW {name} tensor")
                if name == "exp_avg_sq" and (tensor < 0).any():
                    raise CheckpointError("negative AdamW second moment")
            if (
                not isinstance(value["step"], torch.Tensor)
                or value["step"].shape != torch.Size([])
                or value["step"].dtype != torch.float32
                or value["step"].item() != update_counts[index]
            ):
                raise CheckpointError("AdamW step counter mismatch")
