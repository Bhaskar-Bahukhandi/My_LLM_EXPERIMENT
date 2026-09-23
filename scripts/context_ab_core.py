"""Matched TRAIN-only blocks and isolated experiment-local optimizer/checkpoint state."""

import copy
import hashlib
import random
import time
from pathlib import Path

import torch
from context_distribution_study import PRODUCTION
from context_study_metrics import require

from unified_edge.dense_model import DenseByteModel
from unified_edge.training.checkpoint import load_checkpoint, save_checkpoint
from unified_edge.training.device import cuda_rng_state, validate_cuda_rng
from unified_edge.training.memory import training_memory
from unified_edge.training.optimization import (
    WarmupCosine,
    build_optimizer,
    finite_gradients,
    finite_parameters,
    raw_byte_nll,
)
from unified_edge.training.trainer import Trainer

QUOTAS = {"general_text": 256, "code": 128, "documentation": 77, "structured_math": 51}


def select_blocks(documents, read_payload, quotas=None):
    quotas = QUOTAS if quotas is None else quotas
    require(all(d.split == "train" for d in documents), "training selection accepts TRAIN only")
    groups = {domain: [] for domain in quotas}
    for d in documents:
        require(d.domain in groups, "unexpected TRAIN domain")
        for start in range(0, d.byte_count - 127, 128):
            key = hashlib.sha256(f"context-ab-v1|17|{d.path}|{start}".encode()).hexdigest()
            groups[d.domain].append((key, d, start))
    selected = []
    cache = {}
    for domain, count in quotas.items():
        require(len(groups[domain]) >= count, "insufficient nonoverlapping domain capacity")
        for key, d, start in sorted(groups[domain], key=lambda v: v[0])[:count]:
            if d.path not in cache:
                payload = read_payload(d)
                require(
                    len(payload) == d.byte_count
                    and hashlib.sha256(payload).hexdigest() == d.sha256,
                    "TRAIN document mutation",
                )
                cache[d.path] = payload
            raw = cache[d.path][start : start + 128]
            require(len(raw) == 128, "block crosses document")
            selected.append(
                {
                    "document": d.path,
                    "start": start,
                    "end": start + 128,
                    "domain": domain,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "selection_key": key,
                }
            )
    return sorted(
        selected, key=lambda v: hashlib.sha256(("order|" + v["selection_key"]).encode()).hexdigest()
    )


def matched_targets(raw, length, device="cpu"):
    require(length in (32, 64) and len(raw) == 128, "expected 128-byte block and 32/64 context")
    a = torch.tensor(list(raw), dtype=torch.long).reshape(4, 32)
    b = torch.tensor(list(raw), dtype=torch.long).reshape(2, 64)
    left = bytes(a.flatten().tolist())
    right = bytes(b.flatten().tolist())
    require(left == right == raw, "A/B target bytes differ")
    require(
        hashlib.sha256(left).digest() == hashlib.sha256(right).digest(),
        "A/B target digest mismatch",
    )
    return (a if length == 32 else b).to(device)


def trees_equal(a, b):
    if isinstance(a, torch.Tensor):
        return (
            isinstance(b, torch.Tensor)
            and a.dtype == b.dtype
            and a.shape == b.shape
            and torch.equal(a.cpu(), b.cpu())
        )
    if isinstance(a, dict):
        return (
            isinstance(b, dict) and a.keys() == b.keys() and all(trees_equal(a[k], b[k]) for k in a)
        )
    if isinstance(a, (list, tuple)):
        return (
            type(a) is type(b)
            and len(a) == len(b)
            and all(trees_equal(x, y) for x, y in zip(a, b, strict=True))
        )
    return a == b


def storages(value):
    if isinstance(value, torch.Tensor):
        return {(str(value.device), value.untyped_storage().data_ptr())} if value.numel() else set()
    values = (
        value.values()
        if isinstance(value, dict)
        else value
        if isinstance(value, (tuple, list))
        else ()
    )
    return set().union(*(storages(v) for v in values))


def independent(*values):
    seen = set()
    for value in values:
        current = storages(value)
        require(not seen.intersection(current), "tensor storage aliases across branches")
        seen.update(current)


def safe_destination(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    require(not root.is_relative_to(PRODUCTION.resolve()), "production write prohibited")
    require(path.is_relative_to(root) and path != root, "checkpoint outside experiment root")
    require(not path.is_relative_to(PRODUCTION.resolve()), "production write prohibited")
    return path


class Arm:
    """Same accepted optimizer/scheduler; experimental cursor never enters production Trainer."""

    def __init__(self, parent, resolved, config, length, binding_hash, arm, device):
        require(length in (32, 64), "unsupported experimental window")
        self.config = config
        self.length = length
        self.binding_hash = binding_hash
        self.arm = arm
        self.device = torch.device(device)
        self.model = DenseByteModel(resolved).to(self.device)
        self.model.load_state_dict(copy.deepcopy(parent["model"]), strict=True)
        self.optimizer = build_optimizer(self.model, config)
        self.scheduler = WarmupCosine(self.optimizer, config)
        self.parent_step = parent["global_step"]
        Trainer._validate_optimizer(
            self, parent["optimizer"], self.parent_step, parent["optimizer_update_counts"]
        )
        self.optimizer.load_state_dict(copy.deepcopy(parent["optimizer"]))
        self.scheduler.load_state_dict(copy.deepcopy(parent["scheduler"]))
        self.rng = copy.deepcopy(
            {k: parent[k] for k in ("python_rng", "torch_cpu_rng", "cuda_rng")}
        )
        validate_cuda_rng(self.rng["cuda_rng"], self.device)
        self.step = 0
        self.failed = False

    def restore_rng(self):
        random.setstate(self.rng["python_rng"])
        torch.set_rng_state(self.rng["torch_cpu_rng"])
        if self.device.type == "cuda":
            torch.cuda.set_rng_state_all(self.rng["cuda_rng"])

    def capture_rng(self):
        self.rng = {
            "python_rng": random.getstate(),
            "torch_cpu_rng": torch.get_rng_state(),
            "cuda_rng": cuda_rng_state(self.device),
        }

    def update(self, raw, *, advance=True):
        require(not self.failed, "failed arm requires checkpoint restoration")
        self.failed = True
        self.restore_rng()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        finite_parameters(self.model, "before disposable update")
        targets = matched_targets(raw, self.length, self.device)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        # Both forms use one batch: 4x32 or 2x64. Each row starts at fresh BOS.
        summed, count = raw_byte_nll(self.model(targets), targets)
        require(count == 128, "gradient target count mismatch")
        (summed / 128).backward()
        finite_gradients(self.model, "disposable update")
        require(all(p.grad is not None for p in self.model.parameters()), "omitted gradient tensor")
        norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.clip_norm, error_if_nonfinite=True, foreach=False
        ).item()
        lr = self.optimizer.param_groups[0]["lr"]
        if advance:
            self.optimizer.step()
            self.scheduler.advance()
            self.step += 1
        finite_parameters(self.model, "after disposable update")
        mem = training_memory(self.model, self.optimizer, targets.shape[0])
        elapsed = time.perf_counter() - started
        semantic = {
            "update": self.step,
            "sum_nll": summed.item(),
            "mean_nll": summed.item() / 128,
            "target_bytes": 128,
            "cumulative_target_bytes": self.step * 128,
            "learning_rate": lr,
            "gradient_norm": norm,
            "clipped": norm > self.config.clip_norm,
            "finite_gradients": True,
            "gradient_tensors": sum(p.grad is not None for p in self.model.parameters()),
            "scheduler_completed": self.scheduler.completed,
            "target_sha256": hashlib.sha256(raw).hexdigest(),
        }
        self.optimizer.zero_grad(set_to_none=True)
        self.capture_rng()
        self.failed = False
        return {"semantic": semantic, "seconds": elapsed, "memory": mem}

    def state(self):
        require(not self.failed, "failed arm cannot checkpoint")
        require(
            all(p.grad is None for p in self.model.parameters()),
            "checkpoint requires zero gradients",
        )
        return {
            "schema": "2",
            "kind": "DISPOSABLE_AB_ONLY",
            "binding_sha256": self.binding_hash,
            "arm": self.arm,
            "window_length": self.length,
            "experimental_update": self.step,
            "target_bytes": self.step * 128,
            "block_cursor": self.step,
            "parent_step": self.parent_step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "rng": self.rng,
        }

    def save(self, path, root):
        return save_checkpoint(safe_destination(path, root), self.state())

    def restore(self, path, root):
        state = load_checkpoint(safe_destination(path, root))
        for key, value in (
            ("kind", "DISPOSABLE_AB_ONLY"),
            ("binding_sha256", self.binding_hash),
            ("arm", self.arm),
            ("window_length", self.length),
            ("parent_step", self.parent_step),
        ):
            require(state[key] == value, "wrong experimental checkpoint binding")
        step = state["experimental_update"]
        require(
            type(step) is int
            and 0 <= step <= 512
            and state["block_cursor"] == step
            and state["target_bytes"] == 128 * step,
            "experimental cursor mismatch",
        )
        Trainer._validate_optimizer(
            self,
            state["optimizer"],
            self.parent_step + step,
            [self.parent_step + step] * len(list(self.model.parameters())),
        )
        self.model.load_state_dict(state["model"], strict=True)
        finite_parameters(self.model, "restored arm")
        self.optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
        self.scheduler.load_state_dict(state["scheduler"])
        require(
            self.scheduler.completed == self.parent_step + step, "experimental scheduler mismatch"
        )
        self.rng = copy.deepcopy(state["rng"])
        validate_cuda_rng(self.rng["cuda_rng"], self.device)
        self.restore_rng()
        self.step = step
        self.failed = False
