"""One-update CUDA feasibility probes with explicit requested profiles and no resizing."""

import gc
from dataclasses import replace

import torch

from unified_edge.dense_model import DenseByteModel
from unified_edge.parameters import audit_parameters
from unified_edge.preflight import memory_preflight
from unified_edge.training.config import TrainingConfig
from unified_edge.training.device import (
    CUDA_ALLOCATOR_BUDGET_BYTES,
    CUDA_FREE_HEADROOM_BYTES,
    configure_device,
)
from unified_edge.training.memory import cuda_memory
from unified_edge.training.optimization import (
    build_optimizer,
    finite_gradients,
    finite_parameters,
    raw_byte_nll,
)


def probe_candidate(config, *, batch: int = 1, length: int = 32) -> dict:
    training = TrainingConfig(device="cuda:0", batch_size=batch, sequence_length=length)
    result = {
        "target_parameters": config.target_parameters,
        "resolved_config": config.to_dict(),
        "batch_size": batch,
        "sequence_length": length,
        "precision": "fp32",
        "forward_status": "NOT_RUN",
        "backward_status": "NOT_RUN",
        "optimizer_step_status": "NOT_RUN",
        "memory_stages": {},
        "allocator_budget_bytes": CUDA_ALLOCATOR_BUDGET_BYTES,
    }
    stage = "availability"
    model = optimizer = state = targets = logits = loss = None
    try:
        device = configure_device(training.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        result["memory_stages"]["before_model"] = cuda_memory(device)
        stage = "model_load"
        model = DenseByteModel(config)
        audit = audit_parameters(model)
        result["actual_parameters"] = audit["unique_trainable_parameters"]
        planning = memory_preflight(config.shape, audit, replace(config.hardware, batch_size=batch))
        if planning.known_training_payload_bytes > CUDA_ALLOCATOR_BUDGET_BYTES:
            result.update(
                classification="BLOCKED",
                reason="known tensor payload exceeds safety budget",
                failure_stage=stage,
            )
            return result
        model = model.to(device)

        def measure(name):
            snapshot = cuda_memory(device)
            result["memory_stages"][name] = snapshot
            return snapshot

        measure("after_model_load")
        state = model.shared.initialize_state(batch)
        measure("after_recurrent_state")
        result["canonical_recurrent_state_bytes"] = sum(
            tensor.numel() * tensor.element_size()
            for layer in state.layers
            for tensor in (layer.conv, layer.ssm)
        )
        state = None
        targets = (torch.arange(batch * length, device=device).reshape(batch, length) % 256).long()
        optimizer = build_optimizer(model, training)
        stage = "forward"
        logits = model(targets)
        loss, count = raw_byte_nll(logits, targets)
        loss = loss / count
        result["forward_status"] = "PASS"
        measure("after_forward")
        stage = "backward"
        loss.backward()
        finite_gradients(model, "CUDA fit probe")
        result["backward_status"] = "PASS"
        measure("after_backward")
        stage = "optimizer_step"
        optimizer.step()
        finite_parameters(model, "CUDA fit probe update")
        result["optimizer_step_status"] = "PASS"
        measure("after_optimizer_update")
        result["classification"] = "FITS_TRAINING_MECHANICS"
        result["recommended_headroom"] = all(
            row["free_device_bytes"] >= CUDA_FREE_HEADROOM_BYTES
            for row in result["memory_stages"].values()
        )
        result["scope"] = "one synthetic optimizer update; not full-training feasibility"
    except torch.cuda.OutOfMemoryError as error:
        result.update(classification="DOES_NOT_FIT", failure_stage=stage, reason=str(error))
    except RuntimeError as error:
        if stage != "availability":
            raise
        result.update(classification="BLOCKED", failure_stage=stage, reason=str(error))
    finally:
        del model, optimizer, state, targets, logits, loss
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    return result
