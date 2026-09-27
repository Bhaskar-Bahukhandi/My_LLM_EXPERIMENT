"""Resume the prior accounting using meta tensors only; never run a model forward."""

import hashlib
import json
from pathlib import Path

import torch
import yaml
from audit_20m_prebuild import inspect_candidate

from unified_edge.config import AuthoredConfig, digest, load_config
from unified_edge.dense_model import DenseByteModel
from unified_edge.parameters import audit_parameters
from unified_edge.resolve import resolve_config

ROOT = Path(__file__).resolve().parents[1]


def memory(row, batch, length):
    """Planning allowances, not a measured peak or a proven upper bound."""
    s = row["shape"]
    k, d, i, n, h = s["shared_layers"], s["d_model"], row["d_inner"], s["d_state"], row["heads"]
    q, e, t, p = s["decoder_dim"], s["byte_dim"], (length + 7) // 8, row["parameters"]
    terms = {
        "fp32_parameters": 4 * p,
        "fp32_gradients": 4 * p,
        "adam_first_moment": 4 * p,
        "adam_second_moment": 4 * p,
        "adam_scalar_steps_max": row["adam_step_scalar_bytes"],
        "canonical_ssm_state": batch * row["ssm_bytes_per_batch_item_fp32"],
        "canonical_conv_state": batch * row["conv_bytes_per_batch_item_fp32"],
        "canonical_decoder_hidden": batch * q * 4,
        "pending_int64_symbols_max": batch * 7 * 8,
        # Reserve two additional state-sized allocations for incoming/final state and autograd.
        "state_working_copies_allowance": 2
        * batch
        * (row["ssm_bytes_per_batch_item_fp32"] + row["conv_bytes_per_batch_item_fp32"]),
        "trunk_saved_activations_allowance": 4 * batch * t * k * (12 * d + 16 * i + 8 * n + 4 * h),
        "local_saved_activations_allowance": 4 * batch * length * (12 * q + 4 * e + 2 * 267),
        "encoder_saved_activations_allowance": 4 * batch * max(0, t - 1) * 8 * 6 * e,
        "context_vectors": 4 * batch * t * d,
        # Segment/transition/bwd copies and B-C matrices, plus two bool masks per layer.
        "reference_ssd_temporary_allowance": 4 * k * batch * t * t * (6 * h + 2) + 2 * k * t * t,
        "optimizer_temporary_allowance": 8 * p,
        "kernel_workspace_allowance": 1024**3,
        "allocator_fragmentation_reserve": 1024**3,
        "cuda_context_and_framework_allowance": 512 * 1024**2,
    }
    return {
        "batch": batch,
        "length": length,
        "prediction_patches": t,
        "terms_bytes": terms,
        "planning_total_bytes": sum(terms.values()),
        "planning_total_gib": sum(terms.values()) / 1024**3,
        "measured": False,
        "activation_coefficients": (
            "Explicit engineering allowances; allocator/autograd peaks require measurement"
        ),
    }


def main():
    target = ROOT / "reports/20m_prebuild_mechanical.json"
    original_bytes = target.read_bytes()
    prior = json.loads(original_bytes)
    if "resumed_accounting" in prior:
        raise FileExistsError("resumed receipt exists; do not silently overwrite evidence")
    row = inspect_candidate(512, 12, decoder=256)
    old = next(c for c in prior["candidates"] if c["id"] == row["id"])
    if row != old:
        raise ValueError("selected candidate differs from retained original accounting")
    authored = AuthoredConfig.from_dict(row["authored"])
    content = (
        "# RECOMMENDED_FOR_PILOT; not PRODUCTION_APPROVED; no training authorization.\n"
        "# CPU hardware here is the existing inventory schema, not the T4 execution profile.\n"
        "# Baseline bridge; training/runtime gates are in docs/20m_training_plan.md.\n"
        + yaml.safe_dump(authored.to_dict(), sort_keys=False)
    )
    config_path = ROOT / "configs/models/edge_20m_candidate.yaml"
    if config_path.exists():
        raise FileExistsError(config_path)
    config_path.write_text(content, encoding="utf-8")
    resolved = resolve_config(load_config(config_path)).selected
    if resolved is None or resolved.sha256 != row["resolved_sha256"]:
        raise ValueError("authored YAML round-trip changed resolved identity")
    with torch.device("meta"):
        model = DenseByteModel(resolved)
    if audit_parameters(model)["unique_trainable_parameters"] != row["parameters"]:
        raise ValueError("YAML executable count mismatch")
    comparison = []
    ids = {
        "d256_l46_n64_b64_q128",
        "d320_l30_n64_b64_q128",
        "d384_l21_n64_b64_q128",
        "d512_l12_n64_b64_q128",
        "d640_l8_n64_b64_q128",
        "d896_l4_n64_b64_q128",
        "d1024_l3_n64_b64_q128",
        "d512_l12_n64_b64_q256",
        "d512_l12_n64_b128_q128",
        "d512_l12_n128_b64_q128",
        "d512_l11_n128_b64_q128",
    }
    for c in prior["candidates"]:
        if c["id"] not in ids:
            continue
        s = c["shape"]
        projection_macs = s["shared_layers"] * (
            s["d_model"] * (2 * c["d_inner"] + 2 * s["d_state"] + c["heads"])
            + c["d_inner"] * s["d_model"]
        )
        comparison.append(
            {
                "id": c["id"],
                "parameters": c["parameters"],
                "layers": s["shared_layers"],
                "width": s["d_model"],
                "state": s["d_state"],
                "projection_macs_per_patch": projection_macs,
                "recurrent_bytes_per_sequence": c["ssm_bytes_per_batch_item_fp32"]
                + c["conv_bytes_per_batch_item_fp32"]
                + 4 * s["decoder_dim"]
                + 56,
                "trunk_percent": c["component_percent"]["mamba_trunk_excluding_norm"],
            }
        )
    resume = {
        "entry_commit": "6151a9662a9cc40fa7dfcc2266f36adfc8ca8383",
        "prior_receipt_raw_sha256": hashlib.sha256(original_bytes).hexdigest(),
        "prior_grid_repeated": False,
        "new_constructions": "Selected candidate and authored YAML round-trip only, both meta",
        "selected": row,
        "trainable_tensor_count": len(list(model.parameters())),
        "trainable_tensor_structure_sha256": digest(row["tensor_ledger"]),
        "parameter_value_sha256": "NOT_APPLICABLE_META_HAS_NO_VALUES",
        "authored_yaml_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "comparison": comparison,
        "memory": [
            memory(row, b, length)
            for b, length in ((1, 32), (2, 64), (8, 32), (8, 64), (8, 128), (8, 256))
        ],
        "forward_backward_optimizer_executed": False,
        "production_payloads_read": False,
    }
    prior["resumed_accounting"] = resume
    target.write_text(json.dumps(prior, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                k: resume[k]
                for k in (
                    "trainable_tensor_count",
                    "trainable_tensor_structure_sha256",
                    "comparison",
                    "memory",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
