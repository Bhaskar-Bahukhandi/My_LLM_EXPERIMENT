"""Additive, metadata-only prebuild audit. Never trains or opens corpus payloads."""

import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import torch

from unified_edge.config import AuthoredConfig, Hardware, ModelRequest, digest
from unified_edge.dense_model import DenseByteModel
from unified_edge.parameters import (
    audit_parameters,
    formula_parameter_estimate,
    instantiate_inventory,
)
from unified_edge.resolve import resolve_config

ROOT = Path(__file__).resolve().parents[1]
BASE = "3d2698b42404ec67834ead162db0bd10240d270b"


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def inspect_candidate(width, layers, state=64, byte=64, decoder=128):
    authored = AuthoredConfig(
        model=ModelRequest(
            target_parameters=20_000_000,
            parameter_tolerance=0.05,
            d_model=width,
            shared_layers=layers,
            d_state=state,
            byte_dim=byte,
            decoder_dim=decoder,
        ),
        hardware=Hardware(batch_size=8),
    )
    resolution = resolve_config(authored)
    config = resolution.selected
    if config is None:
        raise ValueError("candidate could not resolve")
    with torch.device("meta"):
        model = DenseByteModel(config)
    actual = audit_parameters(model)
    inventory = audit_parameters(instantiate_inventory(config.shape))
    count = actual["unique_trainable_parameters"]
    if count != inventory["unique_trainable_parameters"] or count != formula_parameter_estimate(
        config.shape
    ):
        raise ValueError("STOP: executable/inventory/formula mismatch")
    executable_shapes = sorted(tuple(p.shape) for p in model.parameters())
    inventory_shapes = sorted(
        tuple(p.shape) for p in instantiate_inventory(config.shape).parameters()
    )
    if executable_shapes != inventory_shapes:
        raise ValueError("STOP: tensor shape multiset mismatch")
    ledger = defaultdict(int)
    tensors = []
    for name, p in model.named_parameters():
        if "norm" in name:
            component = "normalization"
        elif name.startswith("shared."):
            component = "mamba_trunk_excluding_norm"
        elif name.startswith("hierarchy.embedding.symbols"):
            component = "symbol_embeddings"
        elif name.startswith("hierarchy.embedding.positions"):
            component = "position_embeddings"
        elif name.startswith("hierarchy.encoder"):
            component = "patch_encoder_excluding_norm"
        elif name.startswith("hierarchy.bootstrap"):
            component = "bootstrap"
        elif name.startswith("hierarchy.decoder.context"):
            component = "context_bridge"
        elif name.startswith("hierarchy.decoder.gru"):
            component = "local_decoder_gru"
        elif name.startswith("hierarchy.output"):
            component = "output_head"
        else:
            raise ValueError(f"unclassified parameter {name}")
        ledger[component] += p.numel()
        tensors.append({"name": name, "shape": list(p.shape), "numel": p.numel()})
    s = config.shape
    return {
        "id": f"d{width}_l{layers}_n{state}_b{byte}_q{decoder}",
        "shape": s.to_dict(),
        "d_inner": s.d_inner,
        "heads": s.nheads,
        "parameters": count,
        "delta_percent": (count - 20_000_000) / 200_000,
        "within_envelope": abs(count - 20_000_000) <= 1_000_000,
        "components": dict(ledger),
        "component_percent": {k: 100 * v / count for k, v in ledger.items()},
        "weight_bytes_fp32": count * 4,
        "gradient_bytes_fp32": count * 4,
        "adam_moment_bytes_fp32": count * 8,
        "adam_step_scalar_bytes": len(tensors) * 4,
        "ssm_bytes_per_batch_item_fp32": layers * s.d_inner * state * 4,
        "conv_bytes_per_batch_item_fp32": layers * (s.d_inner + 2 * state) * 4 * 4,
        "decoder_hidden_bytes_per_batch_item_fp32": decoder * 4,
        "pending_bytes_max_per_batch_item": 7 * 8,
        "activation_pressure_proxy_B8_L64_elements": 8 * 8 * layers * s.d_inner,
        "ssd_matrix_one_copy_B8_L64_bytes": 4 * 8 * s.nheads * 8 * 8 * layers,
        "authored": authored.to_dict(),
        "resolved": config.to_dict(),
        "resolved_sha256": config.sha256,
        "tensor_ledger": tensors,
        "validation": "META_EXECUTABLE_INVENTORY_FORMULA_AND_SHAPES_MATCH",
    }


def baseline():
    report = json.loads((ROOT / "reports/full_training_2m_final.json").read_text())
    source = {
        p.relative_to(ROOT).as_posix(): p.read_text(encoding="utf-8")
        for p in sorted((ROOT / "src").rglob("*.py"))
    }
    if digest(source) != report["amendment"]["source_sha256"]:
        raise ValueError("STOP: frozen model/training source identity differs")
    files = [
        "reports/full_training_2m_final.md",
        "reports/full_training_2m_final.json",
        "docs/project_state.md",
        "Unified_Edge400_Master_Bible_v3.0_FINAL.md",
        "Unified_Edge400_Master_Roadmap_and_Formula_Board_v1.0_FINAL.md",
        "Unified_Edge400_Future_Capabilities_Addendum_v1.0.md",
        "reports/context_distribution_study_2m.md",
        "reports/context_distribution_study_2m.json",
        "reports/context_training_ab_2m.md",
        "reports/context_training_ab_2m.json",
    ] + list(source)
    hashes = {name: sha(ROOT / name) for name in files}
    checked = {}
    for name, expected in report["amendment"]["protected"].items():
        # Read only specs, accepted reports, and checkpoints; never corpus payloads.
        if name.startswith(("reports/", "Unified_")) or "/checkpoints/" in name:
            actual = sha(ROOT / name)
            if actual != expected:
                raise ValueError(f"STOP: frozen evidence mismatch: {name}")
            checked[name] = actual
    endpoint = (
        "data/full-training-2m-v1/continuation-v1/"
        "attempt-1790232260152557800/checkpoints/step_078167"
    )
    manifest = json.loads((ROOT / endpoint / "manifest.json").read_text())
    endpoint_hash = sha(ROOT / endpoint / "state.pt")
    if endpoint_hash != manifest["sha256"] or endpoint_hash != (
        "0d4c822daadf6bb8269a727ac54d54a26e9edf49ff43fedff1b0d91ba6ad2623"
    ):
        raise ValueError("STOP: endpoint checkpoint identity mismatch")
    return {
        "accepted_commit": BASE,
        "source_sha256": digest(source),
        "entry_hashes": hashes,
        "protected_verified": checked,
        "endpoint_path": endpoint,
        "endpoint_state_sha256": endpoint_hash,
        "checkpoint_reload": "NOT_RUN; content matches accepted exact-restore proof",
        "test": "SEALED; no corpus payload read",
    }


def main():
    frozen = baseline()
    candidates = []
    for width in (256, 320, 384, 448, 512, 576, 640, 704, 768, 896, 1024):
        # Enumerate every depth and instantiate even outside-envelope controls.
        for layers in range(2, 49):
            row = inspect_candidate(width, layers)
            if row["within_envelope"]:
                candidates.append(row)
    for width, layers, state, byte, decoder in (
        (512, 12, 64, 64, 256),
        (512, 12, 64, 128, 128),
        (512, 12, 128, 64, 128),
        (512, 11, 128, 64, 128),
    ):
        candidates.append(inspect_candidate(width, layers, state, byte, decoder))
    default = resolve_config(AuthoredConfig())
    output = {
        "schema": "20m-prebuild-mechanical-v1",
        "baseline": frozen,
        "baseline_2m_parameters": inspect_candidate(256, 4),
        "default_search_max_parameters": max(c.parameters for c in default.candidates),
        "main_grid_size": 11 * 47,
        "main_grid": {
            "widths": [256, 320, 384, 448, 512, 576, 640, 704, 768, 896, 1024],
            "depths": [2, 48],
            "fixed_state": 64,
            "fixed_byte": 64,
            "fixed_decoder": 128,
        },
        "candidates": candidates,
        "training_performed": False,
        "git_tracked_file_count_at_audit": len(
            subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
        ),
    }
    target = ROOT / "reports/20m_prebuild_mechanical.json"
    if target.exists():
        raise FileExistsError("mechanical receipt already exists; preserve entry hashes")
    target.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "default_max": output["default_search_max_parameters"],
                "frozen": "PASS",
                "candidates": [
                    {k: r[k] for k in ("id", "parameters", "delta_percent")} for r in candidates
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
