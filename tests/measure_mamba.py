"""Reproduce bounded Mamba evidence using the same independent oracles as the tests.

Run after pytest writes evidence/mamba_integration/full_suite.xml. Pinned upstream
sources must already exist under evidence/mamba_integration/upstream; no downloads occur.
"""

import argparse
import hashlib
import importlib.util
import io
import json
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from test_dense_model import CUTS, finite_logits, stream
from test_mamba import (
    ATOL,
    LENGTHS,
    RTOL,
    accepted_config,
    assert_states,
    equation_step,
    errors,
    state_accounting,
    tensor_reconciliation,
)

from unified_edge.dense_model import DenseByteModel
from unified_edge.mamba import SharedMambaTrunk
from unified_edge.parameters import audit_parameters

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(actual, expected):
    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)
    return errors(actual, expected)


def maxima(rows):
    return {
        key: max((row[key] for row in rows), default=0.0) for key in ("max_abs", "max_relative")
    }


@torch.no_grad()
def measure_recurrence():
    records = []
    for seed, batch in ((7, 1), (29, 2), (101, 4)):
        torch.manual_seed(seed)
        trunk = SharedMambaTrunk(accepted_config()).eval()
        output_errors, state_errors, split_errors, resume_errors, oracle_errors = [], [], [], [], []
        norms, peaks = [], []
        for length in LENGTHS:
            data = torch.randn(batch, length, 256)
            full, final = trunk(data)
            state = trunk.initialize_state(batch)
            steps = []
            for t in range(length):
                value, state = trunk.step(data[:, t], state)
                steps.append(value)
                for layer in state.layers:
                    for tensor in (layer.conv, layer.ssm):
                        assert torch.isfinite(tensor).all()
                        norms.append(tensor.norm().item())
                        peaks.append(tensor.abs().max().item())
            actual = torch.stack(steps, 1) if length else data.clone()
            output_errors.append(compare(actual, full))
            assert_states(state, final)
            for a, b in zip(state.layers, final.layers, strict=True):
                state_errors.extend((compare(a.conv, b.conv), compare(a.ssm, b.ssm)))
            for split in sorted({0, min(1, length), length // 2, max(0, length - 1), length}):
                prefix, middle = trunk(data[:, :split])
                suffix, resumed = trunk(data[:, split:], middle)
                split_errors.append(compare(torch.cat((prefix, suffix), 1), full))
                assert_states(resumed, final)
                buffer = io.BytesIO()
                torch.save(trunk.export_state(middle), buffer)
                buffer.seek(0)
                restored = trunk.restore_state(torch.load(buffer, weights_only=True))
                restored_suffix, restored_final = trunk(data[:, split:], restored)
                resume_errors.append(compare(restored_suffix, suffix))
                assert_states(restored_final, resumed)
        state = trunk.initialize_state(batch)
        for _ in range(4):
            inputs = torch.randn(batch, 256)
            oracle = inputs
            for block, layer in zip(trunk.layers, state.layers, strict=True):
                oracle, _ = equation_step(block, oracle, layer)
            actual, state = trunk.step(inputs, state)
            oracle_errors.append(compare(actual.double(), oracle))
        records.append(
            {
                "seed": seed,
                "batch": batch,
                "lengths": list(LENGTHS),
                "full_vs_step": maxima(output_errors),
                "state_full_vs_step": maxima(state_errors),
                "chunk_split": maxima(split_errors),
                "serialized_resume_vs_same_split": maxima(resume_errors),
                "fp64_equation_oracle": maxima(oracle_errors),
                "max_state_l2_norm": max(norms),
                "max_abs_recurrent_tensor": max(peaks),
            }
        )
    return records


@torch.no_grad()
def measure_integration():
    records = []
    for seed, batch in ((3, 1), (37, 2), (79, 1)):
        torch.manual_seed(seed)
        model = DenseByteModel(accepted_config()).eval()
        data = torch.randint(0, 256, (batch, 137))
        full = model(data)
        incremental, final = stream(model, data)
        parity = compare(finite_logits(incremental), finite_logits(full))
        causal = []
        for cut in CUTS:
            changed = data.clone()
            changed[:, cut:] = (changed[:, cut:] + 113) % 256
            result = model(changed)
            causal.append(
                compare(finite_logits(result[:, : cut + 1]), finite_logits(full[:, : cut + 1]))
            )
        resumes, restored_predictions, restored_states = [], [], []
        for split in (0, 1, 7, 8, 9, 15, 16, 17, 63, 64, 65, 127, 128, 129):
            prefix, state = stream(model, data[:, :split])
            buffer = io.BytesIO()
            torch.save({"state": model.export_state(state), "weights": model.state_dict()}, buffer)
            buffer.seek(0)
            saved = torch.load(buffer, weights_only=True)
            restored_model = DenseByteModel(model.config).eval()
            restored_model.load_state_dict(saved["weights"])
            restored = restored_model.restore_state(saved["state"])
            restored_predictions.append(
                compare(
                    finite_logits(restored_model.predict(restored)),
                    finite_logits(model.predict(state)),
                )
            )
            suffix, end = stream(restored_model, data[:, split:], restored)
            resumes.append(
                compare(finite_logits(torch.cat((prefix, suffix), 1)), finite_logits(incremental))
            )
            assert_states(end.shared, final.shared)
            for a, b in zip(end.shared.layers, final.shared.layers, strict=True):
                restored_states.extend((compare(a.conv, b.conv), compare(a.ssm, b.ssm)))
        records.append(
            {
                "seed": seed,
                "batch": batch,
                "byte_length": 137,
                "mutation_indices": list(CUTS),
                "full_vs_step": parity,
                "future_suffix_prefix_error": maxima(causal),
                "restored_prefix_prediction_error": maxima(restored_predictions),
                "serialized_continuation_error": maxima(resumes),
                "serialized_final_recurrent_state_error": maxima(restored_states),
                "max_abs_admissible_logit": finite_logits(full).abs().max().item(),
            }
        )
    return records


def measure_gradients():
    records = []
    for seed in (13, 59, 97):
        torch.manual_seed(seed)
        model = DenseByteModel(accepted_config())
        data = torch.randint(0, 256, (2, 25))
        logits = model(data)
        loss = F.cross_entropy(logits.flatten(0, 1), data.flatten())
        loss.backward()
        gradients = {}
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            norm = parameter.grad.norm().item()
            assert norm > 0, name
            gradients[name] = {"l2_norm": norm, "max_abs": parameter.grad.abs().max().item()}
        records.append({"seed": seed, "loss": loss.item(), "parameter_tensors": gradients})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"evidence output already exists: {args.output}")
    torch.set_num_threads(1)
    xml = ROOT / "evidence/mamba_integration/full_suite.xml"
    suite = ET.parse(xml).getroot().find("testsuite")
    assert suite is not None and int(suite.attrib["tests"]) >= 143
    assert all(int(suite.attrib[name]) == 0 for name in ("failures", "errors", "skipped"))
    checks = {}
    for name, command in (
        ("ruff", ["ruff", "check", "src", "tests"]),
        ("format", ["ruff", "format", "--check", "src", "tests"]),
        ("compileall", ["compileall", "-q", "src", "tests"]),
        ("dependencies", ["pip", "check"]),
    ):
        result = subprocess.run(
            [sys.executable, "-m", *command], cwd=ROOT, capture_output=True, text=True
        )
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        checks[name] = {"exit_code": result.returncode, "output": result.stdout.strip()}
    upstream_root = ROOT / "evidence/mamba_integration/upstream"
    upstream = json.loads((upstream_root / "sources.json").read_text())
    for source in upstream["sources"]:
        assert sha(upstream_root / source["path"]) == source["sha256"]
    torch.manual_seed(23)
    model = DenseByteModel(accepted_config())
    memory = []
    for batch in (1, 2, 4):
        measured = state_accounting(model.shared.initialize_state(batch))
        assert measured["combined_bytes"] == 565248 * batch
        memory.append({"batch": batch, **measured})
    paths = sorted([*ROOT.glob("src/**/*.py"), *ROOT.glob("tests/*.py")])
    paths += [
        ROOT / p
        for p in (
            "configs/models/edge_2m.yaml",
            "reports/edge_2m_resolved.json",
            "requirements-lock.txt",
            "pyproject.toml",
            "Unified_Edge400_Master_Bible_v3.0_FINAL.md",
            "Unified_Edge400_Master_Roadmap_and_Formula_Board_v1.0_FINAL.md",
        )
    ]
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "starting_byte_commit": "78c179bda571237d3c0d9aa6ec396bf3f6374b0b",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": "cpu",
            "dtype": "float32",
            "threads": 1,
        },
        "tolerance": {"atol": ATOL, "rtol": RTOL, "relative_denominator_floor": 1e-8},
        "tests": suite.attrib,
        "test_xml_sha256": sha(xml),
        "static_checks": checks,
        "upstream_sources": upstream,
        "upstream_package_runtime": {
            "status": "UNVERIFIED_BLOCKED",
            "reason": (
                "Windows CPU profile has no mamba_ssm/Triton runtime; no dependency changes made"
            ),
            "mamba_ssm_present": importlib.util.find_spec("mamba_ssm") is not None,
            "triton_present": importlib.util.find_spec("triton") is not None,
        },
        "parameter_audit": audit_parameters(model),
        "byte_component_audit": audit_parameters(model.hierarchy),
        "shared_tensor_reconciliation": tensor_reconciliation(model.shared),
        "actual_recurrent_state": memory,
        "overall_runtime_memory": "UNKNOWN_REQUIRES_MEASUREMENT",
        "recurrence_measurements": measure_recurrence(),
        "integration_measurements": measure_integration(),
        "gradient_measurements": measure_gradients(),
        "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
        "warnings": ["NumPy optional bridge absent; tensor/autograd/serialization paths validated"],
        "decision": "READY FOR TRAINING INFRASTRUCTURE",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "full_parameters": payload["parameter_audit"]["unique_trainable_parameters"],
                "recurrence": payload["recurrence_measurements"],
                "integration": payload["integration_measurements"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
