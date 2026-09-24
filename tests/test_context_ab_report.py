"""A/B report gates and stale evidence rejection without experiment execution."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import finish_context_training_ab as report  # noqa: E402
import validate_context_ab as validation  # noqa: E402

from unified_edge.training.config import canonical_hash  # noqa: E402


def test_arm_difference_pairs_identity_and_averages_phases():
    rows = []
    for key in ("a", "b"):
        for phase in range(8):
            rows.append(
                {
                    "anchor": {"key": key},
                    "phase": phase,
                    "scores": {str(32 + phase): 3.0, str(64 + phase): 3.0},
                }
            )
    changed = [dict(r, scores=dict(r["scores"], **{str(64 + r["phase"]): 2.5})) for r in rows]
    value = report.arm_difference(rows, changed, benefit=True)
    assert value == {"mean": -0.5, "median": -0.5, "ci95": [-0.5, -0.5], "anchors": 2}
    with pytest.raises(ValueError, match="unpaired"):
        report.arm_difference(rows, changed[:-1])


def test_decision_requires_quality_and_distribution_gates(monkeypatch):
    monkeypatch.setattr(report, "distribution_summary", lambda rows: rows[0])
    summary = {"class_counts": {}, "count": 2, "control_mass_p": 0.0}
    parent = {
        "segmented": {"32": {"nll": 3.0}},
        "position_distribution_summary": summary,
        "generations": [{"steps": [summary]}],
    }

    def arm(nll, history, seconds=100):
        return {
            "full": {
                length: {"nll": nll, "domains": {"code": {"nll": nll}}} for length in ("32", "64")
            },
            "phase": {
                "aggregate": {
                    "64": {"paired_delta": history, "paired_delta_ci95": [history, history]}
                }
            },
            "position_distribution": summary,
            "generation_distribution": summary,
            "training_seconds": seconds,
        }

    delta = {"ci95": [-0.1, -0.01]}
    assert (
        report.decision(parent, arm(2.8, 0), arm(2.7, -0.02), delta, delta)["recommendation"]
        == "REVIEW_64_BYTE_CURRICULUM"
    )
    absolute_worse = {"ci95": [0.01, 0.02]}
    assert (
        report.decision(parent, arm(2.8, 0), arm(2.7, -0.02), delta, absolute_worse)[
            "recommendation"
        ]
        != "REVIEW_64_BYTE_CURRICULUM"
    )
    domain_worse = arm(2.7, -0.02)
    domain_worse["full"]["64"]["domains"]["code"]["nll"] = 3.0
    assert not report.decision(parent, arm(2.8, 0), domain_worse, delta, delta)["gates"][
        "domain_nonregression"
    ]
    b = arm(2.7, -0.02)
    b["position_distribution"] = {
        "class_counts": {"SEVERE_BYTE_MODE_CONCENTRATION": 2},
        "count": 2,
        "control_mass_p": 0.0,
    }
    result = report.decision(parent, arm(2.8, 0), b, delta, delta)
    assert not result["gates"]["distribution_nonregression"]
    assert result["recommendation"] == "MORE_EVIDENCE_REQUIRED"
    assert (
        report.decision(parent, arm(2.999, 0), arm(2.998, 0), delta, delta)["recommendation"]
        == "FREEZE_2M_AND_REVIEW_SCALING"
    )
    assert (
        report.decision(parent, arm(2.8, 0), arm(2.9, 0), delta, delta)["recommendation"]
        == "CONTINUE_32_BYTE_RUN"
    )


def test_final_receipt_rejects_stale_measurement_and_tested_code(tmp_path, monkeypatch):
    monkeypatch.setattr(validation, "ROOT", tmp_path)
    monkeypatch.setattr(validation, "AB", tmp_path / "study")
    validation.AB.mkdir()
    measurement = validation.AB / "measurement.json"
    measurement.write_text("original")
    for name in validation.FILES:
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("tested")
    value = {
        "schema": "context-ab-validation-1",
        "integrity": {"status": "PASS"},
        "measurement_hashes": validation.measurement_hashes(),
        "tested_files": {p: validation.sha(tmp_path / p) for p in validation.FILES},
        "commands": [
            {"name": name, "returncode": 0}
            for name in (
                "pytest",
                "ruff",
                "format",
                "compileall",
                "cpu_dependencies",
                "cuda_dependencies",
                "diff",
            )
        ],
    }
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"value": value, "sha256": canonical_hash(value)}))
    assert validation.verify_receipt(receipt)["schema"] == "context-ab-validation-1"
    measurement.write_text("new")
    with pytest.raises(ValueError, match="measurement evidence stale"):
        validation.verify_receipt(receipt)
    measurement.write_text("original")
    (tmp_path / validation.FILES[0]).write_text("changed after tests")
    with pytest.raises(ValueError, match="code evidence stale"):
        validation.verify_receipt(receipt)
