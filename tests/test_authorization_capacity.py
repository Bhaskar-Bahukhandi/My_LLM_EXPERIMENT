"""Authorization must fail before download when any declared cell lacks capacity."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
_spec = importlib.util.spec_from_file_location(
    "authorization_capacity", ROOT / "scripts/authorization_capacity.py"
)
capacity = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(capacity)


def inputs():
    axes = {"source": ["source"], "domain": ["documentation"], "split": ["train", "test"]}
    quotas = [
        {
            "source": "source",
            "domain": "documentation",
            "split": s,
            "required_bytes": 70 if s == "train" else 0,
        }
        for s in axes["split"]
    ]
    candidates = [
        {
            "source": "source",
            "domain": "documentation",
            "split": "train",
            "source_unit": "document",
            "identity": "hash",
            "eligible_raw_bytes": 100,
            "eligible": True,
        }
    ]
    return candidates, quotas, axes


def test_complete_matrix_requires_headroom_and_uses_exact_integer_rounding():
    candidates, quotas, axes = inputs()
    result = capacity.evaluate(candidates, quotas, axes, [7, 10])
    assert result["status"] == "PASS"
    assert len(result["cells"]) == 2
    train, test = result["cells"]
    assert train["required_raw_headroom_bytes"] == 30
    assert train["estimated_retained_bytes"] == 70
    assert test["minimum_raw_candidate_bytes"] == 0
    quotas[0]["required_bytes"] = 71
    result = capacity.evaluate(candidates, quotas, axes, [7, 10])
    assert result["status"] == "FAIL"
    assert result["cells"][0]["minimum_raw_candidate_bytes"] == 102


def test_requested_but_absent_cell_fails_even_with_surplus_elsewhere():
    candidates, quotas, axes = inputs()
    candidates[0]["eligible_raw_bytes"] = 10000000
    quotas[1]["required_bytes"] = 1
    result = capacity.evaluate(candidates, quotas, axes, [7, 10])
    assert result["status"] == "FAIL"
    assert result["cells"][1]["eligible_raw_candidate_bytes"] == 0


def test_missing_or_duplicate_quota_and_unknown_cell_are_rejected():
    candidates, quotas, axes = inputs()
    with pytest.raises(ValueError, match="explicitly cover"):
        capacity.evaluate(candidates, quotas[:1], axes, [7, 10])
    with pytest.raises(ValueError, match="duplicate quota"):
        capacity.evaluate(candidates, quotas + quotas[:1], axes, [7, 10])
    candidates[0]["domain"] = "invented"
    with pytest.raises(ValueError, match="unknown allocation"):
        capacity.evaluate(candidates, quotas, axes, [7, 10])


@pytest.mark.parametrize("fraction", [[0, 10], [11, 10], [1, 0], [True, 10], [0.7, 1]])
def test_invalid_retention_rejected(fraction):
    with pytest.raises(ValueError):
        capacity.evaluate(*inputs(), fraction)


def test_raw_bytes_must_be_nonnegative_integer_and_eligibility_is_enforced():
    candidates, quotas, axes = inputs()
    candidates[0]["eligible_raw_bytes"] = -1
    with pytest.raises(ValueError, match="nonnegative integer"):
        capacity.evaluate(candidates, quotas, axes, [7, 10])
    candidates[0]["eligible_raw_bytes"] = 100
    candidates[0]["eligible"] = False
    result = capacity.evaluate(candidates, quotas, axes, [7, 10])
    assert result["status"] == "FAIL"
    assert result["cells"][0]["eligible_raw_candidate_bytes"] == 0


def test_duplicate_content_cannot_inflate_capacity_or_cross_cells():
    candidates, quotas, axes = inputs()
    other = dict(candidates[0], source_unit="another-document")
    result = capacity.evaluate(candidates + [other], quotas, axes, [7, 10])
    assert result["duplicate_identities_not_counted"] == 1
    assert result == capacity.evaluate([other] + candidates, quotas, axes, [7, 10])
    assert result["cells"][0]["eligible_raw_candidate_bytes"] == 100
    other["split"] = "test"
    with pytest.raises(ValueError, match="conflicting cell/size"):
        capacity.evaluate(candidates + [other], quotas, axes, [7, 10])


def test_source_unit_cannot_span_splits_even_with_distinct_content():
    candidates, quotas, axes = inputs()
    other = dict(candidates[0], identity="different-hash", split="test")
    with pytest.raises(ValueError, match="source unit crosses splits"):
        capacity.evaluate(candidates + [other], quotas, axes, [7, 10])


def test_candidate_order_does_not_change_result_and_empty_files_add_no_capacity():
    candidates, quotas, axes = inputs()
    candidates.append(dict(candidates[0], identity="empty", eligible_raw_bytes=0))
    assert capacity.evaluate(candidates, quotas, axes, [7, 10]) == capacity.evaluate(
        list(reversed(candidates)), list(reversed(quotas)), axes, [7, 10]
    )


def test_hash_mismatch_and_outside_root_rejected(tmp_path):
    p = tmp_path / "data.json"
    p.write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        capacity.checked_json(tmp_path, {"path": "data.json", "sha256": "0" * 64})
    with pytest.raises(ValueError, match="escapes"):
        capacity.checked_json(tmp_path, {"path": "../outside.json", "sha256": "0" * 64})


def contract():
    return json.loads((ROOT / "docs/authorization_capacity_2m_r2.json").read_text())


def test_real_revision_preserves_budgets_and_exposes_remaining_math_holdout_deficits():
    result = capacity.audit(ROOT, contract())
    assert len(result["cells"]) == 60
    assert all(c["pass"] for c in result["cells"] if c["domain"] == "documentation")
    failures = {(c["source"], c["domain"], c["split"]) for c in result["cells"] if not c["pass"]}
    assert failures == {
        ("sympy", "structured_math", "validation"),
        ("sympy", "structured_math", "test"),
    }
    assert result == json.loads((ROOT / "reports/authorization_capacity_2m_r2.json").read_text())


def test_original_missing_documentation_regression_and_immutable_input(monkeypatch):
    original = contract()
    checked = capacity.checked_json

    def without_additions(root, ref):
        value = checked(root, ref)
        if ref == original["documentation_overlay"]:
            return {"files": []}
        return value

    monkeypatch.setattr(capacity, "checked_json", without_additions)
    result = capacity.audit(ROOT, original)
    by_cell = {(c["source"], c["domain"], c["split"]): c for c in result["cells"]}
    for s in ["cpython", "numpy"]:
        assert by_cell[s, "documentation", "train"]["eligible_raw_candidate_bytes"] == 0
    for split in ["validation", "test"]:
        assert (
            sum(
                c["eligible_raw_candidate_bytes"]
                for c in result["cells"]
                if c["domain"] == "documentation" and c["split"] == split
            )
            == 0
        )
    assert original == contract()


def test_changing_approved_budgets_to_make_capacity_pass_is_rejected():
    c = contract()
    c["source_totals"]["sympy"]["validation"] = 1
    with pytest.raises(ValueError, match="budgets changed"):
        capacity.audit(ROOT, c)
    c = contract()
    c["quotas"][0]["required_bytes"] += 1
    with pytest.raises(ValueError, match="training cell changed"):
        capacity.audit(ROOT, c)


def test_cli_returns_nonzero_for_infeasible_authorization():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/authorization_capacity.py"),
            str(ROOT / "docs/authorization_capacity_2m_r2.json"),
            "--root",
            str(ROOT),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert json.loads(result.stdout)["status"] == "FAIL"
    assert result.stderr == ""


def test_same_totals_cannot_hide_changed_approved_training_cells():
    c = contract()
    changes = {
        ("cpython", "code"): 1,
        ("cpython", "documentation"): -1,
        ("numpy", "code"): -1,
        ("numpy", "documentation"): 1,
    }
    for q in c["quotas"]:
        if q["split"] == "train":
            q["required_bytes"] += changes.get((q["source"], q["domain"]), 0)
    with pytest.raises(ValueError, match="training cell changed"):
        capacity.audit(ROOT, c)


def test_documentation_capacity_cannot_exceed_pinned_file_size(monkeypatch):
    c = contract()
    checked = capacity.checked_json

    def inflated(root, ref):
        value = checked(root, ref)
        if ref == c["documentation_overlay"]:
            f = next(f for f in value["files"] if f["decision"] == "ADMIT")
            f["estimated_eligible_bytes"] = f["raw_bytes_from_tree"] + 1
        return value

    monkeypatch.setattr(capacity, "checked_json", inflated)
    with pytest.raises(ValueError, match="exceeds raw file bytes"):
        capacity.audit(ROOT, c)
