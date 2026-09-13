"""R3 repairs two holdout pools without changing the accepted invariant."""

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
_spec = importlib.util.spec_from_file_location(
    "capacity", ROOT / "scripts/authorization_capacity.py"
)
capacity = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(capacity)


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_all_cells_pass_with_identical_frozen_quota_and_margin():
    old = read("docs/authorization_capacity_2m_r2.json")
    new = read("docs/authorization_capacity_2m_r3.json")
    for field in (
        "quotas",
        "axes",
        "source_totals",
        "domain_totals",
        "retention",
        "headroom_policy",
    ):
        assert new[field] == old[field]
    actual = capacity.audit(ROOT, new)
    assert actual == read("reports/authorization_capacity_2m_r3.json")
    assert len(actual["cells"]) == 60 and all(c["pass"] for c in actual["cells"])
    for cell in actual["cells"]:
        if cell["source"] == "sympy" and cell["domain"] == "structured_math":
            if cell["split"] != "train":
                assert cell["required_final_bytes"] == 25000
                assert cell["minimum_raw_candidate_bytes"] == 35715
                assert cell["required_raw_headroom_bytes"] == 10715
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/authorization_capacity.py"),
            str(ROOT / "docs/authorization_capacity_2m_r3.json"),
            "--root",
            str(ROOT),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == actual


def test_r2_entries_and_accepted_implementation_are_unchanged():
    old = read("docs/documentation_file_allowlist_2m_r2.json")
    new = read("docs/documentation_file_allowlist_2m_r3.json")
    assert new["files"][: len(old["files"])] == old["files"]
    assert len(new["files"]) == len(old["files"]) + 3
    for path, expected in read("reports/download_authorization_2m_r2.json")[
        "implementation"
    ].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected


def test_new_files_have_unique_pinned_identities_and_disjoint_holdout_units():
    base = read("docs/source_file_allowlists_2m.json")
    old = read("docs/documentation_file_allowlist_2m_r2.json")["files"]
    new = read("docs/documentation_file_allowlist_2m_r3.json")["files"][len(old) :]
    existing = [
        (s["source"], f["path"], f["source_unit"]) for s in base["sources"] for f in s["files"]
    ] + [(f["source"], f["path"], f["source_unit"]) for f in old]
    groups = {"validation": set(), "test": set()}
    for f in new:
        assert f["source"] == "sympy" and f["domain"] == "structured_math"
        assert f["commit"] == "b4ce69ad5d40e4e545614b6c76ca9b0be0b98f0b"
        assert f["decision"] == "ADMIT" and len(f["sha256"]) == 64
        assert f["controlling_license"].startswith("BSD-3-Clause")
        assert f["mathematical_domain_justification"] and f["file_local_notices"] == []
        assert not any(
            s == "sympy" and (p == f["path"] or u == f["source_unit"]) for s, p, u in existing
        )
        groups[f["reserved_split"]].add(f["source_unit"])
        end = f["eligible_byte_ranges"][0][1]
        assert f["eligible_byte_ranges"] == [[0, f["estimated_eligible_bytes"]]]
        assert f["known_excluded_ranges"][0]["start"] == end
        assert f["known_excluded_ranges"][0]["end"] == f["raw_bytes_from_tree"]
        assert f["expected_post_cleaning_bytes"] == end * 7 // 10
    assert groups["validation"] and groups["test"]
    assert groups["validation"].isdisjoint(groups["test"])


@pytest.mark.parametrize("split", ["validation", "test"])
def test_removing_new_material_reopens_the_original_failure(monkeypatch, split):
    contract = read("docs/authorization_capacity_2m_r3.json")
    old_count = len(read("docs/documentation_file_allowlist_2m_r2.json")["files"])
    checked = capacity.checked_json

    def without_new_pool(root, ref):
        value = checked(root, ref)
        if ref == contract["documentation_overlay"]:
            value["files"] = value["files"][:old_count] + [
                f for f in value["files"][old_count:] if f["reserved_split"] != split
            ]
        return value

    monkeypatch.setattr(capacity, "checked_json", without_new_pool)
    result = capacity.audit(ROOT, contract)
    failed = [c for c in result["cells"] if not c["pass"]]
    assert [(c["source"], c["domain"], c["split"]) for c in failed] == [
        ("sympy", "structured_math", split)
    ]


def test_history_records_the_quota_correction_without_rewriting_preflight():
    clarification = read("docs/authorization_r3_quota_clarification.json")
    assert clarification["frozen_quotas"] == {"validation": 25000, "test": 25000}
    assert clarification["minimum_eligible_raw_per_cell"] == 35715
    ref = clarification["supersedes_erroneous_instruction"]
    assert hashlib.sha256((ROOT / ref["path"]).read_bytes()).hexdigest() == ref["sha256"]
    assert read(ref["path"])["conflict"]["new_directive_math_quota_each_holdout"] == 50000
