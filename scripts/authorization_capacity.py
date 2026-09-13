"""Offline authorization feasibility; estimates never replace acquisition gates."""

import argparse
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path


def integer(value, label):
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def evaluate(candidates, quotas, axes, retention):
    """Check every cell, with exact integer rounding and no shared-unit leakage.

    Each candidate has one identity, one source unit, one cell and a raw byte
    estimate. Duplicate content counts once; conflicting assignments fail closed.
    H = ceil(Q / retained_fraction) - Q. Zero-quota cells require zero headroom.
    """
    if set(axes) != {"source", "domain", "split"}:
        raise ValueError("expected source/domain/split axes")
    for name, values in axes.items():
        if not values or any(not isinstance(v, str) or not v for v in values):
            raise ValueError(f"invalid {name} axis")
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate {name} axis")
    num, den = retention
    integer(num, "retention numerator")
    integer(den, "retention denominator")
    if not 0 < num <= den:
        raise ValueError("retention must be in (0, 1]")
    cells = list(itertools.product(*(axes[k] for k in ("source", "domain", "split"))))
    required = dict.fromkeys(cells, 0)

    def cell(row):
        key = tuple(row[k] for k in ("source", "domain", "split"))
        if key not in required:
            raise ValueError(f"unknown allocation cell: {key}")
        return key

    seen_quotas = set()
    for row in quotas:
        key = cell(row)
        if key in seen_quotas:
            raise ValueError(f"duplicate quota: {key}")
        seen_quotas.add(key)
        required[key] = integer(row["required_bytes"], "quota")
    # Explicit zero entries prevent a missing quota from silently becoming zero.
    if seen_quotas != set(cells):
        raise ValueError("quota matrix must explicitly cover every cell, including zeros")
    units = {}
    contents = {}
    raw = defaultdict(int)
    counts = defaultdict(set)
    duplicates = 0
    for row in sorted(candidates, key=lambda r: (r["source"], r["source_unit"], r["identity"])):
        key = cell(row)
        size = integer(row["eligible_raw_bytes"], "candidate bytes")
        if not row["identity"] or not row["source_unit"]:
            raise ValueError("candidate identity and source unit required")
        if type(row["eligible"]) is not bool:
            raise ValueError("eligibility must be a boolean")
        if not row["eligible"]:
            continue
        unit = (row["source"], row["source_unit"])
        if unit in units and units[unit] != row["split"]:
            raise ValueError(f"source unit crosses splits: {unit}")
        units[unit] = row["split"]
        if size == 0:
            continue
        identity = row["identity"]
        if identity in contents:
            if contents[identity] != (key, size):
                raise ValueError(f"content identity has conflicting cell/size: {identity}")
            duplicates += 1
            continue
        contents[identity] = (key, size)
        raw[key] += size
        counts[key].add(unit)
    results = []
    for key in cells:
        q = required[key]
        minimum = (q * den + num - 1) // num
        estimated = raw[key] * num // den
        results.append(
            {
                **dict(zip(("source", "domain", "split"), key)),
                "eligible_raw_candidate_bytes": raw[key],
                "estimated_retained_bytes": estimated,
                "required_final_bytes": q,
                "required_raw_headroom_bytes": minimum - q,
                "minimum_raw_candidate_bytes": minimum,
                "raw_surplus_after_headroom_bytes": raw[key] - minimum,
                "estimated_final_surplus_bytes": estimated - q,
                "independent_source_units": len(counts[key]),
                "pass": raw[key] >= minimum,
            }
        )
    return {
        "algorithm": "source-domain-split-capacity-v1",
        "status": "PASS" if all(r["pass"] for r in results) else "FAIL",
        "retention_fraction": [num, den],
        "retention_status": "ESTIMATE; cleaning and deduplication not yet measured",
        "duplicate_identities_not_counted": duplicates,
        "cells": results,
    }


def checked_json(root, ref):
    path = (root / ref["path"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("artifact path escapes project")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != ref["sha256"]:
        raise ValueError(f"artifact hash mismatch: {ref['path']}")
    return json.loads(data)


def audit(root, contract):
    """Load hash-bound approval inputs; no network or training-data reads."""
    base = checked_json(root, contract["base_allowlist"])
    additions = checked_json(root, contract["documentation_overlay"])
    books = checked_json(root, contract["gutenberg_allowlist"])
    previous = checked_json(root, contract["previous_authorization"])
    table = checked_json(root, previous["source_approval_table"])
    approved_train = {row["id"]: row.get("train_domain_bytes", {}) for row in table["source_rows"]}
    for quota in contract["quotas"]:
        if quota["split"] == "train":
            expected = approved_train.get(quota["source"], {}).get(quota["domain"], 0)
            if quota["required_bytes"] != expected:
                raise ValueError("approved source/domain training cell changed")
    expected_domains = {
        domain: {split: amounts[split + "_bytes"] for split in contract["axes"]["split"]}
        for domain, amounts in previous["domain_allocation"].items()
    }
    if (
        contract["source_totals"] != previous["approved_source_allocations"]
        or contract["domain_totals"] != expected_domains
    ):
        raise ValueError("approved source/domain budgets changed")
    if (
        set(contract["axes"]["source"]) != set(contract["source_totals"])
        or set(contract["axes"]["domain"]) != set(contract["domain_totals"])
        or set(contract["axes"]["split"]) != {"train", "validation", "test"}
    ):
        raise ValueError("axes do not cover approved budgets")
    for ref in contract["preserved_evidence"]:
        checked_json(root, ref)
    candidates = []
    paths = set()
    pins = {s["source"]: s["commit"] for s in base["sources"]}
    for source in base["sources"]:
        for f in source["files"]:
            paths.add((source["source"], f["path"]))
            candidates.append(
                {
                    "source": source["source"],
                    "domain": f["domain"],
                    "split": f["reserved_split"],
                    "source_unit": f["source_unit"],
                    "identity": "git-blob-sha1:" + f["git_blob_sha1"],
                    "eligible_raw_bytes": f["raw_bytes_from_tree"],
                    "eligible": True,
                }
            )
    for f in additions["files"]:
        if f["decision"] != "ADMIT":
            continue
        key = (f["source"], f["path"])
        if key in paths or f["commit"] != pins.get(f["source"]):
            raise ValueError(f"duplicate file or changed source pin: {key}")
        if integer(f["estimated_eligible_bytes"], "eligible bytes") > integer(
            f["raw_bytes_from_tree"], "raw file bytes"
        ):
            raise ValueError("eligible capacity exceeds raw file bytes")
        if len(f["sha256"]) != 64 or any(ch not in "0123456789abcdef" for ch in f["sha256"]):
            raise ValueError("documentation SHA-256 required")
        if not f["controlling_license"]:
            raise ValueError("documentation controlling license required")
        paths.add(key)
        candidates.append(
            {
                "source": f["source"],
                "domain": f["domain"],
                "split": f["reserved_split"],
                "source_unit": f["source_unit"],
                "identity": "git-blob-sha1:" + f["git_blob_sha1"],
                "eligible_raw_bytes": f["estimated_eligible_bytes"],
                "eligible": True,
            }
        )
    for work in books["works"]:
        if work["id"] not in contract["approved_gutenberg_ids"]:
            continue
        candidates.append(
            {
                "source": "gutenberg",
                "domain": "general_text",
                "split": work["reserved_split"],
                "source_unit": str(work["id"]),
                "identity": "gutenberg-edition:" + work["files"][0]["url"],
                "eligible_raw_bytes": work["files"][0]["bytes"],
                "eligible": True,
            }
        )
    for source, splits in contract["source_totals"].items():
        for split, target in splits.items():
            actual = sum(
                q["required_bytes"]
                for q in contract["quotas"]
                if q["source"] == source and q["split"] == split
            )
            if actual != target:
                raise ValueError(f"source quota total mismatch: {source}/{split}")
    for domain, splits in contract["domain_totals"].items():
        for split, target in splits.items():
            actual = sum(
                q["required_bytes"]
                for q in contract["quotas"]
                if q["domain"] == domain and q["split"] == split
            )
            if actual != target:
                raise ValueError(f"domain quota total mismatch: {domain}/{split}")
    return evaluate(candidates, contract["quotas"], contract["axes"], contract["retention"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = audit(args.root, json.loads(args.contract.read_text(encoding="utf-8")))
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
