"""Audit a completed early-filter pass without fetching or changing payloads."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def audit(root, source):
    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))

    receipts = sorted((root / "provenance/artifacts" / source).glob("*.json"))
    totals = Counter()
    removed = Counter()
    exclusions = Counter()
    capacities = Counter()
    units = Counter()
    transformations = []
    identities = {}
    for path in receipts:
        receipt = read(path)
        record_path = root / "processed/early-v1_3" / source / path.name
        record = read(record_path)
        raw = (root / receipt["raw_path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == receipt["raw_sha256"]
        assert (
            hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
            == receipt["git_blob_sha1"]
        )
        totals["raw_files"] += 1
        totals["raw_bytes"] += len(raw)
        if record["reason"] not in {
            "UNAPPROVED_FILE_LICENSE",
            "COPIED_MATERIAL_REQUIRES_SEPARATE_RIGHTS",
        }:
            totals["license_eligible_files"] += 1
            totals["license_eligible_bytes"] += len(raw)
        identities[path.name] = hashlib.sha256(record_path.read_bytes()).hexdigest()
        ranges = sorted(record["removed_ranges"])
        previous = 0
        reconstructed = []
        for start, stop, reason in ranges:
            assert previous <= start < stop <= len(raw)
            reconstructed.append(raw[previous:start])
            previous = stop
        reconstructed.append(raw[previous:])
        if record["decision"] == "ELIGIBLE_FOR_DEDUP":
            payload = (root / record["payload_path"]).read_bytes()
            assert b"".join(reconstructed) == payload
            assert hashlib.sha256(payload).hexdigest() == record["cleaned_sha256"]
            totals["cleaned_files"] += 1
            totals["cleaned_bytes"] += len(payload)
            key = (receipt["domain"], receipt["reserved_split"])
            capacities[key] += len(payload)
            units[(*key, receipt["source_unit"])] += len(payload)
            for start, stop, reason in ranges:
                removed[reason] += stop - start
        else:
            exclusions[record["reason"]] += 1
            # Entire rejected document is charged once, not again by partial spans.
            removed[record["reason"]] += len(raw)
        if receipt["reserved_split"] != "train":
            transformations.append(
                {
                    "path": receipt["path"],
                    "source_unit": receipt["source_unit"],
                    "split": receipt["reserved_split"],
                    "decision": record["decision"],
                    "reason": record["reason"],
                    "removed_ranges": ranges,
                    "raw_sha256": receipt["raw_sha256"],
                    "cleaned_bytes": record.get("cleaned_bytes", 0),
                }
            )
    assert totals["raw_bytes"] - totals["cleaned_bytes"] == sum(removed.values())
    quotas = read(root / "provenance/authorization/docs/authorization_capacity_2m_r3.json")[
        "quotas"
    ]
    cells = []
    for quota in quotas:
        if quota["source"] == source:
            measured = capacities[quota["domain"], quota["split"]]
            cells.append(
                dict(
                    quota,
                    measured_early_clean_bytes=measured,
                    deficit=max(0, quota["required_bytes"] - measured),
                )
            )
    return {
        "schema_version": 1,
        "policy": "pilot-clean-v1.3",
        "source": source,
        "totals": dict(totals),
        "exclusions_files": dict(exclusions),
        "removed_bytes_exclusive_categories": dict(removed),
        "measured_early_retention": totals["cleaned_bytes"] / totals["raw_bytes"],
        "cells": cells,
        "source_unit_capacity": [
            dict(domain=d, split=s, source_unit=u, cleaned_bytes=n)
            for (d, s, u), n in sorted(units.items())
        ],
        "holdout_transformations": transformations,
        "record_hashes": identities,
        "capacity_status": "FAIL" if any(c["deficit"] for c in cells) else "PASS",
        "limitations": (
            "Early cleaning only. Deduplication, benchmark exclusion and final corpus "
            "are not certified. Per-unit capacity is reported; quotas apply to frozen "
            "source/domain/split pools."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = audit(Path(__file__).resolve().parents[1] / "data/pilot-2m-r1", args.source)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ("totals", "capacity_status", "cells")}))
    raise SystemExit(1 if result["capacity_status"] == "FAIL" else 0)
