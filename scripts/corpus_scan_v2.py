"""Resumable exhaustive v2 audit of the frozen post-dedup pool; no admission edits."""

import json
import time
from collections import Counter

from corpus_acquisition import PILOT, ROOT, digest, read, require_space, write_new
from corpus_contamination_v2 import load_v2_index
from corpus_global import SNAPSHOT_SHA256, accounting, candidates, capacity, load_stage

POLICY = ROOT / "docs/contamination_policy_v2.json"
OUTPUT = PILOT / "processed/contamination-v2"


def main():
    policy = read(POLICY)
    policy_hash = digest(POLICY.read_bytes())
    for relative, expected in policy["implementation_sha256"].items():
        if digest((ROOT / relative).read_bytes()) != expected:
            raise ValueError("V2 implementation differs from frozen policy")
    for relative, expected in policy["preserved_v1_sha256"].items():
        if digest((ROOT / relative).read_bytes()) != expected:
            raise ValueError("Preserved v1 evidence changed")
    duplicate = load_stage("dedup")
    documents = [d for d in candidates() if duplicate["decisions"][d["identity"]]["keep"]]
    if len(documents) != 3438:
        raise ValueError("Wrong original post-dedup snapshot")
    print("Verified original 3438 candidates; loading frozen content index", flush=True)
    index, registry, fingerprint = load_v2_index()
    records = []
    for number, doc in enumerate(documents, 1):
        key = digest(doc["identity"].encode())
        path = OUTPUT / "documents" / (key + ".json")
        if path.exists():
            record = read(path)
            if (
                record["candidate"] != doc["identity"]
                or record["cleaned_sha256"] != doc["cleaned_sha256"]
                or record["policy_sha256"] != policy_hash
                or digest(path.read_bytes()) != read(path.with_suffix(".sha256.json"))["sha256"]
            ):
                raise ValueError("Resume evidence identity mismatch")
        else:
            start = time.monotonic()
            matches = index.match_all(doc["payload"].decode("utf-8"))
            record = {
                "candidate": doc["identity"],
                "source_unit": doc["source_unit"],
                "source": doc["source"],
                "domain": doc["domain"],
                "split": doc["split"],
                "cleaned_sha256": doc["cleaned_sha256"],
                "bytes": len(doc["payload"]),
                "policy_sha256": policy_hash,
                "matches": matches,
                "scan_seconds": time.monotonic() - start,
            }
            require_space(len(json.dumps(record).encode()) + 4096)
            write_new(path, record)
            write_new(path.with_suffix(".sha256.json"), {"sha256": digest(path.read_bytes())})
        records.append(record)
        if number % 50 == 0:
            print(f"V2 exhaustive scan {number}/3438", flush=True)
    excluded = {r["source_unit"] for r in records if any(m["actionable"] for m in r["matches"])}
    kept = [d for d in documents if d["source_unit"] not in excluded]
    counts = Counter()
    hashes = set()
    for r in records:
        for m in r["matches"]:
            counts["all_document_content_matches"] += 1
            counts["raw_exact_document_content_matches"] += m["raw_exact"]
            hashes.add(m["normalized_sha256"])
            if not m["actionable"]:
                counts["non_actionable_support_import_matches"] += 1
                counts["support_matches_in_otherwise_excluded_units"] += (
                    r["source_unit"] in excluded
                )
                continue
            counts["actionable_document_content_matches"] += 1
            counts["actionable_normalized_only_matches"] += (
                not m["raw_exact"] and "JACCARD" not in m["method"]
            )
            counts["actionable_near_matches"] += "JACCARD" in m["method"]
            for role in {
                a["field_class"] for a in m["associations"] if not a["non_actionable_import"]
            }:
                counts["actionable_" + role + "_matches"] += 1
    cells = capacity(kept)
    before = {
        (c["source"], c["domain"], c["split"]): c["available_bytes"] for c in capacity(documents)
    }
    failures = [
        dict(
            c,
            pre_contamination_bytes=before[c["source"], c["domain"], c["split"]],
            excluded_bytes=before[c["source"], c["domain"], c["split"]] - c["available_bytes"],
        )
        for c in cells
        if c["deficit"]
    ]
    result = {
        "schema_version": 2,
        "policy_sha256": policy_hash,
        "candidate_filter_snapshot_sha256": SNAPSHOT_SHA256,
        "registry": registry,
        "fingerprints": fingerprint,
        "documents_scanned": len(documents),
        "source_units_scanned": len({d["source_unit"] for d in documents}),
        "bytes_scanned": sum(len(d["payload"]) for d in documents),
        "counts": dict(counts),
        "unique_matched_content_identities": len(hashes),
        "excluded_source_units": sorted(excluded),
        "excluded_bytes": sum(len(d["payload"]) for d in documents if d["source_unit"] in excluded),
        "post_exclusion": accounting(kept),
        "cells": cells,
        "failures": failures,
        "document_evidence": {
            r["candidate"]: {
                "path": (OUTPUT / "documents" / (digest(r["candidate"].encode()) + ".json"))
                .relative_to(ROOT)
                .as_posix(),
                "sha256": digest(
                    (
                        OUTPUT / "documents" / (digest(r["candidate"].encode()) + ".json")
                    ).read_bytes()
                ),
            }
            for r in records
        },
        "match_count_semantics": "Unique document/content identities, not every repeated span. "
        "All benchmark associations retained. Exact evidence dominates redundant near evidence "
        "for the same identity. Role counts may overlap; raw exact is a subset "
        "of normalized exact.",
        "limitations": "Frozen bounded paragraph/window near retrieval; "
        "no semantic completeness claim.",
    }
    destination = ROOT / "reports/contamination_v2_existing_pool.json"
    write_new(destination, result)
    write_new(destination.with_suffix(".sha256.json"), {"sha256": digest(destination.read_bytes())})
    print(
        json.dumps(
            {"passing_cells": 60 - len(failures), "failures": failures, "counts": dict(counts)}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
