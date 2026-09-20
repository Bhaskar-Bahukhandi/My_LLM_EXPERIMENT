"""Run frozen pilot duplicate/exclusion gates with reusable immutable stage evidence."""

import argparse
import json
from collections import Counter, defaultdict

from corpus_acquisition import (
    AUTH,
    PILOT,
    ROOT,
    digest,
    read,
    require_space,
    source_files,
    write_new,
)
from corpus_benchmark_index import load_index
from corpus_duplicates import deduplicate

SNAPSHOT_SHA256 = "7f4a00770a3ad22c84a62489d15a18ef625292a083665183753fa44a21d4036f"


def candidates():
    snapshot = PILOT / "provenance/final_filter_snapshot.json"
    if digest(snapshot.read_bytes()) != SNAPSHOT_SHA256:
        raise ValueError("Frozen final filtering snapshot changed")
    expected = {
        (source, f["path"]): f
        for source in ("cpython", "numpy", "sympy", "mathlib")
        for f in source_files(source)
    }
    documents = []
    roles = {}
    for relative, checksum in read(snapshot)["records"].items():
        path = PILOT / relative
        if digest(path.read_bytes()) != checksum:
            raise ValueError("Frozen filtering record changed")
        record = read(path)
        if record["decision"] != "ELIGIBLE_FOR_DEDUP":
            continue
        receipt = read(PILOT / record["artifact_receipt"])
        if record.get("source") == "gutenberg":
            source = "gutenberg"
            identity = f"gutenberg/pg{record['ebook_id']}"
            domain, split, unit = record["domain"], record["reserved_split"], record["source_unit"]
            pin = receipt["authorization"]["files"][0]["url"]
        else:
            source = receipt["source"]
            authorization = expected[source, receipt["path"]]
            if receipt["file_authorization"] != authorization:
                raise ValueError("Receipt authorization differs from frozen file allowlist")
            for field in ("source", "path", "domain", "reserved_split", "source_unit", "commit"):
                if receipt[field] != authorization[field]:
                    raise ValueError("Receipt metadata differs from frozen authorization")
            identity = source + "/" + receipt["path"]
            domain, split, unit = (
                receipt["domain"],
                receipt["reserved_split"],
                receipt["source_unit"],
            )
            pin = receipt["commit"]
        unit = source + ":" + unit
        if unit in roles and roles[unit] != split:
            raise ValueError("A reserved source unit crosses candidate splits")
        roles[unit] = split
        payload = (PILOT / record["payload_path"]).read_bytes()
        if len(payload) != record["cleaned_bytes"] or digest(payload) != record["cleaned_sha256"]:
            raise ValueError("Cleaned candidate payload changed")
        documents.append(
            {
                "identity": identity,
                "source": source,
                "domain": domain,
                "split": split,
                "source_unit": unit,
                "pin": pin,
                "payload": payload,
                "payload_path": record["payload_path"],
                "cleaned_sha256": record["cleaned_sha256"],
                "raw_sha256": receipt["raw_sha256"],
                "filter_record": relative,
                "artifact_receipt": record["artifact_receipt"],
                "cleaning_policy": record["policy"],
            }
        )
    return sorted(documents, key=lambda d: d["identity"])


def accounting(documents):
    counts = Counter()
    records = Counter()
    for doc in documents:
        key = doc["source"], doc["domain"], doc["split"]
        counts[key] += len(doc["payload"])
        records[key] += 1
    return [
        dict(source=s, domain=d, split=k, bytes=n, records=records[s, d, k])
        for (s, d, k), n in sorted(counts.items())
    ]


def capacity(documents):
    available = {(r["source"], r["domain"], r["split"]): r["bytes"] for r in accounting(documents)}
    return [
        dict(
            q,
            available_bytes=available.get((q["source"], q["domain"], q["split"]), 0),
            deficit=max(
                0, q["required_bytes"] - available.get((q["source"], q["domain"], q["split"]), 0)
            ),
        )
        for q in read(AUTH / "docs/authorization_capacity_2m_r3.json")["quotas"]
    ]


def stage_path(name):
    return PILOT / "processed/global-v1" / (name + ".json")


def save_stage(name, value):
    value["filter_snapshot_sha256"] = SNAPSHOT_SHA256
    value["implementation_sha256"] = {
        name: digest((ROOT / "scripts" / name).read_bytes())
        for name in (
            "corpus_global.py",
            "corpus_duplicates.py",
            "corpus_contamination.py",
            "corpus_benchmark_index.py",
            "benchmark_fingerprints.py",
        )
    }
    require_space(len(json.dumps(value).encode("utf-8")) + 4096)
    write_new(stage_path(name), value)
    write_new(
        stage_path(name).with_suffix(".sha256.json"),
        {"sha256": digest(stage_path(name).read_bytes())},
    )


def load_stage(name):
    path = stage_path(name)
    if digest(path.read_bytes()) != read(path.with_suffix(".sha256.json"))["sha256"]:
        raise ValueError("Immutable stage evidence changed")
    result = read(path)
    if result["filter_snapshot_sha256"] != SNAPSHOT_SHA256:
        raise ValueError("Stage belongs to a different filtering snapshot")
    return result


def run_dedup(documents):
    result = deduplicate(documents)
    exact = [d for d in documents if result["exact_decisions"][d["identity"]]["keep"]]
    kept = [d for d in documents if result["decisions"][d["identity"]]["keep"]]
    result.update(
        before=accounting(documents),
        post_exact=accounting(exact),
        post_near=accounting(kept),
        cells=capacity(kept),
    )
    save_stage("dedup", result)
    return result


def run_contamination(documents):
    duplicate = load_stage("dedup")
    if any(c["deficit"] for c in duplicate["cells"]):
        raise ValueError("Deduplication capacity failed; stop before downstream processing")
    surviving = [d for d in documents if duplicate["decisions"][d["identity"]]["keep"]]
    index, registry = load_index(ROOT)
    hits = []
    excluded_units = set()
    by_unit = defaultdict(list)
    for number, doc in enumerate(surviving, 1):
        by_unit[doc["source_unit"]].append(doc["identity"])
        match = index.match(doc["payload"].decode("utf-8"))
        if match is not None:
            excluded_units.add(doc["source_unit"])
            hits.append(dict(identity=doc["identity"], source_unit=doc["source_unit"], **match))
        if number % 100 == 0:
            print(f"Benchmark scan: {number}/{len(surviving)} documents", flush=True)
    kept = [d for d in surviving if d["source_unit"] not in excluded_units]
    result = {
        "registry": registry,
        "documents_scanned": len(surviving),
        "source_units_scanned": len(by_unit),
        "hits": hits,
        "excluded_source_units": sorted(excluded_units),
        "post_exclusion": accounting(kept),
        "cells": capacity(kept),
        "unresolved_hits": 0,
        "resolution": "Every matched source unit excluded in full",
        "limitations": (
            "Frozen textual fingerprints and bounded paragraph/window retrieval; "
            "no semantic completeness claim."
        ),
    }
    save_stage("contamination", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("dedup", "contamination"))
    args = parser.parse_args()
    if stage_path(args.stage).exists():
        result = load_stage(args.stage)
        print("Reused verified completed stage", args.stage)
    else:
        documents = candidates()
        print(f"Loaded {len(documents)} frozen eligible documents", flush=True)
        result = run_dedup(documents) if args.stage == "dedup" else run_contamination(documents)
    failures = [c for c in result["cells"] if c["deficit"]]
    print(
        json.dumps(
            {"stage": args.stage, "passing_cells": 60 - len(failures), "failures": failures}
        ),
        flush=True,
    )
    raise SystemExit(1 if failures else 0)
