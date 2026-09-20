"""Freeze exact byte quotas only after real capacity and independent leakage gates."""

import hashlib
from collections import defaultdict
from itertools import combinations

from benchmark_fingerprints import canonical
from corpus_acquisition import PILOT, ROOT, digest, read, require_space, write_new
from corpus_duplicates import deduplicate

from unified_edge.training.data import (
    BatchStream,
    DatasetManifest,
    Window,
    WindowDataset,
    batches_by_length,
    file_bytes,
)


def select_documents(documents, quotas):
    roles = {}
    for doc in documents:
        if doc["source_unit"] in roles and roles[doc["source_unit"]] != doc["split"]:
            raise ValueError("Source unit crosses splits")
        roles[doc["source_unit"]] = doc["split"]
    selected, cells = [], []
    for quota in quotas:
        pool = sorted(
            (d for d in documents if all(d[k] == quota[k] for k in ("source", "domain", "split"))),
            key=lambda d: d["identity"],
        )
        available = sum(len(d["payload"]) for d in pool)
        remaining = quota["required_bytes"]
        if available < remaining:
            raise ValueError("Insufficient cell capacity; no borrowing")
        for doc in pool:
            count = min(remaining, len(doc["payload"]))
            if count:
                selected.append(
                    dict(doc, selected_payload=doc["payload"][:count], selected_range=[0, count])
                )
                remaining -= count
        cells.append(
            dict(
                quota,
                available_bytes=available,
                selected_bytes=quota["required_bytes"],
                unused_eligible_bytes=available - quota["required_bytes"],
            )
        )
    return selected, cells


def verify_manifest(manifest, root):
    split_hashes = {s: hashlib.sha256() for s in ("train", "validation", "test")}
    totals = defaultdict(int)
    cross_split = {}
    for doc in manifest["documents"]:
        raw = file_bytes(root, doc["path"])
        if len(raw) != doc["byte_count"] or digest(raw) != doc["sha256"]:
            raise ValueError("Frozen corpus byte mutation")
        previous = cross_split.setdefault(doc["sha256"], doc["split"])
        if previous != doc["split"]:
            raise ValueError("Selected byte-identical documents cross splits")
        totals[doc["split"]] += len(raw)
        split_hashes[doc["split"]].update(raw)
    if dict(totals) != manifest["split_bytes"]:
        raise ValueError("Frozen corpus totals differ")
    if {s: h.hexdigest() for s, h in split_hashes.items()} != manifest["split_sha256"]:
        raise ValueError("Frozen corpus ordered split hash differs")
    return dict(totals)


class HeldoutWindows:
    """Test-only view using the existing byte reader and Window/BatchStream primitives."""

    def __init__(self, manifest, root, length):
        payloads = {
            d["path"]: file_bytes(root, d["path"])
            for d in manifest["documents"]
            if d["split"] == "test"
        }
        self.windows = tuple(
            Window(d["path"], start, payloads[d["path"]][start : start + length])
            for d in manifest["documents"]
            if d["split"] == "test"
            for start in range(0, d["byte_count"], length)
        )

    def __len__(self):
        return len(self.windows)


def loader_check(manifest, root):
    verify_manifest(manifest, root)
    files = [
        (d["path"], d["split"], d["domain"]) for d in manifest["documents"] if d["split"] != "test"
    ]
    training = DatasetManifest.create(root, manifest["corpus_id"], files)
    rows = []
    for length in (32, 128):
        for split in ("train", "validation", "test"):
            data = (
                HeldoutWindows(manifest, root, length)
                if split == "test"
                else WindowDataset(training, root, split, length)
            )
            by_document = defaultdict(bytearray)
            for w in data.windows:
                by_document[w.document].extend(w.payload)
            if any(
                bytes(payload) != file_bytes(root, path) for path, payload in by_document.items()
            ):
                raise ValueError("Window byte round-trip failed")
            first, repeated = BatchStream(data, 400, 4), BatchStream(data, 400, 4)
            if first.order != repeated.order:
                raise ValueError("Nondeterministic data ordering")
            left, right = first.next_batch(), repeated.next_batch()
            if left != right:
                raise ValueError("Nondeterministic first batch")
            for tensor in batches_by_length(left):
                if tensor.min().item() < 0 or tensor.max().item() > 255:
                    raise ValueError("Reserved control ID in raw targets")
            rows.append(
                {
                    "split": split,
                    "length": length,
                    "windows": len(data),
                    "bytes_round_tripped": sum(len(v) for v in by_document.values()),
                    "order_sha256": digest(canonical(first.order)),
                    "targets": "raw bytes 0..255",
                }
            )
    return training, rows


def main():
    pool_path = ROOT / "reports/corpus_mode_v2_pool.json"
    pool = read(pool_path)
    if digest(pool_path.read_bytes()) != read(pool_path.with_suffix(".sha256.json"))["sha256"]:
        raise ValueError("Pool report changed")
    if len(pool["cells"]) != 60 or any(c["deficit"] for c in pool["cells"]):
        raise ValueError("Mandatory 60/60 capacity gate failed")
    documents = []
    for row in pool["survivors"]:
        raw = (PILOT / row["payload_path"]).read_bytes()
        if digest(raw) != row["cleaned_sha256"]:
            raise ValueError("Source candidate mutation")
        documents.append(dict(row, payload=raw))
    print("Independent assigned-source exact/normalized/near leakage audit", flush=True)
    audited = deduplicate(documents)
    by_id = {d["identity"]: d for d in documents}
    pairs = []
    for left, right in combinations(("train", "validation", "test"), 2):
        edges = [
            e
            for e in audited["verified_edges"]
            if {by_id[e["left"]]["split"], by_id[e["right"]]["split"]} == {left, right}
        ]
        pairs.append(
            {
                "left": left,
                "right": right,
                "raw_exact": sum(e["method"] == "raw" for e in edges),
                "normalized_exact": sum(e["method"] == "normalized" for e in edges),
                "near": sum(e["method"] == "pilot-near-v1" for e in edges),
                "affected_bytes": sum(
                    len(by_id[i]["payload"])
                    for i in {e[k] for e in edges for k in ("left", "right")}
                ),
            }
        )
    audit = {
        "scope": "Complete assigned source documents before byte selection",
        "pool_sha256": digest(pool_path.read_bytes()),
        "pairs": pairs,
        "unresolved": sum(p["raw_exact"] + p["normalized_exact"] + p["near"] for p in pairs),
        "limitation": audited["limitation"],
    }
    write_new(ROOT / "reports/final_source_leakage_audit.json", audit)
    if audit["unresolved"]:
        raise ValueError("Cross-split leakage gate failed")
    selected, cells = select_documents(documents, pool["cells"])
    require_space(sum(len(d["selected_payload"]) for d in selected) + 2000000)
    licenses = {
        s["source"]: s["license"]
        for s in read(ROOT / "docs/source_file_allowlists_2m.json")["sources"]
    }
    licenses["gutenberg"] = (
        "Approved work-level public-domain clearance; original text only, "
        "edition supplements removed"
    )
    records = []
    hashes = {s: hashlib.sha256() for s in ("train", "validation", "test")}
    totals = defaultdict(int)
    for doc in selected:
        payload = doc["selected_payload"]
        path = "final-corpus-v1/" + doc["split"] + "/" + digest(doc["identity"].encode()) + ".raw"
        target = PILOT / path
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(payload)
        receipt = read(PILOT / doc["artifact_receipt"])
        records.append(
            {
                "path": path,
                "source_identity": doc["identity"],
                "source": doc["source"],
                "source_revision": doc["pin"],
                "source_unit": doc["source_unit"],
                "domain": doc["domain"],
                "split": doc["split"],
                "sha256": digest(payload),
                "byte_count": len(payload),
                "document_count": 1,
                "raw_sha256": doc["raw_sha256"],
                "complete_cleaned_sha256": doc["cleaned_sha256"],
                "selected_range": doc["selected_range"],
                "range_basis": "cleaned bytes, half-open; no re-encoding",
                "cleaning_policy": doc["cleaning_policy"],
                "filter_record": doc["filter_record"],
                "filter_record_sha256": digest((PILOT / doc["filter_record"]).read_bytes()),
                "artifact_receipt": doc["artifact_receipt"],
                "artifact_receipt_sha256": digest((PILOT / doc["artifact_receipt"]).read_bytes()),
                "license_classification": licenses[doc["source"]],
                "historical_acquisition_authorization": receipt.get(
                    "file_authorization", receipt.get("authorization")
                ),
                "file_review": doc.get("file_review"),
            }
        )
        hashes[doc["split"]].update(payload)
        totals[doc["split"]] += len(payload)
    manifest = {
        "schema": "edge-real-corpus-1",
        "corpus_id": "pilot-2m-final-v1",
        "data_root": "configured pilot root",
        "documents": records,
        "cells": cells,
        "split_bytes": dict(totals),
        "split_sha256": {s: h.hexdigest() for s, h in hashes.items()},
        "split_hash_definition": "SHA-256 of selected document bytes concatenated "
        "in this manifest's order, without separators",
        "pool_sha256": digest(pool_path.read_bytes()),
        "mode_policy_sha256": digest((ROOT / "docs/file_mode_policy_v2.json").read_bytes()),
        "contamination_policy_sha256": digest(
            (ROOT / "docs/contamination_policy_v2.json").read_bytes()
        ),
        "exclusion_registry_sha256": digest(
            (ROOT / "docs/data_exclusion_registry.json").read_bytes()
        ),
        "leakage_audit_sha256": digest(
            (ROOT / "reports/final_source_leakage_audit.json").read_bytes()
        ),
    }
    if manifest["split_bytes"] != {"train": 10000000, "validation": 500000, "test": 500000}:
        raise ValueError("Final exact quotas differ")
    verify_manifest(manifest, PILOT)
    path = PILOT / "final-corpus-v1/manifest.json"
    write_new(path, manifest)
    identity = {
        "raw_sha256": digest(path.read_bytes()),
        "canonical_sha256": digest(canonical(manifest)),
    }
    write_new(path.with_suffix(".sha256.json"), identity)
    training, loader = loader_check(manifest, PILOT)
    write_new(PILOT / "final-corpus-v1/training_manifest.json", training.to_dict())
    write_new(
        ROOT / "reports/final_corpus_loader_audit.json",
        {
            "manifest": identity,
            "checks": loader,
            "native_loader_scope": "DatasetManifest/WindowDataset for train and validation; "
            "test-only view uses existing file_bytes, Window, BatchStream and batching. "
            "Test never relabeled as validation.",
            "model_or_optimizer_created": False,
        },
    )
    print(
        {"manifest": identity, "split_hashes": manifest["split_sha256"], "bytes": dict(totals)},
        flush=True,
    )


if __name__ == "__main__":
    main()
