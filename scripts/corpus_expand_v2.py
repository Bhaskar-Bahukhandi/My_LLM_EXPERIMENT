"""Apply reviewed additions through unchanged filters, global dedup and v2 matching."""

from collections import Counter

from corpus_acquisition import PILOT, ROOT, digest, read, require_space, write_new
from corpus_contamination_v2 import load_v2_index
from corpus_duplicates import deduplicate
from corpus_filtering import early_filter
from corpus_global import accounting, candidates, capacity


def main():
    plan = read(ROOT / "docs/targeted_candidates_v2_r1.json")
    review = read(ROOT / "docs/targeted_file_review_v2_r1.json")
    old = read(ROOT / "reports/contamination_v2_existing_pool.json")
    documents = candidates()
    original_ids = {d["identity"] for d in documents}
    filtered = []
    for file in plan["files"]:
        identity = file["source"] + "/" + file["path"]
        if identity in original_ids:
            raise ValueError("Addition repeats an existing file")
        inspected = review["files"][identity]
        receipt_path = (
            PILOT
            / "provenance/artifacts"
            / file["source"]
            / (digest(file["path"].encode()) + ".json")
        )
        receipt = read(receipt_path)
        raw = (PILOT / receipt["raw_path"]).read_bytes()
        if (
            digest(raw) != inspected["raw_sha256"]
            or receipt["file_authorization"] != file
            or inspected["decision"] not in {"ADMIT_TO_FILTERING", "REJECT_FILE_MODE"}
        ):
            raise ValueError("File review or raw identity changed")
        if inspected["decision"] == "REJECT_FILE_MODE":
            payload = None
            result = {
                "policy": "pilot-clean-v1.3",
                "decision": "REJECT",
                "reason": "FROZEN_MODE_100644_ONLY",
                "counts": {
                    "raw": len(raw),
                    "license_eligible": 0,
                    "post_format": 0,
                    "post_security": 0,
                    "post_quality": 0,
                },
                "removed_ranges": [],
            }
        else:
            payload, result = early_filter(raw, file, inspected.get("security_review"))
        result.update(
            identity=identity, artifact_receipt=receipt_path.relative_to(PILOT).as_posix()
        )
        record_path = PILOT / "processed/additions-v2-r1" / (digest(identity.encode()) + ".json")
        require_space(len(raw) + 8192)
        if payload is not None:
            target = PILOT / "processed/payloads" / result["cleaned_sha256"]
            if target.exists():
                if target.read_bytes() != payload:
                    raise ValueError("Existing payload differs")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as stream:
                    stream.write(payload)
            result["payload_path"] = target.relative_to(PILOT).as_posix()
            documents.append(
                {
                    "identity": identity,
                    "source": file["source"],
                    "domain": file["domain"],
                    "split": file["reserved_split"],
                    "source_unit": file["source"] + ":" + file["source_unit"],
                    "pin": file["commit"],
                    "payload": payload,
                    "payload_path": result["payload_path"],
                    "cleaned_sha256": result["cleaned_sha256"],
                    "raw_sha256": receipt["raw_sha256"],
                    "filter_record": record_path.relative_to(PILOT).as_posix(),
                    "artifact_receipt": result["artifact_receipt"],
                    "cleaning_policy": result["policy"],
                    "file_review": "docs/targeted_file_review_v2_r1.json",
                }
            )
        write_new(record_path, result)
        filtered.append(result)
    roles = {}
    for doc in documents:
        if doc["source_unit"] in roles and roles[doc["source_unit"]] != doc["split"]:
            raise ValueError("Source unit spans splits")
        roles[doc["source_unit"]] = doc["split"]
    print("Filtering complete; global exact and near deduplication", flush=True)
    duplicates = deduplicate(documents)
    write_new(PILOT / "processed/expanded-v2-r1/dedup.json", duplicates)
    kept = [d for d in documents if duplicates["decisions"][d["identity"]]["keep"]]
    index, registry, fingerprints = load_v2_index()
    match_records = []
    for doc in kept:
        evidence = old["document_evidence"].get(doc["identity"])
        if evidence:
            path = ROOT / evidence["path"]
            if digest(path.read_bytes()) != evidence["sha256"]:
                raise ValueError("Original v2 evidence changed")
            matches = read(path)["matches"]
        else:
            matches = index.match_all(doc["payload"].decode("utf-8"))
            print("V2 scanned addition", doc["identity"], len(matches), flush=True)
        match_records.append(
            {"identity": doc["identity"], "source_unit": doc["source_unit"], "matches": matches}
        )
    excluded = {
        r["source_unit"] for r in match_records if any(m["actionable"] for m in r["matches"])
    }
    survivors = [d for d in kept if d["source_unit"] not in excluded]
    report = {
        "schema_version": 2,
        "candidate_plan_sha256": digest(
            (ROOT / "docs/targeted_candidates_v2_r1.json").read_bytes()
        ),
        "file_review_sha256": digest((ROOT / "docs/targeted_file_review_v2_r1.json").read_bytes()),
        "implementation_sha256": digest((ROOT / "scripts/corpus_expand_v2.py").read_bytes()),
        "original_v2_sha256": digest(
            (ROOT / "reports/contamination_v2_existing_pool.json").read_bytes()
        ),
        "filtering": filtered,
        "before_dedup": accounting(documents),
        "post_dedup": accounting(kept),
        "post_contamination": accounting(survivors),
        "cells": capacity(survivors),
        "excluded_source_units": sorted(excluded),
        "registry": registry,
        "fingerprints": fingerprints,
        "matches": match_records,
        "survivors": [{k: v for k, v in d.items() if k != "payload"} for d in survivors],
        "new_filter_decisions": dict(Counter(r["decision"] for r in filtered)),
    }
    destination = ROOT / "reports/corpus_expansion_v2_r1.json"
    write_new(destination, report)
    write_new(destination.with_suffix(".sha256.json"), {"sha256": digest(destination.read_bytes())})
    failures = [c for c in report["cells"] if c["deficit"]]
    print({"passing_cells": 60 - len(failures), "failures": failures}, flush=True)


if __name__ == "__main__":
    main()
