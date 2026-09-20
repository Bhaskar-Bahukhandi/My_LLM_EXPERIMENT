"""Recheck cached executable-bit source as inert data and recompute the entire pool."""

from corpus_acquisition import PILOT, ROOT, digest, read, require_space, write_new
from corpus_contamination import ExclusionIndex
from corpus_contamination_v2 import load_v2_index
from corpus_duplicates import deduplicate
from corpus_global import accounting, candidates, capacity
from corpus_mode_v2 import filter_inert, preflight


def main():
    policy = read(ROOT / "docs/file_mode_policy_v2.json")
    for path, checksum in policy["preserved_sha256"].items():
        if digest((ROOT / path).read_bytes()) != checksum:
            raise ValueError("Historical rejection or blocked evidence changed")
    plan = read(ROOT / "docs/targeted_candidates_v2_r1.json")
    previous = read(ROOT / "reports/corpus_expansion_v2_r1.json")
    reviews = read(ROOT / "docs/mode_v2_file_reviews.json")
    trees = {
        s: {
            e["path"]: e
            for e in read(ROOT / f"evidence/download_authorization/{s}_tree.json")["tree"]
        }
        for s in ("cpython", "sympy")
    }
    preflight(plan["files"], trees)
    documents = candidates()
    all_filters = []
    for file in plan["files"]:
        identity = file["source"] + "/" + file["path"]
        prior = next(r for r in previous["filtering"] if r["identity"] == identity)
        if prior["decision"] == "ELIGIBLE_FOR_DEDUP":
            existing = next(d for d in previous["survivors"] if d["identity"] == identity)
            payload = (PILOT / existing["payload_path"]).read_bytes()
            if digest(payload) != existing["cleaned_sha256"]:
                raise ValueError("Previously filtered payload changed")
            documents.append(dict(existing, payload=payload))
            continue
        inspection = reviews["files"][identity]
        receipt_path = (
            PILOT
            / "provenance/artifacts"
            / file["source"]
            / (digest(file["path"].encode()) + ".json")
        )
        receipt = read(receipt_path)
        raw = (PILOT / receipt["raw_path"]).read_bytes()
        if (
            digest(raw) != inspection["raw_sha256"]
            or receipt["file_authorization"] != file
            or inspection["decision"] != "ADMIT_TO_FILTERING"
        ):
            raise ValueError("Cached source/review identity mismatch")
        payload, result = filter_inert(raw, file, trees[file["source"]][file["path"]])
        result.update(
            identity=identity, artifact_receipt=receipt_path.relative_to(PILOT).as_posix()
        )
        record = PILOT / "processed/mode-v2" / (digest(identity.encode()) + ".json")
        require_space(len(raw) + 8192)
        if payload is not None:
            target = PILOT / "processed/payloads" / result["cleaned_sha256"]
            if target.exists():
                if target.read_bytes() != payload:
                    raise ValueError("Payload address collision")
            else:
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
                    "filter_record": record.relative_to(PILOT).as_posix(),
                    "artifact_receipt": result["artifact_receipt"],
                    "cleaning_policy": result["policy"],
                    "file_review": "docs/mode_v2_file_reviews.json",
                }
            )
        write_new(record, result)
        all_filters.append(result)
    print("Cached source filtering complete; global deduplication", flush=True)
    roles = {}
    for d in documents:
        if d["source_unit"] in roles and roles[d["source_unit"]] != d["split"]:
            raise ValueError("Source unit crosses reserved splits")
        roles[d["source_unit"]] = d["split"]
    duplicates = deduplicate(documents)
    write_new(PILOT / "processed/expanded-mode-v2/dedup.json", duplicates)
    kept = [d for d in documents if duplicates["decisions"][d["identity"]]["keep"]]
    index, registry, fingerprints = load_v2_index()
    fixture_registry = read(ROOT / "docs/data_exclusion_registry.json")
    fixture_fields = []
    for entry in fixture_registry["local_fixture_payloads"]:
        raw = (ROOT / entry["path"]).read_bytes()
        if digest(raw) != entry["sha256"]:
            raise ValueError("Project fixture changed")
        fixture_fields.append({"identity": entry["path"], "text": raw.decode("utf-8")})
    fixture_index = ExclusionIndex(fixture_fields)
    old_matches = {r["identity"]: r for r in previous["matches"]}
    matches = []
    fixture_hits = []
    for doc in kept:
        if doc["identity"] in old_matches:
            record = old_matches[doc["identity"]]
        else:
            text = doc["payload"].decode("utf-8")
            record = {
                "identity": doc["identity"],
                "source_unit": doc["source_unit"],
                "matches": index.match_all(text),
            }
            fixture = fixture_index.match(text)
            if fixture:
                fixture_hits.append(
                    {
                        "identity": doc["identity"],
                        "source_unit": doc["source_unit"],
                        "match": fixture,
                    }
                )
            print("Exhaustively scanned", doc["identity"], len(record["matches"]), flush=True)
        matches.append(record)
    excluded = {r["source_unit"] for r in matches if any(m["actionable"] for m in r["matches"])}
    excluded.update(r["source_unit"] for r in fixture_hits)
    survivors = [d for d in kept if d["source_unit"] not in excluded]
    report = {
        "schema_version": 2,
        "policy_sha256": digest((ROOT / "docs/file_mode_policy_v2.json").read_bytes()),
        "previous_report_sha256": digest(
            (ROOT / "reports/corpus_expansion_v2_r1.json").read_bytes()
        ),
        "review_sha256": digest((ROOT / "docs/mode_v2_file_reviews.json").read_bytes()),
        "filters": all_filters,
        "before_dedup": accounting(documents),
        "post_dedup": accounting(kept),
        "post_contamination": accounting(survivors),
        "cells": capacity(survivors),
        "matches": matches,
        "project_fixture_hits": fixture_hits,
        "excluded_source_units": sorted(excluded),
        "registry": registry,
        "fingerprints": fingerprints,
        "survivors": [{k: v for k, v in d.items() if k != "payload"} for d in survivors],
    }
    path = ROOT / "reports/corpus_mode_v2_pool.json"
    write_new(path, report)
    write_new(path.with_suffix(".sha256.json"), {"sha256": digest(path.read_bytes())})
    failures = [c for c in report["cells"] if c["deficit"]]
    print({"passing_cells": 60 - len(failures), "failures": failures}, flush=True)


if __name__ == "__main__":
    main()
