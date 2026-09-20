"""Audit the frozen first-hit ledger; no acquisition or full benchmark rescan."""

import gzip
import json
import re
import sys
from collections import Counter, defaultdict

from benchmark_fingerprints import canonical, load_rows, normalize, sha, strings
from corpus_acquisition import PILOT, ROOT, digest, read, write_new
from corpus_benchmark_index import METADATA_FIELDS
from corpus_global import candidates, capacity, load_stage, stage_path
from corpus_matcher_v2_proposal import isolated_import

REVIEW = PILOT / "provenance/contamination-review-v1"


def preserve():
    paths = [
        "reports/real_corpus_2m_build.md",
        "reports/real_corpus_2m_evidence.json",
        "reports/benchmark_exclusion_coverage.json",
        "docs/data_exclusion_registry.json",
        "data/pilot-2m-r1/provenance/final_filter_snapshot.json",
        *[stage_path(n).relative_to(ROOT).as_posix() for n in ("dedup", "contamination")],
    ]
    hashes = {}
    for relative in paths:
        payload = (ROOT / relative).read_bytes()
        target = REVIEW / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() != payload:
                raise ValueError("Preserved blocked state differs from current input")
        else:
            with target.open("xb") as stream:
                stream.write(payload)
        hashes[relative] = digest(payload)
    return hashes


def matched_fields(targets):
    coverage = read(ROOT / "reports/benchmark_exclusion_coverage.json")
    sidecar = ROOT / coverage["fingerprints"]["path"]
    if digest(sidecar.read_bytes()) != coverage["fingerprints"]["sha256"]:
        raise ValueError("Fingerprint collection changed")
    sys.path.insert(0, str(ROOT / "evidence/download_authorization/reader_deps"))
    found = defaultdict(list)
    counts = Counter()
    with gzip.open(sidecar, "rt", encoding="utf-8") as stream:
        for file in coverage["files"]:
            rows = load_rows(dict(file, local_path=str(ROOT / file["local_path"])))
            for row_index, row in enumerate(rows):
                record = json.loads(next(stream))
                if (
                    record["record_sha256"] != sha(canonical(row))
                    or record["file_sha256"] != file["raw_sha256"]
                    or record["row_index_zero_based"] != row_index
                ):
                    raise ValueError("Benchmark record identity mismatch")
                counts["records"] += 1
                for path, text in strings(row):
                    normalized = normalize(text)
                    checksum = sha(normalized.encode())
                    if checksum != record["fields"][path]["normalized_sha256"]:
                        raise ValueError("Content field fingerprint mismatch")
                    root = path.split("/")[1]
                    metadata = root in METADATA_FIELDS
                    counts["metadata" if metadata else "content"] += 1
                    if checksum not in targets:
                        continue
                    found[checksum].append(
                        {
                            "text": text,
                            "benchmark": record["dataset"],
                            "repository": file["repository"],
                            "revision": file["revision"],
                            "variant": record["variant"],
                            "split": record["split"],
                            "example_id": record["id"],
                            "field_path": path,
                            "identity": f"{record['dataset']}/{record['variant']}/"
                            f"{record['split']}/{record['id']}{path}",
                            "file_sha256": file["raw_sha256"],
                            "record_sha256": record["record_sha256"],
                            "row_index": row_index,
                            "metadata": metadata,
                            "complete_answer_field": root
                            in {"canonical_solution", "completion", "answer", "solution"}
                            or path.endswith("/solution"),
                            "generic_import_support": root == "test_imports"
                            and isolated_import(text),
                        }
                    )
        if next(stream, None) is not None:
            raise ValueError("Unconsumed fingerprint records")
    return found, dict(counts), coverage["fingerprints"]


def project(documents, hits, retain):
    active = [h for h in hits if retain(h)]
    excluded = {h["source_unit"] for h in active}
    kept = [d for d in documents if d["source_unit"] not in excluded]
    cells = capacity(kept)
    return {
        "actionable_recorded_hits": len(active),
        "excluded_source_units": len(excluded),
        "excluded_bytes": sum(len(d["payload"]) for d in documents if d["source_unit"] in excluded),
        "passing_cells": sum(c["deficit"] == 0 for c in cells),
        "cells": cells,
        "deficits": [c for c in cells if c["deficit"]],
    }


def main():
    preserved = preserve()
    frozen = load_stage("contamination")
    targets = {h["normalized_sha256"] for h in frozen["hits"]}
    fields, verified, fingerprint = matched_fields(targets)
    print(
        "Verified pinned benchmark records; counting only four recorded match strings", flush=True
    )
    dedup = load_stage("dedup")
    documents = [d for d in candidates() if dedup["decisions"][d["identity"]]["keep"]]
    by_id = {d["identity"]: d for d in documents}
    frequencies = {checksum: Counter() for checksum in targets}
    unit_frequency = {checksum: set() for checksum in targets}
    raw_variants = {
        checksum: {f["text"] for f in values if not f["metadata"]}
        for checksum, values in fields.items()
    }
    normalized_targets = {
        checksum: normalize(next(iter(values))) for checksum, values in raw_variants.items()
    }
    for doc in documents:
        text = doc["payload"].decode("utf-8")
        lines = Counter(normalize(line) for line in text.splitlines())
        normalized = normalize(text)
        for checksum, target in normalized_targets.items():
            count = lines[target]
            counter = frequencies[checksum]
            counter["normalized_line_occurrences"] += count
            counter["documents_with_v1_short_match"] += bool(count or normalized == target)
            counter["raw_substring_occurrences"] += sum(
                text.count(v) for v in raw_variants[checksum]
            )
            counter["normalized_substring_occurrences"] += normalized.count(target)
            if count or normalized == target:
                unit_frequency[checksum].add(doc["source_unit"])
    hit_unit_counts = Counter(h["source_unit"] for h in frozen["hits"])
    rows = []
    for number, hit in enumerate(sorted(frozen["hits"], key=lambda h: h["identity"]), 1):
        checksum = hit["normalized_sha256"]
        values = [f for f in fields[checksum] if not f["metadata"]]
        if sorted(f["identity"] for f in values) != hit["benchmark_identities"]:
            raise ValueError("Recorded hit associations differ from frozen fields")
        doc = by_id[hit["identity"]]
        generic_support = all(f["generic_import_support"] for f in values)
        answer = any(f["complete_answer_field"] for f in values)
        category = "B" if generic_support and not answer else "C"
        raw = doc["payload"].decode("utf-8")
        variants = sorted(v for v in raw_variants[checksum] if v in raw)
        rows.append(
            {
                "audit_id": number,
                "candidate": hit["identity"],
                "source": doc["source"],
                "source_unit": doc["source_unit"],
                "split": doc["split"],
                "domain": doc["domain"],
                "match_type": hit["method"],
                "normalized_sha256": checksum,
                "raw_matches": [
                    {"sha256": sha(v.encode()), "bytes": len(v.encode())} for v in variants
                ],
                "normalized_bytes": len(normalized_targets[checksum].encode()),
                "token_count": len(re.findall(r"\w+|[^\w\s]", normalized_targets[checksum])),
                "entire_example": False,
                "entire_answer_field": answer,
                "subfield_of_record": True,
                "benchmark_occurrences_including_variants": len(values),
                "benchmark_unique_records": len({f["record_sha256"] for f in values}),
                "candidate_occurrences": dict(frequencies[checksum]),
                "candidate_source_units_with_match": len(unit_frequency[checksum]),
                "generic_support_only": generic_support,
                "generic_syntax_or_scalar": True,
                "classification": category,
                "rationale": "Single parsed import in support-only fields; no answer association"
                if category == "B"
                else "Complete protected answer field, but elementary syntax/scalar shared outside "
                "the benchmark. Full-field equality alone cannot establish copying; "
                "retain exclusion.",
                "unit_excluded_by_any_one_hit": True,
                "sole_recorded_hit_in_unit": hit_unit_counts[hit["source_unit"]] == 1,
                "hits_in_unit": hit_unit_counts[hit["source_unit"]],
                "benchmark_fields": [{k: v for k, v in f.items() if k != "text"} for f in values],
            }
        )
    proposed = project(documents, rows, lambda h: h["classification"] != "B")
    baseline = project(documents, rows, lambda h: True)
    if baseline["cells"] != frozen["cells"]:
        raise ValueError("Frozen capacity replay differs")
    sensitivity = {
        "frozen": baseline,
        "support_import_exception_only": proposed,
        "full_answer_protection_plus_import_exception": proposed,
    }
    for length in (16, 32, 64, 128):
        sensitivity[f"length_only_{length}_UNSAFE"] = project(
            documents, rows, lambda h, n=length: h["normalized_bytes"] >= n
        )
    sensitivity["frequency_at_least_10_documents_UNSAFE"] = project(
        documents,
        rows,
        lambda h: h["candidate_occurrences"]["documents_with_v1_short_match"] < 10,
    )
    spans_by_document = defaultdict(set)
    for row in rows:
        spans_by_document[row["candidate"]].add(row["normalized_sha256"])
    sensitivity["require_two_distinct_recorded_spans_UNSAFE_INCOMPLETE_LEDGER"] = project(
        documents, rows, lambda h: len(spans_by_document[h["candidate"]]) >= 2
    )
    before = {(c["source"], c["domain"], c["split"]): c["available_bytes"] for c in dedup["cells"]}
    failures = [
        dict(
            c,
            pre_contamination_bytes=before[c["source"], c["domain"], c["split"]],
            excluded_bytes=before[c["source"], c["domain"], c["split"]] - c["available_bytes"],
        )
        for c in baseline["deficits"]
    ]
    result = {
        "schema_version": 1,
        "status": "HUMAN REVIEW REQUIRED",
        "preserved": preserved,
        "fingerprint_collection": fingerprint,
        "verified_benchmark_fields": verified,
        "hits": rows,
        "classification_counts": dict(Counter(r["classification"] for r in rows)),
        "unique_content_hashes": len(targets),
        "failing_cells": failures,
        "sensitivity": sensitivity,
        "projection_limit": "Ledger-only upper-bound capacity. Original matcher returned first "
        "hit per document. Suppressing that hit does not establish absence of another protected "
        "match. No excluded unit is restored, and no full benchmark rescan was run.",
        "two_span_limit": "Each recorded document has exactly one first hit. No document can "
        "prove two independent spans in this ledger. The zero-actionable projection is an unsafe "
        "information-loss demonstration, not evidence of no multi-span leakage.",
        "proposal": "Review-only v2 drops a field only when its structured root is test_imports "
        "and Python AST contains exactly one Import/ImportFrom statement. It retains complete "
        "answers even with identical bytes, other content and existing verified near matches. "
        "No raw length/frequency exception. Metadata roots unchanged. No corpus admission hook.",
        "true_contamination": "No recorded match establishes distinctive copying. Complete "
        "answer-field matches remain protected and classified C, not exonerated.",
        "implementation_sha256": {
            p: digest((ROOT / "scripts" / p).read_bytes())
            for p in ("corpus_contamination_review.py", "corpus_matcher_v2_proposal.py")
        },
    }
    write_new(ROOT / "reports/contamination_review_v1.json", result)
    print(
        json.dumps(
            {
                "classifications": result["classification_counts"],
                "frozen": baseline["passing_cells"],
                "proposed_upper_bound": proposed["passing_cells"],
                "remaining": proposed["deficits"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
