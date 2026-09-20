"""Audit registered project fixture content without executing any generator or model."""

import json

from benchmark_fingerprints import normalize
from corpus_acquisition import ROOT, digest, read
from corpus_contamination import ExclusionIndex
from corpus_duplicates import shingles
from corpus_global import candidates, capacity, load_stage, save_stage


def main():
    registry_path = ROOT / "docs/data_exclusion_registry.json"
    registry = read(registry_path)
    for entry in registry["project_tests_and_fixture_generators"]:
        if digest((ROOT / entry["path"]).read_bytes()) != entry["sha256"]:
            raise ValueError("Registered project generator identity changed")
    fields = []
    markers = []
    for entry in registry["local_fixture_payloads"]:
        raw = (ROOT / entry["path"]).read_bytes()
        if digest(raw) != entry["sha256"]:
            raise ValueError("Registered project fixture identity changed")
        text = raw.decode("utf-8")
        fields.append({"text": text, "identity": entry["path"]})
        parts = shingles(text)
        common = set.intersection(*(set(part) for part in parts)) if parts else set()
        markers.append(max(common, key=len) if common else None)
    index = ExclusionIndex(fields)
    duplicate = load_stage("dedup")
    documents = [d for d in candidates() if duplicate["decisions"][d["identity"]]["keep"]]
    hits = []
    full_matches = 0
    for doc in documents:
        text = doc["payload"].decode("utf-8")
        normalized = normalize(text)
        # Every target shingle contains its marker: absence proves no overlap,
        # hence no positive Jaccard or complete-field substring match is possible.
        if all(marker is not None and marker not in normalized for marker in markers):
            continue
        full_matches += 1
        match = index.match(text)
        if match:
            hits.append(dict(identity=doc["identity"], source_unit=doc["source_unit"], **match))
    excluded = {h["source_unit"] for h in hits}
    result = {
        "registry_sha256": digest(registry_path.read_bytes()),
        "checker_sha256": digest((ROOT / "scripts/corpus_fixture_audit.py").read_bytes()),
        "registered_generator_files_verified": len(
            registry["project_tests_and_fixture_generators"]
        ),
        "registered_payload_files_verified": len(fields),
        "unique_payloads": len(index.texts),
        "documents_scanned": len(documents),
        "documents_requiring_full_match": full_matches,
        "hits": hits,
        "excluded_source_units": sorted(excluded),
        "project_path_exclusion": (
            "All candidates require frozen external source authorization; "
            "no project source/test/evidence path can be admitted."
        ),
        "cells": capacity([d for d in documents if d["source_unit"] not in excluded]),
    }
    save_stage("project_fixtures", result)
    print(
        json.dumps(
            {k: result[k] for k in ("documents_scanned", "documents_requiring_full_match", "hits")}
        )
    )


if __name__ == "__main__":
    main()
