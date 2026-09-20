"""Approved role-aware exhaustive matching, separate from immutable v1."""

import gzip
import json
import re
import sys
from collections import defaultdict
from itertools import chain

from benchmark_fingerprints import canonical, load_rows, normalize, sha, strings
from corpus_acquisition import ROOT, digest, read
from corpus_benchmark_index import METADATA_FIELDS
from corpus_contamination import ExclusionIndex
from corpus_duplicates import shingles, signature
from corpus_matcher_v2_proposal import isolated_import


def field_class(path):
    root = path.strip("/").split("/")[0]
    if root in METADATA_FIELDS:
        return "metadata"
    if root == "test_imports":
        return "support_import"
    if root in {"code", "canonical_solution", "completion"}:
        return "code_solution"
    if path.endswith("/solution") and root != "solution":
        return "generated_solution"
    if root in {"answer", "solution"}:
        return "answer"
    if root in {"problem", "question", "prompt", "text"}:
        return "problem"
    return "other_protected_content"


class ExhaustiveIndex(ExclusionIndex):
    """One result per distinct content identity, with all benchmark associations.

    Exact evidence dominates redundant near evidence for the same identity.
    No first-match return: unrelated protected identities are always considered.
    """

    def __init__(self, fields):
        self.associations = defaultdict(list)

        def classified():
            for field in fields:
                role = field_class(field["field_path"])
                if role == "metadata":
                    continue
                checksum = sha(normalize(field["text"]).encode())
                ignored = role == "support_import" and isolated_import(field["text"])
                self.associations[checksum].append(
                    {
                        "identity": field["identity"],
                        "field_class": role,
                        "non_actionable_import": ignored,
                    }
                )
                yield field

        super().__init__(classified())

    def match_all(self, text):
        normalized = normalize(text)
        matches = {}

        def record(checksum, method):
            if checksum in matches:
                return
            associations = sorted(self.associations[checksum], key=lambda a: a["identity"])
            matches[checksum] = {
                "normalized_sha256": checksum,
                "method": method,
                "raw_exact": any(v in text for v in self.raw_variants[checksum]),
                "actionable": any(not a["non_actionable_import"] for a in associations),
                "associations": associations,
            }

        for line in chain((normalized,), (normalize(line) for line in text.splitlines())):
            checksum = sha(line.encode())
            if checksum in self.short:
                record(checksum, "SHORT_EXACT_DOCUMENT_OR_LINE")
        for offset in range(max(0, len(normalized) - 31)):
            for checksum in self.anchors.get(normalized[offset : offset + 32], ()):
                if checksum not in matches and normalized.startswith(self.texts[checksum], offset):
                    record(checksum, "NORMALIZED_FULL_FIELD_SUBSTRING")
        tokens = re.findall(r"\w+|[^\w\s]", normalized)
        windows = (
            " ".join(tokens[start : start + size])
            for size in (128, 256, 512)
            for start in range(0, len(tokens), size // 2)
        )
        for fragment in chain(re.split(r"\n\s*\n", text), windows):
            current = shingles(fragment)
            if not current:
                continue
            candidates = {d for v in signature(current) for d in self.near.get(v, ())}
            for checksum in sorted(candidates - matches.keys()):
                count = self.sizes[checksum]
                if min(len(current), count) * 100 < 85 * max(len(current), count):
                    continue
                prior = self._parts(checksum)
                overlap = len(current & prior)
                total = len(current) + len(prior) - overlap
                if total and overlap * 100 >= 85 * total:
                    record(checksum, "VERIFIED_PARAGRAPH_OR_WINDOW_JACCARD_0.85")
        return [matches[key] for key in sorted(matches)]


def load_v2_index():
    coverage = read(ROOT / "reports/benchmark_exclusion_coverage.json")
    sidecar = ROOT / coverage["fingerprints"]["path"]
    if digest(sidecar.read_bytes()) != coverage["fingerprints"]["sha256"]:
        raise ValueError("Frozen fingerprint sidecar changed")
    sys.path.insert(0, str(ROOT / "evidence/download_authorization/reader_deps"))
    stats = defaultdict(int)

    def fields():
        with gzip.open(sidecar, "rt", encoding="utf-8") as stream:
            for file in coverage["files"]:
                rows = load_rows(dict(file, local_path=str(ROOT / file["local_path"])))
                if len(rows) != file["examples"]:
                    raise ValueError("Benchmark file count changed")
                for number, row in enumerate(rows):
                    record = json.loads(next(stream))
                    if (
                        record["record_sha256"] != sha(canonical(row))
                        or record["file_sha256"] != file["raw_sha256"]
                        or record["row_index_zero_based"] != number
                    ):
                        raise ValueError("Benchmark record changed")
                    stats["records"] += 1
                    for path, text in strings(row):
                        if (
                            sha(normalize(text).encode())
                            != record["fields"][path]["normalized_sha256"]
                        ):
                            raise ValueError("Field identity changed")
                        stats[field_class(path)] += 1
                        yield {
                            "text": text,
                            "field_path": path,
                            "fingerprint": record["fields"][path],
                            "identity": f"{record['dataset']}/{record['variant']}/"
                            f"{record['split']}/{record['id']}{path}",
                        }
            if next(stream, None) is not None:
                raise ValueError("Unconsumed benchmark records")

    index = ExhaustiveIndex(fields())
    return index, dict(stats), coverage["fingerprints"]
