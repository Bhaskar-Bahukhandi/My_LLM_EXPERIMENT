"""Build a matcher from the existing pinned benchmark bytes and fingerprint sidecar."""

import gzip
import hashlib
import json
import sys

from benchmark_fingerprints import canonical, load_rows, sha, strings
from corpus_contamination import ExclusionIndex

METADATA_FIELDS = {"task_id", "entry_point", "source_file", "level", "type"}


def load_index(root):
    coverage = json.loads(
        (root / "reports/benchmark_exclusion_coverage.json").read_text(encoding="utf-8")
    )
    sidecar = root / coverage["fingerprints"]["path"]
    with sidecar.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != coverage["fingerprints"]["sha256"]:
            raise ValueError("Frozen benchmark fingerprint sidecar changed")
    # Reuse the already pinned, isolated parquet reader; never install or execute samples.
    sys.path.insert(0, str(root / "evidence/download_authorization/reader_deps"))
    stats = {
        "records_verified": 0,
        "content_fields": 0,
        "metadata_fields_omitted": 0,
        "fingerprint_sha256": coverage["fingerprints"]["sha256"],
        "metadata_field_roots": sorted(METADATA_FIELDS),
    }

    def fields():
        with gzip.open(sidecar, "rt", encoding="utf-8") as stream:
            for file in coverage["files"]:
                local = dict(file, local_path=str(root / file["local_path"]))
                rows = load_rows(local)
                if len(rows) != file["examples"]:
                    raise ValueError("Benchmark split/file example count changed")
                for row_index, row in enumerate(rows):
                    record = json.loads(next(stream))
                    if (
                        record["record_sha256"] != sha(canonical(row))
                        or record["file_sha256"] != file["raw_sha256"]
                        or record["row_index_zero_based"] != row_index
                    ):
                        raise ValueError("Benchmark raw record/fingerprint identity mismatch")
                    stats["records_verified"] += 1
                    for path, text in strings(row):
                        if path.split("/")[1] in METADATA_FIELDS:
                            stats["metadata_fields_omitted"] += 1
                            continue
                        stats["content_fields"] += 1
                        yield {
                            "text": text,
                            "fingerprint": record["fields"][path],
                            "identity": f"{record['dataset']}/{record['variant']}/"
                            f"{record['split']}/{record['id']}{path}",
                        }
            if next(stream, None) is not None:
                raise ValueError("Unconsumed benchmark fingerprint records")

    index = ExclusionIndex(fields())
    if stats["records_verified"] != coverage["total_records_including_variants_and_examples"]:
        raise ValueError("Benchmark total record coverage changed")
    return index, stats
