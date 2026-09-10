"""Build evaluation-only fingerprints from pinned local files; never execute samples."""

import argparse
import gzip
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ALGORITHM = {
    "version": "edge-exclusion-v1",
    "normalization": "Unicode NFKC, casefold, collapse Unicode whitespace, UTF-8",
    "exact": "SHA-256 of normalized content, separately for every string field",
    "near": "64 minimum distinct BLAKE2b-64 hashes of consecutive 5-token shingles",
    "tokens": r"Python Unicode regex: \w+|[^\w\s]",
    "shingle_encoding": "compact JSON array, ensure_ascii=False, UTF-8",
    "personalization": "edge-excl-v1",
    "short_fields": "fewer than 5 tokens: exact hash only",
    "matching": (
        "Candidate retrieval only; verify original content before exclusion. "
        "No semantic/paraphrase completeness claim."
    ),
    "payload_policy": "Normalization affects detection only, never training bytes.",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def sha(data):
    return hashlib.sha256(data).hexdigest()


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def fingerprint(text):
    normalized = normalize(text)
    tokens = re.findall(r"\w+|[^\w\s]", normalized)
    hashes = {
        hashlib.blake2b(
            canonical(tokens[i : i + 5]), digest_size=8, person=b"edge-excl-v1"
        ).hexdigest()
        for i in range(max(0, len(tokens) - 4))
    }
    return {
        "normalized_sha256": sha(normalized.encode("utf-8")),
        "token_count": len(tokens),
        "distinct_shingles": len(hashes),
        "bottom64": sorted(hashes)[:64],
    }


def strings(value, path=""):
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for key in sorted(value):
            yield from strings(value[key], f"{path}/{key}")
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from strings(item, f"{path}/{i}")


def load_rows(file):
    path = Path(file["local_path"])
    data = path.read_bytes()
    if sha(data) != file["raw_sha256"]:
        raise ValueError(f"Raw file hash mismatch: {path}")
    if path.suffix == ".parquet":
        import duckdb

        with duckdb.connect(config={"enable_external_access": "true", "threads": "1"}) as db:
            cursor = db.execute("SELECT * FROM read_parquet(?)", [str(path)])
            columns = [c[0] for c in cursor.description]
            return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]
    if path.suffix == ".gz":
        data = gzip.decompress(data)
    try:
        value = json.loads(data)
    except json.JSONDecodeError:
        return [json.loads(line) for line in data.splitlines() if line.strip()]
    return value if isinstance(value, list) else [value]


def mbpp_split(task_id):
    if 1 <= task_id <= 10:
        return "prompt"
    if 11 <= task_id <= 510:
        return "test"
    if 511 <= task_id <= 600:
        return "validation"
    if 601 <= task_id <= 974:
        return "train"
    raise ValueError(f"Unexpected MBPP task ID: {task_id}")


def build(manifest, output):
    files = json.loads(Path(manifest).read_text(encoding="utf-8"))
    loaded = [(file, load_rows(file)) for file in files]
    math_sets = {}
    for variant in ["publisher_linked_combined", "split_preserving"]:
        math_sets[variant] = Counter(
            sha(canonical(row))
            for f, rows in loaded
            if f["dataset"] == "math" and f["variant"] == variant
            for row in rows
        )
    if math_sets["publisher_linked_combined"] != math_sets["split_preserving"]:
        raise ValueError("MATH publisher/split distribution row multisets differ")
    math_splits = Counter(
        f["split"]
        for f, rows in loaded
        if f["dataset"] == "math" and f["variant"] == "split_preserving"
        for _ in rows
    )
    if math_splits != {"train": 7500, "test": 5000}:
        raise ValueError(f"MATH split sizes changed: {math_splits}")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    summaries = []
    with (
        output.open("xb") as raw_out,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw_out, mtime=0) as out,
    ):
        for file, rows in loaded:
            counts = Counter()
            for index, row in enumerate(rows):
                split = mbpp_split(row["task_id"]) if file["dataset"] == "mbpp" else file["split"]
                if file["variant"] == "publisher_linked_combined":
                    split = "combined_publisher_distribution_NOT_original_train"
                counts[split] += 1
                native_id = row.get("task_id")
                record = {
                    "dataset": file["dataset"],
                    "variant": file["variant"],
                    "split": split,
                    "file_sha256": file["raw_sha256"],
                    "row_index_zero_based": index,
                    "native_id": native_id,
                    "record_sha256": sha(canonical(row)),
                    "normalized_record_sha256": sha(
                        canonical({key: normalize(value) for key, value in strings(row)})
                    ),
                    "id": str(native_id)
                    if native_id is not None
                    else "sha256:" + sha(canonical(row)),
                    "id_kind": "native"
                    if native_id is not None
                    else "content_identity_no_native_id_in_distribution",
                    "fields": {key: fingerprint(value) for key, value in strings(row)},
                }
                out.write(canonical(record) + b"\n")
            summaries.append({**file, "examples": len(rows), "split_counts": dict(counts)})
            print(file["source_path"], len(rows), flush=True)
    summary = {
        "schema_version": "1",
        "algorithm": ALGORITHM,
        "unicode_database": unicodedata.unidata_version,
        "python": sys.version.split()[0],
        "reader": "duckdb==1.4.4 (isolated ignored tool directory)",
        "files": summaries,
        "total_records_including_variants_and_examples": sum(x["examples"] for x in summaries),
        "math_exact_row_multiset_equal": True,
        "math_original_split_counts": dict(math_splits),
        "math_identity_digest": sha(
            canonical(sorted(math_sets["publisher_linked_combined"].items()))
        ),
        "fingerprints": {
            "path": output.as_posix(),
            "sha256": sha(output.read_bytes()),
            "bytes": output.stat().st_size,
        },
    }
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("output")
    parser.add_argument("summary")
    parser.add_argument("--reader-path", type=Path)
    args = parser.parse_args()
    if args.reader_path:
        sys.path.insert(0, str(args.reader_path))
    result = build(args.manifest, args.output)
    Path(args.summary).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
