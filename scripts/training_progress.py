"""Durable progress publication and strict interrupted-trajectory journal checks."""

import json
import math
import os
import tempfile
import time
from pathlib import Path

SEMANTIC_FIELDS = (
    "step",
    "micro_step",
    "loss",
    "raw_byte_nll",
    "raw_byte_perplexity",
    "valid_target_count",
    "learning_rate",
    "gradient_norm",
    "clip_threshold",
    "clipped",
    "examples_seen",
    "bytes_seen",
)


def publish_progress(path, value, *, replace=os.replace, sleep=time.sleep, attempts=6):
    """Only a complete fsynced JSON becomes authoritative; bounded sharing-error retry."""
    if attempts < 1:
        raise ValueError("publication requires a positive bounded attempt count")
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=".progress-", suffix=".tmp", dir=path.parent)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, allow_nan=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    for attempt in range(attempts):
        try:
            replace(temporary, path)
            return
        except OSError as error:
            if not isinstance(error, PermissionError) and getattr(error, "winerror", None) not in (
                5,
                32,
                33,
            ):
                raise
            if attempt + 1 == attempts:
                raise
            sleep(0.05 * 2**attempt)
    # On failure a non-authoritative temporary is retained for inspection.


def semantic_row(row):
    result = {field: row[field] for field in SEMANTIC_FIELDS}
    for key, value in result.items():
        if key == "clipped":
            if type(value) is not bool:
                raise ValueError("invalid clipping flag")
        elif type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid finite semantic metric: {key}")
    return result


def journal(paths):
    rows = []
    bytes_seen = 0
    for path in paths:
        text = Path(path).read_text(encoding="utf-8")
        if text and not text.endswith("\n"):
            raise ValueError("incomplete observation journal tail")
        for line in text.splitlines():
            row = json.loads(line)
            checked = semantic_row(row)
            expected = len(rows) + 1
            if checked["step"] != expected:
                raise ValueError("duplicate, reordered or missing logical update")
            if checked["micro_step"] != 2 * expected or checked["examples_seen"] != 4 * expected:
                raise ValueError("observation counters disagree with frozen accumulation")
            count = checked["valid_target_count"]
            if type(count) is not int or not 4 <= count <= 128:
                raise ValueError("invalid raw target count")
            bytes_seen += count
            if checked["bytes_seen"] != bytes_seen:
                raise ValueError("observation byte counters are not contiguous")
            rows.append(row)
    return rows


def recovery_range(rows, checkpoint_step, progress_step):
    last = len(rows)
    if not 0 <= checkpoint_step <= progress_step <= last:
        raise ValueError("checkpoint/progress/journal ordering mismatch")
    return rows[checkpoint_step:]


def compare_replay(actual, expected):
    if semantic_row(actual) != semantic_row(expected):
        raise ValueError(f"deterministic replay mismatch at update {expected['step']}")
