"""Actual bounded continuation test/static receipts; no production model execution."""

import argparse
import datetime
import json
import subprocess
import time

from context_distribution_study import ROOT, read, sha
from context_study_metrics import require
from training_progress import publish_progress

from unified_edge.training.config import canonical_hash

TESTS = (
    "tests/test_production_continuation.py",
    "tests/test_training.py",
    "tests/test_training_progress.py",
    "tests/test_full_training_stage_a.py",
    "tests/test_dense_model.py",
    "tests/test_context_distribution_study.py",
    "tests/test_context_study_phases.py",
    "tests/test_context_training_ab.py",
    "tests/test_context_ab_report.py",
)


def files():
    from full_training_stage_b_to_endpoint import TOOLS

    candidates = set(TOOLS) | set(TESTS)
    candidates.update(
        p.relative_to(ROOT).as_posix()
        for p in (ROOT / "scripts").glob("finish_full_training_2m.py")
    )
    return sorted(candidates)


def verify_receipt(path):
    record = read(path)
    value = record["value"]
    require(record["sha256"] == canonical_hash(value), "validation receipt corrupt")
    require(value["files"] == {p: sha(ROOT / p) for p in files()}, "stale validation code")
    required = {
        "pytest",
        "ruff",
        "format",
        "compileall",
        "cpu_dependencies",
        "cuda_dependencies",
        "diff",
    }
    require({v["name"] for v in value["commands"]} == required, "missing validation command")
    require(all(v["returncode"] == 0 for v in value["commands"]), "validation failed")
    return value


def run(phase):
    paths = files()
    before = {p: sha(ROOT / p) for p in paths}
    python = ".venv/Scripts/python.exe"
    commands = (
        ("pytest", [python, "-m", "pytest", *TESTS, "-q", "-p", "no:cacheprovider"]),
        ("ruff", [python, "-m", "ruff", "check", *paths]),
        ("format", [python, "-m", "ruff", "format", "--check", *paths]),
        ("compileall", [python, "-m", "compileall", "-q", *paths]),
        ("cpu_dependencies", [python, "-m", "pip", "check"]),
        ("cuda_dependencies", [".venv-cuda/Scripts/python.exe", "-m", "pip", "check"]),
        ("diff", ["git", "-c", "safe.directory=E:/Theai", "diff", "--check"]),
    )
    records = []
    for name, args in commands:
        start = time.perf_counter()
        stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        result = subprocess.run(
            args,
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
        )
        records.append(
            dict(
                name=name,
                args=args,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                seconds=time.perf_counter() - start,
                started_utc=stamp,
            )
        )
        print(f"CHECK {name}: {result.returncode}", flush=True)
    require(before == {p: sha(ROOT / p) for p in paths}, "files changed during validation")
    value = {
        "schema": "production-continuation-validation-1",
        "phase": phase,
        "files": before,
        "commands": records,
    }
    record = {"value": value, "sha256": canonical_hash(value)}
    root = ROOT / "data/full-training-2m-continuation-validation-v1"
    root.mkdir(exist_ok=True)
    (root / f"{phase}_{time.time_ns()}.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    publish_progress(root / f"{phase}.json", record)
    require(all(r["returncode"] == 0 for r in records), "validation failed; evidence retained")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "closeout"), required=True)
    run(parser.parse_args().phase)
