"""Resume an unchanged measurement producer after its tooling was committed."""

import subprocess

import context_distribution_study as study
from context_study_metrics import require
from training_progress import publish_progress

from unified_edge.training.config import canonical_hash


def bound_source(current, original):
    require(current["source_sha256"] == original["source_sha256"], "model source changed")
    return dict(current, commit=original["commit"])


def run():
    binding = study.read(study.STUDY / "binding.json")
    for name, expected in binding["tools"].items():
        require(study.sha(study.ROOT / "scripts" / name) == expected, "producer tool changed")
    original_identity = study.code_identity
    current = original_identity()
    bound_source(current, binding["source"])
    subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={study.ROOT.as_posix()}",
            "merge-base",
            "--is-ancestor",
            binding["source"]["commit"],
            current["commit"],
        ],
        cwd=study.ROOT,
        check=True,
    )
    directory = study.STUDY / "recovery-v1"
    directory.mkdir(exist_ok=True)
    path = directory / "reconciliation.json"
    if path.exists():
        record = study.read(path)
        require(
            record["binding_sha256"] == study.sha(study.STUDY / "binding.json"),
            "recovery binding changed",
        )
        require(record["resume_tool_sha256"] == study.sha(__file__), "recovery tool changed")
        for name, expected in record["preserved_measurements"].items():
            require(study.sha(study.STUDY / name) == expected, "saved measurement changed")
    else:
        preserved = {}
        for p in study.STUDY.glob("*.json"):
            if p.stem in ("binding", "progress"):
                continue
            row = study.read(p)
            require(row["binding_sha256"] == canonical_hash(binding), "record binding mismatch")
            require(row["sha256"] == canonical_hash(row["value"]), "record integrity mismatch")
            preserved[p.name] = study.sha(p)
        publish_progress(
            path,
            {
                "schema": "study-resume-1",
                "binding_sha256": study.sha(study.STUDY / "binding.json"),
                "original_source": binding["source"],
                "resume_source": current,
                "resume_tool_sha256": study.sha(__file__),
                "preserved_measurements": preserved,
                "reason": "Producer process absent; tooling committed without byte changes. "
                "Retain original launch-commit metadata in existing binding; record new "
                "Git HEAD here. No numerical, data, model or sampling semantics change.",
            },
        )
    # Only provenance metadata is reconciled. The original run() and tool bytes are unchanged.
    study.code_identity = lambda: bound_source(original_identity(), binding["source"])
    try:
        study.run()
    finally:
        study.code_identity = original_identity


if __name__ == "__main__":
    run()
