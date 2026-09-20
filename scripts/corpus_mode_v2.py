"""Approved regular-source mode handling. All source remains inert byte data."""

from pathlib import PurePosixPath

from corpus_filtering import early_filter


def validate_entry(file, entry):
    mode = entry.get("mode")
    if entry.get("type") != "blob" or mode not in {"100644", "100755"}:
        raise ValueError("Unsupported mode or non-regular Git object")
    if mode == "100755" and (
        file["domain"] != "code"
        or PurePosixPath(file["path"]).suffix not in {".py", ".c", ".h", ".lean"}
    ):
        raise ValueError("Executable-bit exception is limited to approved source-code formats")
    if (
        entry["path"] != file["path"]
        or entry["sha"] != file["git_blob_sha1"]
        or entry["size"] != file["raw_bytes_from_tree"]
    ):
        raise ValueError("Candidate differs from pinned Git tree")


def preflight(files, trees):
    for file in files:
        validate_entry(file, trees[file["source"]][file["path"]])


def filter_inert(raw, file, entry, security_review=None):
    validate_entry(file, entry)
    return early_filter(raw, file, security_review)
