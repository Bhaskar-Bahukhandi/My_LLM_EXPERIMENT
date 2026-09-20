"""Preflight every targeted file's frozen tree controls before any network call."""

import argparse

from corpus_acquisition import ROOT, acquire_file, digest, read


def validate_tree_entry(file, entry):
    if entry.get("type") != "blob" or entry.get("mode") != "100644":
        raise ValueError("Frozen candidate policy requires a regular mode-100644 blob")
    if (
        entry["path"] != file["path"]
        or entry["sha"] != file["git_blob_sha1"]
        or entry["size"] != file["raw_bytes_from_tree"]
    ):
        raise ValueError("Candidate metadata differs from pinned tree")


def acquire_targeted(files, trees, acquire=acquire_file):
    """Validate the whole batch first, so a late invalid entry causes no downloads."""
    for file in files:
        validate_tree_entry(file, trees[file["source"]][file["path"]])
    return [acquire(file) for file in files]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    args = parser.parse_args()
    files = read(args.manifest)["files"]
    trees = {}
    sources = {
        s["source"]: s for s in read(ROOT / "docs/source_file_allowlists_2m.json")["sources"]
    }
    for file in files:
        source = file["source"]
        if (
            file["commit"] != sources[source]["commit"]
            or file["repository"] != sources[source]["repository"]
        ):
            raise ValueError("Candidate repository or revision differs from approved pin")
        path = ROOT / f"evidence/download_authorization/{source}_tree.json"
        if digest(path.read_bytes()) != sources[source]["tree_sha256"]:
            raise ValueError("Cached pinned tree changed")
        trees[source] = {e["path"]: e for e in read(path)["tree"]}
    for receipt in acquire_targeted(files, trees):
        print(receipt["source"], receipt["path"], receipt["raw_sha256"])


if __name__ == "__main__":
    main()
