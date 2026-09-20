"""Acquire hash-pinned pilot files as inert bytes; never execute source content."""

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "data/pilot-2m-r1"
AUTH = PILOT / "provenance/authorization"
RAW_WRITE_LOCK = Lock()
DATA_CAP = 800_000_000  # Separate 200 MB run reserve keeps the combined cap at 1 GB.


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def disk_bytes():
    roots = [PILOT] + [
        ROOT / "evidence" / name
        for name in ("download_authorization", "authorization_r2", "authorization_r3")
    ]
    total = 0
    pending = [str(root) for root in roots if root.exists()]
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    pending.append(entry.path)
                elif entry.is_file():
                    total += entry.stat().st_size
    return total


def require_space(additional):
    current = disk_bytes()
    if current + additional > DATA_CAP:
        raise RuntimeError(f"Pilot storage cap: {current} + {additional} > {DATA_CAP}")
    return current


def source_files(source):
    base = read(AUTH / "docs/source_file_allowlists_2m.json")
    entry = next(s for s in base["sources"] if s["source"] == source)
    files = [
        dict(f, source=source, commit=entry["commit"], repository=entry["repository"])
        for f in entry["files"]
    ]
    overlay = read(AUTH / "docs/documentation_file_allowlist_2m_r3.json")
    files += [f for f in overlay["files"] if f["source"] == source and f["decision"] == "ADMIT"]
    return sorted(files, key=lambda f: f["path"])


def fetch(url, limit):
    if not url.startswith("https://raw.githubusercontent.com/"):
        raise ValueError("This stage supports only pinned repository file URLs")
    request = urllib.request.Request(url, headers={"User-Agent": "Unified-Edge-Pilot-Acquisition"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                data = response.read(limit + 1)
                if len(data) > limit:
                    raise ValueError("Response exceeded the declared byte bound")
                return data
        except urllib.error.HTTPError as error:
            if error.code not in {429, 500, 502, 503, 504} or attempt == 2:
                raise
            time.sleep(attempt + 1)
    raise AssertionError("unreachable retry state")


def acquire_file(file):
    source, path, pin = file["source"], file["path"], file["commit"]
    identity = hashlib.sha256(path.encode()).hexdigest()
    receipt = PILOT / "provenance/artifacts" / source / (identity + ".json")
    expected_blob = file["git_blob_sha1"]
    raw_path = PILOT / "raw" / source / expected_blob
    if receipt.exists():
        result = read(receipt)
        raw = raw_path.read_bytes()
        if (
            digest(raw) != result["raw_sha256"]
            or len(raw) != result["raw_bytes"]
            or result["commit"] != pin
            or result["file_authorization"] != file
            or hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
            != expected_blob
        ):
            raise ValueError(f"Immutable acquisition artifact mutated: {source}/{path}")
        return result
    require_space(file["raw_bytes_from_tree"] + 65_537 + 16_384)
    url = file["repository"].replace("https://github.com/", "https://raw.githubusercontent.com/")
    url += "/" + pin + "/" + urllib.parse.quote(path)
    cache = file.get("inspection_path")
    if cache and (ROOT / cache).is_file():
        data = (ROOT / cache).read_bytes()
        method, wire_bytes = "verified authorization inspection cache", 0
    else:
        data = fetch(url, file["raw_bytes_from_tree"] + 65_536)
        method, wire_bytes = "pinned HTTPS file", len(data)
    actual_blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    result = {
        "schema_version": 1,
        "source": source,
        "path": path,
        "commit": pin,
        "repository": file["repository"],
        "url": url,
        "method": method,
        "wire_bytes_this_acquisition": wire_bytes,
        "raw_bytes": len(data),
        "raw_sha256": digest(data),
        "git_blob_sha1": actual_blob,
        "domain": file["domain"],
        "reserved_split": file["reserved_split"],
        "source_unit": file["source_unit"],
        "file_authorization": file,
    }
    if actual_blob != expected_blob or len(data) != file["raw_bytes_from_tree"]:
        quarantine = PILOT / "quarantine" / (source + "-" + identity)
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        with quarantine.open("xb") as stream:
            stream.write(data)
        write_new(quarantine.with_suffix(".json"), result)
        raise ValueError(f"Pinned source identity mismatch, quarantined: {source}/{path}")
    with RAW_WRITE_LOCK:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        if raw_path.exists():
            if raw_path.read_bytes() != data:
                raise ValueError("Existing content-addressed raw artifact differs")
        else:
            try:
                with raw_path.open("xb") as stream:
                    stream.write(data)
            except FileExistsError:
                if raw_path.read_bytes() != data:
                    raise ValueError("Concurrent content-addressed artifact differs")
    result["raw_path"] = raw_path.relative_to(PILOT).as_posix()
    write_new(receipt, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=["cpython", "numpy", "sympy", "mathlib"])
    args = parser.parse_args()
    files = source_files(args.source)
    print(f"Acquiring {args.source}: {len(files)} explicitly eligible files", flush=True)
    # Reserve the worst-case bounded response plus receipt for the whole batch plan.
    require_space(sum(f["raw_bytes_from_tree"] + 65_537 + 16_384 for f in files))
    completed = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for start in range(0, len(files), 4):
            # Submit only four at a time: any failure stops further scheduling.
            completed.extend(pool.map(acquire_file, files[start : start + 4]))
            if start // 50 != (start + 4) // 50:
                print(f"{args.source}: {len(completed)}/{len(files)} verified", flush=True)
    result = {
        "source": args.source,
        "files": len(completed),
        "raw_file_bytes": sum(r["raw_bytes"] for r in completed),
        "wire_bytes": sum(r["wire_bytes_this_acquisition"] for r in completed),
        "pilot_disk_bytes": disk_bytes(),
        "status": "ACQUIRED_IDENTITY_VERIFIED",
    }
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
