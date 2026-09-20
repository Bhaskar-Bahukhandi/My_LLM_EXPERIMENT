"""Acquire one approved Gutenberg catalog file as inert, immutable bytes."""

import argparse
import hashlib
import time
import urllib.request

from corpus_acquisition import AUTH, PILOT, digest, read, require_space, write_new


def download(work):
    file = work["files"][0]
    expected = file["bytes"]
    request = urllib.request.Request(
        file["url"],
        headers={
            "User-Agent": "Unified-Edge-Pilot-Acquisition",
            "Accept-Encoding": "identity",
        },
    )
    # This CLI is called serially: at least two seconds between book requests.
    time.sleep(2)
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.geturl() != file["url"]:
            raise ValueError("Unexpected Gutenberg file redirect")
        data = response.read(expected + 1)
        if len(data) > expected:
            raise ValueError("Gutenberg response exceeded catalog byte bound")
        return data


def acquire(ebook_id):
    allowlist = read(AUTH / "docs/gutenberg_work_allowlist_2m.json")
    work = next((w for w in allowlist["works"] if w["id"] == ebook_id), None)
    if work is None or work["decision"] != "APPROVE_ORIGINAL_TEXT_ONLY":
        raise ValueError("Ebook is not an approved original-text candidate")
    target = PILOT / "raw/gutenberg" / f"pg{ebook_id}.txt"
    receipt_path = PILOT / "provenance/gutenberg" / f"pg{ebook_id}.json"
    if receipt_path.exists():
        receipt = read(receipt_path)
        if receipt["authorization"] != work or digest(target.read_bytes()) != receipt["raw_sha256"]:
            raise ValueError("Immutable Gutenberg artifact or authorization changed")
        return receipt
    require_space(work["files"][0]["bytes"] + 65536)
    raw = download(work)
    receipt = {
        "schema_version": 1,
        "source": "gutenberg",
        "ebook_id": ebook_id,
        "raw_bytes": len(raw),
        "raw_sha256": digest(raw),
        "raw_path": target.relative_to(PILOT).as_posix(),
        "authorization": work,
        "method": "Serial exact catalog-file HTTPS request; no human-page scraping",
        "admission": "RAW_ONLY_PENDING_ORIGINAL_TEXT_CLEANING_REVIEW",
    }
    header = work["header"]
    if (
        len(raw) != work["files"][0]["bytes"]
        or hashlib.sha256(raw[: header["header_bytes"]]).hexdigest() != header["header_sha256"]
    ):
        quarantine = PILOT / "quarantine" / f"gutenberg-{ebook_id}.txt"
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        with quarantine.open("xb") as stream:
            stream.write(raw)
        write_new(quarantine.with_suffix(".json"), receipt)
        raise ValueError("Gutenberg catalog/header identity mismatch; quarantined")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(raw)
    write_new(receipt_path, receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ebook_id", type=int)
    args = parser.parse_args()
    result = acquire(args.ebook_id)
    print({k: result[k] for k in ("ebook_id", "raw_bytes", "raw_sha256", "admission")})
