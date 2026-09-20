"""Apply exact, hash-bound original-text reviews to acquired Gutenberg editions."""

import argparse
import re

from corpus_acquisition import PILOT, digest, read, require_space, write_new
from corpus_filtering import EMAIL, SECRET


def reviewed_payload(raw, review):
    if digest(raw) != review["raw_sha256"]:
        raise ValueError("Book review does not match acquired edition")
    previous = 0
    kept = []
    for start, stop, reason in review["removed_ranges"]:
        if not (previous <= start < stop <= len(raw)) or not reason:
            raise ValueError("Invalid or overlapping book removal range")
        kept.append(raw[previous:start])
        previous = stop
    kept.append(raw[previous:])
    payload = b"".join(kept)
    payload.decode("utf-8")  # Validate, without normalizing or re-encoding.
    if b"\0" in payload or len(payload) < 160:
        raise ValueError("Invalid/insubstantial original book payload")
    if SECRET.search(payload) or re.search(
        rb"[A-Za-z][A-Za-z0-9+.-]*://[^\s/@:]+:[^\s/@]+@", payload
    ):
        raise ValueError("Book security pattern requires review")
    if EMAIL.search(payload):
        raise ValueError("Book contact-like content requires review")
    if any(len(line) > 4000 for line in payload.splitlines()):
        raise ValueError("Book exceeds frozen line-length quality bound")
    return payload


def clean(ebook_id):
    receipt_path = PILOT / "provenance/gutenberg" / f"pg{ebook_id}.json"
    receipt = read(receipt_path)
    review_path = PILOT / "provenance/gutenberg_cleaning_reviews" / f"pg{ebook_id}.json"
    review = read(review_path)
    raw = (PILOT / receipt["raw_path"]).read_bytes()
    if digest(raw) != receipt["raw_sha256"]:
        raise ValueError("Acquired book mutation")
    payload = reviewed_payload(raw, review)
    target = PILOT / "processed/payloads" / digest(payload)
    result = {
        "schema_version": 1,
        "policy": "gutenberg-original-v1",
        "ebook_id": ebook_id,
        "source": "gutenberg",
        "domain": "general_text",
        "reserved_split": receipt["authorization"]["reserved_split"],
        "source_unit": f"gutenberg/work/{ebook_id}",
        "raw_sha256": digest(raw),
        "whole_text_hash_status": "VERIFIED",
        "cleaned_sha256": digest(payload),
        "cleaned_hash_status": "VERIFIED",
        "raw_bytes": len(raw),
        "cleaned_bytes": len(payload),
        "removed_ranges": review["removed_ranges"],
        "review_sha256": digest(review_path.read_bytes()),
        "artifact_receipt": receipt_path.relative_to(PILOT).as_posix(),
        "payload_path": target.relative_to(PILOT).as_posix(),
        "decision": "ELIGIBLE_FOR_DEDUP",
    }
    require_space(len(payload) + 65536)
    if not target.exists():
        with target.open("xb") as stream:
            stream.write(payload)
    elif target.read_bytes() != payload:
        raise ValueError("Existing cleaned payload differs")
    write_new(PILOT / "processed/gutenberg" / f"pg{ebook_id}.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ebook_id", type=int)
    args = parser.parse_args()
    result = clean(args.ebook_id)
    print({k: result[k] for k in ("ebook_id", "raw_bytes", "cleaned_bytes", "cleaned_sha256")})
