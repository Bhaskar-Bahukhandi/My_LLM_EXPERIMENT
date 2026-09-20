"""Deterministic byte-preserving early filters for the approved pilot files."""

import hashlib
import json
import re
from pathlib import Path

ALLOWED_SPDX = {
    "PSF-2.0",
    "Python-2.0",
    "0BSD",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "MIT",
    "Apache-2.0",
}
EMAIL = re.compile(rb"[A-Za-z0-9_.+%-]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})")
SECRET = re.compile(
    rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"
    rb"|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{30,}"
    rb"|github_pat_[A-Za-z0-9_]{40,}|sk-[A-Za-z0-9]{40,}"
)


def early_filter(raw, authorization, security_review=None):
    """Return retained bytes and auditable stage counts; never run source code."""
    counts = {
        "raw": len(raw),
        "license_eligible": 0,
        "post_format": 0,
        "post_security": 0,
        "post_quality": 0,
    }
    result = {
        "policy": "pilot-clean-v1.3",
        "counts": counts,
        "removed_ranges": [],
        "decision": "REJECT",
        "reason": None,
    }

    def reject(reason):
        result["reason"] = reason
        return None, result

    header = raw[:8192]
    spdx = re.findall(rb"SPDX-License-Identifier:\s*([^\r\n*]+)", header)
    result["file_spdx"] = [s.decode("ascii", "replace").strip() for s in spdx]
    if any(s not in ALLOWED_SPDX for s in result["file_spdx"]):
        return reject("UNAPPROVED_FILE_LICENSE")
    if re.search(rb"(?im)^\s*(?:#|\.\.|/\*|\*)[^\r\n]*(?:copied|adapted) from", header):
        return reject("COPIED_MATERIAL_REQUIRES_SEPARATE_RIGHTS")
    counts["license_eligible"] = len(raw)
    if Path(authorization["path"]).suffix not in {".py", ".c", ".h", ".rst", ".md", ".lean"}:
        return reject("UNAPPROVED_FORMAT")
    if b"\0" in raw:
        return reject("BINARY_NUL")
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError:
        return reject("NON_UTF8_TEXT_NO_TRANSCODING")
    ranges = authorization.get("eligible_byte_ranges", [[0, len(raw)]])
    if ranges != [[0, ranges[0][1]]] or not 0 <= ranges[0][1] <= len(raw):
        raise ValueError("Unsupported or invalid authored eligible-byte range")
    end = ranges[0][1]
    if end < len(raw):
        result["removed_ranges"].append([end, len(raw), "AUTHORED_INELIGIBLE_FOOTER"])
    retained = []
    offset = 0
    for line in raw[:end].splitlines(keepends=True):
        stop = offset + len(line)
        if re.match(
            rb"\s*\.\.\s+(?:literalinclude|include|image|figure|automodule|autoclass)::", line
        ):
            result["removed_ranges"].append([offset, stop, "UNEXPANDED_DIRECTIVE"])
        else:
            retained.append((offset, stop, line))
        offset = stop
    counts["post_format"] = sum(len(line) for _, _, line in retained)
    if SECRET.search(raw[:end]):
        return reject("CREDENTIAL_OR_PRIVATE_KEY_PATTERN")
    safe = []
    reviewed = []
    if security_review is not None:
        if hashlib.sha256(raw).hexdigest() != security_review["raw_sha256"]:
            raise ValueError("Security review does not match raw artifact")
        reviewed = security_review["redact_ranges"]
    in_author_directive = False
    for start, stop, line in retained:
        if [start, stop] in reviewed:
            result["removed_ranges"].append([start, stop, "REVIEWED_CONTACT_REDACTION"])
            continue
        in_author_directive = bool(
            re.match(rb"\s*\.\.\s+(?:moduleauthor|sectionauthor|author)::", line)
            or (in_author_directive and line[:1] in (b" ", b"\t"))
        )
        if re.search(rb"[A-Za-z][A-Za-z0-9+.-]*://[^\s/@:]+:[^\s/@]+@", line):
            return reject("EMBEDDED_CREDENTIAL_REVIEW_REQUIRED")
        emails = EMAIL.findall(line)
        if emails and (
            in_author_directive or re.search(rb"(?i)author|copyright|contributor|maintainer", line)
        ):
            # Preserve credits; only contact spans move to unchanged raw provenance.
            cursor = 0
            for match in EMAIL.finditer(line):
                domain = match.group(1).lower()
                if domain in {b"example.com", b"example.org", b"example.net"} or domain.endswith(
                    b".invalid"
                ):
                    continue
                safe.append(line[cursor : match.start()])
                result["removed_ranges"].append(
                    [
                        start + match.start(),
                        start + match.end(),
                        "ATTRIBUTION_CONTACT_TO_PROVENANCE",
                    ]
                )
                cursor = match.end()
            safe.append(line[cursor:])
            continue
        if any(
            not (
                d.lower() in {b"example.com", b"example.org", b"example.net"}
                or d.lower().endswith(b".invalid")
            )
            for d in emails
        ):
            return reject("NON_EXAMPLE_CONTACT_REVIEW_REQUIRED")
        safe.append(line)
    payload = b"".join(safe)
    counts["post_security"] = len(payload)
    # Recognize file-header declarations, not prose about generated language objects.
    if re.search(
        rb'(?im)^\s*(?:[#/*-]+|\.\.|"{3})\s*'
        rb"(?:auto(?:matically |-|)generated\b|Python Character Mapping Codec"
        rb"[^\r\n]*generated from[^\r\n]*with gencodec\.py\.|(?:this )?file (?:is |was )?"
        rb"(?:automatically )?generated\b|generated (?:by|from)\b|do not edit\b)",
        b"".join(raw.splitlines(keepends=True)[:8]),
    ):
        return reject("GENERATED_FILE_HEADER")
    if len(payload) < 160 or len(re.findall(rb"\w+", payload)) < 10:
        return reject("INSUFFICIENT_SUBSTANTIVE_CONTENT")
    if any(len(line) > 4000 for line in payload.splitlines()):
        return reject("EXCESSIVE_LINE_LENGTH")
    counts["post_quality"] = len(payload)
    result.update(
        decision="ELIGIBLE_FOR_DEDUP",
        reason=None,
        cleaned_sha256=hashlib.sha256(payload).hexdigest(),
        cleaned_bytes=len(payload),
    )
    return payload, result


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    args = parser.parse_args()
    from corpus_acquisition import require_space

    root = Path(__file__).resolve().parents[1] / "data/pilot-2m-r1"
    destination = root / "processed/early-v1_3" / args.source
    destination.mkdir(parents=True, exist_ok=True)
    review_path = root / "provenance" / f"security_review_{args.source}_v1.json"
    reviews = json.loads(review_path.read_text(encoding="utf-8")) if review_path.exists() else {}
    for path in sorted((root / "provenance/artifacts" / args.source).glob("*.json")):
        receipt = json.loads(path.read_text(encoding="utf-8"))
        raw = (root / receipt["raw_path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != receipt["raw_sha256"]:
            raise ValueError(f"Raw mutation: {receipt['path']}")
        payload, result = early_filter(
            raw, receipt["file_authorization"], reviews.get(receipt["raw_sha256"])
        )
        result["artifact_receipt"] = path.relative_to(root).as_posix()
        record = destination / path.name
        if record.exists():
            raise FileExistsError(
                "An early-filter result already exists; do not overwrite evidence"
            )
        require_space(
            (len(payload) if payload is not None else 0)
            + len(json.dumps(result).encode("utf-8"))
            + 4096
        )
        if payload is not None:
            target = root / "processed/payloads" / result["cleaned_sha256"]
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                with target.open("xb") as stream:
                    stream.write(payload)
            result["payload_path"] = target.relative_to(root).as_posix()
        with record.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")


if __name__ == "__main__":
    main()
