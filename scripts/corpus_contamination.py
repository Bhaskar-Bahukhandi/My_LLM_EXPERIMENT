"""Verified benchmark-content matching; detection never rewrites corpus bytes."""

import hashlib
import re
from collections import defaultdict
from functools import lru_cache
from itertools import chain

from benchmark_fingerprints import fingerprint, normalize
from corpus_duplicates import shingles, signature


class ExclusionIndex:
    """Index substantive benchmark fields, retaining identifiers rather than log text."""

    def __init__(self, fields):
        self.texts = {}
        self.raw_variants = defaultdict(set)
        self.identities = defaultdict(list)
        self.anchors = defaultdict(list)
        self.short = set()
        self.near = defaultdict(list)
        self.sizes = {}
        for field in fields:
            text = normalize(field["text"])
            if not text:
                continue
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            self.identities[digest].append(field["identity"])
            self.raw_variants[digest].add(field["text"])
            if digest in self.texts:
                continue
            fp = field.get("fingerprint") or fingerprint(text)
            if fp["normalized_sha256"] != digest:
                raise ValueError("Benchmark field fingerprint/content mismatch")
            self.texts[digest] = text
            if len(text) >= 32:
                self.anchors[text[:32]].append(digest)
            else:
                self.short.add(digest)
            self.sizes[digest] = fp["distinct_shingles"]
            for value in fp["bottom64"]:
                self.near[value].append(digest)

    @lru_cache(maxsize=64)
    def _parts(self, digest):
        return shingles(self.texts[digest])

    def match(self, text):
        """Return the first verified match; never log matched benchmark/corpus values."""
        normalized = normalize(text)

        def result(digest, method):
            return {
                "normalized_sha256": digest,
                "raw_exact": any(value in text for value in self.raw_variants[digest]),
                "method": method,
                "benchmark_identities": sorted(self.identities[digest]),
            }

        for line in [normalized] + [normalize(line) for line in text.splitlines()]:
            digest = hashlib.sha256(line.encode("utf-8")).hexdigest()
            if digest in self.short:
                return result(digest, "SHORT_EXACT_DOCUMENT_OR_LINE")
        for offset in range(max(0, len(normalized) - 31)):
            for digest in self.anchors.get(normalized[offset : offset + 32], ()):
                if normalized.startswith(self.texts[digest], offset):
                    return result(digest, "NORMALIZED_FULL_FIELD_SUBSTRING")
        paragraphs = re.split(r"\n\s*\n", text)
        tokens = re.findall(r"\w+|[^\w\s]", normalized)
        # Overlapping bounded windows supplement paragraph matching for wrapped text.
        windows = (
            " ".join(tokens[start : start + size])
            for size in (128, 256, 512)
            for start in range(0, len(tokens), size // 2)
        )
        for fragment in chain(paragraphs, windows):
            current = shingles(fragment)
            if not current:
                continue
            candidates = {d for value in signature(current) for d in self.near.get(value, ())}
            for digest in sorted(candidates):
                count = self.sizes[digest]
                if min(len(current), count) * 100 < 85 * max(len(current), count):
                    continue
                prior = self._parts(digest)
                overlap = len(current & prior)
                total = len(current) + len(prior) - overlap
                if total and overlap * 100 >= 85 * total:
                    return result(digest, "VERIFIED_PARAGRAPH_OR_WINDOW_JACCARD_0.85")
        return None
