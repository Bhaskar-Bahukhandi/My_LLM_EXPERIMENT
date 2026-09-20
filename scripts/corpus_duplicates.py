"""Deterministic exact/near document deduplication for the frozen pilot policy."""

import hashlib
import re
from collections import defaultdict

from benchmark_fingerprints import canonical, normalize

VERSION = "pilot-near-v1"


def shingles(text):
    tokens = re.findall(r"\w+|[^\w\s]", normalize(text))
    return {tuple(tokens[i : i + 5]) for i in range(max(0, len(tokens) - 4))}


def signature(parts):
    return sorted(
        {
            hashlib.blake2b(canonical(part), digest_size=8, person=b"edge-excl-v1").hexdigest()
            for part in parts
        }
    )[:64]


def deduplicate(documents):
    """Return decisions by identity; inputs contain identity, split and raw payload bytes.

    Retrieval is approximate bottom64, verification is exact shingle Jaccard.
    Cross-split connected duplicate clusters lose every member. Within a split,
    lexical identity chooses the survivor without considering its source quota.
    """
    ordered = sorted(documents, key=lambda d: d["identity"])
    if len({d["identity"] for d in ordered}) != len(ordered):
        raise ValueError("Repeated document identity")
    parents = list(range(len(ordered)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left, right):
        parents[find(right)] = find(left)

    exact = {}
    normalized_groups = defaultdict(list)
    inverted = defaultdict(list)
    parts = []
    edges = []
    for index, doc in enumerate(ordered):
        text = doc["payload"].decode("utf-8")
        normalized = normalize(text)
        normalized_groups[hashlib.sha256(normalized.encode("utf-8")).hexdigest()].append(index)
        for kind, key in (("raw", doc["payload"]), ("normalized", normalized.encode("utf-8"))):
            digest = kind, hashlib.sha256(key).hexdigest()
            if digest in exact:
                previous = exact[digest]
                union(previous, index)
                edges.append(
                    {
                        "left": ordered[previous]["identity"],
                        "right": doc["identity"],
                        "method": kind,
                        "jaccard": 1.0,
                    }
                )
            else:
                exact[digest] = index
        current = shingles(text)
        parts.append(current)
        sketch = signature(current)
        candidates = {other for value in sketch for other in inverted[value]}
        for other in sorted(candidates):
            if find(other) == find(index):
                continue
            prior = parts[other]
            # Cardinality bounds avoid impossible comparisons without losing matches.
            if min(len(current), len(prior)) * 100 < 85 * max(len(current), len(prior)):
                continue
            overlap = len(current & prior)
            total = len(current) + len(prior) - overlap
            if total and overlap * 100 >= 85 * total:
                union(other, index)
                edges.append(
                    {
                        "left": ordered[other]["identity"],
                        "right": doc["identity"],
                        "method": VERSION,
                        "jaccard": overlap / total,
                    }
                )
        for value in sketch:
            inverted[value].append(index)
    groups = defaultdict(list)
    for index in range(len(ordered)):
        groups[find(index)].append(index)

    def group_decisions(selected_groups):
        decisions = {}
        for group in selected_groups:
            cross_split = len({ordered[i]["split"] for i in group}) > 1
            survivor = None if cross_split else ordered[group[0]]["identity"]
            for index in group:
                identity = ordered[index]["identity"]
                decisions[identity] = {
                    "keep": identity == survivor,
                    "reason": "CROSS_SPLIT_DUPLICATE"
                    if cross_split
                    else (
                        "UNIQUE_OR_CANONICAL" if identity == survivor else "SAME_SPLIT_DUPLICATE"
                    ),
                    "canonical_identity": survivor,
                }
        return decisions

    return {
        "version": VERSION,
        "decisions": group_decisions(groups.values()),
        "exact_decisions": group_decisions(normalized_groups.values()),
        "verified_edges": edges,
        "limitation": "Bottom64 retrieval does not certify semantic/paraphrase completeness.",
    }
