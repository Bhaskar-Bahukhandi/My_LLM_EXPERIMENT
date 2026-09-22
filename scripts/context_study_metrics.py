"""Study-only byte probabilities, fixed validation anchors and paired statistics."""

import hashlib
import math
import random
import statistics
from collections import defaultdict

import torch

LENGTHS = (32, 64, 128, 256)
PROTOCOL = {
    "version": "context-distribution-1",
    "lengths": list(LENGTHS),
    "baseline_nll": 2.3624953916,
    "baseline_absolute_tolerance": 1e-7,
    "batch_size": 2,
    "anchor_seed": 1729,
    "anchors_per_domain": 128,
    "anchor_algorithm": "lowest SHA256(seed|path|offset), equal domain quotas, offset >=256",
    "uncertainty": "2000 paired percentile bootstrap draws; seed 1730; 95% interval",
    "unchanged_nll_tolerance": 1e-5,
    "concentration_thresholds": [0.5, 0.9, 0.99],
    "classification": {
        "scope": "project-specific diagnostics, not universal scientific thresholds",
        "severe": "q maximum >=0.9 and entropy <=1 nat",
        "moderate": "q(space) >=0.5 unless severe",
        "broad": "space argmax, q(space)<0.5 and entropy>1 nat",
        "otherwise": "MIXED_BY_CONTEXT",
    },
    "tie_policy": "descending probability, ascending byte ID",
    "generation_prompts_hex": [
        p.hex() for p in (b"", b"The ", b"def ", b"import ", b"class ", b"x = ")
    ],
    "generation_bytes": 64,
    "segmentation": "manifest order, nonoverlapping document windows, fresh BOS each window",
    "matched_target": "incremental predict after exactly L prior bytes; same anchor target",
    "distribution_sample": "all anchors at all lengths; additional forced-space successor",
    "state_carry": "first 8 hash-selected anchors/domain, compare last32 reset vs prior256 carry",
    "mechanics": "batch2 validation bytes, fresh disposable parent clone per length, no update",
    "test_policy": "hash-only integrity reads, no TEST bytes delivered to evaluator",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_baseline(value, count):
    require(count == 500000, "baseline must cover exactly 500000 validation bytes")
    require(
        math.isfinite(value)
        and abs(value - PROTOCOL["baseline_nll"]) <= PROTOCOL["baseline_absolute_tolerance"],
        "accepted 32-byte baseline did not reproduce; stop longer-context evaluation",
    )


def validation_documents(manifest, payloads, split="validation"):
    require(split == "validation", "study rejects TEST and training evaluation")
    return [(d, payloads[d.path]) for d in manifest.documents if d.split == split]


def select_anchors(documents, per_domain=128, seed=1729):
    candidates = defaultdict(list)
    for doc, payload in documents:
        require(doc.split == "validation", "anchors require validation documents")
        for offset in range(256, len(payload)):
            key = hashlib.sha256(f"{seed}|{doc.path}|{offset}".encode()).hexdigest()
            candidates[doc.domain].append((key, doc.path, offset, payload[offset]))
    result = []
    for domain, values in sorted(candidates.items()):
        for key, path, offset, target in sorted(values)[:per_domain]:
            result.append(dict(domain=domain, path=path, offset=offset, target=target, key=key))
    require(bool(result), "no eligible anchors")
    return result


def anchor_history(anchor, payloads, length):
    require(length in LENGTHS, "unsupported context length")
    payload = payloads[anchor["path"]]
    offset = anchor["offset"]
    require(256 <= offset < len(payload), "anchor crosses document boundary")
    require(payload[offset] == anchor["target"], "anchor target identity changed")
    return payload[offset - length : offset]


def distribution(logits):
    values = logits.detach().cpu().double().flatten()
    require(values.numel() == 267, "distribution requires full 267-symbol support")
    require(bool(torch.isfinite(values[:256]).all()), "nonfinite byte logits")
    require(not bool(torch.isnan(values).any() or torch.isposinf(values).any()), "invalid logits")
    p = values.softmax(-1)
    mass = p[:256].sum()
    require(bool(mass > 0), "conditional byte distribution undefined")
    q = p[:256] / mass
    order = sorted(range(256), key=lambda i: (-q[i].item(), i))
    a, b = order[:2]
    entropy = -(q[q > 0] * q[q > 0].log()).sum().item()
    space = q[32].item()
    maximum = q[a].item()
    if maximum >= 0.9 and entropy <= 1:
        label = "SEVERE_BYTE_MODE_CONCENTRATION"
    elif space >= 0.5:
        label = "MODERATE_SPACE_CONCENTRATION"
    elif a == 32 and entropy > 1:
        label = "ARGMAX_SPACE_DOMINANCE_WITH_BROAD_DISTRIBUTION"
    else:
        label = "MIXED_BY_CONTEXT"
    return {
        "byte_mass_p": mass.item(),
        "control_mass_p": p[256:].sum().item(),
        "control_probabilities_p": p[256:].tolist(),
        "masked_symbol_ids": torch.where(torch.isneginf(values))[0].tolist(),
        "space_p": p[32].item(),
        "space_q": space,
        "space_rank": order.index(32) + 1,
        "space_top1": a == 32,
        "space_top2": b == 32,
        "top1_byte": a,
        "top2_byte": b,
        "top1_q": maximum,
        "top2_q": q[b].item(),
        "top1_p": p[a].item(),
        "margin_q": (q[a] - q[b]).item(),
        "entropy_nats": entropy,
        "entropy_bits": entropy / math.log(2),
        "effective_bytes": math.exp(entropy),
        "classification": label,
    }


def distribution_summary(rows):
    require(bool(rows), "empty distribution sample")
    keys = (
        "byte_mass_p",
        "control_mass_p",
        "space_p",
        "space_q",
        "space_rank",
        "space_top1",
        "space_top2",
        "top1_q",
        "top2_q",
        "margin_q",
        "entropy_nats",
        "entropy_bits",
    )
    result = {key: statistics.mean(r[key] for r in rows) for key in keys}
    result["count"] = len(rows)
    result["space_q_range"] = [min(r["space_q"] for r in rows), max(r["space_q"] for r in rows)]
    result["concentration_fractions"] = {
        str(t): sum(r["top1_q"] >= t for r in rows) / len(rows)
        for t in PROTOCOL["concentration_thresholds"]
    }
    result["class_counts"] = {
        k: sum(r["classification"] == k for r in rows)
        for k in sorted({r["classification"] for r in rows})
    }
    return result


def add_losses(groups, key, values):
    total, count = groups.get(key, [0.0, 0])
    groups[key] = [total + sum(values), count + len(values)]


def summarize_losses(groups):
    return {
        key: {
            "sum_nll": total,
            "count": count,
            "nll": total / count,
            "bits_per_byte": total / count / math.log(2),
        }
        for key, (total, count) in groups.items()
        if count
    }


def paired_summary(rows):
    result = {}
    for length in LENGTHS:
        values = [r["scores"][str(length)]["nll"] for r in rows]
        deltas = [v - r["scores"]["32"]["nll"] for v, r in zip(values, rows, strict=True)]
        rng = random.Random(1730)
        boot = sorted(statistics.mean(rng.choices(deltas, k=len(deltas))) for _ in range(2000))
        tolerance = PROTOCOL["unchanged_nll_tolerance"]
        result[str(length)] = {
            "count": len(values),
            "mean_nll": statistics.mean(values),
            "median_nll": statistics.median(values),
            "paired_delta": statistics.mean(deltas),
            "paired_delta_ci95": [boot[49], boot[1949]],
            "improved_fraction": sum(d < -tolerance for d in deltas) / len(deltas),
            "worsened_fraction": sum(d > tolerance for d in deltas) / len(deltas),
            "unchanged_fraction": sum(abs(d) <= tolerance for d in deltas) / len(deltas),
        }
    return result
