"""Aggregate checksummed phase diagnostics without model or corpus evaluation."""

import random
import statistics
from collections import Counter, defaultdict

from context_distribution_study import ROOT, STUDY, Store, read, sha
from context_study_metrics import PROTOCOL, require
from context_study_phases import BASES, selections

from unified_edge.training.config import canonical_hash


def paired(rows, aggregate=False):
    result = {}
    for base in BASES:
        grouped = defaultdict(list)
        values = []
        deltas = []
        for row in rows:
            phase = row["phase"]
            value = row["scores"][str(base + phase)]
            difference = value - row["scores"][str(32 + phase)]
            values.append(value)
            deltas.append(difference)
            grouped[row["anchor"]["key"]].append(difference)
        samples = [statistics.mean(v) for v in grouped.values()] if aggregate else deltas
        rng = random.Random(1730)
        # Identical values have an exactly degenerate bootstrap distribution.
        boot = (
            [samples[0]] * 2000
            if min(samples) == max(samples)
            else sorted(statistics.mean(rng.choices(samples, k=len(samples))) for _ in range(2000))
        )
        tolerance = PROTOCOL["unchanged_nll_tolerance"]
        result[str(base)] = {
            "anchor_count": len(grouped),
            "target_phase_count": len(values),
            "mean_nll": statistics.mean(values),
            "median_nll": statistics.median(values),
            "paired_delta": statistics.mean(deltas),
            "paired_delta_ci95": [boot[49], boot[1949]],
            "improved_fraction": sum(v < -tolerance for v in deltas) / len(deltas),
            "worsened_fraction": sum(v > tolerance for v in deltas) / len(deltas),
            "unchanged_fraction": sum(abs(v) <= tolerance for v in deltas) / len(deltas),
        }
    return result


def ranges(rows):
    return {
        key: {
            "min": min(r[key] for r in rows),
            "max": max(r[key] for r in rows),
            "mean": statistics.mean(r[key] for r in rows),
        }
        for key in rows[0]
    }


def forensic_summary(rows):
    values = [r["forensic"] for r in rows]
    logits = [r["logits"] for r in values]
    return {
        "comparisons": len(values),
        "equal_logits_count": sum(v["equal"] for v in logits),
        "same_top1_count": sum(v["top1_short"] == v["top1_long"] for v in logits),
        "top1_byte_pair_counts": dict(
            Counter(f"{v['top1_short']}->{v['top1_long']}" for v in logits)
        ),
        "logits": ranges(
            [
                {
                    k: v[k]
                    for k in (
                        "max_abs",
                        "mean_abs",
                        "l2",
                        "target_logprob_delta",
                        "q_total_variation",
                    )
                }
                for v in logits
            ]
        ),
        "layers": [
            {
                "conv": ranges([v["layers"][i]["conv"] for v in values]),
                "ssm": ranges([v["layers"][i]["ssm"] for v in values]),
            }
            for i in range(len(values[0]["layers"]))
        ],
        "hierarchy": {
            "pending_equal_count": sum(v["hierarchy"]["pending_equal"] for v in values),
            "pending_lengths": sorted({v["hierarchy"]["pending_length_short"] for v in values}),
            "hidden": ranges([v["hierarchy"]["hidden"] for v in values]),
        },
        "shared_state_differs_count": sum(
            any(layer[k]["max_abs"] > 0 for layer in v["layers"] for k in ("conv", "ssm"))
            for v in values
        ),
    }


def summarize():
    root = STUDY / "phase-matched"
    binding = read(root / "binding.json")
    require(
        binding["main_binding_sha256"] == sha(STUDY / "binding.json"), "phase main binding changed"
    )
    require(
        binding["tool_sha256"] == sha(ROOT / "scripts/context_study_phases.py"),
        "phase producer changed",
    )
    require(
        binding["ac008_sha256"]
        == sha(
            ROOT / "docs/architecture_changes/AC-008_Context_and_Distribution_Promotion_Gates.md"
        ),
        "AC-008 changed since phase binding",
    )
    expected = selections(read(STUDY / "binding.json")["anchors"])
    require(
        binding["anchors"] == expected and binding["selection_sha256"] == canonical_hash(expected),
        "phase selection changed",
    )
    store = Store(root, binding)
    require(
        (store.get("integrity_final") or {}).get("status") == "PASS", "phase supplement incomplete"
    )
    rows = []
    seconds = 0
    for batch in range(16):
        for phase in range(8):
            block = store.get(f"batch_{batch:02d}_phase_{phase}")
            require(block is not None, "missing phase block")
            require(
                [v["anchor"] for v in block["rows"]] == expected[batch * 8 : (batch + 1) * 8],
                "phase anchor mismatch",
            )
            for row in block["rows"]:
                require(
                    row["phase"] == phase and set(row["scores"]) == {str(n + phase) for n in BASES},
                    "phase grid mismatch",
                )
                require(("forensic" in row) == (batch % 4 == 0), "forensic selection mismatch")
                if "forensic" in row:
                    h = row["forensic"]["hierarchy"]
                    require(
                        h["pending_equal"]
                        and h["pending_length_short"] == h["pending_length_long"] == phase,
                        "pending phase mismatch",
                    )
            rows.extend(block["rows"])
            seconds += block["seconds"]
    domains = sorted({r["anchor"]["domain"] for r in rows})

    def grouped(selected, aggregate=False):
        return {
            "all": paired(selected, aggregate),
            "domains": {
                d: paired([r for r in selected if r["anchor"]["domain"] == d], aggregate)
                for d in domains
            },
        }

    forensic_rows = [r for r in rows if "forensic" in r]
    f = forensic_summary(forensic_rows)
    all_equal = f["equal_logits_count"] == f["comparisons"]
    mechanism = (
        "INTERNAL_STATE_DIFFERS_OUTPUT_INSENSITIVE"
        if all_equal and f["shared_state_differs_count"] == f["comparisons"]
        else "GENERAL_HISTORY_INSENSITIVITY_AT_2M_STAGE_A"
        if all_equal
        else "MIXED_BY_PHASE_OR_DOMAIN"
    )
    sanity = store.get("architecture_sanity")
    require(
        sanity["status"] == "PASS" and sanity["forensic"]["logits"]["max_abs"] > 0,
        "architecture sanity failed",
    )
    return {
        "binding": binding,
        "measurement_hashes": {
            p.name: sha(p) for p in sorted(root.glob("*.json")) if p.name != "progress.json"
        },
        "phases": {str(p): grouped([r for r in rows if r["phase"] == p]) for p in range(8)},
        "aggregate": grouped(rows, True),
        "forensics": f,
        "forensics_by_phase": {
            str(p): forensic_summary([r for r in forensic_rows if r["phase"] == p])
            for p in range(8)
        },
        "forensics_by_domain": {
            d: forensic_summary([r for r in forensic_rows if r["anchor"]["domain"] == d])
            for d in domains
        },
        "architecture_sanity": sanity,
        "mechanism": mechanism,
        "measured_block_seconds": seconds,
        "uncertainty": (
            "Per-phase paired anchor bootstrap; aggregate averages eight phases "
            "per anchor before resampling. Anchors within documents are not independent."
        ),
        "scope": (
            "FP32, this checkpoint and selected validation histories only; "
            "no architectural-defect or effective-context claim."
        ),
    }


def markdown(report):
    f = report["forensics"]
    lines = [
        "## Separately bound phase-matched supplement",
        "",
        "128 deterministic anchors (32/domain), eight phases, 4,096 target-history scores. "
        "Within phase r, histories are 32+r / 64+r / 128+r / 248+r bytes. "
        "Original boundary measurements above remain unchanged. "
        "Batch size 8; FP32 model and forensic arithmetic.",
        "",
        "| Phase | Anchors | NLL short / 64+r / 128+r / 248+r | Long minus short | "
        "95% paired CI | Improved / worse / unchanged |",
        "|---|---:|---|---:|---|---|",
    ]
    for phase, group in report["phases"].items():
        v = group["all"]["248"]
        means = " / ".join(f"{group['all'][str(n)]['mean_nll']:.9f}" for n in BASES)
        lines.append(
            f"| {phase} | {v['anchor_count']} | {means} | {v['paired_delta']:.9g} | "
            f"{v['paired_delta_ci95']} | {v['improved_fraction']:.3f} / "
            f"{v['worsened_fraction']:.3f} / {v['unchanged_fraction']:.3f} |"
        )
    lines += [
        "",
        report["uncertainty"],
        "The JSON includes all four history summaries, medians, domain-wise "
        "paired intervals and aggregate across phases.",
        "",
        "## Logit equality and internal-state sensitivity",
        "",
        f"Mechanism: **{report['mechanism']}**. {report['scope']}",
        f"Forensic subset: {f['comparisons']} comparisons (8 anchors/domain × 8 phases); "
        f"torch.equal confirms {f['equal_logits_count']} identical logit vectors. "
        f"Shared state differs in {f['shared_state_differs_count']} comparisons.",
        f"Maximum finite-logit absolute delta: {f['logits']['max_abs']['max']:.9g}; "
        f"maximum q total variation: {f['logits']['q_total_variation']['max']:.9g}. "
        "PAD/BOS remain -inf and are excluded only from numeric subtraction, not torch.equal.",
        "",
        "| Layer | Conv max delta | SSM max delta | Conv short / long norm range "
        "| SSM short / long norm range |",
        "|---|---:|---:|---|---|",
    ]

    def norm_range(v):
        return " / ".join(
            f"[{v[k]['min']:.6g}, {v[k]['max']:.6g}]" for k in ("norm_short", "norm_long")
        )

    for i, v in enumerate(f["layers"]):
        lines.append(
            f"| {i} | {v['conv']['max_abs']['max']:.9g} | {v['ssm']['max_abs']['max']:.9g} | "
            f"{norm_range(v['conv'])} | {norm_range(v['ssm'])} |"
        )
    lines += [
        "",
        f"Pending bytes agree in {f['hierarchy']['pending_equal_count']} comparisons; "
        "lengths 0..7 match phase. "
        f"Decoder hidden maximum delta: {f['hierarchy']['hidden']['max_abs']['max']:.9g}. "
        "JSON retains L2, mean absolute delta, state norms, target log-probability differences, "
        "top-1 agreement and phase/domain summaries.",
        "",
        "## Architecture sanity",
        "",
        "Deterministic seed-43 untrained accepted model, eight old zeros "
        "versus eight old 255 bytes, "
        "then the identical 32-byte suffix 0..31. No weights tuned and no optimizer constructed.",
        f"Result: {report['architecture_sanity']['status']}; finite-logit max delta "
        f"{report['architecture_sanity']['forensic']['logits']['max_abs']:.9g}. "
        "This demonstrates architectural transmission in this fixture, "
        "not trained long-context quality.",
        report["binding"]["protocol"]["recency_ablation"],
        "",
    ]
    return lines
