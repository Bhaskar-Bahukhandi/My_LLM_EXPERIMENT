"""Aggregate immutable disposable A/B results; no model execution or promotion."""

import argparse
import json
import random
import statistics
from collections import defaultdict

from context_distribution_study import PARAMETER_SHA, ROOT, STUDY, Store, read, sha
from context_phase_summary import forensic_summary, paired
from context_study_metrics import distribution_summary, paired_summary, require
from context_training_ab import AB, ARMS, CHECKPOINTS
from finish_context_distribution_study import whitespace_summary

from unified_edge.training.checkpoint import load_checkpoint
from unified_edge.training.config import canonical_hash


def phase_rows(store):
    return [
        row
        for batch in range(16)
        for phase in range(8)
        for row in store.get(f"phase_{batch:02d}_{phase}")["rows"]
    ]


def phase_results(rows):
    domains = sorted({r["anchor"]["domain"] for r in rows})
    result = {
        "aggregate": paired(rows, True),
        "domains": {
            d: paired([r for r in rows if r["anchor"]["domain"] == d], True) for d in domains
        },
        "phases": {
            str(p): {
                "all": paired([r for r in rows if r["phase"] == p]),
                "domains": {
                    d: paired([r for r in rows if r["phase"] == p and r["anchor"]["domain"] == d])
                    for d in domains
                },
            }
            for p in range(8)
        },
    }
    selected = [r for r in rows if "forensic" in r]
    if selected:
        result["forensics"] = forensic_summary(selected)
        f = result["forensics"]
        if f["equal_logits_count"] == f["comparisons"]:
            label = (
                "INTERNAL_STATE_DIFFERS_OUTPUT_INSENSITIVE"
                if f["shared_state_differs_count"] == f["comparisons"]
                else "GENERAL_HISTORY_INSENSITIVITY_AT_2M_STAGE_A"
            )
        elif f["equal_logits_count"] == 0:
            label = "LONG_HISTORY_SENSITIVE"
        else:
            label = "MIXED_BY_PHASE_OR_DOMAIN"
        result["mechanism"] = label
        result["forensics_by_phase"] = {
            str(p): forensic_summary([r for r in selected if r["phase"] == p]) for p in range(8)
        }
        result["forensics_by_domain"] = {
            d: forensic_summary([r for r in selected if r["anchor"]["domain"] == d])
            for d in domains
        }
    else:
        result["forensics"] = "NOT_SCHEDULED_AT_128"
    return result


def interval(values):
    rng = random.Random(1730)
    means = (
        [values[0]] * 2000
        if min(values) == max(values)
        else sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(2000))
    )
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "ci95": [means[49], means[1949]],
        "anchors": len(values),
    }


def arm_difference(a, b, length=64, benefit=False):
    index = {(r["anchor"]["key"], r["phase"]): r for r in a}
    require(set(index) == {(r["anchor"]["key"], r["phase"]) for r in b}, "unpaired arm anchors")
    groups = defaultdict(list)
    for right in b:
        left = index[(right["anchor"]["key"], right["phase"])]
        phase = right["phase"]
        delta = right["scores"][str(length + phase)] - left["scores"][str(length + phase)]
        if benefit:
            delta -= right["scores"][str(32 + phase)] - left["scores"][str(32 + phase)]
        groups[right["anchor"]["key"]].append(delta)
    return interval([statistics.mean(v) for v in groups.values()])


def summarize_evaluation(root, step):
    store = Store(root, read(root / "binding.json"))
    require((store.get("complete") or {}).get("status") == "PASS", "incomplete branch evaluation")
    rows = phase_rows(store)
    generations = [store.get(f"generation_{i}") for i in range(6)]
    positions = [store.get(p.stem)["distribution"] for p in sorted(root.glob("position_*.json"))]
    result = {
        "checkpoint_binding": read(root / "binding.json"),
        "integrity": store.get("complete"),
        "small": {str(n): store.get(f"small_{n}") for n in (32, 64)},
        "phase": phase_results(rows),
        "matched": paired_summary([store.get(f"matched_{i:03d}") for i in range(32)]),
        "generations": [
            dict(
                {k: r[k] for k in ("prefix_hex", "output_hex", "escaped", "shared_steps")},
                whitespace=whitespace_summary(r["steps"]),
            )
            for r in generations
        ],
        "generation_distribution": distribution_summary(
            [s for r in generations for s in r["steps"]]
        ),
        "position_distribution": distribution_summary(positions),
    }
    if step == 512:
        result["full"] = {str(n): store.get(f"full_{n}") for n in (32, 64)}
        require(
            all(v["count"] == 500000 for v in result["full"].values()), "full validation incomplete"
        )
    return result, rows


def severe(summary):
    return summary["class_counts"].get("SEVERE_BYTE_MODE_CONCENTRATION", 0) / summary["count"]


def decision(parent, a, b, benefit_difference, absolute_difference):
    fa, fb = a["full"]["32"], b["full"]["32"]
    history = b["phase"]["aggregate"]["64"]
    gains = [parent["segmented"]["32"]["nll"] - v["nll"] for v in (fa, fb)]
    distributions = {
        "positions": (
            parent["position_distribution_summary"],
            a["position_distribution"],
            b["position_distribution"],
        ),
        "generation": (
            distribution_summary([s for g in parent["generations"] for s in g["steps"]]),
            a["generation_distribution"],
            b["generation_distribution"],
        ),
    }
    distribution_ok = all(
        severe(vb) - severe(ref) <= 0.05 and vb["control_mass_p"] - ref["control_mass_p"] <= 0.001
        for p, va, vb in distributions.values()
        for ref in (p, va)
    )
    gates = {
        "beneficial_history": history["paired_delta"] < -1e-5
        and history["paired_delta_ci95"][1] < 0,
        "better_history_use_than_a": benefit_difference["ci95"][1] < 0,
        "better_long_history_prediction_than_a": absolute_difference["ci95"][1] < 0,
        "full32_nonregression": fb["nll"] <= fa["nll"] + 0.01
        and fb["nll"] <= parent["segmented"]["32"]["nll"] + 0.01,
        "domain_nonregression": all(
            b["full"][length]["domains"][d]["nll"] <= a["full"][length]["domains"][d]["nll"] + 0.03
            for length in ("32", "64")
            for d in a["full"][length]["domains"]
        ),
        "distribution_nonregression": distribution_ok,
        "practical_compute": b["training_seconds"] <= 2 * a["training_seconds"]
        and b["training_seconds"] <= 1800,
    }
    if all(gates.values()):
        recommendation = "REVIEW_64_BYTE_CURRICULUM"
    elif max(gains) < 0.01 and not gates["beneficial_history"]:
        recommendation = "FREEZE_2M_AND_REVIEW_SCALING"
    elif fa["nll"] <= fb["nll"] or not gates["beneficial_history"]:
        recommendation = "CONTINUE_32_BYTE_RUN"
    else:
        recommendation = "MORE_EVIDENCE_REQUIRED"
    return {
        "recommendation": recommendation,
        "gates": gates,
        "full32_improvements": gains,
        "distribution_tradeoffs": {
            k: {
                "parent_severe_fraction": severe(p),
                "a_severe_fraction": severe(va),
                "b_severe_fraction": severe(vb),
                "b_minus_a_control_mass": vb["control_mass_p"] - va["control_mass_p"],
            }
            for k, (p, va, vb) in distributions.items()
        },
        "authority": "HUMAN REVIEW ONLY; no promotion or production continuation",
    }


def build(validation):
    binding = read(AB / "binding.json")
    root = Store(AB, binding)
    require((root.get("integrity_final") or {}).get("status") == "PASS", "A/B unfinished")
    schedule = root.get("schedule")["blocks"]
    parent = read(ROOT / "reports/context_distribution_study_2m.json")
    original = Store(STUDY, read(STUDY / "binding.json"))
    selected_keys = {a["key"] for a in binding["evaluation"]["matched_anchors"]}
    parent_matched = [
        original.get(f"anchor_{i:04d}")
        for i, a in enumerate(parent["binding"]["anchors"])
        if a["key"] in selected_keys
    ]
    require(len(parent_matched) == 32, "parent matched subset incomplete")
    arms = {}
    final_rows = {}
    for name, length in ARMS:
        store = Store(AB / name, read(AB / name / "binding.json"))
        observations = [store.get(f"observation_{step:06d}") for step in range(1, 513)]
        require(all(observations), "missing training observation")
        curve = []
        for index, row in enumerate(observations):
            v = row["semantic"]
            require(
                v["update"] == index + 1
                and v["target_bytes"] == 128
                and v["cumulative_target_bytes"] == 128 * (index + 1)
                and v["scheduler_completed"] == 5001 + index
                and v["finite_gradients"]
                and v["target_sha256"] == schedule[index]["sha256"],
                "training schedule/finite gate mismatch",
            )
            curve.append(
                {
                    "update": index + 1,
                    "mean_nll": v["mean_nll"],
                    "moving_average": {
                        str(n): statistics.mean(
                            r["semantic"]["mean_nll"]
                            for r in observations[index + 1 - n : index + 1]
                        )
                        if index + 1 >= n
                        else None
                        for n in (16, 64, 128)
                    },
                }
            )
        checkpoint_ids = {}
        for step in CHECKPOINTS:
            path = AB / name / f"checkpoints/update_{step:06d}"
            state = load_checkpoint(path)
            require(
                state["kind"] == "DISPOSABLE_AB_ONLY"
                and state["binding_sha256"] == canonical_hash(binding)
                and state["experimental_update"] == step
                and state["block_cursor"] == step
                and state["target_bytes"] == 128 * step
                and state["scheduler"]["completed"] == 5000 + step,
                "checkpoint lineage/counter mismatch",
            )
            checkpoint_ids[str(step)] = {
                "manifest_sha256": sha(path / "manifest.json"),
                "state_sha256": sha(path / "state.pt"),
            }
        evaluations = {}
        for step in (128, 256, 512):
            summary, rows = summarize_evaluation(
                AB / "evaluation" / name / f"update_{step:06d}", step
            )
            evaluations[str(step)] = summary
            if step == 512:
                final_rows[name] = rows
        total_seconds = sum(r["seconds"] for r in observations)
        replay_records = [store.get(p.stem) for p in sorted((AB / name).glob("replay_*.json"))]
        replay_seconds = sum(v["seconds"] for v in replay_records)
        final = evaluations["512"]
        final["training_seconds"] = total_seconds
        gain = parent["segmented"]["32"]["nll"] - final["full"]["32"]["nll"]
        arms[name] = {
            "length": length,
            "updates": 512,
            "target_bytes": 65536,
            "training_seconds": total_seconds,
            "recorded_physical_training_seconds": total_seconds + replay_seconds,
            "recorded_replay_seconds": replay_seconds,
            "target_bytes_per_second": 65536 / total_seconds,
            "curve": curve,
            "learning_rate_sequence_sha256": canonical_hash(
                {"values": [v["semantic"]["learning_rate"] for v in observations]}
            ),
            "learning_rate_first_last": [
                observations[i]["semantic"]["learning_rate"] for i in (0, -1)
            ],
            "finite_gradients_all_updates": all(
                v["semantic"]["finite_gradients"] for v in observations
            ),
            "gradient_norm_range": [
                fn(v["semantic"]["gradient_norm"] for v in observations) for fn in (min, max)
            ],
            "clipped_updates": sum(v["semantic"]["clipped"] for v in observations),
            "checkpoints": checkpoint_ids,
            "evaluations": evaluations,
            "learning_efficiency": {
                "nll_gain_per_1000_target_bytes": gain / 65.536,
                "nll_gain_per_update": gain / 512,
                "nll_gain_per_training_minute": gain / ((total_seconds + replay_seconds) / 60),
            },
            "memory": {
                "peak_allocated_bytes": max(
                    r["memory"]["cuda"]["peak_allocated_bytes"] for r in observations
                ),
                "peak_reserved_bytes": max(
                    r["memory"]["cuda"]["peak_reserved_bytes"] for r in observations
                ),
                "minimum_sampled_free_bytes": min(
                    r["memory"]["cuda"]["free_device_bytes"] for r in observations
                ),
                "rss_status": "MEASURED"
                if all(r["memory"]["process_rss_status"] == "MEASURED" for r in observations)
                else "UNVERIFIED",
                "peak_observed_rss_bytes": (
                    max(r["memory"]["process_rss_bytes"] for r in observations)
                    if all(r["memory"]["process_rss_status"] == "MEASURED" for r in observations)
                    else None
                ),
                "tensor_payloads": {
                    k: observations[-1]["memory"][k]
                    for k in (
                        "parameter_bytes",
                        "gradient_bytes",
                        "optimizer_state_bytes",
                        "canonical_recurrent_state_bytes",
                        "recurrent_note",
                        "activation_autograd_bytes",
                    )
                },
            },
            "physical_replay_updates": len(replay_records),
            "recovery": [store.get(p.stem) for p in sorted((AB / name).glob("recovery_*.json"))],
        }
    a, b = [arms[name]["evaluations"]["512"] for name, _ in ARMS]
    require(
        arms[ARMS[0][0]]["learning_rate_sequence_sha256"]
        == arms[ARMS[1][0]]["learning_rate_sequence_sha256"],
        "arms used different LR progression",
    )
    difference = arm_difference(*[final_rows[name] for name, _ in ARMS], benefit=True)
    absolute = arm_difference(*[final_rows[name] for name, _ in ARMS])
    result = {
        "schema": "context-training-ab-review-1",
        "status": "READY_FOR_HUMAN_DECISION",
        "binding": binding,
        "binding_sha256": sha(AB / "binding.json"),
        "schedule_sha256": canonical_hash(schedule),
        "parent": {
            "parameter_sha256": PARAMETER_SHA,
            "step": 5000,
            "matched_panel": paired_summary(parent_matched),
            "small_validation_panel": "NOT_MEASURED_AT_0; accepted full baseline reused",
            "full": {str(n): parent["segmented"][str(n)] for n in (32, 64)},
            "phase": {
                k: parent["phase_supplement"][k]
                for k in ("aggregate", "phases", "forensics", "mechanism")
            },
            "generation_distribution": distribution_summary(
                [s for g in parent["generations"] for s in g["steps"]]
            ),
            "generations": [
                {k: g[k] for k in ("prefix_hex", "output_hex", "escaped", "whitespace")}
                for g in parent["generations"]
            ],
            "position_distribution": parent["position_distribution_summary"],
        },
        "initial_equivalence": root.get("initial_equivalence"),
        "preflights": {str(n): root.get(f"preflight_{n}") for n in (32, 64)},
        "arms": arms,
        "b_minus_a_history_benefit": difference,
        "b_minus_a_long_nll": absolute,
        "decision": decision(parent, a, b, difference, absolute),
        "validation": validation,
        "limitations": [
            "One seed, one matched 65536-byte TRAIN sample; no population-level or SOTA claim.",
            "128 total hash-selected anchors (32/domain); anchors within "
            "documents correlated; bootstrap resamples anchors after phase "
            "averaging.",
            "A then B sequential execution; thermal/cache/order differences "
            "confound timing. Evaluation excluded from measured training time.",
            "No effective-context certification, TEST evaluation, curriculum "
            "implementation or production promotion.",
            "CUDA reference uses quadratic transition storage; no "
            "efficient-kernel inference from short profiles.",
            "No independent 500k baseline rerun: exact identical clone "
            "weights/config/source and optimizer/RNG equality reuse accepted "
            "parent evidence.",
            "Generic process RSS sampled, not inferred from tensor arithmetic; "
            "activations not separately measured.",
        ],
        "production_updates": 0,
        "exposure_note": "Experiment-local TRAIN block schedule; no claim that all selected "
        "bytes were unseen by the parent. Both arms have identical prior-exposure status.",
        "test": "SEALED; hash-only integrity reads",
    }
    result["full_validation_deltas"] = {
        length: {
            "a_minus_parent": a["full"][length]["nll"] - parent["segmented"][length]["nll"],
            "b_minus_parent": b["full"][length]["nll"] - parent["segmented"][length]["nll"],
            "b_minus_a": b["full"][length]["nll"] - a["full"][length]["nll"],
        }
        for length in ("32", "64")
    }
    result["limitations"].append(
        "Training seconds sum measured update intervals including finite/memory checks; "
        "data selection, journal/checkpoint I/O and evaluation are excluded. End-to-end "
        "study wall time was not separately instrumented. Replay timing is separate; "
        "unpublished in-flight work cannot be reconstructed from durable records."
    )
    return result


def markdown(r):
    lines = [
        "# Disposable 2M context-training A/B",
        "",
        "**Recommendation: " + r["decision"]["recommendation"] + " — HUMAN REVIEW ONLY.**",
        "",
        "Production parent remains immutable step 5000. Both branches are "
        "DISPOSABLE_AB_ONLY; no branch is promoted.",
        f"Binding SHA-256: `{r['binding_sha256']}`. Schedule SHA-256: `{r['schedule_sha256']}`.",
        "Exactly 512 non-overlapping 128-byte TRAIN blocks; 65,536 identical "
        "target bytes and 512 optimizer updates per arm. "
        "Quotas: general text 256, code 128, documentation 77, structured math "
        "51. A resets at 0/32/64/96; B at 0/64. "
        "Both sum 128 losses and divide by 128; parent AdamW moments and cosine "
        "scheduler advance locally from 5000 to 5512.",
        "",
        "## Parent / A / B comparison",
        "",
        "| Metric | Parent | A:32 | B:64 |",
        "|---|---:|---:|---:|",
    ]
    finals = [r["arms"][name]["evaluations"]["512"] for name, _ in ARMS]
    for length in ("32", "64"):
        vals = [r["parent"]["full"][length], *[f["full"][length] for f in finals]]
        lines.append(
            f"| Full {length} NLL (500,000 targets) | "
            + " | ".join(f"{v['nll']:.10f}" for v in vals)
            + " |"
        )
        lines.append(
            f"| Full {length} bits/byte | "
            + " | ".join(f"{v['bits_per_byte']:.8f}" for v in vals)
            + " |"
        )
    lines.append(
        "| Phase-matched 64+r history delta | "
        + " | ".join(
            f"{v:.9g}"
            for v in [
                r["parent"]["phase"]["aggregate"]["all"]["64"]["paired_delta"],
                *[f["phase"]["aggregate"]["64"]["paired_delta"] for f in finals],
            ]
        )
        + " |"
    )
    lines.append(
        "| Mechanism | "
        + " | ".join(
            [r["parent"]["phase"]["mechanism"], *[f["phase"]["mechanism"] for f in finals]]
        )
        + " |"
    )
    for field, label in (
        ("space_q", "Validation-context q(space)"),
        ("entropy_nats", "Validation-context q entropy"),
    ):
        lines.append(
            f"| {label} | "
            + " | ".join(
                f"{v[field]:.7f}"
                for v in [
                    r["parent"]["position_distribution"],
                    *[f["position_distribution"] for f in finals],
                ]
            )
            + " |"
        )
    lines += [
        "",
        "Full-validation NLL deltas: `" + json.dumps(r["full_validation_deltas"]) + "`.",
        f"B-A paired history-benefit effect: `{r['b_minus_a_history_benefit']}`.",
        f"B-A absolute long-history target NLL: `{r['b_minus_a_long_nll']}`.",
        "Mechanism labels describe sensitivity, not whether the sensitivity improves prediction.",
        "",
        "| Gate | Result |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in r["decision"]["gates"].items()]
    for name, _ in ARMS:
        arm = r["arms"][name]
        lines += [
            "",
            f"## {name}",
            "",
            f"Training: {arm['training_seconds']:.2f}s, "
            f"{arm['target_bytes_per_second']:.2f} target bytes/s; "
            f"physical replay updates: {arm['physical_replay_updates']}.",
            "Learning efficiency: `" + json.dumps(arm["learning_efficiency"]) + "`.",
            "Memory: `" + json.dumps(arm["memory"]) + "`.",
            "",
            "| Update | MA16 train NLL | MA64 | MA128 | Small32 / Small64 "
            "validation | Long-history64 delta [CI] | Mechanism |",
            "|---|---:|---:|---:|---|---|---|",
        ]
        for step in (128, 256, 512):
            e = arm["evaluations"][str(step)]
            curve = arm["curve"][step - 1]["moving_average"]
            p = e["phase"]["aggregate"]["64"]
            lines.append(
                f"| {step} | {curve['16']:.7f} | {curve['64']:.7f} | {curve['128']:.7f} | "
                f"{e['small']['32']['nll']:.7f} / {e['small']['64']['nll']:.7f} | "
                f"{p['paired_delta']:.9g} {p['paired_delta_ci95']} | "
                f"{e['phase'].get('mechanism', 'forensics not scheduled')} |"
            )
        lines += [
            "",
            "### Final domain and position results",
            "",
            "| Domain | 32 NLL | 64 NLL |",
            "|---|---:|---:|",
        ]
        e = arm["evaluations"]["512"]
        for d, v in e["full"]["32"]["domains"].items():
            lines.append(f"| {d} | {v['nll']:.9f} | {e['full']['64']['domains'][d]['nll']:.9f} |")
        lines += [
            "",
            "64-byte position buckets: `" + json.dumps(e["full"]["64"]["buckets"]) + "`.",
            "Final phase results (full domain/phase confidence intervals and "
            "all curve points in JSON):",
            "",
            "| Phase | Short mean NLL | 64+r delta | 128+r delta | 248+r delta |",
            "|---|---:|---:|---:|---:|",
        ]
        for phase, v in e["phase"]["phases"].items():
            lines.append(
                f"| {phase} | {v['all']['32']['mean_nll']:.8f} | "
                + " | ".join(f"{v['all'][str(n)]['paired_delta']:.9g}" for n in (64, 128, 248))
                + " |"
            )
        f = e["phase"]["forensics"]
        lines += [
            "",
            f"Forensics: {f['equal_logits_count']}/{f['comparisons']} identical logit pairs; "
            f"max logit delta {f['logits']['max_abs']['max']:.9g}; "
            f"max q TV {f['logits']['q_total_variation']['max']:.9g}; "
            f"decoder-hidden max delta {f['hierarchy']['hidden']['max_abs']['max']:.9g}. "
            "Layerwise states/norms are in JSON.",
            "",
            "| Prompt | Greedy output | Longest space run | Mean q(space) | Entropy |",
            "|---|---|---:|---:|---:|",
        ]
        for g in e["generations"]:
            w = g["whitespace"]
            lines.append(
                f"| `{bytes.fromhex(g['prefix_hex'])!r}` | `{g['escaped']}` | "
                f"{w['longest_space_run']} | {w['distribution']['space_q']:.6f} | "
                f"{w['distribution']['entropy_nats']:.6f} |"
            )
    lines += [
        "",
        "## Distribution tradeoffs",
        "",
        json.dumps(r["decision"]["distribution_tradeoffs"]),
        "",
        "## Integrity, validation and limitations",
        "",
    ]
    lines += ["- " + s for s in r["limitations"]]
    lines += ["", "| Check | Exit | Seconds |", "|---|---:|---:|"]
    lines += [
        f"| {v['name']} | {v['returncode']} | {v['seconds']:.2f} |"
        for v in r["validation"]["commands"]
    ]
    lines += [
        "",
        "All ten production checkpoints, progress/cursor/optimizer/scheduler, "
        "accepted Stage-A/context reports, corpus, model source and FINAL files "
        "remain unchanged. TEST stayed sealed. No production continuation or "
        "promotion.",
        "",
        "2M CONTEXT-TRAINING A/B STATUS: READY FOR HUMAN DECISION",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", required=True)
    args = parser.parse_args()
    from validate_context_ab import verify_receipt

    result = build(verify_receipt(ROOT / args.validation))
    output = ROOT / "reports/context_training_ab_2m"
    output.with_suffix(".json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    output.with_suffix(".md").write_text(markdown(result), encoding="utf-8")
