"Aggregate immutable study measurements; never load a model or evaluate data."

import argparse
import json
import statistics
from collections import Counter

from context_distribution_study import ROOT, STUDY, Store, read, sha, verify_inventory
from context_phase_summary import markdown as phase_markdown
from context_phase_summary import summarize as phase_summary
from context_study_metrics import LENGTHS, distribution_summary, paired_summary, require

from unified_edge.training.config import canonical_hash

VALIDATION_FILES = (
    "scripts/context_distribution_study.py",
    "scripts/context_study_metrics.py",
    "scripts/context_study_positions.py",
    "scripts/resume_context_distribution_study.py",
    "scripts/finish_context_distribution_study.py",
    "tests/test_context_distribution_study.py",
    "scripts/context_study_phases.py",
    "scripts/context_phase_summary.py",
    "scripts/validate_context_study.py",
    "tests/test_context_study_phases.py",
    "tests/test_training.py",
    "tests/test_dense_model.py",
    "tests/test_future_capabilities_spec.py",
)


def measurement_sets():
    return {
        name or "main": canonical_hash(
            {
                p.name: sha(p)
                for p in sorted((STUDY / name).glob("*.json"))
                if p.name != "progress.json"
            }
        )
        for name in ("", "position-distributions", "phase-matched")
    }


def validation_closeout(path=None):
    path = path or STUDY / "validation-closeout" / "commands.json"
    envelope = read(path)
    value = envelope["value"]
    require(envelope["sha256"] == canonical_hash(value), "validation evidence hash mismatch")
    require(value["schema"] == "study-validation-1", "validation evidence schema mismatch")
    require(value["final_integrity"]["status"] == "PASS", "final integrity gate failed")
    require(
        value["final_integrity"]["measurement_set_sha256"] == measurement_sets(),
        "validation measurement set is stale",
    )
    expected = {
        "pytest",
        "ruff",
        "format",
        "compileall",
        "cpu_dependencies",
        "cuda_dependencies",
        "diff",
    }
    require(
        {row["name"] for row in value["commands"]} == expected, "missing final validation command"
    )
    require(len(value["commands"]) == len(expected), "duplicate validation commands")
    require(all(row["returncode"] == 0 for row in value["commands"]), "final validation failed")
    require(set(value["files_sha256"]) == set(VALIDATION_FILES), "incomplete tested-file inventory")
    for relative, digest in value["files_sha256"].items():
        require(sha(ROOT / relative) == digest, "validation evidence is stale: " + relative)
    require(
        value["measurement_integrity"]
        == {
            "main": sha(STUDY / "integrity_final.json"),
            "positions": sha(STUDY / "position-distributions" / "integrity_final.json"),
            "phases": sha(STUDY / "phase-matched" / "integrity_final.json"),
            "architecture_sanity": sha(STUDY / "phase-matched" / "architecture_sanity.json"),
        },
        "validation predates final measurements",
    )
    return dict(value, evidence_sha256=sha(path))


def probability_summary(rows):
    summary = distribution_summary(rows)
    summary["mean_control_probabilities_p"] = [
        statistics.mean(r["control_probabilities_p"][i] for r in rows) for i in range(11)
    ]
    summary["masked_symbol_ids"] = sorted({i for r in rows for i in r["masked_symbol_ids"]})
    return summary


def whitespace_summary(rows):
    selected = [r["top1_byte"] for r in rows]
    runs, run = [], 0
    for byte in selected:
        if byte == 32:
            run += 1
        elif run:
            runs.append(run)
            run = 0
    if run:
        runs.append(run)
    return {
        "space_runs": runs,
        "longest_space_run": max(runs, default=0),
        "unique_generated_bytes": len(set(selected)),
        "adjacent_repeat_fraction": sum(a == b for a, b in zip(selected, selected[1:])) / 63,
        "entropy_first_last": [rows[0]["entropy_nats"], rows[-1]["entropy_nats"]],
        "entropy_step_deltas": [
            b["entropy_nats"] - a["entropy_nats"] for a, b in zip(rows, rows[1:])
        ],
        "space_stays_top1_after_generated_space": (
            sum(a["top1_byte"] == b["top1_byte"] == 32 for a, b in zip(rows, rows[1:]))
            / max(1, sum(r["top1_byte"] == 32 for r in rows[:-1]))
        ),
        "distribution": probability_summary(rows),
    }


def build(recommendation, rationale):
    binding = read(STUDY / "binding.json")
    store = Store(STUDY, binding)
    require((store.get("integrity_final") or {}).get("status") == "PASS", "study not complete")
    verify_inventory(binding["parent_before"])
    segmented = {str(n): store.get(f"segmented_{n}") for n in LENGTHS}
    require(all(segmented.values()), "missing segmented measurement")
    rows = [store.get(f"anchor_{i:04d}") for i in range(len(binding["anchors"]))]
    require(all(rows), "missing anchor measurement")
    domains = sorted({r["anchor"]["domain"] for r in rows})
    paired = paired_summary(rows)
    domain_paired = {
        d: paired_summary([r for r in rows if r["anchor"]["domain"] == d]) for d in domains
    }
    distributions = {}
    for n in LENGTHS:
        key = str(n)
        current = [r["scores"][key]["distribution"] for r in rows]
        after = [r["scores"][key]["after_space"] for r in rows]
        space_first = sum(r["space_top1"] for r in current)
        distributions[key] = {
            "all": probability_summary(current),
            "domains": {
                d: probability_summary(
                    [r["scores"][key]["distribution"] for r in rows if r["anchor"]["domain"] == d]
                )
                for d in domains
            },
            "after_forced_space": probability_summary(after),
            "space_top1_retention_after_forced_space": (
                sum(
                    a["space_top1"] and b["space_top1"] for a, b in zip(current, after, strict=True)
                )
                / space_first
                if space_first
                else None
            ),
            "mean_entropy_change_after_forced_space": statistics.mean(
                b["entropy_nats"] - a["entropy_nats"] for a, b in zip(current, after, strict=True)
            ),
        }
    position_root = STUDY / "position-distributions"
    position_binding = read(position_root / "binding.json")
    position_store = Store(position_root, position_binding)
    require(
        (position_store.get("integrity_final") or {}).get("status") == "PASS",
        "position survey incomplete",
    )
    position_rows = [
        position_store.get(f"context_{i:04d}") for i in range(len(position_binding["selections"]))
    ]
    require(all(position_rows), "missing position context")
    position_distributions = {}
    for n in LENGTHS:
        selected = [v for v in position_rows if v["selection"]["length"] == n]
        position_distributions[str(n)] = {
            "all": probability_summary([v["distribution"] for v in selected]),
            "after_space": probability_summary([v["after_space"] for v in selected]),
            "domains": {
                d: probability_summary(
                    [v["distribution"] for v in selected if v["selection"]["anchor"]["domain"] == d]
                )
                for d in domains
            },
            "domain_buckets": {
                d: {
                    str(b): probability_summary(
                        [
                            v["distribution"]
                            for v in selected
                            if v["selection"]["anchor"]["domain"] == d
                            and v["selection"]["position"] // 32 == b
                        ]
                    )
                    for b in range(n // 32)
                }
                for d in domains
            },
            "buckets": {
                str(b): probability_summary(
                    [v["distribution"] for v in selected if v["selection"]["position"] // 32 == b]
                )
                for b in range(n // 32)
            },
        }
    generations = []
    for index in range(6):
        row = store.get(f"generation_{index}")
        require(row is not None, "missing generation")
        generations.append(dict(row, whitespace=whitespace_summary(row["steps"])))
    mechanics = {str(n): store.get(f"mechanics_{n}") for n in LENGTHS}
    require(all(mechanics.values()), "missing mechanical profile")
    carry_rows = [r for d in domains for r in rows if r["anchor"]["domain"] == d]
    carry_rows = [
        r for d in domains for r in [v for v in carry_rows if v["anchor"]["domain"] == d][:8]
    ]
    carry = paired_summary(carry_rows)["256"]
    ratios = {
        f"{b}/{a}": {
            k: mechanics[str(b)][k] / mechanics[str(a)][k]
            for k in ("forward_seconds", "backward_seconds", "total_seconds")
        }
        for a, b in zip(LENGTHS, LENGTHS[1:])
    }
    result = {
        "schema": "context-distribution-review-1",
        "validation": validation_closeout(),
        "probability_precision": (
            "FP32 logits; FP64 probability diagnostics; original FP32 objective. "
            "Space top2 means rank exactly 2."
        ),
        "status": "READY_FOR_HUMAN_DECISION",
        "context_scope": {
            "trained_window_bytes": 32,
            "mechanically_measured_lengths": list(LENGTHS),
            "maximum_scored_conditioning_bytes": 256,
            "effective_context_certification": "UNVERIFIED",
            "retention_benchmarks": "NOT_RUN",
        },
        "interpretation": {
            "A_aggregate_reset_sensitivity": "Segmented NLL range is below 2e-8; effectively "
            "indistinguishable under the fixed protocol. This alone says nothing universal "
            "about useful history.",
            "B_matched_history_benefit": "All 512 paired NLL deltas are exactly 0 at 64/128/256; "
            "95% paired bootstrap intervals are [0,0], including every domain. "
            "The current checkpoint obtains no measurable extra-history benefit for these "
            "boundary targets. The boundary-only result motivates the separate "
            "phase/logit/state supplement; "
            "no universal context claim follows from the original NLL equality.",
            "C_late_position_behavior": "No late bucket degrades against its own first bucket; "
            "differences remain below 0.8%. Buckets contain different targets, so lower "
            "late NLL is not evidence of causal context use.",
            "D_distribution_quality": "Boundary contexts have broad argmax-space dominance. "
            "Greedy whitespace primarily has moderate space concentration: probabilities "
            "and entropy cycle every 8-byte patch instead of tending to a point mass. "
            "Position-stratified results are descriptive and reported separately.",
            "E_compute_cost": "All four forward/backward probes pass with 56 finite gradient "
            "tensors and zero updates. Longest probe peaks below 100 MB allocated. "
            "The first 32-byte cold probe is slower than 64; single-shot timings are not "
            "an asymptotic fit or a sustained training-speed estimate.",
        },
        "binding": binding,
        "study_binding_sha256": sha(STUDY / "binding.json"),
        "measurement_hashes": {
            p.name: sha(p) for p in sorted(STUDY.glob("*.json")) if p.name != "progress.json"
        },
        "segmented": segmented,
        "matched_targets": paired,
        "matched_domains": domain_paired,
        "anchors_per_domain": dict(Counter(r["anchor"]["domain"] for r in rows)),
        "anchor_document_position_buckets": dict(
            Counter(str((r["anchor"]["offset"] % 256) // 32 * 32) for r in rows)
        ),
        "uncertainty_limit": (
            "Anchor bootstrap, not independent document bootstrap; clustered "
            "anchors and equal-domain sampling limit population inference."
        ),
        "state_carry": {
            "status": "MEASURED_BY_REUSING_MATCHED_INCREMENTAL_SCORES",
            "protocol": (
                "First eight hash-selected anchors/domain. Fresh BOS at offset-32 "
                "versus BOS at offset-256 carried through the same last 32 bytes. "
                "Both predict identical next target. No unbounded document "
                "streaming."
            ),
            "paired_256_minus_32": carry,
        },
        "distributions": distributions,
        "position_distributions": position_distributions,
        "position_distribution_summary": probability_summary(
            [v["distribution"] for v in position_rows]
        ),
        "whitespace_diagnosis": "MIXED: broad-distribution argmax attractor at boundaries, "
        "mostly moderate space concentration during greedy loops; rare severe individual "
        "contexts do not establish global probability collapse.",
        "position_binding": position_binding,
        "position_measurement_hashes": {
            p.name: sha(p)
            for p in sorted(position_root.glob("*.json"))
            if p.name != "progress.json"
        },
        "generations": generations,
        "mechanics": mechanics,
        "observed_cost_ratios": ratios,
        "optional_sampling": (
            "NOT_RUN: historical greedy policy preserved; supplemental "
            "sampling not needed for probability measurements"
        ),
        "optional_stream": (
            "NOT_RUN: beyond 256 mechanical streaming would not establish "
            "effective context and would delay primary study"
        ),
        "integrity": store.get("integrity_final"),
        "recommendation": recommendation,
        "rationale": rationale,
        "plan_a": {
            "parent_step": 5000,
            "total_steps": 78167,
            "remaining_updates": 73167,
            "policy": (
                "Retain exact 32-byte config, "
                "optimizer/scheduler/cursor/seed/corpus. No changes authorized by "
                "this study."
            ),
            "historical_training_target_bytes_per_second": 173.757238,
            "training_only_hours_at_stage_a_mean": 73167 * (639613 / 5000) / 173.757238 / 3600,
            "estimate_limit": (
                "Historical production-update throughput; short fresh-clone "
                "mechanics are not a sustained accumulated optimizer benchmark. "
                "Validation/downtime excluded."
            ),
            "proposed_review_cadence": (
                "Full validation and the same six greedy probes every 2500 updates "
                "and final endpoint, checkpoints every 1000 updates; human must "
                "authorize continuation/cadence."
            ),
            "gen0_freeze": (
                "Endpoint checksum/config/corpus/cursor, exact restoration, "
                "validation/distribution/generation evidence, independent review "
                "and immutable lineage; TEST evaluation needs separate "
                "authorization."
            ),
        },
        "plan_b": {
            "candidate_curriculum_for_review": "64 then128 bytes, with256 considered only after "
            "separate phase review; no immediate curriculum change is supported by current "
            "matched-target quality evidence.",
            "compute_basis": "All lengths mechanically pass; measured batch2 forward/backward "
            f"totals at64/128/256 are {mechanics['64']['total_seconds']:.3f}/"
            f"{mechanics['128']['total_seconds']:.3f}/{mechanics['256']['total_seconds']:.3f}s, "
            "without optimizer update. "
            "These single cold probes are not sustained production estimates.",
            "status": "DESIGN_REVIEW_ONLY_NOT_IMPLEMENTED",
            "review_items": [
                (
                    "Select length or staged lengths using matched effects, domain "
                    "regressions and costs; no automatic longest-length preference."
                ),
                (
                    "Define document-window rebuilding, reset and cursor semantics "
                    "without reinterpreting the old cursor."
                ),
                (
                    "Explicitly decide retained AdamW state versus new phase, and "
                    "scheduler continuation/reset."
                ),
                (
                    "Record repeated/skipped byte exposure and new phase budget; "
                    "retain original production counters."
                ),
                (
                    "Version checkpoint binding/schema and validation protocol; "
                    "retain immutable step 5000 rollback."
                ),
                (
                    "Evaluate both accepted32-byte baseline and new protocols for "
                    "comparability; review measured compute impact."
                ),
            ],
        },
        "limitations": [
            "Phase-only batch-8 supplement records inference wall time but no dedicated "
            "allocator/RSS series; memory tables describe the original segmented/mechanics runs.",
            "Segmented seconds sum measured block wall durations and exclude publication/setup "
            "and interruption downtime; throughput uses those measured durations, not total "
            "end-to-end elapsed time.",
            "32-byte timing overlaps focused CPU regression work and includes instrumentation; "
            "do not compare it as pure backend latency with historical validation. "
            "Disposable mechanics run after the other work.",
            (
                "Segment length changes BOS/reset count and historical exposure: "
                "not a same-target experiment."
            ),
            (
                "Matched histories share targets but include different recurrent "
                "trajectories; better NLL does not prove causal long-range "
                "retention."
            ),
            (
                "Stage A trained only 32-byte windows; 256 evaluated is not 256 "
                "effective context certification."
            ),
            (
                "Generic SSD reference retains quadratic transition storage; no "
                "optimized kernels or extrapolated effective context."
            ),
            (
                "Peak allocator memory measured; free VRAM and RSS are sampled, "
                "not continuous minima/maxima."
            ),
            (
                "Mechanics are one cold disposable forward/backward per length, "
                "no optimizer update; ratios include warmup/cache effects."
            ),
            (
                "TEST sealed except streaming hashes; no production training, "
                "curriculum, weights, checkpoints or source-model changes."
            ),
        ],
        "anomalies": [
            "Original process ended after 304 anchors. All 117 segment blocks, four summaries "
            "and 304 anchors verified and reused. Published tooling commit advanced HEAD only; "
            "a checksummed recovery reconciliation preserves the original producer/binding.",
            "Removed prewritten finalizer test claims; actual final command outputs, durations "
            "and tested-file hashes are required from validation-closeout/commands.json.",
            "Before distribution measurement, review identified exact-L matched targets are patch "
            "boundaries. Added separately bound position-stratified survey reusing frozen anchors; "
            "no main measurement/protocol rewritten or restarted.",
            (
                "Initial preflight rejected new verifier's use of config JSON "
                "canonicalization for corpus. Corrected to existing UTF8 corpus "
                "canonicalizer before evaluation; regression added."
            ),
            (
                "Optional unused PyTorch NumPy bridge warning persists; no "
                "environment modifications."
            ),
        ],
        "recovery_reconciliation": read(STUDY / "recovery-v1/reconciliation.json"),
        "stage_a_report_identity": binding["parent_before"][
            "reports/full_training_2m_stage_a.json"
        ],
    }
    result["phase_supplement"] = phase_summary()
    result["future_ab"] = {
        "status": "PROPOSAL_ONLY_NOT_AUTHORIZED_NOT_RUN",
        "parent": "immutable step 5000; separate disposable clones, never production resume",
        "arms": "A:32 bytes; B:64 bytes (128 only if separately chosen before experiment)",
        "budget": "Proposed cap: 65,536 valid target bytes per arm and 30 minutes per arm; "
        "if either cap prevents matched exposure, report the mismatch and do not promote. "
        "Predeclare equal valid target-byte exposure and a common compute ceiling; "
        "report actual compute, do not imply equal bytes guarantee equal wall time. "
        "For a compute-matched comparison, predeclare a separate common wall-time endpoint.",
        "controls": "Same corpus/document eligibility, seed, optimizer-state initialization, "
        "learning-rate policy and target order. Explicitly bind changed window/cursor semantics.",
        "evaluation": "Identical accepted 32-byte validation, segmented and phase-matched "
        "targets, distribution probes, learning gain per target-byte and second; TEST sealed.",
        "promotion": "Discard both unless separately reviewed and explicitly promoted; "
        "no A/B training or production continuation authorized by this report.",
    }
    result["anomalies"].append(
        "Phase runner preflight hit a list-conversion TypeError before saving any result. "
        "Empty failed binding preserved in phase-matched-preflight-001; corrected producer "
        "bound separately before measurement. No completed measurement recomputed."
    )
    return result


def markdown(r):
    lines = [
        "# 2M context and byte-distribution study",
        "",
        ("**Evidence only. No production training or curriculum change. TEST remains sealed.**"),
        "",
        f"Recommendation: **{r['recommendation']}**. {r['rationale']}",
        "",
        "## Immutable parent and protocol",
        "",
        (
            "Parent: "
            "`data/full-training-2m-v1/attempt-003/checkpoints/step_005000`, "
            "update 5000, 1,929,579 trainable parameters."
        ),
        f"Parameter SHA-256: `{r['binding']['parameter_sha256']}`.",
        (
            f"Source commit: `{r['binding']['source']['commit']}`. AC-008 "
            f"SHA-256: `{r['binding']['spec_sha256']}`."
        ),
        (
            "Parent-before/after inventory, exact config/runtime/tool hashes "
            "and split identities are in the JSON report. Production parent, "
            "FINAL specifications and Stage-A reports remain byte-identical."
        ),
        (
            "Raw-byte NLL uses original 267-symbol logits, with BOS/PAD masked "
            "by the model, not conditional-q loss. Manifest-ordered batch 2 "
            "windows reset at BOS and never cross a document boundary. "
            "All 500,000 validation targets included at every length."
        ),
        (
            "32-byte reproduction tolerance was predeclared as absolute 1e-7 "
            "versus 2.3624953916. BOS is separate from completed payload "
            "patches."
        ),
        "",
        "## Segmented validation: context/reset sensitivity",
        "",
        (
            "Changing segmentation changes reset frequency and the prior "
            "history of each target. This is not the controlled same-target "
            "comparison below."
        ),
        "",
        (
            "| Bytes / patches | Windows | Targets | NLL | Bits/byte | "
            "Seconds | Targets/s | Peak allocated/reserved MiB |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n, v in r["segmented"].items():
        m = v["memory"]
        lines.append(
            (
                f"| {n} / {int(n) // 8} | {v['windows']} | "
                f"{v['valid_target_count']} | {v['nll']:.8f} | "
                f"{v['bits_per_byte']:.6f} | {v['seconds']:.2f} | "
                f"{v['target_bytes_per_second']:.1f} | "
                f"{m['peak_allocated_bytes'] / 2**20:.2f}/{m['peak_reserved_bytes'] / 2**20:.2f} |"
            )
        )
    lines += [
        "",
        "Published tooling/resume commit: "
        f"`{r['recovery_reconciliation']['resume_source']['commit']}`. "
        "Original launch binding remains immutable; producer hashes match.",
    ]
    memories = [v["memory"] for v in r["segmented"].values()]
    free_mib = min(v["minimum_sampled_free_bytes"] for v in memories) / 2**20
    lines += ["", f"Minimum sampled free VRAM: {free_mib:.2f} MiB."]
    if all(v["process_rss_status"] == "MEASURED" for v in memories):
        rss_mib = max(v["peak_observed_process_rss_bytes"] for v in memories) / 2**20
        lines.append(
            f"Peak sampled Windows process working set during validation: {rss_mib:.2f} MiB."
        )
    else:
        lines.append("PROCESS_RSS: UNVERIFIED for at least one phase.")
    lines += [
        "",
        "## Matched next-byte targets",
        "",
        (
            "128 lowest-SHA256 anchors per available domain (seed 1729), each "
            "with at least 256 prior document bytes; same target "
            "at 32/64/128/256 history. No result-dependent resampling. Paired "
            "percentile bootstrap: 2000 draws, seed 1730,95% interval; "
            "unchanged tolerance 1e-5 NLL."
        ),
        r["uncertainty_limit"],
        "",
        ("| History | Mean NLL | Median | Delta vs32 [95% CI] | Improved / worsened / unchanged |"),
        "|---|---:|---:|---|---|",
    ]
    for n, v in r["matched_targets"].items():
        lines.append(
            (
                f"| {n} | {v['mean_nll']:.6f} | {v['median_nll']:.6f} | "
                f"{v['paired_delta']:.6f} {v['paired_delta_ci95']} | "
                f"{v['improved_fraction']:.3f}/{v['worsened_fraction']:.3f}/"
                f"{v['unchanged_fraction']:.3f} |"
            )
        )
    lines += [
        "",
        "## Position and domain behavior",
        "",
        (
            "Buckets are absolute target positions within each reset window. "
            "Relative deltas use that length's 0–31 bucket; target composition "
            "differs between buckets."
        ),
        "",
        "| Length | Bucket | Targets | NLL | Bits/byte | Relative delta |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for n, v in r["segmented"].items():
        for bucket, row in sorted(v["buckets"].items(), key=lambda x: int(x[0])):
            lines.append(
                (
                    f"| {n} | {bucket}–{int(bucket) + 31} | {row['count']} | "
                    f"{row['nll']:.6f} | {row['bits_per_byte']:.6f} | "
                    f"{row['relative_delta_from_first']:.2%} |"
                )
            )
    lines += [
        "",
        "| Domain | Length | Segmented NLL | Matched delta vs32 |",
        "|---|---:|---:|---:|",
    ]
    for d, values in r["matched_domains"].items():
        for n in map(str, LENGTHS):
            lines.append(
                (
                    f"| {d} | {n} | {r['segmented'][n]['domains'][d]['nll']:.6f} | "
                    f"{values[n]['paired_delta']:.6f} |"
                )
            )
    lines += [
        "",
        "## State carry diagnostic",
        "",
        r["state_carry"]["protocol"],
        json.dumps(r["state_carry"]["paired_256_minus_32"]),
        "",
        "## Original p and conditional-byte q",
        "",
        (
            "p is softmax over original 267-symbol logits. Byte mass is sum "
            "p(0..255); q(b)=p(b)/byte_mass. Raw-byte NLL remains based on p. "
            "Reported entropy/top-two/margins/concentration below use q. JSON "
            "preserves p-space, byte/control mass and every control "
            "probability for historical probes. Masked IDs are PAD 256 and "
            "BOS 257; EOS 258 remains in control support."
        ),
        (
            "Project-specific predeclared labels: severe if max(q)>=.9 and "
            "entropy<=1nat; otherwise moderate if q(space)>=.5; otherwise "
            "broad space dominance if space is argmax and entropy>1nat; mixed "
            "otherwise. These are not universal collapse thresholds."
        ),
        "",
        (
            "| History | q entropy nats | p(space) | q(space) | Space top1 "
            "fraction | q margin | Control mass | >=.9 concentration fraction "
            "|"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n, v in r["distributions"].items():
        q = v["all"]
        lines.append(
            (
                f"| {n} | {q['entropy_nats']:.6f} | {q['space_p']:.6f} | "
                f"{q['space_q']:.6f} | {q['space_top1']:.3f} | {q['margin_q']:.6f} "
                f"| {q['control_mass_p']:.8f} | "
                f"{q['concentration_fractions']['0.9']:.3f} |"
            )
        )
    lines += [
        "",
        "### Position-stratified validation survey",
        "",
        "Separate fixed sample: up to eight frozen anchor hashes per domain / length / "
        "32-byte position bucket. Context resets at the containing window boundary. "
        "This covers interior positions; the matched comparison above always predicts at "
        "a patch boundary. Neither sample follows the natural corpus domain weights.",
        "",
        "| Window | Contexts | q entropy | p(space) | q(space) | Space top1 | Control mass |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for n, v in r["position_distributions"].items():
        q = v["all"]
        lines.append(
            f"| {n} | {q['count']} | {q['entropy_nats']:.6f} | {q['space_p']:.6f} | "
            f"{q['space_q']:.6f} | {q['space_top1']:.3f} | {q['control_mass_p']:.8f} |"
        )
    lines += [
        "",
        "Joint length/domain/32-byte-bucket statistics, top-1 concentration and "
        "control-vs-byte mass are preserved in JSON; this survey is descriptive.",
        "Position-survey classes: "
        + json.dumps(r["position_distribution_summary"]["class_counts"]),
        r["whitespace_diagnosis"],
        "",
        "## Historical generation and whitespace runs",
        "",
        (
            "The six historical 64-byte greedy outputs reproduce exactly. "
            "Per-step byte/probability/top2/entropy/space/control "
            "measurements and forced-space successor statistics for held-out "
            "contexts are in JSON. No supplemental sampling replaced greedy "
            "results."
        ),
        "",
        ("| Prompt | Output | Longest space run | Mean q(space) | q entropy first→last |"),
        "|---|---|---:|---:|---|",
    ]
    for row in r["generations"]:
        w = row["whitespace"]
        lines.append(
            (
                f"| `{bytes.fromhex(row['prefix_hex'])!r}` | `{row['escaped']}` | "
                f"{w['longest_space_run']} | {w['distribution']['space_q']:.6f} | "
                f"{w['entropy_first_last']} |"
            )
        )
    lines += [
        "",
        "## Disposable mechanics and quadratic reference cost",
        "",
        (
            "Each length uses a fresh independent copy, batch 2, actual "
            "forward/loss/backward and finite-gradient checks. No optimizer "
            "update, saved clone, logical production update or optimizer "
            "migration. One cold observation per length; ratios are measured "
            "costs, not asymptotic proof."
        ),
        "",
        (
            "| Bytes | Forward s | Backward s | Total s | Target bytes/s | "
            "Gradient norm | Peak allocated/reserved MiB |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for n, v in r["mechanics"].items():
        lines.append(
            (
                f"| {n} | {v['forward_seconds']:.4f} | {v['backward_seconds']:.4f} "
                f"| {v['total_seconds']:.4f} | {v['step_bytes_per_second']:.1f} | "
                f"{v['gradient_norm']:.4f} | "
                f"{v['memory']['peak_allocated_bytes'] / 2**20:.2f}/"
                f"{v['memory']['peak_reserved_bytes'] / 2**20:.2f} |"
            )
        )
    lines += [
        "",
        f"Observed forward/backward/total ratios: `{json.dumps(r['observed_cost_ratios'])}`.",
        (
            "Synchronized CUDA peak allocation/reservation, sampled free VRAM "
            "and Windows process working set are recorded per length in JSON. "
            "RSS is measured independently of tensor accounting. No "
            "extrapolation to 512/1024 or effective-context claim."
        ),
        "",
        "## Continuation choices — not authorization",
        "",
        "Plan A retains step 5000,32-byte windows,total_steps78167 and "
        "exact optimizer/scheduler/cursor. **73,167 logical updates remain.** "
        f"Historical production throughput projects "
        f"{r['plan_a']['training_only_hours_at_stage_a_mean']:.2f} training-only hours; "
        "excludes validation/downtime and is not a fresh sustained benchmark.",
        r["plan_a"]["proposed_review_cadence"],
        r["plan_a"]["gen0_freeze"],
        "",
        (
            "Plan B requires a separately reviewed versioned curriculum from "
            "the immutable step 5000 parent:"
        ),
        "",
    ]
    lines += [r["plan_b"]["candidate_curriculum_for_review"], r["plan_b"]["compute_basis"], ""]
    lines += ["- " + s for s in r["plan_b"]["review_items"]]
    lines += ["", "## Recovery, limitations and validation", ""]
    lines += ["- " + s for s in r["anomalies"] + r["limitations"]]
    lines += [
        (
            "- Completed block/anchor files are immutable and checksummed "
            "against binding; progress can be rebuilt. No result is "
            "overwritten on recovery."
        ),
        "- " + r["optional_sampling"],
        "- " + r["optional_stream"],
        "",
        "Final validation results below come from the checksummed command artifact.",
        "",
        (
            "No production training, model architecture, weights, checkpoint, "
            "corpus or environment changes. No TEST evaluation, 20M, RSI or "
            "effort controller. Next action requires human authorization."
        ),
        "",
        "2M CONTEXT-DISTRIBUTION STUDY STATUS: READY FOR HUMAN DECISION",
        "",
    ]
    interpretation_lines = ["## Five separate interpretation questions", ""]
    for name, answer in r["interpretation"].items():
        interpretation_lines += [f"**{name.replace('_', ' ')}:** {answer}", ""]
    lines[-2:-2] = interpretation_lines
    validation_lines = [
        "## Actual final checks",
        "",
        "| Command | Exit code | Wall seconds |",
        "|---|---:|---:|",
    ]
    test_summary = ""
    for row in r["validation"]["commands"]:
        validation_lines.append(f"| {row['name']} | {row['returncode']} | {row['seconds']:.3f} |")
        if row["name"] == "pytest":
            summary = [line for line in row["stdout"].splitlines() if " passed" in line]
            test_summary = "Pytest: " + " ".join(summary)
    validation_lines += ["", test_summary, ""]
    lines[-2:-2] = phase_markdown(r["phase_supplement"])
    lines[-2:-2] = (
        ["## Future disposable A/B proposal — not authorization", ""]
        + [f"- **{k}:** {v}" for k, v in r["future_ab"].items()]
        + [""]
    )
    lines[-2:-2] = validation_lines + [""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recommendation",
        required=True,
        choices=[
            "CONTINUE_32_BYTE_RUN",
            "REVIEW_LONGER_WINDOW_CURRICULUM",
            "MORE_EVIDENCE_REQUIRED",
        ],
    )
    parser.add_argument("--rationale", required=True)
    args = parser.parse_args()
    result = build(args.recommendation, args.rationale)
    path = ROOT / "reports/context_distribution_study_2m"
    path.with_suffix(".json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    path.with_suffix(".md").write_text(markdown(result), encoding="utf-8")
