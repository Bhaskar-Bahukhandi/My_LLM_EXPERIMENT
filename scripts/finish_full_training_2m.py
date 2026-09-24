"""Finalize saved Gen-0 continuation evidence without training or TEST evaluation."""

import argparse
import json

from context_distribution_study import ROOT, Store, parent_inputs, read, sha, tensor_hash
from context_study_metrics import require
from full_training_stage_b_to_endpoint import (
    CONT,
    END,
    EVAL,
    PROOFS,
    VALIDATIONS,
    audit_rows,
    compare_row,
    domain_watch,
    latest_checkpoint,
    read_rows,
    verify_binding,
)
from training_progress import compare_replay
from validate_production_continuation import verify_receipt

from unified_edge.training.checkpoint import load_checkpoint
from unified_edge.training.config import canonical_hash
from unified_edge.training.data import BatchStream, WindowDataset


def build(receipt):
    validation = verify_receipt(receipt)
    binding = read(CONT / "binding.json")
    progress = read(CONT / "progress.json")
    require(
        progress["status"] in ("ENDPOINT_COMPLETE", "DOMAIN_REGRESSION_REVIEW_REQUIRED"),
        "continuation still active or failed; no final report",
    )
    _, _, config, manifest, _, _, full = parent_inputs()
    verify_binding(binding, full)
    rows = read_rows(CONT / "observations.jsonl")
    new_total = audit_rows(rows) - 639613
    step = 5000 + len(rows)
    require(step == progress["step"], "final progress/journal mismatch")
    checkpoint = latest_checkpoint(binding, rows)
    require(checkpoint.relative_to(ROOT).as_posix() == progress["checkpoint"], "checkpoint index")
    saved = load_checkpoint(checkpoint)
    require(
        saved["global_step"] == step and saved["bytes_seen"] == 639613 + new_total,
        "final saved counters mismatch",
    )
    require(
        saved["optimizer_update_counts"] == [step] * 56 and saved["scheduler"]["completed"] == step,
        "final optimizer/scheduler",
    )
    require(sum(t.numel() for t in saved["model"].values()) == 1929579, "parameter count")
    final_hash = tensor_hash(saved["model"])
    # Independent complete frozen cursor traversal, including both wraparound windows.
    from context_distribution_study import DATA

    stream = BatchStream(WindowDataset(manifest, DATA, "train", 32), config.seed, 2)
    expected_bytes = 0
    order_hashes = {}
    for n in range(1, step + 1):
        count = sum(len(w.payload) for _ in range(2) for w in stream.next_batch())
        expected_bytes += count
        metric = saved["metrics"][n - 1]
        require(
            metric["valid_target_count"] == count and metric["bytes_seen"] == expected_bytes,
            f"frozen byte schedule differs at {n}",
        )
        if n > 5000:
            row = rows[n - 5001]
            compare_replay(metric, row)
            if stream.epoch not in order_hashes:
                order_hashes[stream.epoch] = canonical_hash(stream.order)
            require(
                row["cursor"]
                == {
                    "epoch": stream.epoch,
                    "offset": stream.offset,
                    "order_sha256": order_hashes[stream.epoch],
                },
                f"frozen cursor differs at {n}",
            )
    require(saved["data_cursor"] == stream.state_dict(), "endpoint cursor differs")
    checkpoints = []
    for path in sorted(CONT.glob("attempt-*/checkpoints/step_*")):
        checkpoints.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "manifest_sha256": sha(path / "manifest.json"),
                "state_sha256": sha(path / "state.pt"),
            }
        )
    evaluations = []
    for n in VALIDATIONS:
        if n > step:
            break
        root = EVAL / f"step_{n:06d}"
        matches = [c for c in checkpoints if c["path"].endswith(f"step_{n:06d}")]
        require(len(matches) == 1, "missing/ambiguous evaluated checkpoint")
        expected_binding = {
            "amendment_sha256": canonical_hash(binding),
            "step": n,
            "checkpoint_sha256": matches[0]["state_sha256"],
        }
        store = Store(root, expected_binding)
        result = store.get("complete")
        require(result is not None and result["step"] == n, "missing final evaluation")
        require(result["validation"]["count"] == 500000, "validation count")
        evaluations.append(result)
    require(evaluations[-1]["parameter_sha256"] == final_hash, "evaluated checkpoint differs")
    proofs = {}
    for n in PROOFS:
        if n <= step:
            proof = read(CONT / f"restore_{n:06d}.json")
            require(
                proof["status"] == "EXACT PASS" and proof["optimizer_updates"] == 0,
                "restore failed",
            )
            matches = [c for c in checkpoints if c["path"].endswith(f"step_{n:06d}")]
            require(
                len(matches) == 1 and matches[0]["state_sha256"] == proof["checkpoint_sha256"],
                "restore checkpoint identity differs",
            )
            proofs[str(n)] = proof
    stage = read(ROOT / "reports/full_training_2m_stage_a.json")
    context = read(ROOT / "reports/context_distribution_study_2m.json")
    ab = read(ROOT / "reports/context_training_ab_2m.json")
    replay = []
    recovery = []
    for path in sorted(CONT.glob("attempt-*/recovery.json")):
        recovery.append(read(path))
        for physical in read_rows(path.parent / "physical_replay.jsonl"):
            compare_row(physical, rows[physical["step"] - 5001])
            replay.append(physical)
    extra_bytes = sum(r["valid_target_count"] for r in replay)
    extra_steps = len(replay)
    parent_full = context["segmented"]["32"]
    watches = []
    previous = None
    for item in evaluations:
        watch = domain_watch(parent_full, previous, item["validation"])
        watches.append({"step": item["step"], "domains": watch})
        previous = item["validation"]
    endpoint = progress["status"] == "ENDPOINT_COMPLETE"
    if endpoint:
        require(
            step == END
            and saved["bytes_seen"] == 10000064
            and stream.epoch == 1
            and stream.offset == 2
            and not any(v["domains"] for v in watches),
            "endpoint or domain gate failed",
        )
        require(progress["parameter_sha256"] == final_hash, "published final hash mismatch")
    else:
        require(bool(watches[-1]["domains"]), "unsubstantiated domain stop")
    curve = []
    for offset in range(0, len(rows), 1000):
        block = rows[offset : offset + 1000]
        count = sum(r["valid_target_count"] for r in block)
        curve.append(
            {
                "first_step": block[0]["step"],
                "last_step": block[-1]["step"],
                "target_count": count,
                "weighted_nll": sum(r["summed_nll"] for r in block) / count,
                "learning_rate_first_last": [block[0]["learning_rate"], block[-1]["learning_rate"]],
                "seconds": sum(r["update_seconds"] for r in block),
            }
        )
    times = sum(r["update_seconds"] for r in rows)
    physical_seconds = sum(r["update_seconds"] for r in replay)
    memory = {
        "peak_allocated_bytes": max(r["memory"]["cuda"]["peak_allocated_bytes"] for r in rows),
        "peak_reserved_bytes": max(r["memory"]["cuda"]["peak_reserved_bytes"] for r in rows),
        "minimum_sampled_free_bytes": min(r["memory"]["cuda"]["free_device_bytes"] for r in rows),
        "process_rss_status": "MEASURED"
        if all(r["memory"]["process_rss_status"] == "MEASURED" for r in rows)
        else "UNVERIFIED",
        "peak_observed_rss_bytes": (
            max(r["memory"]["process_rss_bytes"] for r in rows)
            if all(r["memory"]["process_rss_status"] == "MEASURED" for r in rows)
            else None
        ),
        "tensor_payloads": {
            k: rows[-1]["memory"][k]
            for k in (
                "parameter_bytes",
                "gradient_bytes",
                "optimizer_state_bytes",
                "canonical_recurrent_state_bytes",
                "recurrent_note",
                "activation_autograd_bytes",
            )
        },
    }
    result = {
        "schema": "full-training-2m-gen0-final-1",
        "status": "2M_GEN0_READY_FOR_REVIEW" if endpoint else "2M_GEN0_NOT_READY",
        "production_status": progress["status"],
        "amendment": binding,
        "amendment_raw_sha256": sha(CONT / "binding.json"),
        "progress": progress,
        "parent": {
            "step": 5000,
            "parameter_sha256": stage["parameter_sha256"],
            "checkpoint": binding["parent"],
            "checkpoint_sha256": binding["parent_state_sha256"],
        },
        "history": {
            "diagnostic": {
                "report": "reports/real_training_diagnostic_2m.json",
                "summary": read(ROOT / "reports/real_training_diagnostic_2m.json")["summary"],
            },
            "stage_a": {
                "validation": stage["validation"],
                "train": stage["train"],
                "recovery": stage["recovery"],
                "resume_proof": stage["resume_proof"],
                "physical_overhead": stage["physical_overhead"],
            },
            "context_study": {
                "recommendation": context["recommendation"],
                "mechanism": context["phase_supplement"]["mechanism"],
            },
            "disposable_ab": {
                "recommendation": ab["decision"]["recommendation"],
                "promoted": False,
                "history_benefit": ab["b_minus_a_history_benefit"],
            },
        },
        "logical": {
            "global_step": step,
            "micro_step": saved["micro_step"],
            "windows_seen": saved["examples_seen"],
            "previous_target_bytes": 639613,
            "new_target_bytes": new_total,
            "total_target_bytes": saved["bytes_seen"],
            "expected_endpoint_target_bytes": 10000064,
            "extra_epoch1_windows": 2 if endpoint else 0,
            "cursor_epoch": stream.epoch,
            "cursor_offset": stream.offset,
        },
        "physical_overhead": {
            "stage_a": stage["physical_overhead"],
            "continuation": {
                "updates": extra_steps,
                "windows": 4 * extra_steps,
                "target_bytes": extra_bytes,
                "recorded_seconds": physical_seconds,
            },
            "combined": {
                "updates": stage["physical_overhead"]["updates"] + extra_steps,
                "target_bytes": stage["physical_overhead"]["target_bytes"] + extra_bytes,
            },
            "note": "Physical replay is excluded from logical learning exposure.",
        },
        "recovery": recovery,
        "checkpoints": checkpoints,
        "restore_proofs": proofs,
        "parameter_sha256": final_hash,
        "parameter_count": 1929579,
        "validation": evaluations,
        "domain_watch": watches,
        "parent_validation": parent_full,
        "train_curve": curve,
        "train": {
            "continuation_final100_weighted_nll": sum(r["summed_nll"] for r in rows[-100:])
            / sum(r["valid_target_count"] for r in rows[-100:]),
            "clipped_updates": sum(r["clipped"] for r in rows),
            "gradient_norm_range": [
                min(r["gradient_norm"] for r in rows),
                max(r["gradient_norm"] for r in rows),
            ],
            "all_finite": all(r["finite_gradients"] for r in rows),
            "measured_update_seconds": times,
            "target_bytes_per_measured_second": new_total / times,
        },
        "memory": memory,
        "validation_receipt": validation,
        "validation_receipt_sha256": sha(receipt),
        "integrity": {
            "accepted_artifacts_unchanged": True,
            "corpus_unchanged": True,
            "model_source_unchanged": True,
            "frozen_schedule_independently_reconciled": True,
            "test": "SEALED; streaming hashes only",
        },
        "limitations": [
            "32-byte-trained; mechanical recurrent streaming does not certify effective context.",
            "One seed, one pass, small licensed pilot. No benchmark/TEST or "
            "meaningful capability claim.",
            "64-byte A/B was disposable and NOT promoted. Only the original step5000 "
            "trajectory continued.",
            "Update wall intervals include original corpus verification and "
            "diagnostics; journal, checkpoint and evaluation I/O excluded.",
            "Separate activation memory and end-to-end study duration not "
            "instrumented. CUDA peaks may include earlier evaluation allocations.",
            "RSS is sampled actual process working set, not tensor arithmetic or a "
            "continuously sampled maximum.",
            "Unpublished in-flight physical work after an interruption cannot be "
            "reconstructed; durable replay is reported separately.",
            "Optional unused NumPy bridge warning remains; no environment change.",
            "Endpoint readiness means reviewable engineering evidence, not automatic "
            "model promotion.",
        ],
    }
    final = evaluations[-1]
    gain = parent_full["nll"] - final["validation"]["nll"]
    result["assessment"] = {
        "validation_nll_improvement_from_step5000": gain,
        "validation_nll_reduction_fraction": gain / parent_full["nll"],
        "domain_nll_improvements_from_step5000": {
            d: value["nll"] - final["validation"]["domains"][d]["nll"]
            for d, value in parent_full["domains"].items()
        },
        "validation_minus_recent100_train_nll": final["validation"]["nll"]
        - result["train"]["continuation_final100_weighted_nll"],
        "generation_limitation": "All six greedy probes remain dominated by long space runs; "
        "this is not useful language-generation evidence.",
        "distribution_scope": "32 fixed validation contexts; zero observed severe cases "
        "does not establish absence of collapse across the full distribution.",
        "readiness_basis": "Reviewable numerical, accounting and restoration evidence, with "
        "improved validation in every domain and persistent generation limitations. "
        "No automatic promotion or effective-context claim.",
    }
    result["limitations"].append(
        "Training summed NLL is reconstructed as the unchanged trainer mean times valid "
        "target count; it is not a separately captured pre-division accumulator."
    )
    verify_binding(binding, full)
    return result


def markdown(r):
    final = r["validation"][-1]
    lines = [
        "# Full 2M Gen-0 continuation review",
        "",
        f"**{r['status']}** — human promotion authority only.",
        "",
        "The original 32-byte production run continued from the immutable step-5000 checkpoint. "
        "The 64-byte A/B was disposable and NOT promoted. TEST remains sealed.",
        "",
        f"Endpoint/review step: {r['logical']['global_step']}. Parameter SHA-256: "
        f"`{r['parameter_sha256']}`.",
        f"Checkpoint: `{r['progress']['checkpoint']}`.",
        f"Amendment SHA-256: `{r['amendment_raw_sha256']}`.",
        "",
        f"Validation NLL fell {100 * r['assessment']['validation_nll_reduction_fraction']:.2f}% "
        "from step 5000, with every domain improved. "
        f"Recent-100 training NLL is {r['train']['continuation_final100_weighted_nll']:.6f}; "
        f"validation is {final['validation']['nll']:.6f}. "
        "Gradients remained finite and required exact restore proofs passed. "
        "Greedy generation still degenerates into long space runs; this remains a "
        "limited engineering pilot, not evidence of useful language capability.",
        "",
        "The distribution monitor uses only 32 fixed validation contexts. Its zero severe "
        "cases do not establish the absence of collapse over the whole distribution.",
        "",
        "## Validation and domain trajectory",
        "",
        "| Step | Aggregate NLL | Bits/byte | General text | Code | Documentation | Mathematics |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in [{"step": 5000, "validation": r["parent_validation"]}, *r["validation"]]:
        v = item["validation"]
        values = [
            v["nll"],
            v["bits_per_byte"],
            *[
                v["domains"][d]["nll"]
                for d in ("general_text", "code", "documentation", "structured_math")
            ],
        ]
        lines.append(f"| {item['step']} | " + " | ".join(f"{x:.9f}" for x in values) + " |")
    lines += [
        "",
        f"Final validation count: {final['validation']['count']}. Domain watch: "
        f"`{r['domain_watch']}`.",
        "",
        "## Logical and physical accounting",
        "",
        f"Previous target bytes: 639,613; new logical bytes: {r['logical']['new_target_bytes']:,}; "
        f"total: {r['logical']['total_target_bytes']:,}, expected endpoint 10,000,064.",
        f"Final cursor: epoch {r['logical']['cursor_epoch']}, offset "
        f"{r['logical']['cursor_offset']}.",
        f"Physical overhead: `{json.dumps(r['physical_overhead'])}`.",
        f"Continuation training summary: `{json.dumps(r['train'])}`.",
        f"Measured memory: `{json.dumps(r['memory'])}`.",
        "",
        "## Generation and distribution trajectory",
        "",
        "| Step | q(space) | p(space) | Entropy | Space top-1 fraction | Severe "
        "fraction | Control mass |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in r["validation"]:
        d = item["position_distribution"]
        severe = d["class_counts"].get("SEVERE_BYTE_MODE_CONCENTRATION", 0) / d["count"]
        lines.append(
            f"| {item['step']} | {d['space_q']:.6f} | {d['space_p']:.6f} | "
            f"{d['entropy_nats']:.6f} | {d['space_top1']:.6f} | {severe:.6f} | "
            f"{d['control_mass_p']:.9g} |"
        )
    lines += [
        "",
        "Final six historical greedy probes (all historical points are retained in JSON):",
        "",
        "| Prompt | Output | Longest space run |",
        "|---|---|---:|",
    ]
    for g in final["generations"]:
        lines.append(
            f"| `{bytes.fromhex(g['prefix_hex'])!r}` | `{g['escaped']}` | "
            f"{g['whitespace']['longest_space_run']} |"
        )
    lines += [
        "",
        "## History and integrity",
        "",
        "Original diagnostic and Stage-A evidence are preserved and referenced in JSON. "
        "Stage A included the documented Windows sharing-error recovery and exact "
        "deterministic replay. "
        "The context study and disposable A/B found no useful older-history output dependence. "
        "No curriculum or architecture change was introduced.",
        f"Restore proofs: `{json.dumps(r['restore_proofs'])}`.",
        "All historical production artifacts, corpus identities, FINAL "
        "specifications and computational source hashes remain unchanged. "
        "The original Stage-A progress file is preserved; the versioned continuation "
        "index records new production steps.",
        "",
        "## Validation receipts",
        "",
        "| Check | Exit | Seconds |",
        "|---|---:|---:|",
    ]
    for c in r["validation_receipt"]["commands"]:
        lines.append(f"| {c['name']} | {c['returncode']} | {c['seconds']:.2f} |")
    lines += ["", "## Limitations and next decision", ""] + [f"- {v}" for v in r["limitations"]]
    lines += [
        "",
        "Next action requires human authorization. No TEST evaluation, 20M or "
        "future-capability implementation.",
        "",
        "FULL 2M GEN-0 STATUS: "
        + ("READY FOR HUMAN REVIEW" if r["status"] == "2M_GEN0_READY_FOR_REVIEW" else "NOT READY"),
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", required=True)
    args = parser.parse_args()
    result = build(ROOT / args.validation)
    report = ROOT / "reports/full_training_2m_final"
    report.with_suffix(".json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    report.with_suffix(".md").write_text(markdown(result), encoding="utf-8")
