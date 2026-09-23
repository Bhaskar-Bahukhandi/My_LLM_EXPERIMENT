"""Separately bound, read-only phase-matched history and state diagnostics."""

import time
from collections import defaultdict
from pathlib import Path

import torch
from context_distribution_study import (
    PARAMETER_SHA,
    ROOT,
    STUDY,
    Store,
    parent_inputs,
    read,
    sha,
    tensor_hash,
    verify_corpus,
    verify_inventory,
)
from context_study_metrics import PROTOCOL, require

from unified_edge.dense_model import DenseByteModel
from unified_edge.training.config import canonical_hash
from unified_edge.training.device import configure_device
from unified_edge.training.trainer import code_identity, environment

BASES = (32, 64, 128, 248)


def selections(anchors):
    groups = defaultdict(list)
    for anchor in anchors:
        require(anchor["offset"] >= 255, "insufficient preceding document bytes")
        groups[anchor["domain"]].append(anchor)
    require(len(groups) == 4, "expected four domains")
    result = []
    for domain in sorted(groups):
        values = sorted(groups[domain], key=lambda a: a["key"])[:32]
        require(len(values) == 32, "insufficient anchors per domain")
        result.extend(values)
    require(len({a["key"] for a in result}) == 128, "duplicate anchors")
    return result


def delta(left, right):
    require(left.dtype == right.dtype == torch.float32, "forensics require unchanged FP32")
    require(bool(torch.isfinite(left).all() and torch.isfinite(right).all()), "nonfinite state")
    difference = right - left
    return {
        "max_abs": difference.abs().max().item(),
        "mean_abs": difference.abs().mean().item(),
        "l2": difference.norm().item(),
        "norm_short": left.norm().item(),
        "norm_long": right.norm().item(),
    }


def forensic(short, long, short_logits, long_logits, index, target):
    a, b = short_logits[index], long_logits[index]
    mask = torch.isfinite(a)
    require(torch.equal(mask, torch.isfinite(b)), "logit support changed")
    require(
        torch.equal(torch.nonzero(~mask).flatten(), torch.tensor([256, 257], device=a.device)),
        "unexpected masked symbols",
    )
    result = {
        "logits": dict(
            delta(a[mask], b[mask]),
            equal=torch.equal(a, b),
            excluded_masked_ids=[256, 257],
            target_logprob_delta=(b.log_softmax(-1)[target] - a.log_softmax(-1)[target]).item(),
            top1_short=a[:256].argmax().item(),
            top1_long=b[:256].argmax().item(),
            q_total_variation=((a[:256].softmax(-1) - b[:256].softmax(-1)).abs().sum() / 2).item(),
        ),
        "layers": [],
    }
    for left, right in zip(short.shared.layers, long.shared.layers, strict=True):
        result["layers"].append(
            {
                "conv": delta(left.conv[index], right.conv[index]),
                "ssm": delta(left.ssm[index], right.ssm[index]),
            }
        )
    a, b = short.hierarchy, long.hierarchy
    result["hierarchy"] = {
        "pending_equal": torch.equal(a.pending[index], b.pending[index]),
        "pending_length_short": a.pending.shape[1],
        "pending_length_long": b.pending.shape[1],
        "hidden": delta(a.hidden[index], b.hidden[index]),
    }
    return result


@torch.inference_mode()
def consume_batch(model, histories, device):
    data = torch.tensor([list(h) for h in histories], dtype=torch.long, device=device)
    state = model.start(len(histories))
    for column in data.unbind(1):
        state = model.consume(column, state)
    return state


def run():
    base = read(STUDY / "binding.json")
    for name, expected in base["tools"].items():
        require(sha(ROOT / "scripts" / name) == expected, "original producer changed")
    for root in (STUDY, STUDY / "position-distributions"):
        require(
            Store(root, read(root / "binding.json")).get("integrity_final")["status"] == "PASS",
            "previous study incomplete",
        )
    saved, resolved, config, _, payloads, before, full = parent_inputs()
    require(before == base["parent_before"], "parent changed")
    configure_device("cuda:0")
    torch.set_num_threads(config.cpu_threads)
    torch.use_deterministic_algorithms(True)
    require(environment(config) == base["environment"], "environment changed")
    anchors = selections(base["anchors"])
    binding = {
        "schema": "phase-matched-1",
        "main_binding_sha256": sha(STUDY / "binding.json"),
        "parent_before": before,
        "parameter_sha256": PARAMETER_SHA,
        "code": code_identity(),
        "tool_sha256": sha(Path(__file__)),
        "original_tools": base["tools"],
        "ac008_sha256": sha(
            ROOT / "docs/architecture_changes/AC-008_Context_and_Distribution_Promotion_Gates.md"
        ),
        "anchors": anchors,
        "selection_sha256": canonical_hash(anchors),
        "protocol": {
            "selection": "first 32 existing frozen hash keys per domain; first 8 forensic",
            "phase_grid": {str(r): [n + r for n in BASES] for r in range(8)},
            "batch_size": 8,
            "unchanged_nll_tolerance": PROTOCOL["unchanged_nll_tolerance"],
            "bootstrap": (
                "2000 paired percentile draws seed1730; "
                "aggregate resamples anchors after phase averaging"
            ),
            "precision": "model and forensic tensor arithmetic FP32",
            "recency_ablation": (
                "NOT_RUN: additional sequential inference cost; "
                "required phase/state probes prioritized"
            ),
            "architecture_sanity": (
                "seed43 untrained model; old 8 zeros versus 8 bytes255; "
                "common suffix bytes0..31; no weight tuning"
            ),
        },
    }
    store = Store(STUDY / "phase-matched", binding)
    model = DenseByteModel(resolved).to("cuda:0")
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    del saved
    with torch.inference_mode():
        for batch in range(16):
            chosen = anchors[batch * 8 : (batch + 1) * 8]
            targets = torch.tensor([a["target"] for a in chosen], device="cuda:0")
            for phase in range(8):
                name = f"batch_{batch:02d}_phase_{phase}"
                if store.get(name) is not None:
                    continue
                started = time.perf_counter()
                rows = [{"anchor": a, "phase": phase, "scores": {}} for a in chosen]
                for length in BASES:
                    histories = []
                    for a in chosen:
                        raw = payloads[a["path"]]
                        require(raw[a["offset"]] == a["target"], "anchor target changed")
                        histories.append(raw[a["offset"] - length - phase : a["offset"]])
                    state = consume_batch(model, histories, "cuda:0")
                    logits = model.predict(state)
                    losses = (
                        (-logits.log_softmax(-1).gather(1, targets[:, None]))
                        .flatten()
                        .cpu()
                        .tolist()
                    )
                    for row, loss in zip(rows, losses, strict=True):
                        row["scores"][str(length + phase)] = loss
                    if length == 32:
                        short, short_logits = state, logits
                    if length == 248 and batch % 4 == 0:
                        for j, row in enumerate(rows):
                            row["forensic"] = forensic(
                                short, state, short_logits, logits, j, row["anchor"]["target"]
                            )
                store.put(name, {"rows": rows, "seconds": time.perf_counter() - started})
            print(f"MILESTONE: phase supplement {batch + 1}/16 batches complete", flush=True)
        if store.get("architecture_sanity") is None:
            torch.manual_seed(43)
            fixture = DenseByteModel(resolved).eval()
            a = consume_batch(fixture, [bytes(8) + bytes(range(32))], "cpu")
            b = consume_batch(fixture, [bytes([255]) * 8 + bytes(range(32))], "cpu")
            evidence = forensic(a, b, fixture.predict(a), fixture.predict(b), 0, 0)
            require(
                evidence["logits"]["max_abs"] > 0
                and any(x["ssm"]["max_abs"] > 0 for x in evidence["layers"]),
                "untrained architecture sensitivity not demonstrated",
            )
            store.put(
                "architecture_sanity",
                {"status": "PASS", "seed": 43, "common_suffix_bytes": 32, "forensic": evidence},
            )
        require(tensor_hash(model.state_dict()) == PARAMETER_SHA, "parent model changed")
        require(verify_corpus(full) == base["split_hashes"], "corpus changed")
        verify_inventory(before)
        if store.get("integrity_final") is None:
            store.put(
                "integrity_final",
                {"status": "PASS", "parent_after": before, "test": "SEALED; hash-only integrity"},
            )
    print("MILESTONE: phase supplement integrity PASS", flush=True)


if __name__ == "__main__":
    run()
