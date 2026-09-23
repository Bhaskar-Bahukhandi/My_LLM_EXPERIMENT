"""Frozen validation-only evaluation panels for disposable A/B checkpoints."""

import math
import time

import torch
import torch.nn.functional as F
from context_distribution_study import consume, memory, memory_summary
from context_study_metrics import PROTOCOL, add_losses, distribution, require, summarize_losses
from context_study_phases import BASES, consume_batch, forensic


@torch.inference_mode()
def segmented_panel(model, documents, length, store, prefix):
    name = f"{prefix}_{length}"
    if (cached := store.get(name)) is not None:
        return cached
    windows = [
        (d.path, d.domain, raw[start : start + length])
        for d, raw in documents
        for start in range(0, len(raw), length)
    ]
    blocks = []
    for offset in range(0, len(windows), 256):
        key = f"{name}_block_{offset:06d}"
        block = store.get(key)
        if block is None:
            torch.cuda.reset_peak_memory_stats()
            samples = [memory()]
            started = time.perf_counter()
            total, count, domains, buckets = 0.0, 0, {}, {}
            for start in range(offset, min(offset + 256, len(windows)), 2):
                groups = {}
                for path, domain, raw in windows[start : start + 2]:
                    groups.setdefault(len(raw), []).append((domain, raw))
                for group in groups.values():
                    targets = torch.tensor([list(raw) for _, raw in group], device="cuda:0")
                    logits = model(targets)
                    summed = F.cross_entropy(
                        logits.flatten(0, 1), targets.flatten(), reduction="sum"
                    )
                    require(bool(torch.isfinite(summed)), "nonfinite evaluation loss")
                    total += summed.item()
                    count += targets.numel()
                    losses = (
                        F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="none")
                        .reshape(targets.shape)
                        .cpu()
                        .tolist()
                    )
                    for (domain, _), row in zip(group, losses, strict=True):
                        add_losses(domains, domain, row)
                        for position in range(0, len(row), 32):
                            add_losses(buckets, str(position), row[position : position + 32])
            samples.append(memory())
            block = {
                "sum_nll": total,
                "count": count,
                "domains": domains,
                "buckets": buckets,
                "seconds": time.perf_counter() - started,
                "memory": memory_summary(samples),
            }
            store.put(key, block)
        blocks.append(block)
    total = sum(b["sum_nll"] for b in blocks)
    count = sum(b["count"] for b in blocks)
    domains, buckets = {}, {}
    for block in blocks:
        for dest, field in ((domains, "domains"), (buckets, "buckets")):
            for key, (value, n) in block[field].items():
                previous = dest.get(key, [0.0, 0])
                dest[key] = [previous[0] + value, previous[1] + n]
    result = {
        "nll": total / count,
        "sum_nll": total,
        "count": count,
        "bits_per_byte": total / count / math.log(2),
        "domains": summarize_losses(domains),
        "buckets": summarize_losses(buckets),
        "seconds": sum(b["seconds"] for b in blocks),
        "block_memory": [b["memory"] for b in blocks],
    }
    store.put(name, result)
    return result


@torch.inference_mode()
def phase_panel(model, anchors, payloads, store, with_forensics):
    for batch in range(16):
        selected = anchors[batch * 8 : (batch + 1) * 8]
        targets = torch.tensor([a["target"] for a in selected], device="cuda:0")
        for phase in range(8):
            name = f"phase_{batch:02d}_{phase}"
            if store.get(name) is not None:
                continue
            started = time.perf_counter()
            rows = [{"anchor": a, "phase": phase, "scores": {}} for a in selected]
            for base in BASES:
                histories = []
                for a in selected:
                    raw = payloads[a["path"]]
                    require(raw[a["offset"]] == a["target"], "evaluation anchor identity changed")
                    histories.append(raw[a["offset"] - base - phase : a["offset"]])
                state = consume_batch(model, histories, "cuda:0")
                logits = model.predict(state)
                losses = (
                    (-logits.log_softmax(-1).gather(1, targets[:, None])).flatten().cpu().tolist()
                )
                for row, loss in zip(rows, losses, strict=True):
                    row["scores"][str(base + phase)] = loss
                if base == 32:
                    short, short_logits = state, logits
                if base == 248 and with_forensics and batch % 4 == 0:
                    for j, row in enumerate(rows):
                        row["forensic"] = forensic(
                            short, state, short_logits, logits, j, row["anchor"]["target"]
                        )
            store.put(name, {"rows": rows, "seconds": time.perf_counter() - started})
        if batch % 4 == 3:
            print(
                f"MILESTONE: {store.root.parent.name}/{store.root.name} phases {batch + 1}/16",
                flush=True,
            )


@torch.inference_mode()
def matched_panel(model, anchors, payloads, store):
    for index, a in enumerate(anchors):
        name = f"matched_{index:03d}"
        if store.get(name) is not None:
            continue
        scores = {}
        for length in (32, 64, 128, 256):
            state = consume(model, payloads[a["path"]][a["offset"] - length : a["offset"]])
            scores[str(length)] = {
                "nll": -model.predict(state).log_softmax(-1)[0, a["target"]].item()
            }
        store.put(name, {"anchor": a, "scores": scores})


@torch.inference_mode()
def distribution_panel(model, selections, payloads, store):
    for index, row in enumerate(selections):
        name = f"position_{index:03d}"
        if store.get(name) is not None:
            continue
        a = row["anchor"]
        state = consume(model, payloads[a["path"]][a["offset"] - row["position"] : a["offset"]])
        store.put(name, {"selection": row, "distribution": distribution(model.predict(state))})


@torch.inference_mode()
def generation_panel(model, store):
    for index, prefix in enumerate(PROTOCOL["generation_prompts_hex"]):
        name = f"generation_{index}"
        if store.get(name) is not None:
            continue
        state = consume(model, bytes.fromhex(prefix))
        rows, output = [], []
        for _ in range(64):
            row = distribution(model.predict(state))
            byte = row["top1_byte"]
            output.append(byte)
            rows.append(row)
            state = model.consume(torch.tensor([byte], device="cuda:0"), state)
        require(
            state.shared.steps == 1 + (len(bytes.fromhex(prefix)) + 64) // 8,
            "generation clock regression",
        )
        store.put(
            name,
            {
                "prefix_hex": prefix,
                "output_hex": bytes(output).hex(),
                "escaped": repr(bytes(output)),
                "steps": rows,
                "shared_steps": state.shared.steps,
            },
        )
