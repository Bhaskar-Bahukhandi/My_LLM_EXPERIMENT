"""Isolated fixed byte-generation diagnostics and endpoint anti-collapse gates."""

import math
import random
import statistics
from collections import Counter
from contextlib import contextmanager
from itertools import combinations, groupby

import torch

from unified_edge.training.pilot_plan import (
    GenerationDiagnosticPlan,
    PromotionGatePlan,
    number,
    require,
)


@contextmanager
def isolated_diagnostics(model):
    device = next(model.parameters()).device
    python, cpu = random.getstate(), torch.get_rng_state()
    gpu = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    mode = model.training
    gradients = [
        (p, None if p.grad is None else p.grad.detach().clone()) for p in model.parameters()
    ]
    try:
        model.eval()
        yield
    finally:
        for parameter, gradient in gradients:
            parameter.grad = gradient
        model.train(mode)
        random.setstate(python)
        torch.set_rng_state(cpu)
        if gpu is not None:
            torch.cuda.set_rng_state(gpu, device)


def longest(values):
    return max((sum(1 for _ in group) for _, group in groupby(values)), default=0)


def longest_true(values):
    return max((sum(1 for _ in group) for value, group in groupby(values) if value), default=0)


def entropy(values):
    return -math.fsum(p * math.log2(p) for p in values if p > 0)


def distribution(probabilities):
    require(len(probabilities) == 267, "retain all 267 unfiltered class probabilities")
    require(
        all(type(p) in (int, float) and math.isfinite(p) and 0 <= p <= 1 for p in probabilities),
        "invalid probability",
    )
    require(abs(math.fsum(probabilities) - 1) <= 1e-12, "probabilities must sum to one")
    mass = math.fsum(probabilities[:256])
    require(mass > 0, "zero byte mass")
    q = [p / mass for p in probabilities[:256]]
    ordered = sorted(q, reverse=True)
    return {
        "q_space": q[32],
        "entropy_bits": entropy(probabilities),
        "margin": ordered[0] - ordered[1],
        "control_mass": math.fsum(probabilities[256:]),
        "byte_distribution": q,
    }


def byte_metrics(payload):
    require(type(payload) is bytes and len(payload) > 0, "nonempty raw bytes required")
    try:
        payload.decode("utf-8", errors="strict")
        valid = True
    except UnicodeDecodeError:
        valid = False
    return {
        "whitespace": sum(b in (9, 10, 11, 12, 13, 32) for b in payload) / len(payload),
        "printable": sum(32 <= b <= 126 or b in (9, 10, 13) for b in payload) / len(payload),
        "longest_run": longest(payload),
        "utf8": valid,
        "empirical_entropy_bits": entropy([n / len(payload) for n in Counter(payload).values()]),
        "distinct_ngrams": {
            str(n): len({payload[i : i + n] for i in range(len(payload) - n + 1)})
            for n in (1, 2, 4)
        },
    }


def trace_metrics(trace):
    payload = bytes.fromhex(trace["output_hex"])
    require(
        len(payload) == 256 and len(trace["probabilities"]) == 256 and trace["valid_state"] is True,
        "incomplete/invalid generation trace",
    )
    rows = [distribution(p) for p in trace["probabilities"]]
    q, margins = [r["q_space"] for r in rows], [r["margin"] for r in rows]
    result = {
        **byte_metrics(payload),
        "mean_q_space": statistics.mean(q),
        "median_q_space": statistics.median(q),
        "control_mass": statistics.mean(r["control_mass"] for r in rows),
        "mean_unfiltered_entropy_bits": statistics.mean(r["entropy_bits"] for r in rows),
        "median_margin": statistics.median(margins),
        "p95_margin": sorted(margins)[math.ceil(0.95 * len(margins)) - 1],
        "space_run": longest_true(v > 0.8 for v in q),
        "margin_run": longest_true(v >= 0.95 for v in margins),
        "first_byte_distribution": rows[0]["byte_distribution"],
        "suffix_hex": payload[-64:].hex(),
        "late_half": {
            **byte_metrics(payload[128:]),
            "median_q_space": statistics.median(q[128:]),
            "mean_q_space": statistics.mean(q[128:]),
            "mean_unfiltered_entropy_bits": statistics.mean(r["entropy_bits"] for r in rows[128:]),
            "median_margin": statistics.median(margins[128:]),
            "control_mass": statistics.mean(r["control_mass"] for r in rows[128:]),
        },
    }
    return result


def jsd(left, right):
    require(len(left) == len(right) and len(left) > 0, "JSD support mismatch")
    for values in (left, right):
        require(
            all(math.isfinite(p) and p >= 0 for p in values)
            and abs(math.fsum(values) - 1) <= 1e-12,
            "JSD requires probability distributions",
        )
    terms = []
    for p, q in zip(left, right, strict=True):
        m = (p + q) / 2
        terms.append((p * math.log(p / m) if p else 0) / 2 + (q * math.log(q / m) if q else 0) / 2)
    return math.fsum(terms)


def generate_panel(model, plan=GenerationDiagnosticPlan()):
    device = next(model.parameters()).device
    traces = []
    with isolated_diagnostics(model), torch.inference_mode():
        for prompt in plan.prompts_hex:
            prefix = bytes.fromhex(prompt)
            for seed in (None, *plan.seeds):
                generator = torch.Generator(device=device)
                generator.manual_seed(0 if seed is None else seed)
                state = model.start()
                for token in prefix:
                    state = model.consume(torch.tensor([token], device=device), state)
                output, probabilities = [], []
                for _ in range(plan.length):
                    logits = model.predict(state).double().flatten()
                    require(
                        logits.numel() == 267
                        and torch.isfinite(logits[:256]).all().item()
                        and not (torch.isnan(logits) | torch.isposinf(logits)).any().item(),
                        "invalid generation logits",
                    )
                    p = logits.softmax(-1)
                    probabilities.append(p.cpu().tolist())
                    q = p[:256] / p[:256].sum()
                    token = (
                        q.argmax().reshape(1)
                        if seed is None
                        else torch.multinomial(q, 1, generator=generator)
                    )
                    output.append(token.item())
                    state = model.consume(token, state)
                consumed = len(prefix) + plan.length
                require(
                    state.shared.steps == 1 + consumed // 8
                    and state.hierarchy.pending.shape[1] == consumed % 8,
                    "generation state clock mismatch",
                )
                traces.append(
                    {
                        "prompt_hex": prompt,
                        "seed": seed,
                        "output_hex": bytes(output).hex(),
                        "probabilities": probabilities,
                        "valid_state": True,
                    }
                )
    return traces


def quality_thresholds(summary, plan=PromotionGatePlan()):
    """Pure threshold evaluator; decimal strings define inclusive boundary comparisons."""
    g = summary["greedy"]
    require(len(g) == 9 and len(summary["prompt_control_mass"]) == 9, "quality panel size")
    checks = {
        "all_traces_complete": summary["complete"] is True,
        "control_mass": all(
            number(x) <= number(plan.control_mass) for x in summary["prompt_control_mass"]
        ),
        "median_whitespace": number(statistics.median(r["whitespace"] for r in g))
        <= number(plan.median_whitespace),
        "maximum_whitespace": all(
            number(r["whitespace"]) <= number(plan.maximum_whitespace) for r in g
        ),
        "runs_and_entropy": sum(
            r["longest_run"] <= 32 and number(r["empirical_entropy_bits"]) >= number(2) for r in g
        )
        >= 7,
        "late_space": number(statistics.median(r["late_half"]["median_q_space"] for r in g))
        <= number(0.35),
        "no_space_attractor": summary["max_space_run"] < 32,
        "printable": sum(number(r["printable"]) >= number(0.9) for r in g) >= 7,
        "utf8": summary["utf8_samples"] >= 33,
        "sample_diversity": sum(n >= 3 for n in summary["unique_sampled_outputs"]) >= 7,
        "greedy_diversity": summary["unique_greedy_suffixes"] >= 7,
        "no_margin_collapse": max(r["margin_run"] for r in g) < 32,
        "prompt_conditioning": number(summary["median_jsd_nats"]) >= number(0.005),
    }
    return {"status": "PASS" if all(checks.values()) else "STOP", "checks": checks}


def evaluate_quality(traces, *, step, endpoint=8000):
    if step != endpoint or len(traces) != 45:
        return {"status": "INCOMPLETE", "reason": "endpoint and all 45 traces required"}
    plan = GenerationDiagnosticPlan()
    expected = {(p, s) for p in plan.prompts_hex for s in (None, *plan.seeds)}
    require(
        {(t["prompt_hex"], t["seed"]) for t in traces} == expected,
        "duplicate/missing/unexpected trace identities",
    )
    groups = {p: [t for t in traces if t["prompt_hex"] == p] for p in plan.prompts_hex}
    metrics = {(t["prompt_hex"], t["seed"]): trace_metrics(t) for t in traces}
    greedy = [metrics[p, None] for p in plan.prompts_hex]
    summary = {
        "complete": True,
        "greedy": greedy,
        "prompt_control_mass": [
            statistics.mean(metrics[p, s]["control_mass"] for s in (None, *plan.seeds))
            for p in plan.prompts_hex
        ],
        "max_space_run": max(r["space_run"] for r in metrics.values()),
        "utf8_samples": sum(metrics[p, s]["utf8"] for p in plan.prompts_hex for s in plan.seeds),
        "unique_sampled_outputs": [
            len({t["output_hex"] for t in groups[p] if t["seed"] is not None})
            for p in plan.prompts_hex
        ],
        "unique_greedy_suffixes": len({r["suffix_hex"] for r in greedy}),
        "median_jsd_nats": statistics.median(
            jsd(a["first_byte_distribution"], b["first_byte_distribution"])
            for a, b in combinations(greedy, 2)
        ),
    }
    return {
        **quality_thresholds(summary),
        "summary": summary,
        "per_prompt": {
            p: {str(s): metrics[p, s] for s in (None, *plan.seeds)} for p in plan.prompts_hex
        },
    }
