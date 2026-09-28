"""Synthetic R9 metric contracts and exact boundaries; no production data."""

import math
import random
from dataclasses import FrozenInstanceError

import pytest
import torch

from unified_edge.training.pilot_context import (
    context_thresholds,
    document_bootstrap,
    evaluate_bridge,
)
from unified_edge.training.pilot_diagnostics import (
    byte_metrics,
    distribution,
    evaluate_quality,
    generate_panel,
    jsd,
    quality_thresholds,
    trace_metrics,
)
from unified_edge.training.pilot_gates import evaluate_promotion, select_lr, unstable_losses
from unified_edge.training.pilot_plan import (
    DOMAINS,
    PROMPTS,
    ContextDiagnosticPlan,
    EvaluationPlan,
    GenerationDiagnosticPlan,
    LRProbePlan,
    PilotPlan,
    PromotionGatePlan,
)


@pytest.mark.parametrize(
    "cls",
    [
        PilotPlan,
        LRProbePlan,
        EvaluationPlan,
        GenerationDiagnosticPlan,
        ContextDiagnosticPlan,
        PromotionGatePlan,
    ],
)
def test_plans_strict_canonical_immutable(cls):
    plan = cls()
    assert cls.from_dict(plan.to_dict()).sha256 == plan.sha256
    for changed in (dict(plan.to_dict(), unknown=True), dict(plan.to_dict(), schema="unexpected")):
        with pytest.raises(ValueError):
            cls.from_dict(changed)
    with pytest.raises(FrozenInstanceError):
        plan.schema = "changed"


def test_exact_plan_and_cadence():
    p = PilotPlan()
    assert p.nominal_bytes_per_update == 8192
    assert p.updates * p.nominal_bytes_per_update == p.train_bytes == 65536000
    assert sum(p.train_quotas) == p.train_bytes
    assert sum(p.validation_quotas) == p.validation_bytes == 4096000
    e = p.evaluation
    assert e.lr_selection_bytes + e.monitor_bytes + e.confirmation_bytes == 4096000
    assert e.events(0) == ("monitor", "generation", "context", "checkpoint")
    assert e.events(250) == ("monitor", "checkpoint")
    assert e.events(500) == ("monitor", "generation", "checkpoint")
    assert e.events(1000) == ("monitor", "confirmation", "generation", "context", "checkpoint")
    assert len(e.events(8000)) == len(set(e.events(8000))) == 5
    assert e.events(7, stopping=True) == ("checkpoint",)
    assert p.training_config(0.0003, "cpu", probe=True).warmup_steps == 32


def arms():
    return [
        dict(
            lr=lr,
            step=256,
            seed=17,
            scheduler="warmup_constant_v1",
            initial_nll=1.0,
            endpoint_nll=0.9,
            initial_parameters_sha256="a" * 64,
            data_order_sha256="b" * 64,
            subset_sha256="c" * 64,
            finite_states=True,
            finite_gradients=True,
            finite_parameters=True,
            losses=[1.0] * 256,
            clipped=[False] * 192 + [True] * 64,
        )
        for lr in LRProbePlan().rates
    ]


def test_lr_boundaries_tie_clipping_and_endpoint_only():
    values = arms()
    assert select_lr(values, 10800)["selected_lr"] == 0.0003
    values[0]["endpoint_nll"], values[1]["endpoint_nll"] = 0.9, 0.89
    assert select_lr(values, 1)["selected_lr"] == 0.0003
    values[1]["endpoint_nll"] = 0.889999
    assert select_lr(values, 1)["selected_lr"] == 0.0006
    for a in values:
        a["endpoint_nll"] = 0.900001
    assert select_lr(values, 1)["status"] == "STOP"
    for a in values:
        a["endpoint_nll"] = 0.9
        a["clipped"][128] = True
    assert select_lr(values, 1)["status"] == "STOP"
    assert select_lr(arms(), 10800.001)["reason"] == "BUDGET_EXCEEDED"
    values = arms()
    values[0]["step"] = 255
    with pytest.raises(ValueError, match="endpoints"):
        select_lr(values, 1)
    assert not unstable_losses([1.0] * 64 + [2.0] * 3)
    assert unstable_losses([1.0] * 64 + [2.000001] * 3)


def test_nll_promotion_exact_inclusive_boundaries():
    e = dict(
        implemented=[f"R{i}" for i in range(1, 10)],
        lr_status="PASS",
        updates=8000,
        data_valid=True,
        exposure_valid=True,
        initial_nll=1.0,
        endpoint_nll=0.9,
        domain_improvements=dict.fromkeys(DOMAINS, 0.02),
        checkpoint_export_consistent=True,
        stop_reason=None,
        endpoint_only=True,
    )
    assert evaluate_promotion(e)["status"] == "PASS"
    assert evaluate_promotion(dict(e, endpoint_nll=0.900001))["status"] == "STOP"
    assert (
        evaluate_promotion(dict(e, domain_improvements=dict.fromkeys(DOMAINS, 0.019999)))["status"]
        == "STOP"
    )
    assert evaluate_promotion(dict(e, updates=7999))["status"] == "STOP"
    assert evaluate_promotion({})["status"] == "INCOMPLETE"


def quality_summary():
    return dict(
        complete=True,
        greedy=[
            dict(
                whitespace=0.65,
                longest_run=32,
                empirical_entropy_bits=2.0,
                late_half={"median_q_space": 0.35},
                printable=0.9,
                margin_run=31,
            )
        ]
        * 9,
        prompt_control_mass=[0.01] * 9,
        max_space_run=31,
        utf8_samples=33,
        unique_sampled_outputs=[3] * 9,
        unique_greedy_suffixes=7,
        median_jsd_nats=0.005,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("utf8_samples", 32),
        ("median_jsd_nats", 0.004999),
        ("prompt_control_mass", [0.010001] * 9),
        ("max_space_run", 32),
        ("unique_greedy_suffixes", 6),
        ("unique_sampled_outputs", [2] * 9),
    ],
)
def test_quality_summary_thresholds(field, value):
    summary = quality_summary()
    assert quality_thresholds(summary)["status"] == "PASS"
    summary[field] = value
    assert quality_thresholds(summary)["status"] == "STOP"


@pytest.mark.parametrize(
    "field,value",
    [
        ("whitespace", 0.650001),
        ("longest_run", 33),
        ("empirical_entropy_bits", 1.99999),
        ("printable", 0.899999),
        ("margin_run", 32),
        ("late_half", {"median_q_space": 0.350001}),
    ],
)
def test_greedy_thresholds(field, value):
    summary = quality_summary()
    summary["greedy"] = [dict(r, **{field: value}) for r in summary["greedy"]]
    assert quality_thresholds(summary)["status"] == "STOP"


def test_metrics_probabilities_utf8_and_analytic_jsd():
    assert jsd([1.0, 0.0], [0.0, 1.0]) == pytest.approx(math.log(2))
    assert jsd([0.5, 0.5], [0.5, 0.5]) == 0
    p = [1 / 267] * 267
    result = distribution(p)
    assert result["entropy_bits"] == pytest.approx(math.log2(267))
    assert result["control_mass"] == pytest.approx(11 / 267)
    assert result["q_space"] == pytest.approx(1 / 256)
    assert byte_metrics(b"\t\n\v\f\r ")["whitespace"] == 1
    assert byte_metrics(b"\t\n\v\f\r ")["printable"] == 4 / 6
    assert not byte_metrics(b"\xff")["utf8"]
    trace = dict(output_hex=(b"abcd" * 64).hex(), probabilities=[p] * 256, valid_state=True)
    row = trace_metrics(trace)
    assert row["empirical_entropy_bits"] == 2
    assert row["distinct_ngrams"] == {"1": 4, "2": 4, "4": 4}
    assert evaluate_quality([], step=8000)["status"] == "INCOMPLETE"


def context_summary():
    return dict(
        history={"mean": 0.01, "ci95": [0.001, 0.02]},
        shuffle={"mean": 0.005, "ci95": [0.001, 0.02]},
        responsive_fraction=0.75,
        domains=dict.fromkeys(DOMAINS, -0.02),
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("history", {"mean": 0.009999, "ci95": [0.001, 0.02]}),
        ("shuffle", {"mean": 0.004999, "ci95": [0.001, 0.02]}),
        ("responsive_fraction", 0.749999),
        ("history", {"mean": 0.01, "ci95": [0.0, 0.02]}),
        ("domains", dict.fromkeys(DOMAINS, -0.020001)),
    ],
)
def test_context_thresholds(field, value):
    summary = context_summary()
    assert context_thresholds(summary)["status"] == "PASS"
    summary[field] = value
    assert context_thresholds(summary)["status"] == "STOP"


def test_document_bootstrap_determinism_and_duplicate_evidence():
    rows = [("a", "a0", 0.01), ("a", "a1", 0.03), ("b", "b0", 0.04)]
    baseline = document_bootstrap(rows)
    assert baseline == document_bootstrap(rows) == document_bootstrap(rows + rows)
    assert baseline["documents"] == 2 and baseline["mean"] == pytest.approx(0.08 / 3)
    with pytest.raises(ValueError, match="conflicting"):
        document_bootstrap(rows + [("a", "a0", 99.0)])


def test_bridge_retains_a_unless_b_earns_promotion():
    args = dict(
        a_status="PASS",
        b_status="PASS",
        paired_confirmation={"mean": 0.01, "ci95": [0.001, 0.02]},
        domain_improvements=dict.fromkeys(DOMAINS, -0.02),
        replication_seed=29,
        replication_improvement=0.001,
    )
    assert evaluate_bridge(**args)["status"] == "PROMOTE_B"
    args["replication_improvement"] = 0
    assert evaluate_bridge(**args)["status"] == "RETAIN_A"
    args.update(a_status="STOP", b_status="STOP")
    assert evaluate_bridge(**args)["status"] == "STOP"


def test_generation_full_fixed_panel_rng_and_controls():
    from types import SimpleNamespace

    class ByteFixture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def state(self, n):
            return SimpleNamespace(
                n=n,
                shared=SimpleNamespace(steps=1 + n // 8),
                hierarchy=SimpleNamespace(pending=torch.empty(1, n % 8)),
            )

        def start(self):
            return self.state(0)

        def predict(self, state):
            random.random()
            torch.rand(1)
            return torch.arange(267).float().remainder(7).reshape(1, -1)

        def consume(self, token, state):
            assert 0 <= token.item() < 256
            return self.state(state.n + 1)

    model = ByteFixture()
    py, cpu = random.getstate(), torch.get_rng_state()
    traces = generate_panel(model)
    assert len(traces) == 45
    assert {bytes.fromhex(t["prompt_hex"]) for t in traces} == set(PROMPTS)
    assert all(len(t["probabilities"]) == 256 and len(t["probabilities"][0]) == 267 for t in traces)
    assert random.getstate() == py and torch.equal(torch.get_rng_state(), cpu)
    result = evaluate_quality(traces, step=8000)
    assert result["status"] == "STOP"
    with pytest.raises(ValueError):
        evaluate_quality(traces[:-1] + [traces[0]], step=8000)
