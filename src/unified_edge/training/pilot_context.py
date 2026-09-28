"""Pinned VALIDATION anchors, phase diagnostics and document-cluster inference."""

import hashlib
import random
import statistics
from dataclasses import dataclass

import torch

from unified_edge.training.pilot_diagnostics import isolated_diagnostics
from unified_edge.training.pilot_plan import DOMAINS, Contract, digest, number, require


@dataclass(frozen=True)
class Anchor(Contract):
    document: str
    document_sha256: str
    domain: str
    offset: int
    target_bytes: int
    target_sha256: str
    split: str = "validation"

    def __post_init__(self):
        require(
            type(self.document) is str and bool(self.document), "immutable document ID required"
        )
        digest(self.document_sha256)
        digest(self.target_sha256)
        require(self.split == "validation" and self.domain in DOMAINS, "VALIDATION anchors only")
        require(
            type(self.offset) is int
            and self.offset >= 135
            and type(self.target_bytes) is int
            and self.target_bytes > 0,
            "insufficient anchor history/target",
        )

    @classmethod
    def from_dict(cls, value):
        require(
            isinstance(value, dict) and set(value) == set(cls.__dataclass_fields__),
            "invalid anchor fields",
        )
        return cls(**value)


@dataclass(frozen=True)
class AnchorManifest(Contract):
    schema: str
    dataset_sha256: str
    anchors: tuple[Anchor, ...]

    def __post_init__(self):
        require(
            self.schema == "edge-anchors-1"
            and type(self.anchors) is tuple
            and len(self.anchors) == 128
            and all(type(a) is Anchor for a in self.anchors),
            "exactly 128 typed anchors required; no sample shrink",
        )
        digest(self.dataset_sha256)
        require(
            len({(a.document_sha256, a.offset) for a in self.anchors}) == 128, "duplicate anchors"
        )
        for domain in DOMAINS:
            group = [a for a in self.anchors if a.domain == domain]
            require(
                len(group) == 32 and len({a.document_sha256 for a in group}) >= 16,
                "require 32 anchors and >=16 independent documents per domain",
            )

    @classmethod
    def from_dict(cls, value):
        require(
            isinstance(value, dict) and set(value) == set(cls.__dataclass_fields__),
            "invalid anchor manifest fields",
        )
        return cls(
            value["schema"],
            value["dataset_sha256"],
            tuple(Anchor.from_dict(a) for a in value["anchors"]),
        )

    def validate(self, manifest):
        require(manifest.sha256 == self.dataset_sha256, "anchor dataset mismatch")
        docs = {d.path: d for d in manifest.documents if d.split == "validation"}
        for a in self.anchors:
            require(a.document in docs, "anchor is not VALIDATION")
            d = docs[a.document]
            require(
                d.sha256 == a.document_sha256
                and d.domain == a.domain
                and a.offset + a.target_bytes <= d.byte_count,
                "anchor identity/range mismatch",
            )


def document_bootstrap(rows):
    """Unique observation IDs; resample whole documents, retaining all paired rows.

    Identical duplicate observations are idempotent; conflicting duplicates reject.
    Point estimates average anchors/phases; the uncertainty unit is the document.
    Percentile order statistics are explicitly indices 49/1949 for 2000 draws.
    """
    unique = {}
    for document, observation, value in rows:
        number(value)
        key = (document, observation)
        require(key not in unique or unique[key] == value, "conflicting duplicate observation")
        unique[key] = value
    groups = {}
    for (document, _), value in sorted(unique.items()):
        groups.setdefault(document, []).append(value)
    require(len(groups) >= 2, "bootstrap needs independent documents")
    values = [(sum(number(x) for x in v), len(v)) for _, v in sorted(groups.items())]
    rng = random.Random(43)

    def mean(selected):
        return float(sum(v for v, _ in selected) / sum(n for _, n in selected))

    draws = sorted(mean(rng.choices(values, k=len(values))) for _ in range(2000))
    return {
        "mean": mean(values),
        "ci95": [draws[49], draws[1949]],
        "documents": len(values),
        "resamples": 2000,
        "seed": 43,
    }


def context_thresholds(summary):
    checks = {
        "history": number(summary["history"]["mean"]) >= number(0.01),
        "history_ci": number(summary["history"]["ci95"][0]) > 0,
        "shuffle": number(summary["shuffle"]["mean"]) >= number(0.005),
        "shuffle_ci": number(summary["shuffle"]["ci95"][0]) > 0,
        "responsive_anchors": number(summary["responsive_fraction"]) >= number(0.75),
        "domain_floor": set(summary["domains"]) == set(DOMAINS)
        and all(number(v) >= number(-0.02) for v in summary["domains"].values()),
    }
    return {"status": "PASS" if all(checks.values()) else "STOP", "checks": checks}


def evaluate_context(rows, manifest):
    expected = {(a.sha256, phase) for a in manifest.anchors for phase in range(8)}
    require(
        len(rows) == len(expected) and {(r["anchor_sha256"], r["phase"]) for r in rows} == expected,
        "missing/duplicated context anchor phases",
    )
    anchors = {a.sha256: a for a in manifest.anchors}

    def paired(selected, field):
        return document_bootstrap(
            [
                (
                    anchors[r["anchor_sha256"]].document_sha256,
                    f"{r['anchor_sha256']}:{r['phase']}",
                    float(number(r[field]) - number(r["long_nll"])),
                )
                for r in selected
            ]
        )

    for r in rows:
        for key in (
            "short_nll",
            "long_nll",
            "shuffled_nll",
            "byte_tv",
            "shared_delta",
            "conv_delta",
            "ssm_delta",
            "pre_tanh_delta",
            "hidden_delta",
            "logit_delta",
            "tanh_saturation",
            "context_gradient_norm",
        ):
            number(r[key])
    tv = {key: max(r["byte_tv"] for r in rows if r["anchor_sha256"] == key) for key in anchors}
    summary = {
        "short_nll": statistics.mean(r["short_nll"] for r in rows),
        "long_nll": statistics.mean(r["long_nll"] for r in rows),
        "history": paired(rows, "short_nll"),
        "shuffle": paired(rows, "shuffled_nll"),
        "responsive_fraction": sum(number(v) > number(0.00001) for v in tv.values()) / 128,
        "domains": {
            d: paired([r for r in rows if anchors[r["anchor_sha256"]].domain == d], "short_nll")[
                "mean"
            ]
            for d in DOMAINS
        },
        "phases": {
            str(p): {
                "history": paired([r for r in rows if r["phase"] == p], "short_nll"),
                "shuffle": paired([r for r in rows if r["phase"] == p], "shuffled_nll"),
            }
            for p in range(8)
        },
    }
    return {**context_thresholds(summary), "summary": summary, "rows": rows}


def measure_anchor(model, anchor, raw):
    require(hashlib.sha256(raw).hexdigest() == anchor.document_sha256, "anchor document changed")
    target = raw[anchor.offset : anchor.offset + anchor.target_bytes]
    require(
        len(target) == anchor.target_bytes
        and hashlib.sha256(target).hexdigest() == anchor.target_sha256,
        "anchor target changed",
    )
    device = next(model.parameters()).device
    rows = []
    with isolated_diagnostics(model):
        for phase in range(8):
            # Phase offsets add the same 0..7 preceding bytes to both base histories.
            short = raw[anchor.offset - 32 - phase : anchor.offset]
            long = raw[anchor.offset - 128 - phase : anchor.offset]
            older = list(long[: -(32 + phase)])
            random.Random(f"43:{anchor.sha256}:{phase}").shuffle(older)
            shuffled = bytes(older) + short
            require(short[-32:] == long[-32:] == shuffled[-32:], "unmatched recent suffix")
            captures = []

            def hook(module, args, output):
                captures.append((args[0], output))

            handle = model.hierarchy.decoder.context.register_forward_hook(hook)

            def score(history):
                captures.clear()
                state = model.start()
                for b in history:
                    state = model.consume(torch.tensor([b], device=device), state)
                context, pre = captures[-1]
                initial_state, logits = state, model.predict(state)
                losses = []
                for b in target:
                    losses.append(-model.predict(state).log_softmax(-1)[0, b])
                    state = model.consume(torch.tensor([b], device=device), state)
                return torch.stack(losses).mean(), initial_state, logits, context, pre

            try:
                with torch.no_grad():
                    a = score(short)
                    c = score(shuffled)
                with torch.enable_grad():
                    b = score(long)
                    grads = torch.autograd.grad(
                        b[0],
                        tuple(model.hierarchy.decoder.context.parameters()),
                        allow_unused=False,
                    )
                    gradnorm = sum(g.detach().square().sum().item() for g in grads) ** 0.5

                def delta(x, y):
                    require(
                        torch.isfinite(x).all().item() and torch.isfinite(y).all().item(),
                        "nonfinite context state",
                    )
                    return (x.detach() - y.detach()).abs().max().item()

                mask = torch.isfinite(a[2])
                require(torch.equal(mask, torch.isfinite(b[2])), "logit support mismatch")
                rows.append(
                    {
                        "anchor_sha256": anchor.sha256,
                        "phase": phase,
                        "short_nll": a[0].item(),
                        "long_nll": b[0].item(),
                        "shuffled_nll": c[0].item(),
                        "byte_tv": (a[2][:, :256].softmax(-1) - b[2][:, :256].softmax(-1))
                        .abs()
                        .sum()
                        .item()
                        / 2,
                        "shared_delta": delta(a[3], b[3]),
                        "pre_tanh_delta": delta(a[4], b[4]),
                        "conv_delta": max(
                            delta(x.conv, y.conv)
                            for x, y in zip(a[1].shared.layers, b[1].shared.layers, strict=True)
                        ),
                        "ssm_delta": max(
                            delta(x.ssm, y.ssm)
                            for x, y in zip(a[1].shared.layers, b[1].shared.layers, strict=True)
                        ),
                        "hidden_delta": delta(a[1].hierarchy.hidden, b[1].hierarchy.hidden),
                        "logit_delta": delta(a[2][mask], b[2][mask]),
                        "tanh_saturation": (b[4].detach().tanh().abs() > 0.99)
                        .float()
                        .mean()
                        .item(),
                        "context_gradient_norm": gradnorm,
                    }
                )
            finally:
                handle.remove()
    return rows


def context_panel(trainer, manifest):
    manifest.validate(trainer.manifest)
    trainer._verify_data(full=True)
    rows = [
        r
        for a in manifest.anchors
        for r in measure_anchor(trainer.model, a, trainer.active_data.buffers[a.document])
    ]
    return evaluate_context(rows, manifest)


def evaluate_bridge(
    *,
    a_status,
    b_status,
    paired_confirmation,
    domain_improvements,
    replication_seed,
    replication_improvement,
):
    if b_status == "INCOMPLETE" or a_status == "INCOMPLETE":
        return {"status": "INCOMPLETE"}
    promote = (
        b_status == "PASS"
        and number(paired_confirmation["mean"]) >= number(0.01)
        and number(paired_confirmation["ci95"][0]) > 0
        and set(domain_improvements) == set(DOMAINS)
        and all(number(x) >= number(-0.02) for x in domain_improvements.values())
        and replication_seed == 29
        and number(replication_improvement) > 0
    )
    return {"status": "PROMOTE_B" if promote else "RETAIN_A" if a_status == "PASS" else "STOP"}
