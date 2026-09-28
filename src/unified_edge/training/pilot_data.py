"""Metadata-only capacity/partition checks and verified subset evaluation."""

import hashlib
from dataclasses import dataclass

import torch

from unified_edge.training.data import DatasetManifest, Window, batches_by_length
from unified_edge.training.optimization import raw_byte_nll
from unified_edge.training.pilot_plan import DOMAINS, Contract, PilotPlan, digest, require


@dataclass(frozen=True)
class SubsetSpan(Contract):
    document: str
    document_sha256: str
    start: int
    stop: int
    payload_sha256: str

    def __post_init__(self):
        digest(self.document_sha256)
        digest(self.payload_sha256)
        require(isinstance(self.document, str) and bool(self.document), "document required")
        require(
            type(self.start) is int and type(self.stop) is int and 0 <= self.start < self.stop,
            "invalid subset span",
        )

    @classmethod
    def from_dict(cls, value):
        require(
            isinstance(value, dict) and set(value) == set(cls.__dataclass_fields__),
            "unknown/missing span fields",
        )
        return cls(**value)


@dataclass(frozen=True)
class ValidationSubsets(Contract):
    schema: str
    dataset_sha256: str
    selection: tuple[SubsetSpan, ...]
    monitor: tuple[SubsetSpan, ...]
    confirmation: tuple[SubsetSpan, ...]

    def __post_init__(self):
        require(self.schema == "edge-validation-subsets-1", "invalid subset schema")
        digest(self.dataset_sha256)
        spans = []
        for group in (self.selection, self.monitor, self.confirmation):
            require(
                type(group) is tuple and bool(group) and all(type(s) is SubsetSpan for s in group),
                "typed nonempty spans required",
            )
            spans.extend(group)
        # Content-identical documents are one source for overlap checks, regardless of path.
        for i, left in enumerate(spans):
            for right in spans[i + 1 :]:
                if left.document == right.document or left.document_sha256 == right.document_sha256:
                    require(
                        max(left.start, right.start) >= min(left.stop, right.stop),
                        "validation subsets overlap",
                    )

    @classmethod
    def from_dict(cls, value):
        require(
            isinstance(value, dict) and set(value) == set(cls.__dataclass_fields__),
            "unknown/missing subset fields",
        )
        return cls(
            value["schema"],
            value["dataset_sha256"],
            *(
                tuple(SubsetSpan.from_dict(s) for s in value[name])
                for name in ("selection", "monitor", "confirmation")
            ),
        )

    def validate(self, manifest: DatasetManifest, *, exact_capacity=True):
        require(self.dataset_sha256 == manifest.sha256, "subset dataset identity mismatch")
        documents = {d.path: d for d in manifest.documents if d.split == "validation"}
        counts = []
        for group in (self.selection, self.monitor, self.confirmation):
            for span in group:
                require(span.document in documents, "subset must belong to VALIDATION")
                doc = documents[span.document]
                require(
                    span.document_sha256 == doc.sha256 and span.stop <= doc.byte_count,
                    "subset range/document identity mismatch",
                )
            counts.append(sum(s.stop - s.start for s in group))
        if exact_capacity:
            require(counts == [262144, 262144, 3571712], "wrong validation subset capacities")
            require(
                sum(d.byte_count for d in documents.values()) == sum(counts),
                "validation partition must cover allocation exactly",
            )
            require(
                {documents[s.document].domain for s in self.confirmation} == set(DOMAINS),
                "confirmation must represent every declared domain",
            )
        return counts


def validate_capacity(manifest: DatasetManifest):
    plan = PilotPlan()
    require(
        len({d.sha256 for d in manifest.documents}) == len(manifest.documents),
        "capacity cannot count duplicated document bytes",
    )
    require({d.domain for d in manifest.documents} == set(DOMAINS), "unknown/missing domains")
    for split, quotas in (("train", plan.train_quotas), ("validation", plan.validation_quotas)):
        counts = tuple(
            sum(d.byte_count for d in manifest.documents if d.split == split and d.domain == domain)
            for domain in DOMAINS
        )
        require(counts == quotas, f"{split} domain capacities do not match pilot")
    return {
        "train": plan.train_bytes,
        "validation": plan.validation_bytes,
        "future_test_reserve_metadata_only": plan.future_sealed_test_reserve_bytes,
    }


def subset_windows(trainer, spans):
    trainer._verify_data(full=True)
    windows = []
    domains = {d.path: d.domain for d in trainer.manifest.documents if d.split == "validation"}
    for span in spans:
        require(span.document in domains, "evaluation excludes TRAIN/TEST")
        raw = trainer.active_data.buffers[span.document][span.start : span.stop]
        require(hashlib.sha256(raw).hexdigest() == span.payload_sha256, "subset payload mismatch")
        for start in range(0, len(raw), trainer.config.sequence_length):
            windows.append(
                Window(
                    span.document,
                    span.start + start,
                    raw[start : start + trainer.config.sequence_length],
                )
            )
    return windows, domains


def evaluate_subset(trainer, spans):
    from unified_edge.training.pilot_diagnostics import isolated_diagnostics

    windows, domains = subset_windows(trainer, spans)
    totals = {}
    with isolated_diagnostics(trainer.model), torch.inference_mode():
        for window in windows:
            target = batches_by_length([window])[0].to(trainer.device)
            loss, count = raw_byte_nll(trainer.model(target), target)
            total, previous = totals.get(domains[window.document], (0.0, 0))
            totals[domains[window.document]] = (total + loss.item(), previous + count)
    count = sum(n for _, n in totals.values())
    require(count > 0, "empty evaluation")
    return {
        "step": trainer.global_step,
        "valid_target_bytes": count,
        "nll": sum(v for v, _ in totals.values()) / count,
        "domains": {k: {"nll": v / n, "bytes": n} for k, (v, n) in totals.items()},
    }
