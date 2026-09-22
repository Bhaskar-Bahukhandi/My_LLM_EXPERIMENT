"""Supplemental position-stratified distributions; immutable main study stays intact."""

from collections import defaultdict
from pathlib import Path

import torch
from context_distribution_study import (
    PARAMETER_SHA,
    ROOT,
    STUDY,
    Store,
    consume,
    parent_inputs,
    read,
    sha,
    tensor_hash,
    verify_corpus,
    verify_inventory,
)
from context_study_metrics import LENGTHS, distribution, require

from unified_edge.dense_model import DenseByteModel
from unified_edge.training.device import configure_device
from unified_edge.training.trainer import environment


def selections(anchors, per_stratum=8):
    """Sample fixed anchors by domain/length/absolute32-byte window-position bucket."""
    groups = defaultdict(list)
    for anchor in anchors:
        for length in LENGTHS:
            position = anchor["offset"] % length
            groups[(anchor["domain"], length, position // 32)].append(
                dict(anchor=anchor, length=length, position=position)
            )
    return [
        row
        for _, rows in sorted(groups.items())
        for row in sorted(rows, key=lambda r: r["anchor"]["key"])[:per_stratum]
    ]


def run():
    base = read(STUDY / "binding.json")
    for name, expected in base["tools"].items():
        require(sha(ROOT / "scripts" / name) == expected, "main measurement tool changed")
    original = Store(STUDY, base)
    require(original.get("integrity_final") is not None, "complete main study first")
    saved, resolved, config, _, payloads, before, full = parent_inputs()
    require(before == base["parent_before"], "parent changed since main study")
    configure_device("cuda:0")
    torch.set_num_threads(config.cpu_threads)
    torch.use_deterministic_algorithms(True)
    require(environment(config) == base["environment"], "study runtime mismatch")
    selection = selections(base["anchors"])
    binding = {
        "schema": "position-distribution-1",
        "main_binding_sha256": sha(STUDY / "binding.json"),
        "tool_sha256": sha(Path(__file__)),
        "parent_before": before,
        "protocol": {
            "selection": "First 8 frozen hashes per domain / length / 32-byte position bucket",
            "reset": "BOS at floor(anchor_offset/L)*L in the same document",
            "target": "Same validation anchor target; history length is offset modulo L",
            "rationale": (
                "Exact-L targets lie on patch boundaries; separately sample window positions "
                "without changing the matched experiment."
            ),
            "classification": base["protocol"]["classification"],
        },
        "selections": selection,
    }
    store = Store(STUDY / "position-distributions", binding)
    model = DenseByteModel(resolved).to("cuda:0")
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    del saved
    try:
        with torch.inference_mode():
            for index, row in enumerate(selection):
                name = f"context_{index:04d}"
                if store.get(name) is not None:
                    continue
                anchor = row["anchor"]
                raw = payloads[anchor["path"]]
                offset = anchor["offset"]
                require(raw[offset] == anchor["target"], "anchor identity mismatch")
                state = consume(model, raw[offset - row["position"] : offset])
                before_space = distribution(model.predict(state))
                state = model.consume(torch.tensor([32], device="cuda:0"), state)
                store.put(
                    name,
                    dict(
                        selection=row,
                        distribution=before_space,
                        after_space=distribution(model.predict(state)),
                    ),
                )
        require(tensor_hash(model.state_dict()) == PARAMETER_SHA, "model changed")
        require(verify_corpus(full) == base["split_hashes"], "corpus changed")
        verify_inventory(before)
        if store.get("integrity_final") is None:
            store.put(
                "integrity_final",
                {"status": "PASS", "parent_after": before, "test": "SEALED; hash-only integrity"},
            )
        print("MILESTONE: position-stratified distribution survey complete", flush=True)
    finally:
        verify_inventory(before)


if __name__ == "__main__":
    run()
