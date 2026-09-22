"""Read-only Stage-A context study; durable per-block results, no production trainer."""

import argparse
import copy
import hashlib
import json
import math
import random
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from benchmark_fingerprints import canonical
from context_study_metrics import (
    LENGTHS,
    PROTOCOL,
    add_losses,
    anchor_history,
    check_baseline,
    distribution,
    require,
    select_anchors,
    summarize_losses,
    validation_documents,
)
from training_progress import publish_progress

from unified_edge.dense_model import DenseByteModel
from unified_edge.resolve import ResolvedConfig
from unified_edge.training.checkpoint import load_checkpoint
from unified_edge.training.config import TrainingConfig, canonical_hash
from unified_edge.training.data import DatasetManifest, Window, batches_by_length, file_bytes
from unified_edge.training.device import configure_device
from unified_edge.training.memory import cuda_memory, process_memory
from unified_edge.training.optimization import finite_gradients, raw_byte_nll
from unified_edge.training.trainer import code_identity, environment

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "data/full-training-2m-v1"
PARENT = PRODUCTION / "attempt-003/checkpoints/step_005000"
STUDY = ROOT / "data/context-study-2m-v1"
DATA = ROOT / "data/pilot-2m-r1"
PARAMETER_SHA = "4b8416333d4faf32b86d20eabd2addf6bb6fb39c52783a3427597657db71584d"
FROZEN = {
    "Unified_Edge400_Master_Bible_v3.0_FINAL.md": (
        "754e99e9feca52c8744b71891d6e71993233dea43c3d7155900334e28c121f75"
    ),
    "Unified_Edge400_Master_Roadmap_and_Formula_Board_v1.0_FINAL.md": (
        "91bb81b09162e8ccdbdd2baa5bcd5c849a912bfe15e53f798c09bd70b788af99"
    ),
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inventory():
    paths = list(PARENT.iterdir()) + [PRODUCTION / "binding.json", PRODUCTION / "progress.json"]
    paths += [ROOT / p for p in FROZEN]
    paths += [ROOT / f"reports/full_training_2m_stage_a.{suffix}" for suffix in ("md", "json")]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths if p.is_file()}


def verify_inventory(expected):
    require(inventory() == expected, "immutable production/specification inventory changed")


def tensor_hash(state):
    h = hashlib.sha256()
    for name, value in state.items():
        require(bool(torch.isfinite(value).all()), f"nonfinite parent tensor {name}")
        h.update(name.encode())
        h.update(bytes(value.detach().cpu().contiguous().view(torch.uint8).flatten().tolist()))
    return h.hexdigest()


def verify_corpus(full):
    """Integrity-only streaming digest; TEST bytes never returned or interpreted."""
    hashes = {s: hashlib.sha256() for s in ("train", "validation", "test")}
    counts = {s: 0 for s in hashes}
    for doc in full["documents"]:
        relative = Path(doc["path"])
        path = (DATA / relative).resolve()
        require(path.is_relative_to(DATA.resolve()), "corpus path escape")
        h, size = hashlib.sha256(), 0
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                h.update(chunk)
                hashes[doc["split"]].update(chunk)
                size += len(chunk)
        require(h.hexdigest() == doc["sha256"] and size == doc["byte_count"], "corpus mutation")
        counts[doc["split"]] += size
    result = {s: h.hexdigest() for s, h in hashes.items()}
    require(result == full["split_sha256"] and counts == full["split_bytes"], "split mismatch")
    return result


def parent_inputs():
    before = inventory()
    for path, expected in FROZEN.items():
        require(before[path] == expected, "FINAL specification changed")
    saved = load_checkpoint(PARENT)
    production = read(PRODUCTION / "binding.json")
    sidecar = read(PARENT / "production_binding.json")
    require(
        sidecar["binding_sha256"] == sha(PRODUCTION / "binding.json"), "parent binding mismatch"
    )
    require(
        sidecar["checkpoint_manifest_sha256"] == sha(PARENT / "manifest.json"), "sidecar mismatch"
    )
    require(saved["global_step"] == 5000 and saved["micro_step"] == 10000, "parent update mismatch")
    require(
        saved["examples_seen"] == 20000 and saved["bytes_seen"] == 639613, "parent counts mismatch"
    )
    require(
        saved["data_cursor"]["epoch"] == 0 and saved["data_cursor"]["offset"] == 20000,
        "parent cursor mismatch",
    )
    require(
        saved["accumulation_position"] == 0 and saved["scheduler"]["completed"] == 5000,
        "parent scheduler mismatch",
    )
    require(saved["optimizer_update_counts"] == [5000] * 56, "parent AdamW counts mismatch")
    require(tensor_hash(saved["model"]) == PARAMETER_SHA, "parent parameter checksum mismatch")
    resolved = ResolvedConfig.from_dict(saved["model_config"])
    config = TrainingConfig.from_dict(saved["training_config"])
    require(resolved.sha256 == production["resolved_sha256"], "model config mismatch")
    require(
        config.to_dict() == production["training_config"]
        and config.sha256 == production["training_sha256"],
        "training config mismatch",
    )
    require(
        config.sequence_length == 32 and config.total_steps == 78167, "production schedule changed"
    )
    require(
        saved["code"]["source_sha256"] == code_identity()["source_sha256"],
        "runtime source mismatch",
    )
    manifest_path = DATA / "final-corpus-v1/manifest.json"
    full = read(manifest_path)
    frozen = production["frozen_inputs"]
    require(sha(manifest_path) == frozen["frozen_manifest"]["raw_sha256"], "full manifest mismatch")
    require(
        hashlib.sha256(canonical(full)).hexdigest()
        == frozen["frozen_manifest"]["canonical_sha256"],
        "canonical mismatch",
    )
    require(
        verify_corpus(full) == sidecar["split_sha256"] == frozen["split_sha256"],
        "split identity mismatch",
    )
    manifest = DatasetManifest.from_dict(read(DATA / "final-corpus-v1/training_manifest.json"))
    order = list(
        range(sum(math.ceil(d.byte_count / 32) for d in manifest.documents if d.split == "train"))
    )
    random.Random(config.seed).shuffle(order)
    require(saved["data_cursor"]["order"] == order, "parent data ordering changed")
    require(
        manifest.sha256 == saved["dataset_sha256"] == sidecar["manifest_sha256"],
        "native manifest mismatch",
    )
    payloads = {
        d.path: file_bytes(DATA, d.path) for d in manifest.documents if d.split == "validation"
    }
    for d in manifest.documents:
        if d.split == "validation":
            require(hashlib.sha256(payloads[d.path]).hexdigest() == d.sha256, "validation mismatch")
    require(sum(map(len, payloads.values())) == 500000, "validation total changed")
    return saved, resolved, config, manifest, payloads, before, full


class Store:
    """Immutable checksummed measurements; progress is a recoverable index only."""

    def __init__(self, root, binding):
        self.root = Path(root).resolve()
        require(not self.root.is_relative_to(PRODUCTION.resolve()), "study cannot write production")
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / "binding.json"
        if path.exists():
            require(read(path) == binding, "existing study binding mismatch; no overwrite")
        else:
            require(not list(self.root.iterdir()), "unbound study directory is not empty")
            publish_progress(path, binding)
        self.binding_hash = canonical_hash(binding)
        self.publish()

    def get(self, name):
        require(name.replace("_", "").isalnum(), "invalid measurement identity")
        path = self.root / f"{name}.json"
        if not path.exists():
            return None
        record = read(path)
        require(record["binding_sha256"] == self.binding_hash, "measurement binding mismatch")
        require(
            record["sha256"] == canonical_hash(record["value"]), "measurement integrity mismatch"
        )
        return record["value"]

    def put(self, name, value):
        require(self.get(name) is None, "refusing to overwrite completed measurement")
        publish_progress(
            self.root / f"{name}.json",
            {
                "binding_sha256": self.binding_hash,
                "sha256": canonical_hash(value),
                "value": value,
            },
        )
        self.publish()

    def publish(self):
        names = sorted(
            p.stem for p in self.root.glob("*.json") if p.stem not in ("binding", "progress")
        )
        for name in names:
            self.get(name)
        publish_progress(
            self.root / "progress.json", {"binding_sha256": self.binding_hash, "completed": names}
        )


def memory():
    return {**cuda_memory(torch.device("cuda:0")), **process_memory()}


def memory_summary(rows):
    result = {k: max(r[k] for r in rows) for k in ("peak_allocated_bytes", "peak_reserved_bytes")}
    result["minimum_sampled_free_bytes"] = min(r["free_device_bytes"] for r in rows)
    result["process_rss_status"] = rows[-1]["process_rss_status"]
    if all(r["process_rss_status"] == "MEASURED" for r in rows):
        result["peak_observed_process_rss_bytes"] = max(r["process_rss_bytes"] for r in rows)
    return result


@torch.inference_mode()
def segmented(model, documents, length, store):
    """Match Trainer.validate ordering and summed FP32 objective; persist 256-window blocks."""
    name = f"segmented_{length}"
    if (complete := store.get(name)) is not None:
        return complete
    windows = [
        (Window(d.path, start, raw[start : start + length]), d.domain)
        for d, raw in documents
        for start in range(0, len(raw), length)
    ]
    blocks = []
    for block_start in range(0, len(windows), 256):
        block_name = f"segment_{length}_{block_start:06d}"
        block = store.get(block_name)
        if block is None:
            torch.cuda.reset_peak_memory_stats()
            mem = [memory()]
            started = time.perf_counter()
            total, count, domains, buckets = 0.0, 0, {}, {}
            for start in range(block_start, min(block_start + 256, len(windows)), 2):
                selected = windows[start : start + 2]
                grouped = {}
                for window, domain in selected:
                    grouped.setdefault(len(window.payload), []).append((window, domain))
                for group in grouped.values():
                    target = batches_by_length([w for w, _ in group])[0].to("cuda:0")
                    logits = model(target)
                    summed, valid = raw_byte_nll(logits, target)
                    total += summed.item()
                    count += valid
                    losses = (
                        F.cross_entropy(logits.flatten(0, 1), target.flatten(), reduction="none")
                        .reshape(target.shape)
                        .cpu()
                        .tolist()
                    )
                    for (_, domain), row in zip(group, losses, strict=True):
                        add_losses(domains, domain, row)
                        for position in range(0, len(row), 32):
                            add_losses(buckets, str(position), row[position : position + 32])
            mem.append(memory())
            block = dict(
                total=total,
                count=count,
                domains=domains,
                buckets=buckets,
                seconds=time.perf_counter() - started,
                memory=memory_summary(mem),
            )
            store.put(block_name, block)
        blocks.append(block)
    total = sum(b["total"] for b in blocks)
    count = sum(b["count"] for b in blocks)
    domains, buckets = {}, {}
    for block in blocks:
        for destination, field in ((domains, "domains"), (buckets, "buckets")):
            for key, (value, n) in block[field].items():
                old, number = destination.get(key, (0.0, 0))
                destination[key] = [old + value, number + n]
    seconds = sum(b["seconds"] for b in blocks)
    result = dict(
        length=length,
        nominal_payload_patches=length // 8,
        documents=len(documents),
        windows=len(windows),
        valid_target_count=count,
        sum_nll=total,
        nll=total / count,
        perplexity=math.exp(total / count),
        bits_per_byte=total / count / math.log(2),
        seconds=seconds,
        target_bytes_per_second=count / seconds,
        domains=summarize_losses(domains),
        buckets=summarize_losses(buckets),
        memory={
            k: max(b["memory"][k] for b in blocks)
            for k in ("peak_allocated_bytes", "peak_reserved_bytes")
        },
    )
    result["memory"]["minimum_sampled_free_bytes"] = min(
        b["memory"]["minimum_sampled_free_bytes"] for b in blocks
    )
    result["memory"]["process_rss_status"] = blocks[-1]["memory"]["process_rss_status"]
    if result["memory"]["process_rss_status"] == "MEASURED":
        result["memory"]["peak_observed_process_rss_bytes"] = max(
            b["memory"]["peak_observed_process_rss_bytes"] for b in blocks
        )
    first = result["buckets"]["0"]["nll"]
    for bucket in result["buckets"].values():
        bucket["relative_delta_from_first"] = (bucket["nll"] - first) / first
    if length == 32:
        check_baseline(result["nll"], count)
    store.put(name, result)
    print(f"MILESTONE: segmented {length}: NLL={result['nll']:.10f}, targets={count}", flush=True)
    return result


@torch.inference_mode()
def consume(model, payload):
    state = model.start()
    for byte in payload:
        state = model.consume(torch.tensor([byte], device="cuda:0"), state)
    return state


@torch.inference_mode()
def anchors_run(model, anchors, payloads, store):
    for index, anchor in enumerate(anchors):
        name = f"anchor_{index:04d}"
        if store.get(name) is not None:
            continue
        scores = {}
        for length in LENGTHS:
            state = consume(model, anchor_history(anchor, payloads, length))
            logits = model.predict(state)
            nll = -logits.log_softmax(-1)[0, anchor["target"]].item()
            diagnostic = distribution(logits)
            after = model.consume(torch.tensor([32], device="cuda:0"), state)
            scores[str(length)] = dict(
                nll=nll, distribution=diagnostic, after_space=distribution(model.predict(after))
            )
        store.put(name, dict(anchor=anchor, scores=scores))
    print("MILESTONE: matched anchors and held-out distributions complete", flush=True)


@torch.inference_mode()
def generations(model, store):
    historical = read(PRODUCTION / "progress.json")["generation"][-1]["samples"]
    for index, prefix in enumerate(PROTOCOL["generation_prompts_hex"]):
        name = f"generation_{index}"
        if store.get(name) is not None:
            continue
        state = consume(model, bytes.fromhex(prefix))
        steps, output = [], []
        for _ in range(64):
            row = distribution(model.predict(state))
            byte = row["top1_byte"]
            steps.append(row)
            output.append(byte)
            state = model.consume(torch.tensor([byte], device="cuda:0"), state)
        require(bytes(output).hex() == historical[index]["hex"], "historical greedy output changed")
        require(
            state.shared.steps == 1 + (len(bytes.fromhex(prefix)) + 64) // 8,
            "stream clock mismatch",
        )
        store.put(
            name,
            dict(
                prefix_hex=prefix,
                output_hex=bytes(output).hex(),
                escaped=repr(bytes(output)),
                steps=steps,
                shared_steps=state.shared.steps,
            ),
        )
    print("MILESTONE: historical greedy distributions complete", flush=True)


def disposable_probe(model, target):
    clone = copy.deepcopy(model).train()
    require(
        all(
            a.data_ptr() != b.data_ptr()
            for a, b in zip(model.parameters(), clone.parameters(), strict=True)
        ),
        "disposable clone aliases parent",
    )
    torch.cuda.reset_peak_memory_stats()
    mem = [memory()]
    started = time.perf_counter()
    loss, count = raw_byte_nll(clone(target), target)
    torch.cuda.synchronize()
    forward = time.perf_counter() - started
    backward_start = time.perf_counter()
    (loss / count).backward()
    torch.cuda.synchronize()
    backward = time.perf_counter() - backward_start
    finite_gradients(clone, "disposable mechanics")
    gradients = [p.grad for p in clone.parameters() if p.grad is not None]
    norm = torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(g) for g in gradients])
    ).item()
    require(math.isfinite(norm) and norm > 0, "invalid probe gradient norm")
    mem.append(memory())
    result = dict(
        status="PASS",
        weights="DISPOSABLE_UNSAVED",
        optimizer_updates=0,
        logical_production_updates=0,
        nll=loss.item() / count,
        target_count=count,
        gradient_tensors=len(gradients),
        gradient_norm=norm,
        forward_seconds=forward,
        backward_seconds=backward,
        total_seconds=forward + backward,
        forward_bytes_per_second=count / forward,
        backward_bytes_per_second=count / backward,
        step_bytes_per_second=count / (forward + backward),
        memory=memory_summary(mem),
    )
    del clone
    return result


def run():
    saved, resolved, config, manifest, payloads, before, full = parent_inputs()
    require(
        str(torch.__version__) == "2.6.0+cu118" and torch.version.cuda == "11.8", "runtime mismatch"
    )
    driver = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version,name,memory.total", "--format=csv,noheader"],
        text=True,
    ).strip()
    require(driver == read(PRODUCTION / "binding.json")["driver"], "GPU driver/profile mismatch")
    configure_device("cuda:0")
    torch.set_num_threads(config.cpu_threads)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config.seed)
    require(saved["environment"] == environment(config), "checkpoint environment mismatch")
    documents = validation_documents(manifest, payloads)
    anchors = select_anchors(documents)
    binding = dict(
        schema="1",
        source=code_identity(),
        protocol=PROTOCOL,
        tools={p.name: sha(p) for p in (Path(__file__), ROOT / "scripts/context_study_metrics.py")},
        parent_before=before,
        parameter_sha256=PARAMETER_SHA,
        resolved_config=resolved.to_dict(),
        production_training_config=config.to_dict(),
        corpus_manifest_sha256=manifest.sha256,
        split_hashes=full["split_sha256"],
        environment=environment(config),
        anchors=anchors,
        spec_sha256=sha(
            ROOT / "docs/architecture_changes/AC-008_Context_and_Distribution_Promotion_Gates.md"
        ),
    )
    # Dirty metadata may change when tests/reports are added, but code and tools must not.
    binding["source"].pop("dirty")
    store = Store(STUDY, binding)
    model = DenseByteModel(resolved).to("cuda:0")
    model.load_state_dict(saved["model"], strict=True)
    require(sum(p.numel() for p in model.parameters()) == 1929579, "parameter count mismatch")
    model.eval()
    del saved
    try:
        baseline = segmented(model, documents, 32, store)
        check_baseline(baseline["nll"], baseline["valid_target_count"])
        for length in LENGTHS[1:]:
            segmented(model, documents, length, store)
        anchors_run(model, anchors, payloads, store)
        generations(model, store)
        for length in LENGTHS:
            if store.get(f"mechanics_{length}") is None:
                raw = documents[0][1][:length]
                target = torch.tensor([list(raw)] * 2, device="cuda:0")
                store.put(f"mechanics_{length}", disposable_probe(model, target))
        require(tensor_hash(model.state_dict()) == PARAMETER_SHA, "evaluation model changed")
        require(verify_corpus(full) == binding["split_hashes"], "final corpus mismatch")
        verify_inventory(before)
        if store.get("integrity_final") is None:
            store.put(
                "integrity_final",
                dict(
                    parent_after=inventory(),
                    parameter_sha256=PARAMETER_SHA,
                    test="SEALED; integrity hash only",
                    status="PASS",
                ),
            )
        print("MILESTONE: measurements complete; report finalization required", flush=True)
    finally:
        verify_inventory(before)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run()
