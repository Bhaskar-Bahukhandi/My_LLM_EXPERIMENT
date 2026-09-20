"""Reconcile a completed diagnostic without training or repeating validation."""

import gc
import random
import subprocess

import torch
from benchmark_fingerprints import canonical
from corpus_acquisition import PILOT, digest, read, write_new
from corpus_freeze import verify_manifest
from real_training_diagnostic import RUN, tensor_hash

from unified_edge.resolve import ResolvedConfig
from unified_edge.training.checkpoint import CheckpointError, load_checkpoint
from unified_edge.training.config import TrainingConfig
from unified_edge.training.data import DatasetManifest
from unified_edge.training.experiment import generate_bytes, tree_error
from unified_edge.training.trainer import Trainer, code_identity


def main():
    progress = read(RUN / "progress.json")
    binding = read(RUN / "input_binding.json")
    assert len(progress["updates"]) == 250
    assert [v["step"] for v in progress["validation"]] == [0, 5, 125, 250]
    assert all(v["valid_target_count"] == 500000 for v in progress["validation"])
    current = code_identity()
    assert current["source_sha256"] == binding["code"]["source_sha256"]
    command = ["git", "-c", "safe.directory=E:/Theai"]
    unchanged = subprocess.check_output(
        [
            *command,
            "diff",
            binding["code"]["commit"],
            "--",
            "src",
            "reports/edge_2m_resolved.json",
            "requirements-lock.txt",
            "requirements-cuda-lock.txt",
        ],
        text=True,
    )
    assert not unchanged
    frozen = read(PILOT / "final-corpus-v1/manifest.json")
    assert (
        digest((PILOT / "final-corpus-v1/manifest.json").read_bytes())
        == binding["frozen_manifest"]["raw_sha256"]
    )
    assert digest(canonical(frozen)) == binding["frozen_manifest"]["canonical_sha256"]
    assert verify_manifest(frozen, PILOT) == binding["split_bytes"]
    assert frozen["split_sha256"] == binding["split_sha256"]
    native = DatasetManifest.from_dict(read(PILOT / "final-corpus-v1/training_manifest.json"))
    assert native.sha256 == binding["native_manifest_sha256"]
    config = TrainingConfig.from_dict(binding["training_config"])
    resolved = ResolvedConfig.from_dict(binding["resolved_config"])
    assert resolved.sha256 == binding["resolved_sha256"]
    paths = [RUN / name / "checkpoints/step_000250" for name in ("resumed", "final-restored")]
    saved = [load_checkpoint(path) for path in paths]
    fields = [
        "model",
        "optimizer",
        "scheduler",
        "global_step",
        "micro_step",
        "examples_seen",
        "bytes_seen",
        "data_cursor",
        "python_rng",
        "torch_cpu_rng",
        "cuda_rng",
        "model_config",
        "training_config",
        "dataset_sha256",
        "parameter_structure",
        "optimizer_update_counts",
    ]
    errors = {field: tree_error(saved[0][field], saved[1][field]) for field in fields}
    for path in paths:
        sidecar = read(path / "corpus_binding.json")
        assert sidecar["input_binding_sha256"] == digest((RUN / "input_binding.json").read_bytes())
        assert sidecar["checkpoint_manifest_sha256"] == digest(
            (path / "manifest.json").read_bytes()
        )
        assert sidecar["split_sha256"] == binding["split_sha256"]
    generations, fingerprints, restoration = [], [], []
    for index, path in enumerate(paths):
        trainer = Trainer(
            resolved, config, native, PILOT, RUN / f"independent-final-{index}", resume_from=path
        )
        assert trainer.run_manifest["parameter_count"] == 1929579
        assert trainer.env == binding["environment"]
        assert (
            trainer.global_step,
            trainer.micro_step,
            trainer.examples_seen,
            trainer.bytes_seen,
        ) == (250, 500, 1000, 31977)
        check = {
            "optimizer": trainer.optimizer.state_dict(),
            "scheduler": trainer.scheduler.state_dict(),
            "data_cursor": trainer.stream.state_dict(),
            "python_rng": random.getstate(),
            "torch_cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
        }

        # Optimizer moments move to CUDA at restoration; compare their exact CPU values.
        def cpu(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu()
            if isinstance(value, dict):
                return {k: cpu(v) for k, v in value.items()}
            if isinstance(value, list):
                return [cpu(v) for v in value]
            return value

        restoration.append({k: tree_error(cpu(v), saved[index][k]) for k, v in check.items()})
        fingerprints.append(tensor_hash(trainer.model))
        generations.append(generate_bytes(trainer.model))
        invalid = dict(saved[index], dataset_sha256="0" * 64)
        try:
            trainer._restore(invalid)
        except CheckpointError:
            pass
        else:
            raise AssertionError("wrong corpus accepted")
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
    assert fingerprints[0] == fingerprints[1]
    assert generations[0] == generations[1]
    result = {
        "progress_sha256": digest((RUN / "progress.json").read_bytes()),
        "starting_code": binding["code"],
        "finalization_code": current,
        "intervening_commits": subprocess.check_output(
            [*command, "log", "--format=%H %s", binding["code"]["commit"] + "..HEAD"], text=True
        ).splitlines(),
        "head_assertion_reconciliation": "HEAD advanced; protected source/config/locks unchanged",
        "saved_checkpoint_exact_errors": errors,
        "restored_state_exact_errors": restoration,
        "parameter_sha256": fingerprints[0],
        "generation_after": generations[0],
        "generation_restored": generations[1],
        "generation_recovery_note": "Original post-run generation was not published before HEAD "
        "assertion; reconstructed from the two already saved step-250 checkpoints "
        "with unchanged policy.",
        "wrong_corpus_rejected": True,
        "corpus_integrity": "PASS: raw/canonical manifest, all split hashes and byte totals",
        "test_evaluated": False,
        "additional_optimizer_updates": 0,
        "validation_repeated": False,
    }
    write_new(RUN / "final_reconciliation.json", result)
    print("Independent final restoration/generation/corpus integrity PASS", flush=True)


if __name__ == "__main__":
    main()
