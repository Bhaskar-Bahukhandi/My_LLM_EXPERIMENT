import json
import uuid
from pathlib import Path

import pytest
from test_training import model_config

from unified_edge.training.data import DatasetManifest
from unified_edge.training.experiment import run_overfit, run_resume_gate


@pytest.mark.overfit
def test_fixed_overfit_and_post_training_generation():
    root = (
        Path(__file__).resolve().parents[1]
        / "evidence/training_infrastructure"
        / ("overfit-" + uuid.uuid4().hex[:12])
    )
    result = run_overfit(root, model_config())
    manifest = DatasetManifest.from_dict(result["dataset_manifest"])
    resume = run_resume_gate(root / "resume_gate", model_config(), manifest, root / "data")
    with (root / "resume_result.json").open("x", encoding="utf-8") as handle:
        json.dump(resume, handle, indent=2, allow_nan=False)
    assert result["training_nll_reduction_fraction"] >= 0.5
    assert result["checkpoint_restore_weights_and_generation_exact"]
    assert result["gradient_tensors_observed_finite"] == 56
    assert result["generation"]["shared_steps"] == 9
    assert result["trained_causality"]["future_suffix_max_abs"] <= 1e-5
    print("Training evidence directory:", root, flush=True)
