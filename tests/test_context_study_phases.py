"""Phase supplement preserves frozen selection, FP32 support and history sensitivity."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from context_study_phases import BASES, consume_batch, delta, forensic, selections  # noqa: E402
from test_mamba import accepted_config  # noqa: E402

from unified_edge.dense_model import DenseByteModel  # noqa: E402


def test_phase_selection_and_grid():
    anchors = [
        dict(domain=d, key=f"{d}{i:03}", offset=256, target=i, path=d)
        for d in "abcd"
        for i in range(40)
    ]
    selected = selections(list(reversed(anchors)))
    assert len(selected) == 128
    assert selected == selections(anchors)
    for phase in range(8):
        assert all((n + phase) % 8 == phase and n + phase <= 255 for n in BASES)
    anchors[0]["offset"] = 254
    with pytest.raises(ValueError, match="preceding"):
        selections(anchors)


def test_delta_requires_fp32_and_finite():
    a = torch.tensor([1.0, 2.0])
    assert delta(a, a)["max_abs"] == 0
    assert delta(a, a + 1)["max_abs"] == 1
    with pytest.raises(ValueError, match="FP32"):
        delta(a.double(), a.double())
    with pytest.raises(ValueError, match="nonfinite"):
        delta(a, a * float("inf"))


@torch.inference_mode()
def test_untrained_old_patch_can_change_logits_after_identical_32_byte_suffix():
    torch.set_num_threads(1)
    torch.manual_seed(43)
    model = DenseByteModel(accepted_config()).eval()
    a = consume_batch(model, [bytes(8) + bytes(range(32))], "cpu")
    b = consume_batch(model, [bytes([255]) * 8 + bytes(range(32))], "cpu")
    result = forensic(a, b, model.predict(a), model.predict(b), 0, 0)
    assert result["hierarchy"]["pending_equal"]
    assert result["logits"]["max_abs"] > 0
    assert not result["logits"]["equal"]
    assert any(layer["ssm"]["max_abs"] > 0 for layer in result["layers"])
    same = forensic(a, a, model.predict(a), model.predict(a), 0, 0)
    assert same["logits"]["equal"]
    assert same["logits"]["q_total_variation"] == 0
    assert same["logits"]["excluded_masked_ids"] == [256, 257]


def test_phase_paired_statistics_keep_anchor_clusters():
    from context_phase_summary import paired

    rows = []
    for i in range(2):
        for phase in range(8):
            rows.append(
                {
                    "anchor": {"key": str(i)},
                    "phase": phase,
                    "scores": {
                        str(n + phase): float(phase) + (0 if n == 32 else i * 2 - 1) for n in BASES
                    },
                }
            )
    result = paired(rows, aggregate=True)
    assert result["248"]["anchor_count"] == 2
    assert result["248"]["target_phase_count"] == 16
    assert result["248"]["paired_delta"] == 0
    assert result["248"]["paired_delta_ci95"] == [-1, 1]
    assert result["248"]["improved_fraction"] == 0.5
    assert result["248"]["worsened_fraction"] == 0.5
    assert result["32"]["unchanged_fraction"] == 1
    assert result == paired(rows, aggregate=True)
