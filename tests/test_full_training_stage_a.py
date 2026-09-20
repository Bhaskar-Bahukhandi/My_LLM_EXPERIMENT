"""Production byte probes and the actual frozen one-pass boundary."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from full_training_stage_a import PREFIXES, inputs, prompted_generation  # noqa: E402
from real_training_diagnostic import tensor_hash  # noqa: E402

from unified_edge.dense_model import DenseByteModel  # noqa: E402
from unified_edge.training.data import BatchStream, WindowDataset  # noqa: E402
from unified_edge.training.experiment import generate_bytes  # noqa: E402


def test_frozen_schedule_crosses_only_two_windows_into_next_epoch():
    _, _, manifest, _, schedule = inputs()
    from corpus_acquisition import PILOT

    data = WindowDataset(manifest, PILOT, "train", 32)
    stream = BatchStream(data, 17, 2)
    stream.offset = len(data) - 2
    final = stream.next_batch() + stream.next_batch()
    assert schedule["one_pass_total_steps"] == 78167
    assert stream.epoch == 1 and stream.offset == 2
    assert len(final) == 4
    assert schedule["one_pass_target_bytes_including_extra"] == 10000000 + sum(
        len(w.payload) for w in final[-2:]
    )


def test_prefix_probes_condition_real_stream_and_preserve_weights_and_mode():
    _, _, _, resolved, _ = inputs()
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        torch.manual_seed(17)
        model = DenseByteModel(resolved)
        before = tensor_hash(model)
        assert prompted_generation(model, b"")["hex"] == generate_bytes(model)["hex"]
        for prefix in PREFIXES[1:]:
            result = prompted_generation(model, prefix)
            with torch.inference_mode():
                state = model.start()
                for byte in prefix:
                    state = model.consume(torch.tensor([byte]), state)
                expected_first = model.predict(state)[:, :256].argmax(-1).item()
            assert bytes.fromhex(result["hex"])[0] == expected_first
            assert result == prompted_generation(model, prefix)
            assert model.training and all(p.grad is None for p in model.parameters())
        assert tensor_hash(model) == before
    finally:
        torch.set_num_threads(old_threads)
