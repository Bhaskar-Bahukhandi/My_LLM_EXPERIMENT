# First real-data diagnostic: ready for human training review

The accepted **1,929,579-parameter** model completed the fixed **250-update** diagnostic on the frozen real corpus. Training and full-validation NLL improved with finite gradients and exact CUDA restart evidence. This authorizes review only; full training has not started.

## Identities and fixed profile

Starting commit: `720af6e63ee8759654b287848173bbd3e74867c4` with the expected recorded dirty corpus working tree. Finalization inspected HEAD `2bacf40f26a2e6ac0d1668618ec89496255cde12`. Intervening commits preserved model source, resolved configuration and dependency locks.

Model-source SHA-256: `7d98de1b0504a1d511f3da95e17ec45dbcca1c5bd2d0dadbd251987261173eb4`. Resolved-config SHA-256: `7fe06b639741003e4c5ad9d57d32c2fd2b86f6805b2048ad1efd6aaa75e3a6f1`. Initial parameter checksum: `19a0bc8848a8362271655061e82b2d7809b8e95eacbd069b547009d01cccb7e2`; trained checksum: `db6c7bd5f2e823e5a9d7d0a1610bec6989632270e2d885355906d2709c62da88`.

RTX 2050, compute capability 8.6, driver 596.21; Python 3.12.14; PyTorch 2.6.0+cu118 / CUDA 11.8. FP32, seed 17, one CPU thread, 32-byte windows, microbatch 2, accumulation 2: at most 128 target bytes/update. Actual short document tails are counted. Deterministic algorithms enabled, TF32 disabled, cuDNN benchmarking off, cuBLAS workspace :4096:8. No environment or architecture changes.

AdamW LR 0.003, five warmup updates then cosine to ratio 0.1; betas (0.9, 0.999), epsilon 1e-8; clip norm 1. Decay group: 22 tensors / 1,921,792 parameters at 0.01. No-decay group: 34 tensors / 7,787 parameters at zero, including tagged Mamba parameters. Identity grouping and optimizer semantics are unchanged. Checkpoints at 3, 5, 125 and 250; replay branches are separately identified.

## Frozen corpus

Manifest SHA-256: `38bbb7266544129b8c1bd14d2e744f233208d55c403135244963024fad6ed86b`. Canonical manifest: `f5f375b0c7b446a39000f5a3fb693dbe111d9621b6818a52a3eea339b8a29ecc`.

| Split | Bytes | SHA-256 |
|---|---:|---|
| train | 10,000,000 | `5204262762bf2c54173d0c60e1cfb82d0d851c5a73f868de32d490d44557c120` |
| validation | 500,000 | `70f52e5d362f4425182b0ad058163e56ac8987971f1db61dde1a224ea59a8614` |
| test | 500,000 | `a3f7bc2b3fa3853a6784aff9e7da78bcf2e3fc535dc15d0ba6cfa1a00f44173f` |

All per-document bytes, split hashes and raw/canonical manifest identities reverified after training. Test was read only for integrity hashing; no test loss, selection or prompt tuning occurred. Native trainer manifest contains train/validation only.

## Full-validation trajectory

Every point uses the same deterministic normal validation path and all 500,000 valid target bytes. NLL is byte-target cross-entropy; perplexity is exp(NLL).

| Update | NLL | Perplexity | Target bytes | Seconds | Peak allocated / reserved bytes |
|---:|---:|---:|---:|---:|---:|
| 0 | 5.6641971744 | 288.356388 | 500,000 | 507.835 | 44,202,496 / 60,817,408 |
| 5 | 4.1410332130 | 62.867744 | 500,000 | 529.913 | 93,208,064 / 127,926,272 |
| 125 | 2.8788662083 | 17.794087 | 500,000 | 415.528 | 93,208,064 / 127,926,272 |
| 250 | 2.7588319022 | 15.781398 | 500,000 | 508.719 | 93,208,064 / 127,926,272 |

Initial-to-final NLL reduction: **2.905365 (51.294%)**. Final is the best recorded point. Update 250 improves on 125 by **0.120034**; no late validation regression is hidden.

## Train trajectory and cost

Initial instantaneous train NLL **5.646677**; final instantaneous **2.874083**. First/final 20-update means: **3.926459 / 2.787159**, reduction **1.139300 (29.016%)**. Moving averages are unweighted means of update NLLs; all per-update target counts are retained. The final moving mean rose from 2.748015 at update 225 to 2.787159 at 250.

Logical run: **250 updates, 500 microsteps, 1,000 windows, 31,977 target bytes** (0.31977% of training-corpus bytes, no full epoch). Authorized two-update replay adds 8 windows and 256 bytes of physical computation, for 252 updates / 32,233 bytes including replay. No new diagnostic update occurred after logical step 250.

Measured update time: **144.004 seconds**, **222.057 target bytes/second**, excluding evaluation. Completed validation passes total **1961.995 seconds**. This is not uninterrupted wall time across the interrupted sessions.

## Gradients, memory and restart proof

All **56 trainable tensors** received finite, nonzero gradients in the actual real-data smoke. No missing or unexpected zero-gradient tensors. Pre-clip norms are recorded per update; the measured post-clip global norm passed the threshold. Loss, raw-byte logits, parameters, optimizer moments and placement checks passed; no OOM or CPU fallback occurred. AdamW scalar step counters retain their normal CPU representation.

Peak CUDA allocated/reserved: **108,640,768 / 127,926,272 bytes**. Minimum sampled driver-free memory: **3,306,894,132 bytes**, with device total 4,294,443,008. The 1 GiB allocator cap and 1 GiB free-headroom gate remained satisfied. JSON contains synchronized model-load, forward, backward/clip, optimizer and all-update measurements, plus host RSS where measured. Peaks include workspaces and autograd; they do not separately attribute activation memory.

Real-data checkpoint 3 → update 5 uninterrupted versus restored continuation matched exactly: model, AdamW state/update counters, schedule, data order/cursor, counters, Python/CPU/CUDA RNG and semantic metrics; maximum error 0.0. Elapsed time and run identity are excluded by the accepted contract. Independent serialized comparison also returned zero for every compared field.

Both saved step-250 checkpoints were independently loaded under the unchanged configuration. Parameter count/checksum, optimizer/scheduler/cursor/counters and all RNG states matched exactly. Wrong-corpus identity was rejected before an update. Checkpoint integrity manifests and corpus-binding sidecars bind the immutable input identity. No optimizer update or validation replay was used for this recovery.

## Generation and limitations

Same empty/BOS prompt, greedy raw-byte classes 0..255, 64 bytes; arbitrary bytes preserved and escaped. Both paths advance eight completed patches and nine shared steps.

Before training:

```text
b'\xbf\xf0\xbf\xf0\xf0\xbb\x8e\r2\xff\x98\xc3\xc3\xc3\xc3\xc3E2\xae\xaaB\xe8\xe1\xae\xf0\xf0\xf0\x1c\xa3\xe0NW2\xff\x98\xc3B\x0cT\xdc\xd3\xf0\xe4\xf0B\x8a\xc6\xdc22\x1a2\xe9\xef\x13\r\xf1\xff\x98}B\xd1\xb9\r'
```

After step 250 and independent restored step 250: **64 ASCII spaces** (`0x20` repeated 64 times), byte-for-byte identical. The original unsaved post-run sample was reconstructed from the two saved checkpoints after the reporting interruption; no missing output was invented.

This is degenerate greedy generation, not useful language capability. It is a warning for the human training review, despite a genuine real-data NLL learning signal. Long-run stability, full pretraining, longer-context behavior and benchmark/test quality remain unvalidated.

## Validation, recovery and boundary

19 relevant CPU tests passed (24.10s); two selected real-CUDA placement/parameter and exact resume tests passed (16.97s), five unrelated GPU cases deselected. Ruff, formatting (70 files), compile and both dependency checks pass. Both test profiles retain the optional unused NumPy bridge warning. Corpus acquisition/dedup/contamination and the 250-update trajectory were not rerun.

The original pre-update process disappeared without a completed evaluation/update; recovery proved identical initialization. Later, HEAD advanced through `f770e29` and `2bacf40`; the runner saved final validation and a restored checkpoint but stopped at its old-HEAD assertion. The read-only finalizer reconciled that change through exact source/config/lock and corpus checks. Original progress and interruption evidence remain preserved.

Changed files for this finalization: `scripts/finish_real_training_diagnostic.py`, this report, its JSON evidence and `docs/project_state.md`. Original runner/history remain intact; generated data/checkpoints remain ignored. No model, optimizer, data-mixture or CUDA changes.

Recommendation: **ready for human full-2M training review**, with explicit review of the all-space greedy sample and very small short-window exposure before approving a full schedule. Do not launch full training, evaluate test, implement RSI or scale to 20M.

REAL-DATA DIAGNOSTIC STATUS: READY FOR FULL 2M TRAINING REVIEW
