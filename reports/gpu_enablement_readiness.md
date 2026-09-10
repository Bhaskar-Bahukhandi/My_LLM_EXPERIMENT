# GPU enablement readiness

The Windows CUDA FP32 tranche passes its bounded gates and is ready for separately authorized small real-data acquisition. No real training data was downloaded. This is execution, restart and short synthetic learning evidence; it is not language-quality or long-training evidence.

## Rollback and scope

Starting approved CPU training commit: **b7eb74d1f9121e40e149d66c65d20e04113c31bb**. The complete approved diff matched its reviewed hashes; all 159 tests, including the fixed 60-update CPU overfit, passed again before commit (143.750 s), together with Ruff, formatting, compile/import and dependency checks. Ledger-only commit **0bc5a6c1c7e2bcaeccb6a0a3759c743a92078176** records the exact SHA without amending history. The working tree was clean before GPU work. Prior Mamba `ddc8306a6d8ba440e259f4c15a6d61aa42f5ce7b`, byte and configuration commits remain ancestors; no remote was added. GPU changes remain uncommitted for review.

The model remains **1,929,579 parameters**: byte hierarchy 201,483 and shared Mamba 1,728,096. CPU/CUDA have identical 56 parameter names/shapes and no device-specific trainable additions. No-weight-decay metadata survives placement. Resolved architecture and model-state schemas remain unchanged. AST comparison confirms that only three Mamba validation methods changed; initialization, recurrence, projection and output equations are identical to the approved CPU source. Byte hierarchy, dense wrapper, optimizer/objective and resolver implementations are unchanged.

The inventory hardware still describes the historical CPU lower-bound planning profile. Actual runtime device is explicit in TrainingConfig, run/checkpoint environment and this measured report; inventory metadata alone does not prove GPU fit. Both FINAL specifications and AC-001..005 remain unchanged. Mamba v2.2.6 / `d7b1ceb3c367ec9022925e812f507bcf706937c6` remains the engineering semantic-reference pin, not a Bible requirement.

## Hardware and reproducible environment

- NVIDIA GeForce RTX 2050; nvidia-smi reports **4,096 MiB**, WDDM, driver **596.21**, compute capability **8.6**.
- Windows 11, build 10.0.26200; Python **3.12.14**.
- Before installation: `.venv` had torch **2.6.0+cpu**, no CUDA runtime, CUDA availability false and zero CUDA devices. Driver visibility alone was not treated as execution proof.
- Driver nvidia-smi header advertises CUDA **13.2** compatibility. The selected PyTorch wheel actually bundles runtime **11.8**, with cuDNN API version **90100** and compiled `sm_86` support.
- `.venv-cuda` now has official **torch 2.6.0+cu118**. A real FP32 CUDA matrix operation passed, followed by actual model gates. `.venv` remains unchanged and independently tested.
- Torch CUDA reports total device memory **4,294,443,008 bytes**; this is recorded separately from nvidia-smi's 4,096 MiB presentation.

Wheel: [torch-2.6.0%2Bcu118-cp312-cp312-win_amd64.whl](https://download-r2.pytorch.org/whl/cu118/torch-2.6.0%2Bcu118-cp312-cp312-win_amd64.whl). SHA-256: `6c040e4181c5dae73b965b61394ec431c93b2018165e2be8f15fc68d44444cb3`. The [official PyTorch previous-version instructions](https://pytorch.org/get-started/previous-versions/) list cu118, cu124 and cu126 for 2.6.0 Windows. The conservative cu118 build preserves the API version, supports this device, and has now executed successfully on the installed driver; [NVIDIA compatibility guidance](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html) describes driver backward compatibility.

Only the torch build differs between `requirements-lock.txt` and `requirements-cuda-lock.txt`; all 17 pinned packages match their respective environments. Both dependency checks and all 20 package imports pass. No Mamba-ssm, causal-conv1d, Triton, unofficial binary, CUDA toolkit, WSL or additional framework was installed. Native Windows uses the project's own generic PyTorch Mamba implementation.

Recreation instructions are in README.md. Create an absent `.venv-cuda` with Python 3.12.14, install `requirements-cuda-lock.txt` using the official cu118 index, install this project with `--no-build-isolation --no-deps -e .`, and run the dedicated CUDA suite. Do not overwrite the existing CPU environment. The saved pip report provides exact download identities, and reports/gpu_enablement_evidence.json includes both profile inventories.

## Runtime and checkpoint semantics

The system boundary initializes model weights on CPU, moves them once with `.to(device)`, then constructs AdamW. All input transfers are explicit at trainer/caller boundaries. Recurrent states, BOS, masks and temporaries derive placement from the owning weights/input; reusable model logic does not insert silent CPU transfers. FP32 remains enforced.

The profile requires exactly one visible GPU and `cuda:0`. TF32 is disabled for matmul/cuDNN; cuDNN benchmarking is off and deterministic execution is enabled. `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set before CUDA initialization. The PyTorch allocator is capped at **1,073,741,824 bytes (1 GiB)**; admission requires that budget plus **1 GiB** of free-device headroom. Other applications and non-allocator memory remain outside that cap, so measured free memory is also reported.

Training checkpoint payload schema **2** adds all visible CUDA RNG states alongside Python/CPU RNG, model, optimizer, scheduler, counters, data order/cursor and strict identities. The integrity manifest and model snapshot schemas remain **1**. Atomic publication and checksums are preserved. Historical schema-1 training checkpoints require the approved CPU rollback source/environment; there is no silent migration or cross-device bitwise-resume claim. The CUDA RNG malformed-state rejection test passes.

## Numerical parity and causality

Predeclared, unchanged gate: **atol=1e-5, rtol=1e-4**. Relative error uses denominator `max(abs(reference), 1e-12)`. Seeds/batches were (17, 1) and (29, 2). Dense lengths: 1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 63, 64, 65, 127, 128 and 129, plus meaningful empty/BOS behavior. Snapshot split points include 7, 16 and 65. Shared full/step and nonzero-state continuation use 17 shared positions split at 5.

| Comparison | Max absolute error | Max relative error |
|---|---:|---:|
| Initial/BOS logits: CPU vs CUDA | 1.221895218e-06 | 0.001123332553 |
| Dense logits: CPU vs CUDA | 2.264976501e-06 | 0.05435591958 |
| Dense CUDA full vs incremental | 2.11596489e-06 | 0.03689469639 |
| Local hidden state: CPU vs CUDA | 1.706182957e-06 | 0.007526817993 |
| Incremental convolution state: CPU vs CUDA | 1.966953278e-06 | 0.02811550152 |
| Incremental SSM state: CPU vs CUDA | 2.086162567e-07 | 0.4285714286 |
| Shared full output: CPU vs CUDA | 2.62260437e-06 | 0.009195402299 |
| Shared CUDA full vs step | 2.026557922e-06 | 0.02267573696 |
| Shared CUDA nonzero-state continuation | 1.907348633e-06 | 0.0162412993 |
| Full convolution state: CPU vs CUDA | 1.072883606e-06 | 0.003314917127 |
| Full SSM state: CPU vs CUDA | 7.450580597e-08 | 0.07089638807 |
| CUDA convolution continuation state | 8.344650269e-07 | 0.005543237251 |
| CUDA SSM continuation state | 1.11758709e-07 | 0.08497936691 |
| CUDA full/step convolution state | 9.834766388e-07 | 0.002734731085 |
| CUDA full/step SSM state | 1.043081284e-07 | 0.0333703201 |

All comparisons pass the combined tolerance. Large relative maxima near zero are retained honestly: the largest relative value is 0.4285714286 in a small SSM value, while the maximum absolute error across all measured categories is only **2.6226043701171875e-6**. Absolute and relative maxima need not occur at the same element. No tolerance was widened.

Incremental convolution/SSM and local hidden states agree within tolerance; pending bytes and shared/completed-patch clocks match exactly. Independent full and incremental code paths remain separate. GPU snapshot export/restore reproduces same-path predictions; a CPU-state snapshot is rejected on GPU until an explicit caller transfer.

Adversarial GPU causality uses batch 2 and 137 bytes, mutating future suffixes at 0, 1, 7, 8, 9, 15, 16, 17, 63, 64, 65, 127, 128 and 129. All unaffected prediction-prefix differences are **0.0** absolute and relative.

## Gradients, AdamW and exact CUDA resume

Real next-byte loss/backward produces finite, positive-norm gradients for all **56** trainable tensors; minimum observed tensor norm is **3.4833105019060895e-5**. Real CUDA AdamW updates complete and parameters remain finite. Parameter identity grouping and tagged no-decay placement pass. The existing loss, accumulation, clipping and finite checks are retained.

Six uninterrupted CUDA optimizer updates were compared with three updates, atomic checkpoint, destruction/recreation and three resumed updates. Python, CPU and CUDA RNG streams are deliberately advanced before the save boundary. Final model/optimizer tensors, scheduler, cursor, next batch, counters, RNG and semantic metrics match **exactly; maximum tensor error 0.0**. Generated bytes match exactly. Only elapsed time and run identity/creation time are excluded from semantic equality. No CPU/CUDA bitwise training equivalence is asserted.

## Short CUDA learning

Unchanged tiny fixture: training `b"edge400: abcdef\n" * 64` (1,024 bytes) and held-out `b"edge401: uvwxyz\n" * 16` (256 bytes). No shared training/validation windows. Config: seed 17, one CPU thread, CUDA FP32, 32-byte windows, microbatch 2, accumulation 2, **10 optimizer updates**, AdamW LR .003, warmup 2, cosine minimum ratio .1, decay .01 and clip norm 1. The predeclared criterion was decreasing whole-training-split NLL, not the CPU 60-update loss target.

- Initial train NLL **5.555967330932617**; final **0.5132922530174255**; reduction **90.76142420493202%**.
- Final raw-byte train perplexity **1.670782789806965**; held-out NLL **3.287210464477539**, perplexity **26.768088876146386**. Held-out quality is informational only.
- Counts: **10 updates, 20 microsteps, 40 examples, 1,280 raw training bytes**. Trainer elapsed through the last update was **6.791681000002427 s**, including initial validation; this is not an isolated optimizer benchmark.
- Actual incremental bytes-only greedy generation emitted `b"edge400: abcdef\n" * 4`: 64 valid bytes, eight completed patches, nine shared steps including BOS; unrestricted head control argmax count zero.
- Trained checkpoint save/destroy/recreate/restore reproduced weights, whole-train NLL and generation exactly. The restored trainer saved another valid checkpoint.

These results establish short learning mechanics and fixture memorization only.

## Synchronized CUDA memory

The following **batch-2, length-32 baseline** lifecycle uses synchronize plus PyTorch allocated/reserved/peak APIs and driver free-memory queries. All quantities are bytes.

| Stage | Allocated | Reserved | Peak allocated | Peak reserved | Device free |
|---|---:|---:|---:|---:|---:|
| before_model | 33,554,432 | 33,554,432 | 33,554,432 | 33,554,432 | 3,411,751,732 |
| after_model_load | 41,280,000 | 58,720,256 | 41,280,000 | 58,720,256 | 3,386,585,908 |
| after_recurrent_state | 42,410,496 | 58,720,256 | 42,410,496 | 58,720,256 | 3,386,585,908 |
| after_forward | 44,061,184 | 62,914,560 | 46,244,864 | 62,914,560 | 3,382,391,604 |
| after_backward | 82,629,632 | 96,468,992 | 84,708,352 | 96,468,992 | 3,338,351,412 |
| after_optimizer_update | 98,080,768 | 123,731,968 | 101,683,200 | 123,731,968 | 3,311,088,436 |

Standalone canonical shared recurrent allocation is **1,130,496 bytes** at batch 2. It is separately created for the state check, not a persistent training cache.

During the 10-update run, peak allocated was **103,754,752**, peak reserved **127,926,272**, and minimum observed free-device memory **3,306,894,132 bytes**. At same-device checkpoint restoration, allocated/peak allocated were **90,285,568**, reserved/peak reserved **98,566,144**, and free memory **3,336,254,260** bytes. The trained-checkpoint restore had allocated **90,285,568**, reserved **127,926,272**; its peak counters include the preceding short-learning segment. Full records are in JSON.

These are process-global PyTorch allocator observations, not total process GPU memory or host RSS. They include library workspaces and any allocations already live at each before-model sample. Sequential probes are not fresh isolated processes; for example, the first matrix probe begins with 90,285,568 allocated bytes, whereas later probes begin with 67,108,864. No baseline subtraction is used to imply model-only usage. Allocation peaks include autograd/activations but do not separately attribute them. WDDM nvidia-smi process memory is N/A; it is not substituted for the allocator measurements. Free-memory values are observations, not guarantees against future desktop/application use.

## Fixed GPU-fit matrix

Each row uses the existing resolver and unchanged architecture family, FP32, batch 1, 32-byte windows, and exactly one synthetic optimizer update. No profile was resized after failure. The 1 GiB allocator safety budget applies throughout. Larger rows are preflight mechanics only.

| Target | Actual parameters | Width/layers | Batch/bytes | Forward/backward/update | Peak allocated | Peak reserved | Minimum device free | Classification |
|---:|---:|---:|---:|---|---:|---:|---:|---|
| 2,000,000 | 1,929,579 | 256/4 | 1 / 32 | PASS / PASS / PASS | 124,825,600 | 132,120,576 | 3,302,699,828 | FITS_TRAINING_MECHANICS |
| 6,000,000 | 5,899,491 | 384/6 | 1 / 32 | PASS / PASS / PASS | 171,424,256 | 180,355,072 | 3,254,465,332 | FITS_TRAINING_MECHANICS |
| 20,000,000 | 20,114,763 | 512/12 | 1 / 32 | PASS / PASS / PASS | 399,809,024 | 438,304,768 | 2,996,515,636 | FITS_TRAINING_MECHANICS |

No actual OOM occurred. A CPU-runnable injected CUDA OutOfMemoryError test verifies `DOES_NOT_FIT`, failure-stage reporting, and preservation of the requested batch/sequence. Hardware OOM was not deliberately induced. Unavailable or budget-rejected candidates return `BLOCKED`; unexpected execution errors propagate. Larger-model one-step fit does not establish sustained training, context-4096 capacity or permission to train 20M.

Recommended current profile: **1,929,579 parameters, CUDA FP32, 32-byte sequences, batch 2, accumulation 2**, deterministic settings above, 1 GiB allocator cap. The measured short-learning profile retained over 3.3 billion free bytes and is well inside the declared headroom gate. The 5.90M/20.11M rows are not promoted training profiles. The full generic SSD path remains quadratic in shared sequence length; no 4096-byte CUDA training or mixed precision was attempted.

## Validation, changed files and remaining limits

- Preserved CPU `.venv`: **161 passed, 7 explicit CUDA skips**, zero failures/errors, XML duration 166.135 s. This includes all unchanged 159 accepted CPU tests and two new CPU-runnable device/OOM tests.
- Dedicated `.venv-cuda` suite: **7 passed, zero skips/failures/errors**, 55.178 s. Across the two profiles all 168 unique cases passed in their applicable environments.
- Ruff, formatting (33 Python files), compileall, both dependency checks and 20 imports per profile passed. Exact 17-package lock inventories are preserved.
- Complete tracked/new-file diff, whitespace, source/evidence identity, artifact exclusions, unchanged math/specifications and rollback history were reviewed. Review artifacts are retained under `evidence/gpu_enablement/`.

Changed files:

- `.gitignore`
- `README.md`
- `docs/gpu_execution_contract.md`
- `docs/project_state.md`
- `docs/training_contract.md`
- `pyproject.toml`
- `reports/gpu_enablement_evidence.json`
- `reports/gpu_enablement_readiness.md`
- `requirements-cuda-lock.txt`
- `src/unified_edge/mamba.py`
- `src/unified_edge/training/checkpoint.py`
- `src/unified_edge/training/config.py`
- `src/unified_edge/training/device.py`
- `src/unified_edge/training/gpu_probe.py`
- `src/unified_edge/training/memory.py`
- `src/unified_edge/training/trainer.py`
- `tests/test_gpu_enablement.py`
- `tests/test_gpu_profile.py`

The optional NumPy bridge remains absent with a visible warning; used operations pass. Upstream optimized Mamba runtime parity, reduced precision, cross-device/version bitwise resume, long-context CUDA training, sustained thermals/power/performance and substantive data quality are unvalidated. The CPU fallback and all rollback commits remain available. No new architectural conflict or remaining blocker for this bounded CUDA tranche was found.

Next best step: review these GPU changes, then separately authorize SMALL real-data acquisition using the recommended baseline. No real-data acquisition or later-stage implementation has started.

GPU ENABLEMENT STATUS: READY FOR SMALL REAL-DATA ACQUISITION
