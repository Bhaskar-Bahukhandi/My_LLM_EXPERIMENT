# GPU enablement: compact evidence closeout

Existing status: **READY FOR SMALL REAL-DATA ACQUISITION**. This closeout reuses the passing measurements; no experiments, installs or model changes were repeated. Detailed evidence: [JSON](gpu_enablement_evidence.json), [readiness report](gpu_enablement_readiness.md).

Evidence SHA-256: `05a9319f8fdbddf65db336818e0336df660d79d9a403092988ec0cc1f1fe42c5`. Tested source SHA-256: `7d98de1b0504a1d511f3da95e17ec45dbcca1c5bd2d0dadbd251987261173eb4`; source commit `0bc5a6c1c7e2bcaeccb6a0a3759c743a92078176`, dirty at measurement. Approved CPU rollback: `b7eb74d1f9121e40e149d66c65d20e04113c31bb`. GPU work remains uncommitted for review.

## Execution and reproducibility

RTX 2050; nvidia-smi VRAM **4,096 MiB**; CUDA API total **4,294,443,008 bytes**; driver **596.21**; compute capability **8.6**. Windows 11 build 10.0.26200. Python **3.12.14**, PyTorch **2.6.0+cu118**, `torch.version.cuda=11.8`, cuDNN API **90100**. The driver header CUDA 13.2 does not identify the wheel runtime.

Executable parameters: **1,929,579** = byte hierarchy **201,483** + shared Mamba **1,728,096**, 56 unique trainable tensors; CPU/CUDA identities match. Architecture and accepted smoke-size exception unchanged.

Recreate the separate `.venv-cuda` using [README](../README.md) and [CUDA lockfile](../requirements-cuda-lock.txt). Official index: <https://download.pytorch.org/whl/cu118>. Only torch differs from the CPU lock (other 16 packages unchanged). No environment, wheel or kernel package is part of this review bundle. Wheel SHA-256: `6c040e4181c5dae73b965b61394ec431c93b2018165e2be8f15fc68d44444cb3`. Exact wheel URL and lock hash are in the companion JSON.

## Numerical gates

All comparisons pass the unchanged combined `atol=1e-5, rtol=1e-4` gate. Relative error uses denominator `max(abs(reference),1e-12)`; near-zero values explain large relative maxima. Absolute and relative maxima can occur at different elements.

| Comparison | Max absolute | Max relative |
|---|---:|---:|
| initial_logits | 1.2218952178955078e-06 | 0.0011233325532412825 |
| cpu_gpu_logits | 2.2649765014648438e-06 | 0.054355919583023084 |
| gpu_full_step | 2.115964889526367e-06 | 0.03689469638739431 |
| hidden | 1.7061829566955566e-06 | 0.0075268179932436435 |
| layers/conv | 1.9669532775878906e-06 | 0.02811550151975684 |
| layers/ssm | 2.086162567138672e-07 | 0.42857142857142855 |
| shared_cpu_gpu | 2.6226043701171875e-06 | 0.009195402298850575 |
| shared_full_step | 2.0265579223632812e-06 | 0.022675736961451247 |
| shared_nonzero_state_continuation | 1.9073486328125e-06 | 0.016241299303944315 |
| full_state_cpu_gpu/conv | 1.0728836059570312e-06 | 0.0033149171270718232 |
| full_state_cpu_gpu/ssm | 7.450580596923828e-08 | 0.0708963880696047 |
| full_state_continuation/conv | 8.344650268554688e-07 | 0.005543237250554324 |
| full_state_continuation/ssm | 1.1175870895385742e-07 | 0.08497936691422628 |
| full_state_step/conv | 9.834766387939453e-07 | 0.0027347310847766638 |
| full_state_step/ssm | 1.043081283569336e-07 | 0.03337032010392093 |

Seeds/batches (17,1), (29,2); lengths through 129 plus BOS/empty behavior, shared sequence and nonzero-state continuation. Adversarial suffix tests: batch 2, length 137, cuts 0,1,7,8,9,15,16,17,63,64,65,127,128,129. Maximum prefix difference **0.0 absolute and relative**.

All 56 gradient tensors finite with positive norms (minimum **3.4833105019060895e-05**); real CUDA AdamW update **PASS**. Checkpoint schema 2 restores Python/CPU/CUDA RNG and rejects malformed CUDA RNG. Six uninterrupted updates versus three + save/destroy/recreate/restore + three match parameters, optimizer, scheduler, RNG, counters, next batch/cursor and semantic metrics **exactly; max tensor error 0.0**. Timing and run identity are observational exclusions; cross-device bitwise resume is not claimed.

## Learning and memory

Ten synthetic CUDA updates: train NLL **5.555967330932617 -> 0.5132922530174255**, reduction **90.76142420493201%**; held-out NLL **3.287210464477539**. 20 microsteps, 40 examples, 1,280 bytes; last update elapsed **6.791681000002427 s**, including initial validation. Real greedy incremental generation produces `edge400: abcdef\n` four times (64 bytes), eight completed patches/nine shared steps. Weights, NLL and generation reproduce after trained checkpoint restoration. This proves mechanics only.

All following values are **bytes**, synchronized CUDA allocator/driver measurements. Peak counters cover each recorded probe boundary.

| Stage (1.93M, B2, length32) | Allocated | Reserved | Peak allocated | Peak reserved | Driver free | Driver total |
|---|---:|---:|---:|---:|---:|---:|
| before_model | 33554432 | 33554432 | 33554432 | 33554432 | 3411751732 | 4294443008 |
| after_model_load | 41280000 | 58720256 | 41280000 | 58720256 | 3386585908 | 4294443008 |
| after_recurrent_state | 42410496 | 58720256 | 42410496 | 58720256 | 3386585908 | 4294443008 |
| after_forward | 44061184 | 62914560 | 46244864 | 62914560 | 3382391604 | 4294443008 |
| after_backward | 82629632 | 96468992 | 84708352 | 96468992 | 3338351412 | 4294443008 |
| after_optimizer_update | 98080768 | 123731968 | 101683200 | 123731968 | 3311088436 | 4294443008 |

Short-learning peaks: allocated **103,754,752**, reserved **127,926,272**; minimum measured free **3,306,894,132**. Resume restore: allocated **90,285,568**, reserved **98,566,144**, free **3,336,254,260**, total **4,294,443,008**.

Observed host process RSS maximum during learning: **898670592**; process peak working set **989241344**, from Windows GetProcessMemoryInfo. These are host-memory observations, not VRAM.

## Fixed fit matrix

| Target | Actual parameters | Batch | Length | Precision | Forward | Backward | AdamW | Peak allocated | Peak reserved | Classification |
|---:|---:|---:|---:|---|---|---|---|---:|---:|---|
| 2000000 | 1929579 | 1 | 32 | fp32 | PASS | PASS | PASS | 124825600 | 132120576 | FITS_TRAINING_MECHANICS |
| 6000000 | 5899491 | 1 | 32 | fp32 | PASS | PASS | PASS | 171424256 | 180355072 | FITS_TRAINING_MECHANICS |
| 20000000 | 20114763 | 1 | 32 | fp32 | PASS | PASS | PASS | 399809024 | 438304768 | FITS_TRAINING_MECHANICS |

**All attempted candidates are shown; actual OOM count 0; no profile was resized.** One forward/backward/update establishes bounded training mechanics, not sustained or full-training feasibility. Maximum recorded peak across these gates: **399,809,024 allocated / 438,304,768 reserved**, from the 20M preflight; minimum driver free there **2,996,515,636** bytes. This does not authorize 20M training.

Allocator totals include process-global workspaces and prior live allocations; sequential probes are not fresh isolated processes. No model-only subtraction is made. Activations/autograd are included but not separately attributed. WDDM process GPU memory is **UNVERIFIED (N/A)**. Driver free memory can change with other applications.

## Safe next profile and decision

Use the accepted 1.93M model, CUDA FP32, **batch 2, sequence 32 bytes, accumulation 2**, one CPU thread; deterministic algorithms/cuDNN, TF32 and cuDNN benchmarking off, `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Enforce **1 GiB allocator cap + 1 GiB free headroom** on admission. Measured short-run headroom passes; sustained throughput, thermals, longer sequences, mixed precision and optimized kernels remain unvalidated. Generic full Mamba reference still uses O(T^2) transition storage.

Existing validation: CPU **161 passed / 7 explicit CUDA skips** (166.135 s); CUDA **7 passed** (55.178 s); **168 unique applicable cases**. Ruff, formatting (33 files), compile/import and both dependency checks passed. No required GPU gate is missing. Optional NumPy warning remains visible. The CPU fallback and all rollback commits remain intact. Next action is human review of the acquisition proposal; no corpus acquisition or training was performed here.

GPU ENABLEMENT STATUS: READY FOR SMALL REAL-DATA ACQUISITION
