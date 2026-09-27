# 20M-I2 training infrastructure

Decision: **20M_I2_READY_FOR_REVIEW**. R4 **PASS**, R5 **PASS**, R6 **PASS**, R7 **PASS**,
R8 **PASS**. R1-R3 remain PASS. **R9 NOT IMPLEMENTED. 20M PILOT NOT AUTHORIZED.**
20M training NOT RUN; T4 hardware UNVERIFIED; Kaggle compute NOT RUN; TEST SEALED;
2M FROZEN. Infrastructure readiness does not certify hardware, a pilot or quality.

Entry: `f9914a84cb2b103b143052ea9ffb7bc0fa18aea6`; clean local master and live origin/master matched.
Implementation: `c096765ff71e728701af3a248e27a6674ee25e1e`. Additive closeout binds the final evidence and current-state
links. Initial I2 source changes were confined to training infrastructure. The narrow
readiness correction below changes status metadata only; architecture and resolver
selection/serialization semantics remain unchanged.

## Narrow closeout reconciliation (2026-09-28)

Issue: **STALE_MACHINE_READINESS_AFTER_I2**. Confirmed at
`a853d0ad4a16161ecdb784eb963a413957f4d9ee`: machine readiness still requested R4-R9
and cited only pre-build/I1 evidence. Resolution:
**READINESS_UPDATED_TO_R9_ONLY_NEXT_IMPLEMENTATION_GATE**. R4-R8 remain **PASS**;
R9 remains **NOT IMPLEMENTED**. This is a status correction, not a Trainer/model refactor.

The reviewed 20M candidate now emits
`IMPLEMENT_R9_THEN_RUN_APPROVED_20M_EXECUTION_GATES`. Its evidence scope states that
R1-R8 infrastructure is implemented/reviewed, R9 is unimplemented, hardware execution
gates remain unpassed, and no pilot/training authorization exists. References include
the pre-build, I1 and I2 JSON receipts. Architecture/accounting/construction remain
RECOMMENDED_FOR_PILOT / META_VERIFIED / META_CONSTRUCTED; training NOT_RUN, quality
UNVERIFIED, pilot_execution_ready=false and training_authorized=false are unchanged.

Frozen 2M's current next action is `PRESERVE_FROZEN_2M_COMPLETE_20M_R9`.
COMPLETED_FROZEN_LINEAGE, ENGINEERING_BASELINE_ONLY and the prohibition on current-source
resume retain their original evidence scope. README now states the current R9-only
implementation gap, unauthorized pilot and unverified T4 without the old workaround.

Current aggregate source SHA is `b54aad86af6771479be2901a71ee67fb89dc84a2e9b4c39347d67fb8a7a36821`; the previous I2 source identity was
`39e71d4a2e772782d0057e9df5bbc611756c8a027dbb1eccf18b2cf7c480eb6d`. Historical/frozen hashes were not rewritten.
The entire training package, loss/optimizer/scheduler/data ordering code and all four
model-mathematics files remain byte-identical to the correction entry. Both resolved
identities, 20,387,531 selected parameters and 128 selected tensors remain unchanged.

Before editing, all **14** recorded artifact hashes and **233** protected hashes
matched. Readiness plus its two existing test files are now explicitly authorized
changes: the JSON records their old/new hashes and removes only these three from
the current unchanged-file map. Every current artifact/protected binding is checked;
there is no concealed mismatch. Final scope is six correction files and 18 cumulative
I2 files. The original mechanics, timings, BITWISE_PASS and historical test records
below are retained; new verification is recorded separately here and in the receipt.

| Correction check | Actual result |
|---|---|
| targeted | 60 passed, 1 warning in 55.56s |
| reviewed_current | 167 passed, 1 warning in 40.22s |
| historical_replay | 30 passed in 1.04s; 1 passed in 1.69s |
| ruff | All checks passed! |
| format | 108 files already formatted |
| compileall | PASS (exit 0) |
| cpu_pip | No broken requirements found. |
| cuda_pip | No broken requirements found. |
| diff | PASS (exit 0) |

The first concurrent targeted attempt reported 59 passes and one timeout in the
optimized-Python subprocess (existing 30-second limit). A serial rerun passed;
no test, timeout, runtime code or numerical tolerance was weakened. Both attempts
are retained in the JSON closeout record.

The required I2 suite exercised bounded synthetic CPU mechanics again. No project
training or overfit experiment was launched; no 20M execution, production TEST read,
dataset acquisition or Kaggle operation occurred. Linux/T4 hardware limitations and
the unchanged missing-NumPy warning remain. R5 equivalence passed without tolerance
relaxation. Revert this correction for a narrow rollback (which restores the stale
wording); full I2 rollback then reverts the prior closeout and implementation in reverse
order. Next work is separately scoped R9, followed by approved execution gates.

Correction files:

- `README.md`
- `reports/20m_i2_training_infrastructure.json`
- `reports/20m_i2_training_infrastructure.md`
- `src/unified_edge/readiness.py`
- `tests/test_cli.py`
- `tests/test_resolver_policies.py`

## R4: active data verification

The existing runtime manifest already required TRAIN and VALIDATION only. It remains
schema 1, preserving its canonical identity. Construction now rejects TEST before
any payload open. Historical corpus evidence with sealed metadata is unchanged.
VerifiedActiveData hashes all active files at startup and stores immutable bytes in
a read-only mapping; frozen TRAIN/VALIDATION window containers share one snapshot.
There are no update-time disk reads to substitute unverified training bytes.

Both modes fully verify at startup, resume and checkpoint publication boundaries.
Paranoid also verifies before every update/validation. Fast checks the bound immutable
manifest during updates and keeps consuming original bytes if disk changes; the next
full boundary detects persistent disk mutation and refuses checkpoint publication.
Tests cover missing TEST paths, same-length mutation, immutable buffers/windows,
manifest substitution and resume rejection. Transient disk changes restored between
checks are not promised detectable, but cannot enter the immutable training snapshot.
Only the future immutable-shard interface contract is documented; R11 is not implemented.

## R5: monitoring and exact semantics

MonitoringPolicy schema 1 has REFERENCE_PARANOID and PRODUCTION_FAST. The default
remains paranoid. Fast detailed tensor inventory/RSS cadence is updates 1-10 and every
100; both modes sample pre/post checkpoint. Scalar metrics remain per update.
All finite loss/gradient/pre-update/post-update parameter guards stay enabled.
Canonical recurrent-state memory is derived analytically instead of allocating
scratch tensors. Validation preserves RNG and model mode even if a diagnostic consumes RNG.

Deterministic synthetic tests compare ten updates, save/restore at five, and an
uninterrupted control: losses, clipped gradients, parameters, optimizer moments/counts,
schedule, cursor, Python/Torch RNG and checkpoint semantic payload are bitwise equal.
An independent 12-update tiny fixture also passed checkpoint equality and next-update
equality when restoring paranoid into fast mode. No tolerance was relaxed.
Only run IDs, elapsed times, monitoring identity and external memory/timing samples
are excluded. Injected nonfinite loss, gradients, updated parameters and optimizer
moments cannot be published as valid checkpoints.

Descriptive 12-update fixture: full verification calls before checkpoint were
**13 paranoid / 1 fast**, then **14 / 2** after checkpoint; detailed memory samples
before checkpoint were **12 / 10**. The receipt separates hashing, compute, memory,
inventory, checkpoint I/O and logging intervals. These host timings are not kernel
benchmarks, a speedup guarantee or a 20M throughput claim.

## R6: portability

Windows RSS uses GetProcessMemoryInfo and was measured natively. Linux VmRSS/VmHWM
parsing uses explicit 1024-byte kB units and controlled parser/dispatch fixtures;
**native Linux execution remains UNVERIFIED**. Unsupported platforms return null RSS
and supported=false; measurement failures remain explicit. RSS is distinct from GPU
memory and isolated activations; OS working-set definitions are not interchangeable.
Relative POSIX logical data paths, pathlib roots, same-parent checkpoint temporaries,
checksummed publication and list-based subprocess calls remain portable. No shell=True,
PowerShell dependency, host absolute path serialization or new dependency was added.
Historical scripts/paths are preserved; their production gates are not current runners.

## R7: discovery and admission policies

Discovery handles 0/1/2+ devices without choosing one. Current training schema 2 permits
explicit cuda:N; named DevicePolicy schema 1 must select the same index before model
allocation. Legacy schema 1 retains CPU/cuda:0 and its one-visible-device fallback.
CUDA seeding/state capture/restore target only the selected execution device.

| Named policy | Allocator cap | Free headroom | Evidence |
|---|---:|---:|---|
| 20m_t4_single_fp32_pilot | 8,589,934,592 bytes | 2,147,483,648 bytes | ANALYTICALLY_PLANNED / HARDWARE_UNVERIFIED |
| 20m_rtx2050_fp32_fallback | 2,684,354,560 bytes | 536,870,912 bytes | ANALYTICALLY_PLANNED / HARDWARE_UNVERIFIED for 20M |

Startup requires cap plus headroom free; observed samples enforce headroom and the
allocator peak cap. Between-sample driver free memory remains unobserved. Diagnostics
report device/version/capability/VRAM/flags/profile metadata without training or
execution approval; an optional fixed nvidia-smi argument list reads driver version.
Missing/unavailable values are explicit. Mock device gates passed; CPU Torch truthfully
reported NO_CUDA (not a claim this physical Windows host lacks an RTX). No real CUDA
run or nvidia-smi diagnostic was performed in I2. No DDP, pooling or implicit selection.
Future T4 execution still requires the approved host boundary and launcher visibility gate.

## R8: parameter coverage and errors

New run manifests are schema 2; new checkpoint payloads schema 3; integrity envelopes
remain schema 1. Inventory schema 1 binds resolved identity, exact unique named
trainable shapes/dtypes, aliases and optimizer order. Counts are model-derived:
frozen 2M remains **56**, selected 20M **128**. No global 56-to-128 replacement occurred.
Lazy AdamW state exists exactly for positive per-parameter counts. Wrong architecture,
missing/unexpected weights, duplicate/missing optimizer coverage and reordered inventory
fail. Publication verifies optimizer moments/counts and schedule alignment too.

The low-level reader still decodes old schema 2; the current Trainer explicitly rejects
its migration and preserves source/config/environment checks. Frozen checkpoint bytes
are untouched. Configuration uses ValueError/TypeError; narrow data, checkpoint,
numerical, admission and evidence errors retain compatible exception ancestry.
All correctness-critical assertions in the training package were replaced with explicit
classified errors; optimized-Python negative gates passed. Frozen scripts' legitimate
2M-specific 56-tensor assertions remain historical, unchanged and unpromoted.

## Validation and preservation

| Group | Actual result |
|---|---|
| targeted | 29 passed, 1 warning in 27.68s |
| reviewed_current | 167 passed, 1 warning in 44.02s |
| historical_replay | 30 passed in 1.16s; 1 passed in 1.96s |
| synthetic_training_regression | 55 passed, 7 skipped, 1 warning in 86.10s (0:01:26) |

Ruff check PASS; format PASS (108 files); compileall PASS; CPU/CUDA pip check PASS;
git diff check PASS. Commands, outputs and exclusions are in the JSON receipt.
The optional missing-NumPy warning persists without environment changes. The seven
CUDA skips are unexecuted gates, not hardware passes. Known production payload-reading
tests, including test_full_training_stage_a.py, were excluded. Historical replay used
verified temporary copies of the immutable authorization snapshot.

Selected 20M resolved SHA:
`6e7ffdc3c3448f7f19fa4466a9780c369c05cc572030c1b026d4eef1503b9359`;
**20,387,531 parameters / 128 tensors**, all meta-only for this shape. Frozen 2M SHA:
`7fe06b639741003e4c5ad9d57d32c2fd2b86f6805b2048ad1efd6aaa75e3a6f1`;
**1,929,579 parameters / 56 tensors**. Both exact identities are unchanged.
All four model-mathematics files, unchanged I1 configuration/accounting and historical
evidence, accepted 2M reports, historical expected hashes and 66 snapshot
files match entry hashes. The endpoint checkpoint was raw-hashed, never loaded.
Optimizer builder/scheduler ASTs are identical; numerical helper equations/guards
are identical apart from the explicit error subtype. Trainer infrastructure changes
are intentional. Current source SHA: `b54aad86af6771479be2901a71ee67fb89dc84a2e9b4c39347d67fb8a7a36821`. Frozen historical source identity
is preserved as evidence, not assigned to the edited checkout.

Authorized fresh synthetic CPU forward/backward/optimizer updates and overfit ran.
No production continuation, 20M forward/training, TEST payload access, dataset acquisition,
Kaggle operation, optimized Mamba, context-bridge change or R9 implementation occurred.

## Limitations, rollback and next step

Native Linux execution, real CUDA parity/admission, T4 availability, 20M GPU fit,
throughput/quality and durable Kaggle export remain UNVERIFIED. Both named profiles
are policies, not measured passes. Monitoring equivalence applies to the deterministic
CPU fixtures; it does not establish cross-backend or cross-hardware equality.

Rollback: revert the additive closeout, then `c096765ff71e728701af3a248e27a6674ee25e1e`; retain all frozen artifacts.
Paranoid mode is the current fallback. Historical 2M reproduction requires its accepted
source checkout; never bypass source identity or migrate weights between architectures.
Next: review I2, then separately scope **R9**. Pilot execution still needs R9, the
remaining approved mechanics/hardware/data gates and explicit compute authorization.

## Files changed

- `README.md`
- `docs/project_state.md`
- `docs/training_infrastructure_policy.md`
- `reports/20m_i2_training_infrastructure.json`
- `reports/20m_i2_training_infrastructure.md`
- `src/unified_edge/training/checkpoint.py`
- `src/unified_edge/training/config.py`
- `src/unified_edge/training/data.py`
- `src/unified_edge/training/device.py`
- `src/unified_edge/training/experiment.py`
- `src/unified_edge/training/memory.py`
- `src/unified_edge/training/monitoring.py`
- `src/unified_edge/training/optimization.py`
- `src/unified_edge/training/trainer.py`
- `tests/test_training_infrastructure_i2.py`

- `src/unified_edge/readiness.py`

- `tests/test_cli.py`

- `tests/test_resolver_policies.py`

Detailed ongoing contract: [training infrastructure policy](../docs/training_infrastructure_policy.md).
