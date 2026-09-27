# Current training infrastructure (20M-I2)

R4-R8 are infrastructure primitives. They do not authorize a 20M pilot, dataset
acquisition, R9 orchestration or Kaggle execution. Model equations, selected shape,
loss normalization, AdamW grouping and schedule mathematics remain unchanged.

## Verified active data

The existing runtime `DatasetManifest` schema 1 already requires exactly TRAIN and
VALIDATION. It is distinct from historical corpus evidence, which can retain sealed
TEST metadata. `create()` now rejects a TEST request before opening any file.
Paths are canonical relative POSIX names, resolved with `pathlib` inside the supplied
root; absolute paths, drive letters, backslashes and traversal are rejected.

`VerifiedActiveData` (`verified-immutable-eager-v1`) reads and hashes each active
document at startup, checks byte lengths and canonical manifest identity, and retains
immutable Python `bytes` behind a read-only mapping. Frozen window objects for both
splits share that snapshot. Updates never reopen a path to obtain training bytes.
The manifest and window container are immutable; the bound manifest identity is
checked on each update. Reverification compares disk against the original manifest
and never replaces the accepted buffers. An unreadable document fails explicitly.

Every new Trainer fully verifies at startup; restore fully verifies again. Every
checkpoint boundary fully verifies before publication. These are active-data checks,
never production TEST reads. The in-memory snapshot has process-local lifetime:
resume must recapture and verify the same bytes. A future indexed implementation
must provide equivalently verified immutable shard handles before first use/reuse;
R11 indexed loading is not implemented here.

| Monitoring schema 1 | Update/validation disk verification | Detailed inventory/RSS |
|---|---|---|
| REFERENCE_PARANOID (default) | Every update/validation | Every update |
| PRODUCTION_FAST | Startup, resume and checkpoint boundaries; immutable binding on updates | Updates 1-10 and every 100 |

Both modes sample memory before and after checkpoint I/O. Scalar loss/count/LR/norm
records remain per update. Fast mode can detect a disk mutation later, at its next
full verification boundary; it cannot silently consume the changed disk bytes.
Paranoid mode detects persistent mutation at its next verification, including before
the next checkpoint. No polling scheme promises to detect a transient mutation that
is restored between checks. Neither mode accepts altered bytes into its snapshot.
Export orchestration does not exist yet; R9 must use a verified checkpoint boundary
and separately verify its durable export, under the approval-bound host policy.

## Observational policies and numerical guards

`MonitoringPolicy` is frozen, typed and versioned. It is separate from `TrainingConfig`
and scheduler identity. Modes cannot change forward/backward, byte-weighted loss,
microbatch order, clipping, AdamW or schedule indexing. All existing per-update finite
loss, gradient, pre-update and post-update parameter guards remain enabled in both
modes. Internal reference-model diagnostics were not removed or deferred.

Validation saves/restores Python, CPU and selected-device RNG plus model mode, even
on failure. Detailed memory no longer allocates a scratch recurrent state: its FP32
canonical byte count derives from the same state axes. Actual parameter, gradient
and lazy optimizer tensor payloads remain separately measured.

`Trainer.observations` counts/times full data verification, inventory checks, memory
sampling, checkpoint serialization/publication, logging and update compute. These
host-wall-clock intervals are descriptive, not disjoint kernel timings or 20M
throughput measurements. Update compute includes numerical checks, data grouping,
transfer, forward/backward and optimizer work. Memory and I/O measurements are
observational and never drive schedule or data order.

## Process memory and paths

Windows uses GetProcessMemoryInfo WorkingSetSize/PeakWorkingSetSize. Linux parses
`/proc/self/status` VmRSS/VmHWM with explicit 1024-byte kB units. Missing, malformed
or unreadable measurements return no invented RSS. Reports include `rss_bytes`,
`source`/`method`, `supported`, and measurement status; old process-memory aliases
remain for evidence consumers. Unsupported OSes return `supported=false` and null RSS.

RSS is host resident memory, not GPU allocator memory or isolated activations.
Windows working set and Linux RSS/HWM are OS-specific measurements, not exact
cross-OS equivalents. Linux parser/dispatch are fixture-tested on the Windows host;
native Linux execution and checkpoint filesystem behavior remain UNVERIFIED until
the separately approved notebook smoke. No new dependency was added.

Runtime identities serialize relative logical names, not corpus or checkpoint host
paths. Checkpoints use same-parent temporary directories, fsync, checksums and rename;
exclusive run ownership and no intentional overwrite remain the contract. Subprocess
calls use argument lists, no shell wrapper. Historical scripts/paths are unchanged;
old production runners must be used only in their accepted frozen checkout.

## Device discovery and admission

`discover_devices()` reports zero, one or multiple visible devices without choosing
one. `hardware_diagnostic()` records count and available version/flag metadata; a
typed policy explicitly selects which device to query. It never configures execution
or runs training. A CUDA information query may initialize that device's driver context.
Without selection it reports SELECTION_REQUIRED; CPU-only Torch reports NO_CUDA.
An optional fixed `nvidia-smi --query-gpu=driver_version --format=csv,noheader` argument
list records the driver; absent, failing or ambiguous results are explicitly null.
No shell wrapper is introduced and the diagnostic performs no Kaggle operation.

| DevicePolicy schema 1 identity | Allocator cap | Required free headroom |
|---|---:|---:|
| 20m_t4_single_fp32_pilot | 8 x 2^30 bytes | 2 x 2^30 bytes |
| 20m_rtx2050_fp32_fallback | 5 x 2^29 bytes | 2^29 bytes |

Both are ANALYTICALLY_PLANNED policies, HARDWARE_UNVERIFIED for 20M. Startup requires
free memory >= cap + headroom; each observed detailed/boundary sample requires the
headroom and an allocator peak no greater than the cap. The cap is also installed
through Torch's allocator fraction. Driver free memory between samples is unobserved.
A profile name is not evidence of GPU model, measured fit or execution approval.

Legacy training schema 1 retains CPU/cuda:0 and its existing one-visible-GPU fallback.
Training schema 2 permits explicit `cuda:N`; a named DevicePolicy must match that
index. Multiple visible GPUs require a named explicit policy before model allocation.
Only that device receives model tensors and CUDA RNG capture/restore/seeding. No DDP,
automatic GPU selection, VRAM pooling or implicit profile substitution is implemented.
The future first T4 launcher must still apply its accepted single-device visibility
boundary before initialization; I2 supplies discovery/selection primitives, not a launcher.

## Checkpoints and explicit errors

New run manifests use schema 2; checkpoint payloads use schema 3; the checksummed
directory envelope remains schema 1. New parameter inventory schema 1 binds resolved
model identity, canonical unique trainable names/shapes/dtypes, alias relationships
and exact optimizer parameter order. It derives counts from construction, never a
fixed 56/128 constant. The historical 2M fixture still has 56; selected 20M has 128.

State exists exactly for parameters whose saved update count is positive, supporting
unused short-tail encoder parameters and unequal lazy AdamW counts. Missing, unexpected,
duplicated or reordered coverage fails. Checkpoint publication checks finite parameters,
optimizer moments/counts, schedule/global-step alignment and cleared gradients. Restore
also checks configuration, data, environment, code identity, names and tensor shape/dtype.
There is no architecture migration. The low-level envelope reader still decodes legacy
schema 2, but the current Trainer explicitly rejects its migration and directs historical
reproduction to the accepted source checkout. Historical bytes are never rewritten.

Monitoring identity is observational and may change on restore. Semantic comparisons
exclude only run IDs, elapsed timestamps, monitoring mode and external memory/timing
samples; weights, gradients at checked boundaries, optimizer, schedule, cursor, metrics
other than elapsed time, and RNG must remain bitwise equal on deterministic CPU fixtures.

Production training-package gates no longer depend on `assert`. Malformed configuration
uses ValueError/TypeError; data mutation uses DataIntegrityError; incompatible/corrupt
state uses CheckpointError; nonfinite numerical state uses NumericalTrainingError;
device admission uses DeviceAdmissionError; failed mechanics/evidence comparisons use
EvidenceIntegrityError. These narrow categories retain ValueError/RuntimeError ancestry
for existing callers and remain active under optimized Python.

Rollback: revert I2 additively. Reference paranoid mode remains the default fallback.
Use the accepted historical checkout for frozen 2M reproduction; never bypass its
source identity guard. Review I2 before separately scoping R9 and execution gates.
