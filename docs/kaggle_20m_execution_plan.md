# Approval-bound Kaggle 20M execution plan

Design only. No Kaggle action, upload, notebook execution or GPU allocation occurred.
Accepted bridge commit supplied by the user:
`4cb387d340b68e9e210bbea1effcaa03108806e1`, **READ-ONLY VERIFIED**.
Kaggle CLI2.2.4, credential isolation and approved host read boundary are accepted
from that statement, not re-audited here. Private writes, notebooks and T4 availability
remain unverified and compute remains disabled.

| Class | Permitted scope and approval |
|---|---|
| LOCAL_ONLY | Inspect/package/hash permitted local artifacts; no implicit upload |
| HOST_READ | Explicit allowlisted host listing/status/download operations when genuinely needed; no compute side effects |
| HOST_WRITE | Separately approved private resource write smoke; approval names payload, resource and visibility |
| HOST_COMPUTE | Separately approved notebook launch with hardware, immutable code/data digests, duration, output and stop budget |

Keep OAuth secrets isolated and out of notebooks, output, repository and logs.
No arbitrary shell bridge, sandbox-network workaround, implicit host fallback or
read-to-write privilege expansion. A CLI write that also runs a kernel counts as
HOST_COMPUTE, regardless of its command name; do not classify by HTTP verb alone.
This document grants neither HOST_WRITE nor HOST_COMPUTE permission.

## Eventual order: single T4 first

1. **Private write smoke:** after explicit HOST_WRITE approval, upload a tiny synthetic
   non-sensitive data artifact to a named private resource. Inspect visibility and
   content hash through HOST_READ. No notebook execution and no real dataset yet.
   If the chosen operation implicitly executes, stop and request compute authority.
2. **CPU notebook smoke:** separate HOST_COMPUTE approval; <=5 minutes, no GPU and
   synthetic bytes only. Test import lockfile, relative POSIX paths, Linux RSS, private
   durable output and a downloaded checksum round-trip. No training. Failure retains
   local reference artifacts and blocks step3.
3. **T4 diagnostic:** separate <=10-minute GPU approval. Record actual GPU count,
   name, capability, visible-device mapping, free/total VRAM, driver/runtime/torch,
   deterministic settings and allocator metrics. Mask one physical T4 before CUDA
   initialization even if two are assigned. Verify sm75 FP32 torch primitives and
   approved parity fixture. Availability is **UNVERIFIED**, never assumed from an
   account setting or listing. No installed Triton path is accepted on faith.
4. **20M mechanics smoke:** only after all pre-pilot refactors and stages0-2 pass,
   separately approve <=30 GPU minutes with a synthetic tiny-overfit fixture and
   approved TRAIN-only smoke subset. Save/restore exact update-boundary checkpoint,
   export privately, download and verify its hashes, then test next-update equality
   within the same supported environment. No production 2M checkpoint is input.
5. **20M bounded pilot:** separately approve LR probes, then the fixed pilot manifest,
   chosen LR and 8,000 updates. Budget <=24 cumulative GPU hours including monitoring
   and exports. Pause before the measured session deadline, reserving at least30
   minutes for durable export. Account/session limits must be verified at execution;
   do not hard-code a documentation limit as availability. No full run follows automatically.

Code bundle must exclude `.git` credentials, local auth directories, private logs,
the frozen 2M weights and TEST payloads. Include exact code/config/lockfile hashes,
TRAIN/VALIDATION-only manifest, license/provenance receipt and diagnostics policy.
Dataset acquisition or transfer is a separate authorization; this audit acquired none.

Atomic local checkpoint + manifest is only the first step. A checkpoint becomes
durable after private output persistence and independent downloaded hash verification.
Keep the last two verified durable checkpoints; do not delete the previous one while
an export is pending. Persist optimizer moments and per-tensor counts, scheduler,
Python/CPU/CUDA RNG, data cursor/order, byte/update counters, architecture/backend,
code/data/config/environment identities. Crash inside an update rolls back to the
last complete checkpoint and replays the same windows; record physical replay bytes.
Reject incompatible environment identities, never weaken restore checks to resume.

## Dual T4: deferred, not pooled VRAM

Two16GB devices do not form one32GB address space. Future DDP replicates parameters,
gradients and moments on each card; communication buckets add memory and overhead.
The analytic ledger adds two FP32 parameter-sized bucket allowances per rank.
Per-rank B=8 and accumulation8 would give the same nominal128 sequences/update
as one T4 B=8/accumulation16, but reduction order and data sharding change. Exact
single-device mathematical/evidence equivalence is not assumed. A new configuration,
rank RNG/cursor checkpoint format, backend parity/learning gates and explicit approval
are required. Current exactly-one-visible-device validation stays the fallback.

Stop on unexpected resource visibility, secret exposure, capability mismatch,
non-finite computation, OOM, parity failure, unavailable durable export, data/config
hash mismatch or exceeded time budget. Record the failure, retain last verified
checkpoint and return to the previous passed stage. Never compensate with unapproved
hardware, a larger budget, public artifacts or sandbox escape.
