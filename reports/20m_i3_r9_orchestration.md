# 20M-I3 R9 orchestration closeout

Decision: **20M_R9_READY_FOR_REVIEW**. **R1-R9 IMPLEMENTED**; R9 is synthetically
verified and ready for review. **20M PILOT NOT AUTHORIZED**. Training authorization
and pilot execution readiness remain false. This is implementation evidence.

Entry: `1a7fc6ea0667eea3f7028ed098a8ac27edb6316a`; clean local master and live remote matched.
Implementation: `a78538ef9bea9043b79237c2f04c7d56fa9ac59a` (`feat: implement bounded 20m pilot orchestration`).
The additive closeout commit binds this report and current README/project state.
Its final SHA and local/live-remote agreement are recorded in Git and the final response.

## Contracts and scope

Seven new focused training modules implement immutable plans, active-data subset
contracts, diagnostic metrics, prerequisite/promotion gates, LR probes and orchestration.
The CLI is `scripts/edge_pilot.py`; plan/inspect/prerequisite checks are read-only.
Launch and resume require explicit execution, approved receipt digests, active-data
paths and the approved durable registry. No generic shell or Kaggle adapter was added.
The existing Trainer receives only narrow scheduler selection, tail accounting and
an optional checkpoint destination; its default loss/optimizer/cosine arithmetic is unchanged.

Pilot plan SHA: `49eb2275b70bcaa002095b381eb23e8cc9ab86a29f8c3420cddc257bf48bd164`.
LR probe plan SHA: `65d86c7834b65a4c0cdf8cbed79c4228ca6e5a29e66fe1fe736e967646d0858d`.
Plan schemas: `edge-pilot-1`, `edge-lr-probe-1`, `edge-evaluation-1`,
`edge-generation-1`, `edge-context-1`, `edge-promotion-1`.
Each rejects unknown fields and changed reviewed settings and serializes canonically.

Pilot: seed 17, L64, patch8, B8/acc16; nominal 128 sequences and 8,192 target bytes
per update. AdamW beta .9/.999, epsilon 1e-8, weight decay .01, clip1; FP32, TF32 off.
Exactly 8,000 updates, 400 linear warmup, cosine to .1 of the selected LR; PRODUCTION_FAST.
REFERENCE_PARANOID remains the default reference fallback. No real plan was run.

LR candidates are exactly **3e-4, 6e-4, 1e-3**. Every fresh seed17 arm uses 256 updates,
32 warmup and the distinct **warmup_constant_v1** scheduler, then constant trial LR.
Identical initial parameter/data-order hashes are enforced. Endpoint improvement >=.1,
finite numerical states, no three post-64 losses >2x prior32 median, and last128 clipping
fraction <=.5 are mandatory. Lowest endpoint NLL wins; within <=.01 choose lower LR.
No stable arm means STOP; no best-checkpoint selection or automatic extra rates.
The orchestration loop is tested with a synthetic Trainer double; scheduler and
Trainer integration are separately exercised on bounded tiny CPU fixtures.

## Data, cadence and execution gates

TRAIN 65,536,000 and VALIDATION 4,096,000 capacities and exact 50/25/15/10 domain
quotas are metadata contracts. Future TEST 4,096,000 is sealed reserve metadata only.
Validation partitions: 262,144 selection +262,144 monitor +3,571,712 confirmation
=4,096,000. Ranges/content hashes are pinned, disjoint and verified against immutable
active buffers during authorized evaluation. No real subsets or corpus were built.

Checkpoint events: 0/every250/endpoint/graceful complete boundary; monitor: 0/every250;
confirmation: every1000/endpoint; generation: 0/every500/endpoint; context:
0/every1000/endpoint. Endpoint events deduplicate. One initial-confirmation observation
supplies the pretraining NLL reference. Numerical monitoring remains every update;
detailed memory first10/every100, checkpoint pre/post and failure.

Explicit receipts gate construction, forward/backward, hardware/time/resume, mechanics,
LR selection, approved data, human compute approval and approved host execution context.
TEST_ONLY receipts cannot authorize REAL. A human-approved registry binds duplicate
protection. Implementation readiness does not supply these unrun receipts.
The construction schema preserves 20,387,531 parameters, 128 tensors, tensor signature,
round-trip and no unclassified tensors. Later hardware/mechanics receipts require the
fixed fixture, 60 updates, B2/acc2/L32/LR.003/warmup5, >=50% reduction, resume30 and
later bounded real-data smoke; no selected-20M stage ran here.

## Accounting, budgets and durable resume

Actual complete updates, microsteps, sequences/examples, valid target bytes and tail
bytes/windows are recorded; BOS/padding are excluded. Nominal 65,536,000 is never
reported as actual. No one-epoch assumption is made. Physical failed-attempt overhead
is separate and explicitly unverified when a partial failure prevents exact counting.

Probe budget is three cumulative GPU hours; pilot budget is 24, including diagnostics,
checkpoint work and initialization/restore intervals. Mock-clock tests pass. There is
no extension; stop at a safe complete boundary and report any necessary boundary
overshoot. Cooperative SIGINT/SIGTERM requests stop without aborting the current update.

Internal per-update recovery snapshots bind Trainer schema3 plus run/plan/LR/data/gate
identity, step, events, budget and exposure. Keep the last two plus cadence checkpoints.
Checksummed pending boundaries can be adopted immediately before publication.
Event receipts are durable before completion marking. All four requested interruption
points preserve exact next state and event sets: before/after event, before/after
checkpoint publication. Missing completed evidence and ambiguous partial intents fail
closed. No completed optimizer update is replayed; no force override exists.
Exclusive OS ownership and immutable registry identity reject duplicate launches.
GRACEFUL_STOP is distinct from FAILED_UPDATE; failed Trainer instances remain unusable.

## Generation, context and promotion

All nine exact prompts use greedy plus seeds101/202/303/404, 256 bytes each,
temperature1 without filters/penalties. Retain every unfiltered 267-class probability;
only bytes0..255 can be emitted. Implement whitespace, printable fraction, longest run,
empirical byte entropy bits, mean/median q(space), unfiltered predictive entropy bits,
conditional-byte top1/top2 margin, control mass, strict UTF-8, distinct1/2/4grams,
sample diversity, greedy suffix diversity and late-half/per-prompt statistics.
JSD uses natural logs over all 36 first-byte prompt pairs; analytic fixtures pass.

20M-5 implements every declared inclusive boundary: per-prompt control <=.01,
greedy median whitespace <=.65 and maximum <=.90; >=7/9 run<=32 AND entropy>=2;
late-half median q(space)<=.35; no 32 consecutive q(space)>.8; printable>=.90 on>=7/9;
UTF-8>=33/36; >=7/9 with >=3 distinct samples; >=7 greedy64-byte suffixes;
no greedy32-position margin>=.95 run; median JSD>=.005. No universal margin floor.

Context: pinned 128 anchors,32/domain, >=16 independent document hashes/domain, no TEST.
Phase p consumes base32+p versus128+p bytes ending at the same anchor and target block,
with the same recent suffix; shuffle only older history. Report NLL, phase TV,
shared/conv/SSM/pre-tanh/hidden/logit changes, tanh saturation and context gradients.
Paired document-cluster bootstrap uses 2,000 resamples, seed43, deterministic percentile
indices49/1949. Duplicate observations do not increase independent evidence.
Require history delta>=.01 and CI lower>0, shuffle advantage>=.005 and CI lower>0,
>=75% responsive anchors with max phase TV>1e-5, no domain delta<-.02.
Bridge evaluator only: B must pass context, gain>=.01 with positive paired document CI,
no domain degradation>.02, and repeat positive direction at seed29. No B architecture.

20M-4 evaluates only the endpoint: R1-R9, LR PASS, exactly8000 updates, valid exposure/data,
confirmation gain>=.10, every domain>=.02, consistent checkpoint/export, no stop.
It returns PASS/STOP/INCOMPLETE without training. Endpoint collection deliberately keeps
export consistency false until separately verified; no pilot or quality PASS is claimed.
Decimal(str(value)) comparisons avoid hidden epsilon rules; differences convert both
operands first. Exact inclusive and strict boundaries have dedicated tests.

Validation, full 45-trace generation and representative eight-phase context measurement
on the tiny real model preserve RNG, weights, gradients, data order, optimizer, scheduler
and the next update exactly. Full protocol metrics also use synthetic panels; no real
validation or quality evaluation was performed. I2 R5 equivalence remains bitwise.

Local export manifest binds plan/model/data/LR/checkpoint/metrics/generation/context,
environment/hardware/source identities, logical/physical exposure and stop/completion.
No upload, host runner, Kaggle CLI, dataset write, kernel or GPU request occurred.

## Validation

| Check | Actual result |
|---|---|
| r9 | 47 passed, 1 warning in 48.88s |
| i1_i2 | 122 passed, 1 warning in 65.65s (0:01:05) |
| checkpoint_training | 25 passed, 1 warning in 14.75s |
| historical_replay | 30 passed in 1.02s; 1 passed in 2.11s |
| reviewed_current | 167 passed, 1 warning in 32.08s |
| ruff | All checks passed! |
| format | 118 files already formatted |
| compileall | PASS (exit 0) |
| cpu_pip | No broken requirements found. |
| cuda_pip | No broken requirements found. |
| diff | PASS (exit 0) |

The reviewed non-TEST allowlist is retained from the accepted I2/provenance receipts.
Historical replay uses verified temporary copies, not mutations to the original snapshot.
No production TEST-reading suite was invoked. Optional missing-NumPy warning remains.
One development resume test initially rejected a synthetic endpoint against8000;
binding the actual endpoint into immutable identity fixed it. Final-review
changes added immutable anchor-ID rejection and cooperative CLI stopping; final R9 and
all static checks were rerun. Unaffected I1/I2/checkpoint/historical/current results are
retained from the immediately preceding full matrix. No test threshold was relaxed.

## Preservation, limitations and rollback

All 241 pre-existing tracked files outside seven explicitly modified
files match entry hashes. All four model-mathematics files are byte-identical.
Legacy optimizer/loss/cosine LR AST equality: True.
Selected meta accounting exactly equals I1; resolved SHA
`6e7ffdc3c3448f7f19fa4466a9780c369c05cc572030c1b026d4eef1503b9359`,
20,387,531 parameters /128 tensors; named tensor signature `3f242a2332fa41d4737f5d184b1593051bf8384a7cb5e954f457c136848dfd61`.
2M resolved SHA `7fe06b639741003e4c5ad9d57d32c2fd2b86f6805b2048ad1efd6aaa75e3a6f1`,
1,929,579 parameters /56 tensors, unchanged. Frozen checkpoint raw SHA
`0d4c822daadf6bb8269a727ac54d54a26e9edf49ff43fedff1b0d91ba6ad2623` was checked without deserialization.
All66 historical snapshot files/membership and original manifest identity
`8461a8d3aaad724a6880627a91520257ab743dae3713c95ad592b5467395ad8d` match. TEST remains sealed.
Current source SHA `313d8217f7b3bee8a82bef9225e144c37cf5912bc11d6bb8cfb60c80c8c325a8` is intentionally different from frozen
source; no checkpoint source-identity bypass or migration was introduced.

- No selected-20M forward/backward, real LR probe, pilot, data acquisition, production TEST access or Kaggle operation occurred.
- T4 availability, actual GPU phase memory/throughput, 20M-1/2/3, native Linux and remote output durability remain unverified.
- Per-update recovery snapshots add unmeasured I/O cost; their full cost must satisfy the later 24-hour hardware gate with the 30% projection reserve.
- An incomplete update or diagnostic intent with ambiguous physical work/time blocks automatic replay/resume; preserve the last durable checkpoint and obtain explicit recovery review. No force bypass.
- Failed partial-update byte overhead can be unverified; it is explicitly null with a reason, never silently zero or logical exposure.
- Intervals conservatively count synchronized wall time, including host work; synthetic mock clocks are not measured GPU-hour evidence.
- Receipt digests require independently reviewed trust inputs; receipt self-assertions or TEST_ONLY objects do not grant real execution authority.
- Synthetic metrics and anti-collapse thresholds do not establish 20M language quality or useful context. No context bridge B is implemented.
- Existing optional missing-NumPy warning remains; no dependencies or CPU/CUDA environments were changed.

Rollback: revert the additive closeout then `a78538ef9bea9043b79237c2f04c7d56fa9ac59a`; retain all frozen
artifacts. Reference paranoid mode and original cosine/loss/optimizer math remain
available. No R10 optimized Mamba, R11 indexed loader or bridge intervention was added.
Next exact gate: **RUN_APPROVED_20M_1_2_3_HARDWARE_AND_MECHANICS_GATES**.
Approved data preparation, LR probing and explicit bounded-pilot compute approval
remain separate. Kaggle still follows approved private-write smoke -> CPU notebook
-> T4 diagnostic ->20M-1/2/3 ->LR probe ->pilot through the approval-bound host bridge.

## Files changed

- `README.md`
- `docs/20m_pilot_orchestration_contract.md`
- `docs/project_state.md`
- `reports/20m_i3_r9_orchestration.json`
- `reports/20m_i3_r9_orchestration.md`
- `scripts/edge_pilot.py`
- `src/unified_edge/readiness.py`
- `src/unified_edge/training/optimization.py`
- `src/unified_edge/training/pilot_context.py`
- `src/unified_edge/training/pilot_data.py`
- `src/unified_edge/training/pilot_diagnostics.py`
- `src/unified_edge/training/pilot_gates.py`
- `src/unified_edge/training/pilot_plan.py`
- `src/unified_edge/training/pilot_probe.py`
- `src/unified_edge/training/pilot_runner.py`
- `src/unified_edge/training/trainer.py`
- `tests/test_cli.py`
- `tests/test_pilot_r9_metrics.py`
- `tests/test_pilot_r9_runtime.py`
- `tests/test_resolver_policies.py`

20M-R9 STATUS: READY FOR REVIEW
