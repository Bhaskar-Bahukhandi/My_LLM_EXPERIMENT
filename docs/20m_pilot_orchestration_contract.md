# R9 bounded pilot orchestration

R9 extends the existing Trainer and schema-3 checkpoint rather than introducing a
second optimizer/data implementation. It does not authorize execution. Plans are
frozen dataclasses with strict complete JSON schemas, canonical SHA-256 identities
and fixed reviewed values. Unknown fields, changed thresholds, mutable containers
and unsupported schema versions fail. `PilotPlan` composes the LR, evaluation,
generation, context and promotion plans. No YAML overlay or environment variable
can change these values.

## Plans, data and admission

The pilot fixes seed17, FP32/TF32 off, L64/patch8/B8/acc16, AdamW
(.9,.999)/epsilon1e-8/decay.01/clip1, 8,000 updates, 400 warmup and cosine to
.1 of the selected LR. Fast monitoring is selected; paranoid remains the default
Trainer fallback. The independent `warmup_constant_v1` probe scheduler warms up
linearly for32 updates then holds its trial LR through update256. Fresh arms at
3e-4/6e-4/1e-3 must have identical initial-parameter and data-order hashes.
Selection uses only endpoint validation: finite states/losses/gradients/parameters,
NLL improvement >=.1, no3 consecutive losses >2x preceding32 median after update64,
and at most64 clipped updates in the last128. Among stable arms within .01 of the
best endpoint NLL, select the lowest LR. No stable arm means STOP.

The TEST-free runtime manifest is unchanged. Capacity checks require exact TRAIN
65,536,000 and VALIDATION4,096,000 domain quotas from the reviewed plan; future
TEST4,096,000 is metadata only. Subset manifests pin document SHA, byte ranges and
payload SHA: selection262,144 +monitor262,144 +confirmation3,571,712 =4,096,000.
All ranges are disjoint, including content-identical documents under alternate
paths. Confirmation must include every domain. Actual evaluation verifies its
spans against the existing immutable active buffers. Capacity validation is not
acquisition, provenance approval, or permission to open payloads during dry-run.

GateReceipt schema edge-gate-1 requires typed stage evidence and SHA-bound content
for construction, forward/backward, measured hardware/budget/resume, mechanics,
LR selection, approved data, human approval and approved execution context.
Each is bound to plan/model/data/subsets/anchors/seed/source. The calling authority
must explicitly supply trusted receipt hashes; a receipt's self-declared PASS is
not trust. Human references and external evidence still require actual review.
TEST_ONLY receipts are restricted to tiny CPU fixtures and rejected for REAL runs.
The approved context pins the durable run registry, preventing a second launch by
choosing a different output folder. Identical run identity refuses duplicate launch;
there is no force flag. Only explicit resume can reopen an existing run.

## Cadence, accounting and recovery

Checkpoint events occur at0/every250/endpoint/graceful complete boundary; monitor
at0/every250; confirmation every1000/endpoint; generation at0/every500/endpoint;
context at0/every1000/endpoint. Endpoints are deduplicated. A separate initial
confirmation observation supplies the pretraining reference required by20M-4.
Numerical checks run every update; I2 detailed memory remains first10/every100,
pre/post checkpoint, plus failure telemetry. Diagnostic RNG and model mode are
restored; generation uses isolated generators. Context gradients use autograd.grad,
never an optimizer update. Existing gradients are restored even on exceptions.

Exposure records actual complete logical updates, microsteps, sequences/examples,
valid target bytes and short-tail bytes/windows. Padding/BOS never count. Nominal
65,536,000 is planning metadata, never reported as measured exposure. Cursor epochs
are unconstrained by the nominal total. Failed partial updates are excluded from
logical counts; their physical overhead is explicitly unverified where a failure
prevents exact byte measurement, never reported as zero.

Durability adds an internal recovery snapshot after every completed update. This
is distinct from the requested retained checkpoint/event cadence. The last two
recovery snapshots and all declared cadence checkpoints are kept. This deliberate
I/O cost must fit the future measured hardware/time gate; no throughput is claimed.
Atomic same-parent directory publication binds the existing Trainer checkpoint,
run/plan/LR/data/gate identities, exposure, completed-event identities and budget.
An already verified pending boundary can be adopted after interruption immediately
before publication. Checksum-bound event receipts precede completion marking, so
interruptions before/after events or publication do not duplicate logical results.
Missing completed receipts fail closed. A per-run OS lock prevents concurrent owners
and releases on process death. No checkpoint source/environment guard is weakened.

An abrupt failure inside an update or diagnostic can leave uncertain physical work
or elapsed time. Such an incomplete intent blocks automatic replay/resume, retains
the last durable complete checkpoint, and requires explicit recovery review. This
is not permission to repeat an optimizer update or reset the budget. Controlled
failed updates produce FAILED_UPDATE and leave Trainer failed; user/budget/stability
stops at safe boundaries produce GRACEFUL_STOP. Diagnostic/checkpoint failures have
separate classified failure receipts. No partial update is published as complete.

Budgets are cumulative3h for all probe arms and24h for the pilot. Synchronized
intervals include training, diagnostics, checkpoint work and initialization/restore;
host time inside them is counted conservatively. Stop before another update once
the budget is exhausted; a necessary complete-boundary export can report overshoot,
never hide it or automatically extend the budget. Mock clocks test this policy.
Native power-loss durability and remote notebook output persistence remain separate
hardware/host gates; local fsync/rename does not claim remote durability.

## Metrics and endpoint decisions

Generation is the exact9 prompts, greedy plus seeds101/202/303/404, 256 bytes each,
temperature1, no sampling filters or repetition penalties. All267 probabilities
are retained before conditioning on byte IDs0..255. Whitespace is9/10/11/12/13/32;
printable is32..126 plus9/10/13. Metrics include raw-byte entropy bits, longest run,
UTF-8, distinct1/2/4grams, q(space), unfiltered predictive entropy bits, conditional
byte top1/top2 margin, unfiltered control mass, late-half and per-prompt summaries.
Pairwise first-byte conditional-distribution JSD uses natural logs across all36
prompt pairs; it does not establish long-history use.

20M-5: all45 valid complete traces; each prompt's mean control mass <=.01;
greedy median whitespace <=.65 and every greedy <=.90; >=7/9 greedy traces with
run<=32 AND entropy>=2bits; median greedy late-half median q(space)<=.35;
no trace with32 consecutive q(space)>.8; printable>=.90 on>=7/9 greedy;
UTF-8 valid>=33/36 samples; >=7/9 prompts with>=3 distinct samples; >=7 distinct
greedy64-byte suffixes; no greedy32-position margin>=.95 run; median JSD>=.005nats.
There is no general margin floor. Invalid/missing endpoint panels cannot promote.

Context manifests require128 unique anchors,32/domain and>=16 distinct source
document hashes/domain, all VALIDATION, pinned before training. Target block SHA
and offsets are fixed. At phase p, consume base32+p versus128+p bytes ending at the
same anchor, preserving identical recent32+p bytes and target block; shuffle only
the older96 bytes with a local deterministic RNG. The base lengths remain32/128;
the common0..7 offset sweeps the patch clock without introducing unequal targets.
Report short/long/shuffled NLL, byte TV, shared/conv/SSM/pre-tanh/hidden/logit deltas,
long-history tanh saturation (abs(tanh)> .99), and context-projection gradient norm.

Inference averages paired anchor/phase observations and resamples whole source
documents, retaining their rows:2000 draws, seed43, percentile indices49/1949.
Identical duplicate observations are idempotent; conflicting duplicates reject.
History improvement>=.01 with CI lower>0; shuffled advantage>=.005 with CI lower>0;
>=75% anchors max phase TV>1e-5; no domain delta<-.02. State changes alone never pass.
Bridge evaluator only: B must pass context, improve confirmation>=.01 with positive
paired document CI, no domain degradation>.02 and positive replication at seed29.
Keep passing A unless B earns promotion; neither passing blocks serious training.

Threshold comparisons use Decimal(str(number)), without epsilon or hidden tolerance.
For differences, convert operands before subtraction. The stated >= and <= bounds
are inclusive; CI positivity, TV>1e-5 and sustained-space>.8 are strict.
20M-4 requires R1-R9, passed LR, exactly8000 updates, valid data/exposure, endpoint
confirmation improvement>=.10 and every domain>=.02, consistent checkpoint/export
and no stop. Missing evidence yields INCOMPLETE; failed gates STOP. Automatic
best-checkpoint selection is absent. Local endpoint evidence remains unreviewed
until independently verified export and human review.

## CLI, export and rollback

`python scripts/edge_pilot.py plan` prints immutable plans, identities and cadence.
`inspect --run-record PATH` checks a local receipt. `validate-prerequisites --bundle
PATH` reads metadata and explicitly trusted receipt hashes only. `launch` and
`resume` also default to validation: execution needs `--execute`, all trusted gate
hashes, the approved registry and explicit active-data root. CLI never invokes a
shell bridge, Kaggle or a host runner. No dry-run constructs real model weights,
opens corpus payloads, writes checkpoints or performs forward/backward.

The local export-manifest function binds complete run identity, selected LR,
checkpoint/metrics/generation/context/environment-hardware SHA, source, actual and
physical exposure, and completion/stop outcome. It does not upload or assert remote
durability. Later private-write smoke -> CPU notebook -> T4 diagnostic ->20M-1/2/3
->LR probe ->pilot must retain explicit approval-bound host execution.

Rollback is additive: revert I3 commits in reverse order. Reference paranoid mode,
legacy cosine arithmetic, model mathematics, frozen2M artifacts and TEST seal remain
the fallback. Old schema-3 checkpoints still require their exact accepted source;
no source-identity migration is provided. R10/R11 and context bridge B are untouched.
