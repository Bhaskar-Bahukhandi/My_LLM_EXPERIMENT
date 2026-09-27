# 20M bounded pilot and promotion contract

Design status: **20M_PREBUILD_READY_WITH_REQUIRED_REFACTOR**. Every execution below
is future work requiring the appropriate authorization; no 20M training ran here.
Use the selected shape in `20m_architecture_contract.md`. Keep frozen 2M unchanged.

## Closeout scope

The completed design at `af4e29bc04d8a2cc776870cbe657948170e50b3a` is preserved.
Closeout changes only prose and validation evidence; R1-R9 remain unimplemented.

## Exact data budgets

Raw bytes mean post-cleaning UTF-8 payload bytes, excluding padding, out-of-band BOS,
manifest bytes and repeated exposure. Decimal numbers below are exact planning
quotas, not a claim these datasets exist or a compute-optimal scaling law.

| Stage | Unique TRAIN bytes | VALIDATION bytes | New sealed TEST reserve | Total corpus bytes | TRAIN bytes/parameter | TRAIN document target |
|---|---:|---:|---:|---:|---:|---:|
| Mechanics smoke | 1,000,000 | 100,000 | 0 | 1,100,000 | 0.04905 | >=200 |
| Bounded pilot | 65,536,000 | 4,096,000 | 4,096,000 | 73,728,000 | 3.21451 | >=10,000 |
| Serious training | 2,000,000,000 | 20,000,000 | 20,000,000 | 2,040,000,000 | 98.09918 | >=200,000 |

Held-out reserves never count as training exposure. Smoke/pilot ratios are for
mechanics/diagnosis, not final competence. Two billion unique training bytes is a
substantially larger proposed first serious budget than 10 MB; learning curves and
quality gates must still justify it. Do not convert published token/parameter rules
to bytes/parameter without a corpus-specific tokenizer measurement. No serious
dataset is acquired, downloaded or built in this tranche.

Keep the authorized mixture **50% general text /25% code /15% technical documentation
/10% structured mathematics**, as recorded in the existing r3 domain totals. For
TRAIN the exact quotas are:

| Stage | General | Code | Documentation | Structured math |
|---|---:|---:|---:|---:|
| Smoke | 500,000 | 250,000 | 150,000 | 100,000 |
| Pilot | 32,768,000 | 16,384,000 | 9,830,400 | 6,553,600 |
| Serious | 1,000,000,000 | 500,000,000 | 300,000,000 | 200,000,000 |

Apply the same percentages independently to each holdout quota. Pilot validation
and future TEST each have 2,048,000 /1,024,000 /614,400 /409,600 bytes; serious each
10,000,000 /5,000,000 /3,000,000 /2,000,000; smoke validation 50,000 /25,000 /15,000
/10,000. Domain labels come from reviewed content, not relabeling code as docs to
fill deficits. Preserve provenance for any deterministic retained span; do not
pad with repeated bytes or synthetic math to meet quotas. If real eligible capacity
is insufficient, report the deficit and stop acquisition planning for review.

Document diversity targets are minimums, not splits of the same source counted as
independent documents. Smoke >=20 TRAIN documents/domain and >=40 validation total;
pilot >=500 TRAIN/domain and >=800 validation total; serious >=10,000 TRAIN/domain
and >=4,000 validation total. Future TEST reserves target the same holdout counts.
Pilot requires >=3 independently curated source families/domain, serious >=5/domain,
with no source family >50% of that domain's bytes; smoke >=2/domain. Repositories,
editions and mirrors of one origin count as one family. Cap a source document's
retained contribution at 20,000 smoke /65,536 pilot /262,144 serious bytes so a small
number of books or repos cannot satisfy diversity by themselves. Counts and exact
byte quotas both gate capacity; no claim current sources meet them.

Split by document and provenance/duplicate cluster before extracting windows;
author/work/version/project grouping prevents cross-split near duplicates. Preserve
original raw SHA, URL, revision, retrieval time, license/redistribution decision,
cleaner version, retained byte offsets and cleaned SHA. Deterministic source lists,
quota choices and random seeds must be frozen before training. Existing frozen
TRAIN/VALIDATION can supply an authorized smoke subset; never read existing TEST.

Dedup: exact raw+cleaned SHA across all candidate sources, normalized-text hashes,
and normalized 5-token-shingle Jaccard>=0.8 candidate clusters (MinHash can shortlist,
exact Jaccard decides), with code/math-aware review before removal. Preserve originals
and normalized fingerprints separately. Pin algorithms, normalization, thresholds,
seeds and exclusions. Holdout cluster membership takes precedence over TRAIN.
Contamination: use the versioned approved benchmark fingerprints and exclusion
rules before selection, quarantine matches, retain audit counts and rejected hashes;
do not silently weaken matching to meet quotas. TEST-aware dedup/splitting is a
separate future preparation authorization by the sealed-data custodian. The runtime
manifest contains TRAIN and VALIDATION only. Existing TEST remains untouched; future
TEST stays inaccessible to training, tuning, prompts and diagnostic selection.

## First real-data pilot settings

| Field | Predeclared setting |
|---|---|
| Initialization / seed | Fresh selected model; seed 17; no frozen 2M loading |
| Precision / device | FP32, TF32 off, deterministic algorithms, one visible T4 |
| Sequence length | **64 bytes**, patch8; reset per independent document window |
| Microbatch / accumulation | **8 /16** |
| Effective sequences/update | **128 nominal**, incomplete dataset batches recorded |
| Effective target bytes/update | **8,192 maximum/nominal**, sum actual valid bytes for tails |
| Optimizer | AdamW, foreach=false, fused=false, beta1=0.9, beta2=0.999, epsilon=1e-8 |
| Weight decay | 0.01 on rank>=2 tensors except tagged no-decay; zero for rank<2 and tagged parameters |
| Gradient clipping | Global norm 1.0 after accumulation, nonfinite error |
| LR | Candidate 6e-4; mandatory bounded selection among 3e-4, 6e-4, 1e-3 before pilot |
| Warmup / decay | 400 updates linear warmup, then cosine to 0.1 x chosen LR at update 8,000 |
| Pilot length | **8,000 logical optimizer updates**; nominal 65,536,000 target-byte exposure |
| Checkpoint cadence | Step 0, every 250 updates, endpoint, and graceful stop at complete update boundary |
| Validation cadence | Step 0, every 250: fixed 262,144-byte monitor; full confirm holdout every 1,000 and endpoint |
| Generation cadence | Step 0, every 500 updates, endpoint; exact prompt/sampling policy below |
| Context diagnostics | Step 0, every 1,000 and endpoint, fixed domains/anchors/phases |
| Cheap monitoring | Finite loss/gradients/update parameters every update; time/count/LR/norm every update |
| Detailed memory | First 10 updates and every 100; pre/post checkpoint and failure |
| Runtime ceiling | <=24 cumulative GPU hours including evaluation/export; no automatic extension |

The data budget and update exposure are distinct. Short documents produce tail
windows; count actual targets, never label padding/BOS as raw-byte exposure and
never drop tails just to match the nominal total. One epoch is not promised by
8,000 updates. Preserve current summed-NLL/total-valid-target normalization across
all accumulation microbatches, cursor order, lazy Adam state and schedule indexing.
Accumulate gradients sequentially; do not retain 16 computation graphs. No precision,
loss, optimizer, data-order or model-equation changes are bundled with fast monitoring.

LR selection: three independent fresh arms, same seed 17, identical data order and
initial parameter values, each 256 updates and <=2,097,152 nominal target bytes;
32 warmup updates, then constant trial LR. A constant post-warmup probe schedule
requires an explicit supported runner; it is not the existing cosine schedule with
silently altered parameters. Bind its identity. Cap total probe compute at 3 GPU hours.
Hold out a disjoint 262,144-byte LR-selection subset from the 4,096,000 validation
bytes; choose monitor 262,144 from the remaining 3,833,856 confirmation bytes. Select
the stable arm with lowest endpoint selection NLL; within 0.01 nats/byte choose the
lower LR. Stability requires finite states/loss/gradients/parameters, >=0.1 nats/byte
improvement over initial selection NLL, no three consecutive updates with loss
>2x median of preceding 32 after update 64, and clipping fraction<=0.5 in last 128.
If none qualify, STOP and revise the experiment before more updates. No cherry-picked
checkpoint or continuous LR sweep. Pilot is reinitialized; probe states are not resumed.
6e-4 is provisional, not a justified optimum or permission to skip probes. Defaults
other than LR retain known optimizer semantics to limit simultaneous changes.

## Analytic memory ledger

Exact payloads for selected shape at B=8,L=64 appear below. Allowances are explicitly
chosen engineering estimates, neither measured peaks nor rigorous upper bounds.
Use GiB=2^30; T=ceil(L/8), K=12,D=512,I=1024,N=64,H=16,Q=256,E=64,V=267.

| Item | Bytes | Basis |
|---|---:|---|
| FP32 parameters | 81,550,124 | 4P exact |
| FP32 gradients | 81,550,124 | 4P maximum populated |
| Adam first moment | 81,550,124 | 4P maximum populated |
| Adam second moment | 81,550,124 | 4P maximum populated |
| Adam scalar steps | 512 | 128x4, lazy actual coverage may be smaller; CPU placement implementation-dependent |
| Canonical SSM state | 25,165,824 | 4BKIN |
| Canonical convolution state | 1,769,472 | 4BK(I+2N)d_conv |
| Decoder hidden / pending symbols | 8,192 /448 | 4BQ /8Bx7 |
| State working copies allowance | 53,870,592 | Two extra SSM+conv copies |
| Trunk saved activations allowance | 70,975,488 | 4BTK(12D+16I+8N+4H) |
| Local saved activations allowance | 7,909,376 | 4BL(12Q+4E+2V) |
| Encoder saved activations allowance | 688,128 | 4B(T-1)x8x6E |
| Context vectors | 131,072 | 4BTD |
| Reference SSD temporary allowance | 2,409,984 | 4KBT^2(6H+2)+2KT^2 bool-mask bytes |
| Optimizer temporary allowance | 163,100,248 | 8P conservative full-model scratch allowance |
| Kernel workspace allowance | 1,073,741,824 | 1GiB, unmeasured |
| Allocator reserve/fragmentation | 1,073,741,824 | 1GiB, not extra live tensors |
| CUDA context/framework allowance | 536,870,912 | 0.5GiB, unmeasured driver/process device overhead |
| **Planning total** | **3,336,584,392** | **3.1074GiB**, not a measured VRAM peak |

The autograd coefficients reserve intermediate arrays, not just final outputs;
einsum contraction choices, norm/GRU saves and allocator reuse can change actual use.
Recurrent states are included conservatively even though the trainer uses full-window
forward, not a persistent streaming cache. Separate planning terms may overlap in
lifetime; do not present this sum as exact allocation. Host RSS is separate from VRAM.
Reserve 2 GiB base host process/framework plus 1 GiB loaders/provenance for the pilot,
then measure the actual window objects/permutation lists and checkpoint serialization.
The current eager loader may cost hundreds of MB beyond raw bytes for ~1M windows;
the serious corpus at 64 would create ~31.25M windows and requires indexed loading.

| Hardware/profile | Analytic planning use | Classification | Measured status |
|---|---:|---|---|
| RTX 2050 4 GB, B2 L64 acc64 | 2.9936 GiB | **FITS_WITH_CONSTRAINTS** | **UNVERIFIED**; display/other processes consume headroom |
| One T4 16 GB, B8 L64 acc16 | 3.1074 GiB | **FITS_COMFORTABLY** analytically | **UNVERIFIED**; actual GPU/runtime still a gate |
| T4x2, per-rank B8 L64 acc8 | 3.2593 GiB/rank plus unmeasured NCCL workspace | **UNVERIFIED** | DDP/runtime/state protocol unimplemented |

DDP allowance adds8P=163,100,248 bytes/rank for two parameter-sized buckets; neither
weights nor Adam states are divided by 2. Two 16 GB cards are not one 32 GB allocation.
For one T4 at B8, predicted totals L32=3.0686GiB, L128=3.1885GiB and
L256=3.3641GiB. These fit estimates do not make the longer lengths useful or fast.
RTX B2/acc64 is a separately bound fallback profile, not a silent mid-run switch;
different microbatch arithmetic grouping needs parity and new evidence.

Before pilot replace the hard 1 GiB allocator policy with versioned named budgets:
T4 allocator cap 8 GiB and >=2 GiB observed free headroom; RTX cap 2.5 GiB and >=0.5 GiB
observed free headroom. Process overhead and other users count against driver free
memory. If the actual phase peak exceeds a cap, STOP; change profile only before a
new run, rerun parity, and never call analytic fit a measured pass. CPU mechanics is
the fallback when the RTX cannot satisfy the gate.

## Reference-paranoid and production-fast

**REFERENCE_PARANOID** verifies the active TRAIN/VALIDATION-only corpus snapshot
at startup, every update and every resume; retains all internal finite diagnostics,
per-update synchronized memory and detailed tensor/state inventory. It is the
correctness oracle on bounded fixtures. Neither mode may open production TEST.

**PRODUCTION_FAST** hashes active immutable TRAIN/VALIDATION files and binds the
manifest at startup/resume, hashes each shard before first use/reuse, checks snapshot
identity cheaply each update and fully verifies at durable-checkpoint/export
boundaries. For the eager pilot, consume the already-hashed immutable in-memory byte
snapshot; changing a disk file must not substitute new unverified training bytes.
For indexed data, cache verified immutable shard buffers. Size/mtime alone is not
proof of integrity. External disk mutation may be detected later than paranoid;
record that detection-latency difference honestly and stop on any mismatch.

Keep finite loss/gradient/post-update parameter guards every update. Detailed layer
finite checks can only be fused/deferred after synthetic fault tests show failure
before checkpoint publication; never remove numerical failure checks silently.
Move expensive tensor inventory/canonical-state allocation off the hot path, and
sample allocator/RSS metrics at the declared cadence. Do not let diagnostic evaluation
consume training RNG: use saved/restored RNG and isolated generation generators,
restore model mode, and include monitoring state in evidence. Checkpoint contents,
atomic writes and verification are identical in both modes.

Required mode equivalence test after implementation: same backend/FP32, synthetic
multi-document tails, identical data order/seed and first 10 updates, save at 5 then
resume. Require bitwise equal loss inputs, weights, moments, RNG, cursor and schedule
at every update (ignore timings/log volume). Test disk mutation, changed manifest,
nonfinite gradient and interrupted checkpoint negative cases. Monitoring parity does
not justify changing fused optimizer, accumulation order, model kernels or precision.

Model bottlenecks separately: layer launches/norms/synchronizations; dense projections;
O(BKHT^2) transition storage/work plus contraction cost; serial byte GRU readouts;
H2D/tail batch formation; startup and boundary corpus hashing; checkpoint hashing and
host serialization; validation/generation; Python/RSS logging. At 2 GB, verifying the
whole corpus per update scales with corpus size even though target bytes/update stay
8,192. Clean execution removes that I/O amplification, not the model's T^2 equations.

Throughput table entries for old/clean reference RTX, clean reference T4, optimized
T4 and dual T4 are all **NEEDS_MEASUREMENT**. The historical 2M 165.36 target bytes/sec
is not a 20M predictor. Future measurement: exclude startup from kernel timing but
report it separately; 20 warmup +100 measured synthetic updates, synchronize timing,
report median/p95 and actual valid bytes/sec. Measure eval/export separately and
also report end-to-end useful bytes/sec including them. Use three repeats and disclose
variance/thermal throttling. The pilot must project <=24GPU hours with30% time reserve;
the serious run must project <=120GPU hours with30% reserve. Equivalently, useful
end-to-end rate must be >=986.1 bytes/sec for nominal pilot exposure, and
>=6,018.6 bytes/sec for 2B serious bytes, including the reserve multiplier 1.3.
Failure blocks the run; optimize or request a revised budget, not optimistic extrapolation.

## Exact refactor gates

These are scoped requirements for a later implementation tranche, not edits performed
by this audit. Preserve the reference path and old configuration/checkpoint readers.

| ID / item | Classification | Required acceptance before proceeding |
|---|---|---|
| R1 resolver ceiling | MUST_FIX_BEFORE_20M_PILOT | Use explicit selected dimensions and versioned 20M search (widths 384,512,640; depths 8,12,21); demonstrate 20M selection and exact executable accounting. The new YAML solves explicit construction; do not change old defaults or 2M results |
| R2 stale resolution metadata | MUST_FIX_BEFORE_20M_PILOT | Separate inventory/construction proof from forward/training readiness; remove false 'not implemented' status only in current resolver output, preserve old reports |
| R3 current README/state | MUST_FIX_BEFORE_20M_PILOT | Link frozen 2M, reconciled history and current pilot restrictions; never describe old blocked provenance as current status. Audit state entry is supplied; README remains pending |
| R4 repeated full corpus verification / sealed TEST | MUST_FIX_BEFORE_20M_PILOT | Immutable verified active-data binding, mode policy, no TEST paths in runtime manifest, corruption/fail-closed tests and startup/resume/export verification |
| R5 paranoid/fast monitoring | MUST_FIX_BEFORE_20M_PILOT | Versioned cadences, RNG/mode isolation, identity-derived static tensor accounting, numerical guards and exact 10-update mode/resume equivalence |
| R6 Linux RSS and path/process portability | MUST_FIX_BEFORE_20M_PILOT | Linux current/peak RSS with explicit units/status; relative POSIX manifest paths and portable subprocess/sys.executable handling; CPU notebook smoke and checkpoint round-trip |
| R7 device budgets/one visible GPU | MUST_FIX_BEFORE_20M_PILOT | Named cap/headroom profiles, device mask before initialization, actual hardware diagnostic; keep single-visible-device invariant for first pilot |
| R8 fixed 56-tensor and production asserts | MUST_FIX_BEFORE_20M_PILOT | New current 20M runner derives names/shapes/counts (selected 128), uses explicit exceptions even under python -O, retains historical scripts unchanged |
| R9 pilot runner and gates | MUST_FIX_BEFORE_20M_PILOT | Implement bound LR probe schedule, manifests/counters, diagnostic metrics, predeclared thresholds, atomic durable checkpoints and stage stopping; no hard-coded expected test counts |
| R10 backend interface and optimization | MUST_FIX_BEFORE_FULL_20M | Versioned forward/step/state/checkpoint backend identity; sm75-compatible optimized implementation passes every parity category and throughput gate; reference remains fallback |
| R11 indexed/sharded data and evidence scaling | MUST_FIX_BEFORE_FULL_20M | Preserve window/tail/order/cursor semantics without eager millions of Python payload objects; measured host memory, shard integrity and resume tests; reusable active evidence helpers |
| R12 true multi-GPU | CAN_DEFER | DDP, rank RNG/cursors and device-count relaxation unnecessary for single-T4 pilot; must be completed before any dual-T4 experiment |
| R13 historical script deduplication | CAN_DEFER | Leave immutable historical evidence generators alone; only share new runner helpers where correctness needs them |

R1-R9 are mandatory before ANY 20M pilot; R10-R11 before serious training. R7 cannot
be satisfied by deleting the exactly-one check; a launcher mask is sufficient for
the first pilot. R8 concerns active runtime checks, not replacing valid test asserts.
R6 includes Windows-specific absolute roots and process assumptions in new orchestration,
not a broad portability rewrite of historical scripts. Run model/trainer/checkpoint
synthetic regressions whenever these infrastructure changes are implemented.

## Generation and context promotion: predeclared before results

Fixed UTF-8 prompts (exact): empty/BOS `b''`, `b'The '`, `b'Once '`, `b'def '`,
`b'import '`, `b'class '`, `b'x = '`, `b'Let x = '`,
`b'Parameters\n----------\n'`. Do not select replacements after inspecting output.
Generate 256 bytes per prompt: greedy plus temperature 1 byte-renormalized sampling
with independent seeds 101,202,303,404; no top-k/top-p, repetition penalty or whitespace
suppression. Always measure unfiltered 267-class probabilities before byte-only
renormalization so control collapse cannot be hidden. Record termination/error separately.

For each trace report whitespace fraction over bytes {9,10,11,12,13,32}, printable
ASCII fraction [32,126] plus tab/LF/CR, longest identical-byte run, empirical byte
entropy in bits, mean/median q(space), unfiltered predictive entropy in bits,
top1-top2 probability margin, sum of control-symbol probability, UTF-8 validity
(whole emitted bytes strictly decoded), distinct 1/2/4-grams and unique outputs.
Also report per-domain/prompt and late-half metrics so averaging cannot conceal the
2M whitespace attractor. PAD/BOS masking remains as the reference defines it;
other control mass is still measured. Never count control IDs as bytes.

Quality gate for stage 20M-5 (not tiny smoke): all 45 traces complete without invalid
state; mean unfiltered control mass<=0.01 on each prompt; greedy median whitespace
<=0.65 and no greedy prompt>0.90; >=7 of 9 greedy prompts have longest run<=32 and
empirical entropy>=2bits; greedy late-half median q(space)<=0.35 and no trace has
32 consecutive positions with q(space)>0.8; printable fraction>=0.90 on>=7/9 greedy
prompts; >=33/36 sampled outputs are valid UTF-8; >=7/9 prompts have>=3 distinct
sampled outputs among 4 seeds; >=7 distinct greedy 64-byte suffixes across 9 prompts.
No universal top1/top2 margin floor is imposed (certainty is context-dependent),
but log median/p95 and stop promotion if margin>=0.95 for 32 consecutive steps in
any greedy trace. These are anti-collapse operational thresholds, not prose-quality
or code-correctness benchmarks. Human review remains necessary.

Prompt conditioning: measure pairwise Jensen-Shannon divergence (natural logs) of
the byte distributions at the first generated position for all prompt pairs;
median must be>=0.005 nats. This can reflect local prefixes and is not a long-history
claim. For history use 128 fixed validation anchors, 32/domain with>=16 documents/domain,
identical last 32 bytes and target block, compare short 32 versus long 128 consumed
history, all 8 patch phases. Pin eligible anchor IDs before training; if insufficient
anchors, stop the context gate, do not weaken its sample size. No TEST anchors.

Require mean(short NLL - long NLL)>=0.01 nats/byte and a positive 95% paired
document-cluster bootstrap CI (2,000 resamples, seed 43); long history must beat shuffled
old-history control by>=0.005 nats/byte with positive CI. On>=75% of anchors require
max byte-distribution total variation across phases>1e-5. Record shared/conv/SSM,
pre-tanh, hidden/logit deltas, tanh saturation and gradients. State differences alone
never satisfy this gate. Evaluate each domain and phase; no domain history delta
below-0.02 nats/byte. For A/B promotion require B satisfy all these gates, improve
confirmation NLL by>=0.01 nats/byte over A with positive paired document CI, and no
domain degradation>0.02. Repeat the direction of benefit at seed 29. If A also passes,
retain A unless B earns this gain; if neither passes, stop full-run promotion.

## Stage decisions

Every stage requires all earlier stages; no favorable threshold selection after
results. Failures produce a receipt, not automatic retries with a different seed.

| Stage | PASS | STOP | ROLLBACK |
|---|---|---|---|
| 20M-0 construction | Exact selected count/signature, no unclassified tensors, config round-trip; reference invariants | Count/shape/config mismatch | Retain frozen 2M and previous candidate receipt |
| 20M-1 forward/backward | Finite synthetic loss/grads, every participating tensor accounted, causal prefixes exact, batch/step outputs/states within architecture parity tolerances | Nonfinite, causal leak, unexpected absent grads, wrong state/masks | Discard disposable 20M instance; return to reference equations |
| 20M-2 GPU memory/throughput | Real hardware profile, phase peaks below cap/headroom, measured pilot budget including30% reserve, exact same-backend resume | OOM, unsupported device/kernel, parity failure, projected time too high | CPU/reference fixture or separately bound smaller-batch profile; no real pilot |
| 20M-3 tiny overfit/mechanics | Fixed TRAIN fixture b'edge400: abcdef\n' repeated 64, 60 updates B2 acc2 L32 LR 0.003 warmup 5; >=50% train-NLL reduction; distinct validation logged; exact resume at 30; real-data smoke up to 128 updates has finite metrics | Mechanical/restore failure; no prose requirement at this stage | Discard smoke outputs and restore last verified disposable checkpoint for debugging only |
| 20M-4 real bounded pilot | R1-R9 complete; LR probes pass; 8,000 updates within budget; full confirmation NLL improves>=0.10 from initialization and every domain improves>=0.02; checkpoint/export/byte counts consistent | Instability rule, data mismatch, budget exceedance, missing evidence or NLL criterion failure | Keep last verified pilot checkpoint unpromoted; freeze for review, no 2M replacement |
| 20M-5 quality/context | All fixed generation, useful-history and bridge A/B criteria above; selected endpoint only, no best-of-many substitution | Whitespace attractor, history insensitivity, heldout regression, invalid UTF-8/control collapse or failed replication | Preserve baseline pilot and rejected arms, no serious-data run |
| 20M-6 human full-run review | R10-R11 complete; approved provenance/capacity, optimized parity, <=120GPU-hour estimate, durable resume, sealed TEST and explicit human compute approval | Any missing item or withheld approval | Stay at bounded pilot; no automatic acquisition/full training |

The tiny fixture's LR 0.003 is a historical mechanics control, not the real-data LR.
Smoke unique corpus budget is not permission to train to convergence on it. Bridge
A/B is required even if the first baseline pilot looks promising; retain baseline
if the intervention does not earn promotion. Stages1-6 are **NOT RUN** in this audit.
Stage 0 construction accounting alone is **PASS**. TEST is never a selection gate.

## Preservation and rollback

Only new design/config/accounting and current audit records change. The completed 2M
weights, computational source, old configurations, Bible/Roadmap/Addendum and
immutable authorization snapshot remain unchanged. Prior partial audit is retained
in Git at aa7e047 and within the machine-readable audit history; its outdated blocker
is explicitly resolved by 6151a966. The original mechanical search fields remain intact.
Revert this additive audit commit to remove the candidate and design without changing
the frozen model or reference fallback. Required refactors are pending, measured
memory/throughput and learning claims are unverified, and this design is not a launch approval.
