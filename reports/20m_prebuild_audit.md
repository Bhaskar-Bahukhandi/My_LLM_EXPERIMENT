# 20M pre-build audit — completed design

Date: 2026-09-27. Decision: **20M_PREBUILD_READY_WITH_REQUIRED_REFACTOR**.
Exactly one candidate is **RECOMMENDED_FOR_20M_PILOT**; none is production approved.
No 20M training, model forward/backward, optimizer update, production corpus payload
read, dataset acquisition or Kaggle action occurred in this resumed tranche.

## Resume scope and evidence

Entry HEAD 6151a9662a9cc40fa7dfcc2266f36adfc8ca8383 clears the former stop through
CASE_A_HISTORICAL_SNAPSHOT_INTACT / CURRENT_2M_LINEAGE_UNAFFECTED. The historical
investigation was not repeated. Mandatory historical replay and preservation checks
verify continued integrity. The original expected report SHA and original snapshot
SHA remain authoritative; see their exact, unchanged identities in the
[reconciliation report](historical_authorization_provenance_review.md).

The previous partial audit at aa7e047 is reused: frozen 2M/checkpoint identity, source
findings, resolver ceiling 5,385,771, reference quadratic SSD analysis, 517 grid shapes
and 4 targeted controls. No repeated grid search or endpoint weight load. The original
mechanical receipt fields are unchanged; `resumed_accounting` appends the selected
meta/YAML check, comparisons and analytic ledgers. JSON retains the old partial
decision as explicitly superseded history; Git retains the entire prior Markdown.
Only the reconciliation harness/docs/tests changed since that source audit; `src/`
and old model configs remain unchanged. The source findings therefore still apply.

Correction retained from reconciliation: old plain pytest included production TEST
integrity reads, so the previous blanket no-access claim was too broad. That was not
TEST evaluation/training. This audit uses the reviewed non-training test allowlist and
never invokes the production corpus integrity modules or reads production payloads.

## Selected design

**512 width x 12 Mamba layers, state 64, byte 64, decoder 256**: **20,387,531 parameters**,
128 unique trainable tensors. The real meta model agrees with formula/inventory,
named tensor shapes and the authored YAML round-trip. The resolved/tensor digests
and complete component ledger are in the mechanical JSON. Meta has no weight values.

Allocation: Mamba excluding norm 19,828,800 (97.2594%); symbols 17,088; positions 512;
encoder 41,856; bootstrap 33,280; bridge 131,328; local GRU 247,296; head 68,619; norms 18,752.
The previous 512 x 12/decoder 128 candidate assigns 98.5783% to the trunk. This is not
proof of bad capacity, because parameter sharing and byte/patch clocks differ, but
the observed trained output insensitivity makes local capacity a reasonable risk
to address. Decoder 256 buys functional capacity for 272,768 parameters with unchanged
equations. It does not prove context adoption or cure tanh suppression.

Retained alternatives: **512 x 12/decoder 128 (20,114,763)** for allocation control and
**384 x 21/decoder 128 (20,061,951)** for depth/width comparison. Widths 256/320 imply
30-46 serial layers and larger state; widths 896/1024 concentrate similar projection
work in only 3-4 layers. Selected width 512 is aligned, moderate-depth and retains
more local capacity. No candidate's GPU utilization or quality has been measured.
Byte 128 and state 128 controls remain available but are not mixed into the first pilot.

Context decision: **REQUIRE_CONTEXT_BRIDGE_AB_BEFORE_FULL_20M**. Start the bounded
pilot with the existing bridge; then compare it against gated residual global readout
at fixed architecture/data/LR/length. Compare unchanged, gated residual, per-byte
injection, FiLM and normalized conditioning with exact parameter/FLOP/cache equations
in the [architecture contract](../docs/20m_architecture_contract.md). No bridge code changed.

**FIRST_PILOT_SEQUENCE_LENGTH=64**; assess 32/64/128/256 separately. At fixed B,
reference SSD patch-time work ratios are 1/4/16/64. Mechanical recurrent history can
exceed a training window; useful learned history remains unproven. No automatic
length curriculum; longer lengths require positive paired heldout context benefit.

## Pilot, data and resource decision

Unique TRAIN budgets: smoke **1,000,000**, pilot **65,536,000**, serious
**2,000,000,000** raw bytes. Validation 100,000 / 4,096,000 / 20,000,000; proposed new
sealed TEST reserve 0 / 4,096,000 / 20,000,000. Total corpus budgets 1,100,000 /
73,728,000 / 2,040,000,000. Retain 50/25/15/10 general/code/docs/math mixture. Exact
domain quotas, document/diversity targets, license/provenance/dedup/contamination
requirements are in the [training plan](../docs/20m_training_plan.md). New TEST
preparation is future separately authorized work, not permission to open current TEST.

First real-data pilot: seed 17, FP32, **microbatch 8 x accumulation 16**, nominal 128
sequences and 8,192 valid bytes/update at length 64; actual tail counts govern loss.
Fresh initialization, 8,000 updates, 400 warmup, cosine to 0.1 x LR; AdamW (0.9, 0.999),
epsilon 1e-8, decay 0.01 with existing exclusions, clip 1.0. Candidate LR 6e-4 is provisional:
three bounded fresh 256-update arms at 3e-4/6e-4/1e-3 must pass the predeclared selection
rule first. Checkpoints/monitor validation every 250 updates, full confirmation/context every 1,000,
generation every 500; maximum 24 GPU hours including monitoring/export. No long training authorized.

The full analytic ledger includes weights, gradients, both moments, scalars, recurrent
states, activations, temporary SSD tensors, optimizer scratch, workspace, allocator
and framework overhead. B8/L64 planning sum is **3,336,584,392 bytes = 3.1074 GiB**.
RTX 2050 B2/L64/acc64: **FITS_WITH_CONSTRAINTS**, planning 2.9936 GiB; single T4:
**FITS_COMFORTABLY analytically**. **Measured fit for both is UNVERIFIED**. T4x2 is
**UNVERIFIED**, about 3.2593 GiB/rank plus unmeasured communication workspace; VRAM
is not pooled. These are allowances, not benchmark results or rigorous upper bounds.

REFERENCE_PARANOID and PRODUCTION_FAST retain loss, optimizer, data order,
accumulation, equations and checkpoint contents. Fast mode consumes verified immutable
TRAIN/VALIDATION buffers, shifts full hashes to declared boundaries and reduces heavy
instrumentation. Integrity-detection latency differs explicitly. Bitwise synthetic
update/resume parity is mandatory. All 20M throughput entries: **NEEDS_MEASUREMENT**.

Optimized backend: **MANDATORY_ONLY_BEFORE_SERIOUS_20M**; reference allowed for
bounded pilot only if it meets measured budget. Current upstream Triton is not an
accepted T4 backend: [NVIDIA lists T4 as 7.5](https://developer.nvidia.com/cuda/gpus),
whereas [Triton lists 8.0+](https://github.com/triton-lang/triton#compatibility), and
[upstream Mamba-2 SSD imports Triton](https://raw.githubusercontent.com/state-spaces/mamba/main/mamba_ssm/ops/triton/ssd_combined.py).
No compatible optimized backend has been validated. Forward/step/state/causality,
gradient, seeded optimizer and checkpoint parity thresholds precede any promotion.

Kaggle remains user-supplied READ-ONLY VERIFIED at bridge 4cb387d...06e1. No bridge
operation was needed. [Execution plan](../docs/kaggle_20m_execution_plan.md): separately
approved private write smoke -> CPU notebook -> T4 diagnostic -> mechanics -> pilot.
Preserve LOCAL_ONLY/HOST_READ/HOST_WRITE/HOST_COMPUTE, isolated credentials and explicit
approval-bound host execution; no arbitrary shell or sandbox-network workaround.

## Mandatory refactors and promotion

Before ANY 20M pilot complete **R1-R9** from the training plan:

1. Explicit/versioned 20M resolver selection and executable accounting.
2. Evidence-scoped resolver readiness metadata.
3. Current README/state index and pilot restrictions.
4. Immutable active-data verification without TEST access or per-update corpus rereads.
5. Paranoid/fast monitoring with exact synthetic update/resume equivalence.
6. Linux RSS and path/process/checkpoint portability.
7. Named GPU memory budgets and one-visible-device launcher/diagnostic.
8. Derived 128-tensor inventory and explicit production errors rather than correctness asserts.
9. Bound LR/pilot runner, counters, diagnostics, stopping and durable checkpoint gates.

The selected explicit YAML already solves construction beyond the old automatic
search ceiling; old default searches and frozen configs are preserved. Infrastructure
refactors are **not implemented** here. Before serious training also complete R10
(optimized backend/interface/parity) and R11 (indexed data/evidence scaling). True DDP
and historical script deduplication can defer. No broad aesthetic refactor is proposed.

Predeclared stages: 20M-0 construction **PASS**; 20M-1 forward/backward, 20M-2 GPU,
20M-3 tiny overfit/mechanics, 20M-4 bounded real data, 20M-5 quality/context all
**NOT RUN**; 20M-6 requires human full-run approval. Each has explicit PASS/STOP/ROLLBACK
in the training plan. Generation uses 9 fixed prompts, greedy plus 4 fixed seeds and
256-byte traces; tracks whitespace attractor, printable fraction, repeated runs,
entropy, q(space), top1/top2 margin, control mass, UTF-8, diversity and divergence.
NLL alone cannot promote. Tiny smoke need not produce polished prose. Context
promotion requires useful paired history benefit, not merely changing shared state.

## Validation, preservation and next step

Closeout preserves substantive commit `af4e29bc04d8a2cc776870cbe657948170e50b3a`.
The requested rerun passed: **140 current non-training tests**, **30 historical
tests** through the mandatory replay, and **1 replay wrapper**. Ruff, format check
(104 files), compileall, CPU/CUDA dependency checks and diff checks passed. The
pre-existing optional NumPy warning remains; no test-count constant determines success.
Fresh meta construction reconfirmed 20,387,531 parameters, 128 unique trainable
tensors, the exact disjoint component sum and YAML/resolved identity.

The frozen source and checkpoint hashes match accepted evidence; the checkpoint
was hashed without deserialization. All 66 historical snapshot files, membership,
manifest identity and expected authorization hashes remain unchanged. No production
TEST payload, model forward/backward, optimizer update, Kaggle operation or dataset
acquisition occurred. R1-R9 implementation has not begun.

The final validation receipt's `final_design_artifact_sha256` binds nine final local
artifacts, including all seven required design artifacts; it excludes itself. A
separate LF-normalized map supports comparison across Git line-ending conversions.
Closeout changes only the audit Markdown/JSON, validation receipt, three design plans
and current project-state prose. Mechanical accounting, YAML and code are unchanged.
Reverting the additive closeout commit retains the substantive design at `af4e29bc`.

Actual commands, outcomes, timings and preservation hashes are recorded in
`20m_prebuild_validation.json`. The current non-training allowlist and mandatory
historical replay run independently; static, format, compileall, CPU/CUDA dependency
and diff checks apply. No hard-coded test count is an acceptance criterion. Broad
training or production TEST integrity suites are deliberately excluded; no infrastructure
source changed. No 20M GPU/forward/backward/performance/learning claim is validated.

The completed design tranche changed four current audit reports; added architecture/training/Kaggle plans;
selected model YAML; additive meta accounting helper; project-state status entry.
Frozen 2M source/weights/configs, historical evidence/snapshot, specs and reference
fallback remain intact. The previous audit's original fields and Git history remain
recoverable. Revert this additive commit for rollback; it neither retrains nor migrates 2M.

Risks: selected local capacity may not improve quality; context bridge may remain
insensitive; memory allowances may miss peaks; reference throughput may miss the
pilot budget; current optimized upstream kernels do not provide a supported T4 path;
data-diversity quotas and notebook durability are unverified. These are enforced
future gates, not hidden execution claims. Next best step is a scoped implementation
of R1-R9 with synthetic model/trainer/checkpoint regression validation, then separately
approved hardware/mechanics gates. No 20M training is authorized by this design decision.

20M PRE-BUILD STATUS: READY WITH REQUIRED REFACTOR
