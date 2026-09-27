# 20M candidate architecture contract

Status: **RECOMMENDED_FOR_PILOT**, never PRODUCTION_APPROVED. Design decision:
**20M_PREBUILD_READY_WITH_REQUIRED_REFACTOR**. No training is authorized here.
Entry: `6151a9662a9cc40fa7dfcc2266f36adfc8ca8383`; frozen 2M source is unchanged.

## Selected shape and scientific question

Exactly one candidate is **RECOMMENDED_FOR_20M_PILOT**:
`d512_l12_n64_b64_q256`: d_model=512, 12 shared layers, d_state=64,
byte_dim=64, decoder_dim=256, expand=2, d_inner=1024, 16 heads of width 64,
d_conv=4, patch_size=8, chunk_patches=16, 267 output symbols, MoE disabled.
Baseline context equations and FP32 arithmetic remain unchanged.
`configs/models/edge_20m_candidate.yaml` is an executable inventory configuration;
its CPU backend field is not a claim of T4 execution readiness.

The real DenseByteModel was constructed on meta twice: selected shape, then YAML
round-trip. Its **20,387,531** unique trainable parameters in **128 tensors** match
the retained search result, independent formula, inventory and shape multiset.
Named tensor signature:
`3f242a2332fa41d4737f5d184b1593051bf8384a7cb5e954f457c136848dfd61`.
Meta establishes tensor identities/shapes/counts, not a parameter-value hash,
forward correctness, trainability or quality. No tensor values were materialized.

| Disjoint component | Parameters | Share |
|---|---:|---:|
| Shared Mamba excluding normalization | 19,828,800 | 97.2594% |
| Symbol embeddings | 17,088 | 0.0838% |
| Position embeddings | 512 | 0.0025% |
| Patch encoder excluding normalization | 41,856 | 0.2053% |
| Bootstrap | 33,280 | 0.1632% |
| Context bridge | 131,328 | 0.6442% |
| Local GRU | 247,296 | 1.2130% |
| Output head | 68,619 | 0.3366% |
| All normalization | 18,752 | 0.0920% |
| **Total** | **20,387,531** | **100%** |

The 98.5783% trunk share of the 512x12/decoder128 control is a risk signal, not
proof of unhealthy allocation: the global path runs once per eight bytes, whereas
small local weights are reused each byte. Parameter shares do not measure useful
capacity or computation shares. Nevertheless, retaining a 128-wide decoder while
greatly enlarging the global network leaves a plausible output bottleneck.
Decoder256 buys an actual higher-rank bridge, a larger nonlinear local state and
a larger readout for 272,768 parameters (+1.36% over that control). It is a single
local-width change, not padding or a new mechanism; it does not cure saturation by
assertion. The trunk is still dominant. Do not widen the encoder and change the
bridge simultaneously in the first pilot. The pilot tests this allocation, not
whether the 2M checkpoint would improve by continued training.

## Retained comparison, using the original constructions

MAC means one multiply-accumulate; approximate projection FLOPs are twice MACs.
The projection column is exact dense in/out projection MACs per global patch
across all layers, excluding norms, conv, SSD and local decoding. State includes
one sequence's FP32 SSM+conv+decoder and seven pending int64 symbols.

| Shape (width x layers; other dimensions baseline) | Parameters | Projection MAC/patch | State bytes/sequence | Assessment |
|---|---:|---:|---:|---|
| 256 x 46 | 20,074,587 | 19,689,472 | 6,500,920 | Most serial launches, longest layer gradient path, narrow GEMMs; reject for first GPU pilot |
| 320 x 30 | 20,119,695 | 19,756,800 | 5,284,408 | Still deep and state-heavy; no measured advantage |
| 384 x 21 | 20,061,951 | 19,708,416 | 4,430,392 | Retained depth/width alternative; 75% more layer boundaries than selected |
| 512 x 12, decoder128 | 20,114,763 | 19,759,104 | 3,367,480 | Retained allocation control; smallest change from the old local path |
| **512 x 12, decoder256** | **20,387,531** | **19,759,104** | **3,367,992** | **Selected: moderate depth, aligned width, stronger local capacity** |
| 640 x 8 | 20,791,275 | 20,418,560 | 2,802,232 | Less serial work, higher projection cost, fewer sequential transforms |
| 896 x 4 | 20,242,779 | 19,826,688 | 1,958,456 | Wide kernels may utilize GPU better, but little depth; unmeasured quality risk |
| 1024 x 3 | 19,807,659 | 19,365,888 | 1,677,880 | Lowest depth/state; not chosen merely because its parameter count is close |
| 512 x 12, byte128 | 20,247,499 | 19,759,104 | 3,367,480 | Encoder/embedding capacity control; does not directly bypass the observed decoder insensitivity |
| 512 x 12, state128 | 20,908,875 | 20,545,536 | 6,537,784 | Roughly doubles recurrent memory; larger latent state is not evidence of decoder adoption |
| 512 x 11, state128 | 19,188,763 | 18,833,408 | 5,993,016 | State/depth tradeoff control; same adoption uncertainty |

All use legal head-aligned widths. Similar projection work does not imply similar
time: narrower/deeper paths have more launches, reductions and serial dependency.
Microbatch and patch count determine GEMM occupancy; GPU utilization for every row
is **NEEDS_MEASUREMENT**. The selected local GRU increases byte-clock compute and
autograd memory; its benefit must justify that cost. All choices use the same tested
dimension-driven implementation, so shape changes have lower implementation risk
than new operators. The middle shape avoids the extremes without a quality claim.

Future MoE: 512/1024 dimensions are conventional partitionable interfaces, but
there is no implemented router, expert block, load-balancing objective or MoE
checkpoint format. None of these candidates is certified MoE-compatible. Keep a
versioned insertion interface as future work; no reserved dummy parameters.

## Context bridge decision

**REQUIRE_CONTEXT_BRIDGE_AB_BEFORE_FULL_20M**. First bounded pilot is baseline A.
The accepted 2M studies establish differing old-history shared state with identical
trained hidden/logits on their probes; they do not prove tanh is the sole cause.
Tenfold scale is not accepted as a remedy.

Definitions for exact design arithmetic: D=512, Q=256, E=64, P=8. Global context c
contains BOS and completed earlier patches only. Baseline u=Wc+b, h0=tanh(u), then
causal GRU updates. A prediction is read before consuming its target byte.
Counts below are incremental relative to A; formulas specify the proposed variants
precisely, although none is implemented or validated here. FLOPs count multiply/add
as one each and list nonlinear calls separately; bias adds are included. These are
forward overheads, not backward FLOPs or a kernel benchmark.

| Option | Equation and exact extra parameters | Forward overhead per full eight-byte patch | Extra persistent inference state per sequence |
|---|---|---|---|
| A unchanged | u=Wc+b; h0=tanh(u); **0 extra** (bridge has 131,328) | **0 extra**; baseline projection 262,144 MAC-convention FLOPs +256 bias adds +256 tanh calls | 0 extra |
| B gated residual global readout | readout=RMSNorm(h+sigmoid(a)*u), learn vector a in R^Q; **256 extra** | Precompute v=sigmoid(a)*u: 256 sigmoid +256 multiplies; add v at each of 8 readouts: 2,048 adds; **2,304 arithmetic +256 sigmoid** | Cache v: **1,024 bytes** |
| C per-byte embedding injection | e'=e+(Wg*c+bg), Wg: D->E; **32,832 extra** | **65,536** projection FLOPs +64 bias adds +7x64 adds for consumed prefix bytes = **66,048 arithmetic** | Cache Wg*c+bg: **256 bytes** |
| D FiLM readout | gamma=tanh(Wgamma*c+bgamma); beta=Wbeta*c+bbeta; readout=RMSNorm((1+gamma)*h+beta); **262,656 extra** | **524,288** projection FLOPs +512 bias +256 offset adds +8x512 element ops = **529,152 arithmetic +256 tanh** | Cache gamma,beta: **2,048 bytes** |
| E normalized conditioning | u=W*RMSNorm(c)+b, retain tanh and GRU; new learned D-vector scale; **512 extra** | Squares/sum/mean/epsilon/two multiplies: **2,049 arithmetic +1 rsqrt** per patch | No extra persistent state |

B's projection reuses u already needed by the baseline, and caches the gated vector
once per patch; it does not reproject at each byte. C counts the seven prefix bytes
consumed for a complete patch's eight predictions (the last consumption is unnecessary
for training output); a streaming implementation may perform an eighth then discard
it, adding 64 operations without changing predictions. For B initialization set
a=-4; use a separate RNG stream for any new weights. The small nonzero gate is
intentional: no claim of exactly identical initial outputs. Original tensors/RNG
initialization are paired across arms. A zero gate limit must recover A algebraically.

Training cache payload B/C/D is respectively 4*B*T*Q, 4*B*T*E and 8*B*T*Q bytes,
before autograd saves; E adds roughly 4*B*T*D normalized-context values and its
backward temporaries. Extra FP32 parameter+gradient+Adam storage is 16 times the
extra parameter count, plus per-tensor Adam step scalars. Gate parameters must use
the documented rank-one/no-decay policy. No extra global recurrent history is needed.

Causality: all options must latch the same previous-patch context, never feed the
current completed patch into its own earlier predictions, and reset caches at every
patch/document boundary. Prefix perturbation and full/stream parity tests are required.
B/C/D need a new streaming-state schema/cache serialization; E still needs a new
architecture/config identity. Every option creates a new checkpoint lineage; never
load a frozen 2M or A checkpoint by ignoring missing keys. Explicit reviewed migrations
may copy unchanged tensors, but cannot claim exact resume into a different architecture.

Failure modes: A can suppress context through tanh/GRU dynamics; B can learn to close
its gate or dominate/shortcut local readout; C can be forgotten by GRU gates or swamp
byte embeddings; D can amplify unstable features, collapse scales, or overfit; E can
erase useful context amplitude and still saturate its projection. Measure pre-tanh
quantiles, saturation fraction abs(tanh(u))>0.99, gradients, phase-wise hidden/logit
sensitivity and held-out benefit, not just state norms. Gate magnitude alone is no proof.

After the first pilot, compare A versus B only, fresh paired initializations and
matched TRAIN windows/valid target-byte exposure, seed17 primary plus seed29 confirm.
Use the same length64, optimizer selected before comparing, and at most 8,000 updates
per arm. Other architecture dimensions stay selected. Require stage 20M-5 criteria
in the training plan. Do not simultaneously change sequence length, mixture, decoder,
precision, backend or LR. No automatic selection of C/D/E after B fails; revise the
predeclared design and obtain a new bounded experiment authorization.

## Sequence and context

**FIRST_PILOT_SEQUENCE_LENGTH=64 bytes**. This is a modest information-exposure
choice: eight patch contexts rather than four, while holding the local eight-byte
clock fixed. It is not evidence of useful learned context. The previous 32/64 2M A/B
did not show adoption, so length is not the proposed solution to that finding.

| Bytes L | Prediction global steps ceil(L/8) | SSD T^2 relative to 32 | Use |
|---|---:|---:|---|
| 32 | 4 | 1 | Mechanics and equal-target-byte length control |
| 64 | 8 | 4 | First bounded pilot |
| 128 | 16 | 16 | Deferred controlled length experiment after bridge/context gate |
| 256 | 32 | 64 | Deferred until measured context benefit and throughput permit |

At equal target bytes per update, quadratic work grows approximately linearly with
L (fewer sequences offset one T factor); linear projection work is roughly constant.
For fixed microbatch, activations scale with L and dense SSD matrices with L^2.
`chunk_patches=16` does not make this implementation chunk-linear: current batch
forward constructs whole patch-time quadratic matrices.

Training resets state for independent document windows, so gradients only span
the chosen window. Streaming mechanical recurrence has fixed-size memory and can
consume histories beyond64; after L bytes its step count is 1+floor(L/8), unlike
prediction forward. Effective learned context requires paired validation benefit.
No automatic 64->128->256 curriculum is approved. A later length-only comparison
must show >=0.01 nats/byte history benefit with a positive document-bootstrap 95% CI,
<=0.02 nats/byte degradation on the fixed short validation, and pass the GPU gate
at equal exposure before its length is scheduled into a new run identity.

## Reference and optimized backend

Decision: **MANDATORY_ONLY_BEFORE_SERIOUS_20M**. Reference FP32 is permitted for
construction, correctness, mechanics and a bounded pilot if it meets the time cap.
For a 2-billion-byte run, unbounded quadratic patch tensors and Python checks are
too much operational risk to approve without a validated optimized path. If the
bounded pilot fails the measured throughput gate, optimization becomes a prerequisite
there too; do not silently extend the compute budget.

Compatibility checked 2026-09-27: NVIDIA lists T4 at compute capability7.5;
current Triton lists supported NVIDIA hardware at8.0+. The upstream Mamba-2 combined
SSD source imports Triton. Thus the current upstream fast path is **not a supported
T4 choice**; installation or import success would not establish kernel support.
Older versions/custom CUDA may differ but no compatible pinned combination has
been verified. Reference torch operations remain the only accepted backend.

Sources: [NVIDIA capability table](https://developer.nvidia.com/cuda/gpus),
[Triton compatibility](https://github.com/triton-lang/triton#compatibility),
[upstream SSD source](https://raw.githubusercontent.com/state-spaces/mamba/main/mamba_ssm/ops/triton/ssd_combined.py),
[Mamba-2 module](https://raw.githubusercontent.com/state-spaces/mamba/main/mamba_ssm/modules/mamba2.py).
These are mutable upstream references, consulted for compatibility, not dependency
pins. Future promotion must record exact commit/wheel/CUDA/driver hashes.

Preferred investigation is a torch chunked SSD or sm75-capable custom implementation
behind forward/step/state import/export, preserving single-group B/C, per-head decay,
conv orientation, dt initialization, RMSNorm after SiLU gating, pre-norm residual,
FP32 and masks. Upstream modules are not a drop-in replacement: wrapper pre-norm,
residual, returned states and strict serialization must match this repository.
No backend installed, ported or promoted in this audit.

Predeclare backend parity on CPU reference versus target FP32, seeds17/29/43,
B=1/2/8, patch lengths0/1/3/4/8/16/17/32, nonzero incoming state and ragged byte tails.
Forward/stream outputs and state: atol=1e-5, rtol=1e-4; gradients: atol=2e-5,
rtol=2e-4; seeded single AdamW update including moments: atol=2e-5, rtol=2e-4.
Report maxima and relative L2, not only allclose. Causal prefix outputs must be exactly
unchanged within a backend. Check gradients for every participating named tensor,
zero-length behavior, PAD/BOS masks, resets and all state fields. Same-backend
checkpoint resume must be bitwise identical over three subsequent updates including
cursor, RNG, scheduler and lazy optimizer coverage. Cross-backend comparison uses
the stated tolerances and is a distinct lineage, never exact resume by relabeling.
Any failure retains reference and blocks promotion; no relaxed thresholds after seeing results.
