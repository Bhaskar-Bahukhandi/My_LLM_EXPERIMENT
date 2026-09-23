# 2M context and byte-distribution study

**Evidence only. No production training or curriculum change. TEST remains sealed.**

Recommendation: **MORE_EVIDENCE_REQUIRED**. The trained step-5000 checkpoint does not measurably use extra history in the selected targets across all eight phases: shared states differ, but all 256 forensic logit pairs are bitwise identical and paired NLL deltas are zero. The untrained architecture sanity fixture transmits older history, and bounded mechanics pass. This does not establish that longer-window training will improve quality. Review a separately authorized disposable 32-vs-64 continuation A/B before committing to either 73,167 more 32-byte updates or a production curriculum.

## Immutable parent and protocol

Parent: `data/full-training-2m-v1/attempt-003/checkpoints/step_005000`, update 5000, 1,929,579 trainable parameters.
Parameter SHA-256: `4b8416333d4faf32b86d20eabd2addf6bb6fb39c52783a3427597657db71584d`.
Source commit: `86d3263d99f6465c41da6b26b8921cd7143727a9`. AC-008 SHA-256: `4b37b71915133012e3172cd5c266d1a353a2ae0b494fac6a7ec468c53d6f121b`.
Parent-before/after inventory, exact config/runtime/tool hashes and split identities are in the JSON report. Production parent, FINAL specifications and Stage-A reports remain byte-identical.
Raw-byte NLL uses original 267-symbol logits, with BOS/PAD masked by the model, not conditional-q loss. Manifest-ordered batch 2 windows reset at BOS and never cross a document boundary. All 500,000 validation targets included at every length.
32-byte reproduction tolerance was predeclared as absolute 1e-7 versus 2.3624953916. BOS is separate from completed payload patches.

## Segmented validation: context/reset sensitivity

Changing segmentation changes reset frequency and the prior history of each target. This is not the controlled same-target comparison below.

| Bytes / patches | Windows | Targets | NLL | Bits/byte | Seconds | Targets/s | Peak allocated/reserved MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| 32 / 4 | 15631 | 500000 | 2.36249539 | 3.408360 | 606.72 | 824.1 | 42.22/60.00 |
| 64 / 8 | 7821 | 500000 | 2.36249539 | 3.408360 | 414.94 | 1205.0 | 42.42/60.00 |
| 128 / 16 | 3914 | 500000 | 2.36249538 | 3.408360 | 280.62 | 1781.8 | 42.84/60.00 |
| 256 / 32 | 1960 | 500000 | 2.36249538 | 3.408360 | 269.23 | 1857.2 | 43.74/62.00 |

Published tooling/resume commit: `84d4f9a49e7225aab408458cbd590fada61597ab`. Original launch binding remains immutable; producer hashes match.

Minimum sampled free VRAM: 3223.70 MiB.
Peak sampled Windows process working set during validation: 779.94 MiB.

## Matched next-byte targets

128 lowest-SHA256 anchors per available domain (seed 1729), each with at least 256 prior document bytes; same target at 32/64/128/256 history. No result-dependent resampling. Paired percentile bootstrap: 2000 draws, seed 1730,95% interval; unchanged tolerance 1e-5 NLL.
Anchor bootstrap, not independent document bootstrap; clustered anchors and equal-domain sampling limit population inference.

| History | Mean NLL | Median | Delta vs32 [95% CI] | Improved / worsened / unchanged |
|---|---:|---:|---|---|
| 32 | 3.595847 | 3.494863 | 0.000000 [0.0, 0.0] | 0.000/0.000/1.000 |
| 64 | 3.595847 | 3.494863 | 0.000000 [0.0, 0.0] | 0.000/0.000/1.000 |
| 128 | 3.595847 | 3.494863 | 0.000000 [0.0, 0.0] | 0.000/0.000/1.000 |
| 256 | 3.595847 | 3.494863 | 0.000000 [0.0, 0.0] | 0.000/0.000/1.000 |

## Position and domain behavior

Buckets are absolute target positions within each reset window. Relative deltas use that length's 0–31 bucket; target composition differs between buckets.

| Length | Bucket | Targets | NLL | Bits/byte | Relative delta |
|---|---|---:|---:|---:|---:|
| 32 | 0–31 | 500000 | 2.362495 | 3.408360 | 0.00% |
| 64 | 0–31 | 250131 | 2.363205 | 3.409384 | 0.00% |
| 64 | 32–63 | 249869 | 2.361785 | 3.407336 | -0.06% |
| 128 | 0–31 | 125199 | 2.369549 | 3.418536 | 0.00% |
| 128 | 32–63 | 125101 | 2.362083 | 3.407765 | -0.32% |
| 128 | 64–95 | 124932 | 2.356847 | 3.400212 | -0.54% |
| 128 | 96–127 | 124768 | 2.361487 | 3.406906 | -0.34% |
| 256 | 0–31 | 62720 | 2.373877 | 3.424781 | 0.00% |
| 256 | 32–63 | 62720 | 2.362370 | 3.408179 | -0.48% |
| 256 | 64–95 | 62681 | 2.358660 | 3.402827 | -0.64% |
| 256 | 96–127 | 62582 | 2.355062 | 3.397637 | -0.79% |
| 256 | 128–159 | 62479 | 2.365204 | 3.412268 | -0.37% |
| 256 | 160–191 | 62381 | 2.361794 | 3.407348 | -0.51% |
| 256 | 192–223 | 62251 | 2.355022 | 3.397579 | -0.79% |
| 256 | 224–255 | 62186 | 2.367953 | 3.416234 | -0.25% |

| Domain | Length | Segmented NLL | Matched delta vs32 |
|---|---:|---:|---:|
| code | 32 | 2.393015 | 0.000000 |
| code | 64 | 2.393015 | 0.000000 |
| code | 128 | 2.393015 | 0.000000 |
| code | 256 | 2.393015 | 0.000000 |
| documentation | 32 | 2.322501 | 0.000000 |
| documentation | 64 | 2.322501 | 0.000000 |
| documentation | 128 | 2.322501 | 0.000000 |
| documentation | 256 | 2.322501 | 0.000000 |
| general_text | 32 | 2.222099 | 0.000000 |
| general_text | 64 | 2.222099 | 0.000000 |
| general_text | 128 | 2.222099 | 0.000000 |
| general_text | 256 | 2.222099 | 0.000000 |
| structured_math | 32 | 3.048170 | 0.000000 |
| structured_math | 64 | 3.048170 | 0.000000 |
| structured_math | 128 | 3.048170 | 0.000000 |
| structured_math | 256 | 3.048170 | 0.000000 |

## State carry diagnostic

First eight hash-selected anchors/domain. Fresh BOS at offset-32 versus BOS at offset-256 carried through the same last 32 bytes. Both predict identical next target. No unbounded document streaming.
{"count": 32, "mean_nll": 3.741223521530628, "median_nll": 3.7649048566818237, "paired_delta": 0.0, "paired_delta_ci95": [0.0, 0.0], "improved_fraction": 0.0, "worsened_fraction": 0.0, "unchanged_fraction": 1.0}

## Original p and conditional-byte q

p is softmax over original 267-symbol logits. Byte mass is sum p(0..255); q(b)=p(b)/byte_mass. Raw-byte NLL remains based on p. Reported entropy/top-two/margins/concentration below use q. JSON preserves p-space, byte/control mass and every control probability for historical probes. Masked IDs are PAD 256 and BOS 257; EOS 258 remains in control support.
Project-specific predeclared labels: severe if max(q)>=.9 and entropy<=1nat; otherwise moderate if q(space)>=.5; otherwise broad space dominance if space is argmax and entropy>1nat; mixed otherwise. These are not universal collapse thresholds.

| History | q entropy nats | p(space) | q(space) | Space top1 fraction | q margin | Control mass | >=.9 concentration fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| 32 | 3.385328 | 0.214796 | 0.214796 | 1.000 | 0.138190 | 0.00000267 | 0.000 |
| 64 | 3.385328 | 0.214796 | 0.214796 | 1.000 | 0.138190 | 0.00000267 | 0.000 |
| 128 | 3.385328 | 0.214796 | 0.214796 | 1.000 | 0.138190 | 0.00000267 | 0.000 |
| 256 | 3.385328 | 0.214796 | 0.214796 | 1.000 | 0.138190 | 0.00000267 | 0.000 |

### Position-stratified validation survey

Separate fixed sample: up to eight frozen anchor hashes per domain / length / 32-byte position bucket. Context resets at the containing window boundary. This covers interior positions; the matched comparison above always predicts at a patch boundary. Neither sample follows the natural corpus domain weights.

| Window | Contexts | q entropy | p(space) | q(space) | Space top1 | Control mass |
|---|---:|---:|---:|---:|---:|---:|
| 32 | 32 | 2.574377 | 0.177161 | 0.177162 | 0.406 | 0.00002027 |
| 64 | 64 | 2.500945 | 0.145834 | 0.145834 | 0.344 | 0.00001314 |
| 128 | 128 | 2.398027 | 0.178324 | 0.178324 | 0.383 | 0.00001137 |
| 256 | 256 | 2.323008 | 0.182925 | 0.182925 | 0.371 | 0.00001290 |

Joint length/domain/32-byte-bucket statistics, top-1 concentration and control-vs-byte mass are preserved in JSON; this survey is descriptive.
Position-survey classes: {"ARGMAX_SPACE_DOMINANCE_WITH_BROAD_DISTRIBUTION": 123, "MIXED_BY_CONTEXT": 292, "MODERATE_SPACE_CONCENTRATION": 48, "SEVERE_BYTE_MODE_CONCENTRATION": 17}
MIXED: broad-distribution argmax attractor at boundaries, mostly moderate space concentration during greedy loops; rare severe individual contexts do not establish global probability collapse.

## Historical generation and whitespace runs

The six historical 64-byte greedy outputs reproduce exactly. Per-step byte/probability/top2/entropy/space/control measurements and forced-space successor statistics for held-out contexts are in JSON. No supplemental sampling replaced greedy results.

| Prompt | Output | Longest space run | Mean q(space) | q entropy first→last |
|---|---|---:|---:|---|
| `b''` | `b'                                                                '` | 64 | 0.677918 | [3.3853281938958215, 1.127350743537607] |
| `b'The '` | `b'made                                                            '` | 60 | 0.629901 | [3.185358289914923, 1.1438791511953539] |
| `b'def '` | `b'the                                                             '` | 61 | 0.639915 | [3.3618289138683712, 1.1438791511953539] |
| `b'import '` | `b'i                                                               '` | 63 | 0.665092 | [2.9578277870894762, 1.1631086403826814] |
| `b'class '` | `b'an                                                              '` | 62 | 0.654707 | [3.1385193650452763, 1.1536940319535802] |
| `b'x = '` | `b'self                                                            '` | 60 | 0.628382 | [3.8669240485096803, 1.1438791511953539] |

## Disposable mechanics and quadratic reference cost

Each length uses a fresh independent copy, batch 2, actual forward/loss/backward and finite-gradient checks. No optimizer update, saved clone, logical production update or optimizer migration. One cold observation per length; ratios are measured costs, not asymptotic proof.

| Bytes | Forward s | Backward s | Total s | Target bytes/s | Gradient norm | Peak allocated/reserved MiB |
|---|---:|---:|---:|---:|---:|---:|
| 32 | 0.5156 | 0.2806 | 0.7962 | 80.4 | 1.8832 | 88.09/96.00 |
| 64 | 0.1296 | 0.1315 | 0.2611 | 490.2 | 1.2376 | 88.09/98.00 |
| 128 | 0.2035 | 0.2557 | 0.4592 | 557.5 | 0.8734 | 88.39/100.00 |
| 256 | 0.2295 | 0.3158 | 0.5454 | 938.8 | 0.7395 | 95.19/108.00 |

Observed forward/backward/total ratios: `{"64/32": {"forward_seconds": 0.2513366171798433, "backward_seconds": 0.4687404791962151, "total_seconds": 0.3279585571163612}, "128/64": {"forward_seconds": 1.5702751715068617, "backward_seconds": 1.9440068100516161, "total_seconds": 1.7585356881707277}, "256/128": {"forward_seconds": 1.1279844093853768, "backward_seconds": 1.2350035154598626, "total_seconds": 1.1875790624515241}}`.
Synchronized CUDA peak allocation/reservation, sampled free VRAM and Windows process working set are recorded per length in JSON. RSS is measured independently of tensor accounting. No extrapolation to 512/1024 or effective-context claim.

## Continuation choices — not authorization

Plan A retains step 5000,32-byte windows,total_steps78167 and exact optimizer/scheduler/cursor. **73,167 logical updates remain.** Historical production throughput projects 14.96 training-only hours; excludes validation/downtime and is not a fresh sustained benchmark.
Full validation and the same six greedy probes every 2500 updates and final endpoint, checkpoints every 1000 updates; human must authorize continuation/cadence.
Endpoint checksum/config/corpus/cursor, exact restoration, validation/distribution/generation evidence, independent review and immutable lineage; TEST evaluation needs separate authorization.

Plan B requires a separately reviewed versioned curriculum from the immutable step 5000 parent:

64 then128 bytes, with256 considered only after separate phase review; no immediate curriculum change is supported by current matched-target quality evidence.
All lengths mechanically pass; measured batch2 forward/backward totals at64/128/256 are 0.261/0.459/0.545s, without optimizer update. These single cold probes are not sustained production estimates.

- Select length or staged lengths using matched effects, domain regressions and costs; no automatic longest-length preference.
- Define document-window rebuilding, reset and cursor semantics without reinterpreting the old cursor.
- Explicitly decide retained AdamW state versus new phase, and scheduler continuation/reset.
- Record repeated/skipped byte exposure and new phase budget; retain original production counters.
- Version checkpoint binding/schema and validation protocol; retain immutable step 5000 rollback.
- Evaluate both accepted32-byte baseline and new protocols for comparability; review measured compute impact.

## Recovery, limitations and validation

- Original process ended after 304 anchors. All 117 segment blocks, four summaries and 304 anchors verified and reused. Published tooling commit advanced HEAD only; a checksummed recovery reconciliation preserves the original producer/binding.
- Removed prewritten finalizer test claims; actual final command outputs, durations and tested-file hashes are required from validation-closeout/commands.json.
- Before distribution measurement, review identified exact-L matched targets are patch boundaries. Added separately bound position-stratified survey reusing frozen anchors; no main measurement/protocol rewritten or restarted.
- Initial preflight rejected new verifier's use of config JSON canonicalization for corpus. Corrected to existing UTF8 corpus canonicalizer before evaluation; regression added.
- Optional unused PyTorch NumPy bridge warning persists; no environment modifications.
- Phase runner preflight hit a list-conversion TypeError before saving any result. Empty failed binding preserved in phase-matched-preflight-001; corrected producer bound separately before measurement. No completed measurement recomputed.
- Phase-only batch-8 supplement records inference wall time but no dedicated allocator/RSS series; memory tables describe the original segmented/mechanics runs.
- Segmented seconds sum measured block wall durations and exclude publication/setup and interruption downtime; throughput uses those measured durations, not total end-to-end elapsed time.
- 32-byte timing overlaps focused CPU regression work and includes instrumentation; do not compare it as pure backend latency with historical validation. Disposable mechanics run after the other work.
- Segment length changes BOS/reset count and historical exposure: not a same-target experiment.
- Matched histories share targets but include different recurrent trajectories; better NLL does not prove causal long-range retention.
- Stage A trained only 32-byte windows; 256 evaluated is not 256 effective context certification.
- Generic SSD reference retains quadratic transition storage; no optimized kernels or extrapolated effective context.
- Peak allocator memory measured; free VRAM and RSS are sampled, not continuous minima/maxima.
- Mechanics are one cold disposable forward/backward per length, no optimizer update; ratios include warmup/cache effects.
- TEST sealed except streaming hashes; no production training, curriculum, weights, checkpoints or source-model changes.
- Completed block/anchor files are immutable and checksummed against binding; progress can be rebuilt. No result is overwritten on recovery.
- NOT_RUN: historical greedy policy preserved; supplemental sampling not needed for probability measurements
- NOT_RUN: beyond 256 mechanical streaming would not establish effective context and would delay primary study

Final validation results below come from the checksummed command artifact.

No production training, model architecture, weights, checkpoint, corpus or environment changes. No TEST evaluation, 20M, RSI or effort controller. Next action requires human authorization.

## Five separate interpretation questions

**A aggregate reset sensitivity:** Segmented NLL range is below 2e-8; effectively indistinguishable under the fixed protocol. This alone says nothing universal about useful history.

**B matched history benefit:** All 512 paired NLL deltas are exactly 0 at 64/128/256; 95% paired bootstrap intervals are [0,0], including every domain. The current checkpoint obtains no measurable extra-history benefit for these boundary targets. The boundary-only result motivates the separate phase/logit/state supplement; no universal context claim follows from the original NLL equality.

**C late position behavior:** No late bucket degrades against its own first bucket; differences remain below 0.8%. Buckets contain different targets, so lower late NLL is not evidence of causal context use.

**D distribution quality:** Boundary contexts have broad argmax-space dominance. Greedy whitespace primarily has moderate space concentration: probabilities and entropy cycle every 8-byte patch instead of tending to a point mass. Position-stratified results are descriptive and reported separately.

**E compute cost:** All four forward/backward probes pass with 56 finite gradient tensors and zero updates. Longest probe peaks below 100 MB allocated. The first 32-byte cold probe is slower than 64; single-shot timings are not an asymptotic fit or a sustained training-speed estimate.

## Separately bound phase-matched supplement

128 deterministic anchors (32/domain), eight phases, 4,096 target-history scores. Within phase r, histories are 32+r / 64+r / 128+r / 248+r bytes. Original boundary measurements above remain unchanged. Batch size 8; FP32 model and forensic arithmetic.

| Phase | Anchors | NLL short / 64+r / 128+r / 248+r | Long minus short | 95% paired CI | Improved / worse / unchanged |
|---|---:|---|---:|---|---|
| 0 | 128 | 3.526271850 / 3.526271850 / 3.526271850 / 3.526271850 | 0 | [0.0, 0.0] | 0.000 / 0.000 / 1.000 |
| 1 | 128 | 3.028552485 / 3.028552485 / 3.028552485 / 3.028552485 | 0 | [0.0, 0.0] | 0.000 / 0.000 / 1.000 |
| 2 | 128 | 2.743031943 / 2.743031943 / 2.743031943 / 2.743031943 | 0 | [0.0, 0.0] | 0.000 / 0.000 / 1.000 |
| 3 | 128 | 2.572614054 / 2.572614054 / 2.572614054 / 2.572614054 | 0 | [0.0, 0.0] | 0.000 / 0.000 / 1.000 |
| 4 | 128 | 2.462623008 / 2.462623008 / 2.462623008 / 2.462623008 | 0 | [0.0, 0.0] | 0.000 / 0.000 / 1.000 |
| 5 | 128 | 2.428548651 / 2.428548651 / 2.428548651 / 2.428548651 | 0 | [0.0, 0.0] | 0.000 / 0.000 / 1.000 |
| 6 | 128 | 2.477357803 / 2.477357803 / 2.477357803 / 2.477357803 | 0 | [0.0, 0.0] | 0.000 / 0.000 / 1.000 |
| 7 | 128 | 2.464861981 / 2.464861981 / 2.464861981 / 2.464861981 | 0 | [0.0, 0.0] | 0.000 / 0.000 / 1.000 |

Per-phase paired anchor bootstrap; aggregate averages eight phases per anchor before resampling. Anchors within documents are not independent.
The JSON includes all four history summaries, medians, domain-wise paired intervals and aggregate across phases.

## Logit equality and internal-state sensitivity

Mechanism: **INTERNAL_STATE_DIFFERS_OUTPUT_INSENSITIVE**. FP32, this checkpoint and selected validation histories only; no architectural-defect or effective-context claim.
Forensic subset: 256 comparisons (8 anchors/domain × 8 phases); torch.equal confirms 256 identical logit vectors. Shared state differs in 256 comparisons.
Maximum finite-logit absolute delta: 0; maximum q total variation: 0. PAD/BOS remain -inf and are excluded only from numeric subtraction, not torch.equal.

| Layer | Conv max delta | SSM max delta | Conv short / long norm range | SSM short / long norm range |
|---|---:|---:|---|---|
| 0 | 0 | 28.1611977 | [245.947, 248.286] / [245.947, 248.286] | [27080.4, 28362] / [27080.6, 28362.2] |
| 1 | 0.211368561 | 30.3751831 | [283.3, 285.316] / [283.498, 285.524] | [6330.42, 6593.96] / [6334.71, 6598.43] |
| 2 | 1.44085145 | 51.0148315 | [309.859, 310.303] / [310.253, 310.603] | [6790.66, 6973.51] / [6847.46, 7030.19] |
| 3 | 1.03919029 | 10.6368675 | [349.896, 350.422] / [354.275, 354.733] | [5.48762, 5.58923] / [42.9915, 43.1391] |

Pending bytes agree in 256 comparisons; lengths 0..7 match phase. Decoder hidden maximum delta: 0. JSON retains L2, mean absolute delta, state norms, target log-probability differences, top-1 agreement and phase/domain summaries.

## Architecture sanity

Deterministic seed-43 untrained accepted model, eight old zeros versus eight old 255 bytes, then the identical 32-byte suffix 0..31. No weights tuned and no optimizer constructed.
Result: PASS; finite-logit max delta 0.782281518. This demonstrates architectural transmission in this fixture, not trained long-context quality.
NOT_RUN: additional sequential inference cost; required phase/state probes prioritized

## Future disposable A/B proposal — not authorization

- **status:** PROPOSAL_ONLY_NOT_AUTHORIZED_NOT_RUN
- **parent:** immutable step 5000; separate disposable clones, never production resume
- **arms:** A:32 bytes; B:64 bytes (128 only if separately chosen before experiment)
- **budget:** Proposed cap: 65,536 valid target bytes per arm and 30 minutes per arm; if either cap prevents matched exposure, report the mismatch and do not promote. Predeclare equal valid target-byte exposure and a common compute ceiling; report actual compute, do not imply equal bytes guarantee equal wall time. For a compute-matched comparison, predeclare a separate common wall-time endpoint.
- **controls:** Same corpus/document eligibility, seed, optimizer-state initialization, learning-rate policy and target order. Explicitly bind changed window/cursor semantics.
- **evaluation:** Identical accepted 32-byte validation, segmented and phase-matched targets, distribution probes, learning gain per target-byte and second; TEST sealed.
- **promotion:** Discard both unless separately reviewed and explicitly promoted; no A/B training or production continuation authorized by this report.

## Actual final checks

| Command | Exit code | Wall seconds |
|---|---:|---:|
| pytest | 0 | 72.847 |
| ruff | 0 | 0.129 |
| format | 0 | 0.119 |
| compileall | 0 | 0.134 |
| cpu_dependencies | 0 | 1.623 |
| cuda_dependencies | 0 | 1.312 |
| diff | 0 | 0.059 |

Pytest: 47 passed, 1 warning in 71.96s (0:01:11)


2M CONTEXT-DISTRIBUTION STUDY STATUS: READY FOR HUMAN DECISION
