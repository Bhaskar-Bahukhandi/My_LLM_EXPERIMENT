# AC-008 — Context and Distribution Promotion Gates

Version 1.0 • 2026-09-22 • **PROPOSED / RESEARCH CONTRACT — NOT IMPLEMENTED OR ENABLED**

## Motivation and existing requirements

The [Stage-A report](../../reports/full_training_2m_stage_a.md) records full-validation NLL improving from 5.6641971744 to 2.3624953916 while greedy continuations remain short fragments followed by spaces. The [diagnostic report](../../reports/real_training_diagnostic_2m.md) independently records improved NLL with 64-space output. These are existing measurements, not experiments rerun in this tranche. They do not establish probability-mass collapse.

The [Bible](../../Unified_Edge400_Master_Bible_v3.0_FINAL.md) §§39, 57–62, 93, 115, 143 and 168.13 already requires quality, context and multi-metric reporting; [Roadmap](../../Unified_Edge400_Master_Roadmap_and_Formula_Board_v1.0_FINAL.md) §8, §§14, 18, 23, 26 and 28 provide information-theory, numerical, efficiency and decision tools. The additions here are explicit context certification fields and distribution diagnostics that distinguish an argmax preference from concentrated probability mass. No existing evaluation is relabeled as having measured these new quantities.

## Three context concepts

1. **Mechanically supported context:** a named runtime can ingest/stream a length under defined resource and numerical conditions. Distinguish measured mechanical tests from theoretical unbounded recurrence.
2. **Trained context:** actual training window lengths, curriculum, boundary/reset policy and state carry-over used by a checkpoint, with exposure counts. It is not the configured maximum capacity alone.
3. **Empirically validated effective context:** task- and distance-specific retention supported by a predeclared evaluation protocol, success criteria and uncertainty for that exact checkpoint/system.

These MUST NOT be conflated. Mechanically streaming one million bytes does not prove one-million-byte effective context. Every future report claiming context needs the following conceptual record, with unknown values represented explicitly rather than invented numbers:

| Field under context | Required interpretation |
|---|---|
| mechanically_supported | Length/range, measured vs theoretical, runtime/profile, resource evidence |
| training_window_bytes | Actual lengths/exposure, curriculum, packing and reset/carry policy |
| validation_window_bytes | Actual validation lengths/reset policy; total corpus size separately |
| maximum_evaluated_context_bytes | Longest tested conditioning span, with protocol and task identity |
| effective_context_evidence | Checkpoint, suite, data/policy hashes, distance curves, thresholds and uncertainty |
| retention_tests | Categories run, pass/fail/unknown, ablations and uncovered categories |
| status | VERIFIED / PARTIAL / UNVERIFIED for the explicitly scoped claim |

VERIFIED requires all predeclared gates for the named tasks/distances/profile, not universal retention. PARTIAL means some relevant evidence exists but the proposed claim is not fully certified; list gaps. UNVERIFIED means no adequate retention evidence. Any unsupported larger context figure must be labeled **mechanical/theoretical only**, including in marketing; it may not appear as an effective-context claim. Maximum tested length is not automatically a passing length. Retrieval-assisted context must be reported separately from neural retention.

Stage A trained and validated with **32-byte windows**. Patch size is **8 bytes**, so an ordinary full window completes **floor(32/8)=4 payload patches** (AC-001); BOS is a separate shared-state bootstrap step, not a fifth payload patch. The 500,000-byte validation split is not a 500,000-byte conditioning context. Sixty-four-byte generation probes are not long-distance retention tests. Effective retention beyond the tested short-window setting remains unverified, and even short-window NLL is not a retention certificate. Stage-A review readiness remains authoritative and unchanged.

## Future context promotion tests

| Category | Required evidence direction |
|---|---|
| Next-byte loss by relative position | Valid-target-weighted curves across positions, lengths and domains; separate controls |
| Increasing retention distances | Controlled information placement and query distance, with short-distance controls |
| Exact-copy / retrieval tasks | Correct source recovery vs distractors; retrieval disabled/enabled labeled |
| Delayed dependencies | Dependency accuracy across delay, not mere local fluency |
| Code symbols/references | Identifier binding and earlier-definition use across distance |
| Structured document retrieval | Evidence localization, document boundaries and provenance |
| State-reset sensitivity | Matched continuous/reset/truncated-history comparisons |
| Recurrent drift | State/logit trajectories and precision/reference comparison |
| Throughput | Ingestion/decode and end-to-end cost at each tested length |
| Memory | Allocated/reserved/state/process measurements with named scope |
| Numerical stability | Finite states/logits and declared absolute/relative tolerance gates |

Use predeclared held-out tasks and source-unit separation, multiple examples and repeats/seeds where feasible, contamination exclusions, causal full/incremental comparison, and report failures and uncertainty. Resolve length grids, thresholds and supported claims before observing candidate results. Benchmarks are not implemented here. The current quadratic reference training path is unchanged; no 4K/32K/1M run is authorized.

## Distribution-quality contract

Let p(i|c) be the declared normalized symbol distribution before sampling transformations at context c. Record allowed/masked symbol support and checkpoint/control schema. For raw-byte diagnostics define q(b|c)=p(b|c)/sum_{j=0..255} p(j|c), b in 0..255, only when raw-byte mass is positive. Report that raw-byte mass separately; a missing/zero mass makes conditional byte metrics undefined, not zero. This is a project diagnostic definition, not a change to the training objective.

Measure q's entropy (natural-log nats), largest and second-largest probabilities, their difference, ASCII-space (0x20) probability, fraction of contexts with space top-1 (declare tie handling), and fraction with any byte above predeclared confidence thresholds. Report support, sample counts, domains, position and context-length strata; averages alone can hide a collapsed subgroup.

Also require conditional generation diversity for fixed prompts/seeds/policies; repeated-byte and run-length statistics; held-out continuation behavior; domain-specific generation behavior; and control-symbol probabilities where applicable. Keep q-based conditional byte metrics, original p-space probability and total control mass distinct. Separate padding/BOS masks and legitimate control emissions from invalid protocol behavior. Raw-byte NLL/perplexity remains the accepted objective's valid-byte-target metric; do not relabel mixed-symbol loss or substitute conditional-q loss into historical reports.

Greedy space can be top-1 by a small margin while most probability lies elsewhere. Conversely, high space mass, low entropy and long runs across diverse held-out contexts may support a concentration diagnosis. **Greedy whitespace is not automatically probability collapse.** The diagnostic must compare both explanations rather than infer probability from argmax samples. Confidence/run-length thresholds are resolved per generation and domain; no universal numerical threshold is introduced.

Sampling settings, repetition penalties and control masks must be recorded. Report pre-transformation distributions and post-policy outputs separately; do not tune a new sampling policy on the hidden evaluator or use it to hide base-model degeneracy. Better NLL alone, and better-looking selected samples alone, are both insufficient for promotion.

## Multi-metric decisions and impact

The [addendum](../../Unified_Edge400_Future_Capabilities_Addendum_v1.0.md) defines CAPABILITY, QUALITY, CONTEXT, EFFICIENCY, ROBUSTNESS, SAFETY and REPRODUCIBILITY. Each generation resolves a policy before evaluation, including hard vetoes, critical regression limits, uncertainty and applicability. Security/integrity and causal correctness precede ranking (consistent with AC-004); no universal scalar weights can buy an exception.

PROMOTE requires credible measured benefit and all applicable gates. HOLD preserves the parent when evidence is incomplete or tradeoffs unresolved. REJECT isolates a failed unpromoted candidate. ROLLBACK withdraws an accepted candidate when a later gate fails and restores a reviewed safe ancestor/system bundle. Disabled features receive explicit not-applicable scope, not invented passing scores. Previously accepted capabilities retain regression protection.

Current parameter/memory/inference/training impact: none. Future measurement adds evaluator cost, which must itself be budgeted; no neural tensors or checkpoint format are changed by this specification. Required future tests include misleading mechanical-context claims, total-split/window confusion, near-tied space argmax vs concentrated mass, raw-byte/control separation, missing evidence, and promotion vetoes despite lower NLL. No such model benchmark runs now.

Rollback preserves the accepted checkpoint, original reports and FINAL files; failed future diagnostics block the requested promotion without rewriting historical measurements. Open issues include task-dependent effective context, adequate distance coverage, calibration across domains, and how distribution diagnostics predict useful generation. Decision: pending human review, no new quality or context capability claimed.
