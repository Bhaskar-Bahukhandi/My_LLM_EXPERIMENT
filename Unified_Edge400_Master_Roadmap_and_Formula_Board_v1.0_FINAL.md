# Unified Edge-400 Master Build Roadmap + Master Formula Board
## v1.0 FINAL — 12-Month Engineering Execution Plan and Verified Mathematical Reference
**Companion baseline:** Unified Edge-400 Master Engineering & Research Bible v3.0 FINAL
**Roadmap window:** 8 September 2026 → 7 September 2027
**Primary target:** reproducible ~400M-parameter byte-native Mamba-2 MoE system, with smaller validation models first and 800M as an optional research ceiling
**Operating rule:** no component is promoted because it sounds advanced; it is promoted only when its interface, test coverage, benchmark value, failure mode, rollback path, and measured cost are known.
**Document role:** this file is the build order. The v3.0 Bible remains the architecture/specification baseline. This document adds execution sequencing, milestones, validation gates, experiment matrices, and a mathematical formula board.
## 0. How to Use This Document
Read the roadmap from top to bottom once. Then work week-by-week. Never jump directly to the 400M model because a later phase looks more interesting. Every major phase begins with a small executable artifact, produces evidence, and only then scales.
The formula board is intentionally written as standalone display equations. Every equation is enclosed by its own double-dollar opening and closing delimiters. No multi-line equation is left open, split across a table cell, or hidden inside prose. Equations are categorized as **reference**, **derived**, or **design-specific** so an implementation does not accidentally treat an approximation as a law.
## 1. Non-Negotiable Project Invariants
- Byte-native input remains causal: completed patch representations cannot leak future bytes to earlier byte predictions.
- Patch size is 8 bytes for the reference design; routing chunks are 16 patches = 128 bytes.
- The first patch has an explicit bootstrap path: request/BOS context → shared state initialization → local byte decoder → completed patch encoder → canonical recurrent update.
- The shared trunk is the canonical cross-domain state path.
- Experts operate on the same residual width when directly attached; heterogeneous capacity is implemented through depth/internal dimensions or explicit audited adapters.
- Expert-local state is isolated unless an explicit state bridge/fusion is declared and tested.
- The exact 400M shape is resolved by parameter search and audit; a fixed layer count is never treated as an exact parameter guarantee.
- RAG is an external subsystem, not hidden inside the model parameter graph.
- Tools and code execution are host-enforced and sandboxed; model text cannot grant itself host privileges.
- Execution success is evidence about supplied tests, not a proof of general correctness.
- MCTS is not used as a label unless real selection, expansion, simulation/evaluation, and backup/backpropagation exist.
- Quantization is validated on the actual recurrent model; AWQ-like methods are candidates, not assumptions.
- Memory claims are benchmark claims: package size, resident weights, recurrent state, runtime buffers, RAG, and sandbox memory are reported separately.
- Every run records config hash, code commit, dataset manifest hash, checkpoint hash, hardware/software versions, seed, metrics, and failure status.
- Every experimental branch has a reversible fallback to the last trusted checkpoint.
## 2. End-of-Year Definition of Done
The year is complete only when all of the following are true:
- The model factory builds at least 20M, 50M, 100M, 200M, and ~400M variants through configuration rather than code forks.
- Causality, patch semantics, recurrent state continuity, router determinism, expert isolation, tool protocol, and sandbox boundaries have automated tests.
- The 400M candidate passes exact parameter-count audit within the configured tolerance after instantiation.
- A clean training run exists from versioned dataset manifests through a reproducible checkpoint.
- A dense Mamba baseline and the sparse model have been compared under a controlled experiment matrix.
- RAG, verification, repair, and latent reasoning each have an isolated ablation with measured quality/cost deltas.
- At least one deployment backend has been demonstrated end-to-end; unsupported backends are documented as unsupported rather than silently approximated.
- INT4 has been evaluated for quality, recurrent stability, memory, and latency on the final checkpoint.
- Measured peak process memory, latency, throughput, and thermal behavior are reported on named hardware.
- A release package contains the model, resolved config, tokenizer/control-ID manifest, dataset provenance record, runtime manifest, benchmark protocol, and reproducibility evidence.
- The final research report states both positive and negative results and includes failed hypotheses.
## 3. One-Year Timeline at a Glance
## 3.0 Exact execution calendar
**Plan start:** 2026-09-08. **52 implementation weeks:** 2026-09-08 through 2027-09-06. **Final archival/review day:** 2027-09-07.

| Cycle | Weeks | Calendar range | Primary outcome |
|---|---|---|---|
| M01 | W01–W04 | 2026-09-08 → 2026-10-05 | Foundations, contracts, and instrumentation |
| M02 | W05–W08 | 2026-10-06 → 2026-11-02 | Byte stream, patching, decoder, and causality |
| M03 | W09–W12 | 2026-11-03 → 2026-11-30 | Mamba-2 reference backbone and stateful inference |
| M04 | W13–W16 | 2026-12-01 → 2026-12-28 | Configuration-driven scaling and hardware resolver |
| M05 | W17–W20 | 2026-12-29 → 2027-01-25 | MoE routing, expert registry, and state continuity |
| M06 | W21–W24 | 2027-01-26 → 2027-02-22 | Data engineering, curriculum, and 200M scaling |
| M07 | W25–W28 | 2027-02-23 → 2027-03-22 | 400M candidate search and full-stack pilot |
| M08 | W29–W32 | 2027-03-23 → 2027-04-19 | Local RAG and provenance |
| M09 | W33–W36 | 2027-04-20 → 2027-05-17 | Tools, sandbox, verification, and repair |
| M10 | W37–W40 | 2027-05-18 → 2027-06-14 | SFT, specialization, and MTP |
| M11 | W41–W44 | 2027-06-15 → 2027-07-12 | Quantization, runtime, and edge deployment |
| M12 | W45–W52 | 2027-07-13 → 2027-09-06 | Latent reasoning, ablations, release, and paper |
| Final | — | 2027-09-07 | Archive, reproducibility review, publication/release handoff |

```text
Month 01  | Foundations, repo, schemas, test harness, mathematical notebook
Month 02  | Byte pipeline, patch encoder/decoder, bootstrap, causality proofs/tests
Month 03  | Mamba-2 reference block, state API, 20M dense language model
Month 04  | Configuration-driven scaling, parameter/memory resolver, 20M→50M→100M
Month 05  | MoE router, expert registry, canonical/isolated recurrent state, 50M/100M ablations
Month 06  | 200M scaling, dataset engineering, training stability, curriculum
Month 07  | 400M candidate search, full pretraining pilot, evaluation harness
Month 08  | RAG subsystem, retrieval/reranking, provenance-aware context injection
Month 09  | Tool broker, code sandbox, verifier, bounded execution-guided repair
Month 10  | Fine-tuning, specialization profiles, code/math specialization, MTP ablations
Month 11  | INT4/AWQ-like quantization, runtime optimization, deployment backends, memory/latency audit
Month 12  | Latent reasoning branch, full ablations, final 400M release, research paper/report
Final review  | 7 Sep 2027: freeze release candidate, archive evidence, publish results
```
## 4. Required Repository and Evidence Layout
```text
unified_edge400/
├── configs/
│   ├── base/
│   ├── models/
│   ├── moe/
│   ├── datasets/
│   ├── hardware/
│   ├── experiments/
│   └── profiles/
├── model/
│   ├── byte_embedding.py
│   ├── patch_encoder.py
│   ├── local_decoder.py
│   ├── mamba_block.py
│   ├── shared_trunk.py
│   ├── router.py
│   ├── expert.py
│   ├── fusion.py
│   └── model.py
├── training/
│   ├── pretrain.py
│   ├── sft.py
│   ├── preference.py
│   ├── curriculum.py
│   └── data_pipeline.py
├── rag/
│   ├── ingest.py
│   ├── embed.py
│   ├── store.py
│   ├── retrieve.py
│   └── rerank.py
├── tools/
│   ├── broker.py
│   ├── calculator.py
│   ├── python_sandbox.py
│   └── test_runner.py
├── verifier/
│   ├── parser.py
│   ├── static_checks.py
│   ├── executor.py
│   └── repair.py
├── runtime/
│   ├── protocol.py
│   ├── streamer.py
│   ├── memory.py
│   └── scheduler.py
├── quant/
│   ├── calibrate.py
│   ├── quantize.py
│   └── audit.py
├── eval/
│   ├── language.py
│   ├── code.py
│   ├── rag.py
│   ├── verifier.py
│   ├── latency.py
│   └── memory.py
├── tests/
├── scripts/
│   ├── resolve_config.py
│   ├── count_params.py
│   ├── memory_audit.py
│   ├── benchmark.py
│   └── export_manifest.py
├── docs/
└── evidence/
    ├── runs/
    ├── checkpoints/
    ├── benchmarks/
    └── releases/
```
The evidence directory is not disposable garbage. It is part of the research apparatus. After release, it can be compressed/archived, but benchmark artifacts required to reproduce claims must remain traceable.
## Month 01 — Foundations, Contracts, and Mathematical Instrumentation
### Monthly exit gate
The month is passed only when the previous month remains reproducible and every listed weekly deliverable has an artifact path or benchmark record.
### W01 — Freeze baseline and establish the repository | 2026-09-08 → 2026-09-14
**Build:** Create repo, branch rules, config schema skeleton, experiment manifest schema, evidence ledger, CI/test command, deterministic seed utilities.
**Gate:** Repo initializes cleanly; one sample config resolves to a machine-readable manifest; all tests run from a clean environment.
**Required evidence:** code commit, resolved config, test output, metric artifact, and a brief decision record stating promote / hold / rollback.
### W02 — Build the numerical notebook and formula tests | 2026-09-15 → 2026-09-21
**Build:** Create a small `math_lab/` or notebook suite implementing every core primitive independently: embedding, RMSNorm, SiLU, softmax, cross-entropy, SSM recurrence, quantization/dequantization, cosine similarity, parameter counting, memory accounting.
**Gate:** Reference implementations agree with hand-checked examples and PyTorch autograd/numerical checks where applicable.
**Required evidence:** code commit, resolved config, test output, metric artifact, and a brief decision record stating promote / hold / rollback.
### W03 — Build dataset manifest pipeline | 2026-09-22 → 2026-09-28
**Build:** Implement immutable dataset manifests, SHA-256 hashing, sample counts, byte counts, domain tags, split policies, deduplication hooks, contamination exclusion hooks, and environment-variable paths.
**Gate:** A dataset can be transformed from raw source to manifest to shard without hidden path assumptions.
**Required evidence:** code commit, resolved config, test output, metric artifact, and a brief decision record stating promote / hold / rollback.
### W04 — Build experiment/evidence system | 2026-09-29 → 2026-10-05
**Build:** Every run gets a run ID, config hash, git commit, data manifest hash, environment manifest, seed, metrics, stdout/stderr, and completion status. Add crash-safe checkpoint naming.
**Gate:** A deliberately failed run remains diagnosable and does not overwrite a prior successful run.
**Required evidence:** code commit, resolved config, test output, metric artifact, and a brief decision record stating promote / hold / rollback.

## Month 02 — Byte Stream, Patching, Decoder, and Causality
### Monthly exit gate
The month is passed only when the previous month remains reproducible and every listed weekly deliverable has an artifact path or benchmark record.
### W05 — Implement byte/control vocabulary contract | 2026-10-06 → 2026-10-12
**Build:** Implement raw byte IDs 0–255, padding ID, versioned control-ID registry, serialization, round-trip tests, and manifest hash.
**Gate:** Byte/control round trip is exact; unknown IDs fail closed.
**Required evidence:** code commit, resolved config, test output, metric artifact, and a brief decision record stating promote / hold / rollback.
### W06 — Implement 8-byte patch representation | 2026-10-13 → 2026-10-19
**Build:** Build byte embedding and completed-patch encoder. Confirm that patch representations are not consumed by the recurrent backbone until the patch is complete.
**Gate:** Patch unit tests cover exact shape, ordering, partial-patch rejection, padding, and boundary cases.
**Required evidence:** code commit, resolved config, test output, metric artifact, and a brief decision record stating promote / hold / rollback.
### W07 — Implement local 8-byte autoregressive decoder | 2026-10-20 → 2026-10-26
**Build:** Implement decoder that predicts byte 1..8 causally within each patch, with teacher forcing and generation mode. Verify that byte k cannot see byte k+1.
**Gate:** Causality property tests pass under random masking/perturbation checks.
**Required evidence:** code commit, resolved config, test output, metric artifact, and a brief decision record stating promote / hold / rollback.
### W08 — Implement first-patch bootstrap and end-to-end byte LM shell | 2026-10-27 → 2026-11-02
**Build:** Connect request context/BOS → initial canonical state → local decoder → patch encoder → next state. Build tiny 1M–5M smoke model.
**Gate:** Model can ingest raw bytes and generate bytes end-to-end with deterministic greedy decoding.
**Required evidence:** code commit, resolved config, test output, metric artifact, and a brief decision record stating promote / hold / rollback.

## Month 03 — Mamba-2 Reference Backbone and Stateful Inference
### Monthly exit gate
The month is passed only when the previous month remains reproducible and every listed weekly deliverable has an artifact path or benchmark record.
### W09 — Implement/pin Mamba-2 reference block | 2026-11-03 → 2026-11-09
**Build:** Use the official Mamba-2 formulation and a pinned source revision. Keep a reference PyTorch path even if optimized kernels are used later.
**Gate:** Reference block output shape and numerical smoke tests pass.
**Required evidence:** code commit, resolved config, test output, metric artifact, and a brief decision record stating promote / hold / rollback.
### W10 — Implement explicit recurrent-state API | 2026-11-10 → 2026-11-16
**Build:** Expose state initialization, state update, reset, save/restore, batch handling, and single-step inference. Test state serialization round trips.
**Gate:** Chunked inference and one-shot inference agree within declared numerical tolerance.
**Required evidence:** code commit, resolved config, test output, metric artifact, and a brief decision record stating promote / hold / rollback.
### W11 — Build dense 20M language-model baseline | 2026-11-17 → 2026-11-23
**Build:** Create the simplest useful dense byte-native Mamba model. Train on a tiny/medium clean corpus.
**Gate:** Loss decreases, validation is stable, checkpoints are loadable, and generation is non-degenerate.
**Required evidence:** code commit, resolved config, test output, metric artifact, and a brief decision record stating promote / hold / rollback.
### W12 — Run baseline diagnostics and checkpoint freeze | 2026-11-24 → 2026-11-30
**Build:** Measure bytes/s, loss, perplexity where meaningful, state size, peak training memory, inference memory, and error taxonomy. Freeze baseline checkpoint B20M-DENSE-01.
**Gate:** Baseline becomes the comparison anchor; later features cannot replace it.
**Required evidence:** code commit, resolved config, test output, metric artifact, and a brief decision record stating promote / hold / rollback.

## Month 04 — Configuration-Driven Scaling and Hardware-Aware Resolver
### Monthly exit gate
The month is passed only when the previous month remains reproducible and every listed weekly deliverable has an artifact path or benchmark record.
### W13 — Build typed configuration schema | 2026-12-01 → 2026-12-07
**Build:** Separate model, training, runtime, MoE, dataset, hardware, and experiment configs. Validate required/optional fields.
**Gate:** Invalid shape combinations are rejected before model instantiation.
**Required evidence:** code commit, resolved config, test output, metric artifact, and a brief decision record stating promote / hold / rollback.
### W14 — Build parameter auditor | 2026-12-08 → 2026-12-14
**Build:** Instantiate candidate graphs and count parameters by component, tying weights correctly. Export per-component parameter ledger.
**Gate:** Audit total equals serialized state-dict count and has no missing/duplicate tensors.
**Required evidence:** code commit, resolved config, test output, metric artifact, and a brief decision record stating promote / hold / rollback.
### W15 — Build memory auditor and feasibility classifier | 2026-12-15 → 2026-12-21
**Build:** Estimate and measure parameter, optimizer, activation, recurrent-state, runtime-buffer, RAG, and sandbox memory separately. Implement VALID_FAST_PATH / VALID_GENERIC_PATH / INVALID classification where backend profile information exists.
**Gate:** A candidate can be rejected before expensive training for known infeasibility.
**Required evidence:** code commit, resolved config, test output, metric artifact, and a brief decision record stating promote / hold / rollback.
### W16 — Scale 20M→50M→100M without source rewrites | 2026-12-22 → 2026-12-28
**Build:** Create configs and regression tests proving the same model factory builds all three sizes. Train short runs to confirm scaling behavior.
**Gate:** No `if size == ...` architecture branches are needed.
**Required evidence:** code commit, resolved config, test output, metric artifact, and a brief decision record stating promote / hold / rollback.

## Month 05 — MoE Router, Expert Registry, and State Continuity
### Monthly exit gate
The month is passed only when the previous month remains reproducible and every listed weekly deliverable has an artifact path or benchmark record.
### W17 — Implement deterministic baseline chunk router | 2026-12-29 → 2027-01-04
**Build:** Routing granularity: 16 patches / 128 bytes. Start with mean pooling + RMSNorm + router logits + top-1 selection.
**Gate:** Routing is deterministic, inspectable, serializable, and benchmarked.
**Required evidence:** code commit, resolved config, test output, metric artifact, and a brief decision record stating promote / hold / rollback.
### W18 — Implement expert registry and direct residual contract | 2027-01-05 → 2027-01-11
**Build:** Experts share canonical d_model on direct residual paths. Expert metadata records internal dimensions, depth, state configuration, and parameter count.
**Gate:** Shape validation catches every illegal expert configuration before training.
**Required evidence:** code commit, resolved config, test output, metric artifact, and a brief decision record stating promote / hold / rollback.
### W19 — Implement canonical/expert state separation | 2027-01-12 → 2027-01-18
**Build:** Canonical state crosses domains; expert-local state is isolated. Add explicit state bridge/fusion with zero-initialized or otherwise base-preserving behavior if needed.
**Gate:** Switching experts does not silently destroy or overwrite canonical continuity.
**Required evidence:** code commit, resolved config, test output, metric artifact, and a brief decision record stating promote / hold / rollback.
### W20 — Run 50M/100M MoE ablation | 2027-01-19 → 2027-01-25
**Build:** Compare dense vs MoE with matched total parameters and matched data. Track expert utilization, router entropy, dropped/rerouted chunks, latency, quality.
**Gate:** MoE is promoted only if benefit exceeds measured cost.
**Required evidence:** code commit, resolved config, test output, metric artifact, and a brief decision record stating promote / hold / rollback.

## Month 06 — Data Engineering, Curriculum, and 200M Scaling
### Monthly exit gate
The month is passed only when the previous month remains reproducible and every listed weekly deliverable has an artifact path or benchmark record.
### W21 — Finalize high-quality pretraining corpus pipeline | 2027-01-26 → 2027-02-01
**Build:** Implement source filtering, license/provenance records, language/domain detection, near-duplicate deduplication, repository/file dedup, contamination exclusion, secrets/malware filtering.
**Gate:** Every training shard traces back to a versioned manifest.
**Required evidence:** code commit, resolved config, test output, metric artifact, and a brief decision record stating promote / hold / rollback.
### W22 — Implement curriculum scheduler | 2027-02-02 → 2027-02-08
**Build:** Phases: byte LM → domain mix → instruction → code → tool protocol → repair data. Make phase boundaries configurable and logged.
**Gate:** A run can resume exactly at a phase boundary.
**Required evidence:** code commit, resolved config, test output, metric artifact, and a brief decision record stating promote / hold / rollback.
### W23 — Stability engineering | 2027-02-09 → 2027-02-15
**Build:** Implement optimizer groups, mixed precision policy, gradient accumulation, gradient clipping, checkpointing, validation cadence, loss-spike detection, automatic rollback to trusted checkpoint.
**Gate:** Injected instability triggers controlled rollback and evidence capture.
**Required evidence:** code commit, resolved config, test output, metric artifact, and a brief decision record stating promote / hold / rollback.
### W24 — Train/evaluate 200M candidate | 2027-02-16 → 2027-02-22
**Build:** Train long enough to determine whether the architecture scales. Compare against dense Mamba and the prior MoE scale.
**Gate:** Promotion requires improved quality-per-compute or quality-per-memory evidence.
**Required evidence:** code commit, resolved config, test output, metric artifact, and a brief decision record stating promote / hold / rollback.

## Month 07 — 400M Candidate Search and Full Pretraining Pilot
### Monthly exit gate
The month is passed only when the previous month remains reproducible and every listed weekly deliverable has an artifact path or benchmark record.
### W25 — Enumerate legal 400M candidates | 2027-02-23 → 2027-03-01
**Build:** Search d_model, depth, d_state, head dimensions, expert depth/internal dimensions within pinned backend legality constraints.
**Gate:** Every candidate is classified before allocation of training resources.
**Required evidence:** code commit, resolved config, test output, metric artifact, and a brief decision record stating promote / hold / rollback.
### W26 — Rank candidates by parameter distance + hardware fit | 2027-03-02 → 2027-03-08
**Build:** Use exact parameter audit, predicted active parameter load, state footprint, and benchmarked kernel path. Select a shortlist, not a single guess.
**Gate:** At least 3 viable candidates exist before final selection.
**Required evidence:** code commit, resolved config, test output, metric artifact, and a brief decision record stating promote / hold / rollback.
### W27 — Short full-stack 400M pilots | 2027-03-09 → 2027-03-15
**Build:** Train several candidates for short controlled runs. Measure learning curves, router behavior, numerical stability, state norms, throughput, and memory.
**Gate:** One candidate is selected on evidence, not aesthetics.
**Required evidence:** code commit, resolved config, test output, metric artifact, and a brief decision record stating promote / hold / rollback.
### W28 — 400M architecture freeze candidate | 2027-03-16 → 2027-03-22
**Build:** Freeze config, control-ID map, expert registry, and baseline training recipe for the main 400M run. Allow only explicitly logged bug fixes.
**Gate:** A release-candidate architecture hash is recorded.
**Required evidence:** code commit, resolved config, test output, metric artifact, and a brief decision record stating promote / hold / rollback.

## Month 08 — Local RAG and Provenance-Aware Context Injection
### Monthly exit gate
The month is passed only when the previous month remains reproducible and every listed weekly deliverable has an artifact path or benchmark record.
### W29 — Build corpus ingestion/index pipeline | 2027-03-23 → 2027-03-29
**Build:** Implement chunking, metadata, content hashing, embedding generation, vector storage, and versioned index manifests. SQLite/embedded vector search is the reference deployment candidate.
**Gate:** Index is reproducible from a manifest and can be rebuilt exactly.
**Required evidence:** code commit, resolved config, test output, metric artifact, and a brief decision record stating promote / hold / rollback.
### W30 — Implement retrieval and reranking | 2027-03-30 → 2027-04-05
**Build:** Support dense retrieval, optional lexical fallback, top-k retrieval, metadata filters, reranking, and context budgeting.
**Gate:** Offline retrieval benchmarks have recall/precision measurements.
**Required evidence:** code commit, resolved config, test output, metric artifact, and a brief decision record stating promote / hold / rollback.
### W31 — Integrate RAG into request pipeline | 2027-04-06 → 2027-04-12
**Build:** Keep retrieval external. Serialize provenance metadata and delimit retrieved context distinctly from model-native text.
**Gate:** A model run can be replayed with the exact retrieved documents.
**Required evidence:** code commit, resolved config, test output, metric artifact, and a brief decision record stating promote / hold / rollback.
### W32 — Run RAG ablation | 2027-04-13 → 2027-04-19
**Build:** Compare no-RAG vs RAG on closed-book, retrieval-heavy, stale-fact, and provenance tasks. Measure latency, memory, retrieval recall, answer faithfulness.
**Gate:** RAG is retained only where its measured benefit justifies its cost.
**Required evidence:** code commit, resolved config, test output, metric artifact, and a brief decision record stating promote / hold / rollback.

## Month 09 — Tools, Sandbox, Verification, and Repair
### Monthly exit gate
The month is passed only when the previous month remains reproducible and every listed weekly deliverable has an artifact path or benchmark record.
### W33 — Implement host-enforced tool broker | 2027-04-20 → 2027-04-26
**Build:** Define structured request/response schema for calculator, documentation lookup, code runner, and test runner. Reject malformed or unauthorized tool calls.
**Gate:** Tool permissions are controlled outside model-generated text.
**Required evidence:** code commit, resolved config, test output, metric artifact, and a brief decision record stating promote / hold / rollback.
### W34 — Build isolated code sandbox | 2027-04-27 → 2027-05-03
**Build:** Implement resource/time/process/file/network policy, filesystem isolation, stdout/stderr capture, exit codes, test limits, and cleanup.
**Gate:** Sandbox escape tests fail safely.
**Required evidence:** code commit, resolved config, test output, metric artifact, and a brief decision record stating promote / hold / rollback.
### W35 — Build verifier pipeline | 2027-05-04 → 2027-05-10
**Build:** Static parse → safety checks → optional dependency checks → test synthesis/selection → sandbox execution → result normalization.
**Gate:** Verification result has explicit categories: syntax failure, safety rejection, runtime failure, test failure, pass.
**Required evidence:** code commit, resolved config, test output, metric artifact, and a brief decision record stating promote / hold / rollback.
### W36 — Build bounded repair loop | 2027-05-11 → 2027-05-17
**Build:** Start with max 3 repair attempts. Record error, patch, rerun result, and stop reason. Call it execution-guided repair, not MCTS.
**Gate:** A benchmark measures pass@1, pass@repair-3, latency, and failure mode.
**Required evidence:** code commit, resolved config, test output, metric artifact, and a brief decision record stating promote / hold / rollback.

## Month 10 — Fine-Tuning, Specialization Profiles, and Multi-Token Prediction
### Monthly exit gate
The month is passed only when the previous month remains reproducible and every listed weekly deliverable has an artifact path or benchmark record.
### W37 — Implement SFT pipeline | 2027-05-18 → 2027-05-24
**Build:** Instruction tuning with strict provenance, role/control tokens, code formatting, tool protocols, and concise verifiable responses.
**Gate:** SFT checkpoint preserves base capabilities within accepted regression thresholds.
**Required evidence:** code commit, resolved config, test output, metric artifact, and a brief decision record stating promote / hold / rollback.
### W38 — Implement specialization profiles | 2027-05-25 → 2027-05-31
**Build:** Create JSON/YAML profiles for Python, C++, math, SQL, data science, and general. Profiles select reproducible training/evaluation recipes; they do not magically transform an existing checkpoint.
**Gate:** Profile execution reproduces the same resolved recipe hash.
**Required evidence:** code commit, resolved config, test output, metric artifact, and a brief decision record stating promote / hold / rollback.
### W39 — Run specialization experiments | 2027-06-01 → 2027-06-07
**Build:** Compare shared model vs specialization-specific checkpoints. Measure code correctness, language quality, retrieval usage, and memory/latency.
**Gate:** Specialization is retained only if specialization gain is meaningful.
**Required evidence:** code commit, resolved config, test output, metric artifact, and a brief decision record stating promote / hold / rollback.
### W40 — MTP ablation | 2027-06-08 → 2027-06-14
**Build:** Implement optional multi-byte/multi-position prediction objective as a training branch. Test whether it improves data efficiency or speculative decoding acceptance enough to justify added complexity.
**Gate:** MTP remains disabled by default unless it wins a predeclared gate.
**Required evidence:** code commit, resolved config, test output, metric artifact, and a brief decision record stating promote / hold / rollback.

## Month 11 — Quantization, Runtime, and Edge Deployment
### Monthly exit gate
The month is passed only when the previous month remains reproducible and every listed weekly deliverable has an artifact path or benchmark record.
### W41 — Build quantization calibration pipeline | 2027-06-15 → 2027-06-21
**Build:** Collect activation statistics on representative domain slices. Implement baseline RTN/symmetric/group-wise quantization and an AWQ-like activation-aware candidate.
**Gate:** Quantization metadata and calibration set are versioned.
**Required evidence:** code commit, resolved config, test output, metric artifact, and a brief decision record stating promote / hold / rollback.
### W42 — Measure recurrent stability under low precision | 2027-06-22 → 2027-06-28
**Build:** Test weight precision and recurrent-state precision separately. Monitor state norm, output drift, long-sequence degradation, loss spikes, NaN/Inf rate.
**Gate:** A quantized checkpoint is rejected if recurrence becomes numerically unstable.
**Required evidence:** code commit, resolved config, test output, metric artifact, and a brief decision record stating promote / hold / rollback.
### W43 — Integrate runtime backend(s) | 2027-06-29 → 2027-07-05
**Build:** Start from the custom reference runtime. Then evaluate llama.cpp/ggml or another backend only if current format and Mamba-2 support are actually available and validated.
**Gate:** Backend equivalence tests compare reference vs deployed outputs within declared tolerance.
**Required evidence:** code commit, resolved config, test output, metric artifact, and a brief decision record stating promote / hold / rollback.
### W44 — Run edge benchmark suite | 2027-07-06 → 2027-07-12
**Build:** Measure cold start, prompt/prefill time, decode time, bytes/s, peak RSS, resident weights, recurrent state, RAG memory, sandbox memory, thermals, and battery/energy where measurable.
**Gate:** All claims are device-specific and workload-specific.
**Required evidence:** code commit, resolved config, test output, metric artifact, and a brief decision record stating promote / hold / rollback.

## Month 12 — Latent Reasoning Research, Final Ablations, Release, and Paper
### Monthly exit gate
The month is passed only when the previous month remains reproducible and every listed weekly deliverable has an artifact path or benchmark record.
### W45 — Implement optional latent recurrent refinement | 2027-07-13 → 2027-07-19
**Build:** Weight-tied recurrent-depth branch. Keep it residual and base-preserving with K=1,2,4,8 experiments. Do not require exposed chain-of-thought.
**Gate:** When disabled, base model path is unchanged bit-for-bit or within declared numerical tolerance.
**Required evidence:** code commit, resolved config, test output, metric artifact, and a brief decision record stating promote / hold / rollback.
### W46 — Run latent compute scaling study | 2027-07-20 → 2027-07-26
**Build:** Evaluate quality vs extra recurrent iterations and latency. Test adaptive halting only after fixed-K results are stable.
**Gate:** No-regression gate covers reasoning, coding, instruction following, retrieval, safety, stability, memory, and latency.
**Required evidence:** code commit, resolved config, test output, metric artifact, and a brief decision record stating promote / hold / rollback.
### W47 — Complete full ablation matrix | 2027-07-27 → 2027-08-02
**Build:** Dense vs MoE; no-RAG vs RAG; no-verifier vs verifier; no-repair vs repair; baseline vs MTP; baseline vs latent reasoning; FP16/BF16 vs INT4.
**Gate:** Every feature has an isolated delta and measured cost.
**Required evidence:** code commit, resolved config, test output, metric artifact, and a brief decision record stating promote / hold / rollback.
### W48 — 400M final training/fine-tuning run | 2027-08-03 → 2027-08-09
**Build:** Run the final selected recipe. Freeze all data manifests and configuration.
**Gate:** Final checkpoint is immutable and hash-recorded.
**Required evidence:** code commit, resolved config, test output, metric artifact, and a brief decision record stating promote / hold / rollback.
### W49 — Final correctness and security audit | 2027-08-10 → 2027-08-16
**Build:** Run causality, state continuity, shape, router, tool, sandbox, RAG provenance, quantization, backend equivalence, reproducibility, and failure-injection tests.
**Gate:** All release blockers are either fixed or explicitly listed.
**Required evidence:** code commit, resolved config, test output, metric artifact, and a brief decision record stating promote / hold / rollback.
### W50 — Final edge audit | 2027-08-17 → 2027-08-23
**Build:** Repeat performance and memory benchmarks across supported devices/backends. Use multiple seeds where practical for confidence intervals.
**Gate:** Headline claims have evidence files attached.
**Required evidence:** code commit, resolved config, test output, metric artifact, and a brief decision record stating promote / hold / rollback.
### W51 — Research report and engineering documentation | 2027-08-24 → 2027-08-30
**Build:** Document architecture, exact configuration, datasets/provenance, formulas, training recipe, ablations, limitations, failed experiments, deployment constraints.
**Gate:** A third party can understand what was actually built.
**Required evidence:** code commit, resolved config, test output, metric artifact, and a brief decision record stating promote / hold / rollback.
### W52 — Release freeze and archive | 2027-08-31 → 2027-09-06
**Build:** Tag release, archive manifests/evidence, freeze repository, export model/runtime package, produce checksums and reproducibility bundle.
**Gate:** Release candidate is reproducible from the archived evidence set.
**Required evidence:** code commit, resolved config, test output, metric artifact, and a brief decision record stating promote / hold / rollback.

## 5. Weekly Operating Loop
- Day 1: define the hypothesis, exact config, dataset slice, expected metric direction, and rollback checkpoint.
- Day 2–3: implement the minimum isolated change; add unit/property tests before long training.
- Day 4: run a deterministic smoke experiment and numerical comparison to the trusted baseline.
- Day 5: run the smallest useful benchmark and inspect failure modes, not only aggregate score.
- Weekend / review boundary: archive evidence, update experiment ledger, decide promote / iterate / reject. Never silently fold an experimental branch into the baseline.
## 6. Build Order Dependencies
```text
Byte/control contract
      ↓
Patch encoder + local byte decoder
      ↓
Reference Mamba-2 block + recurrent state API
      ↓
20M dense baseline
      ↓
Config resolver + parameter/memory audit
      ↓
MoE router + expert state separation
      ↓
50M / 100M / 200M scaling
      ↓
400M candidate search + freeze
      ↓
RAG + provenance
      ↓
Tools + sandbox + verifier + repair
      ↓
SFT + specialization + optional MTP
      ↓
Quantization + runtime backend
      ↓
Latent reasoning branch
      ↓
Full ablation + final release
```
Anything above a node that has not passed its gate is a hard blocker for the dependent node, except for explicitly separated research spikes that do not contaminate the trusted baseline.
## 7. Master Experiment Matrix
```text
A0  Dense Mamba baseline                         | no extras
A1  Dense + byte hierarchy                       | tests hierarchical byte value
A2  MoE + shared canonical state                 | tests specialization value
A3  MoE + RAG                                    | tests external knowledge value
A4  MoE + RAG + verifier                         | tests execution feedback value
A5  A4 + bounded repair                          | tests repair value
A6  A4 + MTP                                    | tests auxiliary future prediction
A7  A4 + latent recurrent compute               | tests inference-time compute scaling
A8  A7 + INT4                                   | tests edge compression
A9  Final full stack                             | integrated candidate

For every pair, report: quality, code correctness, retrieval accuracy, router utilization, active parameters, total parameters, latency, throughput, peak RSS, model package size, recurrent state size, and energy where measurable.
```
## 8. Promotion Gates and Stop Conditions
```text
PROMOTE when:
- primary metric improves or remains statistically/operationally non-inferior,
- cost increase is within predeclared budget,
- failure rate does not worsen beyond threshold,
- regression suite passes,
- rollback path exists,
- evidence is reproducible.

HOLD when:
- result is promising but underpowered,
- benchmark variance is high,
- cost/benefit is ambiguous,
- numerical stability is unresolved.

REJECT/ROLLBACK when:
- causal correctness fails,
- tool/sandbox isolation fails,
- recurrent state becomes unstable,
- quality regresses outside the declared band,
- parameter/memory budget is violated,
- result cannot be reproduced,
- a component cannot be disabled cleanly.
```
Suggested default statistical reporting: mean, standard deviation, number of runs, and a confidence interval when enough independent runs exist. For one-off expensive runs, report that limitation explicitly rather than pretending a single run proves superiority.
## 9. Master Formula Board — Conventions
The equations below are the project-complete working mathematical inventory for implementing, debugging, training, measuring, and researching this system. It is not a claim to contain every formula in all of mathematics or machine learning. They are not all simultaneously active. A formula marked **optional** belongs to a branch and should not be implemented merely because it appears here.
Notation: scalar = lowercase italic; vector = bold lowercase; matrix = uppercase; elementwise product = `⊙`; elementwise division = `/`; transpose = `^T`; identity = `I`; expectation = `E`; indicator = `1{·}`. Dimensions are omitted from equations when they would reduce readability; every implementation must still validate dimensions programmatically.
## 10. Byte Stream and Hierarchical Sequence Geometry
### 10.1. Byte vocabulary
**Use:** Map raw 8-bit bytes to the base vocabulary.
**Status:** Reference / project contract
$$
\mathcal{V}_{\text{byte}} = \{0,1,\ldots,255\}
$$
**Implementation notes:**
- Internal control IDs extend this vocabulary but are versioned.
- Do not treat a concrete prototype control-ID count as a universal language-model law.
### 10.2. Extended control vocabulary
**Use:** Define the total model symbol space after reserving control IDs.
**Status:** Project-specific derived
$$
V = 256 + 1 + N_{\text{control}}
$$
**Implementation notes:**
- The `+1` represents the padding ID in the reference design.
### 10.3. Number of complete patches
**Use:** Convert a byte window of length L into non-overlapping patches of size p.
**Status:** Project geometry
$$
N_{\text{patch}} = \left\lceil \frac{L}{p} \right\rceil
$$
**Implementation notes:**
- Reference patch size p = 8.
- Padding is used only where the implementation explicitly permits it.
### 10.4. Number of routing chunks
**Use:** Convert patches into routing chunks containing c patches.
**Status:** Project geometry
$$
N_{\text{chunk}} = \left\lceil \frac{N_{\text{patch}}}{c} \right\rceil
$$
**Implementation notes:**
- Reference c = 16 patches = 128 bytes.
### 10.5. Bytes per routing chunk
**Use:** Translate routing granularity into raw-byte span.
**Status:** Project geometry
$$
L_{\text{chunk}} = p\,c
$$
**Implementation notes:**
- Reference: 8 × 16 = 128 bytes.
### 10.7. Padded sequence length
**Use:** Determine the storage length after padding to a patch boundary.
**Status:** Project geometry
$$
L_{\text{pad}}=p\left\lceil\frac{L}{p}\right\rceil
$$
### 10.8. Patch index of a byte
**Use:** Map a zero-based byte index t to its non-overlapping patch.
**Status:** Project geometry
$$
q(t)=\left\lfloor\frac{t}{p}\right\rfloor
$$
### 10.9. Chunk index of a patch
**Use:** Map a zero-based patch index q to its routing chunk.
**Status:** Project geometry
$$
c(q)=\left\lfloor\frac{q}{c}\right\rfloor
$$
### 10.6. Bytes represented by W patches
**Use:** Useful when computing receptive span and state transition counts.
**Status:** Derived
$$
L = Wp
$$
## 11. Embeddings, Linear Maps, Activations, and Normalization
### 11.1. Embedding lookup
**Use:** Map discrete byte/control ID to a learned vector.
**Status:** Reference
$$
\mathbf{e}_t = E[x_t]
$$
### 11.2. Linear layer
**Use:** Core affine transformation used throughout the network.
**Status:** Reference
$$
\mathbf{y} = W\mathbf{x} + \mathbf{b}
$$
### 11.3. RMSNorm
**Use:** Normalize a hidden vector by root mean square and apply learned scale.
**Status:** Reference / architecture
$$
\operatorname{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2 + \epsilon}} \odot \mathbf{g}
$$
**Implementation notes:**
- This is the normalization family relevant to the reference Mamba implementation.
- Do not silently substitute LayerNorm without treating it as an architecture change.
### 11.4. LayerNorm
**Use:** Reference normalization for comparison and controlled experiments.
**Status:** Reference / optional comparator
$$
\operatorname{LN}(\mathbf{x}) = \boldsymbol{\gamma} \odot \frac{\mathbf{x}-\mu}{\sqrt{\sigma^2+\epsilon}} + \boldsymbol{\beta}
$$
### 11.5. Mean
**Use:** Statistic used in normalization, pooling, and router summaries.
**Status:** Reference
$$
\mu = \frac{1}{d}\sum_{i=1}^{d} x_i
$$
### 11.6. Variance
**Use:** Statistic used by LayerNorm and diagnostics.
**Status:** Reference
$$
\sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(x_i-\mu)^2
$$
### 11.7. SiLU / Swish
**Use:** Default gate activation used by the official Mamba-2 block family.
**Status:** Reference
$$
\operatorname{SiLU}(x) = x\,\sigma(x) = \frac{x}{1+e^{-x}}
$$
### 11.8. Gated elementwise product
**Use:** Apply an activation-driven gate to sequence-mixer output.
**Status:** Reference / Mamba-2 block pattern
$$
\mathbf{y}_g = \mathbf{y} \odot \phi(\mathbf{z})
$$
### 11.9. Residual update
**Use:** Canonical residual connection.
**Status:** Reference architecture pattern
$$
\mathbf{h}_{\ell+1} = \mathbf{h}_{\ell} + F_{\ell}(\mathbf{h}_{\ell})
$$
### 11.10. Zero-initialized residual branch
**Use:** Base-preserving research branch initializer.
**Status:** Project-specific safety pattern
$$
\mathbf{h}_{\text{new}} = \mathbf{h}_{\text{base}} + g\,F(\mathbf{h}_{\text{base}}), \qquad g_0 = 0
$$
**Implementation notes:**
- Useful for optional latent reasoning or new adapters.
- Promotion still requires training and no-regression evaluation.
## 12. General State Space Model and Mamba-2 Mathematics
### 11.11. Sigmoid
**Use:** Gate probability and derivative checks.
**Status:** Reference
$$
\sigma(x) = \frac{1}{1+e^{-x}}
$$
### 11.12. Softmax Jacobian
**Use:** Debug router/output gradients and numerical checks.
**Status:** Reference
$$
\frac{\partial p_i}{\partial z_j}=p_i(\delta_{ij}-p_j)
$$
### 11.13. Cross-entropy gradient with softmax logits
**Use:** Verify the output-layer gradient for one-hot targets.
**Status:** Reference
$$
\frac{\partial \mathcal{L}}{\partial z_i}=p_i-y_i
$$
### 11.14. Log-sum-exp stabilization
**Use:** Compute log-softmax without overflow/underflow.
**Status:** Reference
$$
\operatorname{LSE}(\mathbf{z}) = m + \log\sum_i e^{z_i-m}, \qquad m=\max_i z_i
$$
### 11.15. Softplus derivative
**Use:** Check gradients of positive-step parameterizations.
**Status:** Reference
$$
\frac{d}{dx}\operatorname{softplus}(x)=\sigma(x)
$$
### 11.16. SiLU derivative
**Use:** Numerical gradient and kernel-equivalence testing.
**Status:** Reference
$$
\frac{d}{dx}\operatorname{SiLU}(x)=\sigma(x)+x\sigma(x)(1-\sigma(x))
$$
### 12.1. Discrete linear SSM recurrence
**Use:** Base state transition equation.
**Status:** Reference
$$
\mathbf{h}_t = A\mathbf{h}_{t-1} + B\mathbf{x}_t
$$
### 12.2. SSM output
**Use:** Map latent state to output.
**Status:** Reference
$$
\mathbf{y}_t = C^T\mathbf{h}_t
$$
### 12.3. Selective time-varying SSM
**Use:** Allow the state-space parameters to vary with the input/time.
**Status:** Reference / Mamba family
$$
\mathbf{h}_t = A_t\mathbf{h}_{t-1} + B_t\mathbf{x}_t, \qquad \mathbf{y}_t = C_t^T\mathbf{h}_t
$$
### 12.4. Continuous-time SSM
**Use:** Continuous dynamical system from which discrete parameters may be derived.
**Status:** Reference theory
$$
\frac{d\mathbf{h}(t)}{dt} = \bar{A}\mathbf{h}(t) + \bar{B}\mathbf{x}(t)
$$
### 12.5. Zero-order-hold discretization for A
**Use:** Standard discretization relationship under a constant input over the interval.
**Status:** Reference derivation
$$
A = e^{\Delta \bar{A}}
$$
### 12.6. Zero-order-hold discretization for B
**Use:** Standard ZOH input term when A is invertible.
**Status:** Reference derivation
$$
B = \bar{A}^{-1}\left(e^{\Delta\bar{A}}-I\right)\bar{B}
$$
### 12.7. Stable decay parameterization
**Use:** Use a non-positive continuous decay parameter so the exponential produces decaying dynamics.
**Status:** Implementation pattern / verify against pinned backend
$$
A = -e^{\alpha}
$$
### 12.8. Positive step size
**Use:** Parameterize the discretization step size so it is strictly positive.
**Status:** Implementation pattern / Mamba family
$$
\Delta = \operatorname{softplus}(\tilde{\Delta})
$$
### 12.9. Mamba-2 input projections
**Use:** Parallel projection pattern highlighted by the Mamba-2 architecture.
**Status:** Reference architecture
$$
\mathbf{x} = \mathbf{u}{W^{(x)}}^T, \qquad \mathbf{z} = \mathbf{u}{W^{(z)}}^T
$$
### 12.10. Mamba-2 depthwise causal convolution
**Use:** Local pre-SSM mixing used in the reference block.
**Status:** Reference architecture
$$
\mathbf{x}_c = \operatorname{Conv1D}_{\text{causal,depthwise}}(\mathbf{x})
$$
### 12.11. Mamba-2 gated state-space output
**Use:** Core reference block flow.
**Status:** Reference architecture
$$
\mathbf{y} = \operatorname{SSM}_{A,B,C,\Delta}(\mathbf{x}_c), \qquad \mathbf{y}_g = \mathbf{y}\odot\operatorname{SiLU}(\mathbf{z})
$$
### 12.12. Mamba-2 final projection
**Use:** Return the mixer output to canonical model width.
**Status:** Reference architecture
$$
\operatorname{out} = \mathbf{y}_g{W^{(o)}}^T
$$
### 12.13. Mamba-2 inner width
**Use:** Derive expanded mixer width from model width and expansion factor.
**Status:** Reference configuration identity
$$
d_{\text{inner}}=\operatorname{expand}\cdot d_{\text{model}}
$$
### 12.14. Mamba-2 head count
**Use:** Derive the number of SSM heads when the implementation specifies a head dimension.
**Status:** Reference configuration identity
$$
H=\frac{d_{\text{inner}}}{d_{\text{head}}}
$$
### 12.15. Rough Mamba block parameter estimate
**Use:** Fast parameter-search screening before exact instantiation.
**Status:** Official-repository approximation; not an exact audit
$$
N_{\text{block}}\approx 3\,e\,d_{\text{model}}^2
$$
The official repository describes this as a rough parameter estimate; exact serialized parameter counts must come from the instantiated graph.
### 12.16. Recurrent state memory planning approximation
**Use:** Early memory screening for the SSM state; exact backend layout must be measured.
**Status:** Project planning approximation
$$
M_{\text{SSM-state}}\approx B\,L\,H\,N\,s_{\text{dtype}}
$$
### 12.17. Associative affine-state composition
**Use:** Reason about scan/chunk composition for recurrences of the form h' = Ah + b.
**Status:** Reference algebra
$$
(A_2,b_2)\circ(A_1,b_1)=\left(A_2A_1,\;A_2b_1+b_2\right)
$$
### 12.18. Generic state norm monitor
**Use:** Monitor recurrent magnitude for stability diagnostics.
**Status:** Project diagnostic
$$
\|\mathbf{h}_t\|_2 = \sqrt{\sum_i h_{t,i}^2}
$$
### 12.19. State drift between two implementations
**Use:** Quantify numerical difference between reference and optimized kernels.
**Status:** Project verification
$$
\delta_{\text{state}}(t) = \frac{\|\mathbf{h}_t^{(A)}-\mathbf{h}_t^{(B)}\|_2}{\max\left(\|\mathbf{h}_t^{(A)}\|_2,\,\eta\right)}
$$
## 13. Patch Encoder / Local Decoder Mathematics
### 13.1. Patch extraction
**Use:** Formalize the j-th patch from a byte stream.
**Status:** Project-specific
$$
\mathbf{p}_j = \left[x_{jp},x_{jp+1},\ldots,x_{jp+p-1}\right]
$$
### 13.2. Patch embedding aggregation
**Use:** Generic differentiable patch aggregation before the shared backbone.
**Status:** Project-specific reference option
$$
\mathbf{e}^{\text{patch}}_j = f_{\text{patch}}(E[\mathbf{p}_j])
$$
### 13.3. Byte-causal local decoder factorization
**Use:** Factorize the probability of the bytes within a patch.
**Status:** Project causality requirement
$$
P(x_{1:p}\mid c) = \prod_{j=1}^{p} P\left(x_j \mid x_{<j},c\right)
$$
### 13.4. Whole-sequence byte likelihood
**Use:** Autoregressive factorization over the full sequence.
**Status:** Core language-model objective
$$
P(x_{1:T}\mid c) = \prod_{t=1}^{T} P(x_t\mid x_{<t},c)
$$
### 13.5. Causal dependency condition
**Use:** A formal test condition for no future-byte leakage.
**Status:** Project invariant
$$
\frac{\partial \mathbf{h}_t}{\partial x_{t+k}} = \mathbf{0}\qquad\forall k>0
$$
### 13.6. Patch causality condition
**Use:** A completed patch may affect only subsequent patch states/predictions.
**Status:** Project invariant
$$
\frac{\partial \mathbf{s}_{j}}{\partial x_{m}} = \mathbf{0}\qquad\forall m>jp+p-1
$$
## 14. Softmax, Probabilities, Sampling, and Information Theory
### 14.1. Softmax
**Use:** Convert logits to a categorical probability distribution.
**Status:** Reference
$$
\operatorname{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{V}e^{z_j}}
$$
### 14.2. Stable softmax
**Use:** Numerically stable implementation by subtracting the maximum logit.
**Status:** Implementation
$$
\operatorname{softmax}(\mathbf{z})_i = \frac{e^{z_i-z_{\max}}}{\sum_j e^{z_j-z_{\max}}}
$$
### 14.3. Temperature sampling
**Use:** Control output distribution sharpness.
**Status:** Inference
$$
P_T(i) = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}
$$
### 14.4. Categorical sampling
**Use:** Sample token i from the model distribution.
**Status:** Inference
$$
i \sim \operatorname{Categorical}(P)
$$
### 14.5. Entropy
**Use:** Measure uncertainty of a categorical distribution.
**Status:** Router/output diagnostic
$$
H(P) = -\sum_i P_i\log P_i
$$
### 14.6. Perplexity
**Use:** Convert mean negative log-likelihood into exponential perplexity.
**Status:** Language-model evaluation
$$
\operatorname{PPL} = \exp\left(-\frac{1}{T}\sum_{t=1}^{T}\log P(x_t\mid x_{<t})\right)
$$
### 14.7. Cross entropy
**Use:** Core next-byte/token training loss.
**Status:** Core training objective
$$
\mathcal{L}_{\text{CE}} = -\frac{1}{N}\sum_{n=1}^{N}\log P_\theta(y_n\mid x_n)
$$
### 14.8. Masked cross entropy
**Use:** Ignore padded positions during loss computation.
**Status:** Training
$$
\mathcal{L}_{\text{masked}} = -\frac{1}{\sum_t m_t}\sum_t m_t\log P_\theta(y_t\mid x_{<t})
$$
### 14.9. KL divergence
**Use:** Compare predicted and target distributions, used in distillation or diagnostics.
**Status:** Distillation / analysis
$$
D_{\mathrm{KL}}(P\|Q) = \sum_i P_i\log\frac{P_i}{Q_i}
$$
### 14.10. Label-smoothed target distribution
**Use:** Optional regularization for classification-like heads.
**Status:** Optional training branch
$$
q_i = (1-\varepsilon)\,\mathbf{1}_{i=y} + \frac{\varepsilon}{V}
$$
## 15. MoE Router, Top-1 Selection, Capacity, and Load Balance
### 15.1. Router logits
**Use:** Compute expert scores from the canonical chunk representation.
**Status:** Project-specific
$$
\mathbf{r} = W_r\mathbf{c} + \mathbf{b}_r
$$
### 15.2. Router probabilities
**Use:** Normalize expert logits.
**Status:** MoE routing
$$
p_i = \frac{e^{r_i}}{\sum_{j=1}^{E}e^{r_j}}
$$
### 15.3. Top-1 expert selection
**Use:** Select one expert per chunk in the baseline MoE.
**Status:** MoE routing
$$
k^* = \operatorname*{arg\,max}_{i\in\{1,\ldots,E\}} p_i
$$
### 15.4. Top-1 gate value
**Use:** Use the winning router probability as the dispatch gate.
**Status:** MoE routing
$$
g = p_{k^*}
$$
### 15.5. Switch fraction routed to expert
**Use:** Exact form used by Switch-style load balancing.
**Status:** Reference — Switch Transformer
$$
f_i = \frac{1}{T}\sum_{x\in\mathcal{B}} \mathbf{1}\{\operatorname*{arg\,max} p(x)=i\}
$$
### 15.6. Mean router probability per expert
**Use:** Exact Switch-style probability fraction.
**Status:** Reference — Switch Transformer
$$
P_i = \frac{1}{T}\sum_{x\in\mathcal{B}} p_i(x)
$$
### 15.7. Switch auxiliary load-balance loss
**Use:** Encourage balanced routing.
**Status:** Reference — Switch Transformer
$$
\mathcal{L}_{\text{aux}} = \alpha E \sum_{i=1}^{E} f_i P_i
$$
**Implementation notes:**
- The paper multiplies the dot product by the number of experts E so uniform routing does not shrink the loss as E grows.
- The coefficient α is a tunable hyperparameter, not a universal constant for this project.
### 15.8. Capacity per expert
**Use:** Estimate fixed expert capacity for a batch.
**Status:** MoE capacity planning
$$
C = \left\lceil \operatorname{capacity\_factor}\cdot\frac{T}{E}\right\rceil
$$
### 15.9. Expert overflow indicator
**Use:** Mark assignments beyond capacity.
**Status:** MoE runtime
$$
o_{t} = \mathbf{1}\{n_{k^*} > C\}
$$
### 15.10. Router load coefficient of variation
**Use:** Quantify imbalance independently of auxiliary loss.
**Status:** MoE diagnostic
$$
\operatorname{CV}_{\text{load}} = \frac{\operatorname{std}(n_1,\ldots,n_E)}{\operatorname{mean}(n_1,\ldots,n_E)}
$$
### 15.11. Normalized routing entropy
**Use:** Compare router confidence across experiments.
**Status:** MoE diagnostic
$$
H_{\text{norm}} = \frac{-\sum_{i=1}^{E}p_i\log p_i}{\log E}
$$
### 15.12. Expert utilization rate
**Use:** Measure how often experts actually receive work.
**Status:** MoE diagnostic
$$
U_i = \frac{n_i}{\sum_j n_j}
$$
### 15.13. Canonical state bridge
**Use:** Generic explicit state fusion between shared and expert branches.
**Status:** Project-specific interface
$$
\mathbf{s}_{\text{next}} = \mathbf{s}_{\text{shared}} + G_\theta\!\left(\mathbf{s}_{\text{expert}},\mathbf{c}\right)
$$
### 15.14. KL divergence from uniform routing
**Use:** Diagnose systematic deviation of routing probabilities from equal allocation.
**Status:** MoE diagnostic
$$
D_{\mathrm{KL}}(U\|P)=\sum_{i=1}^{E}\frac{1}{E}\log\frac{1/E}{P_i}
$$
### 15.15. Gini coefficient of expert loads
**Use:** Measure inequality of token/chunk counts across experts.
**Status:** MoE diagnostic
$$
G=\frac{\sum_{i=1}^{E}\sum_{j=1}^{E}|n_i-n_j|}{2E\sum_{i=1}^{E}n_i}
$$
## 16. Multi-Objective Training, MTP, SFT, Preference Optimization, Distillation
### 16.1. Total weighted training loss
**Use:** Combine independently validated training objectives.
**Status:** Training framework
$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{LM}} + \lambda_{\text{aux}}\mathcal{L}_{\text{aux}} + \lambda_{\text{MTP}}\mathcal{L}_{\text{MTP}} + \lambda_{\text{pref}}\mathcal{L}_{\text{pref}} + \lambda_{\text{dist}}\mathcal{L}_{\text{dist}}
$$
### 16.2. Multi-token prediction loss
**Use:** Predict multiple future positions from the same causal state.
**Status:** Optional MTP branch
$$
\mathcal{L}_{\text{MTP}} = \sum_{k=1}^{D}\lambda_k\left(-\frac{1}{T-k}\sum_{t=1}^{T-k}\log P_\theta(x_{t+k}\mid x_{\le t})\right)
$$
**Implementation notes:**
- The exact MTP architecture can instead use sequential prediction modules; the loss expression is the aggregate conceptual objective.
### 16.3. Supervised fine-tuning loss
**Use:** Train on instruction/output pairs.
**Status:** SFT
$$
\mathcal{L}_{\text{SFT}} = -\frac{1}{N}\sum_{n=1}^{N}\log P_\theta(y_n\mid x_n)
$$
### 16.4. Pairwise logistic preference loss
**Use:** Reference form for a DPO-style preference branch.
**Status:** Optional preference optimization
$$
\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log \sigma\left(\beta\left(\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}-\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right)\right)\right]
$$
### 16.5. Knowledge-distillation loss
**Use:** Match student probabilities to a teacher.
**Status:** Optional distillation
$$
\mathcal{L}_{\text{KD}} = T^2 D_{\mathrm{KL}}\left(P_{\text{teacher}}^{(T)}\|P_{\text{student}}^{(T)}\right)
$$
### 16.6. Temperature-scaled distribution
**Use:** Distillation temperature.
**Status:** Optional distillation
$$
P_i^{(T)} = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}
$$
## 17. Optimization, Gradients, Clipping, Scheduling, and Weight Decay
### 17.1. Gradient definition
**Use:** Parameter gradient used by all gradient-based training.
**Status:** Reference
$$
g_t = \nabla_\theta \mathcal{L}(\theta_t)
$$
### 17.2. Gradient clipping by global norm
**Use:** Prevent catastrophic updates from large gradient norms.
**Status:** Training stability
$$
\tilde{\mathbf{g}} = \mathbf{g}\,\min\left(1,\frac{\tau}{\|\mathbf{g}\|_2}\right)
$$
### 17.3. Adam first moment
**Use:** Exponential moving average of gradients.
**Status:** Adam/AdamW
$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t
$$
### 17.4. Adam second moment
**Use:** Exponential moving average of squared gradients.
**Status:** Adam/AdamW
$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2
$$
### 17.5. Adam bias correction
**Use:** Correct zero initialization bias.
**Status:** Adam/AdamW
$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$
### 17.6. Adam parameter update
**Use:** Base adaptive gradient step.
**Status:** Adam/AdamW
$$
\theta_{t+1} = \theta_t - \eta_t\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
$$
### 17.7. AdamW decoupled weight decay
**Use:** Apply weight decay independently of the gradient normalization.
**Status:** AdamW
$$
\theta_{t+1} = \theta_t - \eta_t\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}+\lambda\theta_t\right)
$$
### 17.8. Warmup learning rate
**Use:** Common linear warmup.
**Status:** Training schedule
$$
\eta_t = \eta_{\max}\frac{t}{T_{\text{warm}}}\qquad 0\le t\le T_{\text{warm}}
$$
### 17.9. Cosine decay
**Use:** Standard cosine schedule after warmup.
**Status:** Training schedule
$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})\left(1+\cos\left(\pi\frac{t-T_w}{T-T_w}\right)\right)
$$
### 17.10. Exponential moving average
**Use:** Generic EMA for model weights or running diagnostics.
**Status:** Optional stabilization / teacher
$$
\theta^{\text{EMA}}_t = \rho\theta^{\text{EMA}}_{t-1} + (1-\rho)\theta_t
$$
### 17.11. Gradient accumulation
**Use:** Equivalent average gradient across K microbatches when loss scaling is done correctly.
**Status:** Training infrastructure
$$
g = \frac{1}{K}\sum_{k=1}^{K}g^{(k)}
$$
## 18. Initialization, Variance, and Numerical Safety
### 18.1. Xavier/Glorot variance heuristic
**Use:** Baseline initialization for linear layers when appropriate.
**Status:** Initialization comparator
$$
\operatorname{Var}(W_{ij}) \approx \frac{2}{n_{\text{in}}+n_{\text{out}}}
$$
### 18.2. He/Kaiming variance heuristic
**Use:** Baseline initialization for rectified activations.
**Status:** Initialization comparator
$$
\operatorname{Var}(W_{ij}) \approx \frac{2}{n_{\text{in}}}
$$
### 18.3. Finite-value predicate
**Use:** Numerical health check.
**Status:** Debugging
$$
\operatorname{finite}(x) \iff x \ne \operatorname{NaN}\;\land\;|x|<\infty
$$
### 18.4. Relative error
**Use:** Core numerical equivalence metric.
**Status:** Kernel/reference validation
$$
\operatorname{RelErr}(a,b) = \frac{\|a-b\|_2}{\max(\|a\|_2,\eta)}
$$
### 18.5. Maximum absolute error
**Use:** Catch localized discrepancies hidden by norm averages.
**Status:** Kernel/reference validation
$$
\operatorname{MaxErr}(a,b) = \max_i |a_i-b_i|
$$
## 19. RAG, Embeddings, Similarity, Retrieval, and Reranking
### 19.1. Dot-product similarity
**Use:** Fast vector similarity when embeddings are appropriately scaled.
**Status:** RAG retrieval
$$
\operatorname{sim}_{\text{dot}}(\mathbf{q},\mathbf{d}) = \mathbf{q}^T\mathbf{d}
$$
### 19.2. Cosine similarity
**Use:** Scale-invariant vector similarity.
**Status:** RAG retrieval
$$
\operatorname{sim}_{\cos}(\mathbf{q},\mathbf{d}) = \frac{\mathbf{q}^T\mathbf{d}}{\|\mathbf{q}\|_2\|\mathbf{d}\|_2}
$$
### 19.3. Euclidean distance
**Use:** Alternative metric for vector retrieval.
**Status:** RAG retrieval comparator
$$
d_2(\mathbf{q},\mathbf{d}) = \sqrt{\sum_i(q_i-d_i)^2}
$$
### 19.4. Top-k retrieval set
**Use:** Select highest scoring documents.
**Status:** RAG retrieval
$$
\mathcal{R}_k(q) = \operatorname{TopK}_{d\in\mathcal{D}}\;s(q,d)
$$
### 19.5. Precision@k
**Use:** Measure fraction of retrieved items that are relevant.
**Status:** RAG evaluation
$$
\operatorname{Precision@k} = \frac{|\mathcal{R}_k\cap\mathcal{G}|}{k}
$$
### 19.6. Recall@k
**Use:** Measure fraction of relevant items retrieved.
**Status:** RAG evaluation
$$
\operatorname{Recall@k} = \frac{|\mathcal{R}_k\cap\mathcal{G}|}{|\mathcal{G}|}
$$
### 19.7. F1@k
**Use:** Combine precision and recall.
**Status:** RAG evaluation
$$
F1@k = \frac{2\,\operatorname{Precision@k}\,\operatorname{Recall@k}}{\operatorname{Precision@k}+\operatorname{Recall@k}}
$$
### 19.8. Reciprocal rank
**Use:** Basic ranked-retrieval metric.
**Status:** RAG evaluation
$$
\operatorname{RR} = \frac{1}{\operatorname{rank}(d^*)}
$$
### 19.9. Mean reciprocal rank
**Use:** Average reciprocal rank over queries.
**Status:** RAG evaluation
$$
\operatorname{MRR} = \frac{1}{N}\sum_{n=1}^{N}\frac{1}{\operatorname{rank}_n(d_n^*)}
$$
### 19.10. Context budget constraint
**Use:** Formalize retrieval context length constraints.
**Status:** Runtime
$$
\sum_{j\in\mathcal{R}_k}|d_j|_{\text{bytes}} + |q|_{\text{bytes}} \le B_{\text{context}}
$$
## 20. Tool Use, Verification, Repair, and Search
### 19.11. BM25 lexical retrieval score
**Use:** Optional lexical fallback or hybrid retrieval component.
**Status:** Reference IR formula / optional
$$
\operatorname{BM25}(D,Q)=\sum_{q_i\in Q}\operatorname{IDF}(q_i)\frac{f(q_i,D)(k_1+1)}{f(q_i,D)+k_1\left(1-b+b\frac{|D|}{\operatorname{avgdl}}\right)}
$$
### 19.12. NDCG@k
**Use:** Evaluate ranked retrieval quality when graded relevance labels exist.
**Status:** Reference IR metric
$$
\operatorname{NDCG}@k=\frac{\operatorname{DCG}@k}{\operatorname{IDCG}@k},\qquad \operatorname{DCG}@k=\sum_{i=1}^{k}\frac{2^{rel_i}-1}{\log_2(i+1)}
$$
### 20.1. Binary execution success
**Use:** Convert a test result into an explicit success signal.
**Status:** Verifier
$$
S = \mathbf{1}\{\text{all required tests pass}\}
$$
### 20.2. Pass@k estimator
**Use:** Estimate probability at least one of k samples succeeds under the standard unbiased estimator form.
**Status:** Code-generation evaluation
$$
\operatorname{pass@}k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}
$$
**Implementation notes:**
- Requires n generated samples and c correct samples under the standard evaluation setup.
- Do not confuse pass@k with pass@repair-k unless the sampling/repair protocol is explicitly defined.
### 20.3. Repair success rate
**Use:** Measure fraction of failures repaired by bounded feedback.
**Status:** Repair evaluation
$$
R_{k} = \frac{N_{\text{repaired within }k}}{N_{\text{initial failures}}}
$$
### 20.4. Expected verification utility
**Use:** Combine correctness gain and execution cost when deciding whether verification is worthwhile.
**Status:** Project-specific decision metric
$$
U = \Delta Q - \lambda_t\Delta t - \lambda_m\Delta M
$$
### 20.5. Repair stopping condition
**Use:** Formalize the bounded repair policy.
**Status:** Project-specific
$$
\text{stop if } S=1\;\lor\;r\ge r_{\max}\;\lor\;\text{policy violation}=1
$$
### 20.6a. Idealized success probability after K independent attempts
**Use:** Upper-level intuition for why bounded retries can increase success; not a claim about correlated repair attempts.
**Status:** Analysis approximation
$$
P_{\le K}=1-(1-p)^K
$$
### 20.6. UCB1
**Use:** Classical exploration/exploitation score for a real tree/search branch if MCTS is eventually implemented.
**Status:** Optional true-search branch
$$
UCB1_i = \bar{X}_i + c\sqrt{\frac{\ln N}{n_i}}
$$
**Implementation notes:**
- Do not call the current bounded repair loop MCTS.
### 20.7. PUCT
**Use:** Policy-guided tree-search selection score for a more modern MCTS branch.
**Status:** Optional true-search branch
$$
\operatorname{PUCT}(i) = Q_i + c_{\text{puct}}P_i\frac{\sqrt{N}}{1+n_i}
$$
### 20.8. Monte Carlo value estimate
**Use:** Average rollout/evaluation value for an actual search node.
**Status:** Optional true-search branch
$$
Q_i = \frac{1}{n_i}\sum_{j=1}^{n_i}v_{i,j}
$$
## 21. Quantization and Low-Precision Mathematics
### 21.1. Uniform symmetric quantization
**Use:** Map a real scalar to signed integer levels.
**Status:** Reference quantization
$$
q = \operatorname{clip}\left(\operatorname{round}\left(\frac{x}{s}\right),q_{\min},q_{\max}\right)
$$
### 21.2. Symmetric scale
**Use:** Choose a scale from the maximum absolute value.
**Status:** Reference quantization
$$
s = \frac{\max_i|x_i|}{q_{\max}}
$$
### 21.3. Dequantization
**Use:** Recover the approximate real value.
**Status:** Reference quantization
$$
\hat{x} = s\,q
$$
### 21.4. Asymmetric affine quantization
**Use:** Map a finite range to an integer interval.
**Status:** Reference comparator
$$
q = \operatorname{clip}\left(\operatorname{round}\left(\frac{x-z}{s}\right),q_{\min},q_{\max}\right)
$$
### 21.5. Group-wise scale
**Use:** Compute independent quantization scales over groups of weights.
**Status:** INT4 deployment
$$
s_g = \frac{\max_{i\in g}|w_i|}{q_{\max}}
$$
### 21.6. Quantization mean squared error
**Use:** Measure squared reconstruction error after quantization.
**Status:** Quantization evaluation
$$
\operatorname{MSE}_{\text{Q}} = \frac{1}{N}\sum_{i=1}^{N}(x_i-\hat{x}_i)^2
$$
### 21.7. Relative quantization error
**Use:** Normalize quantization error by signal energy.
**Status:** Quantization evaluation
$$
\operatorname{RelMSE} = \frac{\|x-\hat{x}\|_2^2}{\max(\|x\|_2^2,\eta)}
$$
### 21.8. Layer output error
**Use:** Measure whether quantization preserves the function of a layer better than weight error alone.
**Status:** Quantization calibration
$$
\mathcal{E}_{\text{layer}} = \frac{\|f(x;W)-f(x;\hat{W})\|_2}{\max(\|f(x;W)\|_2,\eta)}
$$
### 21.9. AWQ-style activation-aware scaling intuition
**Use:** Channel-wise scaling used to reduce quantization error for salient channels.
**Status:** AWQ research branch
$$
\operatorname{Err}\left((Q(W\,\operatorname{diag}(s)))\,\operatorname{diag}(s)^{-1}\mathbf{x}\right)
$$
### 21.10. Weight payload at b bits
**Use:** Theoretical raw weight payload before metadata/packing overhead.
**Status:** Memory accounting
$$
M_{\text{raw}} = \frac{P\,b}{8}
$$
### 21.11. INT4 theoretical weight payload for 400M parameters
**Use:** Raw bit payload under exactly four bits/parameter.
**Status:** Reference budget estimate
$$
M_{\text{raw,INT4}} = \frac{400\times10^6\times4}{8}\;\text{bytes} = 200\times10^6\;\text{bytes}
$$
This is a bit payload only. It is **not** the total resident RAM. Scales, zero-points/metadata, padding, runtime buffers, recurrent state, allocator overhead, RAG, sandbox, and code are measured separately.
## 22. Parameter Counting and Model Scaling
### 22.0. Serialized bits per parameter
**Use:** Measure real model packaging density after packing and metadata.
**Status:** Release accounting
$$
b_{\text{serial}}=\frac{8M_{\text{serialized bytes}}}{N_{\text{params}}}
$$
### 22.0.1. Compression ratio
**Use:** Compare a quantized artifact against a reference-precision artifact.
**Status:** Release accounting
$$
CR=\frac{M_{\text{reference}}}{M_{\text{quantized}}}
$$
### 22.1. Dense linear layer parameters
**Use:** Count a matrix plus optional bias.
**Status:** Parameter audit
$$
N_{\text{params}} = d_{\text{out}}d_{\text{in}} + d_{\text{out}}\,\mathbf{1}_{\text{bias}}
$$
### 22.2. Embedding parameters
**Use:** Count a learned embedding table.
**Status:** Parameter audit
$$
N_{\text{embed}} = Vd
$$
### 22.3. Stacked block parameters
**Use:** Sum per-block parameter counts.
**Status:** Parameter audit
$$
N_{\text{stack}} = \sum_{\ell=1}^{L}N_\ell
$$
### 22.4. MoE total parameter count
**Use:** Total model parameters include all experts, even inactive ones.
**Status:** MoE accounting
$$
N_{\text{total}} = N_{\text{shared}} + \sum_{e=1}^{E}N_{\text{expert},e} + N_{\text{router}} + N_{\text{head}} + N_{\text{other}}
$$
### 22.5. Active parameters per routed chunk
**Use:** Count only the expert actually activated in a top-1 chunk.
**Status:** Sparse compute accounting
$$
N_{\text{active}} = N_{\text{shared-active}} + N_{\text{router}} + N_{\text{selected expert}} + N_{\text{head-active}}
$$
### 22.6. Parameter tolerance
**Use:** Test exact model count against target.
**Status:** Release gate
$$
\operatorname{RelParamErr} = \frac{|N_{\text{actual}}-N_{\text{target}}|}{N_{\text{target}}}
$$
### 22.7. Promotion condition for the 400M model
**Use:** Formalize the size gate.
**Status:** Project-specific release gate
$$
\operatorname{RelParamErr} \le \tau_{\text{param}}
$$
### 22.8. Depth-width search objective
**Use:** Rank candidate configurations under a parameter budget.
**Status:** Hardware-aware search
$$
J(c) = w_p\,\operatorname{RelParamErr}(c) + w_m\,\operatorname{MemoryPenalty}(c) + w_l\,\operatorname{LatencyPenalty}(c)
$$
## 23. Compute, Throughput, Latency, and Memory Accounting
### 23.1. Multiply-add approximation for a dense matrix product
**Use:** Baseline compute estimate for linear layers.
**Status:** Performance model
$$
\operatorname{MACs} \approx d_{\text{in}}d_{\text{out}}T
$$
### 23.2. FLOP approximation from MACs
**Use:** A common engineering approximation counting multiply and add separately.
**Status:** Performance model
$$
\operatorname{FLOPs} \approx 2\times\operatorname{MACs}
$$
### 23.3. Throughput
**Use:** Generated bytes per second.
**Status:** Runtime benchmark
$$
\operatorname{TPS}_{\text{byte}} = \frac{N_{\text{generated bytes}}}{t_{\text{decode}}}
$$
### 23.15. Decimal and binary memory units
**Use:** Keep the 250 MB target distinct from MiB measurements.
**Status:** Measurement convention
$$
1\,\mathrm{MB}=10^6\,\mathrm{bytes},\qquad 1\,\mathrm{MiB}=2^{20}\,\mathrm{bytes}
$$
### 23.16. Effective bytes per second
**Use:** Primary throughput metric for a byte-native language model.
**Status:** Runtime benchmark
$$
R_{\text{bytes}}=\frac{N_{\text{generated bytes}}}{T_{\text{decode}}}
$$
### 23.17. Latency percentile
**Use:** Report p50/p95/p99 latency consistently from repeated measurements.
**Status:** Runtime benchmark
$$
p_q=\text{the }\lceil qN\rceil\text{-th value after sorting }N\text{ latency samples}
$$
### 23.4. Latency per generated byte
**Use:** Reciprocal of throughput.
**Status:** Runtime benchmark
$$
L_{\text{byte}} = \frac{t_{\text{decode}}}{N_{\text{generated bytes}}}
$$
### 23.5. End-to-end latency
**Use:** Total observed request time.
**Status:** Runtime benchmark
$$
T_{\text{E2E}} = T_{\text{normalize}} + T_{\text{retrieve}} + T_{\text{prefill}} + T_{\text{decode}} + T_{\text{verify}} + T_{\text{repair}}
$$
### 23.6. Peak memory decomposition
**Use:** Separate major memory contributors.
**Status:** Edge memory audit
$$
M_{\text{peak}} \approx M_{\text{weights}} + M_{\text{state}} + M_{\text{activations}} + M_{\text{runtime}} + M_{\text{RAG}} + M_{\text{sandbox}} + M_{\text{allocator}}
$$
### 23.7. Recurrent-state memory
**Use:** Generic state-size estimate for batch, layers, state channels, and bytes/element.
**Status:** Edge memory audit
$$
M_{\text{state}} \approx B\,L\,H_{\text{state}}\,s_{\text{dtype}}
$$
### 23.8. Weight memory by precision
**Use:** Raw parameter storage for a chosen bit width.
**Status:** Edge memory audit
$$
M_{\text{weights}} \approx N_{\text{params}}\frac{b}{8}
$$
### 23.9. Resident-memory margin
**Use:** Measure remaining headroom against a device budget.
**Status:** Edge deployment gate
$$
M_{\text{margin}} = M_{\text{budget}} - M_{\text{peak}}
$$
### 23.10. Memory utilization ratio
**Use:** Normalize peak memory to budget.
**Status:** Edge deployment gate
$$
R_M = \frac{M_{\text{peak}}}{M_{\text{budget}}}
$$
### 23.11. Latency improvement
**Use:** Compare baseline and candidate.
**Status:** Benchmarking
$$
\Delta L = \frac{L_{\text{base}}-L_{\text{new}}}{L_{\text{base}}}
$$
### 23.12. Quality-per-memory
**Use:** Core edge research metric.
**Status:** Research thesis
$$
Q_M = \frac{Q}{M_{\text{peak}}}
$$
### 23.13. Quality-per-latency
**Use:** Core edge research metric.
**Status:** Research thesis
$$
Q_L = \frac{Q}{T_{\text{E2E}}}
$$
### 23.14. Quality-per-active-parameter
**Use:** Sparse efficiency metric.
**Status:** MoE research
$$
Q_A = \frac{Q}{N_{\text{active}}}
$$
## 24. Scaling Laws, Data Allocation, and Training Budget
### 24.1. Generic empirical power law
**Use:** A common empirical form for scaling experiments.
**Status:** Research analysis — empirical, not a law of nature
$$
L(N,D,C) \approx L_\infty + aN^{-\alpha} + bD^{-\beta} + cC^{-\gamma}
$$
### 24.2. Compute-optimal proportional scaling heuristic
**Use:** Chinchilla-style empirical guidance for jointly scaling parameters and data under a fixed compute regime.
**Status:** Research prior — not a universal constant
$$
D \propto N
$$
### 24.3. Training compute approximation
**Use:** Simplified compute budget approximation for dense autoregressive training.
**Status:** Planning estimate
$$
C_{\text{train}} \approx \kappa\,N_{\text{params}}\,N_{\text{tokens}}
$$
### 24.4. Training tokens per parameter
**Use:** Track how much data is used relative to the model size.
**Status:** Planning diagnostic
$$
\rho_{\text{data}} = \frac{N_{\text{tokens}}}{N_{\text{params}}}
$$
### 24.5. Loss improvement per compute
**Use:** Compare experiments fairly when compute differs.
**Status:** Experiment analysis
$$
\Delta L/\Delta C = \frac{L_1-L_2}{C_2-C_1}
$$
### 24.6. Scaling-law candidate fit
**Use:** Fit log-log dependence for a parameter sweep.
**Status:** Research analysis
$$
\log(L-L_\infty) \approx \log a - \alpha\log N
$$
## 25. Data Mixtures, Sampling, Deduplication, and Contamination
### 25.1. Mixture sampling probability
**Use:** Choose a corpus according to normalized weights.
**Status:** Dataset pipeline
$$
p_i = \frac{w_i}{\sum_{j=1}^{K}w_j}
$$
### 25.2. Expected sample count
**Use:** Estimate how many examples come from corpus i.
**Status:** Dataset planning
$$
\mathbb{E}[n_i] = Np_i
$$
### 25.3. Jaccard similarity
**Use:** Basic document/token-set overlap measure for deduplication.
**Status:** Data cleaning
$$
J(A,B) = \frac{|A\cap B|}{|A\cup B|}
$$
### 25.4. Hamming distance for byte sequences
**Use:** Useful for exact-near duplicate detection on normalized short sequences.
**Status:** Data cleaning
$$
d_H(x,y) = \sum_{t=1}^{T}\mathbf{1}\{x_t\ne y_t\}
$$
### 25.5. Contamination overlap rate
**Use:** Measure benchmark leakage candidates.
**Status:** Evaluation hygiene
$$
R_{\text{overlap}} = \frac{|D_{\text{train}}\cap D_{\text{eval}}|}{|D_{\text{eval}}|}
$$
## 26. Evaluation, Calibration, Reliability, and Confidence
### 26.1. Accuracy
**Use:** Generic exact-match accuracy.
**Status:** Evaluation
$$
\operatorname{Accuracy} = \frac{N_{\text{correct}}}{N_{\text{total}}}
$$
### 26.2. Precision
**Use:** Binary/multiclass diagnostic when false positives matter.
**Status:** Evaluation
$$
\operatorname{Precision} = \frac{TP}{TP+FP}
$$
### 26.3. Recall
**Use:** Binary/multiclass diagnostic when false negatives matter.
**Status:** Evaluation
$$
\operatorname{Recall} = \frac{TP}{TP+FN}
$$
### 26.4. F1 score
**Use:** Harmonic mean of precision and recall.
**Status:** Evaluation
$$
F1 = \frac{2PR}{P+R}
$$
### 26.5. Brier score
**Use:** Probability calibration metric for probabilistic predictions.
**Status:** Reliability
$$
\operatorname{Brier} = \frac{1}{N}\sum_{n=1}^{N}(p_n-y_n)^2
$$
### 26.6. Expected calibration error
**Use:** Bin-based difference between confidence and accuracy.
**Status:** Reliability
$$
\operatorname{ECE} = \sum_{m=1}^{M}\frac{|B_m|}{N}|\operatorname{acc}(B_m)-\operatorname{conf}(B_m)|
$$
### 26.7. Bootstrap standard error
**Use:** Estimate uncertainty from repeated resampling.
**Status:** Benchmark analysis
$$
\operatorname{SE}_{\text{boot}}(\hat{\theta}) = \operatorname{sd}(\hat{\theta}^{*(1)},\ldots,\hat{\theta}^{*(B)})
$$
For a bootstrap estimate above, the denominator is not a magical population constant: the standard error is the empirical standard deviation of the bootstrap statistic. This board keeps the expression explicit as a notation reminder rather than treating the bootstrap procedure as a single closed-form law.
### 26.8. Approximate 95% normal confidence interval
**Use:** Only appropriate when the approximation assumptions are reasonable.
**Status:** Benchmark reporting
$$
\hat{\mu} \pm 1.96\,\operatorname{SE}(\hat{\mu})
$$
## 27. Latent Recurrent Reasoning and Test-Time Compute
### 27.1. Recurrent latent refinement
**Use:** Weight-tied repeated latent compute without emitting reasoning tokens.
**Status:** Optional latent reasoning branch
$$
\mathbf{s}_k = R_\theta(\mathbf{e},\mathbf{s}_{k-1}), \qquad k=1,\ldots,K
$$
### 27.2. Base-preserving gated refinement
**Use:** Inject optional compute without forcing a new path at initialization.
**Status:** Project-specific safety pattern
$$
\mathbf{h}_{k+1} = \mathbf{h}_k + g_k\,F_\theta(\mathbf{h}_k,\mathbf{c})
$$
### 27.3. Adaptive halting criterion
**Use:** Optional stopping when latent state change becomes small.
**Status:** Optional research branch
$$
\operatorname{stop}(k) \iff \frac{\|\mathbf{h}_{k}-\mathbf{h}_{k-1}\|_2}{\max(\|\mathbf{h}_k\|_2,\eta)} < \tau
$$
### 27.4. Marginal value of an extra iteration
**Use:** Decide whether extra recurrent computation improves the task enough to justify latency.
**Status:** Research analysis
$$
\Delta Q_k = Q(K=k)-Q(K=k-1)
$$
### 27.5. Compute-normalized gain
**Use:** Compare latent reasoning against its compute overhead.
**Status:** Research analysis
$$
G_{\text{latent}} = \frac{\Delta Q}{\Delta T}
$$
The recurrent-depth latent reasoning work motivating this branch formalizes repeated application of a core recurrent block to a latent state and evaluates scaling inference compute; the project adapts the concept to its Mamba-based architecture rather than claiming architectural identity with the published transformer implementation.
## 28. Statistical Decision and Ablation Accounting
### 28.1. Absolute metric delta
**Use:** Simple effect size.
**Status:** Ablation reporting
$$
\Delta Q = Q_{\text{candidate}}-Q_{\text{baseline}}
$$
### 28.2. Relative metric delta
**Use:** Normalize an effect by baseline.
**Status:** Ablation reporting
$$
\Delta Q_{\%} = \frac{Q_{\text{candidate}}-Q_{\text{baseline}}}{|Q_{\text{baseline}}|}
$$
### 28.3. Cost-normalized improvement
**Use:** Normalize quality gain by added latency.
**Status:** Ablation reporting
$$
\operatorname{GainPerMs} = \frac{\Delta Q}{\Delta T_{\text{ms}}}
$$
### 28.4. Pareto dominance
**Use:** Candidate is preferable when it is no worse on all relevant axes and better on at least one.
**Status:** Model selection
$$
a \succ b \iff \forall j:\;m_j(a)\ge m_j(b)\;\land\;\exists k:\;m_k(a)>m_k(b)
$$
### 28.5. Weighted utility score
**Use:** Only use when weights are explicitly declared before reading the result.
**Status:** Decision analysis
$$
U = \sum_{j=1}^{K}w_jm_j, \qquad \sum_j w_j=1
$$
## 29. Release Integrity and Reproducibility Mathematics
### 29.1. SHA-256 digest identity
**Use:** Abstract identity function for content-addressed artifacts.
**Status:** Reproducibility
$$
d = \operatorname{SHA256}(\text{bytes})
$$
### 29.2. Determinism check
**Use:** Same input/config/seed should reproduce within a declared numerical tolerance.
**Status:** Reproducibility
$$
\operatorname{same}(y_1,y_2) \iff \operatorname{RelErr}(y_1,y_2)\le\tau
$$
### 29.3. Run identity tuple
**Use:** Define the minimal evidence identity logically, though the implementation stores the fields as structured metadata.
**Status:** Reproducibility
$$
R = (h_{\text{code}},h_{\text{config}},h_{\text{data}},h_{\text{checkpoint}},\text{hardware},\text{software},\text{seed})
$$
## 30. Formula-to-Module Map
```text
Byte/control contract              → 10.x
Embedding / norms / gates          → 11.x
Mamba / SSM                         → 12.x
Patch causal decoder                → 13.x
Softmax / CE / PPL                  → 14.x
MoE router / capacity / balance     → 15.x
MTP / SFT / preference / KD        → 16.x
Optimization / schedules            → 17.x
Initialization / numerical safety   → 18.x
RAG / retrieval evaluation          → 19.x
Tools / verifier / repair / search  → 20.x
Quantization                        → 21.x
Parameter scaling                   → 22.x
Performance / memory                → 23.x
Scaling laws / data budgets         → 24.x
Data mixture / cleaning             → 25.x
Evaluation / calibration            → 26.x
Latent reasoning                    → 27.x
Ablation decisions                  → 28.x
Release reproducibility             → 29.x
```
## 31. Verified Source Ledger for the Formula Board
These are the primary/reference sources checked while constructing the board. Equations that are project-specific or generic mathematical identities are explicitly marked as such above rather than being attributed to a paper that did not define them.
- Mamba-2: Tri Dao and Albert Gu, “Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality,” ICML 2024 / arXiv:2405.21060 — sections on SSM recurrence, Mamba-2 architecture, parallel projections, gating, head structure, and systems. https://arxiv.org/abs/2405.21060
- Official Mamba repository and implementation — Mamba-2 module, Mamba configuration, current installation/backend requirements, and reference model construction. https://github.com/state-spaces/mamba
- MegaByte: Lample et al., “MegaByte: Predicting Million-byte Sequences with Multiscale Transformers,” arXiv:2305.07185 — hierarchical byte-level sequence modeling precedent. https://arxiv.org/abs/2305.07185
- Switch Transformers: Fedus, Zoph, Shazeer, “Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity,” arXiv:2101.03961 — top-1 routing, capacity considerations, and auxiliary load-balancing loss. https://arxiv.org/abs/2101.03961
- AWQ: Lin et al., “AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration,” arXiv:2306.00978 — activation-aware low-bit weight quantization and salient-channel scaling. https://arxiv.org/abs/2306.00978
- DeepSeek-V3 Technical Report — multi-token prediction, MoE balancing, and large-scale post-training references. https://arxiv.org/abs/2412.19437
- Quiet-STaR — Zelikman et al., “Language Models Can Teach Themselves to Think Before Speaking,” arXiv:2403.09629 — internal predictive thought branch precedent. https://arxiv.org/abs/2403.09629
- Scaling by Thinking in Continuous Space / latent recurrent depth — Geiping et al., arXiv:2502.05171 — repeated latent recurrent computation and inference-time compute scaling. https://arxiv.org/abs/2502.05171
- AdamW: Loshchilov and Hutter, “Decoupled Weight Decay Regularization,” arXiv:1711.05101 — decoupled weight decay. https://arxiv.org/abs/1711.05101
- Layer Normalization: Ba, Kiros, Hinton, arXiv:1607.06450 — normalization reference. https://arxiv.org/abs/1607.06450
- Attention Is All You Need: Vaswani et al., arXiv:1706.03762 — softmax attention and standard transformer mathematical reference where relevant to comparisons. https://arxiv.org/abs/1706.03762
- Scaling Laws for Neural Language Models: Kaplan et al., arXiv:2001.08361 — empirical power-law framing. https://arxiv.org/abs/2001.08361
- Training Compute-Optimal Large Language Models (Chinchilla): Hoffmann et al., arXiv:2203.15556 — empirical model/data allocation result. https://arxiv.org/abs/2203.15556
- InstructGPT / RLHF: Ouyang et al., arXiv:2203.02155 — SFT + preference/RL alignment lineage. https://arxiv.org/abs/2203.02155
## 32. Source-Specific Formula Verification Notes
### 32.1 Mamba-2
The Mamba-2 paper explicitly gives the standard discrete SSM recurrence, introduces the Mamba-2 block with parallel production of A/B/C/X-related quantities, depthwise causal convolution, SSM mixing, SiLU gating, and output projection. The official repository currently exposes Mamba-2 and Mamba-3 code paths. The roadmap therefore pins Mamba-2 as the reference backbone and treats Mamba-3 as an experiment, not as an invisible dependency. citeturn830754view0turn740337view0turn740337view2
### 32.2 Switch-style MoE
The Switch Transformer source gives the load fractions f_i, probability fractions P_i, and auxiliary loss α E Σ_i f_i P_i. The roadmap uses these exact symbols and does not substitute a different balancing objective while calling it “Switch loss.” citeturn740337view3
### 32.3 AWQ
AWQ is used only as a low-bit quantization candidate. Its paper motivates activation-aware scaling and reports INT4 experiments, but the Edge-400 roadmap requires direct validation on the recurrent Mamba checkpoint because recurrent numerical dynamics can make the accuracy/stability trade-off architecture-specific. citeturn740337view4
### 32.4 MTP
DeepSeek-V3 uses a multi-token prediction objective and describes a sequential prediction chain across additional depths. In this project MTP remains an ablation until it beats the declared baseline on both model quality and useful deployment metrics. citeturn740337view6
### 32.5 Latent recurrent reasoning
The recurrent-depth paper explicitly defines a latent state recurrence s_i = R(e, s_{i-1}) and uses repeated application of a recurrent core to increase inference-time computation. Quiet-STaR provides a related precedent for internal predictive thoughts. These papers justify investigation, not a guarantee of reasoning improvement. citeturn740337view5turn710308view6
## 33. Formula Implementation Rules — Preventing Markdown/Math Corruption
- Every display equation in this file is isolated between its own double-dollar opening and closing delimiters.
- Do not place a display equation inside a Markdown table cell.
- Do not break a single LaTeX command across ordinary Markdown line boundaries.
- Use ASCII punctuation around equations where possible; reserve Unicode characters for prose when they do not affect parsing.
- When a formula becomes long, keep it in one display block rather than wrapping it into multiple independent equations that look incomplete.
- Do not mix MathJax-only syntax with raw source syntax from a particular notebook renderer; this file uses standard display-mode LaTeX.
- Every equation added later must pass an automated delimiter-balance check: number of double-dollar delimiters must be even.
- A second check should ensure no double-dollar delimiter appears inside fenced code blocks unless the code block is intentionally demonstrating Markdown.
- For source attribution, cite the paper after the prose definition; do not turn the equation itself into a citation-dependent fragment.
- Project-derived equations must be labelled derived/project-specific so future maintainers do not mistake them for published reference formulas.
## 34. Automated QA Checklist for This Document
```text
1. UTF-8 decode succeeds.
2. double-dollar delimiter count is even.
3. No equation fence is opened and left unclosed.
4. All H2/H3 headings are non-empty.
5. No accidental email/document metadata fragments exist inside formulas.
6. Source URLs are complete and syntactically well formed.
7. Every formula has Use + Status + Equation fields.
8. Reference formulas are separated from project-specific formulas.
9. Every W01..W52 entry has an exact calendar range.
10. The 52-week roadmap contains W01..W52 exactly once.
11. The final file path and checksum are recorded externally in the release ledger.
```
## 35. Year-End Release Sequence
```text
RC-01  Architecture freeze
RC-02  Dataset freeze
RC-03  Training recipe freeze
RC-04  Final checkpoint hash
RC-05  Quantized checkpoint hash
RC-06  Runtime/backend equivalence
RC-07  Causality + state tests
RC-08  Tool/sandbox security tests
RC-09  RAG provenance tests
RC-10  Full benchmark suite
RC-11  Memory/latency/thermal suite
RC-12  Documentation + limitation audit
RC-13  Reproducibility bundle
RC-14  Final tag + archive
```
## 36. Cold Truth About the Roadmap
This plan is intentionally aggressive. The one-year schedule is a disciplined research program, not a guarantee that a 400M model will achieve a particular quality level, fit a particular device, or outperform a larger baseline. The architecture should be considered successful only where the evidence shows it is competitive under the chosen constraints.
The hardest risks are not writing the first Mamba block. They are: dataset quality, training compute, recurrent numerical stability, router/expert specialization without harmful collapse, end-to-end byte generation quality, tool/sandbox engineering, quantized recurrent stability, and the gap between a technically elegant prototype and a useful edge product.
Accordingly, the roadmap places the cheap-to-falsify experiments first. The most expensive 400M training is delayed until the architecture, data, parameter accounting, backend legality, and test infrastructure are already trustworthy.
## 37. Final Project Principle
**Build in the order of dependency, not in the order of excitement. Measure every claim. Keep the trusted baseline alive. Let experiments fail cheaply. Promote only what survives evidence.**
## 38. Canonical Web References
```text
Mamba-2 paper: https://arxiv.org/abs/2405.21060
Official Mamba repository: https://github.com/state-spaces/mamba
MegaByte: https://arxiv.org/abs/2305.07185
Switch Transformers: https://arxiv.org/abs/2101.03961
AWQ: https://arxiv.org/abs/2306.00978
DeepSeek-V3: https://arxiv.org/abs/2412.19437
Quiet-STaR: https://arxiv.org/abs/2403.09629
Latent recurrent depth: https://arxiv.org/abs/2502.05171
AdamW: https://arxiv.org/abs/1711.05101
LayerNorm: https://arxiv.org/abs/1607.06450
Scaling Laws: https://arxiv.org/abs/2001.08361
Chinchilla: https://arxiv.org/abs/2203.15556
InstructGPT: https://arxiv.org/abs/2203.02155
```
End of Unified Edge-400 Master Build Roadmap + Master Formula Board v1.0 FINAL
