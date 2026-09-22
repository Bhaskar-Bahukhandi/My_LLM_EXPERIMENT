# Unified Edge-400 Future Capabilities Addendum v1.0

2026-09-22 • **PROPOSED — READY FOR HUMAN REVIEW, NOT IMPLEMENTED OR ACTIVATED**

## 1. Scope

This specification-only addendum formalizes future host-controlled successor development, adaptive effort, and context/distribution promotion contracts. Its detailed proposals are [AC-006](docs/architecture_changes/AC-006_Verified_RSI_Framework.md), [AC-007](docs/architecture_changes/AC-007_Adaptive_Reasoning_Effort.md) and [AC-008](docs/architecture_changes/AC-008_Context_and_Distribution_Promotion_Gates.md). The compact [JSON contract](docs/future_capabilities_contract_v1.json) is a specification artifact, not runtime configuration. Nothing is automatically enabled by its presence.

## 2. Relationship to FINAL Bible/Roadmap

The [Bible v3.0 FINAL](Unified_Edge400_Master_Bible_v3.0_FINAL.md) and [Roadmap + Formula Board v1.0 FINAL](Unified_Edge400_Master_Roadmap_and_Formula_Board_v1.0_FINAL.md) remain immutable historical specifications. This addendum supplements them; it does not rewrite their equations, chronology or implementation status. Accepted AC-001/002/004/005 retain complete-patch floor geometry, headdim-aware state accounting, legality/memory-before-ranking and unique parameter/buffer/alias/serialization distinctions. [AC-003](docs/architecture_changes/AC-003.md) still defers causal routing resolution; no same-chunk future leakage is authorized.

The accepted planning ladder in [the implementation plan](docs/implementation_plan.md) is 2M → 20M → 40M → 80M → 200M → 400M; older FINAL scale labels stay untouched. The semantic Mamba-2 v2.2.6 / d7b1ceb3c367ec9022925e812f507bcf706937c6 pin remains an engineering choice, not a Bible version requirement. A conflict between a future proposal and accepted behavior must be explicitly reviewed before implementation.

| Existing feature | Authoritative material | New material here |
|---|---|---|
| Byte-native symbols and patch hierarchy | Bible §§6–8, 17–18, 136; Roadmap W05–W08, §§10, 13 | No replacement |
| Mamba trunk and state | Bible §§9, 44, 76, 84–87; Roadmap W09–W10, §12 | Context certification adds evidence scope |
| MoE and shared/expert boundaries | Bible §§11–13, 127–128; Roadmap W17–W20, §15; AC-003 | No router implementation or causality exemption |
| RAG and provenance | Bible §§20–21, 40, 92, 104; Roadmap W29–W32, §19 | Effort envelopes compose existing host budgets |
| Tools, sandbox, verifier, bounded repair | Bible §§22–30, 54, 88–89, 103–105; Roadmap W33–W36, §20 | Contradiction-aware correction and explicit authority boundaries |
| Specialization profiles | Bible §164; Roadmap W38–W39 | Candidate recipe proposals, not instant specialization |
| Quantization | Bible §§45–48; Roadmap W41–W42, §21 | No quantization changes |
| Latent refinement and K=1/2/4/8 research | Bible §165; Roadmap W45–W46, §27 | User/system effort labels, not new recurrence equations |
| Promotion and rollback | Bible §§97, 120, 143, 146, 157, 169; Roadmap §8, §§28–29 | Generation lineage, independent authority, context/distribution detail |

These are project-contract additions, not claims of scientific novelty. Local documents are the sources for this reconciliation; no current external library or paper claims are newly asserted.

## 3. Non-regression principle

Nothing in the future-capabilities layer may weaken or silently mutate the accepted base path. Every experimental subsystem must be independently disableable. Off mode preserves the accepted model/system within its declared compatibility contract: bit-identical on the deterministic reference or an explicitly predeclared numerical tolerance on a named profile. Failure of an experimental subsystem must not corrupt the promoted parent. Runtime proof remains a future gate, not something documentation establishes.

The current 1,929,579-parameter 2M model and [Stage-A evidence](reports/full_training_2m_stage_a.md) remain authoritative. Readiness for continuation review does not mean final model promotion, useful language quality, Gen-0 freeze or RSI activation.

## 4. Base model vs host-controller boundary

| Neural model | Host/system layer |
|---|---|
| Byte hierarchy, recurrent trunk, local decoder, output head | RAG index, provenance handling, tool broker |
| Future expert blocks, only after independent approval | Sandbox, verifier, repair controller, effort controller |
| Optional experimentally validated latent branch | RSI experiment controller, candidate training launcher, independent evaluator, promotion authority, rollback manager |

The neural model proposes structured outputs. The host owns permissions, persistent lineage, budgets, trusted measurements and promotion. A candidate task verifier may be experimental, but it cannot become its own trusted promotion evaluator. Enabling an effort level never grants filesystem, network, evaluator or production-write authority.

## 5. Adaptive reasoning effort

AC-007 defines Auto, Low, Medium, High and Ultra. Labels resolve per model generation and supported profile to a hash-bound bounded compute policy. There is no permanent Low=K1, Medium=K2, High=K4, Ultra=K8 mapping. Unsupported functionality is explicitly unavailable, not silently simulated. Increased effort neither guarantees quality nor requires exposing private hidden reasoning or chain-of-thought.

## 6. Latent-compute budget integration

Retain Bible §165's optional weight-tied, residual, task-level latent branch and fixed-K study before adaptive halting. A host effort policy may select only experimentally admitted iteration limits and decision points. State convergence is a resource signal, not proof of correctness. Disabled latent computation retains the base path; future integration must prove output/state/RNG compatibility and measure latency/memory, including temporary buffers.

## 7. Retrieval/tool/verifier/repair budgets

Every effort label declares finite ceilings for latent iterations, retrieval count/depth/bytes, reranking, verifier passes, tool calls, repair attempts, candidate count, optional self-consistency, end-to-end wall time, memory and generated bytes. Token ceilings require a named unit conversion; raw-byte operation remains authoritative. Nested calls, failed attempts and concurrent candidates share the parent budget and cannot reset it.

Auto may use task class, complexity, confidence, uncertainty, contradictions, failures and remaining resources. It selects actions inside the approved envelope; it is not a separate model and cannot expand its own budget. Audit records explain observable triggers and consumed resources, not private reasoning. Permissions and hard deadlines stay host-enforced.

## 8. Contradiction-aware correction

The host compares a draft with provenance-bearing retrieval/tool/execution/verifier evidence. A material contradiction triggers a bounded revision and re-verification; otherwise the answer may proceed within its verified scope. The controller can terminate explicitly unresolved when evidence conflicts or budget expires. It must not fabricate evidence, treat retrieved instructions as policy, or imply an unrun check passed.

Self-correction revises an answer; self-training updates weights; self-modification changes code/config; RSI is controlled successor lineage and promotion. These are distinct operations and authorizations. Contradiction-aware correction alone is not RSI.

## 9. Verified RSI framework

**PROPOSE != PROMOTE.** A frozen parent Mn leads to a proposal, a host-authorized sandboxed child Mn+1-candidate, bounded modification/training/evaluation, and an immutable independent evaluator. REJECT/HOLD retains the parent. Only external PROMOTE publishes the child as next parent. A further generation requires remaining explicit authority and budget. There is no live self-overwrite or unbounded succession loop.

## 10. Parent/child generation lineage

AC-006 defines GenerationManifest v1: parent/checkpoint, proposal/source/config/data/evaluator hashes, hidden evaluator version, compute/experiment budgets, capability/regression/efficiency/security evidence, decision/authority, rollback parent and timestamp. It also requires candidate artifact hashes and environment/seed/accounting evidence. Pending fields are not success. Every child has an explicit parent; externally frozen Gen-0 alone has no parent. Host-owned immutable revisions preserve decisions and hashes without exposing hidden evaluator content.

## 11. Candidate sandbox

Candidate writable scope is restricted to its own declared artifacts. Parent, production, evaluator, hidden sets, thresholds, permissions and resource controls are outside that scope. The host checks quotas across retries, siblings and generations. Failed children remain quarantined. Candidate source changes must be separately allowlisted and reviewed; proposals cannot quietly edit production. Security/isolation must be tested independently of model cooperation.

## 12. Independent evaluator

Families: general capability, language modeling, code, mathematics, long-context retention, retrieval/RAG, tool use, verifier/repair, latency, throughput, memory, energy/resource cost where measurable, security, robustness, regression, contamination and benchmark integrity. Applicable suites are frozen with exact checkpoint/runtime/data/protocol identities and declared uncertainty. Disabled features are not applicable with a reason; missing required evidence prevents promotion.

## 13. Anti-gaming and hidden evaluation

Visible development tests support debugging; hidden evaluation stays outside candidate prompts, data, RAG, tools and logs. Limit feedback and query budgets; audit access and contamination; version evaluator changes independently. Hashes do not replace access controls. Evaluator gaming and benchmark overfitting remain central risks, including leakage from repeated aggregate feedback. No perfect anti-gaming guarantee is claimed. Require agreement across independent metrics rather than one scalar reward wherever practical.

## 14. Multi-fidelity proxy experimentation

A high-cost parent need not test every idea at full scale. Example planning flow: 400M proposal → authorized 2M/20M proxy → reject/hold if unsupported → 40M/80M replication → target-scale test. Record changed conditions and selection effects. A proxy result is evidence, not proof that an effect scales; target-scale replication is mandatory before final promotion. This neither launches any proxy now nor assumes weight transfer across architectures.

## 15. Promotion / hold / rollback

Resolve a per-generation policy before results, with seven categories:

| Category | Required decision scope |
|---|---|
| CAPABILITY | Task success and retained capabilities |
| QUALITY | Held-out LM/continuation and distribution behavior |
| CONTEXT | Task-specific retention certification |
| EFFICIENCY | Latency, throughput, memory and measurable resource cost |
| ROBUSTNESS | Perturbations, failures and numerical stability |
| SAFETY | Security, integrity, broker/sandbox isolation and permitted use |
| REPRODUCIBILITY | Immutable lineage, data/protocol identities and replay/rollback |

No universal weights or unverified thresholds are assigned. Security/integrity, causality, legality and critical regression gates are hard constraints before tradeoff ranking. Record metrics that disagree and reasons for the decision; lower training loss alone never suffices.

PROMOTE requires independently measured benefit, all applicable gates, external authority and target-scale evidence. HOLD preserves the parent when evidence is incomplete, underpowered or tradeoffs unresolved. REJECT isolates an unpromoted candidate that fails. ROLLBACK withdraws a promoted candidate after a discovered failure and restores a reviewed safe ancestor/system bundle. Retain all promoted ancestors for recovery; an insecure ancestor can be quarantined rather than redeployed. Evaluator or policy changes require separate review and a comparable baseline, not moving thresholds after seeing results.

## 16. Effective-context certification

AC-008 separates mechanical support, actual trained windows and empirically validated effective context. Reports record mechanically_supported, training_window_bytes, validation_window_bytes, maximum_evaluated_context_bytes, effective_context_evidence, retention_tests and VERIFIED/PARTIAL/UNVERIFIED scope. Length claims above empirical support must say mechanical/theoretical only.

Stage A uses 32-byte training and validation windows, about four completed 8-byte payload patches; BOS is separate. Its 500,000 validation bytes are a dataset size, not conditioning length. No 4K/32K/1M effective-context claim follows. Future tests cover positional loss, distance retention, copy/retrieval, delayed dependencies, code references, structured documents, state-reset sensitivity, drift, throughput, memory and stability. No benchmark is implemented or run here.

## 17. Distribution-quality gates

Stage-A NLL improved while greedy generation often produced spaces. AC-008 therefore requires raw-byte entropy, top-1/top-2 probabilities and margin, space probability/argmax frequency, confidence concentration, diversity, run lengths, held-out/domain continuations and separate control-symbol diagnostics. Conditional byte probabilities and original symbol/control mass must be explicitly distinguished.

Space may win by a narrow margin without owning most probability mass. Argmax samples alone do not prove collapse; probability measurements are not yet available in the accepted report. Do not retrofit that claim or change sampling to conceal degeneration. NLL and generation/distribution evidence jointly inform future promotion.

## 18. Scale activation schedule

| Non-binding range | Possible research, only after separate approval and evidence |
|---|---|
| 2M | Foundation, evaluation harness, context certification; RSI-0 infrastructure only after validated/frozen Gen-0 |
| 20M–40M | Improved capability baseline, effort-controller research, RSI-1 reflection/failure-analysis proposals |
| 40M–80M | Bounded recipe/retrieval/verifier/specialization/code/config proposals, RSI-2; MoE/context work independently approved |
| 80M–200M | Development-agent bounded experiments and lower-scale descendants/proxies; RSI-3 candidates |
| 200M–400M | Recursive successor experiments may be considered; RSI-4 only if all prior gates pass |
| 400M | Integrated research candidate; no assumption that RSI or all optional subsystems are enabled |

These associations guarantee nothing. A system may remain below any RSI level indefinitely. A stronger external engineering agent can initially propose RSI-0 experiments while the frozen parent remains passive. No maturity level gives production overwrite authority. Calendar/scale progression never waives evidence.

## 19. Safety/resource ceilings

External controllers enforce finite request, experiment and generation budgets, including wall time, device/host memory, storage, calls, retries and concurrency. The candidate cannot change evaluators, hidden data, thresholds, ceilings or promotion records. Security/integrity veto capability gains. Cancellation and failure preserve the parent. No platform, cloud provider, Kaggle, RTX/T4 infrastructure or future accelerator is mandatory; platform migration is a separate engineering choice.

## 20. Open research questions

- Does proxy-scale improvement transfer to target scales?
- Can this model meaningfully propose architectural improvements?
- How much evaluator information can be exposed without harmful overfitting?
- How can evaluator gaming and cumulative hidden-set leakage be detected?
- How should Auto select effort budgets and handle unreliable confidence signals?
- Does latent recurrent depth earn its latency/memory cost?
- What is useful effective context for a recurrent model across different tasks?
- How should capability-versus-efficiency tradeoffs be resolved?
- At what scale, if any, does model-generated research become useful?
- Can bounded RSI remain stable over repeated generations?
- Does Mamba-2 remain the best trunk at future scales?

These remain unanswered. Mamba-3 is only a possible controlled architecture ablation under the existing Bible branch, never an automatic replacement.

## 21. Explicit non-goals

No training, model code/weights, data, checkpoints, environments, 2M continuation, 20M work, RSI implementation, effort implementation or new evaluation runs in this tranche. No mandatory exposed chain-of-thought, fixed label-to-K mapping, unrestricted self-modification, infinite repair/RSI loop, guarantee of RSI success, claim of revolutionary architecture or guarantee that 400M is sufficient. Documentation is not implementation or activation authorization.

## 22. Required future evidence

Before implementation promotion: reviewed interface/threat/budget contracts; explicit feature disable/off compatibility; parent/evaluator/hidden-set tampering denial; trusted external promotion; lineage reproduction and rollback; finite nested budgets; controlled fixed-effort vs Auto trials; contradiction handling and unresolved exits; target-scale replication; multi-metric regression/security results; context certification and distribution diagnostics with uncertainty. Every result needs checkpoint/config/data/runtime/protocol hashes and measured cost. The current scope validates document integrity only. The next model/training action requires separate authorization.
