# AC-006 — Verified RSI Framework

Version 1.0 • 2026-09-22 • **PROPOSED / RESEARCH CONTRACT — NOT IMPLEMENTED OR ENABLED**

## Purpose and relationship to the baseline

This proposal supplements the immutable [Bible](../../Unified_Edge400_Master_Bible_v3.0_FINAL.md) §§54, 78–79, 97, 103–105, 138, 141–146, 157 and the [Roadmap](../../Unified_Edge400_Master_Roadmap_and_Formula_Board_v1.0_FINAL.md) §§5, 7–8, 28–29. They already require host isolation, reproducibility, ablations, promotion and rollback. The new contract is explicit parent/child succession, separated proposal and promotion authority, independent evaluation, bounded multi-generation operation, and multi-fidelity experiments. It does not claim to invent sandboxing or prove that RSI works.

The accepted dense 1,929,579-parameter model, Mamba-2 semantic-reference pin, corpus and Stage-A evidence remain unchanged. Stage-A readiness for continuation review is not a frozen Gen-0 designation or authorization to launch RSI-0. The [addendum](../../Unified_Edge400_Future_Capabilities_Addendum_v1.0.md) integrates this proposal with AC-007 and AC-008.

## Lifecycle and authority

**PROPOSE != PROMOTE**

```text
immutable promoted parent Mn
  -> proposal / experiment specification
  -> host-authorized sandboxed child Mn+1-candidate
  -> bounded training / modification / evaluation
  -> immutable independent evaluator
       -> REJECT or HOLD: preserve parent; isolate child
       -> external PROMOTE: publish child identity as next parent
            -> next separately budgeted generation, if authorized
```

The candidate label is provisional; rejected children do not consume a promoted generation identity. Multiple siblings have unique IDs and the same explicit parent. A proxy is a separate experiment with its own lineage, not automatically the next production generation. The host admits the experiment, validates its allowed change surface, and launches only the declared operations. The candidate supplies proposals and artifacts, never promotion credentials.

The evaluator is outside the candidate process and writable scope. The host freezes evaluator code, manifests, hidden sets and decision policy before observing candidate results. Authorized evaluator maintenance creates a new independently reviewed version and requires comparable parent/candidate re-evaluation; it is not an in-place candidate edit. Hashes provide identity, not isolation: filesystem/process permissions, tool permissions, network policy and credential separation require future adversarial proof.

## Mandatory invariants

1. The parent checkpoint is immutable and hash-addressed.
2. A candidate cannot overwrite its parent checkpoint, source release or production binding.
3. A candidate cannot modify the evaluator or its measurement implementation.
4. A candidate cannot modify or directly access hidden evaluation data.
5. A candidate cannot modify promotion thresholds or the resolved promotion policy.
6. A candidate cannot silently increase compute, resource or experiment allowances.
7. Every child records one explicit parent; only the externally frozen Gen-0 root has a null parent.
8. Every generation has hashes/manifests sufficient for reproduction, including seeds, environment and declared numerical tolerances; unavailable evidence prevents promotion.
9. Failed or unpromoted children remain isolated from production.
10. Every promoted ancestor remains recoverable with its code, configuration, checkpoint and dependencies. Revoked or insecure ancestors may be restored only into quarantine for analysis, not silently redeployed.
11. Promotion requires independently measured improvement, not self-reported improvement.
12. Predeclared regression gates protect previously accepted capabilities.
13. Security and integrity failures veto capability gains.
14. A candidate cannot mark itself promoted, write the trusted decision record or change the active-parent pointer.
15. External promotion authority is mandatory at every generation, including RSI-4.

The external authority is a named human or separately authorized trusted host service whose policy and credentials the candidate cannot modify. In this project, implementation and activation still require separate human authorization. Promotion is an auditable pointer/publication operation over immutable artifacts, not overwriting the parent. It must verify the expected parent identity to prevent stale or concurrent promotions. A failed publication leaves the accepted parent active.

## GenerationManifest v1 — conceptual fields

This table specifies a future contract; no production schema or loader is introduced. Required field names also appear in [the specification JSON](../future_capabilities_contract_v1.json).

| Field | Meaning / constraint |
|---|---|
| schema_version | Explicit supported manifest version |
| generation_id | Unique child identity, never reused |
| parent_generation_id | Exact parent ID; null only for externally frozen Gen-0 |
| parent_checkpoint_sha256 | Immutable parent payload digest; null only for Gen-0 |
| proposal_id | Hash-bound hypothesis, allowed changes and acceptance plan |
| source_commit | Exact candidate source commit |
| model_config_sha256 | Resolved architecture identity |
| training_config_sha256 | Resolved recipe identity, or explicit not-applicable reason for an inference-only change |
| dataset_manifest_sha256 | Input data identity, or explicit not-applicable reason; no unversioned directory |
| evaluator_manifest_sha256 | Trusted evaluator version, suite, protocol and policy binding |
| hidden_evaluator_version | Opaque host identity; not candidate-readable questions or answers |
| compute_budget | Preauthorized device time, wall time, memory, storage and other resource ceilings; units and scope |
| experiment_budget | Trial, retry, concurrent-child and generation limits; cumulative accounting includes failures |
| capability_metrics | Trusted artifact references with scope, units and uncertainty |
| regression_metrics | Parent comparison and retained-capability gates |
| efficiency_metrics | Latency, throughput, memory and measured resource cost |
| security_metrics | Isolation, integrity and adversarial results; failures cannot be averaged away |
| promotion_decision | External PROMOTE / HOLD / ROLLBACK / REJECT record and reason; pending before decision |
| promotion_authority | Authenticated external decision identity; absent while pending |
| rollback_parent | Recoverable promoted ancestor plus artifacts and compatibility contract |
| creation_timestamp | Host-issued UTC creation time |

A finalized manifest additionally binds candidate checkpoint/output hashes, environment/dependency manifests, seeds/RNG and cursor where applicable, actual budget consumption, evidence bundle hashes, promotion-policy identity and decision timestamp. Pending fields must be explicitly pending, never fabricated success. Host-owned append-only revisions/finalization preserve earlier records and distinguish proposer assertions from trusted measured fields. Hidden manifests stay in host-controlled storage; public provenance may use opaque versions to avoid disclosing evaluation material.

## RSI maturity and scale associations

These are non-binding research ranges, not parameter-count capability guarantees. A system may remain at any lower level indefinitely. Moving levels requires an independent capability/isolation/budget/rollback review, not just a larger checkpoint.

| Level | Research association | Permitted proposal/experiment scope, after authorization |
|---|---|---|
| RSI-0 | After validated/frozen Gen-0, including a possible 2M foundation | External successor-development harness; stronger external engineering agent may propose initial experiments; parent stays passive; evaluation and promotion infrastructure established first |
| RSI-1 | Approximately 20M–40M | Self-reflection, failure analysis, policy/prompt/config repair proposals; no unrestricted source mutation |
| RSI-2 | Approximately 40M–80M | Bounded training recipes, retrieval settings, verifier rules, specialization recipes, selected code/config proposals; host execution and external promotion |
| RSI-3 | Approximately 80M–200M | Development-agent design and sandboxed execution of bounded experiments, including lower-scale proxies; no production overwrite authority |
| RSI-4 | Approximately 200M–400M | A promoted generation may participate in producing its successor; immutable external evaluation, promotion and resource controls remain mandatory |

Verifier-rule proposals at RSI-2 apply to the candidate's task verifier, not the trusted promotion evaluator. Proposed recipe changes cannot rewrite the current experiment budget or decision policy. No level grants unrestricted host access or authority to change its own maturity level.

## Multi-fidelity experimentation

Use the accepted planning ladder **2M → 20M → 40M → 80M → 200M → 400M**, without rewriting the historical FINAL ladder. A 400M parent proposal may first receive an authorized 2M/20M proxy trial, followed by 40M/80M replication and then a target-scale test. If the effect fails at a proxy, reject or hold the proposal under the predeclared screening policy; any further investigation needs a new bounded experiment.

A proxy result is evidence, not proof of scaling. Record differences in data, optimizer, architecture, objective and resource regime so apparent transfer is not attributed to scale alone. Require replication against the parent at the actual target scale before final promotion. Cheap proxies neither waive target gates nor imply checkpoint tensor compatibility (Bible §141). Finite total experiment/generation budgets and a host stop switch bound the whole chain, including failed trials and retries.

## Independent evaluator and anti-gaming

Separate evaluator families cover: general capability; language-modeling quality; code; mathematics; long-context retention; retrieval/RAG; tool use; verifier/repair success; latency; throughput; memory; energy/resource cost where measurable; security; robustness; regression; contamination; benchmark integrity. Enabled components require applicable tests; disabled components receive an explicit not-applicable scope and no capability claim. Missing required evidence yields HOLD, not a fabricated pass.

Candidate-visible development evaluation supports debugging. Candidate-hidden evaluation is host-only and must not enter prompts, training data, retrieval indexes, tool outputs or candidate logs. Restrict query counts and feedback detail; separate evaluation workspaces/credentials and audit access. Benchmark contamination checks, held-out source-unit separation, evaluator versioning and adversarial canaries are future evidence requirements, not implemented protections here.

**Evaluator gaming and benchmark overfitting are central RSI risks.** Hidden access controls do not guarantee immunity, and repeated aggregate feedback can leak information. Promotion should use agreement across independent metrics and evaluator families wherever practical, with predeclared uncertainty/repetition rules and no post-result threshold changes. Capability gains cannot compensate for security/integrity failure. The addendum's multi-metric policy defines PROMOTE, HOLD, ROLLBACK and REJECT without universal weights.

## Impact, evidence and rollback

Current parameter, memory, training and inference impact: **none**. Future host controllers add measured storage/runtime/experiment cost; neural modifications require separate architectural and parameter audits. No production RSI code, launcher, dataset, evaluator or checkpoint is created now.

Future acceptance requires adversarial attempts to mutate parent/evaluator/hidden sets/thresholds/budgets, self-promotion denial, budget exhaustion across retries, concurrent-publication safety, lineage replay and ancestor recovery, target-scale replication and retained-capability tests. A documentation integrity test cannot satisfy these runtime gates.

Rejected alternatives: live in-place self-overwrite; a single self-reported score; candidate-controlled evaluation; unlimited generations; promotion on a cheap proxy alone. If the future harness fails, stop candidate execution, quarantine its artifacts and retain the promoted parent. Revert or supersede this proposal as documentation without changing either FINAL file. Decision remains **pending human review**; no assurance that RSI will become useful at any scale.
