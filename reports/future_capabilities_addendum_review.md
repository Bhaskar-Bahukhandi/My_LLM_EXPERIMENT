# Future-capabilities specification review

**READY FOR HUMAN REVIEW. DOCUMENTATION != IMPLEMENTATION.** This closeout completes the interrupted validation of the existing published specification; it does not recreate or activate it.

Previous HEAD and existing specification commit: `dabe922ecd09762ccbbbe3e11b484ca8c4177024` (`Changes`). Published history and its subject remain unchanged. The finalization is a separate commit.

## Existing features, not duplicated

| Already in FINAL Bible/Roadmap | Cross-reference |
|---|---|
| byte-native architecture, patch hierarchy | Bible sections 6-8, 17-18, 136; Roadmap W05-W08, sections 10, 13 |
| Mamba, MoE planning | Bible sections 9, 11-13, 44, 76, 127-128; Roadmap W09-W10, W17-W20, sections 12, 15; AC-003 remains deferred |
| RAG, provenance | Bible sections 20-21, 40, 92, 104; Roadmap W29-W32, section 19 |
| tools, sandbox, verifier, bounded repair | Bible sections 22-30, 54, 88-89, 103-105; Roadmap W33-W36, section 20 |
| specialization profiles, quantization | Bible sections 164, 45-48; Roadmap W38-W39, W41-W42, section 21 |
| optional latent recurrent reasoning, K=1/2/4/8 research | Bible section 165; Roadmap W45-W46, section 27 |
| promotion/rollback concepts | Bible sections 97, 120, 143, 146, 157, 169; Roadmap section 8, sections 28-29 |

## Newly formalized contracts

| New specification | Purpose |
|---|---|
| verified RSI lineage, PROPOSE != PROMOTE, RSI-0..RSI-4 maturity model, immutable parent/child generations | Make successor identity, authority, bounded succession and recoverable parent preservation explicit. Scale does not guarantee maturity. |
| independent promotion evaluator, candidate-hidden evaluation, anti-gaming contract | Separate candidate optimization from trusted measurement and promotion; evaluator gaming remains a central unresolved risk. |
| multi-fidelity RSI experimentation | Screen ideas cheaply while requiring actual target-scale replication before promotion. |
| Auto/Low/Medium/High/Ultra effort contract | Compose finite host compute envelopes per generation without permanent label-to-K mapping or private reasoning disclosure. |
| contradiction-aware correction | Require provenance-bearing evidence, bounded revision and explicit unresolved termination; distinguish correction from training/modification/RSI. |
| mechanical/trained/effective context certification | Prevent streaming capacity or aggregate corpus size from being presented as demonstrated retention; preserve the 32-byte Stage-A limitation. |
| distribution-collapse diagnostics, expanded multi-metric promotion gate | Lower NLL and whitespace argmax do not establish useful generation or probability collapse; examine probabilities and hard regression/security gates separately. |

All new feature states remain **PROPOSED_NOT_IMPLEMENTED_NOT_ENABLED**. No fixed effort-to-K mapping, universal promotion weights, scale-based RSI guarantee or automatic Gen-0 designation is introduced. Security/integrity override capability gains. Parent/evaluator/hidden-data/threshold/budget mutation and self-promotion remain prohibited. Infinite loops and mandatory private chain-of-thought disclosure are prohibited.

## Content audit against the directive

| Area | Result and inspected evidence |
|---|---|
| A_verified_rsi | PASS: AC-006 lifecycle, 15 invariants, maturity table and anti-gaming sections; addendum sections 9-13, 18-21 |
| B_multi_fidelity | PASS: AC-006 multi-fidelity section; addendum sections 2, 14: accepted six-scale ladder, proxy caveat, mandatory target-scale replication |
| C_adaptive_effort | PASS: AC-007 resolved policy, budget table and bounded Auto; addendum sections 5-7 |
| D_contradiction_correction | PASS: AC-007 correction flow and distinct correction/training/modification/RSI definitions; addendum section 8 |
| E_context | PASS: AC-008 three context concepts and fields; 32/8=4 payload patches; no effective 4K/32K/1M claim; addendum section 16 |
| F_distribution | PASS: AC-008 conditional byte probabilities, entropy/top-1/top-2/margin/space/concentration/diversity/runs; separate original symbol/control mass; no collapse inference from argmax |
| G_base_path | PASS: AC-007 base/off section and addendum section 3; independent disablement and declared compatibility contract |
| H_host_boundary | PASS: Addendum section 4 and AC-006 external authority; host owns controllers, sandbox and promotion; no neural promotion privilege |

The five artifacts were read in full. No substantive missing requirement, contradictory wording, malformed field or broken local link was found. No specification wording was changed for style.

## Validation and corrected defects

- `.venv/Scripts/python.exe -m ruff format tests/test_future_capabilities_spec.py` — Formatting completed; hash literals then wrapped to fix E501; final format check passes.
- `.venv/Scripts/python.exe -m ruff check tests/test_future_capabilities_spec.py` — PASS after two E501 line-length corrections.
- `.venv/Scripts/python.exe -m pytest tests/test_future_capabilities_spec.py -q -p no:cacheprovider` — 3 passed in 0.23s.
- `json.loads(Path("docs/future_capabilities_contract_v1.json").read_text(encoding="utf-8"))` — PASS; explicit standalone parse.
- `AST comparison of test file with dabe922e; hash-object comparison of five spec artifacts and both Stage-A reports` — PASS; test behavior unchanged; published specification and Stage-A content unchanged.
- `Focused tests: local Markdown link resolution and balanced code fences` — PASS for addendum and all three AC documents.

The first lint attempt stopped on two E501 hash-literal lines before pytest ran. Parentheses around the unchanged strings fixed those lines. Ruff also expanded a set literal. Comparing Python ASTs against the published test proves the final changes are formatting-only. All three focused tests then passed. They check exact effort/RSI/promotion/context enumerations, uniqueness, lineage/evaluator/context fields, prohibited authorities, resource bounds, proposal-only states, local links and immutable FINAL hashes.

## FINAL file preservation

| File | Pre SHA-256 | Post SHA-256 | Result |
|---|---|---|---|
| Unified_Edge400_Master_Bible_v3.0_FINAL.md | `754e99e9feca52c8744b71891d6e71993233dea43c3d7155900334e28c121f75` | `754e99e9feca52c8744b71891d6e71993233dea43c3d7155900334e28c121f75` | EXACT MATCH |
| Unified_Edge400_Master_Roadmap_and_Formula_Board_v1.0_FINAL.md | `91bb81b09162e8ccdbdd2baa5bcd5c849a912bfe15e53f798c09bd70b788af99` | `91bb81b09162e8ccdbdd2baa5bcd5c849a912bfe15e53f798c09bd70b788af99` | EXACT MATCH |

Both existing Stage-A reports and all five published specification artifacts remain unchanged. Their local hashes are recorded in the companion JSON. Stage-A evidence remains authoritative and ready for continuation review, not promoted by this documentation.

## File inventory

Created in the existing specification commit:

- `Unified_Edge400_Future_Capabilities_Addendum_v1.0.md`.
- `docs/architecture_changes/AC-006_Verified_RSI_Framework.md`.
- `docs/architecture_changes/AC-007_Adaptive_Reasoning_Effort.md`.
- `docs/architecture_changes/AC-008_Context_and_Distribution_Promotion_Gates.md`.
- `docs/future_capabilities_contract_v1.json`.
- `tests/test_future_capabilities_spec.py`.

This closeout creates only `reports/future_capabilities_addendum_review.md` and `.json`, prepends the current planning entry to `docs/project_state.md`, and formats `tests/test_future_capabilities_spec.py`. Historical ledger entries are preserved. No model/training/data/config source or specification semantics change.

## Implementation status, limits and rollback

No RSI implementation exists. No effort controller exists. No context-certification benchmark has been run. No distribution-probability study has been run. No model weights changed. No training occurred. No model code, checkpoint, corpus/data or environment changed; no 2M continuation or 20M work started. No expensive model/GPU suite, corpus acquisition, Stage-A run or model validation was executed.

Document tests establish consistency, links and identity only; they do not establish runtime isolation, evaluator resistance to gaming, RSI efficacy, effort benefits or long-context retention. All remain optional research contracts with separate implementation and activation gates. Greedy whitespace remains distinct from demonstrated probability-mass collapse.

Rollback: the published specification commit and the accepted Stage-A history/checkpoints remain available; this closeout changes only reports, a new ledger entry and test formatting. The remaining risk is treating a written contract as a working capability. Human review is the next step; model/training work still requires separate authorization.

## Unresolved research questions

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

FUTURE-CAPABILITIES SPEC STATUS: READY FOR HUMAN REVIEW
