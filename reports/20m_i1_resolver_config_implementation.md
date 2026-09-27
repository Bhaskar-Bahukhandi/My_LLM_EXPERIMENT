# 20M-I1 resolver/configuration implementation closeout

Decision: **20M_I1_READY_FOR_REVIEW**. R1/R2/R3: **PASS**. R4-R9: **NOT IMPLEMENTED**.
20M pilot **NOT AUTHORIZED**; 20M training **NOT RUN**; 2M **FROZEN**; TEST **SEALED**.
Kaggle compute **NOT RUN**. This is configuration evidence, not pilot readiness.

## Entry and scope

Pre-I1 entry: `fad209fccb4add87e66770fb0470f257cf787f2d`. Interrupted implementation: `57e19c78a9bdec8abc72826ef300b7a254d9b9f1`.
At closeout entry, local master, origin/master and live remote matched the latter with
a clean tree. The implementation referenced this report pair, but both files were
absent. The full 14-file implementation diff was reviewed; no implementation defect
was found and no R1-R3 code was rewritten. This additive closeout creates the report
pair and updates only the current-state next-step link/scope.

## R1: resolver, schema and accounting

`edge_dense_v1_2m` fixes the legacy grid (widths 128/192/256; depths 2/3/4/6/8/12).
`edge_dense_v2_20m` fixes widths 256/320/384/448/512/576/640/704/768/896/1024 and
depths 3/4/8/12/21/30/46. Explicit mode requires pinned width and depth, evaluates
exactly one candidate and publishes no selected config on tolerance failure. Search
requires both dimensions auto; conflicting custom spaces, unknown policy/mode and
incomplete sections fail. No target-size policy heuristic or MoE is permitted.

Schema remains `"1"`. Missing resolver preserves legacy behavior and omits the field
from serialization. Present resolver is typed and policy-bound in authored/resolved
identity. The exact pre-I1 parser rejects it with `config: unknown fields ['resolver']`.
Five representative legacy authored serializations match the old parser; the entire
frozen 2M resolved object and inventory accounting match retained evidence exactly.
The unchanged branch semantics and strict parser rules make this optional extension
coherent; no schema bump or checkpoint migration is needed.

Pinned candidate: **512 x 12**, state **64**, byte **64**, decoder **256**, expand **2**,
inner **1024**, head **64**, heads **16**, conv **4**, patch **8**, chunk **16**, MoE **false**.
Formula = inventory = real unique meta Parameters = **20,387,531**; unique trainable
tensors = **128**. Exact component and named tensor ledgers equal the accepted
pre-build mechanical receipt. All real-model parameters were checked to be meta;
forward was guarded against execution, RNG stayed unchanged, no model values or
padding were introduced. Synthetic accounting tests may allocate tiny toy tensors.

Selected resolved SHA-256: `6e7ffdc3c3448f7f19fa4466a9780c369c05cc572030c1b026d4eef1503b9359`.
Pre-I1 selected resolved SHA-256: `c0b58d464ac815bd0b44669e1e62e8d51884f7ebb4ec34c4e7fa47ec2bd939a4`.
This authorized identity change binds policy metadata, preserving the reviewed shape.
Old pre-build reports/hash maps remain historical evidence and are not rebound.

Frozen 2M: **1,929,579** parameters; complete resolved dictionary equals
`reports/edge_2m_resolved.json`; SHA-256
`7fe06b639741003e4c5ad9d57d32c2fd2b86f6805b2048ad1efd6aaa75e3a6f1`. No resolver field is inserted. The original
OUTSIDE_TARGET result retains its historical nearest candidate and 2% tolerance.

Diagnostic search: **SEARCH_COMPLETE**, **selected=null**, **77 candidates**;
repeated output is identical. All seven requested audited families are represented.
Distance/depth/width ranking is explicitly diagnostic, never an architecture
recommendation and never a replacement for the reviewed pinned candidate.

## R2: evidence-scoped status

Before: scope claimed executable model not implemented; blanket `training_ready=false`;
next gate was inventory reconciliation. After: current 20M is
`20M_PILOT_CANDIDATE / RECOMMENDED_FOR_PILOT / META_VERIFIED / META_CONSTRUCTED /
NOT_RUN / UNVERIFIED`, with `pilot_execution_ready=false` and
`training_authorized=false`. Next gate is R4-R9 followed by approved execution gates.
The vague field is removed. Exact frozen 2M alone reports COMPLETED_FROZEN_LINEAGE,
scoped to its accepted historical run; it does not authorize edited-source resume.
Memory preflight remains an analytic payload screen, without a runtime-fit claim.

## R3: current documentation and reference repair

The implementation rewrote README as a concise current-state guide with frozen 2M,
proposed 20M, safe configuration commands, gate restrictions and evidence links.
Project state keeps the historical ledger and adds the R1-R3 status/index. This
closeout supplies the missing evidence pair and narrows the next implementation
tranche to R4-R8 after review; R9 remains mandatory before any approved pilot.
Local Markdown links and I1 report references in README, project state, resolver
documentation and readiness source are checked after writing these artifacts.

## Validation

| Check | Actual result |
|---|---|
| targeted | 108 passed, 1 warning in 41.73s |
| reviewed_current | 143 passed, 1 warning in 40.03s |
| historical_replay | 30 passed in 1.23s; 1 passed in 2.01s |
| ruff | All checks passed! |
| format | 106 files already formatted |
| compileall | exit 0; no output |
| cpu_pip | No broken requirements found. |
| cuda_pip | No broken requirements found. |
| diff_check | exit 0; no output |

The reviewed current allowlist is taken verbatim from the accepted provenance receipt.
Targeted R1-R3 tests and the mandatory historical wrapper run separately; the wrapper
runs the unchanged historical tests in verified temporary copies. Full commands,
outputs, counts and warnings are retained in the JSON receipt. Production corpus
payload suites and all model forward/training suites were excluded. No 20M forward,
backward, optimizer, training, dataset acquisition or Kaggle operation occurred.
Negative coverage includes unknown/incomplete policy, invalid modes, pinned search,
auto explicit, conflicting search, tolerance rejection, no search override, legacy
retention, disabled MoE, malformed resolved policy and false execution readiness.

## Preservation, limitations and rollback

All **227** pre-existing tracked files outside the ten modified implementation files
retain their entry raw hashes. Four model mathematics files and all training-package
files are byte-identical to pre-I1. Existing formula and architecture validation
function ASTs are unchanged. All **66** historical snapshot files and the original
manifest identity match; the endpoint checkpoint was raw-hashed, never deserialized.
Historical expected hashes, accepted pre-build artifacts and state ledger remain intact.

Current tooling source SHA-256: `e56c334efc81a40eae0eb7839deb54e3f870a4787f76b782bd7d32cb7ab4c3cb`. The frozen historical source identity
remains `7d98de1b0504a1d511f3da95e17ec45dbcca1c5bd2d0dadbd251987261173eb4`.
TOOLING SOURCE CHANGED; MODEL MATHEMATICS / TRAINER SEMANTICS UNCHANGED. Source identity
checks are not bypassed; use an accepted frozen checkout for historical reproduction.

Not validated: 20M learning/quality, forward parity, GPU/T4 fit, throughput, optimized
backend, private Kaggle writes or compute. R4-R9 remain unimplemented. The optional
NumPy warning is recorded without changing environments. The only closeout-checker
repair was explicit UTF-8 decoding of historical Markdown; no evidence was repaired.

Rollback is additive: revert the closeout commit to remove this evidence/link update;
that restores the known missing-report gap. To roll back the complete I1 tooling,
also revert `57e19c78a9bdec8abc72826ef300b7a254d9b9f1` in reverse commit order. No reset or checkpoint migration is
needed. The reference fallback and frozen 2M artifacts remain available. Review this
receipt before separately authorizing **R4-R8**; R9 and approved execution gates are
still required before a pilot. The approval-bound Kaggle host boundary is unchanged.

## Files changed

Closeout only:

- `docs/project_state.md`
- `reports/20m_i1_resolver_config_implementation.md`
- `reports/20m_i1_resolver_config_implementation.json`

Previously pushed implementation (reviewed, not rewritten here):

- `README.md`
- `configs/models/edge_20m_candidate.yaml`
- `configs/models/edge_20m_search.yaml`
- `docs/project_state.md`
- `docs/resolver_policy.md`
- `src/unified_edge/cli.py`
- `src/unified_edge/config.py`
- `src/unified_edge/parameters.py`
- `src/unified_edge/preflight.py`
- `src/unified_edge/readiness.py`
- `src/unified_edge/resolve.py`
- `tests/test_cli.py`
- `tests/test_config.py`
- `tests/test_resolver_policies.py`
