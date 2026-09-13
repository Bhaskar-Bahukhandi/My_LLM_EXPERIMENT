# Final corrected download authorization — r3

**READY: 60 / 60 feasibility cells pass using the unchanged invariant.** Only three SymPy mathematical guides were added to repair the two holdout pools. No corpus acquisition, Gutenberg book download, model training, architecture change or environment change occurred in this finalization pass.

## Frozen quota and history

Rollback `8c5cb375fced7cf1452f3007514a32cbc9678922` is unchanged. The original documentation deficit, r2 58/60 report and r3 quota-conflict preflight remain immutable. The human clarification explicitly rejects the erroneous 50,000-mathematics instruction: each SymPy holdout remains **50,000 total = 12,500 code + 12,500 documentation + 25,000 mathematics**. The complete 10,000,000 / 500,000 / 500,000 source budgets and 50/25/15/10 domain mixture remain unchanged.

History and artifact hashes are recorded in [the final JSON report](download_authorization_2m_r3.json) and [the quota clarification](../docs/authorization_r3_quota_clarification.json). R3 supersedes r2 without modifying it. The capacity contract retains v1 as the immutable budget authority and separately records its r2 predecessor.

## Exact admitted files and license findings

All three files are pinned to **SymPy 1.13.3**, tag `sympy-1.13.3`, commit `b4ce69ad5d40e4e545614b6c76ca9b0be0b98f0b`:

- `doc/src/guides/solving/find-roots-polynomial.md` — validation, 21,371 raw bytes, 21,068 eligible bytes after known footer exclusion; SHA-256 `9a252fa01e75b1f40d0ba28ae46e6e4f4d5ece5462858e00adc69513f47a49c1`.
- `doc/src/guides/solving/solve-diophantine-equation.md` — validation, 10,484 raw bytes, 10,232 eligible bytes after known footer exclusion; SHA-256 `1d7b841aefd979f9fb654e91c00355d13baa620c7fb49c1e5607cd414bea8cc2`.
- `doc/src/guides/solving/solve-ode.md` — test, 18,727 raw bytes, 18,480 eligible bytes after known footer exclusion; SHA-256 `61f83d28becd52d9d32fe8ec4f58ca805f27119a1fe3ce1e89a88bbb4a1bd3bb`.

The complete [versioned allowlist](../docs/documentation_file_allowlist_2m_r3.json) retains the 58 r2 entries unchanged and appends these three records. It is a materialized overlay for the unchanged loader; apply it once alongside the base allowlist, not in addition to a second copy of r2.

The full pinned SymPy root LICENSE, its Git blob identity and SHA-256 were verified. The guides inherit BSD-3-Clause terms; no overriding file-local license, copied-from attribution or permission-only restriction was found during the preserved inspection. Root notices, disclaimers and non-endorsement conditions remain required. Listed bundled-material notices are preserved and do not authorize unrelated files. External references are links, not acquired or incorporated source material.

These are authored mathematical expositions: polynomial roots and multiplicities; integer solutions and parameterized Pythagorean triples; and differential equations, initial conditions and integration constants. They are neither implementation source nor generated API output. Embedded examples and plot directives are not executed or expanded. The nonmathematical, repeated “Report a Bug” footer is excluded from each guide's eligible range before capacity accounting. No cleaned corpus payload was produced.

## Split feasibility and duplicate review

- Validation adds `math-holdout/algebraic-integer-equations`, containing the polynomial-roots and Diophantine guides. Existing geometry material remains validation.
- Test adds `math-holdout/ordinary-differential-equations`, containing the ODE guide. Existing combinatorics material remains test.
- Each repaired cell has **two source units**. No new path or unit was present in either older allowlist, and validation/test groups are disjoint. Each whole guide belongs to one split; footer exclusions do not split documents across roles. Existing training files and units were not renamed or reassigned.

The preserved bounded duplicate review covers the three candidates, seven related existing training documents and three r2 SymPy prose documents. No normalized-identical documents or near-duplicate whole-document derivative was identified in that reviewed set. Maximum exact five-token-shingle Jaccard was 0.090677; maximum shorter-document shingle containment was 0.256098. These are review evidence, not a new acceptance threshold. Shared syntax/examples and guide boilerplate are explicitly acknowledged; shared bug-report footers are ineligible.

This bounded review does **not** certify a full corpus free of duplicates or benchmark contamination. The unchanged global exact/normalized/near-duplicate and benchmark rules still apply before final splits/windows. Cross-role duplicate clusters must be excluded, never moved between roles.

## Capacity

Unchanged policy: `H = ceil(Q × 10 / 7) − Q`, with `C >= Q + H`. For each mathematical holdout, **Q = 25,000**, **H = 10,715**, and minimum eligible raw material is **35,715 bytes**. Expected retained capacity is `floor(7C/10)` per cell, an estimate incorporating cleaning and deduplication, not a measured outcome.

| Split | Final quota | Eligible raw C | Expected retained | Required H | Achieved raw headroom C−Q | Surplus over minimum | Units |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation | 25,000 | 39,387 | 27,570 | 10,715 | 14,387 | 3,672 | 2 |
| test | 25,000 | 44,341 | 31,038 | 10,715 | 19,341 | 8,626 | 2 |

Expected final surpluses are **2,570 validation bytes** and **6,038 test bytes**. All other cells are unchanged. The [full 60-cell result](authorization_capacity_2m_r3.json) was independently recomputed and matches the persisted evidence.

```powershell
.venv\Scripts\python.exe scripts/authorization_capacity.py docs/authorization_capacity_2m_r3.json
```

The unchanged CLI returns **exit 0** and **60/60 PASS**. No SymPy exception or altered headroom was introduced.

## Validation, changed files and boundary

The interrupted test file did not exist. It was created through a separate repository patch. **30 focused tests pass** (0.85 s): all 24 existing tests plus six r3 cases. Removing either newly added pool reproduces precisely the corresponding original failed cell. Tests also cover unchanged allocations/margin, disjoint and previously unused identities, preserved r2 entries, quota history and the CLI result.

Ruff passes; all 38 Python files pass formatting; compilation and dependency checks pass in both environments. Three new candidate SHA-256 values, complete raw sizes, Git blob IDs, footer ranges and pinned root-license identity were verified. All recorded starting hashes, benchmark raw files/fingerprint sidecar, project fixture identities, model source, FINAL specifications and rollback remain unchanged. No model/GPU/overfit experiment was rerun because this phase changes no runtime implementation.

R3 files: `tests/test_authorization_r3.py`; the clarification, allowlist and capacity-contract JSONs under `docs/`; the capacity, source-table and final authorization reports under `reports/`; and `docs/project_state.md`. Earlier reports remain intact. The accepted invariant and its existing tests are byte-for-byte unchanged. No new Git commit or remote was created.

Conservative projected combined disk use is 801,080,613 bytes, including the preserved review evidence and separate 200,000,000-byte run allowance. This remains below the unchanged combined 1,000,000,000-byte ceiling and data/preprocessing ceiling. Actual acquisition/preprocessing peak use and retained yield remain unmeasured.

Authorization repair is complete. The next step is the separately authorized bounded acquisition phase, which must enforce all existing license, security, contamination, deduplication and hash gates. Stop here; this report does not initiate acquisition.

CORRECTED DOWNLOAD AUTHORIZATION STATUS: READY
