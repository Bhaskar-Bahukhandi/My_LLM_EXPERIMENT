# Real corpus v2: blocked at 58/60

The approved exhaustive matcher-v2 scan is complete. The targeted eligible additions passed their gates, but both SymPy code holdouts remain short by **12,500 bytes**. Final corpus construction stops at the required capacity gate. V1 evidence, its 75 hits/46 exclusions, the 53/60 result and the prior NOT READY reports remain byte-identical.

## Actual exhaustive scan

All 3,438 original post-dedup documents / 231 source units / 57,415,744 bytes were scanned. There are 81 distinct document/content matches (80 raw-exact), covering five unique content identities. These are not occurrence counts for every repeated span. All associations are retained; exact evidence dominates redundant near evidence for the same identity.

26 support-import matches are intrinsically non-actionable; 16 occur in units excluded for other protected content. There are 55 actionable matches: 52 code-solution and three generated-solution matches. One actionable match is normalized-only; none is a verified near match. Actionable problem and ordinary-answer-class matches are zero in this actual pool. All 49 original protected answer/solution hits remain actionable. V2 excludes 36 source units / 14,784,656 bytes.

Near retrieval remains the frozen bounded paragraph/window policy, with verified five-token Jaccard >=0.85. This is exhaustive over matched identities under that policy, not a semantic completeness claim. No minimum-length/frequency exception was introduced.

## Actual pre-expansion deficits

| Source | Domain | Split | Required | Before exclusion | Excluded | Remaining | Deficit |
|---|---|---|---:|---:|---:|---:|---:|
| cpython | documentation | train | 1000000 | 1474384 | 634248 | 840136 | 159864 |
| sympy | code | validation | 12500 | 339738 | 339738 | 0 | 12500 |
| sympy | code | test | 12500 | 677810 | 677810 | 0 | 12500 |
| sympy | structured_math | test | 25000 | 43308 | 24828 | 18480 | 6520 |

## Targeted additions and new control conflict

19 bounded files totaling 292,332 raw bytes were acquired from the exact existing CPython and SymPy pins. No new dataset, Gutenberg work or source version was added. Required raw capacity used the unchanged 7/10 policy; whole-document rounding supplied the small reserve. Exact paths, blob/raw hashes, sizes, unit roles and review decisions are recorded in the linked plan and file-review reports.

The first targeted batch incorrectly checked file modes after downloading. Eleven SymPy code examples (41,984 bytes) are mode `100755`, contrary to the frozen allowlist rule “Mode 100644 only.” They remain rejected raw evidence and contribute zero eligible capacity. This is a control-policy conflict, not a claim that reading executable-marked text executes it. The new targeted acquisition helper validates the entire batch before its first network call; regression tests prove an invalid later file causes no downloads.

Seven CPython documents and one SymPy mathematical guide pass the mode and file-level licensing/control checks: 250,348 raw bytes -> 250,276 cleaned bytes. The 72 removed bytes are attribution contacts retained in raw provenance; author names remain. Generated/copy language in the CPython guides refers to runtime behavior, not file generation or third-party copying. No nested alternate-license material or unresolved file-local attribution was admitted.

All eight additions passed unchanged quality/privacy filtering, exact and near dedup against the whole existing candidate pool, exhaustive v2 benchmark matching and the frozen project-fixture matcher. New actionable benchmark matches: zero. New project-fixture hits: zero. Existing v2 results were reused by hash for unchanged originals. No mode-ineligible file was grandfathered.

## Final measured capacity

All 60 cells were recomputed from actual surviving bytes: **58/60 PASS**. CPython train documentation is now **1,079,137 / 1,000,000** bytes; SymPy test mathematics is **29,755 / 25,000** bytes. Mathematical quota remains 25,000, not 50,000.

| Remaining cell | Required | Available | Deficit |
|---|---:|---:|---:|
| SymPy / code / validation | 12,500 | 0 | 12,500 |
| SymPy / code / test | 12,500 | 0 | 12,500 |

The conservative cached metadata inventory contains at most 26,336 unused mode-100644 Python bytes outside frozen exclusion scopes, even counting unreviewed small helpers optimistically. The two holdouts require at least 35,716 raw candidate bytes under the unchanged headroom rule, a 9,380-byte shortfall before further filtering. This is not an admission allowlist. The earlier inventory omitted the explicit external-directory exclusion; its corrected r2 and original r1 are both retained. No additional downloads were justified by this inventory.

## Validation and boundary

62 affected tests pass (1.04s); Ruff, format (63 files), compile and both dependency checks pass. Hash checks verify preserved v1 inputs, versioned stage evidence, every surviving payload and the unchanged model source. Historical authorization tests replay unchanged against their frozen inputs, as in earlier phases. Final review found and corrected the late mode-check path; no production matcher-v1 or model source was edited.

Measured logical acquisition/preprocessing footprint: 326,456,162 bytes against the 800,000,000-byte data cap plus separate 200,000,000-byte reserve. This is disk accounting, not RSS.

Final source assignment, independent final cross-split leakage audit, quota selection, corpus hashes, manifest hash and loader dry run were **not run** because 60/60 is mandatory first. No final manifest was created. Selected bytes remain 0/0/0; final targets remain 10,000,000 / 500,000 / 500,000. No backward, optimizer, model-weight, architecture, CUDA, RAG or later-scale work occurred.

Files added in this continuation include the v2 policy/hash, exhaustive matcher and resumable scanner, candidate/file-review records, expansion processor and evidence, targeted acquisition control/guard/tests, corrected capacity inventory, fixture audit and this versioned report/evidence. The project ledger records the actual stopped state. Previously modified v1 build reports were not changed during this continuation.

Rollback `720af6e63ee8759654b287848173bbd3e74867c4` and earlier commits are unchanged; no new commit or remote was created. Raw rejected candidates and all failed/intermediate evidence remain available. There is no fallback that silently restores excluded units.

**Next decision:** review whether a new explicit control version may accept regular mode-100755 source-text blobs as inert data while retaining all other gates. This report does not approve the exception. Alternatively, an eligible scope must be separately established within the existing pins. No quota borrowing, headroom relaxation or additional dataset acquisition is proposed.

REAL CORPUS BUILD STATUS: NOT READY
