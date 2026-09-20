# Real corpus build: blocked after benchmark exclusion

The completed scan leaves **53 / 60 PASS**. No final corpus or manifest was created. The required totals remain 10,000,000 / 500,000 / 500,000 bytes; selected totals are **0 / 0 / 0**. No training, backward, optimizer, model or environment change occurred.

## Exact blocking cells

| Source | Domain | Split | Required | Available | Deficit |
|---|---|---|---:|---:|---:|
| cpython | code | test | 75,000 | 0 | 75,000 |
| cpython | documentation | train | 1,000,000 | 840,136 | 159,864 |
| cpython | documentation | test | 50,000 | 0 | 50,000 |
| numpy | code | train | 750,000 | 117,370 | 632,630 |
| sympy | code | validation | 12,500 | 0 | 12,500 |
| sympy | code | test | 12,500 | 0 | 12,500 |
| sympy | structured_math | test | 25,000 | 18,480 | 6,520 |

## Measured exclusion result

Scanned 3,438 documents in 231 source units / 57,415,744 bytes. There were 75 first-hit raw exact short-content matches (also normalized matches), zero normalized-only first hits and zero near first hits. These are first-hit-per-document counts, not an exhaustive inventory. Excluding all 46 affected units removes 898 records / 17,354,845 bytes. All affected source/domain/split counts and all 60 cells are recorded in the JSON report.

Some matches are common short benchmark support content, including import statements. Under the frozen matcher they remain content, and whole-unit exclusions remain in force. No thresholds or field eligibility were changed to recover capacity. There are zero unresolved admitted hits because every matched unit is excluded; this does not claim semantic contamination completeness. The matcher uses frozen normalization and bounded paragraph/window near-duplicate retrieval.

The independent registered-project-fixture scan verified 17 generator/source identities and 14 fixture aliases (two unique payloads), scanned all 3,438 candidates, and found zero hits. Global dedup previously removed five same-role near-duplicate documents / 10,932 bytes, with no exact or cross-role clusters. That prior result does not substitute for the blocked final independent leakage audit.

## Evidence and boundaries

All five source groups completed acquisition/filtering. Detailed raw, license-eligible, cleaning-category and per-unit evidence is summarized and hash-linked in the machine-readable report. Cleaning estimates remain historical; actual post-cleaning capacity is measured without a second retention haircut. Seven selected Gutenberg works have raw and cleaned hashes; reserve ebook 158 remains unused, rejected 1260 remains excluded, Wikipedia contributes zero.

Measured logical acquisition/preprocessing footprint: **320,468,815 bytes**, below the 800,000,000-byte data cap plus separate 200,000,000 reserve. This is disk accounting, not process memory.

Final split assignment, independent final leakage audit, quota selection, corpus hashes, manifest hash and loader dry run are **NOT RUN / NOT CREATED** because the prerequisite gate failed. No placeholder dataset manifest was written. Frozen reserved roles, budgets, policies and source pins are unchanged.

## Validation, preservation and review

51 relevant tests passed. Ruff, formatting (54 files), compile, both dependency checks and whitespace checks passed. Historical authorization tests execute unchanged against their frozen input snapshot because their contracts bind the old report hash. They establish preservation, not post-acquisition sufficiency. Model source identity and every authorization snapshot file were verified; current global-stage implementation hashes and candidate payload hashes were rechecked. All 2,202 historical filtering receipt hashes remain unchanged. The final affected test replay passed again (51 tests, 0.76s). No model/training test was run in this data-only pass.

Rollback `720af6e63ee8759654b287848173bbd3e74867c4` and earlier commits remain untouched. Historical authorization reports are preserved under `data/pilot-2m-r1/provenance/authorization` and in Git; failed/intermediate filtering remains preserved. The exact changed/added file list is in the JSON report. Changed files comprise corpus-only scripts/tests/policies/filter reports, this report, its JSON evidence and the project ledger. No production model source changed.

The final review found no reason to loosen exclusions. Remaining risk is conservative loss of large source-unit pools from common short content; any remedy requires separately reviewed candidate/policy decisions. Next step: review the seven explicit deficits, with no automatic acquisition or replacement.

REAL CORPUS BUILD STATUS: NOT READY
