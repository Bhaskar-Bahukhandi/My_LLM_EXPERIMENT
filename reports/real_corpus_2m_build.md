# First real-corpus build: blocked before acquisition

The frozen source allocations and file allowlists disagree. The directive explicitly requires stopping before downloading in this case. No substantive source content was acquired, no corpus was built, and no training ran.

Reviewed inputs are preserved in commit `8c5cb375fced7cf1452f3007514a32cbc9678922`. The working tree was clean immediately after this commit. It includes the previously reviewed GPU integration and authorization/exclusion artifacts, without rewriting earlier commits or adding a remote. The historical READY authorization report remains frozen; this capacity finding supersedes its readiness conclusion.

## Blocking evidence

| Source / split | Documentation target | Maximum allowlisted raw documentation bytes | Minimum deficit |
|---|---:|---:|---:|
| CPython train | 1,000,000 | 0 | 1,000,000 |
| NumPy train | 250,000 | 0 | 250,000 |
| All sources validation | 75,000 | 0 | 75,000 |
| All sources test | 75,000 | 0 | 75,000 |

The previous checks verified hashes and allocation sums but missed this cross-artifact capacity conflict. All 665 CPython and 84 NumPy approved files are classified as code. SymPy has only 311,995 raw documentation bytes in its training candidates and none in validation/test. Cleaning cannot create the missing authorized documentation. This exceeds a minor intact-document rounding adjustment.

The earlier metadata filter treated `Doc/license.rst` and `doc/source/license.rst` as nested license scope boundaries and excluded the surrounding CPython and NumPy documentation. Their actual scope needs contextual review. Removing that exclusion indiscriminately would not establish per-file eligibility. No allowlist or source classification was changed during this pass.

## Frozen sources and budgets

Exact commits, license-evidence hashes, approved Gutenberg editions, and frozen artifact SHA-256 values are retained in `real_corpus_2m_evidence.json` and the linked authorization artifacts at the rollback commit. Gutenberg IDs 1400, 766, 1023, 145, 1342, 158, 768 and 1661 remain candidates; edition 1260 remains rejected. Wikipedia remains deferred with zero bytes.

| Source | Target train | Target validation | Target test | Selected bytes, all splits |
|---|---:|---:|---:|---:|
| Gutenberg | 5,000,000 | 250,000 | 250,000 | 0 |
| CPython | 2,500,000 | 125,000 | 125,000 | 0 |
| NumPy | 1,000,000 | 50,000 | 50,000 | 0 |
| SymPy | 1,000,000 | 50,000 | 50,000 | 0 |
| mathlib4 | 500,000 | 25,000 | 25,000 | 0 |
| Total | 10,000,000 | 500,000 | 500,000 | 0 |

Domain targets remain 50% general text, 25% code, 15% documentation and 10% structured mathematics. No alternative mixture or acquisition order was executed.

## Validation and unexecuted work

Five fingerprint tests passed (0.06 s); Ruff, formatting (35 files), whitespace, frozen-input hashes, allocation arithmetic, 25 benchmark raw-file hashes, fingerprint sidecar and local fixture identities passed. Model source and both FINAL specification hashes remain unchanged. Full CPU/GPU regression was not rerun because no runtime code changed. The new capacity gate failed as detailed above.

Downloaded-file licensing, safe extraction, cleaning, security/quality filtering, exact/near deduplication, benchmark scanning, split assignment, cross-split leakage checks and loader dry-run are **NOT_RUN**. Retention is **NOT_MEASURED**. There are no raw/cleaned corpus hashes, selected source-unit counts or corpus hash. A final dataset manifest is **NOT_CREATED** because no admitted corpus exists. Existing benchmark fingerprints are evaluation infrastructure, not evidence that an unbuilt corpus passes contamination checks.

Gutenberg whole-text hashes remain `PENDING_AUTHORIZED_ACQUISITION`; raw and cleaned payload hashes are mandatory immediately following a future authorized download. Existing metadata/header provenance remains intact.

Existing authorization evidence occupies 182,287,216 logical bytes. Newly acquired training data occupies zero bytes. The 800,000,000-byte data/preprocessing cap plus separate 200,000,000-byte run allowance remains unchanged; actual acquisition/preprocessing peak disk use is unmeasured.

## Rollback, risks and next step

Changes in this pass are this report, its JSON evidence and `docs/project_state.md`; the ignored pre-acquisition audit records the capacity check. The coherent rollback commit above preserves the previously reviewed state. No model, environment, recurrence, FINAL specification, approved source scope or exclusion identity changed.

The next step is a narrow documentation-license scope review and an explicitly amended file/holdout allowlist with adequate source/domain/split capacity, followed by human approval of that scope change. Building now would require unauthorized substitutions or a materially different domain mixture. The current corpus is not ready for training review.

REAL CORPUS BUILD STATUS: NOT READY
