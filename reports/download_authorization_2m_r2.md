# Corrected download authorization — revision 2

**NOT READY.** The documentation expansion resolves the original missing-capacity conflict. The new generic feasibility gate checks all 60 source/domain/split cells and exposes two additional undersized SymPy mathematical holdouts. No substantive corpus acquisition or corpus-scale preprocessing occurred.

The immutable blocked rollback remains `8c5cb375fced7cf1452f3007514a32cbc9678922`. Revision 1 and its failed build evidence are retained unchanged. This revision supersedes the old readiness conclusion through new files and hashes; it does not rewrite history, change model weights/architecture or modify either environment or FINAL specification.

## Root cause and correction

The original tree filter treated any nested license-looking file as grounds to exclude its containing directory. That removed all CPython `Doc/` and NumPy `doc/source/` candidates despite planned documentation quotas. It left at least 1,400,000 prescribed documentation bytes without candidates. The new file overlay admits only inspected authored documents from the same pinned revisions; implementation code is never relabeled.

The full scope is the immutable original allowlist plus ADMIT entries in [the revision-2 overlay](../docs/documentation_file_allowlist_2m_r2.json). Every added or rejected file records repository, exact commit/tag/path, Git blob identity, SHA-256, license evidence, local notices, inclusion directives, classification, raw byte estimate and split unit. This avoids duplicating or silently editing the old 3,698-file manifest.

| Source | Added documentation files |
|---|---:|
| cpython | 35 |
| numpy | 18 |
| sympy | 3 |
| mathlib | 0 |

A bounded review inspected 58 files totaling 2,336,362 bytes; 56 are admitted and two rejected. These copies live only in ignored authorization evidence. No Gutenberg full text or source archive was downloaded, and no corpus was assembled. Original Gutenberg edition decisions, pending full-text hashes and benchmark identities remain unchanged.

## License and source-unit findings

CPython's pinned [Doc/license.rst](https://raw.githubusercontent.com/python/cpython/0cc81280367df838c4b199f8f0378837165071c2/Doc/license.rst) explicitly covers documentation under PSF License Version 2 and offers 0BSD for documentation code. Incorporated material remains excluded. The selected standard-library and language-reference files contain explanatory prose, with contributor credits recorded separately. Email documents stay in the existing validation unit; urllib documents stay in its test unit. Reference prose and other selected API topics remain training candidates.

NumPy inherits the pinned [BSD-3-Clause license](https://raw.githubusercontent.com/numpy/numpy/3b377854e8b1a55f15bda6f1166fe9954828231b/LICENSE.txt), subject to file-local and bundled-material exclusions. Its documentation license page contains a relative include whose `doc/LICENSE.txt` target is absent at this source pin (verified HTTP 404). The root license is explicitly identified as inherited evidence; no generated target or build result is assumed. `basics.indexing.rst` and `basics.ufuncs.rst` credit adaptation from *Guide to NumPy* and are rejected pending specific rights evidence. Selected user-guide prose is training material; `arrays.nditer.rst` and `arrays.classes.rst` are independent validation/test file units.

SymPy's pinned [BSD-3-Clause license](https://raw.githubusercontent.com/sympy/sympy/b4ce69ad5d40e4e545614b6c76ca9b0be0b98f0b/LICENSE) covers the reviewed first-party explanatory additions. `guides/custom-functions.md` joins the existing training prose unit; `explanation/best-practices.md` and `explanation/glossary.md` form separate validation/test file units. Existing selected files and holdout roles are unchanged. mathlib's pinned tree supplies no proposed standalone prose addition; existing theorem files remain structured mathematics.

Include/autodoc/autosummary directives will not fetch or expand additional content. Image assets, external manuals, generated indexes and linked examples do not gain eligibility. Original byte spans are preserved if retained; removed metadata and required notices remain in separate provenance/license records. Actual secret/PII/benchmark scans remain acquisition gates. Cross-role duplicate clusters must be excluded, not reassigned. File-unit separation is planned here; content leakage cannot be certified before scanning acquired content.

## Capacity and margin

The fixed budgets remain 10,000,000 / 500,000 / 500,000 bytes. The complete matrix preserves the approved train source/domain allocation and uses its within-source proportions for validation/test (one twentieth of train). Source totals and domain totals are both checked against the hash-bound original approval, not merely against each other. The 50/25/15/10 mixture is unchanged.

For **every** `(source, domain, split)` cell, including explicit zeros:

`H = ceil(Q × 10 / 7) − Q`; require `C >= Q + H`.

The 70% retained yield is an **ESTIMATE**: 80% after cleaning times 87.5% after deduplication. It requires at least 42.857% raw oversupply over final quota. The report floors retained capacity separately per cell and never borrows surplus from another cell. Actual yield may be lower and must be measured later.

| Split | Old raw docs | Corrected raw docs | Estimated retained | Required final | Estimated surplus |
|---|---:|---:|---:|---:|---:|
| train | 311,995 | 2,266,962 | 1,586,872 | 1,500,000 | 86,872 |
| validation | 0 | 170,842 | 119,588 | 75,000 | 44,588 |
| test | 0 | 164,014 | 114,809 | 75,000 | 39,809 |

| Source | Split | Raw candidates | Estimated retained | Final quota | Required raw headroom H | Estimated surplus |
|---|---|---:|---:|---:|---:|---:|
| cpython | train | 1,484,562 | 1,039,193 | 1,000,000 | 428,572 | 39,193 |
| cpython | validation | 94,375 | 66,062 | 50,000 | 21,429 | 16,062 |
| cpython | test | 97,563 | 68,294 | 50,000 | 21,429 | 18,294 |
| numpy | train | 403,439 | 282,407 | 250,000 | 107,143 | 32,407 |
| numpy | validation | 25,755 | 18,028 | 12,500 | 5,358 | 5,528 |
| numpy | test | 27,399 | 19,179 | 12,500 | 5,358 | 6,679 |
| sympy | train | 378,961 | 265,272 | 250,000 | 107,143 | 15,272 |
| sympy | validation | 50,712 | 35,498 | 12,500 | 5,358 | 22,998 |
| sympy | test | 39,052 | 27,336 | 12,500 | 5,358 | 14,836 |

All documentation cells pass. This is candidate capacity, not a claim that every existing conditionally approved file will survive acquisition review. Some existing SymPy mathematical candidates are short reference stubs; no generated expansion is counted.

## Generic invariant result and remaining blockers

**58/60 cells pass.** SymPy structured-math validation has 8,087 raw bytes (5,660 estimated retained); test has 25,861 raw bytes (18,102 estimated retained). Each requires 25,000 final bytes, or 35,715 raw bytes with the declared margin. Raw shortfalls are 27,628 and 9,854 bytes. The test pool exceeds its nominal quota slightly but fails headroom, as intended.

These deficits cannot be hidden by mixing aggregate surpluses. With CPython/NumPy contributing code/docs and mathlib's entire 25,000-byte holdout contribution already mathematical, the remaining 25,000 mathematical bytes in each holdout must come from SymPy under the frozen source/domain budgets. No further mathematical scope or reclassification was silently introduced. A subsequent explicit review should identify eligible mathematical documents and independent holdout groups at the same pin, then rerun the same gate.

Run the offline pre-authorization gate with:

```powershell
.venv\Scripts\python.exe scripts/authorization_capacity.py docs/authorization_capacity_2m_r2.json
```

It returns exit code **1** for this revision. Acquisition must not proceed from either the superseded READY report or an unchecked allocation table. Passing this necessary gate later will still not establish measured yield, license compliance of unseen files or contamination clearance.

## Validation, hashes and preservation

- Focused suite: **24 passed** (0.47 s) after final invariant tightening, including the original absent-documentation case, missing cells, exact rounding, unknown allocations, duplicate content, split conflicts, hash mutation, budget mutation and a CLI failure exit.
- Ordinary CPU suite: **182 passed**, eight overfit/CUDA cases deselected (168.09 s), before the last two invariant tests and checker tightening; existing optional NumPy warning retained. No overfit or GPU experiment rerun.
- Ruff, formatting (37 files), compilation and dependency checks in both environments pass.
- All 58 inspected files match pinned Git blob IDs, byte counts and SHA-256. Raw frozen artifact hashes match the earlier blocker report. Git clean-filter identities match the rollback; `core.autocrlf=true` explains raw Git-blob versus working-file line endings. Model source and FINAL specifications remain unchanged.
- New artifact hashes and inherited source-table references are in [download_authorization_2m_r2.json](download_authorization_2m_r2.json). The [capacity result](authorization_capacity_2m_r2.json) records every cell; the [contract](../docs/authorization_capacity_2m_r2.json) fixes quotas and margin.

Additional review evidence occupies 2,546,187 logical bytes at measurement. Adding it conservatively to the prior storage projection gives 800,830,696 bytes including the separate 200,000,000-byte run allowance, within the unchanged 1,000,000,000-byte combined ceiling and 800,000,000-byte data/preprocessing ceiling. This is a projection; actual corpus preprocessing peaks are unmeasured.

Changed files are the new invariant script and tests, versioned documentation overlay and capacity contract, versioned source approval and authorization reports, capacity evidence, and project ledger. Earlier blocked-build reports remain intact. No new commit was made during this correction pass; the rollback remains available unchanged.

CORRECTED DOWNLOAD AUTHORIZATION STATUS: NOT READY
