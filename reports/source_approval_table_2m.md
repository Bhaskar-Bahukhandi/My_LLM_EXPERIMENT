# Final source approval table — reviewed allocation

**READY for scoped download authorization; no substantive training corpus downloaded.** Wikipedia is deferred entirely. This table supersedes the prior pending source recommendations. [Final authorization report](download_authorization_2m.md) provides exact works, split/fingerprint evidence and acquisition order.

| Source | Train bytes | Validation bytes | Test bytes | Decision |
|---|---:|---:|---:|---|
| gutenberg | 5,000,000 | 250,000 | 250,000 | APPROVE |
| wikipedia | 0 | 0 | 0 | DEFER |
| cpython | 2,500,000 | 125,000 | 125,000 | APPROVE |
| numpy | 1,000,000 | 50,000 | 50,000 | APPROVE |
| sympy | 1,000,000 | 50,000 | 50,000 | APPROVE |
| mathlib | 500,000 | 25,000 | 25,000 | APPROVE |
| Total | 10,000,000 | 500,000 | 500,000 | |

The domain targets remain exactly 50% general text /25% code /15% technical-scientific documentation /10% structured mathematics. File-level categories are disjoint. Global dedup precedes final split/window generation; reserved heldout source units can never supply training bytes.

## Project Gutenberg: selected English books

| Field | Detail |
|---|---|
| 1. Source | Project Gutenberg: selected English books |
| 2. Publisher | Project Gutenberg Literary Archive Foundation / volunteers |
| 3. Authoritative URL | [Source](https://www.gutenberg.org/) |
| 4. Version/edition | allowlist: docs/gutenberg_work_allowlist_2m.json; sha256: beaacb53ac5d813bf3d35a42441642587d8145fd732c51f7bd0efea1a8e4604b; accepted ids: 1400, 766, 1023, 145, 1342, 158, 768, 1661; rejected edition ids: 1260 |
| 5. License/version | Original authored prose screened outside copyright term under approved conservative policy; retain Gutenberg license/provenance separately. No blanket license for supplements. |
| 6. Planned-use eligibility | Original-English authored prose passes the approved conservative India/US catalog screening. No permission-only items, translations, posthumous originals or supplementary material admitted. Recheck complete notices on authorized acquisition. |
| 7. Attribution/redistribution | Retain complete raw license and ebook/author/title/source metadata. Any boilerplate removal must satisfy the particular ebook license and trademark conditions. |
| 8. Domain allocation (train bytes) | general text: 5000000 |
| 9. Clean train/validation/test bytes | train: 5000000; validation: 250000; test: 250000 |
| 10. Download size | catalog plain file bytes all accepted candidates: 9794897; downloaded full books: 0; compressed bytes: 0; status: Catalog sizes, not a measured corpus download; whole-text SHA-256 PENDING_AUTHORIZED_ACQUISITION. |
| 11. Decompressed size ESTIMATE | 9794897, 9794897 |
| 12. Retention ESTIMATE | original text unique fraction estimate: 0.7, 0.9 |
| 13. Selective/range access | Individual sanctioned plain-text book downloads possible; no arbitrary website scraping. No assumption of byte-range support; partial books would complicate licenses/grouping. |
| 14. Dedup risks | Repeated editions, translations, shared introductions and duplicated chapters. |
| 15. Benchmark contamination | Classics, school texts, translations and alternate editions often recur in evaluations. Deny registered benchmark works and cluster editions/translations by work. |
| 16. PII/privacy | Avoid modern personal records. Historical/fictional names are not indiscriminately stripped. Country-specific copyright review remains required. |
| 17. Secrets/security | Plain text only; reject embedded active content and misleading formats. |
| 18. Cleaning | Original chapters only; remove all front/back matter, prefaces, annotations, editorial additions, illustrations/captions and Gutenberg boilerplate with audited spans. Do not normalize retained bytes. Full original-file and cleaned-payload SHA-256 immediately after acquisition. Reject item-specific permission/restriction notices. |
| 19. Classification | BOTH |
| 20. Human decision | APPROVE — Human-approved scoped source; mandatory content/license/provenance checks remain on actual retrieval, not a blanket license grant. |

## English Wikipedia: selected stable article revisions

| Field | Detail |
|---|---|
| 1. Source | English Wikipedia: selected stable article revisions |
| 2. Publisher | Wikimedia contributors / Wikimedia Foundation |
| 3. Authoritative URL | [Source](https://en.wikipedia.org/) |
| 4. Version/edition | Freeze page IDs + revision IDs + timestamps + retrieved SHA-256. Article selection pending; do not bind to a moving latest revision. |
| 5. License/version | CC BY-SA 4.0 default text reuse; GFDL alternative where applicable; imported material may differ |
| 6. Planned-use eligibility | NOT APPLICABLE: no bytes authorized in this pilot. |
| 7. Attribution/redistribution | Retain page ID, revision ID, article/history URL, applicable license and change record; preserve required imported-content notices. No images. |
| 8. Domain allocation (train bytes) |  |
| 9. Clean train/validation/test bytes | train: 0; validation: 0; test: 0 |
| 10. Download size | pilot bytes: 0; status: DEFERRED |
| 11. Decompressed size ESTIMATE | 0, 0 |
| 12. Retention ESTIMATE | status: NOT APPLICABLE to this pilot |
| 13. Selective/range access | Revision API retrieves selected page IDs/revision IDs. No full dump/shard required; do not rely on arbitrary HTTP byte ranges. |
| 14. Dedup risks | Revision/redirect families, mirrors, syndicated passages and repeated templates. |
| 15. Benchmark contamination | Wikipedia-derived benchmarks and mirrored/revised articles can leak. Group page/redirect/revision families; exclude registered evaluation identities. |
| 16. PII/privacy | Exclude biographies of living people and personal contact/user/talk pages for this pilot. |
| 17. Secrets/security | Treat markup and links as data; do not follow links, render active content or execute templates. |
| 18. Cleaning | No acquisition or cleaning in this pilot; future CC BY-SA policy separately reviewed. |
| 19. Classification | RAG CANDIDATE |
| 20. Human decision | DEFER — Human deferred Wikipedia entirely; any future training/RAG use requires a separately reviewed CC BY-SA attribution/share-alike policy. |

## CPython 3.12.10 source and selected documentation

| Field | Detail |
|---|---|
| 1. Source | CPython 3.12.10 source and selected documentation |
| 2. Publisher | Python Software Foundation |
| 3. Authoritative URL | [Source](https://github.com/python/cpython) |
| 4. Version/edition | tag: v3.12.10; commit: 0cc81280367df838c4b199f8f0378837165071c2; license sha256: 3b2f81fe21d181c499c59a256c8e1968455d6689d269aa85373bfb6af41da3bf; license bytes: 13936 |
| 5. License/version | PSF License Version 2 plus historical/per-file notices; documentation examples additionally offer 0BSD |
| 6. Planned-use eligibility | YES for verified files actually covered by the stated permissive license, with notices/conditions preserved; engineering eligibility assessment, not approval of every repository file or a guarantee about generated outputs. |
| 7. Attribution/redistribution | Retain PSF/historical notices and required change summary; third-party files require separate review. No endorsement. |
| 8. Domain allocation (train bytes) | code: 1500000; documentation: 1000000 |
| 9. Clean train/validation/test bytes | train: 2500000; validation: 125000; test: 125000 |
| 10. Download size | compressed wire bytes: 20520960; uncompressed wire bytes: 0; status: Official tar.xz; index reports exact compressed size. SHA-256 must be recorded and official signature/checksum provenance verified before parsing.; size source: https://www.python.org/ftp/python/3.12.10/; url: https://www.python.org/ftp/python/3.12.10/Python-3.12.10.tar.xz; sha256: None |
| 11. Decompressed size ESTIMATE | 100000000, 140000000 |
| 12. Retention ESTIMATE | eligible clean fraction of extracted: 0.15, 0.3; unique fraction of clean: 0.8, 0.95; status: Estimates before quota; exclusions and actual duplicate rates not yet measured. |
| 13. Selective/range access | Pinned files can be selected from official repository endpoints; proposed official tar.xz requires full 20,520,960-byte archive. Compressed range reads do not yield arbitrary members. |
| 14. Dedup risks | Cross-version copies, standard examples, headers/license boilerplate and shared vendor content. |
| 15. Benchmark contamination | Large overlap with public code completion examples; reject tests, benchmark suites and copied tutorial exercises. |
| 16. PII/privacy | Scan for credentials, private keys and personal contact details. Keep attribution separately; reject suspect documents, not just the matched substring. |
| 17. Secrets/security | Source/archive only; never execute configure, setup, builds, tests or install scripts. |
| 18. Cleaning | Allowlisted first-party Python/C/headers and selected RST prose only. Exclude vendor/third-party trees, tests, benchmark cases, generated tables/build outputs, copied exercises, installers and unresolved file headers. Keep indentation, line endings and notices; do not import or build. |
| 19. Classification | BOTH |
| 20. Human decision | APPROVE — Human-approved scoped source; mandatory content/license/provenance checks remain on actual retrieval, not a blanket license grant. |

## NumPy 2.2.4 source and selected documentation

| Field | Detail |
|---|---|
| 1. Source | NumPy 2.2.4 source and selected documentation |
| 2. Publisher | NumPy Developers |
| 3. Authoritative URL | [Source](https://github.com/numpy/numpy) |
| 4. Version/edition | tag: v2.2.4; commit: 3b377854e8b1a55f15bda6f1166fe9954828231b; license sha256: 01fb016849aa427edb1bbbbd55f91c26ca6cadb32a5b20e7f000655dd05b0760; license bytes: 1543 |
| 5. License/version | BSD 3-Clause; bundled/per-file notices may differ |
| 6. Planned-use eligibility | YES for verified files actually covered by the stated permissive license, with notices/conditions preserved; engineering eligibility assessment, not approval of every repository file or a guarantee about generated outputs. |
| 7. Attribution/redistribution | Retain copyright/license/disclaimer; no endorsement. Review vendored components separately and exclude them by default. |
| 8. Domain allocation (train bytes) | code: 750000; documentation: 250000 |
| 9. Clean train/validation/test bytes | train: 1000000; validation: 50000; test: 50000 |
| 10. Download size | compressed wire bytes: 20270701; uncompressed wire bytes: 0; status: Official PyPI sdist gzip; size and SHA-256 verified from metadata, body not fetched.; size source: https://pypi.org/pypi/numpy/2.2.4/json; filename: numpy-2.2.4.tar.gz; size bytes: 20270701; sha256: 9ba03692a45d3eef66559efe1d1096c4b9b75c0986b5dff5530c378fb8331d4f; url: https://files.pythonhosted.org/packages/e1/78/31103410a57bc2c2b93a3597340a8119588571f6a4539067546cb9a0bfac/numpy-2.2.4.tar.gz |
| 11. Decompressed size ESTIMATE | 70000000, 120000000 |
| 12. Retention ESTIMATE | eligible clean fraction of extracted: 0.05, 0.15; unique fraction of clean: 0.8, 0.95; status: Estimates before quota; exclusions and actual duplicate rates not yet measured. |
| 13. Selective/range access | Pinned files possible; proposed official gzip sdist requires full 20,270,701-byte archive. No selective compressed-member range claim. |
| 14. Dedup risks | Generated kernels, repeated docs/examples, code copied across numerical libraries. |
| 15. Benchmark contamination | Code benchmarks and duplicated examples; exclude tests, benchmarks and replicated API examples. |
| 16. PII/privacy | Scan for credentials, private keys and personal contact details. Keep attribution separately; reject suspect documents, not just the matched substring. |
| 17. Secrets/security | Do not run packaging hooks, Meson, code generators or package installation. |
| 18. Cleaning | Allowlisted first-party Python/C/Cython and RST prose only. Exclude bundled dependencies, generated/minified outputs, tests, benchmarks, build/install hooks and repeated API tables. Keep numerical notation/indentation; no package installation. |
| 19. Classification | BOTH |
| 20. Human decision | APPROVE — Human-approved scoped source; mandatory content/license/provenance checks remain on actual retrieval, not a blanket license grant. |

## SymPy 1.13.3 source and selected documentation

| Field | Detail |
|---|---|
| 1. Source | SymPy 1.13.3 source and selected documentation |
| 2. Publisher | SymPy Development Team |
| 3. Authoritative URL | [Source](https://github.com/sympy/sympy) |
| 4. Version/edition | tag: sympy-1.13.3; commit: b4ce69ad5d40e4e545614b6c76ca9b0be0b98f0b; license sha256: 07a5e9819f727b4986ad2829c7a29a6320d42575f720eb24d71b7fef573a0286; license bytes: 7885 |
| 5. License/version | BSD 3-Clause main license plus listed third-party notices |
| 6. Planned-use eligibility | YES for verified files actually covered by the stated permissive license, with notices/conditions preserved; engineering eligibility assessment, not approval of every repository file or a guarantee about generated outputs. |
| 7. Attribution/redistribution | Retain copyright/license/disclaimer and applicable bundled notices; no endorsement. Per-file scope review required. |
| 8. Domain allocation (train bytes) | code: 250000; documentation: 250000; structured math: 500000 |
| 9. Clean train/validation/test bytes | train: 1000000; validation: 50000; test: 50000 |
| 10. Download size | compressed wire bytes: 7533196; uncompressed wire bytes: 0; status: Official PyPI sdist gzip; size and SHA-256 verified from metadata, body not fetched.; size source: https://pypi.org/pypi/sympy/1.13.3/json; filename: sympy-1.13.3.tar.gz; size bytes: 7533196; sha256: b27fd2c6530e0ab39e275fc9b683895367e51d5da91baa8d3d64db2565fec4d9; url: https://files.pythonhosted.org/packages/11/8a/5a7fd6284fa8caac23a26c9ddf9c30485a48169344b4bd3b0f02fef1890f/sympy-1.13.3.tar.gz |
| 11. Decompressed size ESTIMATE | 35000000, 60000000 |
| 12. Retention ESTIMATE | eligible clean fraction of extracted: 0.2, 0.35; unique fraction of clean: 0.8, 0.95; status: Estimates before quota; exclusions and actual duplicate rates not yet measured. |
| 13. Selective/range access | Pinned files possible; proposed official gzip sdist requires full 7,533,196-byte archive. No selective compressed-member range claim. |
| 14. Dedup risks | Repeated algebra examples/answers, docstrings duplicated into manuals, copied algorithms. |
| 15. Benchmark contamination | Mathematical evaluation problems and answer strings may be embedded in tests/examples. Exclude tests/benchmarks and known problem identities. |
| 16. PII/privacy | Scan for credentials, private keys and personal contact details. Keep attribution separately; reject suspect documents, not just the matched substring. |
| 17. Secrets/security | No symbolic evaluation, doctest execution, imports or install scripts. |
| 18. Cleaning | Allowlisted first-party Python and selected explanatory/math prose only; disjoint code/doc/math subsets. Exclude tests, exercise-answer/problem banks, copied external examples, vendored files, generated tables and unresolved headers. No symbolic evaluation or doctests. |
| 19. Classification | BOTH |
| 20. Human decision | APPROVE — Human-approved scoped source; mandatory content/license/provenance checks remain on actual retrieval, not a blanket license grant. |

## mathlib4 v4.19.0 selected Lean files

| Field | Detail |
|---|---|
| 1. Source | mathlib4 v4.19.0 selected Lean files |
| 2. Publisher | Lean community |
| 3. Authoritative URL | [Source](https://github.com/leanprover-community/mathlib4) |
| 4. Version/edition | tag: v4.19.0; commit: c44e0c8ee63ca166450922a373c7409c5d26b00b; license sha256: b40930bbcf80744c86c46a12bc9da056641d722716c378f5659b9e555ef833e1; license bytes: 11357 |
| 5. License/version | Apache License 2.0 (January 2004); review per-file notices |
| 6. Planned-use eligibility | YES for verified files actually covered by the stated permissive license, with notices/conditions preserved; engineering eligibility assessment, not approval of every repository file or a guarantee about generated outputs. |
| 7. Attribution/redistribution | Keep LICENSE, applicable NOTICE, per-file attribution and modification records. |
| 8. Domain allocation (train bytes) | structured math: 500000 |
| 9. Clean train/validation/test bytes | train: 500000; validation: 25000; test: 25000 |
| 10. Download size | compressed wire bytes: 0; uncompressed wire bytes estimate: 2000000, 4000000; full archive compressed bytes: None; full archive compressed estimate: 15000000, 35000000; status: Selective transfer estimate; archive HEAD 200, Content-Length unavailable, ETag is not an archive checksum. No body fetched. |
| 11. Decompressed size ESTIMATE | 2000000, 4000000 |
| 12. Retention ESTIMATE | clean fraction of selected raw: 0.75, 0.95; unique fraction of clean: 0.8, 0.95; status: Unmeasured; full repository extraction could be 80-200 MB, not needed for selective method. |
| 13. Selective/range access | Select explicit .lean paths at pinned commit; no history clone or dependency fetch. Full archive/range fallback is not authorized. |
| 14. Dedup risks | Boilerplate proof patterns, renamed equivalent theorems and benchmark-mirrored proofs. |
| 15. Benchmark contamination | Formal-math benchmarks and mirrored proofs; exclude evaluation/problem collections and identify overlap with any chosen future proof benchmark before admission. |
| 16. PII/privacy | Scan author/contact metadata and secrets; retain license attribution separately. |
| 17. Secrets/security | No Lean/lake toolchain, dependency fetching, compilation or proof execution. |
| 18. Cleaning | Allowlisted first-party Lean source at pinned commit, with resolved per-file notices. Exclude benchmark/problem collections, downloaded dependencies, build outputs and autogenerated repetition. Preserve Unicode symbols; no lake/Lean execution. Group namespaces/proof provenance before split. |
| 19. Classification | TRAINING SOURCE |
| 20. Human decision | APPROVE — Human-approved scoped source; mandatory content/license/provenance checks remain on actual retrieval, not a blanket license grant. |

## Eligibility, evidence and boundary

Exact [Gutenberg work/edition records](../docs/gutenberg_work_allowlist_2m.json) contain catalog/header SHA-256 values. Whole-text and cleaned-payload SHA-256 are explicitly `PENDING_AUTHORIZED_ACQUISITION` and mandatory immediately after retrieval/extraction. #1260 is rejected; #1661 supplies the reserved validation work. Novel payloads exclude all supplemental material.

The [per-file allowlists](../docs/source_file_allowlists_2m.json) pin exact repository commit/blob identities and license provenance. Vendor, copied, generated, differently licensed or unresolved files cannot inherit permission silently from root licenses. Contradictory evidence on retrieval requires rejection.

All four benchmark fingerprint sets and downstream source-unit identities are registered in the [exclusion registry](../docs/data_exclusion_registry.json). The MATH publisher-linked and split-preserving artifacts are separately identified and matched exactly; no split recreation occurred. Fingerprint files remain ignored evaluation infrastructure; verified hashes are required before acquisition.

Current projected data/preprocessing plus evaluation infrastructure: 598284509 bytes; plus separate 200,000,000-byte run allowance. Hard cap: 800,000,000 data bytes +200,000,000 run bytes. No substantive source, full book, model training or GPU environment change occurred.

Acquisition order and required runtime checks are in the final authorization report. No unresolved authorization blocker remains; later file/hash/notice checks cannot be skipped.

DOWNLOAD AUTHORIZATION STATUS: READY
