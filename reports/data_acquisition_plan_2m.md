# First real-data acquisition proposal: 1.93M byte model

**Proposal only; mixture unselected.** Recommend **10,000,000 train bytes + 500,000 validation + 500,000 test**, subject to human review. No corpus archives/shards were downloaded, preprocessing or training started, or model/environment changed. Only source metadata, license pages and headers were inspected. This is a pilot of stable learning from heterogeneous real bytes, not final model quality or the 20M stage.

Companion artifacts: [machine-readable proposal](data_acquisition_plan_2m.json), [compact GPU closeout](gpu_enablement_closeout.md), [GPU measurements](gpu_enablement_closeout.json), [exclusion registry](../docs/data_exclusion_registry.json).

## Why start below the preferred 100–250 MB

The saved synthetic CUDA run processed **1,152 bytes in 5.2024273999995785 seconds** between updates 1 and 10, approximately **221.44 bytes/second** for updates 2–10. A linear extrapolation gives **12.54 hours per 10 MB pass**, or **125.44 hours per 100 MB pass**, before additional evaluation/data overhead. This instrumented short interval is not a sustained-throughput benchmark. Thermal behavior, real IO and different sequence sizes remain unmeasured.

The preferred 100–250 MB training target remains a later option. A separately reviewed expansion could use 100 MB train + 5 MB validation + 5 MB test, but its source inventory and retention must be re-estimated; these six sources may not meet every expanded quota. Do not repeat pilot documents or automatically multiply downloads. The first-stage hard ceiling is **500,000,000 cleaned usable bytes across all splits**. Decimal MB = 1,000,000 bytes throughout.

Target the validated **1,929,579-parameter model**, CUDA FP32, batch 2, sequence **32 bytes**, accumulation 2, CPU threads 1; deterministic algorithms/cuDNN, TF32/benchmarking off, cuBLAS workspace `:4096:8`. Keep the 1 GiB allocator cap and 1 GiB free headroom admission rule. No optimized Mamba/Triton kernels or mixed precision. The quadratic full-reference path remains unchanged. This is a proposed future training profile, not authorization to train.

## Three choices, none selected

Each row gives **percent; train bytes / validation bytes / test bytes**. Budgets are whole-group upper targets; report underfill rather than split a document family or add duplicates. Zero instruction/dialogue is intentional: it adds no necessary pilot gate.

| Domain | A: balanced | B: code-leaning | C: structured text |
|---|---|---|---|
| natural language | 55%; 5,500,000 / 275,000 / 275,000 | 35%; 3,500,000 / 175,000 / 175,000 | 45%; 4,500,000 / 225,000 / 225,000 |
| code | 25%; 2,500,000 / 125,000 / 125,000 | 40%; 4,000,000 / 200,000 / 200,000 | 15%; 1,500,000 / 75,000 / 75,000 |
| documentation | 15%; 1,500,000 / 75,000 / 75,000 | 20%; 2,000,000 / 100,000 / 100,000 | 20%; 2,000,000 / 100,000 / 100,000 |
| structured math | 5%; 500,000 / 25,000 / 25,000 | 5%; 500,000 / 25,000 / 25,000 | 20%; 2,000,000 / 100,000 / 100,000 |
| instruction dialogue | 0%; 0 / 0 / 0 | 0%; 0 / 0 / 0 | 0%; 0 / 0 / 0 |
| Total | 100%; 10,000,000 / 500,000 / 500,000 | 100%; 10,000,000 / 500,000 / 500,000 | 100%; 10,000,000 / 500,000 / 500,000 |

- **A — Balanced general-byte pilot:** Broad prose with moderate code/docs; historical/encyclopedic style may dominate.
- **B — Software/code-leaning pilot:** More software syntax and documentation; less general prose and no claim of code-task competence.
- **C — Structured-text pilot:** More mathematical/formal structure, with risk of poor language transfer; this is not a reasoning benchmark.

Proposed **post-dedup unique bytes across all three splits**, by source. These are alternative allocations; do not add columns. The JSON also records disjoint source/domain training allocations. For example, SymPy code, documentation and structured-math subsets cannot be counted twice.

| Source | A | B | C |
|---|---:|---:|---:|
| gutenberg | 3,630,000 | 2,310,000 | 2,970,000 |
| wikipedia | 2,420,000 | 1,540,000 | 1,980,000 |
| cpython | 2,750,000 | 3,960,000 | 2,310,000 |
| numpy | 1,100,000 | 1,760,000 | 935,000 |
| sympy | 825,000 | 1,155,000 | 1,265,000 |
| mathlib | 275,000 | 275,000 | 1,540,000 |
| Total | 11,000,000 | 11,000,000 | 11,000,000 |

## Candidate source review

Official metadata/license pages were checked on **2026-09-09**; no corpus/archive body was downloaded. Archive sizes below are publisher metadata. Complete uncompressed collection sizes are **UNKNOWN**; repository Git storage is not corpus size. Extraction, retention, quality and eligible-yield estimates require later sample validation. Exact download URLs and license-file SHA-256 values are preserved in the companion JSON.

### Project Gutenberg: selected English books

Publisher: **Project Gutenberg Literary Archive Foundation / volunteers**; [authoritative source](https://www.gutenberg.org/). Format/domain: plain text; English, natural-language prose. Choose ebook ID + edition/update date + raw SHA-256 before acquisition; not yet selected. Metadata example only: Flatland #201, updated 2022-06-26.

**License:** Work-specific public-domain status (US); Project Gutenberg License/trademark terms, as embedded in each book. US public-domain labels do not establish rights in India or other jurisdictions. Reject permission-only works unless separately cleared. Human review must approve jurisdiction, commercial use and removal/retention of Gutenberg marks and license; do not equate free access with unrestricted worldwide reuse. Retain complete raw license and ebook/author/title/source metadata. Any boilerplate removal must satisfy the particular ebook license and trademark conditions.

**Byte estimate:** Select 4–6 MB of plain-text books; compressed archive bytes 0 under the proposed method. Record any HTTP Content-Encoding separately. Decoded/extracted bytes: **4,000,000–6,000,000**. Eligible cleaned inventory before quota: **3,000,000–5,000,000**. Estimated retention: clean fraction of selected raw **85–95%**; unique fraction of clean **70–90%**. The source/split quota table determines what is retained; inventory is not an instruction to consume all eligible bytes. These ranges can underfill after filtering and dedup.

**Access and fit:** Use sanctioned robot/harvest or an authorized mirror with requested wait/rate rules; no scraping the main human website. Select books individually. Long-form edited prose; historical language and social bias, weak modern dialogue/technical coverage. Small natural-language anchor, subject to rights clearance.

**Contamination/privacy/security:** Classics, school texts, translations and alternate editions often recur in evaluations. Deny registered benchmark works and cluster editions/translations by work. Avoid modern personal records. Historical/fictional names are not indiscriminately stripped. Country-specific copyright review remains required. Plain text only; reject embedded active content and misleading formats.

**BOTH WITH TRACKED PROVENANCE**: selected prose for training; canonical literary reference could support future RAG.

Primary references: [Gutenberg license policy](https://www.gutenberg.org/policy/license.html), [Authorized robot access](https://www.gutenberg.org/policy/robot_access.html), [Example book metadata (#201)](https://www.gutenberg.org/ebooks/201).

### English Wikipedia: selected stable article revisions

Publisher: **Wikimedia contributors / Wikimedia Foundation**; [authoritative source](https://en.wikipedia.org/). Format/domain: Action API JSON with wikitext extraction; English, encyclopedic natural-language text. Freeze page IDs + revision IDs + timestamps + retrieved SHA-256. Article selection pending; do not bind to a moving latest revision.

**License:** CC BY-SA 4.0 default text reuse; GFDL alternative where applicable; imported material may differ. Commercial reuse is permitted under applicable license conditions. Human review must approve attribution/share-alike handling, training/output implications and downstream redistribution policy. Exclude fair-use material and imported text whose additional terms cannot be resolved. Retain page ID, revision ID, article/history URL, applicable license and change record; preserve required imported-content notices. No images.

**Byte estimate:** Select API responses estimated at 4–8 MB decoded JSON; compressed wire bytes are UNKNOWN until measured. Decoded/extracted bytes: **4,000,000–8,000,000**. Eligible cleaned inventory before quota: **2,000,000–4,000,000**. Estimated retention: clean fraction of decoded response **35–60%**; unique fraction of clean **80–95%**. The source/split quota table determines what is retained; inventory is not an instruction to consume all eligible bytes. These ranges can underfill after filtering and dedup.

**Access and fit:** Official revision API supports selective retrieval; respect identification/rate limits/Retry-After, one connection. Full dumps are unnecessary; current dump service is moving from XML to MediaWiki Content File Exports. Broad factual prose, uneven article quality, templating and version-sensitive facts; use a curated article allowlist. Diversity beyond historical books, conditional on license decision.

**Contamination/privacy/security:** Wikipedia-derived benchmarks and mirrored/revised articles can leak. Group page/redirect/revision families; exclude registered evaluation identities. Exclude biographies of living people and personal contact/user/talk pages for this pilot. Treat markup and links as data; do not follow links, render active content or execute templates.

**BOTH WITH TRACKED PROVENANCE**: selected prose for training; frequently changing facts are better future **RAG CANDIDATE** material.

Primary references: [Wikimedia terms of use](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use), [Revision API](https://www.mediawiki.org/wiki/API:Revisions), [Official dump service](https://dumps.wikimedia.org/).

### CPython 3.12.10 source and selected documentation

Publisher: **Python Software Foundation**; [authoritative source](https://github.com/python/cpython). Format/domain: source text / RST; archive manifest inspected before extraction; English, Python/C source, headers, selected English RST tutorials. `v3.12.10` at commit `0cc81280367df838c4b199f8f0378837165071c2`. License file SHA-256: `3b2f81fe21d181c499c59a256c8e1968455d6689d269aa85373bfb6af41da3bf`. These historical pins are intentional reproducibility choices.

**License:** PSF License Version 2 plus historical/per-file notices; documentation examples additionally offer 0BSD. Permissive commercial/research reuse under stated notices; this is not a blanket clearance of bundled files. Review selected per-file licenses and combined-corpus distribution before acquisition. Retain PSF/historical notices and required change summary; third-party files require separate review. No endorsement.

**Byte estimate:** Official tar.xz: **20,520,960 compressed bytes**; no separate uncompressed transfer. Archive SHA-256/signature verification remains required before parsing. Decoded/extracted bytes: **100,000,000–140,000,000**. Eligible cleaned inventory before quota: **20,000,000–35,000,000**. Estimated retention: eligible clean fraction of extracted **15–30%**; unique fraction of clean **80–95%**. The source/split quota table determines what is retained; inventory is not an instruction to consume all eligible bytes. These ranges can underfill after filtering and dedup.

**Access and fit:** Pinned release archive is small enough for a later bounded download. Selective commit/path retrieval is possible; an approved acquisition method must be frozen before execution. High-quality real source and prose, but generated tables, tests and vendor code dominate some regions. Authentic compact programming and technical structure; no need to install these packages.

**Contamination/privacy/security:** Large overlap with public code completion examples; reject tests, benchmark suites and copied tutorial exercises. Scan for credentials, private keys and personal contact details. Keep attribution separately; reject suspect documents, not just the matched substring. Source/archive only; never execute configure, setup, builds, tests or install scripts.

**BOTH WITH TRACKED PROVENANCE**: source/tutorial prose for training; version-specific API reference tables are **RAG CANDIDATE**.

Primary references: [Pinned license](https://raw.githubusercontent.com/python/cpython/0cc81280367df838c4b199f8f0378837165071c2/LICENSE), [Official archive index](https://www.python.org/ftp/python/3.12.10/), [Pinned source tree](https://github.com/python/cpython/tree/0cc81280367df838c4b199f8f0378837165071c2).

### NumPy 2.2.4 source and selected documentation

Publisher: **NumPy Developers**; [authoritative source](https://github.com/numpy/numpy). Format/domain: source text / RST; archive manifest inspected before extraction; English, Python/C/Cython numerical code and English RST. `v2.2.4` at commit `3b377854e8b1a55f15bda6f1166fe9954828231b`. License file SHA-256: `01fb016849aa427edb1bbbbd55f91c26ca6cadb32a5b20e7f000655dd05b0760`. These historical pins are intentional reproducibility choices.

**License:** BSD 3-Clause; bundled/per-file notices may differ. Permissive commercial/research reuse under stated notices; this is not a blanket clearance of bundled files. Review selected per-file licenses and combined-corpus distribution before acquisition. Retain copyright/license/disclaimer; no endorsement. Review vendored components separately and exclude them by default.

**Byte estimate:** Official PyPI tar.gz: **20,270,701 compressed bytes**; no separate uncompressed transfer. Publisher metadata SHA-256: `9ba03692a45d3eef66559efe1d1096c4b9b75c0986b5dff5530c378fb8331d4f`. Decoded/extracted bytes: **70,000,000–120,000,000**. Eligible cleaned inventory before quota: **5,000,000–12,000,000**. Estimated retention: eligible clean fraction of extracted **5–15%**; unique fraction of clean **80–95%**. The source/split quota table determines what is retained; inventory is not an instruction to consume all eligible bytes. These ranges can underfill after filtering and dedup.

**Access and fit:** Pinned release archive is small enough for a later bounded download. Selective commit/path retrieval is possible; an approved acquisition method must be frozen before execution. Idiomatic numerical software; generated/vendor/build content needs strict exclusion. Authentic compact programming and technical structure; no need to install these packages.

**Contamination/privacy/security:** Code benchmarks and duplicated examples; exclude tests, benchmarks and replicated API examples. Scan for credentials, private keys and personal contact details. Keep attribution separately; reject suspect documents, not just the matched substring. Do not run packaging hooks, Meson, code generators or package installation.

**BOTH WITH TRACKED PROVENANCE**: source/tutorial prose for training; version-specific API reference tables are **RAG CANDIDATE**.

Primary references: [Pinned license](https://raw.githubusercontent.com/numpy/numpy/3b377854e8b1a55f15bda6f1166fe9954828231b/LICENSE.txt), [Official sdist metadata](https://pypi.org/pypi/numpy/2.2.4/json), [Pinned source tree](https://github.com/numpy/numpy/tree/3b377854e8b1a55f15bda6f1166fe9954828231b).

### SymPy 1.13.3 source and selected documentation

Publisher: **SymPy Development Team**; [authoritative source](https://github.com/sympy/sympy). Format/domain: source text / RST; archive manifest inspected before extraction; English, Python mathematical code, English RST and mathematical expressions. `sympy-1.13.3` at commit `b4ce69ad5d40e4e545614b6c76ca9b0be0b98f0b`. License file SHA-256: `07a5e9819f727b4986ad2829c7a29a6320d42575f720eb24d71b7fef573a0286`. These historical pins are intentional reproducibility choices.

**License:** BSD 3-Clause main license plus listed third-party notices. Permissive commercial/research reuse under stated notices; this is not a blanket clearance of bundled files. Review selected per-file licenses and combined-corpus distribution before acquisition. Retain copyright/license/disclaimer and applicable bundled notices; no endorsement. Per-file scope review required.

**Byte estimate:** Official PyPI tar.gz: **7,533,196 compressed bytes**; no separate uncompressed transfer. Publisher metadata SHA-256: `b27fd2c6530e0ab39e275fc9b683895367e51d5da91baa8d3d64db2565fec4d9`. Decoded/extracted bytes: **35,000,000–60,000,000**. Eligible cleaned inventory before quota: **8,000,000–20,000,000**. Estimated retention: eligible clean fraction of extracted **20–35%**; unique fraction of clean **80–95%**. The source/split quota table determines what is retained; inventory is not an instruction to consume all eligible bytes. These ranges can underfill after filtering and dedup.

**Access and fit:** Pinned release archive is small enough for a later bounded download. Selective commit/path retrieval is possible; an approved acquisition method must be frozen before execution. Useful symbolic expression structure; avoid repetitive test answers, generated documentation and over-weighted formulas. Authentic compact programming and technical structure; no need to install these packages.

**Contamination/privacy/security:** Mathematical evaluation problems and answer strings may be embedded in tests/examples. Exclude tests/benchmarks and known problem identities. Scan for credentials, private keys and personal contact details. Keep attribution separately; reject suspect documents, not just the matched substring. No symbolic evaluation, doctest execution, imports or install scripts.

**BOTH WITH TRACKED PROVENANCE**: source and mathematical prose for training; exhaustive API reference is **RAG CANDIDATE**.

Primary references: [Pinned license](https://raw.githubusercontent.com/sympy/sympy/b4ce69ad5d40e4e545614b6c76ca9b0be0b98f0b/LICENSE), [Official sdist metadata](https://pypi.org/pypi/sympy/1.13.3/json), [Pinned source tree](https://github.com/sympy/sympy/tree/b4ce69ad5d40e4e545614b6c76ca9b0be0b98f0b).

### mathlib4 v4.19.0 selected Lean files

Publisher: **Lean community**; [authoritative source](https://github.com/leanprover-community/mathlib4). Format/domain: .lean source text; Lean, Unicode mathematical/structured text, English comments. `v4.19.0` at commit `c44e0c8ee63ca166450922a373c7409c5d26b00b`. License file SHA-256: `b40930bbcf80744c86c46a12bc9da056641d722716c378f5659b9e555ef833e1`. These historical pins are intentional reproducibility choices.

**License:** Apache License 2.0 (January 2004); review per-file notices. Permissive commercial/research use subject to license/NOTICE/change requirements and patent terms. Do not infer trademark endorsement. Keep LICENSE, applicable NOTICE, per-file attribution and modification records.

**Byte estimate:** Select 2–4 MB of plain files; compressed archive bytes 0 under the proposed method. Full-archive HEAD succeeded, but Content-Length is UNKNOWN; its ETag is not a checksum. A 15–35 MB archive estimate is informational only; that fallback is not selected. Decoded/extracted bytes: **2,000,000–4,000,000**. Eligible cleaned inventory before quota: **1,500,000–3,500,000**. Estimated retention: clean fraction of selected raw **75–95%**; unique fraction of clean **80–95%**. The source/split quota table determines what is retained; inventory is not an instruction to consume all eligible bytes. These ranges can underfill after filtering and dedup.

**Access and fit:** Select namespace/file allowlist at pinned commit via official repository endpoints. Bound metadata traversal; never clone history. Full-archive fallback needs separate approval. Precisely structured definitions/proofs; differs from natural-language reasoning and may consume byte capacity disproportionately. Small controlled structured-text component; not evidence of reasoning ability.

**Contamination/privacy/security:** Formal-math benchmarks and mirrored proofs; exclude evaluation/problem collections and identify overlap with any chosen future proof benchmark before admission. Scan author/contact metadata and secrets; retain license attribution separately. No Lean/lake toolchain, dependency fetching, compilation or proof execution.

**TRAINING SOURCE** for a small structured-byte component; no claim of learned proof capability.

Primary references: [Pinned license](https://raw.githubusercontent.com/leanprover-community/mathlib4/c44e0c8ee63ca166450922a373c7409c5d26b00b/LICENSE), [Pinned source tree](https://github.com/leanprover-community/mathlib4/tree/c44e0c8ee63ca166450922a373c7409c5d26b00b).

## Byte accounting and storage

Keep wire-compressed bytes, wire-uncompressed bytes, decoded/extracted bytes, cleaned pre-dedup bytes and unique post-dedup bytes as different counters. Do not add decoded JSON or extracted tar bytes to wire transfers. The three release archives total **48,324,857 compressed bytes**; selected Gutenberg/plain mathlib transfers are estimated at **6–10 MB combined**, and Wikipedia responses at **4–8 MB decoded**, with wire compression unknown. A conservative transfer allowance is 80 MB. No substantial payload has been fetched.

Expected selected pre-dedup cleaned corpus: **12–16 MB**; desired unique result **11 MB**, comprising 10 MB train, 0.5 MB validation and 0.5 MB test. Source filtering and global duplicate rates may force underfill. Record each reduction separately; do not claim these target ratios have been measured.

| Storage component | Projected bytes |
|---|---:|
| retained compressed archives | 60,000,000 |
| retained plain sources and api responses | 20,000,000 |
| temporary extraction one archive | 250,000,000 |
| cleaned before dedup | 20,000,000 |
| unique train | 10,000,000 |
| unique validation | 500,000 |
| unique test | 500,000 |
| manifests indexes notices | 5,000,000 |
| preprocessing scratch | 50,000,000 |
| Data working total | **416,000,000** |
| Checkpoints / run artifacts, separate | **200,000,000** |
| Combined projection | **616,000,000** |

The projection includes temporary and retained copies conservatively; validation/test are already included as unique split bytes. Checkpoint allowance: three at 32 MB each (96 MB), 20 MB metrics/manifests and 84 MB reserve. These are model-run storage, not training-data bytes. Process only one archive at a time, retaining provenance before deleting temporary extracted copies.

Hard review stops: **80 MB transfer**, **800 MB data working footprint**, and **11 MB selected unique pilot bytes**. A separately reviewed 100 MB expansion would need a fresh component estimate and remain below **2 GB data + 250 MB run artifacts**. The 40 GB long-term allowance is not allocated.

## Proposed pipeline — design, not implementation

1. **source acquisition:** After human approval, freeze source allowlist, exact revisions/releases, method and byte caps. Download through official endpoints with rate limits, retry bounds and descriptive user agent. No arbitrary crawling or automatic source substitutions.

2. **checksum/provenance verification:** Record URL, UTC retrieval, compressed transfer count, encoding, publisher checksum/signature if provided, and raw SHA-256. A checksum created after download proves identity, not publisher authenticity. A checksum/commit mismatch stops the source.

3. **safe decompression:** Inspect every archive entry before extraction; reject absolute/drive/parent paths, links, devices, nested archives and destination escapes. Limits: 50,000 entries, 8 MB per accepted text member, 250 MB expanded archive, ratio 200:1; reject and review exceptions. Process one archive at a time. Never execute downloaded code, installer or build scripts.

4. **format extraction:** Select source-specific paths and document boundaries. Extract API text fields using an inert parser. Record JSON unescaping/markup removal or any transcoding explicitly with input/output hashes and offsets where available. No Unicode normalization, lowercasing, silent replacement decoding or generic whitespace/newline rewriting. Expected UTF-8 failures are quarantined.

5. **license and source metadata:** Retain immutable original bytes and LICENSE/NOTICE sidecars. Attach per-document license, revision and authorship/provenance. Reject unresolved permissions. Remove repeated license/navigation boilerplate from training text only with a reviewed source-specific rule, retained notices and transformation log.

6. **binary/text validity:** Text-only source allowlist for this pilot; reject executables, binary media and disguised archives by signature, not extension alone. This selection policy does not change the byte model: arbitrary bytes, NUL, high bytes and invalid UTF-8 remain supported and covered by excluded local mechanics fixtures. Raw source bytes are never repaired silently.

7. **secrets/credentials:** Scan private-key blocks, recognizable token/API-key patterns and suspicious high-entropy strings with contextual review. Reject suspect whole documents and record only hash/reason/count, not secret values. No verification requests to external services.

8. **PII-sensitive filtering:** Exclude personal contact/user pages and living-person biographies; review contact details in code/docs. Separate legally needed public attribution from training payload. No indiscriminate erasure of fictional names, identifiers or mathematical constants.

9. **malware and generated content:** Reject executable payloads, obfuscated/minified blobs, malware/exploit collections, vendored dependencies, generated tables/build outputs and install scripts. Content is inert throughout; no imports, tests, proof execution or package installs. Pattern scans are not a guarantee of benign content.

10. **quality filtering:** Apply auditable source/domain thresholds for empty/boilerplate/repetitive content and markup residue. Do not apply prose heuristics to code or Lean. Preserve indentation and Unicode mathematical notation. Inspect bounded stratified samples after approval; quality and retention remain unmeasured now.

11. **exact dedup:** SHA-256 of cleaned document bytes globally across all sources; retain provenance for duplicate origins and select one canonical payload. Full raw payload hashes remain separate. Exclude every local fixture identity/payload before admission.

12. **near dedup:** Propose byte-shingle MinHash: 64-byte shingles, 128 permutations, seed 17, candidate Jaccard >=0.85; verify matched candidates against actual shingles. A normalized whitespace detection view may supplement exact bytes but never rewrites training payloads. Calibrate false positives on code licenses/short math before freezing; thresholds are unvalidated proposals. Connected duplicate clusters must stay in one split.

13. **benchmark/evaluation exclusion:** Use docs/data_exclusion_registry.json identity denylist plus later licensed content fingerprints. Match exact documents and sufficiently long shared spans (initial 64-byte threshold, review short-answer false positives), inspect candidates, quarantine matches from all splits. Content indexes and downstream holdout IDs must be frozen before training manifest admission. Hashes cannot prove absence of paraphrased contamination.

14. **split assignment:** Assign source/work/document families and duplicate clusters before windows, with deterministic seeded hash ordering and atomic group quotas. Prefer whole repositories/work/author families. If repo counts cannot support quotas, require explicit review for package/namespace-level holdouts; label same-repository validation accordingly. Never split revisions, neighboring regions or duplicate clusters across train/validation/test.

15. **immutable manifest:** Freeze provenance, exclusion registry, pipeline rules, identities and byte counts. Produce compatible train/validation projection for the existing schema-1 trainer only after validation. Keep test data in a separate sealed index and never use it for tuning.

16. **byte statistics:** Report downloaded compressed and uncompressed bytes separately, decoded/extracted bytes, pre-dedup cleaned bytes, rejected bytes by reason, post-dedup unique bytes, per-split/domain/source counts, document/group counts, lengths and byte-frequency/control-ID checks. No token counts needed. Count raw-byte targets in [0,255] only.

## Splits and exclusion registry

- **seed:** 17
- **hash order:** SHA-256(corpus ID, split seed, stable group ID); algorithm and encoding frozen before implementation.
- **gutenberg group:** work/edition/translation family, preferably author
- **wikipedia group:** page/redirect/revision family
- **code group:** repository preferred; module/package holdout only after explicit review, all versions/path descendants grouped
- **mathlib group:** namespace dependency/provenance family; same-library holdout is not cross-library generalization
- **windows:** Generate bounded windows only after split. No shared source byte region across splits; retain source offsets. Do not join unrelated documents without explicit boundaries.
- **budget semantics:** Whole-group upper targets; report actual underfill rather than cut groups across splits or duplicate material. Source/domain quotas are mutually exclusive.
- **evaluation:** Use valid-target-weighted raw-byte NLL/perplexity and per-domain metrics. 500k bytes per holdout is a proposal, not a guarantee of statistical independence; report group counts and uncertainty. Test remains sealed.

The [exclusion registry](../docs/data_exclusion_registry.json) contains SHA-256 identities for project tests/generators and actual CPU/GPU synthetic fixture payloads. Entire project source, tests, generated responses, reports and evidence are denied as corpus sources. Generator hashes identify recipes, not every randomly generated sequence. Do not globally blacklist single byte values or common short literals.

Future benchmark exclusions pin repository commits and paths for **HumanEval**, **GSM8K**, **MATH** and **MBPP**. Benchmark bodies were not downloaded; content/near-duplicate fingerprint indexes are explicitly pending and must be obtained under approved access/terms before training admission. Planned downstream validation/test has reserved identity `edge400-real-byte-pilot-v1-heldout`, but its source groups/hashes await mixture selection. If formal-math evaluation is planned, add its exact benchmark identities before admitting Lean data. The registry alone does not implement filtering or prove absence of paraphrase contamination.

## Provenance contract

Design only; proposed provenance schema version 1, separate from existing trainer DatasetManifest schema 1. No trainer contract changed.

Per-document fields: `schema_version`, `corpus_id`, `source_id`, `publisher`, `authoritative_url`, `release_tag`, `commit_or_revision_id`, `retrieved_utc`, `license_id_version`, `license_notice_relative_paths`, `raw_relative_path`, `raw_sha256`, `wire_compressed_bytes`, `wire_uncompressed_bytes`, `decoded_extracted_bytes`, `clean_relative_path`, `clean_sha256`, `clean_byte_count`, `document_count`, `domain`, `language`, `stable_group_id`, `duplicate_cluster_id`, `split`, `source_offsets`, `transformation_rule_version_and_hash`, `filter_counts_by_reason`, `exclusion_registry_sha256`, `benchmark_index_sha256`, `pipeline_code_commit`, `selection_seed`, `unique_bytes`, `parent_manifest_sha256`.

All file paths relative to configured data root, no credentials or machine-specific absolute paths. URLs carry no access tokens.

Canonical JSON SHA-256, atomic publication to unused corpus ID, source/clean file re-hash on consumption. Record retrieval metadata in append-only provenance; never mutate an admitted manifest.

Later implement and test a projection to existing train/validation schema-1 manifest. Bind its hash and parent provenance hash in run metadata; test index separate. Existing byte preservation and mutation detection must remain. This adapter and corpus preprocessing are not implemented by this proposal.

## Acquisition order and stop conditions

1. Human selects mixture/budget, jurisdiction/license policy and evaluation identities.
2. Freeze per-file licenses, release/revision allowlists, checksums and retention/transfer caps.
3. After separate authorization, pilot a small CPython allowlist, then NumPy/SymPy and measure actual yield/security exclusions; do not execute packages.
4. Acquire Gutenberg/Wikipedia only after their specific rights and attribution decisions; one source at a time.
5. Acquire mathlib selected files only up to the chosen mixture quota; archive fallback requires separate approval.
6. Calibrate filtering/dedup, freeze benchmark/holdout fingerprints, assign groups, then freeze manifests.
7. Review actual storage/yield and a separately authorized bounded real-byte learning plan before training or 100 MB expansion.

- No human source/mixture/license approval: no acquisition.
- Unavailable identity, changed bytes, checksum mismatch, unresolved rights or prohibited access: quarantine source and stop; no silent substitute.
- Archive traversal/link/device/size/ratio violation, discovered credential or active payload: quarantine without execution and review.
- Pilot transfer >80 MB, data working footprint >800 MB, selected unique bytes >11 MB or any cleaned usable total >500 MB: stop before exceeding cap.
- Insufficient eligible unique groups, poor retention or unexpected benchmark overlap: report underfill and request revised plan; no repetition to fill quota.
- Benchmark exclusion indexes or held-out identities missing: no frozen training admission.
- No sustained throughput/thermal evidence: do not scale data or sequence length based solely on free VRAM.

## Human decisions still required

- Choose A, B or C; no mixture has been selected.
- Approve 10 MB train +0.5 MB validation +0.5 MB test, or request a revised larger pilot; 100-250 MB preference is deferred for measured throughput reasons.
- Resolve relevant jurisdiction/commercial/redistribution policy, Gutenberg work eligibility and Wikimedia attribution/share-alike/training treatment. If a source is rejected, revise mixture explicitly.
- Approve pinned releases, source subsets and license allowlist; resolve archive checksum and per-file exceptions before download.
- Choose downstream evaluation identities; approve benchmark reference access/fingerprint indexing and stricter formal-math exclusions if that domain will be evaluated.
- Approve whole-repository holdouts or explicitly accept reviewed module/namespace holdouts and their limited interpretation.
- Review near-dedup thresholds on later small licensed samples and approve retention/storage caps.

## Validation, rollback and next step

This stage adds `reports/gpu_enablement_closeout.md/.json`, `reports/data_acquisition_plan_2m.md/.json` and `docs/data_exclusion_registry.json`, and updates `docs/project_state.md`. Existing GPU runtime changes were already present and are not new implementation in this stage.

Document checks cover JSON parsing, 100% mixture sums, source/domain/split byte totals, storage arithmetic, local exclusion hashes, exact GPU-summary agreement with saved evidence, and preservation of tested source/lock/specification hashes. GPU/CPU experiments are reused, not rerun for a documentation proposal.

**Not validated:** actual source acquisition/extraction/retention, per-file rights clearance, benchmark content indexes, final holdout groups, filtering/manifest adapter implementation, sustained real-data performance or learning quality. These are explicit next-stage gates, not missing GPU measurements.

Existing model, environments, GPU evidence and accepted Git commits preserved. This stage adds documentation only. No new Git commit or remote.

Human review of mixture, smaller budget, source rights and holdout policy; only then separately authorize bounded acquisition.

DATA ACQUISITION PLAN STATUS: READY FOR HUMAN REVIEW
