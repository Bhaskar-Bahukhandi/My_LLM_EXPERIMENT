# Download authorization — first 1.93M byte pilot

**READY for the next separately authorized acquisition pass. No substantive training corpus has been downloaded.** Wikipedia is deferred with zero bytes in all splits. Model, CPU/CUDA environments, recurrence and Git rollback commits remain unchanged.

Authoritative details: [source table](source_approval_table_2m.json), [Gutenberg work/edition allowlist](../docs/gutenberg_work_allowlist_2m.json), [exact repository file allowlists](../docs/source_file_allowlists_2m.json), [benchmark coverage](benchmark_exclusion_coverage.json), [exclusion registry](../docs/data_exclusion_registry.json).

## Approved sources and exact budgets

| Source | Train bytes | Validation bytes | Test bytes |
|---|---:|---:|---:|
| gutenberg | 5,000,000 | 250,000 | 250,000 |
| cpython | 2,500,000 | 125,000 | 125,000 |
| numpy | 1,000,000 | 50,000 | 50,000 |
| sympy | 1,000,000 | 50,000 | 50,000 |
| mathlib | 500,000 | 25,000 | 25,000 |
| Total | **10,000,000** | **500,000** | **500,000** |

Domain balance remains 50% general text /25% code /15% technical-scientific documentation /10% structured mathematics. Exact file-level categories are recorded in the allowlist; SymPy code, docs and structured text are disjoint. Quantities are post-cleaning targets, not measured yield. Report underfill rather than duplicate data or share heldout regions.

## Exact Gutenberg work screening

Original works must be English, first published by 1930 and published during their identified author’s lifetime; last original author death by 1965. All selected originals satisfy the stricter dates below. The operating jurisdiction screen is India (Copyright Act section 22) plus Gutenberg’s US catalog status; no worldwide or supplement clearance is inferred. Later reprint dates are not posthumous original works. [India Code](https://www.indiacode.nic.in/bitstream/123456789/1367/1/a195714.pdf), [Gutenberg policy](https://www.gutenberg.org/policy/license.html).

| Ebook | Work | Author death | First publication / collected book | Role | Catalog file bytes | Decision |
|---:|---|---:|---|---|---:|---|
| 1400 | [Great Expectations](https://www.gutenberg.org/cache/epub/1400/pg1400.rdf) | 1870 | 1860 / 1861 | train | 1,038,512 | APPROVE_ORIGINAL_TEXT_ONLY |
| 766 | [David Copperfield](https://www.gutenberg.org/cache/epub/766/pg766.rdf) | 1870 | 1849 / 1850 | train | 2,013,603 | APPROVE_ORIGINAL_TEXT_ONLY |
| 1023 | [Bleak House](https://www.gutenberg.org/cache/epub/1023/pg1023.rdf) | 1870 | 1852 / 1853 | train | 2,025,044 | APPROVE_ORIGINAL_TEXT_ONLY |
| 145 | [Middlemarch](https://www.gutenberg.org/cache/epub/145/pg145.rdf) | 1880 | 1871 / 1872 | train | 1,812,793 | APPROVE_ORIGINAL_TEXT_ONLY |
| 1342 | [Pride and Prejudice](https://www.gutenberg.org/cache/epub/1342/pg1342.rdf) | 1817 | 1813 / 1813 | train | 738,046 | APPROVE_ORIGINAL_TEXT_ONLY |
| 158 | [Emma](https://www.gutenberg.org/cache/epub/158/pg158.rdf) | 1817 | 1815 / 1815 | train | 897,582 | APPROVE_ORIGINAL_TEXT_ONLY |
| 1260 | [Jane Eyre: An Autobiography](https://www.gutenberg.org/cache/epub/1260/pg1260.rdf) | 1855 | 1847 / 1847 | excluded | 1,044,063 | REJECT_EDITION |
| 768 | [Wuthering Heights](https://www.gutenberg.org/cache/epub/768/pg768.rdf) | 1848 | 1847 / 1847 | test | 661,813 | APPROVE_ORIGINAL_TEXT_ONLY |
| 1661 | [The Adventures of Sherlock Holmes](https://www.gutenberg.org/cache/epub/1661/pg1661.rdf) | 1930 | 1891 / 1892 | validation | 607,504 | APPROVE_ORIGINAL_TEXT_ONLY |

The JSON records every exact text-file URL, catalog modification/size, catalog SHA-256, header SHA-256 and bibliographic evidence URL. All nine inspected headers used mandatory HTTP 206 byte ranges, 4,096 bytes each; **36,864 header bytes**, no full book. The ibiblio mirror returned 403, recorded as failed; catalog-linked direct text-file ranges succeeded. This was not human-page scraping.

**#1260 Jane Eyre is rejected** because its 1897 edition header separately reserves illustration copyright. It is replaced by #1661 for validation. #1342 includes Saintsbury’s preface and Thomson’s illustrations: only Austen’s novel chapters are eligible. Exclude every illustration/caption, editorial preface, annotation and supplement across all books. Recheck the full notice at actual acquisition and reject item-specific permission/restriction notices.

Eight accepted candidates total **9,794,897 catalog bytes**. Six training candidates total **8,525,580 bytes**; estimated 70–90% eligible unique retention gives **5,967,906–7,673,022 bytes**, above the 5 MB target at the conservative estimate. Doyle/Emily Bronte heldout works each exceed 250k bytes after the same estimate. Retention and exact extraction boundaries remain unmeasured.

**Whole-text SHA-256: `PENDING_AUTHORIZED_ACQUISITION`. Cleaned-payload SHA-256: `PENDING_AUTHORIZED_ACQUISITION`.** Compute the former immediately after each authorized file download and the latter immediately after extraction. Metadata/header hashes are not substitutes. Preserve original bytes/notices outside the payload. Any changed identity or uncertain supplemental boundary stops that item.

## Per-file repository eligibility and heldouts

| Source | Pinned commit | Candidate files | Reserved validation / test unit |
|---|---|---:|---|
| cpython | `0cc81280367df838c4b199f8f0378837165071c2` | 665 | email / urllib |
| numpy | `3b377854e8b1a55f15bda6f1166fe9954828231b` | 84 | fft / polynomial |
| sympy | `b4ce69ad5d40e4e545614b6c76ca9b0be0b98f0b` | 1059 | geometry / combinatorics |
| mathlib | `c44e0c8ee63ca166450922a373c7409c5d26b00b` | 1890 | Order / Topology |

Pins remain CPython 3.12.10, NumPy 2.2.4, SymPy 1.13.3 and mathlib4 v4.19.0. Every exact candidate file has a Git blob SHA-1, metadata size, domain, reserved source-unit role and controlling license hash. Git SHA-1 identifies the pinned repository blob; it is not mislabeled as a downloaded-file SHA-256.

Human approval applies only to verified first-party material under its actual per-file license. Vendored/external/copied/generated/test material and nested alternate-license scopes are excluded from the candidate lists. Any contradictory header/provenance discovered on retrieval means rejection; root licenses cannot override it. Keep LICENSE/NOTICE/change and attribution records outside payload. Never execute repository code, installers, tests or Lean proofs.

Reserved roles name source units now; global dedup precedes final split/window generation. A duplicate cluster spanning roles is excluded instead of moving heldout bytes into training. Unused Doyle/Emily Bronte work bytes also remain unavailable to training. These are within-source-unit evaluations, not claims of repository-independent generalization.

## Completed benchmark fingerprints

**25 raw evaluation files; 24,450,646 bytes; 45,475 fingerprint records including variants and illustrative/generated examples—not 45,475 unique benchmark problems.** All are evaluation-only, outside training sources.

| Benchmark | Preserved identities and splits |
|---|---|
| HumanEval | 164 canonical test tasks; one illustrative problem and six sample solutions also excluded |
| GSM8K | 7,473 train /1,319 test; both main and Socratic forms;1,319 example model-solution records also excluded |
| MBPP | 974 main tasks and 427 sanitized tasks; original task IDs 1–10 prompt, 11–510 test, 511–600 validation, 601–974 train |
| MATH | 7,500 train /5,000 test across 14 category/split files; publisher-linked combined 12,500-row artifact separately retained |

MATH publisher repo: `hendrycks/math@985bdc1696e88e8643f081a0ff4719da39f2ae2a`. Its README links `qwedsacf/competition_math@e839825f9ec5c6cfa585c654a59610969ec13993`, a combined artifact. The split-preserving payload is independently distributed by `EleutherAI/hendrycks_math@21a5633873b6a120296cce3e2df9d5550074f4a3`. **Complete-row multiset equality passes for all 12,500 records, including all four fields and duplicate multiplicities.** Original distribution split labels were copied unchanged; no random split or reconstruction. The two distributions are explicitly different artifacts.

Each raw file has its authoritative revision, split identity and SHA-256. Every example has native ID where available, otherwise an explicitly labeled content-derived ID, source row index, canonical-record hash, normalized-record hash and per-field normalized hashes/signatures. Detection normalization v1 is NFKC +casefold +whitespace collapse; near signatures retain 64 minimum distinct BLAKE2b-64 hashes of consecutive 5-token shingles. Fields shorter than 5 tokens use exact hashes only. These transformations never modify training bytes; candidate matches require original-content verification.

Final fingerprint sidecar: `evidence/download_authorization/benchmark_fingerprints_v1.jsonl.gz`; **59,289,417 bytes**, SHA-256 `1415778eac25407107bfbe0aae312d55573b79f87882688c91a48528fface90d`. It remains ignored evaluation infrastructure. Missing or corrupt sidecars block acquisition. All 45,475 normalized-record hashes were independently recomputed; all pre-existing fingerprint fields match the first build exactly.

Local exclusions retain **17 test/generator/tool identities** and 14 fixture-file records representing 2 unique synthetic payloads, with normalized signatures. Randomized causality fixtures are additionally protected by generator/seed identities; possible random outputs are not falsely claimed to be exhaustively enumerated. No additional downstream benchmark is currently planned beyond the four registered.

## Storage, validation and acquisition order

Observed evaluation-infrastructure logical bytes, including raw files, two index builds, reader and metadata: **182,284,509**. Estimated acquisition/preprocessing adds 416,000,000 bytes; projected data total **598,284,509**, or **798,284,509** with the separate 200MB run allowance. Stop at 800MB data/preprocessing plus 200MB run artifacts (**1 GB combined**); existing model environments/history are not new acquisition data. Corpus transfer cap 80MB; long-term 40GB allowance unused. These are logical-file accounting and projections, not RSS.

Validation: 5 focused tests pass; Ruff passes;35 files formatting-clean; compile and both CPU/CUDA dependency checks pass. Exact source, specification and document arithmetic/hash checks are recorded in the final validation evidence. Existing model tests were not rerun for this isolated evaluation tool; model source hashes remain unchanged.

The sole added reader is `duckdb==1.4.4`, isolated under ignored `evidence/download_authorization/reader_deps`; exact official wheel URL/hash and recreation command are in the coverage JSON. No CUDA/CPU package replacement occurred. To reproduce fingerprints from the hash-verified local evaluation files:

```powershell
.venv/Scripts/python.exe scripts/benchmark_fingerprints.py evidence/download_authorization/benchmark_downloads.json evidence/download_authorization/rebuilt.jsonl.gz evidence/download_authorization/rebuilt_summary.json --reader-path evidence/download_authorization/reader_deps
```

1. Verify registry, raw benchmark and fingerprint sidecar hashes and pinned source-unit allowlists.
2. After human download authorization, process CPython 3.12.10 eligible files.
3. Process NumPy 2.2.4 eligible files.
4. Process SymPy 1.13.3 eligible files.
5. Process selected mathlib4 v4.19.0 files; no whole-repository download.
6. Process Gutenberg train candidates in order 1400, 766, 1023, 145, 1342, 158 as needed; validation1661; test768. Never acquire rejected1260 or Wikipedia. Retain whole work/author separation.
7. Verify full-file hashes/notices; extract and hash only eligible payloads. Globally deduplicate/exclude benchmark matches before final source-unit split/window generation. Cross-role duplicate clusters are excluded.
8. Freeze training/validation/test manifests and report measured underfill/retention. Stop before any training.

No unresolved authorization blocker remains. Full-payload notice checks, raw/clean SHA-256 and measured retention are mandatory acquisition-time conditions, not fabricated completed evidence. No model training is authorized by this artifact.

DOWNLOAD AUTHORIZATION STATUS: READY
