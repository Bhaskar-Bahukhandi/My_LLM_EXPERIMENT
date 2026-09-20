# Final pilot corpus: ready for training review

The measured pipeline passes **60 / 60** frozen source/domain/split cells. The immutable corpus contains exactly **10,000,000 train / 500,000 validation / 500,000 test bytes**, selected from 370 documents. This is an engineering pilot, not evidence of model quality. No training ran.

## Approved mode correction

`docs/file_mode_policy_v2.json` admits regular source-code blobs with mode 100755 as inert data, subject to every existing gate. Symlinks, gitlinks and unsupported modes still fail closed. The original 100644-only policy, rejected-file receipts, 58/60 report and historical batch-preflight rejection remain hash-verified and unchanged.

All 11 cached SymPy 1.13.3 files were re-reviewed against commit `b4ce69ad5d40e4e545614b6c76ca9b0be0b98f0b`; no new download or source execution occurred. Exact paths, Git blobs, SHA-256, license controls, notices, authorship and split assignments are in `docs/mode_v2_file_reviews.json`. All passed unchanged filtering, global deduplication, exhaustive contamination v2 and project-fixture exclusions. Root licenses do not override file-local exclusions.

| SymPy code pool | Final quota | Raw and measured eligible bytes | Minimum raw | Required headroom | Achieved headroom | Independent units |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 12,500 | 20,392 | 17,858 | 5,358 | 7,892 | 3 |
| Test | 12,500 | 21,592 | 17,858 | 5,358 | 9,092 | 5 |

The existing 7/10 planning rule is unchanged. No second retention haircut is applied to measured cleaned bytes. The pools use the already reviewed small file set with margin; no unused inventory or new source was admitted. Source-unit groups are disjoint from each other and training.

## Measured accounting

| Stage | Documents | Bytes |
|---|---:|---:|
| Filtered original plus targeted candidates | 3,462 | 57,718,936 |
| Global deduplication | 3,457 | 57,708,004 |
| Exhaustive contamination exclusions | 2,698 | 42,923,348 |
| Final selected payload | 370 | 11,000,000 |
| Unselected eligible capacity | — | 31,923,348 |

The five removed duplicate documents account for 10,932 bytes; contamination source-unit exclusions account for 14,784,656 bytes. Original per-source raw/license/filter accounting is linked by hash in `real_corpus_final_evidence.json`, alongside all 60 final cells and unused capacity. The mode correction retained all 41,984 cached bytes after every gate.

| Source | Train | Validation | Test |
|---|---:|---:|---:|
| Cleared Gutenberg | 5,000,000 | 250,000 | 250,000 |
| CPython | 2,500,000 | 125,000 | 125,000 |
| NumPy | 1,000,000 | 50,000 | 50,000 |
| SymPy | 1,000,000 | 50,000 | 50,000 |
| mathlib4 | 500,000 | 25,000 | 25,000 |

Domain proportions remain 50% general, 25% code, 15% documentation and 10% structured mathematics in each split. Wikipedia contributes zero bytes.

## Frozen identities

Manifest: `data/pilot-2m-r1/final-corpus-v1/manifest.json`.

| Identity | SHA-256 |
|---|---|
| Manifest file | `38bbb7266544129b8c1bd14d2e744f233208d55c403135244963024fad6ed86b` |
| Canonical manifest | `f5f375b0c7b446a39000f5a3fb693dbe111d9621b6818a52a3eea339b8a29ecc` |
| Train | `5204262762bf2c54173d0c60e1cfb82d0d851c5a73f868de32d490d44557c120` |
| Validation | `70f52e5d362f4425182b0ad058163e56ac8987971f1db61dde1a224ea59a8614` |
| Test | `a3f7bc2b3fa3853a6784aff9e7da78bcf2e3fc535dc15d0ba6cfa1a00f44173f` |

Split hashes cover concatenated selected bytes in manifest order, without separators. Relative paths, per-document hashes, exact cleaned-byte ranges, raw/cleaned hashes, source pins, unit roles, licenses and provenance receipts are retained. Native `training_manifest.json` exports train/validation only; the test set stays separately identified and is never relabeled.

## Validation and limits

- Independent full-source-document audit: zero unresolved raw-exact, normalized-exact or near matches for all three split pairs. Bottom64 retrieval does not certify semantic/paraphrase completeness. Final selected payloads additionally passed byte-identical cross-split checks; normalized/near audit scope is the complete documents before selection.
- Independent verification checked all selected payloads against their source ranges, raw and cleaned hashes, receipts, filter records, 60 exact quotas, split hashes and manifest identities.
- Loader dry-run round-tripped all 11 MB at 32-byte and 128-byte windows, reproduced deterministic ordering and batches, and kept raw targets in 0..255. Exact byte selection may end a document mid-UTF8 sequence; no re-encoding occurs.
- **69 tests passed** in 3.10 seconds. Historical authorization tests replayed unchanged from their immutable snapshot because their contracts bind historical report hashes. Ruff, formatting (68 files), compileall and CPU/CUDA dependency checks passed. The optional absent NumPy bridge produced one warning; it is unused by this loader.
- Measured acquisition/evidence storage: **339,883,666 bytes**, below the 800 MB data cap plus separate 200 MB run reserve. Generated corpus remains ignored, not committed.
- Model source identity and rollback `720af6e63ee8759654b287848173bbd3e74867c4` are unchanged. No model, backward, optimizer, CUDA environment, architecture or RAG work occurred. No training quality or GPU performance claim is made.

## Changes and boundary

Added mode-policy/review artifacts, `corpus_mode_v2.py`, `corpus_complete_mode_v2.py`, `corpus_freeze.py`, their focused regression tests, pool/leakage/loader evidence and these final reports. Updated `docs/project_state.md`. All earlier NOT READY reports remain preserved; final evidence is additive. The working tree remains available for review without a new commit or history rewrite.

No outstanding corpus-build blocker was found. Next step: human real-data training review and separately authorized bounded training configuration. Stop before training.

REAL CORPUS BUILD STATUS: READY FOR REAL-DATA TRAINING REVIEW
