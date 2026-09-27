# Historical authorization provenance review

Decision: **CASE_A_HISTORICAL_SNAPSHOT_INTACT**.
Impact: **CURRENT_2M_LINEAGE_UNAFFECTED** within the metadata/checkpoint verification
scope below. Entry commit: `aa7e04726332a09024568ab7bfd8c02a0fcb7eb2`.

The expected `6658…` digest is valid historical evidence. The prior 20M audit
correctly failed closed when it encountered a mismatch, but stopped before resolving
the documented historical replay context. The mismatch is not evidence of current
corpus corruption. This review permits the 20M **audit** to resume; it does not
authorize training or declare the unfinished design ready.

## Historical artifact and exact representations

The file first appears in tracked history at
`720af6e63ee8759654b287848173bbd3e74867c4`, committed 2026-09-13 13:34:12 +05:30,
subject `Freeze approved r3 corpus authorization and feasibility evidence`.
Its own creation timestamp is 2026-09-10T01:36:21.988963+00:00. It is schema 1,
status `NOT READY`, stage `PRE_ACQUISITION_BLOCKED`.

| Representation | Bytes | SHA-256 |
|---|---:|---|
| Exact historical Git blob (LF) | 9,208 | `2b8f6c4fd59716434ff5a2e49030b082306a4120f5c8f6afa66423b8d3b05a9a` |
| Deterministic LF→CRLF | 9,503 | `6658bb3c6f1e973870c744ad26288880626132dbc6aefb96a0b32c5433ec52e9` |
| R2/R3 expected historical digest | — | `6658bb3c6f1e973870c744ad26288880626132dbc6aefb96a0b32c5433ec52e9` |
| Preserved snapshot file | 9,503 | `6658bb3c6f1e973870c744ad26288880626132dbc6aefb96a0b32c5433ec52e9` |

The CRLF historical representation and the snapshot are byte-identical. Git bytes
were obtained with `subprocess.check_output(["git", "show", commit+":"+path])`,
avoiding shell output encoding/newline conversion. Neither representation was
written over a working or historical file. Expected contract hashes remain unchanged.

## Snapshot inventory and original identity

Root: `data/pilot-2m-r1/provenance/authorization`.
Original identity file: `data/pilot-2m-r1/provenance/authorization_snapshot.json`.

The recorded “snapshot SHA” is **SHA-256 of the exact original 6,552-byte manifest
file**, not a newly invented tree digest or canonical-JSON digest:

`8461a8d3aaad724a6880627a91520257ab743dae3713c95ad592b5467395ad8d`

It exactly matches `reports/real_corpus_2m_evidence.json` →
`historical_authorization.snapshot_sha256`. That original manifest contains the
historical commit plus a relative-path→SHA-256 map of **57 files**. Every listed
file matches its recorded raw-byte digest. Independently, all 57 match a historical
Git representation: 16 LF and 41 CRLF. No JSON reserialization or replacement
aggregation was used to make the identity pass.

The directory actually contains **66 regular files**: the 57 manifest-bound files,
four supplemental replay files, and five pre-existing bytecode cache files. Thus
`files_verified=57` describes manifest coverage, not total directory membership.
The JSON companion inventories all 66 paths, byte sizes and SHA-256 values.

Supplemental replay files are `scripts/benchmark_fingerprints.py` and the three
historical test modules. All four match the historical Git commit (one LF, three
CRLF) and current unchanged copies. The new harness pins these separately; it
does not modify or expand the original manifest. Bytecode caches are not copied
or executed. Original files, including the caches, remain unchanged.

The exact original manifest-generation shell command is not retained in inspected
tracked source/local receipts. The original serialized manifest, its matching
recorded raw-file digest, its per-file digests, and historical Git counterparts are
available and were verified. This review does not claim to have recovered a missing
command transcript or substitute a new aggregation algorithm.

## Complete tracked history of the report

`git log --all -- reports/real_corpus_2m_evidence.json` identifies exactly two
commits changing the path:

| Commit | Date (+05:30) | Content evidence |
|---|---|---|
| `720af6e63ee8759654b287848173bbd3e74867c4` | 2026-09-13 13:34:12 | Adds schema-1 pre-acquisition blocked report; historical authorization capacity estimates |
| `f770e29c0a5c2e07e556f121aae0db0d83cbe7b0` | 2026-09-20 07:32:54 | Replaces root report with schema 2, `post_contamination_capacity`, still `NOT READY`; 53/60 cells pass after exclusions |

The later commit has the uninformative subject `Changes`, so intent is established
from content rather than that subject. Its report introduces the snapshot root,
digest and count, explicitly says it supersedes operational status rather than
historical evidence, and reports the post-contamination deficits. The accompanying
build report and project state explain why the frozen authorization tests run
against historical inputs. Later final-corpus reports retain this newer root report
as preserved evidence; the final corpus readiness decision lives in separate files.

Current schema-2 report identity:

- Working CRLF file: 70,753 bytes,
  `7ff340a96a861538319554883543eaf2eacb299828743379c2504b9ce1a5db76`.
- Git LF blob: 68,098 bytes,
  `c9bb670c2f9cc11447f17f58d199f032397732acb73014cbfe3702345878fd65`.
- The file is unchanged since the superseding commit. Neither identity is supposed
  to equal the historical schema-1 report's hash.

## Exact declarations and original test context

At `reports/real_corpus_2m_evidence.json` → `historical_authorization.note`:

> Historical estimates and blocked reports remain immutable here and in Git. Current report supersedes their operational status, not their evidence.

At `validation.historical_test_context` in that same report:

> Unchanged authorization capacity/r3/benchmark tests replayed under frozen authorization snapshot, preserving hash-bound historical report inputs. Current-root historical tests would reject the intentionally updated report hash; not a current corpus readiness test.

Corroborating existing records:

- `reports/real_corpus_2m_build.md`, validation paragraph: frozen-input historical
  replay; 51 focused tests, not post-acquisition sufficiency.
- `reports/real_corpus_final_build.md`, checks: unchanged historical replay;
  69 tests passed during final corpus closeout.
- `reports/real_corpus_final_validation.json` → `historical_test_scope` and
  `tests`: explicit immutable snapshot context and `69 passed … in 3.10s`.
- Preserved `docs/project_state.md` entries for the real-corpus phases repeat
  the historical replay contract.

The original mechanism is visible in the preserved directory: copies of
`test_authorization_capacity.py`, `test_authorization_r3.py` and
`test_benchmark_fingerprints.py` reside under its `tests/` directory. Their
`Path(__file__).parents[1]` expressions resolve imports and hash-bound inputs under
the snapshot root. Historical receipts record that execution context, but do not
retain the exact shell command. This review reproduces the mechanism, not an
unavailable command transcript.

Running the same ROOT-relative modules from current `tests/` resolves the
schema-2 report instead. Seven tests fail at the unchanged hash check (one CLI
case subsequently sees missing JSON). This is expected context mismatch after
the intentional supersession, not an invalid historical expected hash.

## Minimal additive test fix

Original tests, scripts, contracts and reports remain unchanged.

- New `scripts/historical_authorization_replay.py` checks the pinned original
  manifest identity, all 57 entries and four separately pinned replay files.
- It copies only verified metadata/code bytes to a disposable temporary root and
  invokes the three original test modules there. Cache writing/plugin autoload is
  disabled in the child. No snapshot mutation, corpus access, download, training
  or fallback to current metadata occurs. Original inputs are checked again afterward.
- New `tests/conftest.py` removes the three context-specific modules from ordinary
  current-root collection and registers `historical_replay`. Their tests are not
  dropped: the mandatory wrapper runs all **30** in the verified historical context.
  A default full pytest collection includes the wrapper; absent/corrupt snapshots
  fail rather than skip.
- New current provenance tests verify the superseding root report's final-corpus
  binding, the final validation receipt, unchanged replay code, and rejection of
  missing snapshots, mutated manifests/artifacts and path escapes.

See [replay instructions](../docs/historical_authorization_replay.md). A full
validation must require both groups. Explicitly selecting the historical modules
at their obsolete current-root paths is not the supported replay command.

## Current 2M/final-corpus lineage

The current schema-2 report is explicitly bound by
`reports/real_corpus_final_evidence.json.preserved_sha256`. That final report hashes
to `68eed41234f7296ead6712391532d0ebe0712f5a96b0c0ada6d7cfaf4b3a7dd9`, matching
its final validation receipt. The following checks passed without reading payloads:

- Final manifest raw SHA:
  `38bbb7266544129b8c1bd14d2e744f233208d55c403135244963024fad6ed86b`;
  canonical SHA `f5f375b0c7b446a39000f5a3fb693dbe111d9621b6818a52a3eea339b8a29ecc`.
  Both match final evidence and production training bindings.
- Manifest document-byte totals and bound split metadata remain **10,000,000 TRAIN,
  500,000 VALIDATION, 500,000 sealed TEST**. All three recorded split digests agree
  across final manifest, final report and production binding. Exact values are in JSON.
- Native training manifest raw hash matches the final corpus report; its canonical
  hash matches the production binding. These distinct serialization identities
  (`64e154…` raw versus `33511a…` canonical) are consistent, not another mismatch.
- Frozen computational-source canonical SHA remains `7d98de1b…7173eb4`.
- Endpoint state file SHA remains
  `0d4c822daadf6bb8269a727ac54d54a26e9edf49ff43fedff1b0d91ba6ad2623`, matching
  both checkpoint manifest and accepted exact-restore receipt.
- Accepted parameter identity remains `75b1a342…8be173`; 1,929,579 parameters,
  78,167 logical updates, 10,000,064 logical target bytes, and final validation
  NLL approximately 2.1701512726. These metrics are retained from accepted evidence;
  parameter tensors and validation loss were not recomputed.

Classification: **CURRENT_2M_LINEAGE_UNAFFECTED** by this historical-contract issue.
This is a provenance-impact conclusion, not a new full corpus corruption scan.
No TRAIN/VALIDATION/TEST payload was rehashed, no model was loaded, and no training,
backward or optimizer step occurred in this tranche. TEST remains sealed.

## Validation and limitations

- Current applicable non-training suite: **140 passed**, one pre-existing optional
  NumPy initialization warning, in 15.67 seconds.
- Independent historical replay: **30 passed**; mandatory wrapper **1 passed**.
- Default pytest collection confirmed the required historical wrapper is present.
- Ruff, format, compileall, CPU and CUDA dependency checks and diff checks pass
  after correcting four line-length violations in the new helper. Initial failure
  output and successful rerun receipts remain in the JSON companion.
- The full model/training suite was deliberately not executed. Training/overfit
  tests perform updates; `test_full_training_stage_a.py` invokes real corpus
  verification including sealed TEST integrity reads. The JSON records the exact
  current file list. Synthetic data unit-test fixtures are not production TEST.
- Retrospective scope correction: the previous audit ran plain `pytest`, which
  included those real-corpus integrity tests. Its blanket statement that no real
  TEST payload was accessed was therefore too broad; the metadata audit helper
  avoided it, but that full test command included integrity reads. This is not
  TEST evaluation or training exposure. No such reads were performed in this
  reconciliation, and the old report is preserved rather than silently rewritten.
- Historical generator/replay shell command transcripts were not recovered. The
  retained bytes, algorithms, test source, declarations and independent replay
  establish the relationship without claiming those transcripts exist.
- Snapshot is a required local validation input; a fresh checkout lacking it fails
  closed. No silent reconstruction/download is built into the harness.

Files added: this Markdown/JSON report, replay helper, replay documentation,
collection hook and two current test modules. `docs/project_state.md` receives an
additive reconciled entry. Old evidence, expected hashes, source/model code, original
tests, specifications, checkpoints and snapshot are untouched. Before closeout,
all old tracked file hashes (except the authorized state entry) and all 66 snapshot
file hashes are compared with entry identities.

Rollback: revert this additive commit's harness/reports/status entry together;
the frozen model and reference fallback remain available. The previous root-context
failures would return. Risk is future misuse of historical tests as current corpus
readiness checks, or running a broad training/TEST-reading suite under a narrower
authorization. The explicit group separation and documented commands address that.

Next step: resume the blocked 20M architecture/design audit using these reconciled
provenance results. No full 20M design work, training or Kaggle operation was resumed
in this narrow tranche. The approval-bound Kaggle host boundary remains unchanged.

HISTORICAL PROVENANCE STATUS: RECONCILED — 20M AUDIT MAY RESUME
