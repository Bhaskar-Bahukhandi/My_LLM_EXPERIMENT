# Contamination review v1: human review required

The accepted corpus remains **NOT READY, 53/60**. All 75 recorded hits, 46 excluded source units, seven deficits, candidate snapshot and previous reports remain unchanged. This diagnostic acquired no new data and did not repeat the completed benchmark scan, cleaning or deduplication. It counted only the four already-recorded match strings across the existing post-dedup candidates.

**Finding:** 26 hits are generic support imports (B); 49 are ambiguous complete-answer fields (C); no recorded hit establishes distinctive copying (A: 0). This is not proof that the candidates contain no genuine leakage. Complete short answers remain protected regardless of apparent simplicity.

## Frozen failing cells

| Source | Domain | Split | Required | Before contamination | Excluded | After contamination | Deficit |
|---|---|---|---:|---:|---:|---:|---:|
| cpython | code | test | 75000 | 163435 | 163435 | 0 | 75000 |
| cpython | documentation | train | 1000000 | 1474384 | 634248 | 840136 | 159864 |
| cpython | documentation | test | 50000 | 97382 | 97382 | 0 | 50000 |
| numpy | code | train | 750000 | 1867286 | 1749916 | 117370 | 632630 |
| sympy | code | validation | 12500 | 339738 | 339738 | 0 | 12500 |
| sympy | code | test | 12500 | 677810 | 677810 | 0 | 12500 |
| sympy | structured_math | test | 25000 | 43308 | 24828 | 18480 | 6520 |

## Matched-content families and frequencies

No benchmark answers are reproduced here. Full raw/normalized SHA-256 values, raw lengths per variant, pinned repositories/revisions, benchmark splits, record IDs and exact structured field paths are recorded for every hit in `contamination_review_v1.json`. Family identifiers below refer to those hash-bound entries.

| Family | Normalized SHA-256 | Bytes | Tokens | Benchmark field occurrences | Candidate matching documents | Matching lines | Units | Category |
|---|---|---:|---:|---:|---:|---:|---:|---|
| F1 | b67c9ddef3e0c3715a3708ecfa296cc461a164bde94c8ba3fd112789b5bed926 | 11 | 2 | 13 | 26 | 27 | 17 | B |
| F2 | 1acf9060bbcfa6cd0f44158462967f04dd31d08ff3afb00aac42b0eeaddaebab | 12 | 4 | 1 | 1 | 3 | 1 | C |
| F3 | 486d9affb60dbb0063b03d8e23a6ccf6364ce203dc3a9f56f20e750eb41ecade | 8 | 2 | 4 | 49 | 80 | 32 | C |
| F4 | b7a56873cd771f2c446d369b649430b65a756ba278ff97ec81bb6f55b2e73569 | 2 | 1 | 1 | 3 | 3 | 3 | C |

F1 is one AST-verified Python import in MBPP `test_imports`; it is not an answer. F2 is a complete HumanEval canonical solution consisting of elementary arithmetic return syntax. F3 is a complete elementary constant-return solution/completion in HumanEval illustrative examples, including repeated sample variants. F4 is a complete short scalar generated-solution field in GSM8K. The latter three remain C because generic-looking full-answer equality does not establish copying, yet the required policy protects complete answers. None is a complete multi-field benchmark example; each is a complete string subfield of a larger record.

Benchmark occurrence counts include frozen distribution/sample variants, not unique independent tasks. Candidate counts cover all 3,438 post-dedup documents / 57,415,744 bytes in 231 units. Matching-line counts use the frozen normalized whole-line rule. Raw substring totals in JSON count non-overlapping occurrences per raw variant (variant overlap may double-count spans); normalized substring totals include embedded fragments and are descriptive only, not v1 actionable-hit counts. F3 appears in 49 documents although only 46 recorded first hits name it; F4 appears in three documents although only two first hits name it. These differences demonstrate first-hit censorship.

## Why the exclusions expanded

The registry fingerprints every string field and also retains whole-record identities. The matching index uses individual content fields; it does not use whole-record identities as independent matching triggers. Five metadata roots (`task_id`, `entry_point`, `source_file`, `level`, `type`) are omitted. The diagnostic reverified all 45,475 records, 101,634 content fields and 50,763 omitted metadata fields against pinned raw files and the frozen sidecar. No recorded hit came from an omitted metadata field. IDs, source paths, categories and entry-point names were not accidentally matched.

However, `test_imports` is intentionally included as content. A field shorter than 32 normalized characters becomes actionable on exact normalized document/line equality, without a role/information-content safeguard. Every hit then excludes its entire source-unit group, rather than only the matching line or document. There is no quota-aware repair. Common complete answers trigger the same mechanism. This explains the amplification without establishing that all excluded material is contaminated.

## Sensitivity: ledger projections only

All numbers below are counterfactual projections of recorded hits, never changes to corpus admission. They are upper bounds on usable capacity: an ignored first hit can reveal an unrecorded protected match. No alternative full benchmark scan was performed.

| Rule | Actionable recorded hits | Excluded units | Excluded bytes | Cells passing (upper bound) | Protection retained or lost |
|---|---:|---:|---:|---:|---|
| frozen | 75 | 46 | 17354845 | 53/60 | All existing exclusions retained; actual frozen outcome |
| support_import_exception_only | 49 | 34 | 13787314 | 56/60 | Retains complete answers and other content; suppresses support-only parsed imports |
| full_answer_protection_plus_import_exception | 49 | 34 | 13787314 | 56/60 | Retains complete answers and other content; suppresses support-only parsed imports |
| length_only_16_UNSAFE | 0 | 0 | 0 | 60/60 | Loses complete-answer protection; diagnostic only, rejected |
| length_only_32_UNSAFE | 0 | 0 | 0 | 60/60 | Loses complete-answer protection; diagnostic only, rejected |
| length_only_64_UNSAFE | 0 | 0 | 0 | 60/60 | Loses complete-answer protection; diagnostic only, rejected |
| length_only_128_UNSAFE | 0 | 0 | 0 | 60/60 | Loses complete-answer protection; diagnostic only, rejected |
| frequency_at_least_10_documents_UNSAFE | 3 | 3 | 963612 | 58/60 | Loses complete-answer protection; diagnostic only, rejected |
| require_two_distinct_recorded_spans_UNSAFE_INCOMPLETE_LEDGER | 0 | 0 | 0 | 60/60 | Loses complete-answer protection; diagnostic only, rejected |

Length-only cutoffs of 16/32/64/128 normalized bytes remove every recorded hit, including entire answers. The frequency example (10 matching documents) suppresses a complete constant-return answer. Neither safeguard is proposed. A two-independent-span requirement cannot be established from a one-hit-per-document ledger; its zero-hit result demonstrates missing evidence, not safety. With the mandatory full-answer override, these exceptions cannot suppress F2/F3/F4 and the bounded support exception remains at most 56/60.

### Remaining deficits under the narrow proposal

| Source | Domain | Split | Required | Projected available (upper bound) | Minimum remaining deficit |
|---|---|---|---:|---:|---:|
| cpython | documentation | train | 1000000 | 905876 | 94124 |
| sympy | code | validation | 12500 | 0 | 12500 |
| sympy | code | test | 12500 | 0 | 12500 |
| sympy | structured_math | test | 25000 | 18480 | 6520 |

The import-only change removes 26 recorded triggers but only 12 unit exclusions: other protected hits keep five of the 17 import-containing units excluded. Projected excluded bytes fall from 17,354,845 to 13,787,314. This cannot solve all seven deficits; even optimistic recovery leaves four. Additional approved capacity would be necessary under unchanged full-answer protection, subject first to human review of the proposed policy and remaining ambiguity. No candidate expansion is performed or authorized here.

## All 75 recorded hit decisions

This deterministic table is keyed to the complete machine-readable audit. Every row retains its frozen exclusion. B does not authorize restoration. “Sole” means the sole recorded trigger in that source unit; any one hit was sufficient to exclude the whole unit. Other unrecorded matches remain possible. Raw matched lengths can vary by benchmark representation; they are listed as a set per row. Candidate identity, family, unit and cell make each exclusion traceable.

| Audit | Candidate | Unit | Split / domain | Family | Raw bytes | Entire answer | Category | Sole trigger |
|---:|---|---|---|---|---|---|---|---|
| 1 | cpython/Doc/library/itertools.rst | cpython:itertools | train / documentation | F1 | 11 | False | B | True |
| 2 | cpython/Doc/library/typing.rst | cpython:typing | train / documentation | F2 | 17 | True | C | True |
| 3 | cpython/Doc/reference/lexical_analysis.rst | cpython:documentation/reference | train / documentation | F3 | 10,12 | True | C | True |
| 4 | cpython/Lib/_pydecimal.py | cpython:_pydecimal | train / code | F3 | 10,12 | True | C | True |
| 5 | cpython/Lib/_pyio.py | cpython:_pyio | train / code | F3 | 10,12 | True | C | True |
| 6 | cpython/Lib/aifc.py | cpython:aifc | train / code | F1 | 11 | False | B | True |
| 7 | cpython/Lib/asyncio/windows_events.py | cpython:asyncio | train / code | F1 | 11 | False | B | True |
| 8 | cpython/Lib/calendar.py | cpython:calendar | train / code | F3 | 10,12 | True | C | True |
| 9 | cpython/Lib/curses/textpad.py | cpython:curses | train / code | F3 | 10,12 | True | C | True |
| 10 | cpython/Lib/fractions.py | cpython:fractions | train / code | F1 | 11 | False | B | True |
| 11 | cpython/Lib/idlelib/tree.py | cpython:idlelib | train / code | F3 | 10,12 | True | C | True |
| 12 | cpython/Lib/ipaddress.py | cpython:ipaddress | train / code | F3 | 10,12 | True | C | True |
| 13 | cpython/Lib/lib2to3/main.py | cpython:lib2to3 | train / code | F3 | 10,12 | True | C | True |
| 14 | cpython/Lib/logging/handlers.py | cpython:logging | train / code | F3 | 10,12 | True | C | True |
| 15 | cpython/Lib/multiprocessing/shared_memory.py | cpython:multiprocessing | train / code | F3 | 10,12 | True | C | True |
| 16 | cpython/Lib/numbers.py | cpython:numbers | train / code | F3 | 10,12 | True | C | True |
| 17 | cpython/Lib/pstats.py | cpython:pstats | train / code | F3 | 10,12 | True | C | True |
| 18 | cpython/Lib/selectors.py | cpython:selectors | train / code | F1 | 11 | False | B | True |
| 19 | cpython/Lib/statistics.py | cpython:statistics | train / code | F1 | 11 | False | B | True |
| 20 | cpython/Lib/threading.py | cpython:threading | train / code | F3 | 10,12 | True | C | True |
| 21 | cpython/Lib/unittest/case.py | cpython:unittest | train / code | F3 | 10,12 | True | C | True |
| 22 | cpython/Lib/urllib/parse.py | cpython:urllib | test / code | F1 | 11 | False | B | True |
| 23 | cpython/Lib/xml/dom/expatbuilder.py | cpython:xml | train / code | F3 | 10,12 | True | C | False |
| 24 | cpython/Lib/xml/sax/expatreader.py | cpython:xml | train / code | F3 | 10,12 | True | C | False |
| 25 | numpy/numpy/_core/_internal.py | numpy:_core | train / code | F1 | 11 | False | B | False |
| 26 | numpy/numpy/_core/_machar.py | numpy:_core | train / code | F1 | 11 | False | B | False |
| 27 | numpy/numpy/_core/numeric.py | numpy:_core | train / code | F1 | 11 | False | B | False |
| 28 | numpy/numpy/lib/__init__.py | numpy:lib | train / code | F1 | 11 | False | B | True |
| 29 | sympy/sympy/categories/diagram_drawing.py | sympy:categories | train / code | F3 | 10,12 | True | C | True |
| 30 | sympy/sympy/codegen/approximations.py | sympy:codegen | train / code | F1 | 11 | False | B | True |
| 31 | sympy/sympy/combinatorics/partitions.py | sympy:combinatorics | test / code | F3 | 10,12 | True | C | False |
| 32 | sympy/sympy/combinatorics/perm_groups.py | sympy:combinatorics | test / code | F4 | 2 | True | C | False |
| 33 | sympy/sympy/combinatorics/permutations.py | sympy:combinatorics | test / code | F3 | 10,12 | True | C | False |
| 34 | sympy/sympy/combinatorics/schur_number.py | sympy:combinatorics | test / code | F1 | 11 | False | B | False |
| 35 | sympy/sympy/core/basic.py | sympy:core | train / code | F3 | 10,12 | True | C | False |
| 36 | sympy/sympy/core/evalf.py | sympy:core | train / code | F1 | 11 | False | B | False |
| 37 | sympy/sympy/core/intfunc.py | sympy:core | train / code | F1 | 11 | False | B | False |
| 38 | sympy/sympy/core/numbers.py | sympy:core | train / code | F1 | 11 | False | B | False |
| 39 | sympy/sympy/core/sorting.py | sympy:core | train / code | F3 | 10,12 | True | C | False |
| 40 | sympy/sympy/crypto/crypto.py | sympy:crypto | train / code | F3 | 10,12 | True | C | True |
| 41 | sympy/sympy/diffgeom/diffgeom.py | sympy:diffgeom | train / code | F3 | 10,12 | True | C | True |
| 42 | sympy/sympy/functions/combinatorial/factorials.py | sympy:functions | train / code | F3 | 10,12 | True | C | False |
| 43 | sympy/sympy/functions/combinatorial/numbers.py | sympy:functions | train / code | F3 | 10,12 | True | C | False |
| 44 | sympy/sympy/functions/special/tensor_functions.py | sympy:functions | train / code | F3 | 10,12 | True | C | False |
| 45 | sympy/sympy/geometry/line.py | sympy:geometry | validation / code | F3 | 10,12 | True | C | True |
| 46 | sympy/sympy/integrals/heurisch.py | sympy:integrals | train / code | F3 | 10,12 | True | C | False |
| 47 | sympy/sympy/integrals/laplace.py | sympy:integrals | train / code | F3 | 10,12 | True | C | False |
| 48 | sympy/sympy/integrals/transforms.py | sympy:integrals | train / code | F3 | 10,12 | True | C | False |
| 49 | sympy/sympy/liealgebras/weyl_group.py | sympy:liealgebras | train / code | F3 | 10,12 | True | C | True |
| 50 | sympy/sympy/matrices/common.py | sympy:matrices | train / code | F3 | 10,12 | True | C | False |
| 51 | sympy/sympy/matrices/reductions.py | sympy:matrices | train / code | F3 | 10,12 | True | C | False |
| 52 | sympy/sympy/ntheory/elliptic_curve.py | sympy:ntheory | train / code | F3 | 10,12 | True | C | False |
| 53 | sympy/sympy/ntheory/factor_.py | sympy:ntheory | train / code | F1 | 11 | False | B | False |
| 54 | sympy/sympy/ntheory/partitions_.py | sympy:ntheory | train / code | F1 | 11 | False | B | False |
| 55 | sympy/sympy/ntheory/residue_ntheory.py | sympy:ntheory | train / code | F3 | 10,12 | True | C | False |
| 56 | sympy/sympy/parsing/c/c_parser.py | sympy:parsing | train / code | F3 | 10,12 | True | C | True |
| 57 | sympy/sympy/physics/biomechanics/activation.py | sympy:physics | train / code | F3 | 10,12 | True | C | False |
| 58 | sympy/sympy/physics/continuum_mechanics/truss.py | sympy:physics | train / code | F1 | 11 | False | B | False |
| 59 | sympy/sympy/physics/paulialgebra.py | sympy:physics | train / code | F3 | 10,12 | True | C | False |
| 60 | sympy/sympy/physics/quantum/qubit.py | sympy:physics | train / code | F1 | 11 | False | B | False |
| 61 | sympy/sympy/physics/quantum/shor.py | sympy:physics | train / code | F1 | 11 | False | B | False |
| 62 | sympy/sympy/physics/secondquant.py | sympy:physics | train / code | F3 | 10,12 | True | C | False |
| 63 | sympy/sympy/plotting/textplot.py | sympy:plotting | train / code | F1 | 11 | False | B | True |
| 64 | sympy/sympy/polys/domains/integerring.py | sympy:polys | train / code | F1 | 11 | False | B | False |
| 65 | sympy/sympy/polys/groebnertools.py | sympy:polys | train / code | F3 | 10,12 | True | C | False |
| 66 | sympy/sympy/polys/polyroots.py | sympy:polys | train / code | F1 | 11 | False | B | False |
| 67 | sympy/sympy/polys/ring_series.py | sympy:polys | train / code | F1 | 11 | False | B | False |
| 68 | sympy/sympy/polys/subresultants_qq_zz.py | sympy:polys | train / code | F3 | 10,12 | True | C | False |
| 69 | sympy/sympy/series/gruntz.py | sympy:series | train / code | F3 | 10,12 | True | C | True |
| 70 | sympy/sympy/simplify/powsimp.py | sympy:simplify | train / code | F3 | 10,12 | True | C | False |
| 71 | sympy/sympy/simplify/simplify.py | sympy:simplify | train / code | F3 | 10,12 | True | C | False |
| 72 | sympy/sympy/simplify/sqrtdenest.py | sympy:simplify | train / code | F3 | 10,12 | True | C | False |
| 73 | sympy/sympy/strategies/tree.py | sympy:strategies | train / code | F4 | 2 | True | C | True |
| 74 | sympy/sympy/tensor/tensor.py | sympy:tensor | train / code | F3 | 10,12 | True | C | True |
| 75 | sympy/sympy/utilities/timeutils.py | sympy:utilities | train / code | F1 | 11 | False | B | True |

## Proposal, tests and preservation

The unadopted policy is specified in `docs/contamination_matcher_v2_proposal.md`; its isolated executable prototype is `scripts/corpus_matcher_v2_proposal.py`. The exception uses the structured support role and AST statement kind, not a manual list of current strings. Other fields with identical bytes remain indexed. Complete problems/answers, normalized variants and existing verified near matches remain protected. No production matcher or admission code was edited.

All 57 relevant tests pass (1.14s), including six proposal cases covering complete problems/answers, normalization, near matches, isolated imports, generic import plus genuine copied answer, metadata exclusion, deterministic field order and short-answer protection. Existing acquisition/filter/dedup/fingerprint tests pass. Historical authorization tests run unchanged in their frozen snapshot, as before this tranche. Ruff and format pass (57 files), compile and both CPU/CUDA dependency checks pass. The review search found no TODO/FIXME, broad exception swallowing or shell execution in the new code. Whitespace checks pass.

Every blocked-state hash in the diagnostic was rechecked after the audit. Byte-identical copies are under `data/pilot-2m-r1/provenance/contamination-review-v1`; the large frozen fingerprint sidecar is hash-verified in place. The executed audit script is archived there with its recorded hash. The current script differs only in formatting of one string literal, verified by identical Python ASTs. No expensive scan was rerun for that formatting correction.

Files added this tranche: this report, `reports/contamination_review_v1.json`, `reports/contamination_review_v1_validation.json`, the proposal document, two review scripts and `tests/test_contamination_review_v1.py`. The project ledger adds the current review state; existing NOT READY build reports remain byte-identical. Rollback `720af6e63ee8759654b287848173bbd3e74867c4` and earlier commits are unchanged.

**Not validated:** a full proposed-v2 corpus scan, absence of secondary protected hits, final leakage, selected quotas, corpus/manifest hashes or loader behavior. None was authorized at this boundary. There is no final corpus and no model training.

**Recommended next action:** human review of the narrow support-field exception and the 49 protected-but-ambiguous answer matches. If the exception is approved, separately authorize a versioned protected-content scan before reconsidering any admission. Plan additional independently approved capacity for the four residual deficits under unchanged answer protection; do not acquire it yet.

CONTAMINATION REVIEW STATUS: HUMAN REVIEW REQUIRED
