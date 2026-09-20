# Contamination matcher v2: unadopted review proposal

This proposal changes no corpus admission decision. The v1 candidate snapshot,
fingerprints, 75 first-hit records, 46 excluded units, 53/60 capacity result and
NOT READY build reports remain authoritative and unchanged.

The proposed exception is defined by field role and syntax, not a list of current
strings or a length threshold. A field is non-actionable only when its original
structured root is `test_imports` and Python's AST parser produces exactly one
`Import` or `ImportFrom` statement. Parsing never executes the code. This includes
ordinary aliases and from-imports but excludes statements followed by assertions,
assignments, calls or other code. A missing or different content role receives no
exception. Existing metadata roots remain excluded from text matching:
`task_id`, `entry_point`, `source_file`, `level`, `type`.

Complete problem/example content and complete answer/solution fields remain
actionable, even when short, frequent or identical to suppressed support content.
The exception is applied per field before indexing; a protected answer with the
same bytes still enters the index. Matching continues past ignored support
material and detects an answer surrounded by boilerplate. All other exact,
normalized and verified near matching semantics remain v1: NFKC/casefold/whitespace
normalization for detection only, bottom-64 retrieval with verified five-token
Jaccard at least 0.85, and the existing paragraph/window coverage limitations.

No minimum length, frequency cutoff or two-span requirement is adopted. Those
diagnostics can suppress complete short answers and cannot establish absence of
leakage. No general exception for return statements or numerical answers is
proposed. Whole-record hashes remain preserved in the benchmark registry; this
prototype does not introduce a new record serializer or claim a new full-record
matching gate. Existing constituent problem/answer fields remain protected.

The implementation is isolated in `scripts/corpus_matcher_v2_proposal.py` and is
used only by proposal tests. It is not connected to corpus acquisition, selection
or the production v1 scan. The unchanged v1 index verifies content after retrieval.

Capacity projections use only the recorded first-hit ledger. Removing a generic
first hit could expose a second, protected match not recorded by v1. Therefore
projected capacity is an upper bound, not a certified corrected corpus. Adopting
this proposal would require human review, a separate versioned evidence run that
examines protected content after ignored fields, and all subsequent capacity and
leakage gates. No such scan or restoration is authorized by this report.
