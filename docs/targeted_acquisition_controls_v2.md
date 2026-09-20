# Targeted acquisition: preserve existing file controls before download

The existing source allowlist explicitly limits candidates to regular Git blobs
with mode `100644`. This control remains unchanged. Future targeted batches must
use `scripts/corpus_targeted_acquisition.py`, which validates every entry against
the pinned tree before the first download. An ineligible later entry prevents the
entire batch from downloading. The helper also checks the repository/revision,
tree hash, blob identity and declared size. Content licensing, privacy, quality,
deduplication and contamination checks remain subsequent mandatory gates.

The first v2 targeted batch checked file modes too late. Eleven SymPy example
files, totaling 41,984 raw bytes, were fetched before discovering their `100755`
modes. They remain preserved raw candidates with explicit rejection records;
none entered the eligible pool. Their executable bit is not a claim that reading
these inert bytes executes them, nor a finding of malicious code. The reason for
rejection is the unchanged explicit eligibility contract.

The seven CPython documents and one SymPy mathematical guide satisfy the mode
control. File-level review and measured downstream outcomes are recorded in
`docs/targeted_file_review_v2_r1.json` and `reports/corpus_expansion_v2_r1.json`.

The remaining two code holdouts require 12,500 final bytes each. Under the unchanged
7/10 policy, they require at least 17,858 raw candidate bytes each. The cached
inventory contains at most 26,336 otherwise potentially usable unused mode-100644
Python bytes after the frozen exclusion scopes, before further rights/quality
review. This optimistic bound is below 35,716; it cannot establish both pools.
The inventory is not an allowlist and does not authorize downloads.

A possible next human decision is whether a separately versioned control may
admit mode-100755 **regular source-text blobs** stored as inert data, retaining
all other gates and rejecting symlinks/submodules. This document does not approve
that exception or restore the eleven rejected files. No new dataset, source pin,
quota, domain allocation or headroom change is proposed.
