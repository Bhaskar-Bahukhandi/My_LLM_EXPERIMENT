# Revision-3 preflight: quota clarification required

The new directive requires unchanged budgets and allocations but specifies **50,000 mathematical bytes per SymPy holdout**. The accepted contract instead allocates each 50,000-byte SymPy holdout as 12,500 code + 12,500 documentation + **25,000 mathematics**.

Applying the new mathematical number without changing the other cells would produce 75,000 SymPy bytes and 525,000 overall bytes in each holdout, exceeding both frozen budgets. At the unchanged 70% retained-yield estimate, the accepted mathematical quota requires 35,715 raw bytes; the new number requires 71,429. Neither quotas nor margin were edited to hide this disagreement.

A clarification was requested: retain 25,000 mathematical bytes per holdout, or explicitly revise budgets. No response has been assumed. The current revision still has 58/60 passing cells.

The 24 focused authorization/fingerprint tests pass (0.34s). The accepted invariant source and tests retain their recorded hashes. Cached pinned SymPy tree metadata was inspected for solving guides; no new files were downloaded, admitted or assigned to holdouts. Content/license and cross-group duplicate review of new candidates remains unperformed.

Changed files: this preflight report, its JSON evidence, and the current project ledger. Revision-2 reports/allowlists, the original failures and rollback `8c5cb375fced7cf1452f3007514a32cbc9678922` are preserved. Model and environments remain unchanged. The next step is to resolve the quota contradiction before continuing the two-cell repair.

CORRECTED DOWNLOAD AUTHORIZATION STATUS: NOT READY
