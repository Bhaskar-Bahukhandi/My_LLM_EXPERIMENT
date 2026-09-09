# Project state

- CURRENT STAGE: bounded configuration/parameter/preflight tranche validated; stopped before executable byte-model work.
- CURRENT MODEL SCALE: declared dense inventory 1,929,579 parameters; target 2,000,000, delta −70,421 (−3.52105%). OUTSIDE_TARGET; nominal 2% tolerance unchanged.
- LAST VERIFIED COMMIT: none. Local Git initialized, unborn master, no remote; metadata writes denied because .git belongs to sandbox account. Read-only checks use command-local safe.directory. No commit/staging claimed. Source hashes: reports/configuration_evidence.json.
- PASSING TESTS: 81 passed, 0 failed/skipped; Ruff lint/format, compileall, pip check pass. CLI determinism, strict schema, legal/illegal candidates, real instantiated metadata counts, independent PyTorch component counts, alias/buffer/serialization accounting and state-memory axes verified.
- FAILED TESTS: no remaining test failures. Initial lint findings corrected. Git branch metadata write failed with permission denied. Standard sandbox helper still cannot launch commands; narrow reviewed commands used.
- KNOWN ISSUES: default candidate misses 2% band; memory preflight UNKNOWN_REQUIRES_MEASUREMENT; optional PyTorch NumPy-import warning; no executable model, dataset or checkpoint.
- OPEN RESEARCH QUESTIONS: same-chunk routing schedule, future byte/control framing and causality, executable-vs-inventory reconciliation, actual runtime/state cost, upstream numerical parity.
- ACTIVE EXPERIMENTS: none. Structural candidate census and tests completed; no training run.
- ARCHITECTURE CHANGES: FINAL Bible/Roadmap unchanged (original hashes verified). AC-001/002/004/005 accounting/math/precedence clarifications adopted; AC-003 causal routing remains deferred. Reference v2.2.6 / d7b1ceb3c367ec9022925e812f507bcf706937c6 is an engineering pin, not a Bible requirement.
- BLOCKERS: Git metadata permission/ownership for checkpointing. Final-model count and runtime memory cannot be validated until the next authorized byte-model tranche. Fresh install, CUDA and non-Windows execution unverified.
- NEXT PROMOTION GATE: restore Git writability and checkpoint this foundation; implement byte hierarchy/model only in the next tranche and reconcile every parameter component against inventory v1. No training readiness and no 20M promotion.
- FALLBACK/ROLLBACK: original specifications preserved; changes additive; no prior checkpoint exists; reports/edge_2m_resolved.json preserves the selected inventory config. No system/global Git settings or user data altered.

Detailed evidence and decision: reports/configuration_tranche.md.
