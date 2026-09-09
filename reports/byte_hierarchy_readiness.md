# Byte hierarchy evidence and readiness

The byte hierarchy is ready to receive Mamba-2. This tranche stops before any shared recurrence. The accepted candidate stays at 1,929,579 structural parameters (-3.52105% from 2M), with its unchanged 2% tolerance and explicit smoke-stage exception.

## Git, files and rollback

Clean foundation commit: **5265179e84e3b3fcc38eb22528fcb349c78f7fee** on existing master; no remote. All 29 committed files were checked byte-for-byte against the pre-byte snapshot plus reviewed .gitignore. Normal index/object/ref writes succeed with command-local safe.directory=E:/Theai. Windows still denies replacing the old HEAD or changing its ACL; neither is needed for the verified commit path. No global configuration or security setting was changed.

The ignored archive evidence/byte_hierarchy/rollback/configuration_foundation.zip has SHA-256 792bb2e693a7276e9bdaf8718df3348e87706f45654d424242d6ba8d11698627. A manifest, original ACL SDDL and foundation-commit verification sit beside it. The source foundation and snapshot provide rollback; byte changes remain uncommitted for review. The reviewed .gitignore excludes the local environment, caches, builds, large evidence, datasets and checkpoint formats.

Added: src/unified_edge/{symbols,byte_layers,byte_hierarchy}.py; tests/test_symbols.py; tests/test_byte_hierarchy.py; docs/byte_hierarchy_contract.md; reports/{mamba_reference_verification,byte_hierarchy_evidence}.json; this report. Updated: README.md, docs/backend_reference.md, docs/project_state.md. The .gitignore update is already in the foundation commit. Existing configuration code, tests, configs, resolved snapshot and dependency lock are unchanged.

## Symbol, bootstrap and patch contracts

Version 1 derives controls from the existing canonical registry: raw bytes 0..255; PAD 256; BOS 257; EOS 258; tool_call 259; tool_result 260; rag_begin 261; rag_end 262; code_begin 263; code_end 264; error 265; retry 266. Vocabulary: 267 entries, including 256 bytes, one padding entry and ten other controls. Binary conversion performs no Unicode decoding or normalization. Unknown IDs fail specifically. PAD is storage only, BOS is out-of-band conditioning, and both are masked from predictions. EOS and remaining controls occupy slots; the caller owns stopping/protocol interpretation.

The first conditioning vector is a learned projection of the shared BOS embedding. The local decoder starts at tanh(context_projection(context)), predicts a symbol, and only then consumes its embedding through a GRUCell. Completed encoding uses local positions, left-padded depthwise convolution, a gate, RMSNorm, mean pooling over all eight positions and projection.

Only eight observed symbols emit a patch event. Complete count is floor(L/8); explicit padded storage uses ceil(L/8). Partial symbols stay local. At a completed event the hierarchy waits for real external conditioning; no dummy shared state or substitute trunk exists. Synchronous batches require caller grouping for ragged lengths. Local state snapshots are detached inference state, require identical model weights, and require separately persisted external context/outbound events at a waiting boundary.

## Validation and numerical evidence

- Full suite: **118 passed, 0 failed, 0 skipped**, including all unchanged 81 foundation cases; 21.091 seconds. One optional NumPy warning.
- Ruff lint and formatting check, Python compileall and pip check: PASS.
- Required lengths 0,1,2,7,8,9,15,16,17,127,128,129 plus randomized lengths passed. All 256 bytes, arbitrary binary, invalid UTF-8, zeros, high bytes, controls, padding, invalid IDs, shape errors, partial state and serialization are covered.
- Independent GRU/RMSNorm and convolution/gate/pooling equations, future-input gradient checks, cross-patch isolation and all-position encoder gradients provide independent oracles.
- Adversarial future suffixes cover the first patch, byte-8 boundary and 127/128/129 boundaries. Across seeds 19,27,73, maximum prefix-logit difference was **0**, restored-prefix difference **0**, and independent GRU logit error **9.5367431640625e-7**. Declared FP32 tolerance is atol=1e-6, rtol=1e-5; adversarial equal-prefix checks require exact equality.
- Multi-patch tests use an explicitly test-only causal accumulator. These results establish hierarchy causality under causal caller conditioning; they do not establish Mamba behavior.
- Actual tensor execution, finite backward gradients for every parameter, and weights/state save/restore passed. No source tests were weakened. Full new-source/test review found no placeholder production systems, swallowed exceptions or dead code.

Machine-readable measurements and source/config hashes: [byte_hierarchy_evidence.json](byte_hierarchy_evidence.json). Full-suite XML and additional measurement script are preserved under ignored evidence/byte_hierarchy.

## Partial parameter reconciliation

Counts are actual unique trainable nn.Parameter objects; implemented groups have no buffers or aliases. Per-tensor shapes also match the structural inventory.

| Component | Structural | Executable | Difference | Reason |
|---|---:|---:|---:|---|
| Embedding | 17,600 | 17,600 | 0 | Exact |
| Completed-patch encoder | 25,280 | 25,280 | 0 | Exact |
| Shared Mamba | 1,728,096 | Not instantiated | N/A | Outside tranche |
| Local decoder | 107,520 | 107,520 | 0 | Exact |
| Output head | 34,443 | 34,443 | 0 | Exact |
| BOS bootstrap | 16,640 | 16,640 | 0 | Exact |
| Implemented subtotal | **201,483** | **201,483** | **0** | Fully reconciled |
| Eventual full inventory | **1,929,579** | **Not yet available** | N/A | Shared recurrence absent |

## Pins, warnings and remaining gates

No dependency changed. Tested environment: project-local Python 3.12.14, torch 2.6.0+cpu, PyYAML 6.0.2, pytest 8.3.5 and Ruff 0.11.13; exact transitive pins remain in requirements-lock.txt. NumPy is absent: the optional tensor-to-NumPy bridge explicitly reports unavailable. The hierarchy uses no such bridge; supported tensor/autograd/serialization paths passed without suppressing the warning.

Official semantic reference remains state-spaces/mamba tag v2.2.6, commit d7b1ceb3c367ec9022925e812f507bcf706937c6. Local git ls-remote verified the relationship on 2026-09-09 at 00:09:05 UTC. The immutable release preserves auditable parameter/state semantics compatible with the declared generic subset. It is an engineering reference, not a Bible mandate or installed implementation dependency; see [verification record](mamba_reference_verification.json) and ../docs/backend_reference.md.

No FINAL specification changed; existing correction proposals remain intact. Same-chunk routing is still deferred. An initial endpoint encoder ignored five positions; independent gradient tests exposed the implementation defect and mean pooling corrected it without altering counts. Explicit-context boolean batch validation was also corrected. Neither changes a FINAL requirement.

No blockers remain for this byte boundary. Memory preflight remains UNKNOWN_REQUIRES_MEASUREMENT: full-model runtime memory, Mamba parity, GPU execution, fresh installation, cross-platform behavior and training are not validated. External conditioning must remain causal and each completed event must be processed exactly once. The next authorized tranche must implement/reconcile the shared recurrence and own its state; no later system was added here.

BYTE HIERARCHY STATUS: READY FOR MAMBA INTEGRATION