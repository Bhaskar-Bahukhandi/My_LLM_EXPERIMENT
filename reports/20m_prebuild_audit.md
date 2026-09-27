# 20M pre-build audit — BLOCKED

Date: 2026-09-27. Decision: **20M_PREBUILD_NOT_READY**.

This is a partial audit stopped under section 29 of the supplied phase prompt:
“STOP immediately for human review if … existing frozen evidence is inconsistent.”
No architecture was promoted and no 20M training occurred. The unresolved historical
evidence binding prevents a READY decision even though the endpoint checkpoint itself
matches its accepted digest.

## Blocking finding and reproduction

`docs/authorization_capacity_2m_r2.json` and `_r3.json` bind
`reports/real_corpus_2m_evidence.json` to a SHA-256 that does not match the current file.
`scripts/authorization_capacity.py:119` correctly rejects it. Seven existing tests fail.

| Identity | SHA-256 |
|---|---|
| Historical contract expects | `6658bb3c6f1e973870c744ad26288880626132dbc6aefb96a0b32c5433ec52e9` |
| Working file, CRLF, 70,753 bytes | `7ff340a96a861538319554883543eaf2eacb299828743379c2504b9ce1a5db76` |
| Accepted commit Git blob, LF, 68,098 bytes | `c9bb670c2f9cc11447f17f58d199f032397732acb73014cbfe3702345878fd65` |

The working file equals the accepted Git blob after CRLF normalization. Git reports
`core.autocrlf=true` and `i/lf w/crlf`. Rendering that blob as CRLF reproduces the working
hash, **not** the contract hash. Thus ordinary checkout line endings explain the
working/blob difference but do not reconcile the contract. The mismatch predates this
audit; those files have no working-tree diff. Its historical origin has not been traced.
This finding does not establish checkpoint corruption or invalidate the measured loss.
It does establish that the repository-wide evidence chain is not presently reconciled.

Reproduce from the existing CPU environment:

```powershell
.\.venv\Scripts\python.exe -m pytest -x -q tests/test_authorization_capacity.py
```

Observed: `1 failed, 13 passed in 0.13s`; first failure is
`test_real_revision_preserves_budgets_and_exposes_remaining_math_holdout_deficits`.
Do not replace the expected digest, regenerate old evidence, normalize frozen files,
or weaken the tests merely to make this pass. Human review should authorize a separate
provenance investigation and an additive reconciliation record.

## Frozen baseline verification completed

- Initial working tree was clean. HEAD, origin/master, and live `git ls-remote` all
  identified `3d2698b42404ec67834ead162db0bd10240d270b`, subject
  `train: finalize frozen 2m gen0 one-pass evidence`.
- Frozen computational source canonical SHA-256 matches the final report:
  `7d98de1b0504a1d511f3da95e17ec45dbcca1c5bd2d0dadbd251987261173eb4`.
- Local endpoint `data/full-training-2m-v1/continuation-v1/attempt-1790232260152557800/checkpoints/step_078167/state.pt`
  hashes to `0d4c822daadf6bb8269a727ac54d54a26e9edf49ff43fedff1b0d91ba6ad2623`,
  matching its manifest and accepted endpoint report.
- Accepted parameter identity remains
  `75b1a342f3acf3b10c7dadb7f1d86feef09642cf15201a282aa7ec8b758be173`.
  This parameter hash is cited from accepted evidence, not recomputed from a fresh load.
- The metadata audit reconciled the 2M executable model at 1,929,579 parameters.
  Published endpoint: 78,167 updates, 10,000,064 target bytes, validation NLL
  2.1701512726 and BPB 3.130866479. Weak generation and unverified effective context
  remain explicitly unpromoted.
- Entry hashes for final reports, project state, all computational source, Bible,
  Roadmap, Addendum and both context reports are in
  [the mechanical receipt](20m_prebuild_mechanical.json). Its `protected_verified`
  section records the subset of final-amendment report/spec/checkpoint hashes checked.
  This subset check is not a claim that every historical authorization binding passed.
- No real corpus payload, including TEST, was opened by the audit helper. Endpoint
  deserialization and reconstruction were not repeated; the content hash matches the
  accepted exact-restore proof. The wider lineage remains blocked by the finding above.

## Source audit completed before the stop

| Component | Fresh finding | Required treatment after reconciliation |
|---|---|---|
| config / resolve | Default widths 128/192/256, depths 2/3/4/6/8/12; exact maximum **5,385,771** | Version an explicit larger search/config. Existing explicit dimensions already work; no model rewrite is required |
| resolve metadata | `executable model not implemented` and tensor-reconciliation next gate are stale | Use evidence-scoped status; do not set all shapes training-ready merely because 2M trained |
| parameters | Real meta Parameters, unique identity accounting, formula cross-check | Reuse; reconcile against executable tensors for each new shape |
| byte layers / hierarchy / dense model | Dimension-driven; completed patches only; BOS out of band; 267 symbols, patch 8 | Preserve model mathematics and frozen source |
| Mamba batch forward | `segment` and `transition` are B×H×T×T; B/C contraction is B×T×T | Quadratic patch-time reference retained as oracle. Recurrent step has fixed state; it is a different execution path |
| Mamba checks | Many finite checks and Python conditionals can synchronize CUDA | Profile before changing monitoring; preserve failure detection |
| Trainer.step / data | Full manifest reread and SHA verification every update; cached byte windows already exist | Design verified immutable snapshot binding, cheap identity checks and boundary verification; metadata-only checks cannot promise identical corruption latency |
| Data RAM | Materialized Window objects and payload copies plus Python permutation lists | Serious-corpus RAM needs indexed/sharded design preserving exact document/window order and cursor semantics |
| Device | Exactly one visible CUDA device, `cuda:0`, FP32; **1 GiB allocator cap**, plus 1 GiB free headroom | Named RTX/T4 budget profiles; mask visible devices before CUDA initialization |
| Memory | Windows RSS only; constructs canonical recurrent state each update; activation memory unmeasured | Linux RSS and measured phase peaks; separate analytic state size from monitoring allocation |
| Checkpoint | Strict source/config/environment identity; Python/CPU/CUDA RNG, cursor and optimizer restored | New 20M lineage, fresh initialization. Cross-platform exactness must be proven, not bypassed |
| Evidence helpers | Older `experiment.py` / Stage-A helpers use correctness asserts and fixed 56-tensor assumptions | New reusable validation helpers with explicit errors; preserve historical scripts |
| README / historical contracts | Current scope omits completed real-data endpoint; some old stage descriptions remain | Add current index/status later; retain historical reports and published paths |

For a byte length L, differentiable full prediction processes
`1 + floor((L-1)/8) = ceil(L/8)` global inputs for L>0, including BOS and excluding
the final completed patch if it cannot influence any requested prediction. Streaming
after consuming L bytes has `1 + floor(L/8)` shared steps. This distinction matters
for state/parity and memory tests.

Binding rules inspected include Bible §§5–9, 31–35, 126, 133–135, 141–142, 154–155,
168; Roadmap invariants and scaling/data formulas; implementation plan; Addendum;
AC-001 through AC-005 and AC-008. Byte causality, two clocks, exact instantiated
accounting, reference fallback and strict checkpoint compatibility are binding.
Illustrative final-model widths/depths and generic scaling ratios are not mandatory
20M choices. The current ladder is 2M→20M→40M→80M→200M→400M; 800M remains research.
The full requested document/maintainability audit was not completed after the stop.

## Exact candidate results — construction evidence only

`scripts/audit_20m_prebuild.py` instantiated the real `DenseByteModel` on the meta
device for 517 grid shapes (11 widths × depths 2..48) and four targeted variants.
For each, executable unique trainable counts, inventory counts, independent formulas
and parameter-shape multisets agreed. It ran no forward/backward or optimizer step.
The receipt retains all 17 within-envelope grid candidates and all four variants,
with full named executable tensors, components, percentages and analytic memory terms.
The other 500 grid shapes were audited but their per-candidate receipts were not retained.

| Family / construction control | d_model | Layers | d_state | Byte / decoder | Exact parameters | Difference |
|---|---:|---:|---:|---:|---:|---:|
| Depth-first | 256 | 46 | 64 | 64 / 128 | 20,074,587 | +0.373% |
| Intermediate depth | 320 | 30 | 64 | 64 / 128 | 20,119,695 | +0.598% |
| Intermediate depth | 384 | 21 | 64 | 64 / 128 | 20,061,951 | +0.310% |
| Balanced | 512 | 12 | 64 | 64 / 128 | 20,114,763 | +0.574% |
| Wider | 640 | 8 | 64 | 64 / 128 | 20,791,275 | +3.956% |
| Width-first | 896 | 4 | 64 | 64 / 128 | 20,242,779 | +1.214% |
| Width-first | 1024 | 3 | 64 | 64 / 128 | 19,807,659 | −0.962% |
| Decoder capacity control | 512 | 12 | 64 | 64 / 256 | 20,387,531 | +1.938% |
| Byte encoder control | 512 | 12 | 64 | 128 / 128 | 20,247,499 | +1.237% |
| State capacity control | 512 | 12 | 128 | 64 / 128 | 20,908,875 | +4.544% |
| State / shallower control | 512 | 11 | 128 | 64 / 128 | 19,188,763 | −4.056% |

All preserve expand=2, headdim=64, d_conv=4, patch_size=8, chunk_patches=16 and
MoE disabled. `d_inner=2*d_model`, heads=`d_inner/64`. The comparison is not a
quality ranking. Deep candidates imply more serial layer work and state; wide
candidates concentrate capacity in fewer layers and change projection costs.
No final selection or authored pilot YAML is issued because the stop gate intervened.

For the balanced construction **only**, the non-overlapping ledger is:

| Subsystem | Parameters | Share |
|---|---:|---:|
| Mamba trunk excluding normalization | 19,828,800 | 98.5783% |
| Symbol embeddings | 17,088 | 0.0850% |
| Position embeddings | 512 | 0.0025% |
| Patch encoder excluding normalization | 41,856 | 0.2081% |
| Bootstrap | 33,280 | 0.1655% |
| Context bridge | 65,664 | 0.3264% |
| Local GRU | 74,496 | 0.3704% |
| Output head | 34,443 | 0.1712% |
| All normalization | 18,624 | 0.0926% |
| Total | **20,114,763** | **100%** |

Weights and gradients would each be 80,459,052 FP32 bytes; Adam moments
160,918,104 bytes, before scalar steps, activations, buffers, allocator or process
overhead. Per batch item: SSM 3,145,728 bytes; convolution 221,184 bytes; decoder
hidden 512 bytes, plus pending symbols. These are tensor arithmetic, **not** VRAM
fit measurements. The old preflight adds a planned context vector that the current
HierarchyState does not persist separately. No RTX/T4 fit category beyond UNVERIFIED
is established by this partial audit.

## Context finding and uncompleted design

The inspected bridge is Linear(d_model,decoder_dim) → tanh → initial GRU hidden;
local byte embeddings then drive GRUCell updates and RMSNorm/output projection.
The accepted studies report differing old-history shared states but identical trained
hidden/logits, while an untrained sanity case transmits history. This is evidence of
output insensitivity, not proof that tanh saturation alone caused it. Scaling the
trunk while holding decoder width at 128 does not establish a remedy.

Context-bridge A/B policy, sequence curriculum, smoke/pilot/serious data budgets,
LR range experiment, exact pilot microbatch/accumulation, full memory ledger,
20M promotion thresholds, generation diagnostics and final backend choice are
**NOT FINALIZED**. Throughput for current/cleaned reference on RTX, cleaned reference
on T4 and optimized T4 is **NEEDS_MEASUREMENT**. The accepted 2M continuation's
165.36 target bytes per measured update-second is historical evidence, not a 20M
benchmark or an end-to-end throughput estimate.

## Research and Kaggle boundaries preserved

The reference Mamba must remain available. Current
[Triton compatibility](https://github.com/triton-lang/triton#compatibility) lists
NVIDIA compute capability 8.0+; the
[T4 hardware specification](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet.pdf)
identifies Turing and 16 GB. An optimized upstream Mamba/Triton path must therefore
not be assumed to work on the intended T4 profile. Kernel compatibility and forward,
state, gradient, causality and seeded-update parity remain untested. No optimized
library was installed. Whether optimization is mandatory for a bounded first pilot
has not been decided; serious-training speed must be measured.

Research consulted before stopping: [Mamba-2/SSD](https://arxiv.org/abs/2405.21060),
[MEGABYTE](https://arxiv.org/abs/2305.07185),
[MambaByte](https://arxiv.org/abs/2401.13660),
[BLT](https://arxiv.org/abs/2412.09871),
[TinyStories](https://arxiv.org/abs/2305.07759), and
[compute-optimal scaling](https://arxiv.org/abs/2203.15556).
These are prior art, not validations of this implementation or a byte-native 20M
optimum. No SOTA or novelty claim is made.

The supplied Kaggle-Bridge status is accepted as user-provided evidence at commit
`4cb387d340b68e9e210bbea1effcaa03108806e1`: HOST_READ verified; private writes,
notebook execution and T4 availability unverified; compute disabled. The bridge
was not independently inspected or invoked. No Chrome interaction, upload, host
Kaggle read, write, compute, or sandbox-network fallback occurred.

Any future plan must keep LOCAL_ONLY / HOST_READ / HOST_WRITE / HOST_COMPUTE typed,
host execution explicit and approval-bound, OAuth credentials isolated, and no
arbitrary shell bridge. Read authorization does not grant write or compute authority.
First target remains one visible T4; dual T4 is future design only. Kaggle's
[notebook documentation](https://www.kaggle.com/docs/notebooks) describes 12-hour
CPU/GPU sessions and up to 20 GB saved outputs; account availability and persistence
must be reverified before execution. Local checkpoint creation alone must not be
treated as durable export. No execution-ready notebook or portability claim is issued.

## Validation, preservation and next decision

Full existing CPU-environment `pytest -q`: **7 failed, 264 passed, 7 skipped,
1 warning in 110.54s**. All seven failures arise in historical authorization
capacity tests from the evidence-hash mismatch (one CLI failure surfaces as missing
JSON output). The seven CUDA tests skipped because this interpreter is CPU-only;
this is not a claim that the host has no GPU. Existing synthetic fixture tests
performed their own disposable updates; frozen 2M was not resumed and 20M was not trained.
The pre-existing optional NumPy warning remains. Static/dependency/diff receipts
are recorded in `20m_prebuild_validation.json`.

Files added: this Markdown report, JSON decision report, mechanical receipt,
validation receipt and the metadata-only audit helper. `docs/project_state.md`
receives only a blocked-status entry, retaining its historical content. No `src/`,
existing tests/configs/reports, weights, corpus, Bible, Roadmap or Addendum was edited.
No cleanup, backend refactor or training change was implemented. New local synthetic
test artifacts remain ignored in `evidence/`; they are not published.

Fallback: accepted commit and all existing weights/evidence remain available; the
reference Mamba and original 2M configuration are unchanged. Reverting this audit's
additive commit removes its reports/helper and state entry without changing the
accepted baseline. Risk: changing historical hashes without explaining provenance
would conceal this failure. Incomplete design must not be mistaken for pilot authority.

Next best step: human review of the historical binding mismatch, followed by an
authorized read-only Git-history/provenance investigation and additive reconciliation.
Then rerun the failing tests and resume the remaining 20M design, hardware/memory,
data, context, backend, generation and promotion-gate work. The source findings above
are a preliminary refactor inventory, not permission to bypass the evidence gate.

20M PRE-BUILD STATUS: BLOCKED
