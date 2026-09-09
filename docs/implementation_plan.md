# Unified Edge implementation plan

Baseline read in full on 2026-09-08: Bible v3.0 (7,581 lines) and Roadmap v1.0 (1,780 lines). The current user directive supersedes the older scale ladder: **2M → 20M → 40M → 80M → 200M → 400M**, with 800M optional. This session stops at the 2M readiness boundary. Calendar milestones are not claimed complete merely because a component exists.

## Reconnaissance

- Initial workspace: the two specification documents only; no source, configuration, tests, datasets, manifests, generated artifacts, hidden project directories, or Git repository. Nothing deleted.
- Bible SHA-256: `754e99e9feca52c8744b71891d6e71993233dea43c3d7155900334e28c121f75`.
- Roadmap SHA-256: `91bb81b09162e8ccdbdd2baa5bcd5c849a912bfe15e53f798c09bd70b788af99`.
- Windows; NVIDIA RTX 2050, 4,096 MiB VRAM, driver 596.21. GPU hardware visibility is not proof of a working PyTorch/CUDA backend.
- `python` unavailable on PATH. Store Python 3.13 launcher fails with access denied. Bundled Python 3.12.14 executes, with no torch, Mamba, Triton, pytest, PyYAML, psutil, or Ruff installed. Use an isolated project environment, not changes to the bundled environment.
- CIM CPU/RAM queries denied. Collect portable process/hardware information through Python where possible; retain unavailable fields explicitly.
- No existing implementation to preserve or compare. No training data supplied; controlled fixtures will establish mechanics, not language/code capability.

## Minimum architecture and boundaries

The first stage is a **dense hierarchical byte Mamba-2 model**, supported by Bible §§98 phases 1–2, 142 stage 0, 152.1 and Roadmap W05–W10 (W08 explicitly permits 1M–5M). MoE is disabled and rejected if requested until its own implementation gate. This is a staged foundation for the same configuration-driven family, not a claim of completed MoE.

Data path: versioned bytes/control IDs → shared byte embedding → completed non-overlapping 8-symbol patch encoder → pre-normalized residual Mamba-2 shared layers → local autoregressive decoder → unified symbol head. Routing chunk geometry is 16 patches, recorded for future compatibility; dense execution has no routing decisions. BOS conditions the shared initial context before any payload patch. A prediction at position t consumes only symbols before t. Partial patches stay in local state and never advance the shared clock. Padding masks loss and must not silently become EOS or payload. Controls and byte conversion semantics are documented and tested.

Engineering selection (not a Bible version requirement): use Mamba-2 v2.2.6 at commit d7b1ceb3c367ec9022925e812f507bcf706937c6 as the semantic reference; rationale and supported subset are recorded in docs/backend_reference.md. Implement a small differentiable PyTorch reference path without custom kernels, restricted to a declared supported subset (full SSM width, one group, no distributed path). Classify it VALID_GENERIC_PATH; do not claim CUDA/Triton fast-path validation. Independent equation/full-scan tests and upstream comparison where available are mandatory. CPU FP32 is the initial explicit validation profile; GPU/low precision are separate evidence gates.

Module ownership:

- `configs/`: authored YAML, immutable resolved dataclasses, legality and deterministic budget resolution.
- `model/`: byte/control contract, encoder, decoder, Mamba recurrence/state, integrated model; no training or file I/O inside forward.
- `training/`: manifest-backed byte batches, objectives, optimizer/scheduler, finite checks, checkpoint/resume, run records.
- `eval/`: byte-weighted loss/perplexity and generation evaluation.
- `runtime/`: byte streaming/sampling and measured memory where implemented.
- `scripts/`: resolve/audit/preflight, bounded smoke/benchmark and dataset-manifest commands.
- `tests/`: config, causality, equations/state, data integrity, training/checkpoint/resume, CLI integration.
- `docs/`, `reports/`: plans, decisions, operational ledger and measured evidence. Runtime artifacts go to ignored `evidence/`; retain compact reproducible measurement summaries in reports.

No empty RAG/tool/verifier/quantization directories. No later-scale implementation copies or speculative backend adapters.

## Ordered dependency graph and implementation tranches

1. This plan → state ledger → explicit architecture/tensor/control/backend decision record (Bible §§123–136, 163–168; Roadmap W01).
2. Typed authored/resolved schemas → reject unknown fields/versions → derived dimensions and backend legality → formula screening → exact instantiated parameter ledger → memory preflight. Target 2,000,000, tolerance 2%; legality first, then tolerance and known memory rejection, then deterministic target-distance ranking. Unknown activation memory remains UNKNOWN_REQUIRES_MEASUREMENT. Preserve rejected-candidate explanations.
3. Byte contract → completed patch encoder → local causal decoder → reference Mamba-2 → canonical state → full integration (Bible §§5–9, 17, 76, 84–87; Roadmap W05–W10). Model construction accepts resolved config only. Explicit state permits continuation at arbitrary byte boundaries, reset and serialization.
4. Masked next-symbol loss → deterministic manifest data pipeline → AdamW with special recurrent parameters excluded from weight decay → warmup/cosine schedule, accumulation, clipping, validation, seed/environment/logging → atomic checkpoint and strict compatibility resume (Bible §§31, 74–79, 129–132, 141; Roadmap W03–W04). No silent device or precision fallback.
5. Generation → evaluation → adversarial and independent numerical tests → static checks → exact 2M audit → forward/backward/update → short controlled overfit → save/resume equivalence → memory/runtime measurement → independent review → readiness report.

The source tree begins only after this plan and state ledger exist. Independent component work may proceed in parallel after interfaces are agreed; integration follows these dependencies.

## Tests and predeclared gates

- Quick tests vary batch sizes, widths, sequence lengths (including 0/1/7/8/9/127/128/129), seeds and recurrent split points. Byte IDs 0–255 round-trip exactly, including invalid UTF-8; unknown IDs reject.
- Adversarial prefix/suffix perturbation: predictions for an identical prefix remain equal at every tested byte, patch and chunk boundary. Test restored states/checkpoints too.
- Independent recurrence oracle and batch/full versus stepped processing: FP32 `atol=1e-5, rtol=1e-4`; FP64 equation checks tighter. Report maximum observed errors. Do not rely solely on comparing a loop with itself.
- Every trained parameter/gradient/loss finite; actual optimizer updates occur. Loss on a tiny deterministic training corpus should fall by at least 50% in the bounded overfit experiment. Held-out data remains distinct; overfit is not held-out quality evidence.
- Compare uninterrupted and interrupted/resumed training with the same total schedule, exact data cursor and RNG state. Restore model, optimizer, scheduler, scaler where applicable, global step, counts and manifest/config identity. Reused output directories or incompatible checkpoints fail specifically.
- Evaluate weighted symbol loss, raw-byte loss/perplexity separately from controls; never label mixed-symbol metrics raw-byte perplexity.
- Exact parameter ledger matches unique instantiated trainable tensors and falls inside the configured band. State allocation bytes must match tensors; RSS is measured independently. Benchmark with fixed warmup/repeats and all samples, hardware, precision and sizes recorded.
- Training-readiness requires the supported local profile to pass architecture, causal, forward/backward/update, overfit, evaluation, generation, checkpoint/resume, finite, metadata and memory measurement gates. Unsupported hardware is explicitly unverified. No 20M promotion without a user-trained 2M checkpoint, curves and full evaluation review.

## Research issues and specification reconciliation

- Same-chunk mean routing (Bible §§11, 82) can leak future patches into earlier predictions if dispatched back into the same chunk. Dense 2M defers this; resolve before MoE by a separate evidenced proposal.
- Roadmap §10.3 labels ceil(L/P) complete patches; truly complete patches use floor and ceil is padded storage count. Preserve original document; record correction separately.
- Roadmap §12.16 omits headdim when H denotes number of heads. State tensor size follows Bible §44: B × layers × heads × headdim × d_state, plus convolution state. Record a mathematical correction proposal before using the corrected ledger.
- Roadmap §22.8 weighted search cannot override Bible §§163/168 legality/memory priority. Hard gates precede distance ranking.
- Roadmap W14 serialized tensor counts need distinction among parameters, buffers and aliases; unique trainable tensors determine model size (Bible §§18, 133).
- Remaining research: hierarchy information loss, long-context state retention, held-out learning, dense-versus-MoE benefit, GPU parity, low-precision recurrence. None is settled by a tiny overfit run.

## Artifacts, risks and rollback

Expected: `configs/models/edge_2m.yaml`, tested source and CLI, installation pins, README, resolved config and parameter/memory reports, immutable dataset/run manifests, structured metrics, checkpoint hashes, short overfit and resume evidence, `reports/edge_2m_readiness.md` with an honest READY/NOT READY decision.

Initialize local Git without remote or rewriting history, preserve original specs, and make staged additive changes. No pre-existing checkpoint exists; the first validated smoke checkpoint becomes the mechanical rollback anchor. Never overwrite a checkpoint or run. Source hashes cover uncommitted runs. Keep the original specifications byte-identical; proposal documents record factual corrections without editing the Bible. If dependency downloads fail, finish independently verifiable work and label runtime validation BLOCKED. If equations, causality or resume fail, stop scaling and fix the cause. No long training, dataset downloads, paid resources or deployment in this session.

## Bounded continuation review — 2026-09-08

The current user instruction limits this tranche to Git/environment, typed authored/resolved configuration, legality, deterministic candidate resolution, exact structural parameter auditing and preliminary preflight. Executable byte/Mamba/decoder/training work is deferred. Approximately 2M and its nominal 2% tolerance are subordinate to architectural/backend legality; if no clean candidate fits, report the closest clean legal candidate without silently widening tolerance. Correction proposals AC-001 through AC-005 precede dependent accounting. Inventory-v1 is a provisional structural contract and must be reconciled against the future executable model; it is not a model implementation or training-readiness claim.