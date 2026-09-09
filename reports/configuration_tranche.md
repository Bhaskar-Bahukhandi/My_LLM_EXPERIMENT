# Configuration and accounting tranche — 2026-09-08

**Decision: configuration/accounting is validated for the next byte-model implementation tranche. Git checkpointing is BLOCKED by repository metadata permissions. The nominal 2% parameter band is NOT met. No training readiness or executable-model claim is made.**

## Completed scope

Project-local Python environment and installable src package; typed authored/resolved contracts; strict version/type/unknown-field/duplicate-key checks; explicit CPU structural backend profile; deterministic candidate enumeration; unique instantiated parameter auditing; alias/buffer/serialization accounting; preliminary memory screening; CLI and regression tests. No byte encoder, decoder, Mamba recurrence, training, RAG, tools, quantization, latent reasoning or later-scale implementation was added.

Git was initialized in the prior continuation and remains on unborn master with no remote or commits. The requested .gitignore excludes environment/cache/build/evidence artifacts. Read-only Git inspection uses a command-local safe.directory setting. Changing the unborn branch to main failed with permission denied; no ownership, global configuration or ACL change was attempted. Source remains untracked; no staged/committed diff is claimed.

## Selected candidate

18 declared clean width/depth candidates were evaluated. None fits 2,000,000 ±2%. The closest in this search is **1,929,579**, delta **−70,421 (−3.52105%)**; status OUTSIDE_TARGET. This does not claim a global optimum over every possible architecture. Fixed local dimensions and the target tolerance were retained. A legal width 256 with four shared layers is preferred to arbitrary parameter inflation or unsupported shapes.

| Dimension | Resolved value |
|---|---:|
| d_model / shared layers | 256 / 4 |
| byte_dim / decoder_dim | 64 / 128 |
| expand / d_inner | 2 / 512 |
| d_state / d_conv | 64 / 4 |
| headdim / nheads | 64 / 8 |
| patch size / chunk patches | 8 / 16 |
| unified symbol count | 267 |
| MoE | disabled, unsupported |

| Component | Unique trainable inventory parameters |
|---|---:|
| Shared | 1,728,096 |
| Decoder | 107,520 |
| Output | 34,443 |
| Encoder | 25,280 |
| Embedding/positions | 17,600 |
| Bootstrap | 16,640 |
| **Total** | **1,929,579** |

The inventory has 56 actual metadata-device Parameters, zero frozen parameters, zero buffers and zero aliases. Dense active inventory count equals total inventory count; runtime activity is not measured. The next tranche must reconcile executable tensors with the declared inventory before calling this a model parameter count. Audits on independent real nn.Modules prove tied-parameter, frozen-parameter, buffer, nonpersistent-buffer and non-tensor state metadata handling.

## Dependency and reference pins

- Python **3.12.14**, existing bundled executable used only to create project .venv; no system environment upgraded.
- PyTorch **2.6.0+cpu**, official CPU wheels; used for real Parameter metadata and independent primitive accounting.
- PyYAML **6.0.2**, strict authored YAML loading; pytest **8.3.5** and Ruff **0.11.13**, test/static validation.
- Setuptools **78.1.0**, local editable build; full transitive lock in requirements-lock.txt.
- Mamba semantic reference **v2.2.6**, commit **d7b1ceb3c367ec9022925e812f507bcf706937c6**. GitHub's tag target and immutable source hash were independently rechecked. Source SHA-256: **605e4439ff0baec8d8acaf4a191d9f0570eea9900065a065909124c472b08707**.

The Mamba pin is an engineering reproducibility decision, not a Bible version mandate. Its default single-process tensor layout and native recurrence-state shapes are inspectable. No official Mamba package, Triton or causal-conv1d was installed. Executable upstream parity is unvalidated and outside this tranche.

## Validation

- `python -m pytest -q --junitxml=evidence/config_tranche/tests_final.xml`: **81 passed, 0 failed, 0 skipped**, 17.95 seconds (pytest display).
- `python -m ruff check src tests`: PASS.
- `python -m ruff format --check src tests`: PASS, 9 files.
- `python -m compileall -q src tests`: PASS.
- `python -m pip check`: PASS.
- Default CLI: expected exit **2**, OUTSIDE_TARGET; all four exit/result paths covered by tests.
- Deterministic resolution across repeated runs, reordered candidate enumeration and different process hash seeds; immutable round-trip resolved metadata; RNG state unchanged by inventory allocation.
- Independent nn.Linear/Conv1d/RMSNorm/GRUCell structural counts agree with inventory components across four shape families. Corrected state-axis accounting agrees with actual metadata tensors across batch/head/state variations.
- Both original FINAL specification SHA-256 hashes match the initial baseline exactly.
- Source/test/config/lock hashes and runtime identity: reports/configuration_evidence.json. Raw test XML and rejection examples: evidence/config_tranche/. Source, tests and documentation were reviewed, with per-file added-file diff and whitespace checks because Git staging is unavailable.

One warning remains: PyTorch reports unavailable optional NumPy initialization. NumPy conversion is not used in this tranche. The warning was not suppressed. Initial lint failures were corrected; no lint failure remains.

## Rejections and memory priority

- Width 130, expand 2, head 64: d_inner=260 is not divisible by 64 → INVALID.
- Width 160, head 32: mathematically valid head division, but violates the declared width-multiple-64 search policy → INVALID with policy-specific explanation (not labelled a universal backend law).
- patch_size=4, chunk_patches=8, enabled MoE or CUDA/BF16 requests → explicit unsupported-contract rejection.
- int64-overflow tensor dimensions and excessive inventory depth → rejected before metadata construction.
- A 1-byte memory budget rejects all candidates with explicit known-payload reasons.
- With widths [192,256], depth 4 and budget 19,391,080 bytes, the closer width-256 candidate is rejected for known memory overflow and width 192 is selected. Parameter closeness cannot override memory rejection.

Default preflight: **UNKNOWN_REQUIRES_MEASUREMENT**.

| Planned payload | Bytes |
|---|---:|
| FP32 parameter payload | 7,718,316 |
| Gradients | 7,718,316 |
| Adam first/second moments | 15,436,632 |
| SSM state, including headdim | 524,288 |
| Convolution state | 40,960 |
| Planned local/context/pending-symbol state | 1,592 |
| Known inference payload | 8,285,156 |
| Known training payload | 31,440,104 (~29.984 MiB) |

These are declared storage payloads, not resident RAM. Activations, transient buffers, allocator/runtime overhead, serialized file size, peak process RSS and actual optimizer execution remain unmeasured. There is no configured memory ceiling in the default request. Passing a known-payload bound never becomes FITS.

## Specifications, risks and rollback

AC-001 through AC-005 record the complete-patch floor/ceil distinction, missing headdim factor, same-chunk routing hazard, hard-gate search precedence, and unique/serialized parameter distinction. The routing proposal remains deferred; the accounting/precedence clarifications are adopted for this tranche. Both FINAL documents remain byte-identical.

Additional boundary: exact final model auditing conflicts with forbidding model implementation if represented as a completed-model claim. This tranche resolves that by explicitly auditing only a declared parameter inventory. Proposed local component shapes are documented engineering choices under the Bible's allowed components, not implemented functionality.

Risks/remaining blockers: executable tensor reconciliation; Git metadata permissions; outside-2% candidate status; unknown actual runtime memory; no clean-environment reinstall or non-Windows validation. CUDA, numerical Mamba parity, byte causality, training and checkpoint tests are not part of the completed evidence. No model/checkpoint exists to roll back. Changes are additive, original specs preserved, generated artifacts isolated, and the immutable resolved snapshot provides a configuration rollback reference. No user data was deleted.

**Next best step:** restore normal workspace/Git writability and create the first reviewable checkpoint, then implement the byte-model tranche against this inventory contract. Preserve the outside-target status unless an explicitly revised clean search policy produces a better candidate. Stop here; do not start model implementation or training in this tranche.
