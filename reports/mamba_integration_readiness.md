# Mamba integration readiness

The generic CPU FP32 dense architecture is ready for training-infrastructure implementation. This is architecture and gradient-mechanics readiness, not training readiness, quality evidence, long-context validation or permission to start training. This tranche stops here.

## Rollback and files

Starting approved byte-hierarchy commit: **78c179bda571237d3c0d9aa6ec396bf3f6374b0b**. Before implementation the accepted diff was reviewed, all 118 tests and static/dependency checks passed, and the byte tranche was committed as one coherent commit. The working tree was clean at that boundary. The earlier configuration commit 5265179e84e3b3fcc38eb22528fcb349c78f7fee was preserved; no remote or global/security configuration was added. Current integration changes remain uncommitted for review and can be isolated from that rollback point.

Added: src/unified_edge/mamba.py, src/unified_edge/dense_model.py, tests/test_mamba.py, tests/test_dense_model.py, tests/measure_mamba.py, docs/mamba_integration_contract.md, reports/mamba_integration_evidence.json and this report. Updated: README.md and docs/project_state.md. Accepted configuration and byte implementations/tests, dimensions, dependency pins, resolved snapshot and both FINAL specifications are unchanged. Large source copies, XML and local diff evidence remain ignored under evidence/mamba_integration.

## Reference and supported semantics

Official repository: [state-spaces/mamba](https://github.com/state-spaces/mamba), tag **v2.2.6**, immutable commit **d7b1ceb3c367ec9022925e812f507bcf706937c6**. This remains the approved engineering/reproducibility reference, not a Bible version mandate or an installed dependency. Upstream sources retrieved at 2026-09-09T01:39:57.196521+00:00; the previously verified tag/commit relationship is preserved in mamba_reference_verification.json.

The native [Mamba2 source](https://github.com/state-spaces/mamba/blob/d7b1ceb3c367ec9022925e812f507bcf706937c6/mamba_ssm/modules/mamba2.py) SHA-256 is 605e4439ff0baec8d8acaf4a191d9f0570eea9900065a065909124c472b08707. SSD, gated normalization, residual-block and license source identities/hashes are also verified and recorded in the JSON evidence. The implementation uses reviewed equations; it does not vendor or import optimized upstream kernels.

Supported subset: d_model=256, four layers, expand=2, d_inner=512, d_state=64, headdim=64, derived nheads=8, d_conv=4, one B/C group, full SSM width, per-head D, no input/output projection bias, biased depthwise convolution, SiLU, gate-before-RMSNorm (eps=1e-5), pre-normalized residual blocks. No final trunk norm beyond the accepted inventory. No distributed, fused, GPU or reduced-precision path. The dense wrapper is this project's architecture, not the upstream language-model wrapper.

The step path uses A=-exp(A_log), dt=softplus(dt_raw+dt_bias), S'=exp(dt*A)*S+dt*x*B, then contraction with C, D skip, gating, norm, output projection and residual. The independent O(T^2) full path uses batched convolution and the dense causal SSD identity, not the step loop. Initialization follows the approved upstream defaults. dt_bias/A_log/D retain no-weight-decay metadata for future optimizer work.

BOS is projected by the existing bootstrap and processed through all four shared layers before first-patch local conditioning. Each completed payload patch is processed once, internally, before the next patch is conditioned. Partial patches never advance the shared clock. The integrated invariant is shared.steps=1+completed_patches, including BOS. Canonical typed shared state owns convolution and SSM tensors; byte state remains separately owned. Reset clears shared history and reprocesses BOS. Snapshots validate schema/config, batch, shapes, dtype/device, finiteness and clock consistency, require identical weights, and are detached inference state.

## Executable reconciliation

All counts use actual unique trainable Parameter identities. There are 56 tensors, zero frozen parameters, zero buffers and zero aliases. Logical state-dict elements equal trainable elements; serialized file size is not inferred from this equality.

| Component | Structural | Executable | Difference |
|---|---:|---:|---:|
| Embedding | 17,600 | 17,600 | 0 |
| Completed-patch encoder | 25,280 | 25,280 | 0 |
| Local decoder | 107,520 | 107,520 | 0 |
| Output head | 34,443 | 34,443 | 0 |
| BOS bootstrap | 16,640 | 16,640 | 0 |
| Shared Mamba | 1,728,096 | 1,728,096 | 0 |
| **Full dense model** | **1,929,579** | **1,929,579** | **0** |

The JSON contains all 36 shared tensors individually: layer, name, expected/actual shape, inventory/actual count and difference. Each layer has 432,024 parameters. No formula or dimension was changed to force agreement. The full candidate remains -3.52105% from nominal 2M, explicitly accepted for smoke work; the 2% tolerance is unchanged.

## Actual recurrent-state allocation

Per layer: SSM [B,8,64,64], convolution [B,640,4], oldest-to-newest raw projected convolution history. Both use FP32. Measured tensor payload and backing-storage bytes agree, with distinct storage per layer. These are live canonical-state tensors, not buffers/parameters, saved-file size, autograd-history size or process RSS.

| Batch | SSM bytes | Convolution bytes | Combined bytes |
|---:|---:|---:|---:|
| 1 | 524,288 | 40,960 | 565,248 |
| 2 | 1,048,576 | 81,920 | 1,130,496 |
| 4 | 2,097,152 | 163,840 | 2,260,992 |

The SSM result matches B*4*8*64*64*4; convolution matches B*4*640*4*4. Overall runtime memory remains **UNKNOWN_REQUIRES_MEASUREMENT**. Dense SSD transition storage is quadratic in patch length; this path is for bounded reference/chunk execution, not a long-sequence efficiency claim.

## Validation and measured errors

**143 passed, 0 failed/skipped**, including all unchanged 118 accepted cases, in 61.611 seconds. Ruff lint/format, Python compile/import checks and pip check pass. The regression suite covers all required geometry, full/step and chunk continuity, saved/restored state, reset, every tensor, first-patch shared bootstrap, one-event/one-update, prefix dependence, integrated causality and gradients.

Shared measurements use (seed,batch)=(7,1),(29,2),(101,4), each with patch-step lengths 0,1,2,7,8,9,15,16,17,31,32 and multiple split points. Integrated measurements use (3,1),(37,2),(79,1), 137 symbols, mutations at 0,1,2,6,7,8,9,15,16,17,62,63,64,65,126,127,128,129, plus serialized continuation at arbitrary partial/complete boundaries. These later boundaries validate dense state integrity; no routing exists.

| Comparison | Maximum absolute error | Maximum relative error |
|---|---:|---:|
| Shared full vs step | 3.21865082e-06 | 0.0482758619 |
| Shared final state | 1.31130219e-06 | 0.178692266 |
| Chunk split vs whole | 2.50339508e-06 | 0.035502959 |
| FP64 equation oracle vs step | 1.88503632e-06 | 0.00122874218 |
| Shared serialized resume vs same split | 0 | 0 |
| Integrated full vs step | 2.2649765e-06 | 0.0292682927 |
| Integrated future-suffix prefix | 0 | 0 |
| Integrated serialized continuation | 0 | 0 |
| Integrated restored final recurrent state | 0 | 0 |

All comparisons pass atol=1e-5 plus rtol=1e-4. Reported relative error divides by max(abs(reference),1e-8); near-zero references explain relative maxima above rtol while absolute differences remain below atol. No tolerance was relaxed. Saved identical execution paths reproduce predictions and final recurrent state exactly. The maximum recurrent-tensor magnitude was 2.97910953; maximum aggregate tensor L2 norm was 59.6529655. All observed outputs/states were finite, excluding deliberate PAD/BOS output masks.

Cross-entropy fixtures (seeds 13,59,97; batch 2; 25 symbols) exercise every one of the 56 trainable tensors: all gradients are present, finite and have nonzero norm. The minimum observed parameter-gradient norm is 5.31823025085032e-6. Per-tensor norms/peaks and objective values are recorded. Full/step gradient agreement is also tested. This is mechanics evidence only; no optimizer, dataset or training pipeline was added.

## Issues, deviations and remaining gates

An adversarial numerical test reproduced overflow in dt*(-exp(A_log)) despite finite factors. Explicit layer-labelled checks now reject it before exponentiation, including sequence segment accumulations. No arbitrary clamp was added. Initial import/format lint findings were corrected. No remaining test failure or unexplained accounting mismatch exists.

Upstream **package-runtime parity is UNVERIFIED/BLOCKED**: mamba_ssm and Triton are absent, and the pinned package imports optimized runtime modules on this Windows CPU profile. Independent mathematical/source-semantic parity is measured; no claim of executing the upstream package is made. Dependencies remain Python 3.12.14, torch 2.6.0+cpu, PyYAML 6.0.2, pytest 8.3.5 and Ruff 0.11.13 with the existing lock. The optional NumPy bridge remains absent and its warning is visible; used tensor/autograd/serialization paths pass.

No FINAL specification change or new accounting proposal is needed. Existing AC-001..005 remain authoritative; same-chunk MoE routing remains deferred. Earlier usage-review interruption did not modify the test file; after explicit continuation the missing tests were written and validated. Nothing remains blocked by that interruption.

Remaining unvalidated work: process RSS/peak activation memory, GPU/reduced precision, upstream package-runtime parity, fresh/cross-platform environment recreation, optimization/learning behavior, long-context retention and training checkpoint semantics. The reference full path is quadratic; streaming inference callers should use inference_mode/no_grad to avoid retaining autograd history. State objects are functional values whose tensors must be treated as read-only. The next authorized work is bounded training-infrastructure implementation with measured memory and optimizer/checkpoint gates, not training or later-size promotion.

Reproduce tests with .venv/Scripts/python.exe -m pytest -q --junitxml=evidence/mamba_integration/full_suite.xml. The numerical runner is tests/measure_mamba.py --output <new-file.json>, invoked through the project Python; it verifies the locally retained pinned source cache and runs static/dependency checks. It refuses to overwrite an existing evidence file. Full compact results: [mamba_integration_evidence.json](mamba_integration_evidence.json). Tensor/lifecycle details: [integration contract](../docs/mamba_integration_contract.md).

MAMBA INTEGRATION STATUS: READY FOR TRAINING INFRASTRUCTURE
