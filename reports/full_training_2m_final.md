# Full 2M Gen-0 continuation review

**2M_GEN0_READY_FOR_REVIEW** — human promotion authority only.

The original 32-byte production run continued from the immutable step-5000 checkpoint. The 64-byte A/B was disposable and NOT promoted. TEST remains sealed.

Endpoint/review step: 78167. Parameter SHA-256: `75b1a342f3acf3b10c7dadb7f1d86feef09642cf15201a282aa7ec8b758be173`.
Checkpoint: `data/full-training-2m-v1/continuation-v1/attempt-1790232260152557800/checkpoints/step_078167`.
Amendment SHA-256: `4b07f33a23a9ecc621aaa1b7d21c0a5ebb6d473593705ac441e55cef75e199ae`.

Validation NLL fell 8.14% from step 5000, with every domain improved. Recent-100 training NLL is 2.066720; validation is 2.170151. Gradients remained finite and required exact restore proofs passed. Greedy generation still degenerates into long space runs; this remains a limited engineering pilot, not evidence of useful language capability.

The distribution monitor uses only 32 fixed validation contexts. Its zero severe cases do not establish the absence of collapse over the whole distribution.

## Validation and domain trajectory

| Step | Aggregate NLL | Bits/byte | General text | Code | Documentation | Mathematics |
|---|---:|---:|---:|---:|---:|---:|
| 5000 | 2.362495392 | 3.408360386 | 2.222098905 | 2.393015102 | 2.322500877 | 3.048170306 |
| 10000 | 2.414324449 | 3.483133910 | 2.271128070 | 2.453133964 | 2.382645456 | 3.080801086 |
| 20000 | 2.286950327 | 3.299371895 | 2.133066223 | 2.341383540 | 2.254125527 | 2.969524959 |
| 30000 | 2.260600047 | 3.261356477 | 2.108424448 | 2.318521821 | 2.221838800 | 2.934815418 |
| 40000 | 2.235053854 | 3.224501112 | 2.100813315 | 2.280467239 | 2.183059254 | 2.870714953 |
| 50000 | 2.211794109 | 3.190944392 | 2.078944674 | 2.254668700 | 2.160045299 | 2.846478049 |
| 60000 | 2.187458375 | 3.155835349 | 2.048825338 | 2.241622782 | 2.141179073 | 2.814631498 |
| 70000 | 2.176495606 | 3.140019417 | 2.041508312 | 2.226584737 | 2.127203386 | 2.800147593 |
| 78167 | 2.170151273 | 3.130866479 | 2.039018865 | 2.217153403 | 2.117555751 | 2.787201266 |

Final validation count: 500000. Domain watch: `[{'step': 10000, 'domains': []}, {'step': 20000, 'domains': []}, {'step': 30000, 'domains': []}, {'step': 40000, 'domains': []}, {'step': 50000, 'domains': []}, {'step': 60000, 'domains': []}, {'step': 70000, 'domains': []}, {'step': 78167, 'domains': []}]`.

## Logical and physical accounting

Previous target bytes: 639,613; new logical bytes: 9,360,451; total: 10,000,064, expected endpoint 10,000,064.
Final cursor: epoch 1, offset 2.
Physical overhead: `{"stage_a": {"updates": 182, "windows": 728, "target_bytes": 23236}, "continuation": {"updates": 0, "windows": 0, "target_bytes": 0, "recorded_seconds": 0}, "combined": {"updates": 182, "target_bytes": 23236}, "note": "Physical replay is excluded from logical learning exposure."}`.
Continuation training summary: `{"continuation_final100_weighted_nll": 2.066719864010811, "clipped_updates": 163, "gradient_norm_range": [0.46791672706604004, 6.470909118652344], "all_finite": true, "measured_update_seconds": 56606.68978950266, "target_bytes_per_measured_second": 165.3594484116228}`.
Measured memory: `{"peak_allocated_bytes": 116423680, "peak_reserved_bytes": 132120576, "minimum_sampled_free_bytes": 3304796980, "process_rss_status": "MEASURED", "peak_observed_rss_bytes": 1200246784, "tensor_payloads": {"parameter_bytes": 7718316, "gradient_bytes": 7718316, "optimizer_state_bytes": 15436856, "canonical_recurrent_state_bytes": 1130496, "recurrent_note": "standalone canonical allocation; not persistent trainer cache", "activation_autograd_bytes": "UNMEASURED_SEPARATELY"}}`.

## Generation and distribution trajectory

| Step | q(space) | p(space) | Entropy | Space top-1 fraction | Severe fraction | Control mass |
|---|---:|---:|---:|---:|---:|---:|
| 10000 | 0.161976 | 0.161974 | 2.688397 | 0.312500 | 0.000000 | 5.77206705e-05 |
| 20000 | 0.174519 | 0.174519 | 2.529193 | 0.437500 | 0.000000 | 1.05956886e-06 |
| 30000 | 0.195822 | 0.195822 | 2.413125 | 0.375000 | 0.000000 | 9.72811854e-07 |
| 40000 | 0.188708 | 0.188708 | 2.477242 | 0.468750 | 0.000000 | 3.72382813e-08 |
| 50000 | 0.189826 | 0.189826 | 2.412290 | 0.468750 | 0.000000 | 8.72124311e-08 |
| 60000 | 0.189235 | 0.189235 | 2.360998 | 0.468750 | 0.000000 | 3.54238356e-08 |
| 70000 | 0.178444 | 0.178444 | 2.394707 | 0.468750 | 0.000000 | 2.96426941e-08 |
| 78167 | 0.183788 | 0.183788 | 2.405323 | 0.468750 | 0.000000 | 4.26462524e-08 |

Final six historical greedy probes (all historical points are retained in JSON):

| Prompt | Output | Longest space run |
|---|---|---:|
| `b''` | `b'                                                                '` | 64 |
| `b'The '` | `b'same                                                            '` | 60 |
| `b'def '` | `b'____                                                            '` | 60 |
| `b'import '` | `b't                                                               '` | 63 |
| `b'class '` | `b'an                                                              '` | 62 |
| `b'x = '` | `b'np.a                                                            '` | 60 |

## History and integrity

Original diagnostic and Stage-A evidence are preserved and referenced in JSON. Stage A included the documented Windows sharing-error recovery and exact deterministic replay. The context study and disposable A/B found no useful older-history output dependence. No curriculum or architecture change was introduced.
Restore proofs: `{"20000": {"status": "EXACT PASS", "step": 20000, "checkpoint_sha256": "ecd2bbd060aa832de2aa57c4d004db3e8b64b6ad9cf7222b1f4fb48fa8fc95f6", "model_optimizer_scheduler_counters_cursor_rng": "EXACT", "six_generations": "EXACT", "optimizer_updates": 0}, "40000": {"status": "EXACT PASS", "step": 40000, "checkpoint_sha256": "277dd8fd37867e29aa14ac5edfdc7397ac6a3cc38e8559e210238d34db5c4d48", "model_optimizer_scheduler_counters_cursor_rng": "EXACT", "six_generations": "EXACT", "optimizer_updates": 0}, "60000": {"status": "EXACT PASS", "step": 60000, "checkpoint_sha256": "48eca8be0079a83c622232cf0ced6b42fe0044e2ccb3393f5f20aa10307d833c", "model_optimizer_scheduler_counters_cursor_rng": "EXACT", "six_generations": "EXACT", "optimizer_updates": 0}, "78167": {"status": "EXACT PASS", "step": 78167, "checkpoint_sha256": "0d4c822daadf6bb8269a727ac54d54a26e9edf49ff43fedff1b0d91ba6ad2623", "model_optimizer_scheduler_counters_cursor_rng": "EXACT", "six_generations": "EXACT", "optimizer_updates": 0}}`.
All historical production artifacts, corpus identities, FINAL specifications and computational source hashes remain unchanged. The original Stage-A progress file is preserved; the versioned continuation index records new production steps.

## Validation receipts

| Check | Exit | Seconds |
|---|---:|---:|
| pytest | 0 | 92.42 |
| ruff | 0 | 0.12 |
| format | 0 | 0.11 |
| compileall | 0 | 0.11 |
| cpu_dependencies | 0 | 0.55 |
| cuda_dependencies | 0 | 0.54 |
| diff | 0 | 0.04 |

## Limitations and next decision

- 32-byte-trained; mechanical recurrent streaming does not certify effective context.
- One seed, one pass, small licensed pilot. No benchmark/TEST or meaningful capability claim.
- 64-byte A/B was disposable and NOT promoted. Only the original step5000 trajectory continued.
- Update wall intervals include original corpus verification and diagnostics; journal, checkpoint and evaluation I/O excluded.
- Separate activation memory and end-to-end study duration not instrumented. CUDA peaks may include earlier evaluation allocations.
- RSS is sampled actual process working set, not tensor arithmetic or a continuously sampled maximum.
- Unpublished in-flight physical work after an interruption cannot be reconstructed; durable replay is reported separately.
- Optional unused NumPy bridge warning remains; no environment change.
- Endpoint readiness means reviewable engineering evidence, not automatic model promotion.
- Training summed NLL is reconstructed as the unchanged trainer mean times valid target count; it is not a separately captured pre-division accumulator.

Next action requires human authorization. No TEST evaluation, 20M or future-capability implementation.

FULL 2M GEN-0 STATUS: READY FOR HUMAN REVIEW
