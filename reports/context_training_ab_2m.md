# Disposable 2M context-training A/B

**Recommendation: CONTINUE_32_BYTE_RUN — HUMAN REVIEW ONLY.**

Production parent remains immutable step 5000. Both branches are DISPOSABLE_AB_ONLY; no branch is promoted.
Binding SHA-256: `32cc4bb6d45fe514c8c3e13abff913c4fff82a6c063578a0b53211af75cd2f79`. Schedule SHA-256: `3029d8c9a6dbc333ebc9ea50d457cb3b8b5d8128781cb486d2ee8be92d367cef`.
Exactly 512 non-overlapping 128-byte TRAIN blocks; 65,536 identical target bytes and 512 optimizer updates per arm. Quotas: general text 256, code 128, documentation 77, structured math 51. A resets at 0/32/64/96; B at 0/64. Both sum 128 losses and divide by 128; parent AdamW moments and cosine scheduler advance locally from 5000 to 5512.

## Parent / A / B comparison

| Metric | Parent | A:32 | B:64 |
|---|---:|---:|---:|
| Full 32 NLL (500,000 targets) | 2.3624953916 | 2.3516379566 | 2.3514031205 |
| Full 32 bits/byte | 3.40836039 | 3.39269642 | 3.39235762 |
| Full 64 NLL (500,000 targets) | 2.3624953904 | 2.3516379470 | 2.3514031097 |
| Full 64 bits/byte | 3.40836038 | 3.39269640 | 3.39235761 |
| Phase-matched 64+r history delta | 0 | 0 | 0 |
| Mechanism | INTERNAL_STATE_DIFFERS_OUTPUT_INSENSITIVE | INTERNAL_STATE_DIFFERS_OUTPUT_INSENSITIVE | INTERNAL_STATE_DIFFERS_OUTPUT_INSENSITIVE |
| Validation-context q(space) | 0.1763687 | 0.1821727 | 0.1837951 |
| Validation-context q entropy | 2.3834959 | 2.2989983 | 2.2950009 |

Full-validation NLL deltas: `{"32": {"a_minus_parent": -0.010857435068130528, "b_minus_parent": -0.011092271137237653, "b_minus_a": -0.00023483606910712496}, "64": {"a_minus_parent": -0.010857443410873469, "b_minus_parent": -0.011092280704498414, "b_minus_a": -0.00023483729362494543}}`.
B-A paired history-benefit effect: `{'mean': 0.0, 'median': 0.0, 'ci95': [0.0, 0.0], 'anchors': 128}`.
B-A absolute long-history target NLL: `{'mean': 0.003328422681761367, 'median': -0.0009448113851249218, 'ci95': [-0.008152062816634498, 0.01482514686790637], 'anchors': 128}`.
Mechanism labels describe sensitivity, not whether the sensitivity improves prediction.

| Gate | Result |
|---|---|
| beneficial_history | False |
| better_history_use_than_a | False |
| better_long_history_prediction_than_a | False |
| full32_nonregression | True |
| domain_nonregression | True |
| distribution_nonregression | True |
| practical_compute | True |

## arm-a-32

Training: 116.36s, 563.24 target bytes/s; physical replay updates: 0.
Learning efficiency: `{"nll_gain_per_1000_target_bytes": 0.00016567131146439404, "nll_gain_per_update": 2.1205927867442437e-05, "nll_gain_per_training_minute": 0.005598730078624442}`.
Memory: `{"peak_allocated_bytes": 148823040, "peak_reserved_bytes": 178257920, "minimum_sampled_free_bytes": 3256562484, "rss_status": "MEASURED", "peak_observed_rss_bytes": 932749312, "tensor_payloads": {"parameter_bytes": 7718316, "gradient_bytes": 7718316, "optimizer_state_bytes": 15436856, "canonical_recurrent_state_bytes": 2260992, "recurrent_note": "standalone canonical allocation; not persistent trainer cache", "activation_autograd_bytes": "UNMEASURED_SEPARATELY"}}`.

| Update | MA16 train NLL | MA64 | MA128 | Small32 / Small64 validation | Long-history64 delta [CI] | Mechanism |
|---|---:|---:|---:|---|---|---|
| 128 | 2.2741224 | 2.3151154 | 2.2978976 | 2.5508314 / 2.5508313 | 0 [0.0, 0.0] | forensics not scheduled |
| 256 | 2.3158720 | 2.2408556 | 2.2692568 | 2.5460115 / 2.5460113 | 0 [0.0, 0.0] | INTERNAL_STATE_DIFFERS_OUTPUT_INSENSITIVE |
| 512 | 2.1613836 | 2.2850825 | 2.2758827 | 2.5626503 / 2.5626504 | 0 [0.0, 0.0] | INTERNAL_STATE_DIFFERS_OUTPUT_INSENSITIVE |

### Final domain and position results

| Domain | 32 NLL | 64 NLL |
|---|---:|---:|
| structured_math | 3.086943268 | 3.086943267 |
| code | 2.412650938 | 2.412650937 |
| documentation | 2.312801584 | 2.312801583 |
| general_text | 2.185721297 | 2.185721296 |

64-byte position buckets: `{"0": {"sum_nll": 588432.4175345792, "count": 250131, "nll": 2.35249696173037, "bits_per_byte": 3.3939357003947586}, "32": {"sum_nll": 587386.555689865, "count": 249869, "nll": 2.3507780304474144, "bits_per_byte": 3.3914558067572096}}`.
Final phase results (full domain/phase confidence intervals and all curve points in JSON):

| Phase | Short mean NLL | 64+r delta | 128+r delta | 248+r delta |
|---|---:|---:|---:|---:|
| 0 | 3.54990987 | 0 | 0 | 0 |
| 1 | 3.06357305 | 0 | 0 | 0 |
| 2 | 2.70006795 | 0 | 0 | 0 |
| 3 | 2.53495081 | 0 | 0 | 0 |
| 4 | 2.49830403 | 0 | 0 | 0 |
| 5 | 2.47046531 | 0 | 0 | 0 |
| 6 | 2.46007909 | 0 | 0 | 0 |
| 7 | 2.50535975 | 0 | 0 | 0 |

Forensics: 256/256 identical logit pairs; max logit delta 0; max q TV 0; decoder-hidden max delta 0. Layerwise states/norms are in JSON.

| Prompt | Greedy output | Longest space run | Mean q(space) | Entropy |
|---|---|---:|---:|---:|
| `b''` | `b'                                                                '` | 64 | 0.594864 | 2.015862 |
| `b'The '` | `b'come                                                            '` | 60 | 0.549987 | 2.049167 |
| `b'def '` | `b'the                                                             '` | 61 | 0.560939 | 2.012989 |
| `b'import '` | `b'o                                                               '` | 63 | 0.583773 | 2.033496 |
| `b'class '` | `b'of                                                              '` | 62 | 0.572265 | 2.043501 |
| `b'x = '` | `b'self                                                            '` | 60 | 0.550776 | 2.042284 |

## arm-b-64

Training: 152.20s, 430.60 target bytes/s; physical replay updates: 0.
Learning efficiency: `{"nll_gain_per_1000_target_bytes": 0.00016925462550716633, "nll_gain_per_update": 2.166459206491729e-05, "nll_gain_per_training_minute": 0.004372901577351473}`.
Memory: `{"peak_allocated_bytes": 149601280, "peak_reserved_bytes": 180355072, "minimum_sampled_free_bytes": 3254465332, "rss_status": "MEASURED", "peak_observed_rss_bytes": 457146368, "tensor_payloads": {"parameter_bytes": 7718316, "gradient_bytes": 7718316, "optimizer_state_bytes": 15436856, "canonical_recurrent_state_bytes": 1130496, "recurrent_note": "standalone canonical allocation; not persistent trainer cache", "activation_autograd_bytes": "UNMEASURED_SEPARATELY"}}`.

| Update | MA16 train NLL | MA64 | MA128 | Small32 / Small64 validation | Long-history64 delta [CI] | Mechanism |
|---|---:|---:|---:|---|---|---|
| 128 | 2.2741224 | 2.3151153 | 2.2978976 | 2.5508313 / 2.5508313 | 0 [0.0, 0.0] | forensics not scheduled |
| 256 | 2.3158719 | 2.2408555 | 2.2692567 | 2.5460115 / 2.5460115 | 0 [0.0, 0.0] | INTERNAL_STATE_DIFFERS_OUTPUT_INSENSITIVE |
| 512 | 2.1560124 | 2.2832768 | 2.2733773 | 2.5609003 / 2.5609004 | 0 [0.0, 0.0] | INTERNAL_STATE_DIFFERS_OUTPUT_INSENSITIVE |

### Final domain and position results

| Domain | 32 NLL | 64 NLL |
|---|---:|---:|
| structured_math | 3.090548956 | 3.090548957 |
| code | 2.414779067 | 2.414779068 |
| documentation | 2.311584354 | 2.311584353 |
| general_text | 2.183831594 | 2.183831594 |

64-byte position buckets: `{"0": {"sum_nll": 588380.6131505439, "count": 250131, "nll": 2.3522898527193505, "bits_per_byte": 3.393636905251637}, "32": {"sum_nll": 587320.9433368485, "count": 249869, "nll": 2.3505154434397566, "bits_per_byte": 3.3910769737834596}}`.
Final phase results (full domain/phase confidence intervals and all curve points in JSON):

| Phase | Short mean NLL | 64+r delta | 128+r delta | 248+r delta |
|---|---:|---:|---:|---:|
| 0 | 3.55599291 | 0 | 0 | 0 |
| 1 | 3.08019932 | 0 | 0 | 0 |
| 2 | 2.69396013 | 0 | 0 | 0 |
| 3 | 2.53216966 | 0 | 0 | 0 |
| 4 | 2.50001183 | 0 | 0 | 0 |
| 5 | 2.47152634 | 0 | 0 | 0 |
| 6 | 2.45718449 | 0 | 0 | 0 |
| 7 | 2.51829257 | 0 | 0 | 0 |

Forensics: 256/256 identical logit pairs; max logit delta 0; max q TV 0; decoder-hidden max delta 0. Layerwise states/norms are in JSON.

| Prompt | Greedy output | Longest space run | Mean q(space) | Entropy |
|---|---|---:|---:|---:|
| `b''` | `b'                                                                '` | 64 | 0.593187 | 2.022433 |
| `b'The '` | `b'come                                                            '` | 60 | 0.548557 | 2.053011 |
| `b'def '` | `b'the                                                             '` | 61 | 0.559910 | 2.015928 |
| `b'import '` | `b'o                                                               '` | 63 | 0.582205 | 2.039283 |
| `b'class '` | `b'of                                                              '` | 62 | 0.570742 | 2.050457 |
| `b'x = '` | `b'self                                                            '` | 60 | 0.549226 | 2.048962 |

## Distribution tradeoffs

{"positions": {"parent_severe_fraction": 0.035416666666666666, "a_severe_fraction": 0.04583333333333333, "b_severe_fraction": 0.04583333333333333, "b_minus_a_control_mass": -2.0461144064920386e-07}, "generation": {"parent_severe_fraction": 0.0026041666666666665, "a_severe_fraction": 0.0026041666666666665, "b_severe_fraction": 0.0026041666666666665, "b_minus_a_control_mass": -2.512715527601149e-07}}

## Integrity, validation and limitations

- One seed, one matched 65536-byte TRAIN sample; no population-level or SOTA claim.
- 128 total hash-selected anchors (32/domain); anchors within documents correlated; bootstrap resamples anchors after phase averaging.
- A then B sequential execution; thermal/cache/order differences confound timing. Evaluation excluded from measured training time.
- No effective-context certification, TEST evaluation, curriculum implementation or production promotion.
- CUDA reference uses quadratic transition storage; no efficient-kernel inference from short profiles.
- No independent 500k baseline rerun: exact identical clone weights/config/source and optimizer/RNG equality reuse accepted parent evidence.
- Generic process RSS sampled, not inferred from tensor arithmetic; activations not separately measured.
- Training seconds sum measured update intervals including finite/memory checks; data selection, journal/checkpoint I/O and evaluation are excluded. End-to-end study wall time was not separately instrumented. Replay timing is separate; unpublished in-flight work cannot be reconstructed from durable records.

| Check | Exit | Seconds |
|---|---:|---:|
| pytest | 0 | 26.51 |
| ruff | 0 | 0.23 |
| format | 0 | 0.15 |
| compileall | 0 | 0.15 |
| cpu_dependencies | 0 | 2.07 |
| cuda_dependencies | 0 | 1.76 |
| diff | 0 | 0.05 |

All ten production checkpoints, progress/cursor/optimizer/scheduler, accepted Stage-A/context reports, corpus, model source and FINAL files remain unchanged. TEST stayed sealed. No production continuation or promotion.

2M CONTEXT-TRAINING A/B STATUS: READY FOR HUMAN DECISION
