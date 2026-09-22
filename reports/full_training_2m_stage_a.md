# Full 2M Stage-A production evidence

**READY FOR CONTINUATION REVIEW.** The existing run completed 5,000 logical updates and stopped. Finalization reused completed validation, generation and exact destruction/recreation/restore evidence; it performed no optimizer updates. TEST remains sealed except integrity hashes.

## Identity and fixed schedule

Production `edge-2m-one-pass-v1`; original production code commit `d3ea23c0ecc60f641bad585333a2df04e386982d`. Recovery correction is already committed as `5879c142a295e8beac38e0839eec91eb9dbfd5ab`. Original and corrected runner SHA-256 identities:

- Original: `bb466cd50b72a805d119765c4593372643057efd3780c6cf00a9d38a47d6b1b4`.
- Corrected: `295b7f39661a9a04d9698e1fe2ad3e190257961697e5b9b4b954f64b128ab974`.
- Immutable binding: `3e73bbae47f291e3fa7e3e1faffb01332349f2a6601c9e2ac448e1e5c32da986`.
- Model source identity: `7d98de1b0504a1d511f3da95e17ec45dbcca1c5bd2d0dadbd251987261173eb4`.

The real 1,929,579-parameter dense model is unchanged (201,483 byte hierarchy + 1,728,096 shared Mamba; 56 trainable tensors). Fresh seed-17 initialization was used, not diagnostic weights. Full configuration and environment identities are included in the JSON report.

CUDA FP32; RTX 2050; PyTorch 2.6.0+cu118 / CUDA 11.8; driver 596.21; deterministic execution, TF32 off. Sequence 32, batch 2, accumulation 2, AdamW LR 0.003, betas 0.9/0.999, epsilon 1e-8, decay 0.01 with existing metadata exclusions, clip norm 1, five-update warmup then cosine decay to minimum ratio 0.1. The horizon remains **78,167** updates, not 5,000. The frozen corpus has 312,666 train windows; a complete four-window-update schedule crosses into epoch 1 by two windows, totaling 10,000,064 bytes. Warmup 5 is preserved from the accepted diagnostic.

## Logical and physical accounting

Logical: 5,000 updates, 10,000 microsteps, 20,000 windows, **639,613 raw-byte targets**. Cursor epoch 0 / offset 20,000; scheduler completed 5,000; all 56 AdamW parameter update counts are 5,000; accumulation position 0.

The original WinError 5 occurred while publishing progress after update 432. Durable checkpoint 252, authoritative progress 431, valid temporary progress/journal 432 were preserved. The locking process is UNKNOWN. Updates 253–432 were replayed exactly: 180 physical updates / 720 windows / 22,980 bytes; every semantic metric and cursor ordering matched. This added zero logical updates. Recovery checkpoint 432 and all original snapshots remain preserved. The separate 250-to-252 resume proof adds two updates / eight windows / 256 bytes. Total replay overhead: **182 updates / 728 windows / 23,236 bytes**; physical total: **5,182 updates / 20,728 windows / 662,849 bytes**. No recovery was repeated during finalization.

Publication correction: unique temporary files, flush/fsync, six bounded sharing-error attempts with 0.05/0.1/0.2/0.4/0.8-second backoff, preservation of the previous valid file, and publication every 100 updates plus checkpoint/evaluation/status boundaries. Per-update append-only journals remain unchanged. Model, optimizer and scheduler semantics were not modified.

## Full validation and training trajectory

Each validation evaluates the same complete, separate 500,000-byte split, aggregated by valid target count. Absolute changes below are current minus previous NLL; negative is improvement.

| Update | Raw-byte NLL | Perplexity | NLL change | Assessment |
|---:|---:|---:|---:|---|
| 0 | 5.6641971744 | 288.35638827 | — | Baseline |
| 250 | 2.7356225964 | 15.41934047 | -2.9285745780 | Improved |
| 1000 | 2.5240430368 | 12.47894765 | -0.2115795596 | Improved |
| 2500 | 2.4254003321 | 11.30675497 | -0.0986427047 | Improved |
| 5000 | 2.3624953916 | 10.61741303 | -0.0629049405 | Improved |

Initial-to-final NLL reduction: 3.3017017828 (58.2907%). Every scheduled interval improves; gains diminish, with no observed validation regression. No claim of statistical significance or TEST performance is made.

| Update | Trailing 100-update mean train NLL | LR used | Windows | Bytes | Preclip norm | Clipped |
|---:|---:|---:|---:|---:|---:|---|
| 250 | 2.7838597167 | 0.002999935077 | 1000 | 31977 | 1.07052433 | True |
| 1000 | 2.5030081259 | 0.002998922696 | 4000 | 127909 | 1.05217755 | True |
| 2500 | 2.3411758037 | 0.002993222766 | 10000 | 319789 | 0.83833128 | False |
| 5000 | 2.3284301907 | 0.002972894234 | 20000 | 639613 | 0.89372247 | False |

First/final 100-update means: 3.2958777392 / 2.3284301907. These are arithmetic moving means of update NLL, not a complete training-corpus evaluation. Minimum individual update NLL 1.5850710273; final individual update NLL 2.6959750652. The final validation NLL exceeds the trailing train mean by 0.0340652010; these cover different data and are not a matched generalization-gap estimate. JSON retains every 100-update moving mean and hashes the complete original journals.

Maximum preclip gradient norm 10.7526836395; 1049 of 5,000 updates clipped at 1. All recorded losses/gradient norms are finite; trainer parameter/gradient gates and runner optimizer-state checks remained active. Final model and optimizer tensors were independently checked finite.

## Fixed incremental generation trajectory

Policy unchanged: greedy argmax over raw-byte classes 0–255, 64 continuation bytes for each fixed hand-authored prefix. Below is a lossless compact representation: `fragment + SPACE*n`; the JSON contains exact hex and escaped bytes for all 30 samples. All probes advance the real recurrent path (eight completed patches, nine shared steps); empty/BOS is preserved.

| Update | Prefix | Continuation |
|---:|---|---|
| 0 | `b''` | `b'\xbf\xf0\xbf\xf0\xf0\xbb\x8e\r2\xff\x98\xc3\xc3\xc3\xc3\xc3E2\xae\xaaB\xe8\xe1\xae\xf0\xf0\xf0\x1c\xa3\xe0NW2\xff\x98\xc3B\x0cT\xdc\xd3\xf0\xe4\xf0B\x8a\xc6\xdc22\x1a2\xe9\xef\x13\r\xf1\xff\x98}B\xd1\xb9\r'` |
| 0 | `b'The '` | `b'B\xe8\x99\x0b\xdb\xff\xd3\xce\xdf__\xab2\x8d]\xaaB\xe8\xe1\xae\xf9\x95J\xc8]N\x1a\x0c\xf1\xe2s2U\tU\x89\xe7~\x13\xf4I\xb5T\xdc\xf1u\xe5\xaaB\xb9.\x10J\xe5]\x0c\x0c\xca\x85WJ\x95]\xdf'` |
| 0 | `b'def '` | `b'\x0c\x85\xefb2\xff\xd3\xce\x18\xa8\xc3\xc32F\xd3\x7f\xc3\xc3\xc3\xc3s\x95A\xef]\xe0\x13\xdcHH\xe3BB\xe8\xe1 \x0c\n  \xeffff2\x8d\x8d\x1dB\x0c\xfadN\x8d]\x8d\x0c\x85\xefbE\xf0\xe4\xf0'` |
| 0 | `b'import '` | `b'=z\xbc\xb7\xaaB\xe8\xe1\xaer\xef]\x13\x0c\x85\xef\xb8\xf1u\xe5\xaaB\xb9.4J\x95J\x00\xe0\xe0J\x13\xbd2\xf1\x9e\x9e\x9e\x9e\xf72r\xe92\xe9\xefT\xdc,,[\x9cffff\xc3\xc3\xc3\xc3\xc3\xc3\xc3'` |
| 0 | `b'class '` | `b'\xf6d\xb2\xff\xd3\xce\xe8\xc6T\xdc2$]\xd3\x0c\xd1\xf6\xdc\xdau\x95\xfe\xad\x1f\xdc\x0c\xab2\x1a_\x0c\x85\xefb2\x8d\x8d\x1dB\x0c\xfadn\x95\x1a6\x0c\xef\xb8-\xf1u\xe5\xaaB\xb9.\x10i2\xe9UB\x0c'` |
| 0 | `b'x = '` | `b'\x0c\xbe\x86\r\xa8\xff\xd3\xce\x18\xa8\xc3\xc3\xd3\xf0s2\xe9\xef\xf6 \xf0\xf0\xff\xce\x0c\x02\x13W\xa8us\xdbBN\x1ab\xf0\xf0\xf0\xf0\xbb\xbb\x8e\r\x862\xe92\xe9\xefT\xdcuu\xa8\xc3\xc3\xc3\xc3\xc3E\xe2**'` |
| 250 | `b''` | `b'' + SPACE*64` |
| 250 | `b'The '` | `b'and' + SPACE*61` |
| 250 | `b'def '` | `b'and' + SPACE*61` |
| 250 | `b'import '` | `b'a' + SPACE*63` |
| 250 | `b'class '` | `b'an' + SPACE*62` |
| 250 | `b'x = '` | `b'' + SPACE*64` |
| 1000 | `b''` | `b'' + SPACE*64` |
| 1000 | `b'The '` | `b'a fo' + SPACE*60` |
| 1000 | `b'def '` | `b'a fo' + SPACE*60` |
| 1000 | `b'import '` | `b't' + SPACE*63` |
| 1000 | `b'class '` | `b'an' + SPACE*62` |
| 1000 | `b'x = '` | `b'the' + SPACE*61` |
| 2500 | `b''` | `b'' + SPACE*64` |
| 2500 | `b'The '` | `b'self' + SPACE*60` |
| 2500 | `b'def '` | `b'and' + SPACE*61` |
| 2500 | `b'import '` | `b'i' + SPACE*63` |
| 2500 | `b'class '` | `b'an' + SPACE*62` |
| 2500 | `b'x = '` | `b'next' + SPACE*60` |
| 5000 | `b''` | `b'' + SPACE*64` |
| 5000 | `b'The '` | `b'made' + SPACE*60` |
| 5000 | `b'def '` | `b'the' + SPACE*61` |
| 5000 | `b'import '` | `b'i' + SPACE*63` |
| 5000 | `b'class '` | `b'an' + SPACE*62` |
| 5000 | `b'x = '` | `b'self' + SPACE*60` |

Conditioned fragments become recognizable, but no sustained structure emerges. Empty/BOS stays at 64 spaces after update 250; other prefixes also collapse into long space runs. This is clear qualitative greedy byte-mode degeneracy, not proof that the learned distribution has collapsed: full validation improves throughout. It is neither useful language capability nor grounds to hide the healthy loss trajectory. Sampling policy was not changed to improve appearances.

## Checkpoints and exact restoration

All ten saved checkpoints passed payload SHA-256, readability, configuration, corpus and production-binding checks. The required 2,000/2,500/3,000/4,000/5,000 boundaries exist. Full path/hash inventory is in JSON. The final checkpoint is `data/full-training-2m-v1/attempt-003/checkpoints/step_005000`.

- Final state.pt SHA-256: `e1104c74d6593af12a7c521394bea452a3a7a9d4f4912c4740443415904c5726`.
- Final parameter SHA-256: `4b8416333d4faf32b86d20eabd2addf6bb6fb39c52783a3427597657db71584d`.
- Final destruction/recreation/restore: **EXACT PASS**, including model, optimizer, scheduler, global/microsteps, examples/bytes, cursor, Python/CPU/CUDA RNG and semantic metrics; all six fixed generation probes reproduced exactly.
- The completed runner publishes completion only after those assertions. The restored run manifest records the same final checkpoint at step 5,000. Finalization verified that manifest and all 5,000 semantic checkpoint metrics against the original logical journals without another training update.

## Measured cost and memory

Summed original logical-update duration: 3681.072545 seconds; target throughput 173.757238 bytes/s. Full validation durations total 2105.398620 seconds. These exclude replay, generation, downtime and other overhead; they are not a single uninterrupted wall-clock measurement. At this average rate the remaining 73,167 updates project to about 14.96 training-only hours, excluding validation and interruptions; this is an estimate, not authorization.

| Quantity | Measured bytes |
|---|---:|
| maximum_allocated_bytes | 99,920,384 |
| maximum_peak_allocated_bytes | 108,640,768 |
| maximum_peak_reserved_bytes | 127,926,272 |
| minimum_sampled_free_device_bytes | 3,306,894,132 |
| maximum_sampled_process_rss_bytes | 1,010,270,208 |
| maximum_process_peak_working_set_bytes | 1,034,629,120 |
| parameter_bytes | 7,718,316 |
| gradient_bytes | 7,718,316 |
| optimizer_state_bytes | 15,436,856 |
| canonical_recurrent_state_bytes | 1,130,496 |

RSS is measured with Windows GetProcessMemoryInfo, not inferred from tensors. CUDA allocator peaks and sampled device free memory remained inside the 1 GiB allocation cap / 1 GiB free-headroom gates. Canonical recurrent-state bytes describe a standalone allocation, not a persistent trainer cache. Activations/autograd are **UNMEASURED_SEPARATELY**. These measurements apply only to this short-window profile; memory evidence does not establish longer-sequence safety.

## Validation, rollback and limits

- 22 focused CPU production/recovery/trainer/profile tests PASS; 2 selected CUDA placement/parameter/resume tests PASS (5 expensive unrelated gates deselected).
- Ruff, format, compile and both environment dependency checks PASS. Optional unused NumPy bridge warning remains; no environment change.
- Frozen manifest raw/canonical hashes and all train/validation/test integrity hashes PASS. Model source, resolved config, diagnostic artifacts, original interruption snapshots, recovered journal and every checkpoint PASS. No acquisition, deduplication, contamination rerun or TEST evaluation.
- Rollback remains available through preserved Git history, recovery commit, immutable original runner, diagnostic artifacts and original/recovery/final checkpoints. Generated corpus/checkpoints/journals remain local ignored artifacts.
- Files added: `scripts/finish_full_training_stage_a.py`, this report and `reports/full_training_2m_stage_a.json`; ledger updated in `docs/project_state.md`. No `src/`, production runner, model, corpus, training configuration or dependency edits.

**Context limitation:** production training uses 32-byte windows, exposing the shared recurrent trunk to approximately four completed 8-byte payload patches per ordinary full window. Stage A does not establish long-context retention. No 4K/32K/1M effective-context claim is supported. The existing reference sequence implementation remains quadratic; no longer-window run was attempted.

**Decision and next step:** Stage A is ready for human continuation review, with no unresolved mechanical blocker. Human review must choose completing the existing one-pass 32-byte schedule or a separately reviewed longer-window curriculum. Persistent greedy whitespace degeneracy and limited context exposure should inform that decision. Neither continuation is launched; no RSI, TEST evaluation, architecture change or later scale is authorized by this report.

FULL-2M STAGE-A STATUS: READY FOR CONTINUATION REVIEW
