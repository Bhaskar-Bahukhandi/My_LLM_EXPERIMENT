# Unified Edge-400

## Current status

- **2M Gen-0 is complete and frozen:** 1,929,579 parameters; final validation NLL
  approximately 2.17015. Generation remains weak and whitespace-heavy. This is an
  engineering baseline, not a production language model.
- **20M pre-build design is complete:** 20,387,531 parameters, width 512 x 12 layers,
  state 64, byte 64, decoder 256. **RECOMMENDED_FOR_PILOT**, not trained or production
  approved. **20M-I1 implements R1-R3; 20M-I2 implements R4-R8 infrastructure**.
  **20M-I3 implements R9 orchestration** with synthetic verification. Execution gates
  remain outstanding. No pilot is authorized; T4 hardware
  and measured 20M fit/throughput remain unverified.
- **Kaggle:** the separate bridge's read-only host execution is verified according
  to the accepted bridge record. Compute has not been approved or exercised.
- **TEST remains sealed.** MoE, RAG, tools, RSI and quantization are not implemented.

Start with the [current-state index](docs/project_state.md),
[frozen 2M report](reports/full_training_2m_final.md),
[20M design](reports/20m_prebuild_audit.md), and
[I1 implementation evidence](reports/20m_i1_resolver_config_implementation.md), and
[I2 infrastructure evidence](reports/20m_i2_training_infrastructure.md), and
[I3 orchestration evidence](reports/20m_i3_r9_orchestration.md).
The [training plan](docs/20m_training_plan.md) defines the remaining gates;
the [Kaggle plan](docs/kaggle_20m_execution_plan.md) preserves approval-bound host execution.

## Configuration inspection

Using the existing local CPU environment in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m unified_edge.cli configs/models/edge_20m_candidate.yaml
.\.venv\Scripts\python.exe -m unified_edge.cli configs/models/edge_20m_search.yaml
.\.venv\Scripts\python.exe -m unified_edge.cli configs/models/edge_2m.yaml
```

The 20M candidate explicitly pins its shape under `edge_dense_v2_20m`. It is validated
against the declared envelope and a real meta-device model. The separate search
config returns ranked candidates without choosing a model. Ranking by parameter
distance is diagnostic, not an architecture recommendation.

Schema 1 configs without `resolver` keep their original behavior and serialized
identity. The unchanged 2M YAML still reports OUTSIDE_TARGET at its original 2%
tolerance; the frozen 1.93M baseline was an explicitly accepted exception. See the
[resolver policy contract](docs/resolver_policy.md) for compatibility and status fields.

CLI exits: 0 for WITHIN_TARGET or SEARCH_COMPLETE, 2 for OUTSIDE_TARGET, 3 for no
legal/memory-admissible candidate, 1 for malformed input/output errors. No exit code
authorizes training. `--output evidence/my_review/audit.json` writes a new report
without overwriting an existing file. Readiness separates architecture, accounting,
construction, training and quality. R1-R9 are implemented; I3 is ready for review.
The pilot remains unauthorized, and T4 hardware remains unverified.
See the [I2 runtime policy](docs/training_infrastructure_policy.md)
for immutable TRAIN/VALIDATION snapshots, paranoid/fast monitoring, checkpoint schema
and planned device admission profiles.

Metadata-only pilot planning: `python scripts/edge_pilot.py plan`. The
[orchestration contract](docs/20m_pilot_orchestration_contract.md) describes strict
plans, receipt admission, durable event recovery, exact metrics and local export.
Validation never launches training. Hardware/mechanics gates, approved data, bounded
LR selection and explicit compute approval must precede real execution.

## Safe configuration checks

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q tests/test_config.py tests/test_accounting.py tests/test_resolver_policies.py tests/test_cli.py
.\.venv\Scripts\python.exe -B -m pytest -q -s tests/test_historical_authorization_replay.py
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m ruff format --check src tests scripts
```

These checks do not train models or read production TEST. Broad pytest includes
training and corpus integrity suites, so use the reviewed allowlist in the I1 receipt
for this scope. Historical authorization uses the [mandatory snapshot replay](docs/historical_authorization_replay.md).

The reference CPU/CUDA FP32 byte/Mamba model remains the semantic oracle. Its batch
SSD path is quadratic in patch time; meta accounting does not measure runtime fit.
See the [model contract](docs/mamba_integration_contract.md),
[training contract](docs/training_contract.md), and [historical GPU contract](docs/gpu_execution_contract.md).
The original Bible, Roadmap, accepted configs, historical receipts and checkpoints
remain preserved. Python 3.12 and the repository lockfiles define the existing
environments; no dependency installation or environment migration is part of I1.
