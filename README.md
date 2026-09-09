# Unified Edge-400

Current scope: configuration resolution, exact parameter accounting, the generic CPU FP32 dense byte/Mamba-2 model, and bounded deterministic CPU training infrastructure. A fixed tiny synthetic overfit experiment exercises learning and restart mechanics; it is not evidence of language-model quality. The original FINAL Bible and Roadmap remain unchanged.

The closest inventory in the default 18-candidate search is **1,929,579 parameters**, 3.52105% below 2M. The nominal 2% gate is honestly rejected. Width/depth are 256/4; no dimensions were distorted to meet the target. The full instantiated model now matches this inventory exactly; the candidate remains an explicit smoke-stage exception to the unchanged tolerance.

## Use the existing local environment (PowerShell)

```powershell
.\.venv\Scripts\python.exe -m pytest -q -m "not overfit"
.\.venv\Scripts\python.exe -m pytest -q -s tests/test_training_overfit.py -m overfit
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m ruff format --check src tests
.\.venv\Scripts\python.exe -m unified_edge.cli configs/models/edge_2m.yaml
```

The CLI returns 0 for WITHIN_TARGET, 2 for OUTSIDE_TARGET, 3 when no legal/memory-admissible candidate exists, and 1 for malformed input or output errors. Exit 2 is expected for the default config. None of these statuses authorizes training. To retain an audit, use `--output evidence/my_review/audit.json`; an existing file is never overwritten.

## Recreate the supported environment

Use Python 3.12 (tested: 3.12.14) to create a project-local `.venv`, then its Python executable for:

```text
python -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-build-isolation --no-deps -e .
```

Replace `python` with the new environment's executable. No system environment modification is needed. The repository lock captures the tested Windows CPU dependency versions; a clean reinstall on other operating systems has not been validated. PyTorch emits an optional NumPy initialization warning because NumPy is not installed; this tranche uses neither NumPy conversion nor NumPy numerical routines. The warning is recorded, not suppressed.

## Contracts and evidence

- Authored configuration: `configs/models/edge_2m.yaml`; unknown fields, wrong types, unsupported versions, duplicate YAML keys and unsupported backends fail explicitly.
- Immutable resolved config: `unified_edge.resolve.ResolvedConfig`, with derived dimensions, control registry, backend/reference identity and SHA-256.
- Parameter inventory: real metadata-device Parameters, counted by identity. Auditing also works on ordinary PyTorch modules, distinguishes frozen parameters/buffers/aliases, and reports logical state-dict entries separately from serialized file size.
- Memory: the configuration preflight remains a lower-bound check. Bounded CPU training now records actual Windows process working set separately from parameter, gradient, AdamW and canonical recurrent tensor payloads. Activation/autograd storage is not separately measured; short-run evidence does not establish long-context capacity.
- The candidate search sorts explicit width/depth choices, rejects legality/policy/memory violations, then ranks absolute parameter distance; ties prefer fewer layers then narrower width. Explicit dimensions are never silently replaced.

See [tranche report](reports/configuration_tranche.md), [inventory contract](docs/parameter_inventory.md), [reference pin](docs/backend_reference.md), [correction proposals](docs/architecture_changes), [implementation plan](docs/implementation_plan.md), and [project state](docs/project_state.md).

The byte hierarchy has 201,483 executable parameters and the shared Mamba trunk has 1,728,096, totaling 1,929,579. The accepted 143 architecture tests are preserved. Training tests and a separate fixed overfit gate cover the CPU mechanics. MoE remains deferred pending its causal scheduling decision.

## Byte hierarchy

```python
import json
from pathlib import Path
import torch
from unified_edge.resolve import ResolvedConfig
from unified_edge.byte_hierarchy import ByteHierarchy

config = ResolvedConfig.from_dict(json.loads(Path("reports/edge_2m_resolved.json").read_text()))
hierarchy = ByteHierarchy(config)
logits = hierarchy(torch.tensor([[0, 255, 128, 1]], dtype=torch.long))  # [1,4,267]
```

This teacher-forced API predicts at most one patch using shifted local inputs and learned BOS conditioning. For streaming, `start`, `predict` and `consume` emit a completed event only after eight observed symbols. The hierarchy then waits for explicit external conditioning through `condition`; no global trunk is supplied yet. Raw binary conversion and padding helpers live in `unified_edge.symbols`.

Read the [byte contract](docs/byte_hierarchy_contract.md) for state/serialization and caller obligations, and the [byte readiness report](reports/byte_hierarchy_readiness.md) for reconciliation, causality evidence and rollback. The standalone byte hierarchy remains available; the full model supplies real shared conditioning.

## Integrated dense model

```python
from unified_edge.dense_model import DenseByteModel

model = DenseByteModel(config).eval()  # resolved config loaded above
with torch.inference_mode():
    state = model.start()  # learned BOS is processed through all four Mamba layers
    first_logits = model.predict(state)
    state = model.consume(torch.tensor([65]), state)
```

The full differentiable call `model(targets)` accepts equal-length [B,T] symbol rows. Its independent dense SSD reference path uses quadratic storage in patch length; use bounded sequences/chunks. Incremental `predict`/`consume` manages completed patches internally. Reset and tagged export/restore cover both local and shared state. Use inference mode for streaming inference; exported state is detached and requires identical weights.

See [Mamba readiness](reports/mamba_integration_readiness.md), [numerical evidence](reports/mamba_integration_evidence.json), and [state/math contract](docs/mamba_integration_contract.md). Upstream package-runtime parity remains unverified on this Windows CPU environment. Bounded training memory is now measured; no GPU code was added.


## Bounded CPU training

Read [training contract](docs/training_contract.md), [readiness report](reports/training_infrastructure_readiness.md), and [machine-readable evidence](reports/training_infrastructure_evidence.json). The two pytest commands above separate ordinary regressions from the fixed 60-update experiment; plain pytest runs both. The experiment creates a fresh ignored directory under `evidence/training_infrastructure/` containing its tiny fixtures, manifests, JSONL metrics and checkpoints. Existing runs are never overwritten. No data is downloaded.

For explicit local runs, use the same tested API:

```python
from unified_edge.training.config import TrainingConfig
from unified_edge.training.data import create_tiny_fixture
from unified_edge.training.trainer import Trainer

# `config` is the resolved model configuration loaded above.
root = Path("evidence/my_cpu_run")  # choose an unused run identity
manifest = create_tiny_fixture(root / "data")
training = TrainingConfig()  # 32-byte windows, 60 optimizer updates, CPU FP32
trainer = Trainer(config, training, manifest, root / "data", root / "initial")
trainer.train_until(30)
checkpoint = trainer.save()  # completed update boundary only

del trainer
trainer = Trainer(
    config, training, manifest, root / "data", root / "resumed", resume_from=checkpoint
)
trainer.train_until(60)
validation = trainer.validate()
```

Resume requires identical model/training semantics, manifest hash, source identity and supported environment. It restores optimizer state (including lazy per-parameter update counts), scheduler, RNG and data cursor. A failed update requires restoration from the last complete checkpoint. GPU enablement, large data and substantive model training remain separate work.
