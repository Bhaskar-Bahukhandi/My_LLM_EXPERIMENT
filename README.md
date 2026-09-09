# Unified Edge-400

Current scope: configuration resolution, structural parameter inventory and preliminary memory screening for the approximately 2M dense proof. There is **no executable language model or training pipeline yet**. The original FINAL Bible and Roadmap remain unchanged.

The closest inventory in the default 18-candidate search is **1,929,579 parameters**, 3.52105% below 2M. The nominal 2% gate is honestly rejected. Width/depth are 256/4; no dimensions were distorted to meet the target. This is a declared parameter budget, not the final executable model count.

## Use the existing local environment (PowerShell)

```powershell
.\.venv\Scripts\python.exe -m pytest -q
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
- Memory: declared FP32 payload and recurrent-state shapes only. Passing a lower-bound budget check remains UNKNOWN_REQUIRES_MEASUREMENT. Runtime RSS, activations and allocator costs are not measured.
- The candidate search sorts explicit width/depth choices, rejects legality/policy/memory violations, then ranks absolute parameter distance; ties prefer fewer layers then narrower width. Explicit dimensions are never silently replaced.

See [tranche report](reports/configuration_tranche.md), [inventory contract](docs/parameter_inventory.md), [reference pin](docs/backend_reference.md), [correction proposals](docs/architecture_changes), [implementation plan](docs/implementation_plan.md), and [project state](docs/project_state.md).

Next: implement the byte-model tranche against the documented structural contract, then reconcile actual tensors and parameter counts before any training readiness claim. MoE remains deferred pending its causal scheduling decision.
