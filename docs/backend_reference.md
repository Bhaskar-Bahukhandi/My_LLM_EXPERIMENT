# Mamba semantic-reference selection

This is an engineering pin, not a Bible-prescribed version. Bible §§9.2, 76, 106 and 163.5 require reference semantics and versioned backend legality; they do not select v2.2.6.

- Upstream: state-spaces/mamba, release tag **v2.2.6**.
- Immutable commit: **d7b1ceb3c367ec9022925e812f507bcf706937c6**.
- [Mamba-2 source](https://github.com/state-spaces/mamba/blob/d7b1ceb3c367ec9022925e812f507bcf706937c6/mamba_ssm/modules/mamba2.py).
- Source SHA-256: **605e4439ff0baec8d8acaf4a191d9f0570eea9900065a065909124c472b08707** (rechecked during this tranche).

Rationale: a fixed release and immutable source make the default single-process Mamba-2 parameter layout auditable. Its native single-step fallback clearly exposes convolution and SSM state shapes. We intentionally avoid depending on evolving main-branch code. This is not a claim that this is the newest or fastest release.

The current inventory supports the reference subset: full SSM width (d_ssm=d_inner=expand*d_model), ngroups=1, D per head, RMSNorm enabled after SiLU gating, bias-free input/output projections, biased depthwise convolution, specialized dt/A/D parameter shapes. Width must divide into whole heads. State is [B,heads,headdim,d_state]; convolution state is [B,d_inner+2*d_state,d_conv]. These constraints describe a declared generic PyTorch subset, not an optimized kernel guarantee.

Profile: **torch_reference_inventory_v1**, classification **VALID_GENERIC_PATH**. This tranche implements parameter/state structure only. No Mamba forward, recurrence, kernel, initialization or numerical-parity claim is made. CUDA/Triton profiles and reduced precision are rejected explicitly. Official mamba-ssm, causal-conv1d and Triton are not installed because executable Mamba is outside this tranche.

Local verification pins: Python 3.12.14 from the existing bundled runtime used to create project .venv; torch 2.6.0+cpu from the official CPU wheel index; PyYAML 6.0.2 for authored YAML; pytest 8.3.5 and Ruff 0.11.13 for validation. Versions are fixed to reproduce this tested environment, not asserted to be Bible mandates. Transitive versions are captured in requirements-lock.txt. This CPU structural profile has no CUDA requirement; the visible RTX 2050/driver 596.21 remains unvalidated for model execution.

## Byte-hierarchy continuation verification

Retrieval/check date (UTC): 2026-09-09T00:09:05.154254+00:00. Official repository identity: https://github.com/state-spaces/mamba. Local git ls-remote independently confirmed the exact v2.2.6 tag/commit relationship; command and response are preserved in reports/mamba_reference_verification.json. The pin remains a semantic reference, not an installed Mamba implementation dependency. No version upgrade or Mamba implementation occurs in this tranche.
