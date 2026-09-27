# Versioned resolver and configuration status

20M-I1 implements R1-R3 only. Schema stays `"1"`: the optional typed `resolver`
section is backward compatible, while its policy name versions the new semantics.
Older tools reject that unknown section rather than silently interpreting it. No
target-magnitude heuristic, automatic migration or checkpoint format change is introduced.

## Policy and selection

The complete pinned configuration is `configs/models/edge_20m_candidate.yaml`:

```yaml
resolver:
  policy: edge_dense_v2_20m
  mode: explicit
```

It pins width 512, 12 layers, state 64, byte 64, decoder 256, expand 2, head width 64,
conv 4, patch 8 and chunk 16. Target remains 20,000,000 with tolerance 0.05. The policy
and mode participate in authored and resolved serialization and both SHA identities.

| Policy | Search widths | Search depths |
|---|---|---|
| `edge_dense_v1_2m` | 128,192,256 | 2,3,4,6,8,12 |
| `edge_dense_v2_20m` | 256,320,384,448,512,576,640,704,768,896,1024 | 3,4,8,12,21,30,46 |

Versioned spaces are immutable definitions. YAML normally omits `search`; serialized
authored config includes the effective space for evidence. A supplied `search` must
exactly agree with its policy or fail. New scales require new supported policy names,
tests and review; unknown names never fall through to legacy behavior.

`explicit` requires both dimensions pinned and validates exactly that shape. It
never substitutes another candidate after legality, memory or tolerance failure.
OUTSIDE_TARGET publishes no selected resolved model under the new policy. Other
explicit shapes can be validated but do not inherit the reviewed pilot recommendation.
There is no padding, new model weight, MoE or mathematical model change.

`search` requires both dimensions `auto`; a pinned dimension is an error. The separate
`configs/models/edge_20m_search.yaml` returns SEARCH_COMPLETE, no selected model,
deterministic candidates and a diagnostic ranking of legal memory-admissible rows
by absolute parameter distance, depth, then width. Rows retain their envelope verdict.
Out-of-envelope controls are not preferred architectures. All audited width/depth
families are represented; explicitly setting decoder_dim to 128 in a separate search
config evaluates the retained allocation controls without changing the pinned YAML.

## Legacy compatibility

Absent `resolver` means exactly the old schema-1 behavior, including custom search
lists and automatic distance/depth/width selection. The new field is omitted from
legacy serialization, not serialized as null or added retroactively. A supplied
`resolver: null`, incomplete section or unsupported value fails. No target magnitude
chooses a policy implicitly. These rules preserve both old inputs and old hashes.

The unchanged 2M YAML resolves to the same shape, inventory, 1,929,579 parameters and
entire retained `reports/edge_2m_resolved.json` object. Resolved SHA remains
`7fe06b639741003e4c5ad9d57d32c2fd2b86f6805b2048ad1efd6aaa75e3a6f1`.
Its original 2% OUTSIDE_TARGET result still includes the historical nearest candidate.
This preserves legacy behavior instead of imposing the new explicit rejection
contract on frozen inputs. Historical resolved dictionaries round-trip unchanged.

The selected 20M YAML intentionally migrates to explicit policy metadata. Its shape,
parameter count, tensor ledger and equations are unchanged; its authored/resolved
digests change because policy identity is now bound. The I1 receipt records both.
Pre-build receipts and their original hash map remain immutable evidence at the
accepted closeout commit; they are not silently rebound to new implementation bytes.
Project state points to the new implementation receipt.

## Accounting and readiness

Selected configurations receive independent formula, declared inventory and real
DenseByteModel meta accounting. Unique Parameter totals, tensor-shape multisets,
named executable tensors and disjoint component totals must reconcile or raise an
explicit error. No forward, backward, optimizer, checkpoint load or parameter-value
allocation occurs. Search uses inventory-only accounting, not a meta-construction
or training claim for unselected rows.

Readiness lives outside immutable resolved-model serialization:

| Scope | Architecture | Accounting / construction | Training | Quality | Next |
|---|---|---|---|---|---|
| Exact frozen 2M config | IMPLEMENTED | META_VERIFIED / META_CONSTRUCTED | COMPLETED_FROZEN_LINEAGE | ENGINEERING_BASELINE_ONLY | Preserve frozen lineage; implement 20M R4-R9 |
| Reviewed pinned 20M | RECOMMENDED_FOR_PILOT | META_VERIFIED / META_CONSTRUCTED | NOT_RUN | UNVERIFIED | R4-R9, then approved execution gates |
| Other selected config | IMPLEMENTED_REFERENCE | META_VERIFIED / META_CONSTRUCTED | NOT_RUN | UNVERIFIED | Architecture and execution review |
| Search/rejection | NO_SELECTED_ARCHITECTURE | CANDIDATE_INVENTORY_ONLY / NOT_CONSTRUCTED | NOT_RUN | UNVERIFIED | Review candidates or fix config |

The 2M completion field cites accepted historical evidence and matches only its exact
configuration identity. It does not claim the edited workspace was trained, that a
similarly sized model completed training, or that current-source resume is authorized.
All reports keep `pilot_execution_ready=false` and `training_authorized=false`.
The obsolete single `training_ready` field is removed from current reports. Consumers
should use scoped readiness fields. Legacy CLI exit behavior is unchanged.

Memory preflight remains an analytic payload screen with unmeasured activations,
temporaries and process overhead. Its old arithmetic is unchanged for compatibility.
Metadata labels the planned context-vector allowance as conservative, not a measured
persistent hierarchy tensor. This screen cannot certify runtime fit or pilot readiness.

I1 edits configuration/accounting/status source, so aggregate current `src/` identity
necessarily changes. The frozen Gen-0 identity remains at its accepted Git commit;
mathematical model and trainer/checkpoint code stay byte-identical. Never bypass the
trainer source-identity check to resume a frozen checkpoint from this edited checkout.
Use the accepted frozen checkout for historical reproduction.

Rollback: revert the coherent I1 commit to restore prior tooling and the old 20M YAML.
Frozen 2M artifacts and reference fallback remain unchanged. R4-R9 implementation,
GPU measurements, training, dataset work and Kaggle are outside this tranche.
