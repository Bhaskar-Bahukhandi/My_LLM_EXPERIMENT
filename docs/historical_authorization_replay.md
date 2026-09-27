# Current tests and historical authorization replay

The r2/r3 authorization contracts bind the pre-acquisition report from commit
`720af6e63ee8759654b287848173bbd3e74867c4`. The root report was deliberately
superseded in `f770e29c0a5c2e07e556f121aae0db0d83cbe7b0`. Current corpus evidence
and frozen authorization evidence must therefore be validated in their own contexts.

`tests/conftest.py` routes the three unchanged historical modules through
`tests/test_historical_authorization_replay.py`. A normal full pytest run includes
that required wrapper. The wrapper verifies the original snapshot manifest digest,
all 57 listed artifacts and four supplemental replay files, copies those verified
bytes to a temporary directory, and runs the unchanged three modules there. Their
`Path(__file__).parents[1]` roots then resolve to the historical inputs. The snapshot
is verified again after execution. Neither snapshot caches nor expected hashes are
modified. An absent or corrupt snapshot fails; there is no skip or current-root fallback.

The original `snapshot_sha256` is SHA-256 over the exact bytes of
`data/pilot-2m-r1/provenance/authorization_snapshot.json`, a manifest mapping
57 relative paths to file digests and recording the historical commit. It is not
a newly defined sorted-tree or canonical-JSON hash. Existing raw-file SHA-256
semantics are retained. Supplemental replay code was independently matched to the
historical Git commit and is pinned separately; it is not retroactively added to
the original manifest. Five pre-existing bytecode files are inventoried by the
provenance review but neither copied nor executed.

Independent historical group:

```powershell
.\.venv\Scripts\python.exe -B scripts/historical_authorization_replay.py
.\.venv\Scripts\python.exe -m pytest -q -s tests/test_historical_authorization_replay.py
```

Current checks, excluding the required historical group (both groups must pass for
a full validation):

```powershell
.\.venv\Scripts\python.exe -m pytest -q -m "not historical_replay"
```

That last command includes existing model/training regressions and real-corpus
integrity checks. It must **not** be used for a no-training/no-TEST-access tranche.
This provenance tranche instead uses the explicit non-training current file list
recorded in the review JSON. `test_full_training_stage_a.py` calls real corpus
verification, which includes sealed TEST integrity reads; trainer/overfit tests run
synthetic updates. Neither category was executed here. Synthetic byte fixtures in
corpus unit tests are not the sealed production TEST payload.

The unchanged historical test files remain in `tests/` for traceability, but must
not be interpreted as current-corpus readiness tests. New current provenance tests
bind the superseding report through final corpus evidence and require that the
historical scripts/tests remain byte-identical to their reviewed copies. If those
implementations need changes later, add current behavior tests and explicitly review
the compatibility rule; never edit the immutable historical snapshot to accommodate them.

The snapshot lives in ignored local provenance storage and is a required validation
input. A fresh checkout without it fails explicitly; do not download data or silently
rebuild a snapshot in the test harness. Git reproduction of the known LF/CRLF bytes
is documented in the provenance report for a separately reviewed recovery if needed.

Rollback: remove the added harness, current-context tests and collection routing as
one change. Original scripts/tests, contracts, reports and snapshot remain untouched.
The old current-root failures would return, as expected; rollback does not make the
superseded inputs compatible. No model/training mathematics changes are involved.
