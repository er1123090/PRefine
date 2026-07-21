# Runnable experiments7 environment

The practical entry point is `scripts/experiment.py`. It is independent from
the older strict-evidence facade under `scripts/runtime/`.

The initial `exp6_base_eval6` profile is a no-replace, read-only snapshot of
the experiments6 runtime subset. It contains 288 regular files (67,815,845
bytes) and 26 safe internal compatibility symlinks. The publication is sealed
to the fresh CP0 and source-pre identities from strict run
`exp7-strict-v6-20260718T091006Z-e9ff93d0c9fe0d6f15fb03c1214dd6e2`.
It excludes `artifacts/**`, generated outputs/logs/results/runs, caches, and
three symlinks whose targets are excluded artifacts.

## Commands

Run these from `/data/minseo/experiments7`:

```bash
python -B scripts/experiment.py list --json
python -B scripts/experiment.py list --profile exp6_base_eval6 --json
python -B scripts/experiment.py doctor --profile exp6_base_eval6 --json
python -B scripts/experiment.py dry-run --profile exp6_base_eval6 --recipe vanilla_llm_singleturn_api_eval6 --json
python -B scripts/experiment.py run --profile exp6_base_eval6 --recipe smoke_snapshot_eval6 --run-id smoke-001
```

Profile and recipe names carry the `eval6` label explicitly. Future exp4 and
exp5 overlays must keep their output-affecting behavior under distinct labels,
such as `eval4`, instead of silently replacing this base.

Every actual run creates a new `runs/<run-id>` directory. Existing run
directories are rejected. The runner records a credential-redacted invocation
manifest and result, puts temporary files under the run directory, executes
from the immutable snapshot, and rejects declared output paths outside the run
directory.

## Exact publication

The checked publisher validates fresh CP0/source-pre hashes, every selected
source descriptor, SHA-256, and size before publishing. It uses Linux
`renameat2(RENAME_NOREPLACE)` and fails when the versioned destination exists.

```bash
python -B scripts/publish_environment_snapshot.py --json
```

The publication manifest and seal live inside the snapshot at
`.experiment-env/`. The `doctor` command recomputes the payload, recipe, CP0,
source-pre, mode, symlink, and containment checks.

## Legacy strict facade

`scripts/runtime/facade.py` remains available for legacy registry, fixture,
and strict provenance workflows. It is not used as an implicit execution
backend by `scripts/experiment.py`.
