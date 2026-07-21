# Script entrypoints

Run active commands from the repository root.

## Canonical workflow

- `prepare.py` admits the configured mix600/query/preference/schema inputs and
  atomically writes six prepared JSONL files plus `manifest.json` and
  `validation_report.json`.
- `run.py --config ...` is the canonical experiment entrypoint. It admits one
  strict config, runs all six VanillaLLM conditions, preserves the exact config
  bytes, snapshots and SHA-binds both tool schemas, records the resolved Python
  runtime, and automatically evaluates terminal predictions.
- `evaluate.py` performs the same strict, provider-free evaluation as a
  standalone operation on an admitted run.
- `validate.py` is the API-free active-cutover phase/final validator.

```bash
python -B scripts/prepare.py
python -B scripts/run.py --config configs/experiments/vanilla_mix600.json
python -B scripts/evaluate.py \
  --run-dir artifacts/runs/<run-id> \
  --variant exp6_slot_value_or_v1
python -B scripts/validate.py --mode phase-readiness
python -B scripts/validate.py --mode final-completion
```

Phase-readiness currently exits 0 with `status=phase_ready` and
`final_complete=false`. Final-completion currently exits 2 with
`status=final_blocked` because the planning-only historical archive cleanup and the missing verified
cutover receipt are blockers. Only blocker-free final-completion with both `--cutover-trust-bundle` and its
exact `--cutover-trust-sha256` emits `status=final_complete` and exits 0. The
trust bundle must be outside the repository; there is no local fallback.

Phase-readiness has `completion_domain=project_structure`; final-completion has
`completion_domain=project_physical_cutover`. Both report
`experiment_completion=not_evaluated` and
`actual_experiment_completion_claimed=false`. They validate repository cutover
state and never claim that an actual provider experiment completed.

`run.py` also has a lower-level single-condition CLI, but it is not the default
six-condition experiment surface.

Config runs use their configured output root by default. An explicit external
directory must be acknowledged:

```bash
python -B scripts/run.py \
  --config configs/experiments/vanilla_mix600.json \
  --run-dir /approved/external/run \
  --allow-external-output
```

The config command exits nonzero when every provider call fails, while preserving
the terminal `all_provider_errors` status and any completed evaluation artifacts.

## Compatibility path

- `run_suite.py` is the older six-condition argument-driven entrypoint.
- `run_vanilla_llm_mix600_all_difficulties.sh` is a thin wrapper around it.

Those paths retain resume and manifest checks, but they do not preserve an
experiment config or auto-evaluate. Their metrics status remains pending;
standalone `evaluate.py` writes real metrics but does not rewrite that legacy
orchestration status. Prefer the config command for new runs.

## Separate validation and audit tools

- `validate_final_paper_outputs.py` validates the historical final paper-output
  evidence and 521 raw bindings. It is not the active phase validator.
- `archive_map.py --check` verifies the non-destructive archive plan. It never
  moves, deletes, or externalizes data.
- `archive_preflight.py` strictly validates supplied off-host immutable backup
  evidence and hashes a restored copy. Its exact arguments and schemas are in
  [the archive boundary](../archive/README.md#read-only-backup-preflight). It
  writes nothing and never authorizes externalization or changes the map.
  Success means only `evidence_bundle_structurally_valid`; provider-authenticated
  proof, independent approval, and physical cutover remain external blockers.
- `runtime/facade.py` and provenance/environment scripts retain the historical
  audited variant surface; they are not imported by the canonical runner.

The `paper_outputs/raw` externalization gate is closed until independent
off-host immutable backup versions and a verified restore transcript exist.
