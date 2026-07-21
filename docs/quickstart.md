# mix600 VanillaLLM quickstart

Run every command from `/data/minseo/experiments7`.

## 1. Prepare the six conditions

```bash
python -B scripts/prepare.py
```

The default source is external:
`/data/minseo/experiments4/data/mix600.json`. Preparation admits that file and
the canonical query, preference, and schema inputs only when their byte sizes
and SHA-256 values match the configs. It then atomically creates
`artifacts/prepared/mix600-v1` and refuses to overwrite it.

Use `--mix600 PATH` only for a byte-identical source. A deliberate rebuild uses
`--replace-existing`; replacement is allowed only when the existing regular
manifest identifies `mix600-v1`.

```bash
python -B -m json.tool \
  artifacts/prepared/mix600-v1/validation_report.json
```

`contract_parity.status` must be `ok`. The fixed counts are:

| Condition | Instances |
| --- | ---: |
| singleturn.easy | 554 |
| singleturn.medium | 293 |
| singleturn.hard | 472 |
| multiturn.easy | 554 |
| multiturn.medium | 293 |
| multiturn.hard | 472 |
| total | 2,638 |

The report deliberately retains 463 GT/schema warnings. They are known
evaluation-boundary warnings, not permission to remove or rewrite GT. See the
[dataset contract](dataset-contract.md).

## 2. Inspect the plan without provider calls

```bash
python -B scripts/run.py \
  --config configs/experiments/vanilla_mix600.json \
  --dry-run \
  --run-dir /tmp/exp7-vanilla-plan --allow-external-output
```

Dry-run prints exactly six commands in canonical order and creates no run
directory. It does not perform the real prepared-bundle admission done at run
time. Config mode requires `--allow-external-output` whenever an explicit
`--run-dir` is supplied, including disposable `/tmp` plans and resume targets.

## 3. Run the canonical experiment

Install the optional `openai` package in the inference environment and set the
credential named by the config (`OPENAI_API_KEY` by default):

```bash
export OPENAI_API_KEY='...'
python -B scripts/run.py --config configs/experiments/vanilla_mix600.json
```

The command prints its generated `artifacts/runs/<run-id>` path. Model,
endpoint, reasoning effort, limits, evaluator variant, prepared paths, and
output policy come from the admitted config. Config mode accepts only
`--run-dir`, `--resume`, and `--dry-run` as runtime controls; edit or copy the
config instead of applying hidden model/provider overrides.

For an OpenAI-compatible local endpoint, set `provider.base_url` in a copied
repository-local config. When the endpoint needs no credential, the provider
uses the placeholder key `EMPTY`.

## 4. Understand completion and artifacts

Config-driven runs preserve the exact admitted config bytes and automatically
evaluate after all six prediction conditions reach a terminal provider status:

```text
artifacts/runs/<run-id>/
├── experiment_config.json
├── resolved_config.json
├── dataset_manifest.json
├── run.log
├── status.json
├── predictions/
│   ├── singleturn/{easy,medium,hard}/
│   │   ├── checkpoint.jsonl
│   │   └── predictions.json
│   └── multiturn/{easy,medium,hard}/
│       ├── checkpoint.jsonl
│       └── predictions.json
└── metrics/
    ├── metrics.json
    └── status.json
```

`experiment_config.json` is byte-for-byte identical to the admitted source
config. `resolved_config.json` binds its filename and SHA-256, the evaluator
variant, dataset manifest SHA-256, resolved paths, provider settings, and six
conditions. `dataset_manifest.json` is the exact admitted prepared manifest.

Top-level `status.json` reports provider execution truth:
`complete`, `complete_with_provider_errors`, `all_provider_errors`, or
`failed`. Provider error rows remain in predictions and count as evaluation
errors; they are not silently discarded.

`metrics/status.json` moves from `pending` to `running`, then to `complete` with
the evaluator variant, `metrics.json` SHA-256, and condition, row, parse-failure,
and row-error counts. Evaluation failure writes `failed` with the variant and
error and makes the run command fail. It does not replace the provider status.

## 5. Resume or evaluate explicitly

A non-terminal config-driven run can resume only with its original explicit
directory and the same admitted config and prepared manifest:

```bash
python -B scripts/run.py \
  --config configs/experiments/vanilla_mix600.json \
  --run-dir artifacts/runs/<run-id> --allow-external-output \
  --resume
```

Terminal runs are non-resumable. Completed conditions are verified and skipped;
partial checkpoints must be exact ordered prefixes of the prepared IDs.

The evaluator is also available as an offline standalone command:

```bash
python -B scripts/evaluate.py \
  --run-dir artifacts/runs/<run-id> \
  --variant exp6_slot_value_or_v1
```

It strictly joins all predictions to manifest-bound prepared truth by
`instance_id` and writes real slot/value metrics to `metrics/metrics.json`.
Unlike config-mode orchestration, the standalone command does not own the
`pending` → `running` → terminal metrics-status transition.

## 6. Validate the cutover

```bash
python -B scripts/validate.py --mode phase-readiness
python -B scripts/validate.py --mode final-completion
```

The API-free phase command currently exits 0 with `status=phase_ready` and
`final_complete=false`. The final command currently fails closed with exit 2
and `status=final_blocked` because the planning-only historical archive cleanup
and the missing verified cutover receipt are blockers. Only a blocker-free final-completion run supplied with an external
`--cutover-trust-bundle` and exact `--cutover-trust-sha256` emits
`status=final_complete`. The archive receipt contract is documented in
[the archive boundary](../archive/README.md#durable-final-completion-receipt). These modes are different from the final paper-output
evidence validator described in
[the documentation index](README.md#validation-boundaries).

`phase-readiness` reports `completion_domain=project_structure`, while
`final-completion` reports
`completion_domain=project_physical_cutover`. Both explicitly report
`experiment_completion=not_evaluated` and
`actual_experiment_completion_claimed=false`; neither mode requires or claims a
successful provider run.
