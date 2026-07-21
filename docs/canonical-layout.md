# Canonical active layout

The active experiment surface is deliberately small:

- `configs/`: admitted dataset, query, preference, schema, model, and experiment
  inputs;
- `src/exp7/`: reusable dataset, method, experiment, evaluation, and provenance
  code;
- `scripts/prepare.py`, `scripts/run.py`, `scripts/evaluate.py`, and
  `scripts/validate.py`: canonical entrypoints;
- `tests/`: dataset, methods, evaluation, and integration contracts;
- `artifacts/`: prepared datasets and new canonical runs.

Historical `methods/`, `experiments/`, `environments/`, `runs/`, `results/`, and
`paper_outputs/` remain audit/provenance surfaces. Active modules must not import
their implementations.

## Prepared bundle

```text
artifacts/prepared/mix600-v1/
├── manifest.json
├── validation_report.json
├── singleturn/{easy,medium,hard}.jsonl
└── multiturn/{easy,medium,hard}.jsonl
```

## Config-driven run

```text
artifacts/runs/<run-id>/
├── experiment_config.json
├── resolved_config.json
├── dataset_manifest.json
├── run.log
├── status.json
├── predictions/<turn>/<difficulty>/
│   ├── predictions.json
│   └── checkpoint.jsonl
└── metrics/
    ├── metrics.json
    └── status.json
```

The experiment config and dataset manifest are exact admitted byte copies.
`resolved_config.json` binds their digests and all resolved runtime values.
`status.json` reports provider execution truth, while `metrics/status.json`
separately reports evaluation progress and its final metrics digest/counts.
Existing run paths are never overwritten.

The legacy argument-driven suite omits `experiment_config.json` and leaves
its orchestration metrics status pending. Standalone evaluation may write real
metrics but does not claim the config-mode status transition. New work should
use the config entrypoint documented in [the quickstart](quickstart.md).

## Archive boundary

`archive/archive-map.json` and `.md` are planning records only. No planned
disposition authorizes a move, deletion, or externalization. The raw-paper
externalization gate remains closed until independently recoverable off-host
evidence exists.
