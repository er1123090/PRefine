# Canonical artifacts

`prepared/` contains manifest-bound dataset bundles. `runs/` contains one new,
self-contained directory per canonical experiment.

A config-driven VanillaLLM run contains:

- exact `experiment_config.json` and `dataset_manifest.json` byte copies;
- `resolved_config.json` with config/manifest digests and resolved runtime
  values;
- atomic provider `status.json` and `run.log`;
- six prediction/checkpoint pairs below
  `predictions/<turn>/<difficulty>/`;
- real `metrics/metrics.json` plus digest/count-bearing
  `metrics/status.json` after automatic evaluation.

Provider and evaluator status are independent. A provider-terminal run can have
`complete`, `complete_with_provider_errors`, or `all_provider_errors`, while an
evaluation failure is recorded under `metrics/status.json` and makes the config
command fail.

The legacy argument-driven suite does not copy an experiment config and leaves
its orchestration status pending. `scripts/evaluate.py` can write real metrics
without claiming the config-mode status transition. Existing canonical run
paths are never overwritten.

The existing top-level `runs/`, `results/`, and `paper_outputs/` trees are audit
and historical-result surfaces, not destinations for the canonical runner.
