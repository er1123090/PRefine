# Canonical experiment contract

The config-driven suite consumes one verified `mix600-v1` prepared bundle and
executes exactly six conditions in this order:

| Order | Condition | Count | Schema |
| ---: | --- | ---: | --- |
| 1 | `singleturn.easy` | 554 | `schema_easy.json` |
| 2 | `singleturn.medium` | 293 | `schema_easy.json` |
| 3 | `singleturn.hard` | 472 | `schema_easy.json` |
| 4 | `multiturn.easy` | 554 | `schema_all.json` |
| 5 | `multiturn.medium` | 293 | `schema_all.json` |
| 6 | `multiturn.hard` | 472 | `schema_all.json` |

The total is 2,638 instances. Every prepared row carries canonical
`dataset_id`, `turn`, `difficulty`, `instance_id`, `source_example_id`, `query`,
and `ground_truth`. `ground_truth` is a non-empty list of service-call strings.
Adapters and evaluation must preserve those fields exactly; evaluation joins on
`instance_id` rather than deriving truth from dialogue or prediction text.

Difficulty-specific query and GT construction is defined in the
[dataset contract](dataset-contract.md). The retained 463 GT/schema warnings are
reported evidence and are never silently filtered.

## Config admission

The canonical invocation is:

```bash
python -B scripts/run.py --config configs/experiments/vanilla_mix600.json
```

The config must be a stable, non-symlink, repository-relative UTF-8 JSON file.
Admission hashes the exact bytes and rejects traversal, historical roots,
unknown or missing keys, non-canonical condition order, unsupported evaluator
variants, unsafe output paths, invalid provider values, and any method other
than `vanilla_llm`. Prepared root and manifest must have the configured
ownership relationship.

Config mode derives every output-affecting setting from the document. Its only
CLI controls are a fresh `--run-dir`, `--dry-run`, or `--resume` with an explicit
existing run directory. It does not accept a hidden model or endpoint override.

## Admission and provider execution

Before creating a real run directory, the suite verifies all six prepared JSONL
files against `manifest.json`: path, SHA-256, row count, unique non-empty IDs,
condition fields, and manifest membership must agree. Dry-run only prints the
six planned commands and creates no artifacts.

A new run directory must not exist. Per-condition prediction files are never
overwritten. Each prediction row preserves canonical identity/query/GT,
provider output, token counts, reasoning content, latency, and `ok` or `error`
status. Provider exceptions become evaluator-readable error rows.

Top-level `status.json` reflects provider execution only:

- `complete`: every row received an `ok` provider result;
- `complete_with_provider_errors`: both `ok` and `error` rows exist;
- `all_provider_errors`: every prediction row is an error row;
- `failed`: the suite could not produce and verify the required prediction set.

## Run artifact contract

Config-driven runs contain:

```text
<run-dir>/
├── experiment_config.json
├── resolved_config.json
├── dataset_manifest.json
├── run.log
├── status.json
├── predictions/<turn>/<difficulty>/
│   ├── checkpoint.jsonl
│   └── predictions.json
└── metrics/
    ├── metrics.json
    └── status.json
```

`experiment_config.json` is the exact admitted config byte sequence.
`resolved_config.json` binds `experiment_config: experiment_config.json`, its
SHA-256 as `experiment_config_sha256`, `evaluator_variant`, the admitted dataset
manifest hash, resolved paths, provider settings, and condition order.
`dataset_manifest.json` is an exact copy of the admitted prepared manifest.

Config-driven execution automatically evaluates after provider execution has a
terminal status. `metrics/status.json` transitions `pending` → `running` →
`complete`. Complete status binds the evaluator variant, `metrics.json` path and
SHA-256, condition and row counts, parse failures, and error rows. Evaluation
failure writes `failed` with its variant and error, then raises; the already
recorded provider terminal status remains truthful.

The legacy `run_suite.py`/shell-wrapper mode does not copy an experiment config
and does not auto-evaluate. Its orchestration-owned `metrics/status.json`
remains `pending`; standalone `evaluate.py` can write real `metrics.json` but
does not claim the config-mode status transition. This compatibility path must
not be confused with the canonical config command.

## Evaluation contract

The offline evaluator can be invoked independently:

```bash
python -B scripts/evaluate.py \
  --run-dir artifacts/runs/<run-id> \
  --variant exp6_slot_value_or_v1
```

It re-admits the run config, dataset manifest, prepared files, predictions, and
preference metadata; requires an exact per-condition `instance_id` join; and
writes real aggregate and per-condition slot/value metrics to
`metrics/metrics.json`. It does not call a provider.

## Resume contract

Resume requires a non-terminal real run directory, a byte-identical experiment
config and dataset manifest, equal resolved configuration, and pending
evaluation. Verified complete conditions are skipped. A partial checkpoint must
be a unique exact prefix of prepared IDs. Provider-terminal or evaluated runs
are non-resumable.

## Method boundary

The shared registry contains `vanilla_llm`, `rag`, `mem0`, `langmem`, and
`preference_memory`. Only VanillaLLM is wired to the canonical config-driven CLI.
The four stateful methods require injected storage/retrieval/refinement and
generation backends; their adapter contracts are implemented and tested, but
the CLI does not fabricate provider configuration for them.
