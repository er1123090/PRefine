# Canonical provenance contract

The active mix600 chain is source bytes → prepared manifest/records → admitted
experiment config → predictions → metrics. Historical experiments4/5/6 code is
audit evidence, not an import dependency of canonical scripts or `src/exp7`.

## Prepared dataset evidence

`artifacts/prepared/mix600-v1/manifest.json` binds:

- byte size, path, and SHA-256 for mix600, both query files, preference inputs,
  and schemas;
- active dataset/query/preference/schema config hashes;
- active builder module hashes;
- path, row count, byte size, and SHA-256 for each of six JSONL outputs;
- validation-report digest and condition counts.

Preparation verifies every input before building and publishes the bundle with
one rename. Replacement is limited to a real directory whose regular manifest
already identifies `mix600-v1`.

## Run evidence

Config admission stable-reads and hashes the exact JSON bytes. A config-driven
run stores those bytes unchanged as `experiment_config.json` and records its
SHA-256, path, and evaluator variant in `resolved_config.json`.

Before inference, the suite re-verifies all prepared members and copies the
admitted manifest bytes to `dataset_manifest.json`. `resolved_config.json` also
binds the prepared root, manifest digest, model, endpoint, reasoning effort,
API-key environment name, Python executable, runner arguments, run path, and
six condition labels. Credential values are never persisted.

`status.json` records atomic provider and condition transitions. Predictions
retain canonical identity, query, GT, source example, provider response, token
counts, latency, and row status. Provider errors remain explicit rows.

Automatic or standalone evaluation re-admits the manifest-bound prepared and
prediction files. `metrics/metrics.json` binds evaluator code/config,
per-condition file digests, prepared/prediction counts, and resolved-config
digest. `metrics/status.json` binds the evaluator variant, metrics digest, and
condition/row/parse/error counts, or records a truthful evaluation failure.

## Verification

Inspect the prepared report:

```bash
python -B -m json.tool \
  artifacts/prepared/mix600-v1/validation_report.json
```

Run the API-free active phase validator:

```bash
python -B scripts/validate.py
```

The frozen historical parity and final paper evidence are separate. Use the
[legacy query/GT contract](mix600-legacy-contract.md),
[legacy validation audit](mix600-legacy-validation.md), and
[historical runtime navigation](runnable-environment.md) when reviewing those
surfaces.
