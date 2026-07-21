# Prepared dataset contract

`scripts/prepare.py` is the inference-free entrypoint for `mix600-v1`. It reads
the external 20 MB mix600 source plus canonical query, preference, and schema
bytes under `configs/`, then materializes all six turn/difficulty conditions.

```bash
python scripts/prepare.py \
  --mix600 /data/minseo/experiments4/data/mix600.json
```

Until the audited `data/` layout is expanded atomically, the safe default is
`artifacts/prepared/mix600-v1`. A different non-existing directory can be
selected with `--output-dir`. Existing directories fail closed. Explicit
`--replace-existing` accepts only a real directory whose regular
`manifest.json` identifies the same dataset.

The output contains `singleturn/{easy,medium,hard}.jsonl`,
`multiturn/{easy,medium,hard}.jsonl`, `manifest.json`, and
`validation_report.json`. The manifest binds source, query, preference,
schema, builder, config, validation, and output hashes. The validation report
preserves the 463 known schema warnings, the ignored `eco` and `prefers_star`
groups, and the zero multi-turn base/preference merge-conflict result.

Canonical `instance_id` values include dataset, turn, difficulty, source ID,
and semantic content hash. `legacy_example_id_sub` remains available for
compatibility, but no semantic consumer should treat its positional suffix as
identity.
