# Experiment configs

`vanilla_mix600.json` is the only config currently admitted by the canonical
end-to-end runner:

```bash
python -B scripts/run.py --config configs/experiments/vanilla_mix600.json
```

It fixes the dataset bundle, method, provider settings, all six conditions in
canonical order, evaluator variant, output root/name prefix, and Python
executable. The supported method is currently `vanilla_llm`; RAG, Mem0,
LangMem, and preference-memory adapters require injected backends and do not yet
have config-driven CLI runtimes.

## Admission rules

The config path must be a safe repository-relative path to a stable,
non-symlink regular UTF-8 JSON file. Admission rejects:

- absolute, escaping, symlinked, or historical-root paths;
- unknown or missing fields at every schema level;
- duplicate decoded keys at any depth, non-finite JSON numbers, or booleans in
  integer version fields;
- any condition set or ordering other than single-turn easy/medium/hard then
  multi-turn easy/medium/hard;
- unsupported method, provider type, or evaluator variant;
- invalid environment-variable names, provider URLs, numeric limits, output
  paths, or executable names;
- a manifest not owned by the configured prepared root.

The exact admitted bytes are SHA-256 bound and copied unchanged to
`<run-dir>/experiment_config.json`. `resolved_config.json` records that copy's
path and digest plus the selected evaluator variant. Before dry-run output or
artifact creation, the suite core strictly reparses those same bytes and rejects
any mismatch with the prepared root/manifest, provider/model/runtime settings,
evaluator, or exact child-runner arguments.

Config mode accepts only `--run-dir`, `--allow-external-output`, `--resume`, and
`--dry-run` in addition to `--config`. An explicit `--run-dir` is classified as
external output and is rejected unless `--allow-external-output` is also present;
the default remains the configured `output.root`. To change the model or endpoint,
create a separate repository-local config so the change is visible, admitted,
and preserved with the run.

At suite start, both tool-schema files are admitted once and copied byte-for-byte
into the run tree. All six child commands use those snapshots with SHA-256 checks.
The resolved config also records the requested Python executable and a fingerprint
of the resolved executable path, implementation, version, and environment prefixes.
