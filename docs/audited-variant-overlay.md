# Audited experiments4/5 variant overlay

This environment preserves the immutable experiments6 base snapshot while treating it semantically as an experiments4 release/path alias. It does not introduce `*_eval6` or `*_infer6` variants.

The overlay is built from the audit document with SHA256 `782df2d84077f93b69aed4c79b334643f7ef5cb33470fe5f38fc412434f90b6b`. It contains exactly 108 origin-preserving files: 62 from experiments4 and 46 from experiments5. Five were already represented by the earlier snapshot plan and 103 are newly selected. The overlay contains code, shell wrappers, prompts, and derived Table 15/16/Figure 5 audit artifacts; raw paper inference outputs are handled separately.

The canonical v2 overlay is already published at `environments/exp45-audited-overlay-v2-782df2d84077f93b`. The commands below document the one-time procedure for a clean reconstruction only, where that canonical destination does not yet exist:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B scripts/build_audited_environment_registry.py
PYTHONDONTWRITEBYTECODE=1 python -B scripts/publish_audited_environment_overlay.py
```

Publication is no-replace. If the canonical destination already exists, as it does in this workspace, the publisher raises `FileExistsError` and fails closed; it is not a rebuild-in-place or refresh command.

Inspect and validate:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B scripts/experiment_variants.py doctor
PYTHONDONTWRITEBYTECODE=1 python -B scripts/experiment_variants.py list
PYTHONDONTWRITEBYTECODE=1 python -B scripts/experiment_variants.py dry-run --variant eval4_single_legacy
PYTHONDONTWRITEBYTECODE=1 python -B scripts/experiment_variants.py run --run-id environment-overlay-smoke-v2
```

Dry-run resolves a sealed source entrypoint but deliberately does not execute legacy experiment code. Those scripts may require external providers, GPUs, datasets, or hard-coded legacy output paths. The only directly executable recipe is the contained smoke run, whose writes are restricted to a fresh experiments7 run directory.
