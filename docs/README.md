# experiments7 documentation

New experiment work starts with the canonical prepared-data and config-driven
runtime, not the historical variant facade.

## Active workflow

1. [Quickstart](quickstart.md) — prepare, run, resume, evaluate, and validate.
2. [Dataset contract](dataset-contract.md) — six counts, query/GT generation,
   identity, and known schema warnings.
3. [Experiment contract](experiment-contract.md) — config admission, provider
   status, artifacts, evaluation, and resume rules.
4. [Canonical layout](canonical-layout.md) — active code and output boundaries.
5. [Provenance](provenance.md) — source-to-prepared-to-run hash bindings.

The canonical command is:

```bash
python -B scripts/run.py --config configs/experiments/vanilla_mix600.json
```

It currently supports VanillaLLM end to end. RAG, Mem0, LangMem, and preference
memory are shared, provider-neutral adapters with injected backends; their CLI
experiment configs have not been enabled.

## Validation boundaries

`scripts/validate.py` is the API-free cutover validator. It checks the frozen
dataset contract, active-code and config boundaries, launcher, Markdown links,
layout, and targeted tests. The phase-readiness mode is the default:

```bash
python -B scripts/validate.py --mode phase-readiness
python -B scripts/validate.py --mode final-completion
```

The current phase command exits 0 with `status=phase_ready` and
`final_complete=false`. Final-completion fails closed with exit 2 and
`status=final_blocked` while the planning-only historical archive cleanup and the missing verified
cutover receipt remain blockers. Only a blocker-free final-completion run supplied with an external
`--cutover-trust-bundle` and exact `--cutover-trust-sha256` emits
`status=final_complete` and exits 0. There is no repository-local trust fallback;
see [the receipt contract](../archive/README.md#durable-final-completion-receipt).

The report makes the boundary machine-readable. Phase-readiness uses
`completion_domain=project_structure`; final-completion uses
`completion_domain=project_physical_cutover`. Both set
`experiment_completion=not_evaluated` and
`actual_experiment_completion_claimed=false`. These commands validate project
reorganization state, not the success of an actual provider experiment.

The paper-output validator is a separate final-evidence check. It does not prove
that the active experiment runtime works:

```bash
python -B scripts/validate_final_paper_outputs.py
```

`--fast` skips only the 521 raw content hashes; it retains metadata and
structural checks.

## Historical and audit navigation

The following documents are retained as audit navigation, not as the default
experiment entrypoint:

- [Runnable historical environments](runnable-environment.md)
- [Audited variant overlay](audited-variant-overlay.md)
- [Runtime closure](runtime-closure.md)
- [Legacy query/GT contract](mix600-legacy-contract.md)
- [Legacy validation audit](mix600-legacy-validation.md)
- [Strict-run decision record](decisions/0001-versioned-strict-run.md)

The historical facade remains available through `scripts/runtime/facade.py` for
explicit variant inspection. It is independent of the canonical config-driven
VanillaLLM workflow.

## Archive gate

[The archive map](../archive/archive-map.md) is non-destructive and
planning-only. It authorizes no move, deletion, or externalization.
`paper_outputs/raw` remains local until independent off-host immutable object
versions and a restore transcript verifying all 521 hashes are recorded. The
[read-only backup preflight](../archive/README.md#read-only-backup-preflight)
documents the exact command and evidence schemas; a passing preflight does not
mutate the map or authorize externalization. Provider-authenticated proof,
independent approval, and the physical cutover remain external blockers and are
not simulated by local attestations.
