# Professor–critic rubric: fresh anchored PReFine validation

The candidate is acceptable for one final fresh-holdout run only if all of the
following are true.

- The candidate preserves the source PReFine latent abstraction and accumulated
  API history; no-evidence cases fall back byte-for-byte to baseline memory.
- Additional evidence is repeated, typed, public-query-routed, confidence/
  conflict-gated, and masked when the current query explicitly sets a mapped
  target slot. No free-form latent reserialization or evaluator field appears
  in a candidate prompt.
- `dev_3` validation IDs are deterministically absent from `1229_dev_6`, and
  task/gold construction is evaluator-owned. The GPU runner has no gold-vault
  path or evaluator target field.
- Both paired arms use the same model snapshot, schemas, seed formula,
  temperature, max tokens, and exactly one action call per frozen public case.
- Before the gold file is read, prediction hashes and equal action budget are
  sealed. Evaluation opens gold once and the registered/independent metrics
  agree.
- The implementation seal is unchanged from GPU preflight through the critic.
  A performance failure is a valid research outcome, but it must stop further
  tuning of this candidate.
