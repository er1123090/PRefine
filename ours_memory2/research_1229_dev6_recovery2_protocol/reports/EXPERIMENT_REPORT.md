# ours_memory2 — 1229_dev6 sealed recovery evaluation

## Result

The registered paired evaluation completed but did **not** meet the
pre-registered performance contract.  This is a negative result, not a
re-tuned result.

| Metric | PReFine baseline | ours_memory2 candidate | Candidate − baseline |
| --- | ---: | ---: | ---: |
| BMF1 | 0.425020 | 0.330178 | -0.094842 |
| Single-turn F1 | 0.376638 | 0.385509 | +0.008872 |
| Multi-turn F1 | 0.473401 | 0.274846 | -0.198555 |
| Preference F1 | 0.333650 | 0.264264 | -0.069386 |
| Non-preference F1 | 0.532853 | 0.320519 | -0.212334 |
| Parse-failure rate | 0.425706 | 0.558341 | +0.132634 |

The paired BMF1 bootstrap 95% CI was [-0.112328, -0.076865] (10,000
clustered draws; 245 clusters), and the one-sided paired-randomization
p-value was 1.0 (100,000 draws).  All performance gates therefore fail.

## Integrity and execution

- Exact paired coverage: 2,194 cases per arm, 100% in each arm.
- Equal action budget: one Qwen/Qwen3-8B action call per case, temperature
  0, TP=4; original runtime receipt bound CUDA_VISIBLE_DEVICES=0,1,2,3 and
  four distinct GPU UUIDs.
- Original provider generation completed before an unrelated post-provider
  NVML failure prevented its normal evaluator launch.  The recovery reused
  the immutable prediction bytes exactly; it made no provider calls and did
  not regenerate predictions.
- Final recovery preflight passed with zero target content reads.  The final
  evaluator opened the gold target once using O_NOFOLLOW, and the strict
  critic did not reopen it.
- Registered evaluator and independent metric implementation agreed exactly.
  The final recovery result is `execution=COMPLETE`, `integrity=PASS`, and
  `performance=FAIL`; the strict critic audit is PASS.

The post-provider GPU topology was unavailable because NVML reported an
unknown error after generation.  No post-hoc topology claim was fabricated.

## Interpretation and next valid experiment

The candidate's single-turn gain does not offset a substantial multi-turn and
non-preference regression, together with a 13.26 percentage-point increase in
parse failures.  Because this gold target has now been opened once, no further
prompt or mechanism tuning may use this 1229_dev6 evaluation.  Any subsequent
improvement attempt must be pre-registered and evaluated on a fresh unseen
split or separate held-out target.

## Evidence

- `RECOVERY2_AMENDMENT.json` SHA-256:
  `aea123ce7a23f24378535d3d09fdd88834e373e7888836b1575054718d1cfd23`
- `reports/summary.json` SHA-256:
  `9057871b191ec4a0a5dbc437e7b15e4fc1eb80762bc104c4c3adf62937e6f348`
- `reports/final_evaluation.evidence.json` SHA-256:
  `090dc94c021586ac784a72d7daea2fdc27cb0b4b8c662336ef646090c4680650`
- `artifacts/recovery2.critic.json` SHA-256:
  `26073a157d83e6330d0ecbe9a2a23b34d4f0be79acbd40c092d1e759709b8129`
