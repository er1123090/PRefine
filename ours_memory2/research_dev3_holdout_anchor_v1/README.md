# Fresh holdout: PReFine-anchored typed overlay

This is a separate, one-shot validation of a single mechanism motivated by the
sealed 1229_dev6 outcome. It does not rerun, tune on, or read per-query
1229_dev6 answers, predictions, or gold rows.

The baseline is PReFine's retained latent abstraction plus accumulated API-call
history. The candidate starts from exactly that same block. It appends compact
typed facts only when repeated history evidence passes a public-query relevance
gate, confidence/conflict thresholds, and an explicit-current-query slot mask.
When no fact survives, the candidate block is byte-identical to baseline.

`fresh_prepare.py` is evaluator-owned and creates sanitized history/public
tasks plus a private gold vault from the 95 `dev_3` IDs absent from
`1229_dev_6`. `holdout_runner.py` cannot import or open the vault. After the
paired GPU run, `holdout_evaluator.py` opens gold once, writes aggregate
metrics, and checks an independently implemented metric audit.

No output from the gold vault, per-case predictions, or model responses should
be inspected during development or after the one-shot evaluation.
