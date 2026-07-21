# Iteration Cap Comparison

## Goal

Show that allowing up to 10 refinement iterations in `0312_MEMORY1` does not provide a clear or consistent advantage over the 3-iteration setting in `1231_MEMORY3`.

Compared settings:

- `0312_MEMORY1_max10`: `/data/minseo/experiments6/ours_memory/inference/0312_MEMORY1`
- `1231_MEMORY3_max3`: `/data/minseo/experiments6/ours_memory/inference/1231_MEMORY3`

Comparison script:

- `/data/minseo/experiments6/ours_memory/compare_iteration_caps.py`
- `/data/minseo/experiments6/ours_memory/plot_refinement_tail.py`

Main figure:

- `/data/minseo/experiments6/ours_memory/iteration_cap_comparison/0312_MEMORY1_max10_refinement_tail.png`

## 1. Refinement Efficiency

- `0312_MEMORY1_max10`
  - Average attempts per example-session pair: `1.0936`
  - Verifier-valid pairs: `99.9802%`
  - Cases that only became valid after step 3: `69 / 10100 = 0.6832%`
  - Extra attempts beyond step 3: `165`

- `1231_MEMORY3_max3`
  - Average attempts per example-session pair: `1.0609`
  - Verifier-valid pairs: `99.3267%`
  - Cases that only became valid after step 3: `0 / 10099 = 0.0000%`
  - Extra attempts beyond step 3: `0`

Interpretation:

- The 10-iteration setting spends more refinement budget.
- However, the number of cases that actually benefit from going beyond 3 steps is very small: only `0.68%` of all example-session pairs.
- This means the larger iteration budget mostly adds cost, while changing the final validity outcome for only a tiny fraction of cases.

## 2. Multiturn Downstream Performance

Common subset:

- `45` model combinations
- `5` memory models
- `3` action models

Overall:

- F1
  - `0312_MEMORY1_max10 = 0.542830`
  - `1231_MEMORY3_max3 = 0.531233`
  - Delta: `+0.011597`

- Row-wise F1 wins
  - `0312_MEMORY1_max10 = 28`
  - `1231_MEMORY3_max3 = 17`

- Mean `pref_EM`
  - `0312_MEMORY1_max10 = 0.162901`
  - `1231_MEMORY3_max3 = 0.180285`
  - Delta: `-0.017384`

By difficulty:

- `easy`
  - F1: `0312 = 0.608795`, `1231 = 0.615088`
  - `pref_EM`: `0312 = 0.309036`, `1231 = 0.335141`

- `medium`
  - F1: `0312 = 0.568463`, `1231 = 0.574537`
  - `pref_EM`: `0312 = 0.146758`, `1231 = 0.172241`

- `hard`
  - F1: `0312 = 0.484406`, `1231 = 0.450051`
  - `pref_EM`: `0312 = 0.032910`, `1231 = 0.033475`

Interpretation:

- `0312_MEMORY1_max10` has a small F1 gain overall.
- But that gain is not consistent across metrics.
- On the stricter multiturn metric `pref_EM`, `1231_MEMORY3_max3` is better overall.
- For `easy` and `medium`, `1231_MEMORY3_max3` is better on both F1 and `pref_EM`.
- The F1 gain of `0312_MEMORY1_max10` is mostly concentrated in `hard` subsets.

## 3. Singleturn Downstream Performance

Common subset:

- `48` model combinations
- `4` memory models
- `4` action models

Overall:

- F1
  - `0312_MEMORY1_max10 = 0.327529`
  - `1231_MEMORY3_max3 = 0.324942`
  - Delta: `+0.002587`

- Row-wise F1 wins
  - `0312_MEMORY1_max10 = 26`
  - `1231_MEMORY3_max3 = 22`

By difficulty:

- `easy`
  - `0312 = 0.533566`
  - `1231 = 0.529643`

- `medium`
  - `0312 = 0.401775`
  - `1231 = 0.400000`

- `hard`
  - `0312 = 0.111105`
  - `1231 = 0.108526`

Interpretation:

- The singleturn difference is extremely small.
- The observed gain is too small to claim that the 10-iteration memory is meaningfully superior.

## 4. Main Takeaway

The results do not support the claim that increasing the refinement cap from 3 to 10 yields a clear overall improvement.

More precise conclusion:

- The 10-iteration setting slightly improves some F1 numbers.
- But the improvement is small.
- It is not consistent across metrics.
- It does not improve the stricter multiturn exact-match metric.
- It requires extra refinement attempts, while only `0.68%` of example-session pairs actually benefit from going beyond step 3.

## 5. Recommended Claim for Writing

Recommended wording:

> Allowing up to 10 refinement iterations does not yield a clear or consistent downstream advantage over the 3-iteration setting. While the larger budget slightly improves F1 in some subsets, the gains are small, concentrated mostly in harder cases, and do not translate into better strict multiturn preference exact match. Moreover, only 0.68% of example-session pairs benefit from refinement beyond step 3, indicating that the extra budget adds cost with limited practical payoff.

Safer shorter version:

> Increasing the refinement cap from 3 to 10 is not clearly beneficial overall.

## 6. Output Files

- Refinement totals:
  - `/data/minseo/experiments6/ours_memory/iteration_cap_comparison/refinement_totals.csv`
- Refinement by model:
  - `/data/minseo/experiments6/ours_memory/iteration_cap_comparison/refinement_by_model.csv`
- Multiturn common subset:
  - `/data/minseo/experiments6/ours_memory/iteration_cap_comparison/multiturn_common_subset.csv`
- Multiturn by difficulty:
  - `/data/minseo/experiments6/ours_memory/iteration_cap_comparison/multiturn_by_pref.csv`
- Singleturn common subset:
  - `/data/minseo/experiments6/ours_memory/iteration_cap_comparison/singleturn_common_subset.csv`
- Singleturn by difficulty:
  - `/data/minseo/experiments6/ours_memory/iteration_cap_comparison/singleturn_by_pref.csv`
- Refinement tail figure:
  - `/data/minseo/experiments6/ours_memory/iteration_cap_comparison/0312_MEMORY1_max10_refinement_tail.png`
- First-valid-step table for the figure:
  - `/data/minseo/experiments6/ours_memory/iteration_cap_comparison/0312_MEMORY1_max10_first_valid_steps.csv`
