# PREFINE Preference Shift and Conflict Rebuttal Notes

## Reviewer Question

Reviewers asked whether the MPT dataset includes cases where user preferences fundamentally reverse over time, for example low-price choices in early sessions followed by high-price choices in later sessions, and how PREFINE handles discarding previous assumptions and establishing new ones.

They also asked whether PREFINE can quickly adjust when preferences shift between sessions, and how it resolves conflicting preferences.

## Short Answer

The safest response is that MPT/Table 3 is not designed as a systematic abrupt preference-reversal benchmark. It mainly evaluates stable latent preference inference across sessions. In the Table 3 source data, I did not find explicit labeled cases where the same `group_preference` has both `low_cost` and `high_cost` as `api_calls_pref` evidence.

However, the raw session histories do include heterogeneous or conflict-like preference evidence. On those slices, PREFINE is consistently stronger than base prompting. This supports the more modest claim that PREFINE is more robust under noisy or heterogeneous preference evidence, while fully controlled abrupt-reversal evaluation is an important future extension.

## Source Files Audited

- `/data/minseo/experiments4/_paper/appendix_table3_combined_with_sources.csv`
- `/data/minseo/experiments4/_paper/table3_recomputed_from_appendix.csv`
- `/data/minseo/experiments4/data/1229_dev_6.json`
- `/data/minseo/experiments4/pref_group.json`
- Table 3 upstream prediction/evaluation sources listed in `appendix_table3_combined_with_sources.csv`, including:
  - `experiments4/evaluation/eval_results/results_vanillaLLM_1229-1_single.csv`
  - `experiments4/evaluation/eval_results/results_vanillaLLM_1231-1_single.csv`
  - `experiments4/evaluation/eval_results/results_vanillaLLM_1230-1_multi2_parse.csv`
  - `experiments4/evaluation/eval_results/results_memory_singleturn2.csv`
  - `experiments4/evaluation/eval_results/results_memory_multiturn3.csv`
  - `experiments4/evaluation/eval_results/results_memory_multiturn4.csv`
  - `experiments4/evaluation/eval_results/results_memory_singleturn_0306_gpt5.csv`
  - `experiments4/evaluation/eval_results/results_memory_multiturn_0306_gpt5.csv`

## Dataset Audit

Dataset: `/data/minseo/experiments4/data/1229_dev_6.json`

- User profiles: 265
- Sessions/dialogues: 2020
- Explicit `api_calls_pref` same-preference low/high reversal cases: 0
- Raw session API profiles containing both `low_cost` and `high_cost` budget evidence: 14 / 265
- Profiles with ordered low/high budget transitions: 5 / 265
- Strong reversal profiles, defined as at least 2 consecutive pure sessions for one budget pattern followed by at least 2 consecutive pure sessions for the opposite pattern: 0 / 265

Interpretation: MPT has limited heterogeneous/conflict-like evidence, but not systematic fundamental preference reversal scenarios of the type described by the reviewer.

## Table 3 Paired Deltas

PREFINE vs. base prompting, paired by inference model in Table 3.

### Context-Guided Results

| Metric | Avg. gain |
|---|---:|
| Recall OA-F1 | +11.92 |
| Induction OA-F1 | +10.57 |
| Transfer OA-F1 | +9.33 |
| Recall P-EM | +13.03 |
| Induction P-EM | +6.90 |
| Transfer P-EM | +2.95 |

### Context-Free Results

| Metric | Avg. gain |
|---|---:|
| Recall F1 | +11.87 |
| Induction F1 | +9.66 |
| Transfer F1 | +3.41 |
| Recall precision | +14.80 |
| Induction precision | +9.81 |
| Transfer precision | +4.83 |

These are not reversal-specific numbers, but they show the overall advantage of memory abstraction over base prompting in Table 3.

## Conflict-Like Slice Results

I recomputed metrics directly from the linked Table 3 upstream prediction JSON files.

### `budget_both` Slice

Definition: profiles where raw session API history contains both `low_cost` and `high_cost` budget evidence. This slice contains 14 source profiles and maps to 14 recall/easy, 20 induction/medium, and 33 transfer/hard query instances because the query construction creates different evaluation subsets.

PREFINE vs. base prompting averaged over linked multiturn sources:

| Setting | P-EM gain | Pref-slot F1 gain | OA-F1 gain |
|---|---:|---:|---:|
| Recall/easy | +17.31 | +16.62 | +13.25 |
| Induction/medium | +9.05 | +7.69 | +10.86 |
| Transfer/hard | +3.33 | +3.00 | +8.63 |

Model-level win counts on this slice:

| Setting | P-EM wins | Pref-slot F1 wins | OA-F1 wins |
|---|---:|---:|---:|
| Recall/easy | 6/8 | 6/8 | 7/8 |
| Induction/medium | 8/8 | 6/8 | 5/8 |
| Transfer/hard | 6/8 | 7/8 | 7/8 |

### `budget_majority_conflict` Slice

Definition: profiles where raw session API history contains both `low_cost` and `high_cost` budget evidence, but the counts are unequal. This excludes tied cases such as `low_cost=1, high_cost=1`. There are 8 such profiles, all `low_cost` majority: five profiles have `low_cost=2, high_cost=1`, and three profiles have `low_cost=3, high_cost=1`.

In Table 3, these 8 profiles map to 61 context-guided query instances: 8 recall/easy instances from 5 unique profiles, 20 induction/medium instances from 8 unique profiles, and 33 transfer/hard instances from 8 unique profiles.

PREFINE vs. base prompting averaged over linked multiturn sources:

| Setting | Query inst. | Unique profiles | P-EM base -> PREFINE | Pref-slot F1 base -> PREFINE | OA-F1 base -> PREFINE |
|---|---:|---:|---:|---:|---:|
| Recall/easy | 8 | 5 | 31.25 -> 49.79 (+18.54) | 52.48 -> 67.40 (+14.92) | 57.63 -> 70.02 (+12.39) |
| Induction/medium | 20 | 8 | 15.62 -> 25.17 (+9.54) | 41.51 -> 49.70 (+8.19) | 50.35 -> 61.37 (+11.02) |
| Transfer/hard | 33 | 8 | 4.55 -> 8.30 (+3.75) | 9.21 -> 13.01 (+3.80) | 42.93 -> 51.82 (+8.89) |
| Macro average | 61 | 8 | 17.14 -> 27.75 (+10.61) | 34.40 -> 43.37 (+8.97) | 50.30 -> 61.07 (+10.77) |

Model-level win counts on this slice:

| Setting | P-EM wins | Pref-slot F1 wins | OA-F1 wins |
|---|---:|---:|---:|
| Recall/easy | 6/8 | 7/8 | 6/8 |
| Induction/medium | 8/8 | 6/8 | 5/8 |
| Transfer/hard | 6/8 | 7/8 | 7/8 |

Interpretation: Even after removing tied low/high cases and keeping only majority-conflict profiles, PREFINE remains better on average in every setting and metric. This is useful rebuttal evidence, but it should still be framed as heterogeneous/conflicting budget evidence rather than a controlled abrupt reversal benchmark.

### `any_pref_slot_change` Slice

Definition: profiles where any preference slot has more than one observed value across the profile history.

PREFINE vs. base prompting averaged over linked multiturn sources:

| Setting | P-EM gain | Pref-slot F1 gain | OA-F1 gain |
|---|---:|---:|---:|
| Recall/easy | +12.83 | +12.95 | +12.22 |
| Induction/medium | +6.79 | +6.96 | +10.68 |
| Transfer/hard | +1.83 | +1.86 | +9.20 |

Model-level win counts on this slice:

| Setting | P-EM wins | Pref-slot F1 wins | OA-F1 wins |
|---|---:|---:|---:|
| Recall/easy | 7/8 | 7/8 | 8/8 |
| Induction/medium | 8/8 | 8/8 | 8/8 |
| Transfer/hard | 7/8 | 6/8 | 7/8 |

### `budget_transition` Slice

Definition: profiles where pure low-cost and high-cost budget evidence appear in different temporal positions. This is a small slice: 5 recall/easy profiles, 2 induction/medium profiles, and 4 transfer/hard profiles.

PREFINE improves overall OA-F1 on this small slice:

| Setting | P-EM gain | Pref-slot F1 gain | OA-F1 gain |
|---|---:|---:|---:|
| Recall/easy | +13.04 | +16.50 | +18.05 |
| Induction/medium | +12.50 | +13.04 | +22.06 |
| Transfer/hard | -4.93 | -1.29 | +6.18 |

Because this slice is very small, use it only as supporting evidence, not as the main claim.

## Mechanism Evidence

PREFINE's temporal behavior is implemented in the memory update prompt and verifier rather than as a separate explicit change-point detector.

Relevant source:

- `/data/minseo/experiments4/ours_memory/prompt_update3.py`
- `/data/minseo/experiments4/ours_memory/Preference_Memory_step1_LATENTPREF_VLLM.py`

Mechanism summary:

- Each session update conditions on the previous belief, full dialogue history, and full API-call history.
- The generator is instructed to prioritize the most recent stable pattern if contradictions exist.
- The verifier checks Temporal Consistency: if behavior changed, the inferred preference must reflect the latest stable pattern.
- Invalid, over-specific, or unsupported abstractions are rejected and refined.

Important caveat: PREFINE should not be described as immediately flipping after one contradictory observation. It is better described as conservative: it revises the latent preference when the newer evidence forms a stable pattern.

## Recommended Rebuttal Text

We thank the reviewer for raising this point. MPT is primarily designed to evaluate stable latent preference inference across sessions, rather than systematic abrupt reversals such as several low-price sessions followed by several high-price sessions. We audited the Table 3 source dataset and found no explicit `api_calls_pref` cases where the same preference dimension contains both low-cost and high-cost value groups as a labeled reversal. At the raw session level, however, 14/265 user profiles contain both low- and high-cost budget signals, and 5 profiles show an ordered low/high transition.

PREFINE handles such cases through its Temporal Consistency verifier: after each session, it regenerates the latent preference conditioned on the previous belief plus the full accumulated dialogue/API history, and the verifier requires the abstraction to reflect the latest stable behavioral pattern when behavior changes. Thus, PREFINE is conservative: it does not overwrite a preference after a single contradictory observation, but it can revise the memory once the newer pattern becomes stable.

Empirically, even on conflict-like profiles containing both low- and high-cost evidence, PREFINE improves over base prompting. On this slice, PREFINE improves P-EM/OA-F1 by +17.31/+13.25 in Recall, +9.05/+10.86 in Induction, and +3.33/+8.63 in Transfer. On the broader slice where any preference slot changes across the user history, PREFINE also improves OA-F1 by +12.22, +10.68, and +9.20 across Recall, Induction, and Transfer. These results suggest that PREFINE is more robust than base prompting under heterogeneous preference evidence, while we agree that explicit abrupt preference-reversal benchmarks are an important future extension.

## Suggested Paper/Appendix Caveat

MPT does not currently isolate abrupt preference-reversal scenarios as a controlled benchmark factor. We will clarify this limitation and add it as future work. The present evidence supports PREFINE's robustness to heterogeneous and partially conflicting historical evidence, not a claim of instantaneous adaptation to arbitrary preference reversals.
