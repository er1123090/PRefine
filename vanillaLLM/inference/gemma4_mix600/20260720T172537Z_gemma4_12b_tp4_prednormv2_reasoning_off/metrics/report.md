# Gemma 4 12B Vanilla LLM Evaluation

Primary metric: preference-slot value-OR micro F1. Deltas are multi − single.

| Mode | Difficulty | Cases | P | R | F1 | Preference EM | Full-call EM | Function acc. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| multi | easy | 554 | 0.7904 | 0.3105 | 0.4458 | 0.2671 | 0.1462 | 0.6191 |
| multi | hard | 472 | 0.0505 | 0.0106 | 0.0175 | 0.0085 | 0.0000 | 0.6504 |
| multi | medium | 293 | 0.5814 | 0.3413 | 0.4301 | 0.2355 | 0.0341 | 0.6416 |
| single | easy | 554 | 0.8627 | 0.4528 | 0.5939 | 0.4007 | 0.3430 | 0.8610 |
| single | hard | 472 | 0.1081 | 0.0169 | 0.0293 | 0.0169 | 0.0127 | 0.9979 |
| single | medium | 293 | 0.6131 | 0.4164 | 0.4959 | 0.3140 | 0.2594 | 0.9522 |

## Paired source-cluster bootstrap

| Stratum | Pairs | Sources | ΔF1 | 95% CI |
|---|---:|---:|---:|---:|
| pooled | 1319 | 359 | -0.0994 | [-0.1271, -0.0722] |
| easy | 554 | 335 | -0.1481 | [-0.1941, -0.1039] |
| medium | 293 | 123 | -0.0658 | [-0.1113, -0.0194] |
| hard | 472 | 123 | -0.0118 | [-0.0336, 0.0091] |
