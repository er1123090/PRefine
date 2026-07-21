# Gemma 4 12B Vanilla LLM Evaluation

Primary metric: preference-slot value-OR micro F1. Deltas are multi − single.

| Mode | Difficulty | Cases | P | R | F1 | Preference EM | Full-call EM | Function acc. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| multi | easy | 554 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| multi | hard | 472 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| multi | medium | 293 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| single | easy | 554 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| single | hard | 472 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| single | medium | 293 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Paired source-cluster bootstrap

| Stratum | Pairs | Sources | ΔF1 | 95% CI |
|---|---:|---:|---:|---:|
| pooled | 1319 | 359 | 0.0000 | [0.0000, 0.0000] |
| easy | 554 | 335 | 0.0000 | [0.0000, 0.0000] |
| medium | 293 | 123 | 0.0000 | [0.0000, 0.0000] |
| hard | 472 | 123 | 0.0000 | [0.0000, 0.0000] |
