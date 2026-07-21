# Gemma 4 12B Vanilla LLM Evaluation

Primary metric: preference-slot value-OR micro F1. Deltas are multi − single.

| Mode | Difficulty | Cases | P | R | F1 | Preference EM | Full-call EM | Function acc. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| multi | easy | 554 | 0.5247 | 0.2367 | 0.3262 | 0.1877 | 0.1209 | 0.7960 |
| multi | hard | 472 | 0.2431 | 0.0932 | 0.1348 | 0.0847 | 0.0000 | 0.8665 |
| multi | medium | 293 | 0.5764 | 0.3993 | 0.4718 | 0.2901 | 0.0000 | 0.8908 |
| single | easy | 554 | 0.6605 | 0.4906 | 0.5630 | 0.3736 | 0.1625 | 0.7780 |
| single | hard | 472 | 0.1780 | 0.1992 | 0.1880 | 0.0975 | 0.0127 | 0.9280 |
| single | medium | 293 | 0.4652 | 0.6382 | 0.5381 | 0.3140 | 0.0853 | 0.9863 |

## Paired source-cluster bootstrap

| Stratum | Pairs | Sources | ΔF1 | 95% CI |
|---|---:|---:|---:|---:|
| pooled | 1319 | 359 | -0.1185 | [-0.1442, -0.0945] |
| easy | 554 | 335 | -0.2368 | [-0.2791, -0.1959] |
| medium | 293 | 123 | -0.0664 | [-0.1120, -0.0211] |
| hard | 472 | 123 | -0.0532 | [-0.0906, -0.0151] |
