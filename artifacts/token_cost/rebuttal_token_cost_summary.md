# Rebuttal Token Cost Accounting

All token estimates use `cl100k_base`. Provider usage is used when present; otherwise the stored prompt text is counted offline.

Important caveat: Mem0 and LangMem construction logs do not expose their internal LLM/embedding token usage. Their construction values below are lower bounds from session text sent into the memory writer plus positive stored-memory growth. PREFINE construction is reconstructed from the paper code path: generator prompts are reconstructed, verifier prompts are logged.

## Construction Cost Per Session Update

| method | n updates | avg est total | avg lower-bound total | avg lower-bound input | avg lower-bound output | avg stored/injected memory after | avg compact pref only |
|---|---:|---:|---:|---:|---:|---:|---:|
| langmem | 10100 |  | 585.9 | 300.2 | 285.7 | 1298.6 |  |
| mem0 | 1434 |  | 366.4 | 301.6 | 64.8 | 289.6 |  |
| ours_memory | 12120 | 4118.4 | 4118.4 | 3962.8 | 155.6 | 122.8 | 23.7 |

## Inference Cost Per Test Query

| method | turn | context | n queries | avg provider input | avg estimated input | avg retrieved memory/context | avg memory payload |
|---|---|---|---:|---:|---:|---:|---:|
| langmem | multi | memory_api | 2221 | 4371.4 | 4346.4 | 225.7 | 225.7 |
| langmem | single | memory_api | 19763 | 2301.3 | 2288.7 | 215.6 | 215.6 |
| mem0 | multi | memory_only | 1097 | 3862.2 | 3862.2 | 133.1 | 133.1 |
| mem0 | single | memory_api | 1097 | 1931.8 | 1931.8 | 109.9 | 109.9 |
| mem0 | single | memory_diag | 1010 | 1931.2 | 1931.2 | 109.3 | 109.3 |
| mem0 | single | memory_only | 1097 | 1931.8 | 1931.8 | 109.9 | 109.9 |
| ours_memory | multi | memory_api | 21940 |  | 4313.2 | 521.7 | 119.7 |
| ours_memory | multi | memory_only | 21940 |  | 3915.1 | 123.7 | 119.7 |
| ours_memory | single | memory_api | 21940 |  | 2336.8 | 521.7 | 119.7 |
| ours_memory | single | memory_only | 21940 |  | 1938.7 | 123.7 | 119.7 |

## Files

- Construction detail: `/data/minseo/experiments6/rebuttal_token_cost_construction_detail.csv`
- Inference detail: `/data/minseo/experiments6/rebuttal_token_cost_inference_detail.csv`
- Summary CSV: `/data/minseo/experiments6/rebuttal_token_cost_summary.csv`
