# Gemma Local Memory Construction Cost

- LLM: `google/gemma-3-12b-it` via `http://127.0.0.1:8000/v1`
- Embedding: `text-embedding-3-small`

| method | dialogues | LLM input/dialogue | LLM output/dialogue | embedding input/dialogue | final memory content tokens | final memory serialized tokens |
|---|---:|---:|---:|---:|---:|---:|
| langmem | 265 | 12971.9 | 29501.4 | 1189.8 | 619.5 | 1440.4 |
| mem0 | 265 | 24615.4 | 5416.3 | 387.2 | 351.9 | 4468.5 |
| ours_memory | 265 | 29184.5 | 1297.0 | 0.0 | 131.7 | N/A |

Note: `ours_memory` uses the existing `google_gemma-3-12b-it` PREFINE construction trace in `/data/minseo/experiments4/rebuttal_token_cost_construction_detail.csv`, grouped by dialogue. It does not use an embedding model during memory construction, and serialized-memory token counts are not applicable in that trace.
