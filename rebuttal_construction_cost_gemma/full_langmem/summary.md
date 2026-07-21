# Gemma Local Memory Construction Cost

- LLM: `google/gemma-3-12b-it` via `http://127.0.0.1:8000/v1`
- Embedding: `text-embedding-3-small`

| method | dialogues | LLM input/dialogue | LLM output/dialogue | embedding input/dialogue | final memory content tokens | final memory serialized tokens |
|---|---:|---:|---:|---:|---:|---:|
| langmem | 265 | 12971.9 | 29501.4 | 1189.8 | 619.5 | 1440.4 |
