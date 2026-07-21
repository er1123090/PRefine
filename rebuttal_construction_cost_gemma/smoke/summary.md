# Gemma Local Memory Construction Cost

- LLM: `google/gemma-3-12b-it` via `http://127.0.0.1:8000/v1`
- Embedding: `text-embedding-3-small`

| method | dialogues | LLM input/dialogue | LLM output/dialogue | embedding input/dialogue | final memory content tokens | final memory serialized tokens |
|---|---:|---:|---:|---:|---:|---:|
| langmem | 1 | 2538.0 | 4469.0 | 148.0 | 76.0 | 185.0 |
| mem0 | 1 | 3191.0 | 497.0 | 69.0 | 69.0 | 1190.0 |
