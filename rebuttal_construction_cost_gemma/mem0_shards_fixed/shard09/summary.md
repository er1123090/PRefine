# Gemma Local Memory Construction Cost

- LLM: `google/gemma-3-12b-it` via `http://127.0.0.1:8000/v1`
- Embedding: `text-embedding-3-small`

| method | dialogues | LLM input/dialogue | LLM output/dialogue | embedding input/dialogue | final memory content tokens | final memory serialized tokens |
|---|---:|---:|---:|---:|---:|---:|
| mem0 | 17 | 25158.6 | 5650.9 | 415.0 | 381.5 | 4695.1 |
