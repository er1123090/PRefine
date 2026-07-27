# A-MEM upstream provenance

This Experiment8 implementation adapts the official A-MEM reproduction:

- Repository: <https://github.com/WujiangXu/A-mem>
- Pinned commit: `0c8039f28fdcc08189a23c07a3437d9d2482f9c2`
- License: MIT; see `LICENSE.A-MEM`
- Primary reference: `memory_layer.py`
- Evaluation reference: `test_advanced.py`

## Preserved algorithm

The adapter keeps the official execution order and mutation semantics:

1. Each dialogue turn is an immutable `MemoryNote.content`.
2. `analyze_content` generates `keywords`, `context`, and `tags`.
3. The raw new content retrieves the five nearest older notes.
4. `process_memory` may `strengthen` the new note and/or
   `update_neighbor` metadata.
5. Strengthened links point from the new note to older notes; no backlink is
   synthesized.
6. Retrieval returns semantic hits and expands their linked neighborhood.

## Experiment8 adapter boundaries

- OpenAI calls are asynchronous Chat Completions Batch requests.
- Stable string IDs replace process-local integer indexes.
- Chroma's local `all-MiniLM-L6-v2` function supplies embeddings.
- Memory is isolated by Experiment8 `example_id`.
- Updated neighbors are immediately re-embedded so persisted vectors match
  persisted metadata.
- Construction uses the official `k=5`; inference defaults to `k=10`.

The Experiment8 evaluator prompt and result contract are intentionally not
replaced by the upstream LoCoMo evaluation wrapper.
