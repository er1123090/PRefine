# Historical method audit tree

This top-level `methods/` directory preserves origin-specific experiments4/5/6
method files and `methods_index.json` for audit and provenance. It is not the
active Python package, and canonical scripts must not import it.

Active shared adapters live under `src/exp7/methods`:

| Method ID | Canonical lifecycle | Runtime support |
| --- | --- | --- |
| `vanilla_llm` | stateless generation over canonical prepared rows | complete config-driven CLI |
| `rag` | build a bound index, retrieve identity-scoped context, generate | injected index backend and generator |
| `mem0` | build namespace-scoped memories, retrieve, generate | injected memory backend and generator; explicit namespace required |
| `langmem` | build a bound snapshot, embed/search, generate | injected store, embedder, and model |
| `preference_memory` | refine source memory session by session, then generate | injected refiner and generator |

All adapters use the common contract in `src/exp7/methods/contracts.py`. They
preserve canonical IDs, query, and GT; bind persistent state to the dataset and
prepared record digest; reject missing, tampered, or cross-source state; and do
not import historical modules. Built-in registry import does not instantiate an
optional SDK or contact a network.

The registry is lazy. Stateless/default-config adapters can be obtained with
`get()`. Configured construction uses `create()`; for example, Mem0 fails closed
without an explicit namespace. Only VanillaLLM is presently exposed through
`configs/experiments/vanilla_mix600.json`. The other adapters are reusable
Python surfaces, not advertised as end-to-end CLI experiments.

For origin-specific method labels and historical behavior, use
`methods_index.json` and [the audited variant overlay](../docs/audited-variant-overlay.md).
