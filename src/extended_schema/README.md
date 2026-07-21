# Extended Schema Fixed-400 Runs

This directory contains wrapper entrypoints for the fixed-400 single-turn and multi-turn experiments.
The wrappers keep the original script structure and reuse the source data and
memory files from the release root.

## Layout

- `vanillaLLM/`
  - fixed-pairs Python wrappers
  - rerun shell wrappers
- `ours_memory/`
  - fixed-pairs Python wrappers
  - rerun shell wrappers
- `langmem/`
  - fixed-pairs Python wrappers
  - rerun shell wrappers
- `mem0/`
  - fixed-pairs Python wrappers
  - rerun shell wrappers
- `RAG/`
  - fixed-pairs Python wrappers
  - rerun shell wrappers

## What Changed

- The wrapper Python entrypoints support `--fixed_pairs_path`.
- The fixed-400 shell wrappers derive their paths from the release root, so
  they can be launched from a cloned release without editing absolute local
  paths.
- Multi-turn fixed pairs rebuild `reference_ground_truth` from the original multiturn template base API call plus `slot_values_map`.

## Run Commands

Run these from the `experiments6` release root. The top-level public wrappers in
`scripts/` are the primary method interface, and the fixed-400 shell files here
are the supported entrypoints for reproducing the extended-schema batches. Both
surfaces resolve data, config, script, and default output paths from the release
directory. Lower-level copied legacy scripts that are not listed here are kept
for provenance and may still need explicit path arguments or method-specific
runtime state.

VanillaLLM:

```bash
bash extended_schema/vanillaLLM/run_vanillaLLM-api-single-extended-400.sh
bash extended_schema/vanillaLLM/run_vanillaLLM-api-multi-extended-400.sh
```

Ours memory:

```bash
bash extended_schema/ours_memory/Preference_Memory_step2_ACTION_singleturn_api5_extended_hard.sh
bash extended_schema/ours_memory/Preference_Memory_step2_ACTION_multiturn_api2_extended_hard.sh
```

LangMem:

```bash
bash extended_schema/langmem/run_langmem-single-extended-400.sh
bash extended_schema/langmem/run_langmem-multi-extended-400.sh
```

Mem0:

```bash
bash extended_schema/mem0/run_mem0-single-extended-400.sh
bash extended_schema/mem0/run_mem0-multi-extended-400.sh
```

RAG:

```bash
bash extended_schema/RAG/run_rag-single-extended-400.sh
bash extended_schema/RAG/run_rag-multi-extended-400.sh
```

Notes:

- `langmem` requires an existing snapshot such as `langmem/memory_snapshots/semantic-custom/<memory_model>/langmem_1229_dev_6.jsonl`.
- `mem0` requires `MEM0_API_KEY` and assumes the user memories for the dataset examples have already been ingested into Mem0.
- `RAG` defaults to `RAG/chroma_db_rag` with collection `user_memories` and requires `OPENAI_API_KEY` for retrieval embeddings.
- The new shell wrappers accept `BASE_URL` and `API_KEY` so the same runners can target either direct API calls or an OpenAI-compatible vLLM endpoint.

## Optional Environment Variables

- `TEST_TAG`
- `CONCURRENCY`
- `MAX_QUERIES`
- `BASE_URL`
- `API_KEY`
- `TARGET_MODELS`
- `TARGET_MEMORY_MODELS`
- `MEMORY_ROOT`
- `MEMORY_TOP_K`
- `EMBEDDING_MODEL`
- `STOP_AFTER_MEMORY_FOLDER`
- `STOP_RUN_AFTER_TARGET_MEMORY`

## Output Locations

- Vanilla: `extended_schema/vanillaLLM/output/`
- Ours memory: `extended_schema/ours_memory/output/`
- LangMem: `extended_schema/langmem/output/`
- Mem0: `extended_schema/mem0/output/`
- RAG: `extended_schema/RAG/output/`

Logs are written under the matching `logs/` subdirectories beneath those run roots.
