# experiments6: Paper Release For Preference-Memory Experiments

This folder is a cleaned public release of the experiments originally developed in
`/data/minseo/experiments4`.

Scope is intentionally simple:

- Included: all non-`e-mem` experiment code, datasets, configs, baseline code,
  evaluation scripts, and small paper/analysis artifacts.
- Excluded: `e-mem`, generated inference outputs, logs, caches, bytecode,
  vector stores, virtualenvs, and nested git metadata.
- Preserved: legacy run paths through top-level symlinks such as
  `ours_memory`, `vanillaLLM`, `RAG`, `mem0`, `langmem`, and `evaluation`.

## What Is Implemented

The release contains these experiment families:

| Area | Release path | Purpose |
| --- | --- | --- |
| Preference-memory method | `src/preference_memory/` | Build/refine preference memories and run single-turn or multi-turn action inference. |
| Vanilla LLM/LRM baselines | `src/baselines/vanilla_llm/` | Run direct LLM/LRM inference without explicit memory. |
| RAG baseline | `src/baselines/rag/` | Build a Chroma retrieval index and run retrieval-augmented inference. |
| Mem0 baseline | `src/baselines/mem0/` | Add dialogue memories to Mem0 and evaluate retrieved memories. |
| LangMem baseline | `src/baselines/langmem/` | Build LangMem snapshots and run LangMem inference. |
| Extended schema tests | `src/extended_schema/` | Fixed 400-pair comparisons for the method and baselines. |
| Evaluation | `src/evaluation/` | F1, preference-group, reasoning, slot-count, and aggregate evaluation scripts. |
| Dataset tools | `src/dataset_tools/` | SGD-to-experiment conversion and validation utilities. |
| Self-refine baseline | `src/self_refine/` | Preference self-refinement through API models. |
| Analysis | `src/analysis/` | Token-cost, construction-cost, and rebuttal analysis scripts. |

The source-to-release mapping is recorded in
[`docs/source_manifest.md`](docs/source_manifest.md).

## Setup

From the cloned or extracted release directory:

```bash
cd experiments6
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The root `requirements.txt` is a release-oriented dependency list. The original
local freeze from `experiments4` is preserved at
`docs/original_requirements_from_experiments4.txt` for environment archaeology.

Most full experiment runs require one or more external services:

```bash
export OPENAI_API_KEY=...
export GOOGLE_API_KEY=...
export MEM0_API_KEY=...
```

For vLLM runs, install vLLM in a GPU environment and start from the copied shell
entrypoints or pass `--base_url` to a compatible OpenAI-style endpoint.

## Data And Configs

Core inputs are included:

- Dataset: `data/1229_dev_6.json`
- Single-turn queries: `configs/query_singleturn.json`
- Multi-turn queries: `configs/query_multiturn-domain.json`
- Preference lists/groups: `configs/pref_list.json`, `configs/pref_group.json`
- Tool schemas: `configs/schema_easy.json`, `configs/schema_all.json`
- Extended 400-pair files: `data/fixed_singleturn_example_query_pairs_400.json`,
  `data/fixed_multiturn_example_query_pairs_400.json`

The same config files are also symlinked at the release root so copied legacy
scripts that expect root-level files such as `schema_all.json` still resolve.

For public reruns, prefer the wrapper scripts under `scripts/` and the
documented fixed-400 extended-schema wrappers. They pass release-root-derived
paths into the original modules. Some lower-level legacy scripts under `src/`
are kept for provenance and may still require explicit path arguments or
method-specific runtime state.

## Representative Commands

The public wrappers resolve their defaults from the release directory. The
examples below still spell out the core data and config files so the experiment
inputs are easy to audit and override.

Build preference memories with the API path:

```bash
scripts/build_preference_memory_api.sh \
  --input data/1229_dev_6.json \
  --output runs/preference_memory/memory.jsonl \
  --verifier_output runs/preference_memory/verifier.jsonl \
  --refinement_output runs/preference_memory/refinement.jsonl \
  --provider openai \
  --model gpt-4o-mini
```

Run the preference-memory method on single-turn examples:

```bash
scripts/run_preference_memory_singleturn_api.sh \
  --input_path data/1229_dev_6.json \
  --query_path configs/query_singleturn.json \
  --pref_list_path configs/pref_list.json \
  --pref_group_path configs/pref_group.json \
  --tools_schema_path configs/schema_easy.json \
  --memory_path runs/preference_memory/memory.jsonl \
  --pref_type easy \
  --context_type memory_api \
  --output_path runs/preference_memory/singleturn_easy.json \
  --log_path runs/preference_memory/singleturn_easy.log
```

Run the vanilla LLM baseline:

```bash
scripts/run_vanilla_llm_singleturn_api.sh \
  --input_path data/1229_dev_6.json \
  --query_path configs/query_singleturn.json \
  --pref_list_path configs/pref_list.json \
  --pref_group_path configs/pref_group.json \
  --tools_schema_path configs/schema_easy.json \
  --pref_type easy \
  --output_path runs/vanilla_llm/singleturn_easy.json \
  --log_path runs/vanilla_llm/singleturn_easy.log
```

Build and run the RAG baseline:

```bash
scripts/build_rag_index.sh \
  --input_path data/1229_dev_6.json \
  --db_path runs/rag/chroma_db

scripts/run_rag_singleturn.sh \
  --input_path data/1229_dev_6.json \
  --query_path configs/query_singleturn.json \
  --pref_list_path configs/pref_list.json \
  --pref_group_path configs/pref_group.json \
  --tools_schema_path configs/schema_easy.json \
  --use_rag \
  --db_path runs/rag/chroma_db \
  --pref_type easy \
  --output_path runs/rag/singleturn_easy.json \
  --log_path runs/rag/singleturn_easy.log
```

Build and run LangMem:

```bash
scripts/build_langmem_memory.sh \
  --input_path data/1229_dev_6.json \
  --output_path runs/langmem/langmem_1229_dev_6.jsonl \
  --manifest_path runs/langmem/langmem_1229_dev_6.manifest.json

scripts/run_langmem_singleturn.sh \
  --input_path data/1229_dev_6.json \
  --query_path configs/query_singleturn.json \
  --pref_list_path configs/pref_list.json \
  --pref_group_path configs/pref_group.json \
  --tools_schema_path configs/schema_all.json \
  --memory_path runs/langmem/langmem_1229_dev_6.jsonl \
  --pref_type easy \
  --context_type memory_api \
  --output_path runs/langmem/singleturn_easy.json \
  --log_path runs/langmem/singleturn_easy.log
```

Run Mem0:

```bash
scripts/build_mem0_memory.sh --input_path data/1229_dev_6.json

scripts/run_mem0_singleturn.sh \
  --input_path data/1229_dev_6.json \
  --query_path configs/query_singleturn.json \
  --pref_list_path configs/pref_list.json \
  --pref_group_path configs/pref_group.json \
  --tools_schema_path configs/schema_easy.json \
  --pref_type easy \
  --context_type memory_api \
  --output_path runs/mem0/singleturn_easy.json \
  --log_path runs/mem0/singleturn_easy.log
```

Evaluate a single-turn result JSON:

```bash
scripts/evaluate_singleturn_f1.sh \
  --json_path runs/preference_memory/singleturn_easy.json
```

Aggregate multi-turn F1:

```bash
scripts/evaluate_multiturn_f1.sh \
  --pref_list_path configs/pref_list.json \
  --json_path runs/preference_memory/multiturn_easy.json \
  --out_csv runs/evaluation/multiturn_easy_f1.csv
```

## Legacy Compatibility

The clearer public layout lives under `src/`, `configs/`, `data/`, `scripts/`,
and `artifacts/`. For compatibility, these legacy names are symlinked at the
release root:

```text
ours_memory -> src/preference_memory
vanillaLLM -> src/baselines/vanilla_llm
RAG -> src/baselines/rag
mem0 -> src/baselines/mem0
langmem -> src/baselines/langmem
evaluation -> src/evaluation
extended_schema -> src/extended_schema
memory-dev -> src/dataset_tools
self-refine -> src/self_refine
```

This keeps old hard-coded family paths runnable after they were retargeted from
`experiments4` to `experiments6`.

## Verification Scope

Release verification checks syntax, shell entrypoints, stale path references,
source coverage, exclusion of `e-mem`, and unchanged `experiments4` /
`experiments5` status. It does not run full API, Mem0, RAG, LangMem, or GPU/vLLM
experiments because those require credentials, external services, and substantial
runtime.
