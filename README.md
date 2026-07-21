# Implicit Preference-Aware API Calling

Research code for evaluating whether LLM-based agents can infer implicit user preferences from dialogue and past API calls, then produce correct API calls in new situations.

## Layout

```text
experiments5/
├── src/                     # Shared loaders, prompt builders, clients, metrics
├── methods/
│   ├── vanilla_llm/
│   ├── rag/
│   ├── our_memory/
│   ├── mem0/
│   ├── self_refine/
│   ├── langmem/
│   ├── emem/
│   └── remem/
├── evaluation/             # Canonical evaluators and aggregate scripts
├── analysis/               # Token/slot analysis + rebuttal plotting scripts
├── config/                 # Canonical config, schema, query, and extended assets
├── data/                   # Local datasets and legacy experiment inputs
├── extended_schema/        # Fixed-pairs / extended-schema experiment variants
├── outputs/
├── logs/
└── results/
```

`experiments5` is the canonical public home. Method runners use the split
`inference_api_*`, `inference_vllm_*`, and `step1_*` entrypoints directly; old
compatibility wrappers and generated run artifacts are intentionally excluded
from the public tree.

## Install

```bash
pip install -r requirements.txt
```

Environment variables commonly used here:

```bash
export OPENAI_API_KEY=...
export GOOGLE_API_KEY=...
export ANTHROPIC_API_KEY=...
export MEM0_API_KEY=...
```

## Core Runs

Vanilla LLM:

```bash
cd methods/vanilla_llm
bash run_api_singleturn.sh
```

RAG:

```bash
cd methods/rag
bash run_api_singleturn.sh
```

Our Memory:

```bash
cd methods/our_memory
bash run_api_singleturn.sh
```

Conflict-majority Our Memory:

```bash
python scripts/build_conflict_majority_tasks.py \
  --input_path data/e_dev_conflict_ordered_ratio.json \
  --out_dir outputs/conflict_majority/ordered

python scripts/run_conflict_majority_exp5_memory_inference.py \
  --tasks_path outputs/conflict_majority/ordered/singleturn/hard.json \
  --memory_path outputs/our_memory/memory.jsonl \
  --output_path outputs/conflict_majority/ordered/singleturn/hard_exp5_memory.json \
  --context_type memory_only

python scripts/run_conflict_majority_exp4_memory_inference.py \
  --tasks_path outputs/conflict_majority/ordered/multiturn/hard.json \
  --memory_path outputs/our_memory/memory.jsonl \
  --output_path outputs/conflict_majority/ordered/multiturn/hard_exp4_memory.json \
  --context_type memory_only
```

The builder writes strict-majority task bundles by default and skips tied
majorities, e.g. `dev_0028` in the conflict dev files. Use `--include_ties`
only when tied max-count value groups should all be evaluated. The inference
outputs keep `reference_ground_truth` and `llm_output`, so the existing
`evaluation/eval_singleturn.py` and `evaluation/eval_multiturn.py` scripts can
score them with `config/pref_list.json`.

mem0:

```bash
cd methods/mem0
bash run_api_singleturn.sh
```

Self-Refine:

```bash
cd methods/self_refine
bash run_api_singleturn.sh
```

## Additional Baselines

LangMem:

```bash
cd methods/langmem
bash run_langmem_build.sh
bash run_langmem_single.sh
```

EMem:

```bash
cd methods/emem
bash run_emem_build.sh
bash run_emem_single.sh
```

ReMem:

```bash
cd methods/remem
python script.py --help
```

Extended-schema fixed-pairs runners live under `extended_schema/` and call the
canonical `experiments5/methods/*/inference_*` scripts.

## Evaluation

Single file:

```bash
python evaluation/eval_singleturn.py \
  --input_path outputs/vanilla_llm/api/singleturn/.../result.json \
  --pref_list_path config/pref_list.json
```

Batch summary:

```bash
bash evaluation/run_eval.sh
```

Aggregate and reasoning scripts are available under `evaluation/`.

## Notes

- `config/` includes the extended query/schema assets used by the fixed-pairs runners.
- `config/schema_hard.json` is provided for runners that expect a dedicated hard schema file.
- `methods/remem/` and `data/remem/sgd2ours/` provide the ReMem baseline and its supporting SGD conversion assets.
- Runtime outputs, logs, indexes, vector DBs, and memory snapshots should be written under `outputs/`, `logs/`, or ignored method-local cache directories rather than committed under `methods/`.
- Some heavy baselines such as `langmem` and `emem` may still require external services, GPU resources, or additional model downloads even after dependency installation.
