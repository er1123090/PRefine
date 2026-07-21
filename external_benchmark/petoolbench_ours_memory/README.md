# PEToolBench ours_memory Adapter

This adapter tests whether `experiments5/methods/our_memory` can be applied to
PEToolBench without changing the original `ours_memory` runners.

The adapter keeps PEToolBench's native output contract:

```json
{"tool_name": "...", "parameters": {...}}
```

## Smoke Test

```bash
python experiments5/external_benchmark/petoolbench_ours_memory/validate_smoke.py \
  --work-dir /tmp/petoolbench_ours_memory_smoke
```

The smoke path normalizes two examples from each PEToolBench history type, builds
mock `ours_memory`-compatible memory records for the `p` split, emits mock
predictions, and evaluates tool/parameter exact match.

## Manual Steps

```bash
python experiments5/external_benchmark/petoolbench_ours_memory/prepare_petoolbench.py \
  --input experiments5/external_benchmark/PEToolBench/dataset_test/user_entries_test_p.json \
  --history_type p \
  --limit 5 \
  --output /tmp/petoolbench_p_5.jsonl

python experiments5/external_benchmark/petoolbench_ours_memory/build_memory.py \
  --input /tmp/petoolbench_p_5.jsonl \
  --output /tmp/petoolbench_memory.jsonl \
  --mock

python experiments5/external_benchmark/petoolbench_ours_memory/infer.py \
  --input /tmp/petoolbench_p_5.jsonl \
  --memory /tmp/petoolbench_memory.jsonl \
  --output /tmp/petoolbench_predictions.json \
  --mock

python experiments5/external_benchmark/petoolbench_ours_memory/evaluate.py \
  --predictions /tmp/petoolbench_predictions.json \
  --output_summary /tmp/petoolbench_summary.json
```

## Live Inference

After memory records exist, omit `--mock` from `infer.py` and set
`OPENAI_API_KEY` or the provider credentials supported by `experiments5/src/llm_client.py`.
The live path uses `tools_schema=None` and includes PEToolBench candidate tools in
the prompt text because PEToolBench tool names are not valid OpenAI function
names.
