# PEToolMemory

`petool_memory` is a PEToolBench-specific memory adapter. It does not modify
`experiments5/methods/our_memory`.

The memory schema is tuned for PEToolBench's personalized tool-selection task:

- parse `<Category>.<Provider>.<Operation>` tool names into provider namespaces
- store positive provider evidence from preferred histories
- store rating-0 negative provider evidence from ratings histories
- store later-provider preference shifts from chronological histories
- apply memory only after the current query semantics are satisfied

## Smoke Test

```bash
python experiments5/external_benchmark/petool_memory/validate_smoke.py \
  --work-dir /tmp/petool_memory_smoke
```

## Manual Run

```bash
python experiments5/external_benchmark/petoolbench_ours_memory/prepare_petoolbench.py \
  --input experiments5/external_benchmark/PEToolBench/dataset_test/user_entries_test_p.json \
  --history_type p \
  --limit 5 \
  --output /tmp/petoolbench_p_5.jsonl

python experiments5/external_benchmark/petool_memory/build_memory.py \
  --input /tmp/petoolbench_p_5.jsonl \
  --output /tmp/petool_memory_p_5.jsonl \
  --mode deterministic

python experiments5/external_benchmark/petool_memory/infer.py \
  --input /tmp/petoolbench_p_5.jsonl \
  --memory /tmp/petool_memory_p_5.jsonl \
  --output /tmp/petool_predictions_p_5.json \
  --mock

python experiments5/external_benchmark/petoolbench_ours_memory/evaluate.py \
  --predictions /tmp/petool_predictions_p_5.json \
  --output_summary /tmp/petool_summary_p_5.json
```

For live memory extraction or live inference, omit `--mock` in `infer.py` and
use `--mode llm_with_draft` in `build_memory.py` with credentials supported by
`experiments5/src/llm_client.py`.

