# Performance Evaluator: exp8-gptoss-memory-throughput-retune

## Objective
Maximize sustained experiment8 gpt-oss-20b ours_memory throughput on GPUs 0-3 from the current checkpoint without output corruption

## Evaluator Command
```sh
/data/minseo/.venvs/vllm/bin/python /data/minseo/experiment8/scripts/evaluate_ours_memory_gpu_throughput.py
```

## Pass/Fail Contract
PASS when a sustained live sample improves aggregate generation tokens/s or completed model calls/hour over the mature c512 baseline, all four replicas remain healthy, backend failures are zero, and memory.jsonl remains valid and duplicate-free

This evaluator must exist and produce concrete pass/fail evidence before the performance goal can be completed.
