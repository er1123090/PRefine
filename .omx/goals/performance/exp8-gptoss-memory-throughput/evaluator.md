# Performance Evaluator: exp8-gptoss-memory-throughput

## Objective
Maximize experiment8 gpt-oss-20b ours_memory construction throughput on GPUs 0-3 without output corruption or request failures

## Evaluator Command
```sh
/data/minseo/.venvs/vllm/bin/python /data/minseo/experiment8/scripts/evaluate_ours_memory_gpu_throughput.py
```

## Pass/Fail Contract
PASS when all four replicas are healthy, each sustains 64 in-flight requests, backend failures are zero, and recent aggregate generation throughput is at least 6500 tokens/s (2x the c128 baseline)

This evaluator must exist and produce concrete pass/fail evidence before the performance goal can be completed.
