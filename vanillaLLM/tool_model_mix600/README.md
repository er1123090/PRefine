# Gemma 4 mix600 vanilla LLM evaluation

This package materializes 1,319 mode-neutral targets and paired single/multi
renderings, runs Gemma 4 tool calling through a TP=4 vLLM OpenAI server, and
scores preference-slot value-OR metrics plus paired source-cluster bootstrap
intervals.

Offline rescore:

```bash
python /data/minseo/experiments4/vanillaLLM/gemma4_mix600/score.py \
  --run-root RUN_ROOT
```

Independent audit:

```bash
python /data/minseo/experiments4/vanillaLLM/gemma4_mix600/critic_audit.py \
  --latest-root /data/minseo/experiments4/vanillaLLM/inference/gemma4_mix600
```
