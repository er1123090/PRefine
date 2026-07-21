# LangMem Baseline

This folder adds a `langmem`-based semantic memory baseline for `experiments4`.

## What is included

- `step1_build_memory.py`
  Builds user-scoped LangMem semantic memories from the dataset and exports a local snapshot JSONL.
- `langmem_inference_single.py`
  Runs single-turn inference from a saved snapshot.
- `langmem_inference_multi.py`
  Runs multi-turn inference from a saved snapshot.
- `measure_session_memory_token_delta.py`
  Replays sessions and records how stored memory tokens and retrieved memory tokens change after each session.
- `plot_session_memory_token_growth.py`
  Aggregates the session delta CSV into a plot and per-session summary CSV.
- `run_langmem_build.sh`, `run_langmem_single.sh`, `run_langmem_multi.sh`, `run_langmem_measure_token_delta.sh`
  Convenience launchers that also configure `.run.log` execution logs.
- `run_langmem_build_vllm.sh`
  Launches a local vLLM server for the memory-generation model and writes LangMem snapshots per model.

## Installation

Use an isolated environment for this baseline. The root repo pins older package versions and `langmem==0.0.30` pulls newer LangChain/OpenAI dependencies.

```bash
python -m venv .venv-langmem
source .venv-langmem/bin/activate
pip install -r /data/minseo/experiments4/langmem/requirements.txt
```

## Python 3.10 note

`langmem==0.0.30` imports `typing.NotRequired` directly, which breaks on Python 3.10. The scripts in this folder include a compatibility shim in [`_compat.py`](/data/minseo/experiments4/langmem/_compat.py) so they can still import the runtime on Python 3.10.

## Example workflow

Build the snapshot:

```bash
python /data/minseo/experiments4/langmem/step1_build_memory.py \
  --memory_model gpt-4o-mini \
  --output_path /data/minseo/experiments4/langmem/memory_snapshots/langmem_1229_dev_6.jsonl \
  --manifest_path /data/minseo/experiments4/langmem/memory_snapshots/langmem_1229_dev_6.manifest.json
```

Build the snapshot with a vLLM-served memory model:

```bash
bash /data/minseo/experiments4/langmem/run_langmem_build_vllm.sh
```

If the vLLM server only hosts the chat model, keep embeddings separate by setting `OPENAI_API_KEY` or exporting `EMBEDDING_BASE_URL` and `EMBEDDING_API_KEY` before running the script.

Run single-turn inference:

```bash
python /data/minseo/experiments4/langmem/langmem_inference_single.py \
  --memory_path /data/minseo/experiments4/langmem/memory_snapshots/langmem_1229_dev_6.jsonl \
  --pref_type easy \
  --context_type memory_only \
  --model_name gpt-5
```

Run multi-turn inference:

```bash
python /data/minseo/experiments4/langmem/langmem_inference_multi.py \
  --memory_path /data/minseo/experiments4/langmem/memory_snapshots/langmem_1229_dev_6.jsonl \
  --pref_type easy \
  --context_type memory_only \
  --model_name gpt-5
```

Measure session memory growth:

```bash
python /data/minseo/experiments4/langmem/measure_session_memory_token_delta.py \
  --memory_model gpt-4o-mini \
  --output_csv /data/minseo/experiments4/langmem/session_memory_token_deltas_1229_dev_6.csv \
  --summary_json /data/minseo/experiments4/langmem/session_memory_token_deltas_1229_dev_6.summary.json
```

Plot memory growth:

```bash
python /data/minseo/experiments4/langmem/plot_session_memory_token_growth.py \
  --input_csv /data/minseo/experiments4/langmem/session_memory_token_deltas_1229_dev_6.csv
```

## Logging

Each runnable script now supports a human-readable execution log via `--run_log_path`.

- Build and measurement scripts automatically create `<output_stem>.run.log` if `--run_log_path` is omitted.
- Inference scripts keep the structured per-example JSONL log in `--log_path` and write progress/error messages to `--run_log_path`.
- The helper shell scripts already pass explicit `.run.log` paths and use `.jsonl` for structured inference logs.

## Endpoint Split

The LangMem scripts support separate endpoints for the memory LLM and embeddings.

- Step 1 build and session-delta measurement:
  `--base_url` / `--api_key` control the memory-generation model.
  `--embedding_base_url` / `--embedding_api_key` optionally override the embedding endpoint.
- Step 2 inference:
  `--base_url` / `--api_key` control the inference model.
  `--embedding_base_url` / `--embedding_api_key` optionally override the retrieval embedding endpoint used when restoring the snapshot.

Gemini inference can now run directly with `GOOGLE_API_KEY` only. Set `--base_url` (or `GEMINI_CHAT_BASE_URL` in the helper scripts) only when you want to use an OpenAI-compatible Gemini proxy. Retrieval embeddings are still configured separately through `--embedding_base_url` / `--embedding_api_key` or the helper-script defaults based on `OPENAI_API_KEY`.
