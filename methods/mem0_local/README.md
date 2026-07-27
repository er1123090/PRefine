# Mem0 Local

This method constructs Experiment8 memories without the managed Mem0 API. It
imports the official `mem0ai/mem0` source checkout pinned in `UPSTREAM.json`,
uses `gpt-oss-20b` through a local vLLM OpenAI-compatible endpoint, and stores
vectors in embedded Qdrant.

## Setup

```bash
bash methods/mem0_local/setup_local_env.sh
```

The official checkout is placed under `methods/mem0_local/upstream/mem0` and is
gitignored. The adapter refuses to run if its commit differs from
`UPSTREAM.json`. Setup also installs the official Mem0 `nlp` extra and
`en_core_web_sm`, so entity linking and BM25 lemmatization are not silently
disabled.

## Construction

When a vLLM endpoint is already running:

```bash
/data/minseo/.venvs/experiment8/bin/python \
  methods/mem0_local/build_memory.py \
  --input_path data/MPT_v2_0725.json \
  --base_url http://127.0.0.1:8000/v1 \
  --model gpt-oss-20b \
  --reasoning_effort low \
  --snapshot_mode all \
  --resume
```

To start one vLLM replica and then construct the artifact:

```bash
CUDA_DEVICE=0 bash methods/mem0_local/run_gpt_oss20b.sh
```

The default embedder is the local CPU/ONNX
`fastembed` provider with `BAAI/bge-small-en-v1.5`. To use the optional
Hugging Face `sentence-transformers` provider, install `sentence-transformers`
and pass `--embedding_provider huggingface`. To retain the previous OpenAI
embedding choice, pass:

```bash
--embedding_provider openai \
--embedding_model text-embedding-3-small \
--embedding_dims 1536 \
--embedding_api_key "$OPENAI_API_KEY"
```

No `MEM0_API_KEY` is read in either mode.

## Outputs and token accounting

- `MPT_v2_0725_construction.jsonl`: one completed user per line.
- `.manifest.json`: exact dataset/configuration fingerprint and upstream SHA.
- `.errors.jsonl`: resumable per-user failures.
- `MPT_v2_0725_qdrant`: persistent memory vectors.
- `MPT_v2_0725_history.sqlite`: official Mem0 history and rolling messages.

With `--snapshot_mode all`, every session records its post-session memory
snapshot, memory-count delta, stored-memory token delta, and vLLM usage. This
supports memory accumulation and session/token ablations. Construction lower
bound remains:

`session message tokens + max(0, stored-memory tokens after - before)`.

Embedding-internal tokens are not included in the lower bound or vLLM usage.
