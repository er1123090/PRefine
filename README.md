# Experiment 8 — MPT_v2

Experiment 8 fixes one comparison contract:

- method construction and inference semantics: **experiment4**
- parsing, normalization, and evaluation metrics: **experiment5**
- dataset/configuration: **MPT_v2 mix600**

No experiment7 manifest/facade layer is used here. Outputs and logs stay under
this directory.

The completed experiment4 vanilla runs for Qwen3 8B, Gemma4 12B, and
GPT-OSS 20B are preserved byte-for-byte under
`outputs/vanilla_llm/imported_experiment4/`. See the import note in that
directory for the exact source paths and integrity hashes.

## Layout

```text
data/                         MPT_v2 mix600
config/                       schema, preference map, hint/no-hint queries
methods/
  vanilla_llm/                exp4 prompt + compatible exp4 inference runtime
  ours_memory/                exp4 G-V-R builder and single/multi inference
  rag/                        exp4 Chroma ingestion + compatible retrieval runner
  mem0/                       exp4 Mem0 ingestion and single/multi inference
  langmem/                    exp4 memory builder/runtime + thin inference CLI
src/evaluation/metrics.py     exact experiment5 canonical metric implementation
evaluation/                   experiment5 single/multi evaluation CLIs
ablations/                    G-V-R, token, and API-argument analyses
scripts/run_inference.py      common inference CLI
```

## Source policy

The preserved files were copied from:

- `/data/minseo/experiment4/vanillaLLM`
- `/data/minseo/experiment4/ours_memory`
- `/data/minseo/experiment4/RAG`
- `/data/minseo/experiment4/mem0`
- `/data/minseo/experiment4/langmem`
- `/data/minseo/experiment5/src/evaluation`
- `/data/minseo/experiment5/evaluation/eval_{singleturn,multiturn}.py`

Several exp4 shell scripts referenced Python files that are no longer present in
the exp4 directory or its Git history: vanilla API inference, RAG inference,
LangMem inference CLIs, and `prompt_inference.py`. Experiment8 does not conceal
that gap:

- LangMem uses the complete inference implementation already present in exp4
  `langmem/common.py`, exposed through a thin CLI.
- vanilla and RAG use the data preparation, provider call, token accounting, and
  output contract from that same exp4 runtime.
- the two memory prompt templates are isolated in `src/exp4_prompts.py`.

These compatibility files are clearly separated from byte-preserved source
copies.

## Environment

```bash
cd /data/minseo/experiment8
python3 -m venv /data/minseo/.venvs/experiment8
source /data/minseo/.venvs/experiment8/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/verify_layout.py
```

The common inference and GVR launchers automatically prefer this Experiment8
environment when it exists. Local vLLM GPU launchers continue to use the
separate `/data/minseo/.venvs/vllm` environment. API keys are read from the
normal provider environment variables. Local token re-counting also requires
the selected tiktoken encoding to be cached; provider-reported usage works
without that cache by adding `--skip-local`.

## Build memory/index

Ours Memory:

```bash
python methods/ours_memory/build_memory.py \
  --input data/MPT_v2_mix600.json \
  --output outputs/ours_memory/memory.jsonl \
  --verifier_output outputs/ours_memory/verifier.jsonl \
  --refinement_output outputs/ours_memory/refinement.jsonl \
  --provider openai --model gpt-4o-mini
```

RAG:

```bash
python methods/rag/build_index.py \
  --input_path data/MPT_v2_mix600.json \
  --db_path outputs/rag/chroma
```

To build the RAG index with OpenRouter embeddings:

```bash
export OPENROUTER_API_KEY='...'

python methods/rag/build_index.py \
  --provider openrouter \
  --input_path data/MPT_v2_conflict_mixed_noise.json \
  --db_path outputs/rag/conflict_noise_chroma \
  --embedding_model openai/text-embedding-3-small
```

Mem0:

```bash
python methods/mem0/build_memory.py \
  --input_path data/MPT_v2_mix600.json \
  --metrics_output outputs/mem0/construction_metrics.jsonl
```

Local Mem0 OSS with `gpt-oss-20b`:

```bash
bash methods/mem0_local/setup_local_env.sh
CUDA_DEVICE=0 bash methods/mem0_local/run_gpt_oss20b.sh
```

This path uses the pinned official `mem0ai/mem0` checkout, local vLLM, local
Qdrant, and does not require `MEM0_API_KEY`. See
`methods/mem0_local/README.md` for resume and token-ablation details.

LangMem:

```bash
python methods/langmem/build_memory.py \
  --input_path data/MPT_v2_mix600.json \
  --output_path outputs/langmem/memory.jsonl \
  --manifest_path outputs/langmem/memory.manifest.json
```

For OpenRouter-based LangMem construction, add `--provider openrouter`, use an
OpenRouter chat-model slug for `--memory_model`, and use an OpenRouter
embedding-model slug for `--embedding_model`.

To resume the checkpointed 0725 LangMem construction through ordinary OpenAI
API calls without submitting new Batch jobs:

```bash
export OPENAI_API_KEY='...'

python methods/langmem/build_memory_batch.py direct \
  --input_path data/MPT_v2_0725.json \
  --output_dir outputs/langmem/MPT_v2_0725_gpt-5-mini_batch \
  --model gpt-5-mini \
  --reasoning_effort minimal \
  --direct_concurrency 10
```

Completed ordinary responses are checkpointed individually. Re-running the
same command resumes only missing responses, while preserving construction
usage, memory accumulation, and token/session ablation metadata.

A-MEM:

```bash
export OPENAI_API_KEY='...'

python methods/amem/build_memory_batch.py run \
  --input_path data/MPT_v2_0725.json \
  --output_dir outputs/amem/MPT_v2_0725_gpt-5-mini_batch \
  --model gpt-5-mini \
  --reasoning_effort minimal
```

This adapter is pinned to the official
[`WujiangXu/A-mem`](https://github.com/WujiangXu/A-mem) reproduction at commit
`0c8039f28fdcc08189a23c07a3437d9d2482f9c2`. Each dialogue turn is an
immutable note. Construction preserves the official two-call order:
metadata analysis, raw-content `k=5` neighbor search, then memory evolution.
The two calls become separate causal Batch jobs. For a bounded smoke test on
the first dataset row's first dialogue, add `--max_examples 1
--max_sessions 1`. Provenance and adapter boundaries are recorded in
`methods/amem/UPSTREAM.md`.

Vanilla LLM has no construction stage.

## Run inference

The common CLI supplies dataset/config defaults and keeps the method-specific runtime.

```bash
python scripts/run_inference.py \
  --method vanilla_llm \
  --turn single \
  --query hint \
  --schema easy \
  --pref_type easy \
  --model gpt-4o-mini
```

In Experiment8 methods, OpenRouter can be used for inference through its
OpenAI-compatible endpoint via `--provider openrouter`. Use an OpenRouter model
slug and keep the key out of the command line:

```bash
export OPENROUTER_API_KEY='...'

# Vanilla
python scripts/run_inference.py \
  --method vanilla_llm \
  --provider openrouter \
  --turn single \
  --input_path data/MPT_v2_conflict_mixed_noise.json \
  --query hint \
  --schema easy \
  --pref_type medium \
  --model openai/gpt-4o
```

```bash
# RAG
python scripts/run_inference.py \
  --method rag \
  --provider openrouter \
  --turn single \
  --input_path data/MPT_v2_conflict_mixed_noise.json \
  --db_path outputs/rag/chroma \
  --query hint \
  --schema all \
  --pref_type medium \
  --model openai/gpt-4o
```

```bash
# Mem0 (single)
python scripts/run_inference.py \
  --method mem0 \
  --provider openrouter \
  --turn single \
  --input_path data/MPT_v2_conflict_mixed_noise.json \
  --query hint \
  --schema easy \
  --pref_type medium \
  --model openai/gpt-4o
```

```bash
# LangMem
python scripts/run_inference.py \
  --method langmem \
  --provider openrouter \
  --turn single \
  --input_path data/MPT_v2_conflict_mixed_noise.json \
  --memory_path outputs/langmem/memory.jsonl \
  --query hint \
  --schema easy \
  --pref_type medium \
  --model openai/gpt-4o
```

```bash
# A-MEM
python scripts/run_inference.py \
  --method amem \
  --turn single \
  --input_path data/MPT_v2_0725.json \
  --memory_path outputs/amem/MPT_v2_0725_gpt-5-mini_batch/memory.jsonl \
  --query hint \
  --schema easy \
  --pref_type medium \
  --model gpt-5-mini \
  --reasoning_effort minimal
```

A-MEM inference defaults to the official-style semantic top 10 plus one-hop
directed-link expansion (up to 10 linked notes per semantic hit). The common
Experiment8 evaluation prompt and per-`example_id` memory isolation remain
unchanged.

To smoke-test the final A-MEM prediction through Batch API as well, use the
same flags for every lifecycle command:

```bash
python methods/amem/inference_batch.py prepare \
  --memory_path outputs/amem/MPT_v2_0725_gpt-5-mini_batch/memory.jsonl \
  --input_path data/MPT_v2_0725.json \
  --output_dir outputs/amem/MPT_v2_0725_gpt-5-mini_batch/inference_smoke \
  --model gpt-5-mini --reasoning_effort minimal --max_queries 1

python methods/amem/inference_batch.py submit \
  --memory_path outputs/amem/MPT_v2_0725_gpt-5-mini_batch/memory.jsonl \
  --input_path data/MPT_v2_0725.json \
  --output_dir outputs/amem/MPT_v2_0725_gpt-5-mini_batch/inference_smoke \
  --model gpt-5-mini --reasoning_effort minimal --max_queries 1

python methods/amem/inference_batch.py status \
  --memory_path outputs/amem/MPT_v2_0725_gpt-5-mini_batch/memory.jsonl \
  --input_path data/MPT_v2_0725.json \
  --output_dir outputs/amem/MPT_v2_0725_gpt-5-mini_batch/inference_smoke \
  --model gpt-5-mini --reasoning_effort minimal --max_queries 1

python methods/amem/inference_batch.py collect \
  --memory_path outputs/amem/MPT_v2_0725_gpt-5-mini_batch/memory.jsonl \
  --input_path data/MPT_v2_0725.json \
  --output_dir outputs/amem/MPT_v2_0725_gpt-5-mini_batch/inference_smoke \
  --model gpt-5-mini --reasoning_effort minimal --max_queries 1
```

`--provider openrouter` defaults to `https://openrouter.ai/api/v1` and reads
`OPENROUTER_API_KEY`. Explicit `--base_url` and `--api_key` values still
override those defaults. Use a concrete OpenRouter model slug when experiment
reproducibility matters.

The same provider flag works for `ours_memory`, RAG, Mem0 single/multi, and
LangMem. For `ours_memory` and LangMem, add `--memory_path`. For RAG, use the
same embedding model at index and inference time and pass `--db_path`. Mem0
still uses `MEM0_API_KEY` for its managed memory store; OpenRouter supplies
only the inference LLM. Add `--dry_run` to inspect the exact command without
calling a model.

For local `gpt-oss-20b` reasoning-high inference on GPUs 0–3, use the
optimized four-replica launcher. It runs one TP1 vLLM replica per GPU, routes
requests through the least-loaded proxy, and uses concurrency 256 with async
scheduling:

```bash
scripts/run_gpt_oss20b_high_dp4_inference.sh \
  --method vanilla_llm \
  --turn single \
  --query hint \
  --schema easy \
  --pref_type easy

scripts/run_gpt_oss20b_high_dp4_inference.sh \
  --method ours_memory \
  --turn single \
  --query hint \
  --schema easy \
  --pref_type easy \
  --memory_path outputs/ours_memory/.../memory.jsonl
```

The inference prompt, memory retrieval, and prediction output schema are
unchanged; only server placement, request scheduling, and client timeout/retry
settings differ.

## Evaluate with experiment5

```bash
python evaluation/eval_singleturn.py \
  --input_path outputs/.../predictions.json \
  --pref_list_path config/pref_list.json

python evaluation/eval_multiturn.py \
  --input_glob 'outputs/**/predictions.json' \
  --pref_list_path config/pref_list.json \
  --csv_output outputs/evaluation/summary.csv
```

Keep this evaluator fixed for every method and ablation. In particular, pass
model output as a string in `llm_output`; the experiment5 parser does not treat
a native Python/JSON dict in that field as a valid prediction.

## Reproducible vanilla Batch API sample

`scripts/run_vanilla_batch_sample.py` prepares one shared random sample across
single/multi and easy/medium/hard, then submits the exact same prompts to
OpenAI and Anthropic. GPT-5 uses `reasoning_effort=minimal`; Claude Haiku 4.5
omits `thinking`, so it is the non-reasoning condition.

```bash
source /data/minseo/.config/experiment8/secrets.env

python scripts/run_vanilla_batch_sample.py prepare \
  --sample-size 1000 \
  --seed 20260723

python scripts/run_vanilla_batch_sample.py submit --provider both
python scripts/run_vanilla_batch_sample.py status --provider both
python scripts/run_vanilla_batch_sample.py collect --provider both
```

The default run directory is
`outputs/vanilla_llm/batch_sample_1000_seed_20260723/`. It contains the shared
sample manifest and checksum, provider request files, batch IDs/status,
raw results, normalized `predictions.json`, `inference.jsonl`, and per-condition
plus aggregate `evaluation.json`. Submission refuses to create another billable
batch when a provider state file already exists unless
`--force-resubmit` is explicitly supplied.

For the provider-specific reusable checkpoints in `MPT_v2_0725`, use the
resume runner. It computes a separate exact complement for OpenAI and
Anthropic, excludes easy conflict rows, and merges completed Batch results
back into the existing 4,695-row checkpoints:

```bash
python scripts/run_mpt0725_vanilla_api_resume.py prepare
python scripts/run_mpt0725_vanilla_api_resume.py submit --provider both
python scripts/run_mpt0725_vanilla_api_resume.py status --provider both
python scripts/run_mpt0725_vanilla_api_resume.py collect --provider both
```

Preparation and submission refuse to overwrite recorded batch state. Provider
credentials are read only from `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`.

## MPT_v2 ablations

For G-V-R, experiment5 is the best source among experiments 4–7:

- exp4 only sweeps verifier retry count;
- exp5 explicitly separates generator-only and blind-refinement modes;
- exp6 is mainly an exp4 path/release re-layout;
- exp7 adapters are useful for contracts but change the experimental surface.

Experiment8 therefore keeps the main method at exp4 semantics, while the
isolated ablation builder comes from exp5 and adds the missing G+V/no-refine
condition.

```bash
bash ablations/gvr/run_mpt_v2.sh --dry-run
bash ablations/gvr/run_mpt_v2.sh
```

The same construction ablation can run through OpenRouter:

```bash
export OPENROUTER_API_KEY='...'

PROVIDER=openrouter \
MODEL=openai/gpt-4o \
INPUT_PATH=data/MPT_v2_conflict_mixed_noise.json \
bash ablations/gvr/run_mpt_v2.sh
```

Each resulting `memory.jsonl` can be passed to
`scripts/run_inference.py --method ours_memory --provider openrouter` for the
inference side of the ablation. Token-count and API-argument analyses consume
the resulting artifacts and are provider-independent.

The four controlled conditions are:

- `g_only`: generator
- `gv_no_refine`: generator + verifier, no refinement
- `gr_no_verifier`: generator + one blind refinement
- `gvr`: generator + verifier-feedback refinement

The complete matrix is in `config/ablation_matrix.json`. Use the same model,
temperature policy, query/schema pair, evaluator, and sample set across the
four conditions.

Token accounting:

```bash
python ablations/token_count/analyze.py \
  'outputs/**/*.json*' \
  --input-cost-per-million INPUT_USD_PER_1M \
  --cached-input-cost-per-million CACHED_INPUT_USD_PER_1M \
  --output-cost-per-million OUTPUT_USD_PER_1M \
  --csv_output outputs/ablation_tokens/rows.csv \
  --session_csv_output outputs/ablation_tokens/sessions.csv \
  --json_output outputs/ablation_tokens/report.json
```

Use the rates for the exact provider/model and experiment date; the analyzer
does not hard-code prices. Omit all three cost flags when only token counts are
needed. Input and output rates must be supplied together; the cached-input rate
is optional and falls back to the normal input rate.

Construction outputs now preserve two different kinds of evidence:

- `construction_token_usage` is provider-reported usage, split into total,
  component, session, and call records. Ours records generator/refiner/verifier
  calls. LangMem records memory-manager/extractor and hosted embedding calls.
- LangMem's manifest also stores one-time runtime setup usage in
  `setup_token_usage`; include the manifest in the analyzer input when that
  setup cost should be reported.
- local measurements use `cl100k_base`: Mem0 stores
  `local_construction_input_tokens`, and every method stores
  `stored_memory_tokens_after_session` when a session snapshot is available.

At inference time Mem0 and LangMem store `retrieved_memory_tokens`, the token
count of actual retrieved memories, plus `retrieval_prompt_tokens`, the exact
inserted retrieval block including the no-result placeholder. The analyzer
reports `retrieval_estimated_input_cost_usd`. This is the retrieved portion of
the full inference input cost, not an extra cost to add on top of the request's
provider-reported input cost.

Important interpretation details:

- Mem0's managed API does not always return its internal LLM token usage. When
  it does not, `provider_usage_coverage` is `0` and
  `construction_estimated_cost_usd` is `null`, not zero. The locally counted
  construction payload and its input-rate equivalent remain available as
  `local_construction_input_estimated_cost_usd`; that value is a comparison
  proxy, not a provider invoice.
- `stored_memory_estimated_input_cost_usd` is the input-rate equivalent of the
  cumulative serialized memory after a session. Storage itself is not
  necessarily billed per token.
- Mem0 construction performs one `get_all` read after each session to obtain
  the cumulative memory snapshot used for token measurement.
- LangMem local embedding backends have no provider token charge. Hosted
  OpenAI-compatible embedding usage is recorded only when the endpoint returns
  usage metadata.

API-argument matching with the experiment5 parser:

```bash
python ablations/api_arguments/analyze.py \
  --input_path outputs/.../predictions.json \
  --pref_list_path config/pref_list.json \
  --csv_output outputs/api_arguments/rows.csv \
  --json_output outputs/api_arguments/report.json
```
