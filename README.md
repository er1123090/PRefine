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
source /data/minseo/.venvs/vllm/bin/activate
cd /data/minseo/experiment8
python scripts/verify_layout.py
```

Install only the method-specific missing dependencies from `requirements.txt`;
API keys are read from the normal provider environment variables. The copied
JSON loaders have a small pandas-free compatibility path, so the base vLLM
environment can run vanilla/Ours CLI parsing and the offline checks. RAG still
requires `chromadb`, Mem0 requires `mem0ai`, and LangMem construction/inference
requires the LangMem/LangGraph packages. Local token re-counting also requires
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

Mem0:

```bash
python methods/mem0/build_memory.py \
  --input_path data/MPT_v2_mix600.json \
  --metrics_output outputs/mem0/construction_metrics.jsonl
```

LangMem:

```bash
python methods/langmem/build_memory.py \
  --input_path data/MPT_v2_mix600.json \
  --output_path outputs/langmem/memory.jsonl \
  --manifest_path outputs/langmem/memory.manifest.json
```

Vanilla LLM has no construction stage.

## Run inference

The common CLI fixes dataset/config paths and keeps the method-specific runtime.

```bash
python scripts/run_inference.py \
  --method vanilla_llm \
  --turn single \
  --query hint \
  --schema easy \
  --pref_type easy \
  --model gpt-4o-mini
```

For `ours_memory` and `langmem`, add `--memory_path`. For RAG, use
`--db_path`; Mem0 uses `MEM0_API_KEY`. Add `--dry_run` to inspect the exact
command without calling a model.

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
