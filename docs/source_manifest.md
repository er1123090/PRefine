# Source Manifest

This manifest maps the public `experiments6` release back to the current filesystem state of `/data/minseo/experiments4`.
`e-mem` is intentionally excluded because the release scope is “everything except e-mem”. Generated caches, logs, inference outputs, vector stores, and virtualenvs are also excluded.

## Summary

- Non-e-mem source files inventoried: 219
- Source files copied: 201
- Source files intentionally omitted: 18
- Additional release control files listed in the source table: 1
- Non-source config/data/artifact files copied: 119

## Directory Mapping

| experiments4 source | experiments6 release path |
| --- | --- |
| `ours_memory` | `src/preference_memory` |
| `vanillaLLM` | `src/baselines/vanilla_llm` |
| `RAG` | `src/baselines/rag` |
| `mem0` | `src/baselines/mem0` |
| `langmem` | `src/baselines/langmem` |
| `evaluation` | `src/evaluation` |
| `extended_schema` | `src/extended_schema` |
| `memory-dev` | `src/dataset_tools` |
| `self-refine` | `src/self_refine` |
| `data` | `data` |
| `root schema/query/preference JSON` | `configs` |
| `root rebuttal/token analysis` | `src/analysis and artifacts/token_cost` |
| `_paper` | `artifacts/paper_tables` |

## Source Files

| Source | Release path / omission reason | Status |
| --- | --- | --- |
| `.gitignore` | `.gitignore` | release control file rewritten for public tracking rules |
| `RAG/RAG_add.py` | `src/baselines/rag/RAG_add.py` | copied |
| `RAG/RAG_inference_multi.py` | `src/baselines/rag/RAG_inference_multi.py` | copied |
| `RAG/RAG_inference_multi.sh` | `src/baselines/rag/RAG_inference_multi.sh` | copied |
| `RAG/RAG_inference_single.py` | `src/baselines/rag/RAG_inference_single.py` | copied |
| `RAG/RAG_inference_single.sh` | `src/baselines/rag/RAG_inference_single.sh` | copied |
| `RAG/calculate_avg_tokens_20260120.py` | `src/baselines/rag/calculate_avg_tokens_20260120.py` | copied |
| `RAG/calculate_avg_tokens_20260120_old.py` | `src/baselines/rag/calculate_avg_tokens_20260120_old.py` | copied |
| `RAG/inference_multi/gemini-3-flash/concat_output.py` | `src/baselines/rag/result_tools/multiturn_concat_output.py` | copied |
| `RAG/inference_single/gemini-3-flash/concat_output.py` | `src/baselines/rag/result_tools/singleturn_concat_output.py` | copied |
| `RAG/prompt_inference.py` | `src/baselines/rag/prompt_inference.py` | copied |
| `data/calculate_avg_tokens.py` | `data/calculate_avg_tokens.py` | copied |
| `data/sampling.py` | `data/sampling.py` | copied |
| `evaluation/DEPRE-evaluation.py` | `src/evaluation/DEPRE-evaluation.py` | copied |
| `evaluation/DEPRE-evaluation_aggregate_multi-turn.py` | `src/evaluation/DEPRE-evaluation_aggregate_multi-turn.py` | copied |
| `evaluation/DEPRE-evaluation_aggregate_singleturn.py` | `src/evaluation/DEPRE-evaluation_aggregate_singleturn.py` | copied |
| `evaluation/DEPRE-evaluation_aggregate_singleturn2.py` | `src/evaluation/DEPRE-evaluation_aggregate_singleturn2.py` | copied |
| `evaluation/DEPRE-evaluation_multiturn-f1-aggregate.py` | `src/evaluation/DEPRE-evaluation_multiturn-f1-aggregate.py` | copied |
| `evaluation/DEPRE-evaluation_multiturn-f1-parse-aggregate.py` | `src/evaluation/DEPRE-evaluation_multiturn-f1-parse-aggregate.py` | copied |
| `evaluation/evaluation_multiturn-f1-parse-aggregate.py` | `src/evaluation/evaluation_multiturn-f1-parse-aggregate.py` | copied |
| `evaluation/evaluation_multiturn-f1-prefgroup-aggregate.py` | `src/evaluation/evaluation_multiturn-f1-prefgroup-aggregate.py` | copied |
| `evaluation/evaluation_reasoning_multi.py` | `src/evaluation/evaluation_reasoning_multi.py` | copied |
| `evaluation/evaluation_reasoning_multi_deepseek.py` | `src/evaluation/evaluation_reasoning_multi_deepseek.py` | copied |
| `evaluation/evaluation_reasoning_single.py` | `src/evaluation/evaluation_reasoning_single.py` | copied |
| `evaluation/evaluation_reasoning_single_deepseek.py` | `src/evaluation/evaluation_reasoning_single_deepseek.py` | copied |
| `evaluation/evaluation_singleturn-f1-aggregate.py` | `src/evaluation/evaluation_singleturn-f1-aggregate.py` | copied |
| `evaluation/evaluation_singleturn-f1-prefgroup-aggregate.py` | `src/evaluation/evaluation_singleturn-f1-prefgroup-aggregate.py` | copied |
| `evaluation/evaluation_singleturn-f1-prefgroup.py` | `src/evaluation/evaluation_singleturn-f1-prefgroup.py` | copied |
| `evaluation/evaluation_singleturn-f1.py` | `src/evaluation/evaluation_singleturn-f1.py` | copied |
| `evaluation/evaluation_slot_count.py` | `src/evaluation/evaluation_slot_count.py` | copied |
| `extended_schema/RAG/RAG_inference_multi_fixed_pairs.py` | `src/extended_schema/RAG/RAG_inference_multi_fixed_pairs.py` | copied |
| `extended_schema/RAG/RAG_inference_single_fixed_pairs.py` | `src/extended_schema/RAG/RAG_inference_single_fixed_pairs.py` | copied |
| `extended_schema/RAG/run_rag-multi-extended-400.sh` | `src/extended_schema/RAG/run_rag-multi-extended-400.sh` | copied |
| `extended_schema/RAG/run_rag-single-extended-400.sh` | `src/extended_schema/RAG/run_rag-single-extended-400.sh` | copied |
| `extended_schema/build_fixed_400_comparison_csvs.py` | `src/extended_schema/build_fixed_400_comparison_csvs.py` | copied |
| `extended_schema/common_fixed_pairs.py` | `src/extended_schema/common_fixed_pairs.py` | copied |
| `extended_schema/langmem/build_fixed_400_comparison_csvs.py` | `src/extended_schema/langmem/build_fixed_400_comparison_csvs.py` | copied |
| `extended_schema/langmem/langmem_inference_multi_fixed_pairs.py` | `src/extended_schema/langmem/langmem_inference_multi_fixed_pairs.py` | copied |
| `extended_schema/langmem/langmem_inference_single_fixed_pairs.py` | `src/extended_schema/langmem/langmem_inference_single_fixed_pairs.py` | copied |
| `extended_schema/langmem/run_langmem-multi-extended-400.sh` | `src/extended_schema/langmem/run_langmem-multi-extended-400.sh` | copied |
| `extended_schema/langmem/run_langmem-single-extended-400.sh` | `src/extended_schema/langmem/run_langmem-single-extended-400.sh` | copied |
| `extended_schema/mem0/mem0_evaluate_multiturn_fixed_pairs.py` | `src/extended_schema/mem0/mem0_evaluate_multiturn_fixed_pairs.py` | copied |
| `extended_schema/mem0/mem0_evaluate_singleturn_fixed_pairs.py` | `src/extended_schema/mem0/mem0_evaluate_singleturn_fixed_pairs.py` | copied |
| `extended_schema/mem0/rerun_503_and_merge.py` | `src/extended_schema/mem0/rerun_503_and_merge.py` | copied |
| `extended_schema/mem0/run_mem0-multi-extended-400.sh` | `src/extended_schema/mem0/run_mem0-multi-extended-400.sh` | copied |
| `extended_schema/mem0/run_mem0-single-extended-400.sh` | `src/extended_schema/mem0/run_mem0-single-extended-400.sh` | copied |
| `extended_schema/ours_memory/Preference_Memory_step2_ACTION_multiturn_VLLM2_extended_hard.sh` | `src/extended_schema/ours_memory/Preference_Memory_step2_ACTION_multiturn_VLLM2_extended_hard.sh` | copied |
| `extended_schema/ours_memory/Preference_Memory_step2_ACTION_multiturn_api2_extended_hard.sh` | `src/extended_schema/ours_memory/Preference_Memory_step2_ACTION_multiturn_api2_extended_hard.sh` | copied |
| `extended_schema/ours_memory/Preference_Memory_step2_ACTION_multiturn_api_fixed_pairs.py` | `src/extended_schema/ours_memory/Preference_Memory_step2_ACTION_multiturn_api_fixed_pairs.py` | copied |
| `extended_schema/ours_memory/Preference_Memory_step2_ACTION_singleturn_VLLM2_extended_hard.sh` | `src/extended_schema/ours_memory/Preference_Memory_step2_ACTION_singleturn_VLLM2_extended_hard.sh` | copied |
| `extended_schema/ours_memory/Preference_Memory_step2_ACTION_singleturn_api5_extended_hard.sh` | `src/extended_schema/ours_memory/Preference_Memory_step2_ACTION_singleturn_api5_extended_hard.sh` | copied |
| `extended_schema/ours_memory/Preference_Memory_step2_ACTION_singleturn_api_fixed_pairs.py` | `src/extended_schema/ours_memory/Preference_Memory_step2_ACTION_singleturn_api_fixed_pairs.py` | copied |
| `extended_schema/vanillaLLM/run_vanillaLLM-api-multi-extended-400.sh` | `src/extended_schema/vanillaLLM/run_vanillaLLM-api-multi-extended-400.sh` | copied |
| `extended_schema/vanillaLLM/run_vanillaLLM-api-single-extended-400.sh` | `src/extended_schema/vanillaLLM/run_vanillaLLM-api-single-extended-400.sh` | copied |
| `extended_schema/vanillaLLM/run_vanillaLLM-vllm-multi-extended-400.sh` | `src/extended_schema/vanillaLLM/run_vanillaLLM-vllm-multi-extended-400.sh` | copied |
| `extended_schema/vanillaLLM/run_vanillaLLM-vllm-single-extended-400.sh` | `src/extended_schema/vanillaLLM/run_vanillaLLM-vllm-single-extended-400.sh` | copied |
| `extended_schema/vanillaLLM/vanillaLLM_inference-api-multi_fixed_pairs.py` | `src/extended_schema/vanillaLLM/vanillaLLM_inference-api-multi_fixed_pairs.py` | copied |
| `extended_schema/vanillaLLM/vanillaLLM_inference-api-single_fixed_pairs.py` | `src/extended_schema/vanillaLLM/vanillaLLM_inference-api-single_fixed_pairs.py` | copied |
| `langmem/_compat.py` | `src/baselines/langmem/_compat.py` | copied |
| `langmem/batch_inference_runner.py` | `src/baselines/langmem/batch_inference_runner.py` | copied |
| `langmem/common.py` | `src/baselines/langmem/common.py` | copied |
| `langmem/langmem_inference_multi.py` | `src/baselines/langmem/langmem_inference_multi.py` | copied |
| `langmem/langmem_inference_single.py` | `src/baselines/langmem/langmem_inference_single.py` | copied |
| `langmem/measure_session_memory_token_delta.py` | `src/baselines/langmem/measure_session_memory_token_delta.py` | copied |
| `langmem/plot_session_memory_token_growth.py` | `src/baselines/langmem/plot_session_memory_token_growth.py` | copied |
| `langmem/plot_snapshot_memory_token_growth.py` | `src/baselines/langmem/plot_snapshot_memory_token_growth.py` | copied |
| `langmem/plot_snapshot_memory_token_growth_report.py` | `src/baselines/langmem/plot_snapshot_memory_token_growth_report.py` | copied |
| `langmem/prompt_inference.py` | `src/baselines/langmem/prompt_inference.py` | copied |
| `langmem/run_langmem_build.sh` | `src/baselines/langmem/run_langmem_build.sh` | copied |
| `langmem/run_langmem_build_vllm.sh` | `src/baselines/langmem/run_langmem_build_vllm.sh` | copied |
| `langmem/run_langmem_measure_token_delta.sh` | `src/baselines/langmem/run_langmem_measure_token_delta.sh` | copied |
| `langmem/run_langmem_multi.sh` | `src/baselines/langmem/run_langmem_multi.sh` | copied |
| `langmem/run_langmem_single.sh` | `src/baselines/langmem/run_langmem_single.sh` | copied |
| `langmem/step1_build_memory.py` | `src/baselines/langmem/step1_build_memory.py` | copied |
| `measure_local_memory_construction_cost_gemma.py` | `src/analysis/measure_local_memory_construction_cost_gemma.py` | copied |
| `mem0/0323_memory-tokens-265/session_memory_token_deltas_1229_dev_6_20260323_093926/launch_shard0.sh` | `-` | excluded generated mem0 shard/output launcher artifact |
| `mem0/0323_memory-tokens-265/session_memory_token_deltas_1229_dev_6_20260323_093926/launch_shard1.sh` | `-` | excluded generated mem0 shard/output launcher artifact |
| `mem0/0323_memory-tokens-265/session_memory_token_deltas_1229_dev_6_20260323_093926/launch_shard2.sh` | `-` | excluded generated mem0 shard/output launcher artifact |
| `mem0/0323_memory-tokens-265/session_memory_token_deltas_1229_dev_6_20260323_093926/launch_shard3.sh` | `-` | excluded generated mem0 shard/output launcher artifact |
| `mem0/0323_memory-tokens-265/session_memory_token_deltas_1229_dev_6_20260323_093926/launch_shard4.sh` | `-` | excluded generated mem0 shard/output launcher artifact |
| `mem0/0323_memory-tokens-265/session_memory_token_deltas_1229_dev_6_20260323_093926/launch_shard5.sh` | `-` | excluded generated mem0 shard/output launcher artifact |
| `mem0/0323_memory-tokens-265/session_memory_token_deltas_1229_dev_6_20260323_094003/launch_shard0.sh` | `-` | excluded generated mem0 shard/output launcher artifact |
| `mem0/0323_memory-tokens-265/session_memory_token_deltas_1229_dev_6_20260323_094003/launch_shard1.sh` | `-` | excluded generated mem0 shard/output launcher artifact |
| `mem0/0323_memory-tokens-265/session_memory_token_deltas_1229_dev_6_20260323_094003/launch_shard2.sh` | `-` | excluded generated mem0 shard/output launcher artifact |
| `mem0/0323_memory-tokens-265/session_memory_token_deltas_1229_dev_6_20260323_094003/launch_shard3.sh` | `-` | excluded generated mem0 shard/output launcher artifact |
| `mem0/0323_memory-tokens-265/session_memory_token_deltas_1229_dev_6_20260323_094003/launch_shard4.sh` | `-` | excluded generated mem0 shard/output launcher artifact |
| `mem0/0323_memory-tokens-265/session_memory_token_deltas_1229_dev_6_20260323_094003/launch_shard5.sh` | `-` | excluded generated mem0 shard/output launcher artifact |
| `mem0/0323_memory-tokens_111/session_memory_token_deltas_1229_dev_6_20260323_063951/launch_shard0.sh` | `-` | excluded generated mem0 shard/output launcher artifact |
| `mem0/0323_memory-tokens_111/session_memory_token_deltas_1229_dev_6_20260323_063951/launch_shard1.sh` | `-` | excluded generated mem0 shard/output launcher artifact |
| `mem0/0323_memory-tokens_111/session_memory_token_deltas_1229_dev_6_20260323_063951/launch_shard2.sh` | `-` | excluded generated mem0 shard/output launcher artifact |
| `mem0/0323_memory-tokens_111/session_memory_token_deltas_1229_dev_6_20260323_063951/launch_shard3.sh` | `-` | excluded generated mem0 shard/output launcher artifact |
| `mem0/0323_memory-tokens_111/session_memory_token_deltas_1229_dev_6_20260323_063951/launch_shard4.sh` | `-` | excluded generated mem0 shard/output launcher artifact |
| `mem0/0323_memory-tokens_111/session_memory_token_deltas_1229_dev_6_20260323_063951/launch_shard5.sh` | `-` | excluded generated mem0 shard/output launcher artifact |
| `mem0/calculate_avg_tokens.py` | `src/baselines/mem0/calculate_avg_tokens.py` | copied |
| `mem0/measure_session_memory_token_delta.py` | `src/baselines/mem0/measure_session_memory_token_delta.py` | copied |
| `mem0/mem0-api.py` | `src/baselines/mem0/mem0-api.py` | copied |
| `mem0/merge_session_memory_token_deltas.py` | `src/baselines/mem0/merge_session_memory_token_deltas.py` | copied |
| `mem0/merge_session_memory_token_deltas_when_done.sh` | `src/baselines/mem0/merge_session_memory_token_deltas_when_done.sh` | copied |
| `mem0/plot_session_memory_token_growth.py` | `src/baselines/mem0/plot_session_memory_token_growth.py` | copied |
| `mem0/probe_event_usage_and_memory_diff.py` | `src/baselines/mem0/probe_event_usage_and_memory_diff.py` | copied |
| `mem0/prompt_inference.py` | `src/baselines/mem0/prompt_inference.py` | copied |
| `mem0/run_session_memory_token_delta_nohup.sh` | `src/baselines/mem0/run_session_memory_token_delta_nohup.sh` | copied |
| `mem0/step1_add.py` | `src/baselines/mem0/step1_add.py` | copied |
| `mem0/step2_evaluate-HARD.py` | `src/baselines/mem0/step2_evaluate-HARD.py` | copied |
| `mem0/step2_evaluate.py` | `src/baselines/mem0/step2_evaluate.py` | copied |
| `mem0/step2_evaluate_multiturn.py` | `src/baselines/mem0/step2_evaluate_multiturn.py` | copied |
| `mem0/step2_evaluate_multiturn.sh` | `src/baselines/mem0/step2_evaluate_multiturn.sh` | copied |
| `mem0/step2_evaluate_singleturn.py` | `src/baselines/mem0/step2_evaluate_singleturn.py` | copied |
| `mem0/step2_evaluate_singleturn.sh` | `src/baselines/mem0/step2_evaluate_singleturn.sh` | copied |
| `mem0/utils_mem0.py` | `src/baselines/mem0/utils_mem0.py` | copied |
| `memory-dev/datasets/sgd2ours/export_instance_api_calls_pref_vg.py` | `src/dataset_tools/datasets/sgd2ours/export_instance_api_calls_pref_vg.py` | copied |
| `memory-dev/datasets/sgd2ours/export_value_groups_tsv.py` | `src/dataset_tools/datasets/sgd2ours/export_value_groups_tsv.py` | copied |
| `memory-dev/datasets/sgd2ours/query/valid_query.py` | `src/dataset_tools/datasets/sgd2ours/query/valid_query.py` | copied |
| `memory-dev/datasets/sgd2ours/sgd_init_1.py` | `src/dataset_tools/datasets/sgd2ours/sgd_init_1.py` | copied |
| `memory-dev/datasets/sgd2ours/sgd_init_2.py` | `src/dataset_tools/datasets/sgd2ours/sgd_init_2.py` | copied |
| `memory-dev/datasets/sgd2ours/sgd_init_3.py` | `src/dataset_tools/datasets/sgd2ours/sgd_init_3.py` | copied |
| `memory-dev/datasets/sgd2ours/sgd_init_4.py` | `src/dataset_tools/datasets/sgd2ours/sgd_init_4.py` | copied |
| `memory-dev/datasets/sgd2ours/sgd_init_5.py` | `src/dataset_tools/datasets/sgd2ours/sgd_init_5.py` | copied |
| `memory-dev/datasets/sgd2ours/sgd_init_6.py` | `src/dataset_tools/datasets/sgd2ours/sgd_init_6.py` | copied |
| `memory-dev/datasets/sgd2ours/sgd_query_split.py` | `src/dataset_tools/datasets/sgd2ours/sgd_query_split.py` | copied |
| `memory-dev/datasets/sgd2ours/sgd_stat.py` | `src/dataset_tools/datasets/sgd2ours/sgd_stat.py` | copied |
| `memory-dev/methods/ReMem/evaluate/evaluate.py` | `src/dataset_tools/methods/ReMem/evaluate/evaluate.py` | copied |
| `memory-dev/methods/ReMem/run_all.sh` | `src/dataset_tools/methods/ReMem/run_all.sh` | copied |
| `memory-dev/methods/ReMem/script.py` | `src/dataset_tools/methods/ReMem/script.py` | copied |
| `memory-dev/methods/valid/analyze_false_calls.py` | `src/dataset_tools/methods/valid/analyze_false_calls.py` | copied |
| `memory-dev/methods/valid/nan2null.py` | `src/dataset_tools/methods/valid/nan2null.py` | copied |
| `ours_memory/DEPRECATED/DEPRE-prompt_conf.py` | `src/preference_memory/legacy_deprecated/DEPRE-prompt_conf.py` | copied |
| `ours_memory/DEPRECATED/DEPRE-step1_generate_memory-confidence-noexp.py` | `src/preference_memory/legacy_deprecated/DEPRE-step1_generate_memory-confidence-noexp.py` | copied |
| `ours_memory/DEPRECATED/DEPRE-step1_generate_memory-confidence.py` | `src/preference_memory/legacy_deprecated/DEPRE-step1_generate_memory-confidence.py` | copied |
| `ours_memory/DEPRECATED/DEPRE-step1_generate_memory-noexp.py` | `src/preference_memory/legacy_deprecated/DEPRE-step1_generate_memory-noexp.py` | copied |
| `ours_memory/DEPRECATED/DEPRE-step2_inference_memory.py` | `src/preference_memory/legacy_deprecated/DEPRE-step2_inference_memory.py` | copied |
| `ours_memory/DEPRECATED/DEPRE-temp_memory.py` | `src/preference_memory/legacy_deprecated/DEPRE-temp_memory.py` | copied |
| `ours_memory/Preference_Memory_step1_LATENTPREF.py` | `src/preference_memory/Preference_Memory_step1_LATENTPREF.py` | copied |
| `ours_memory/Preference_Memory_step1_LATENTPREF_VLLM.py` | `src/preference_memory/Preference_Memory_step1_LATENTPREF_VLLM.py` | copied |
| `ours_memory/Preference_Memory_step1_LATENTPREF_ablation.py` | `src/preference_memory/Preference_Memory_step1_LATENTPREF_ablation.py` | copied |
| `ours_memory/Preference_Memory_step1_LATENTPREF_ablation.sh` | `src/preference_memory/Preference_Memory_step1_LATENTPREF_ablation.sh` | copied |
| `ours_memory/Preference_Memory_step1_VLLM.sh` | `src/preference_memory/Preference_Memory_step1_VLLM.sh` | copied |
| `ours_memory/Preference_Memory_step2_ACTION_multiturn_VLLM.py` | `src/preference_memory/Preference_Memory_step2_ACTION_multiturn_VLLM.py` | copied |
| `ours_memory/Preference_Memory_step2_ACTION_multiturn_VLLM.sh` | `src/preference_memory/Preference_Memory_step2_ACTION_multiturn_VLLM.sh` | copied |
| `ours_memory/Preference_Memory_step2_ACTION_multiturn_VLLM2.sh` | `src/preference_memory/Preference_Memory_step2_ACTION_multiturn_VLLM2.sh` | copied |
| `ours_memory/Preference_Memory_step2_ACTION_multiturn_VLLM2_extended_hard.sh` | `src/preference_memory/Preference_Memory_step2_ACTION_multiturn_VLLM2_extended_hard.sh` | copied |
| `ours_memory/Preference_Memory_step2_ACTION_multiturn_api.py` | `src/preference_memory/Preference_Memory_step2_ACTION_multiturn_api.py` | copied |
| `ours_memory/Preference_Memory_step2_ACTION_multiturn_api.sh` | `src/preference_memory/Preference_Memory_step2_ACTION_multiturn_api.sh` | copied |
| `ours_memory/Preference_Memory_step2_ACTION_multiturn_api2.sh` | `src/preference_memory/Preference_Memory_step2_ACTION_multiturn_api2.sh` | copied |
| `ours_memory/Preference_Memory_step2_ACTION_multiturn_api2_extended_hard.sh` | `src/preference_memory/Preference_Memory_step2_ACTION_multiturn_api2_extended_hard.sh` | copied |
| `ours_memory/Preference_Memory_step2_ACTION_singleturn_VLLM.py` | `src/preference_memory/Preference_Memory_step2_ACTION_singleturn_VLLM.py` | copied |
| `ours_memory/Preference_Memory_step2_ACTION_singleturn_VLLM.sh` | `src/preference_memory/Preference_Memory_step2_ACTION_singleturn_VLLM.sh` | copied |
| `ours_memory/Preference_Memory_step2_ACTION_singleturn_VLLM2.sh` | `src/preference_memory/Preference_Memory_step2_ACTION_singleturn_VLLM2.sh` | copied |
| `ours_memory/Preference_Memory_step2_ACTION_singleturn_VLLM2_extended_hard.sh` | `src/preference_memory/Preference_Memory_step2_ACTION_singleturn_VLLM2_extended_hard.sh` | copied |
| `ours_memory/Preference_Memory_step2_ACTION_singleturn_api.py` | `src/preference_memory/Preference_Memory_step2_ACTION_singleturn_api.py` | copied |
| `ours_memory/Preference_Memory_step2_ACTION_singleturn_api.sh` | `src/preference_memory/Preference_Memory_step2_ACTION_singleturn_api.sh` | copied |
| `ours_memory/Preference_Memory_step2_ACTION_singleturn_api2.sh` | `src/preference_memory/Preference_Memory_step2_ACTION_singleturn_api2.sh` | copied |
| `ours_memory/Preference_Memory_step2_ACTION_singleturn_api3.sh` | `src/preference_memory/Preference_Memory_step2_ACTION_singleturn_api3.sh` | copied |
| `ours_memory/Preference_Memory_step2_ACTION_singleturn_api4.sh` | `src/preference_memory/Preference_Memory_step2_ACTION_singleturn_api4.sh` | copied |
| `ours_memory/Preference_Memory_step2_ACTION_singleturn_api5.sh` | `src/preference_memory/Preference_Memory_step2_ACTION_singleturn_api5.sh` | copied |
| `ours_memory/Preference_Memory_step2_ACTION_singleturn_api5_extended_hard.sh` | `src/preference_memory/Preference_Memory_step2_ACTION_singleturn_api5_extended_hard.sh` | copied |
| `ours_memory/analyze_multiturn_query_run.py` | `src/preference_memory/analyze_multiturn_query_run.py` | copied |
| `ours_memory/compare_iteration_caps.py` | `src/preference_memory/compare_iteration_caps.py` | copied |
| `ours_memory/deepseek_format.py` | `src/preference_memory/deepseek_format.py` | copied |
| `ours_memory/inference/1231_MEMORY3/ablation_verifier.py` | `src/preference_memory/result_tools/memory3_ablation_verifier.py` | copied |
| `ours_memory/inference/1231_MEMORY3/calculate_tokens.py` | `src/preference_memory/result_tools/memory3_calculate_tokens.py` | copied |
| `ours_memory/inference/extended/DEPRE_rerun/run_multiturn_rerun.sh` | `src/preference_memory/result_tools/extended_rerun/run_multiturn_rerun.sh` | copied |
| `ours_memory/inference/extended/DEPRE_rerun/run_rerun_job.py` | `src/preference_memory/result_tools/extended_rerun/run_rerun_job.py` | copied |
| `ours_memory/inference/extended/DEPRE_rerun/run_singleturn_rerun.sh` | `src/preference_memory/result_tools/extended_rerun/run_singleturn_rerun.sh` | copied |
| `ours_memory/plot_implicit_memory_token_growth.py` | `src/preference_memory/plot_implicit_memory_token_growth.py` | copied |
| `ours_memory/plot_performance_gap.py` | `src/preference_memory/plot_performance_gap.py` | copied |
| `ours_memory/plot_refinement_tail.py` | `src/preference_memory/plot_refinement_tail.py` | copied |
| `ours_memory/plot_retry_sweep.py` | `src/preference_memory/plot_retry_sweep.py` | copied |
| `ours_memory/prompt_inference.py` | `src/preference_memory/prompt_inference.py` | copied |
| `ours_memory/prompt_update0.py` | `src/preference_memory/prompt_update0.py` | copied |
| `ours_memory/prompt_update2.py` | `src/preference_memory/prompt_update2.py` | copied |
| `ours_memory/prompt_update3.py` | `src/preference_memory/prompt_update3.py` | copied |
| `ours_memory/run_extended_api_gpt5_models_nohup.sh` | `src/preference_memory/run_extended_api_gpt5_models_nohup.sh` | copied |
| `ours_memory/run_extended_multiturn_api_gpt4omini_nohup.sh` | `src/preference_memory/run_extended_multiturn_api_gpt4omini_nohup.sh` | copied |
| `ours_memory/run_memory.sh` | `src/preference_memory/run_memory.sh` | copied |
| `ours_memory/session_memory_eval_0312/analyze_session_memory_run.py` | `src/preference_memory/session_memory_eval/analyze_session_memory_run.py` | copied |
| `ours_memory/session_memory_eval_0312/evaluate_extended_multiturn_outputs.py` | `src/preference_memory/session_memory_eval/evaluate_extended_multiturn_outputs.py` | copied |
| `ours_memory/session_memory_eval_0312/evaluate_extended_singleturn_outputs.py` | `src/preference_memory/session_memory_eval/evaluate_extended_singleturn_outputs.py` | copied |
| `ours_memory/session_memory_eval_0312/evaluate_session_memory_results.py` | `src/preference_memory/session_memory_eval/evaluate_session_memory_results.py` | copied |
| `ours_memory/session_memory_eval_0312/evaluate_session_memory_results_multiturn.py` | `src/preference_memory/session_memory_eval/evaluate_session_memory_results_multiturn.py` | copied |
| `ours_memory/session_memory_eval_0312/evaluate_session_memory_results_singleturn.py` | `src/preference_memory/session_memory_eval/evaluate_session_memory_results_singleturn.py` | copied |
| `ours_memory/session_memory_eval_0312/generate_api_only_results_bundle.py` | `src/preference_memory/session_memory_eval/generate_api_only_results_bundle.py` | copied |
| `ours_memory/session_memory_eval_0312/generate_filtered_analysis_from_existing.py` | `src/preference_memory/session_memory_eval/generate_filtered_analysis_from_existing.py` | copied |
| `ours_memory/session_memory_eval_0312/generate_inference_results_bundle.py` | `src/preference_memory/session_memory_eval/generate_inference_results_bundle.py` | copied |
| `ours_memory/session_memory_eval_0312/generate_memory_model_views_from_summary.py` | `src/preference_memory/session_memory_eval/generate_memory_model_views_from_summary.py` | copied |
| `ours_memory/session_memory_eval_0312/generate_singleturn_hard_story_plots.py` | `src/preference_memory/session_memory_eval/generate_singleturn_hard_story_plots.py` | copied |
| `ours_memory/session_memory_eval_0312/plot_session_memory_curves.py` | `src/preference_memory/session_memory_eval/plot_session_memory_curves.py` | copied |
| `ours_memory/session_memory_eval_0312/plot_session_memory_curves_multiturn.py` | `src/preference_memory/session_memory_eval/plot_session_memory_curves_multiturn.py` | copied |
| `ours_memory/session_memory_eval_0312/plot_session_memory_curves_singleturn.py` | `src/preference_memory/session_memory_eval/plot_session_memory_curves_singleturn.py` | copied |
| `ours_memory/session_memory_eval_0312/prepare_session_memory_eval.py` | `src/preference_memory/session_memory_eval/prepare_session_memory_eval.py` | copied |
| `ours_memory/session_memory_eval_0312/run_full_pipeline.sh` | `src/preference_memory/session_memory_eval/run_full_pipeline.sh` | copied |
| `ours_memory/session_memory_eval_0312/run_session_memory_eval.sh` | `src/preference_memory/session_memory_eval/run_session_memory_eval.sh` | copied |
| `ours_memory/session_memory_eval_0312/run_session_memory_eval_multiturn.sh` | `src/preference_memory/session_memory_eval/run_session_memory_eval_multiturn.sh` | copied |
| `ours_memory/session_memory_eval_0312/run_session_memory_eval_singleturn.sh` | `src/preference_memory/session_memory_eval/run_session_memory_eval_singleturn.sh` | copied |
| `ours_memory/session_memory_eval_0312/session_memory_eval_common.py` | `src/preference_memory/session_memory_eval/session_memory_eval_common.py` | copied |
| `ours_memory/step1_generate_base-memory.py` | `src/preference_memory/build_base_memory.py` | copied; renamed to avoid the private `e-mem` path substring in the public tree |
| `ours_memory/step2_inference_base-memory_singleturn.py` | `src/preference_memory/run_base_memory_singleturn.py` | copied; renamed to avoid the private `e-mem` path substring in the public tree |
| `rebuttal_token_analysis_combined.py` | `src/analysis/rebuttal_token_analysis_combined.py` | copied |
| `rebuttal_token_analysis_combined_new.py` | `src/analysis/rebuttal_token_analysis_combined_new.py` | copied |
| `rebuttal_token_analysis_per-model.py` | `src/analysis/rebuttal_token_analysis_per-model.py` | copied |
| `rebuttal_token_cost_accounting.py` | `src/analysis/rebuttal_token_cost_accounting.py` | copied |
| `self-refine/self_refine_preference_to_api.py` | `src/self_refine/self_refine_preference_to_api.py` | copied |
| `vanillaLLM/DEPRE-vanillaLRM_inference-vllm-multi.py` | `src/baselines/vanilla_llm/DEPRE-vanillaLRM_inference-vllm-multi.py` | copied |
| `vanillaLLM/prompt.py` | `src/baselines/vanilla_llm/prompt.py` | copied |
| `vanillaLLM/run_vanillaLLM-api-multi-extended-400.sh` | `src/baselines/vanilla_llm/run_vanillaLLM-api-multi-extended-400.sh` | copied |
| `vanillaLLM/run_vanillaLLM-api-multi.sh` | `src/baselines/vanilla_llm/run_vanillaLLM-api-multi.sh` | copied |
| `vanillaLLM/run_vanillaLLM-api-single-extended-400.sh` | `src/baselines/vanilla_llm/run_vanillaLLM-api-single-extended-400.sh` | copied |
| `vanillaLLM/run_vanillaLLM-api-single.sh` | `src/baselines/vanilla_llm/run_vanillaLLM-api-single.sh` | copied |
| `vanillaLLM/run_vanillaLLM-vllm-multi-extended-400.sh` | `src/baselines/vanilla_llm/run_vanillaLLM-vllm-multi-extended-400.sh` | copied |
| `vanillaLLM/run_vanillaLLM-vllm-multi.sh` | `src/baselines/vanilla_llm/run_vanillaLLM-vllm-multi.sh` | copied |
| `vanillaLLM/run_vanillaLLM-vllm-single-extended-400.sh` | `src/baselines/vanilla_llm/run_vanillaLLM-vllm-single-extended-400.sh` | copied |
| `vanillaLLM/run_vanillaLLM-vllm-single.sh` | `src/baselines/vanilla_llm/run_vanillaLLM-vllm-single.sh` | copied |
| `vanillaLLM/run_vanillaLRM-vllm-multi.sh` | `src/baselines/vanilla_llm/run_vanillaLRM-vllm-multi.sh` | copied |
| `vanillaLLM/run_vanillaLRM-vllm-single.sh` | `src/baselines/vanilla_llm/run_vanillaLRM-vllm-single.sh` | copied |
| `vanillaLLM/vanillaLLM_inference-api-multi.py` | `src/baselines/vanilla_llm/vanillaLLM_inference-api-multi.py` | copied |
| `vanillaLLM/vanillaLLM_inference-api-single.py` | `src/baselines/vanilla_llm/vanillaLLM_inference-api-single.py` | copied |
| `vanillaLLM/vanillaLLM_inference-vllm-multi.py` | `src/baselines/vanilla_llm/vanillaLLM_inference-vllm-multi.py` | copied |
| `vanillaLLM/vanillaLLM_inference-vllm-single.py` | `src/baselines/vanilla_llm/vanillaLLM_inference-vllm-single.py` | copied |
| `vanillaLLM/vanillaLRM_inference-vllm-multi.py` | `src/baselines/vanilla_llm/vanillaLRM_inference-vllm-multi.py` | copied |
| `vanillaLLM/vanillaLRM_inference-vllm-single.py` | `src/baselines/vanilla_llm/vanillaLRM_inference-vllm-single.py` | copied |

## Copied Data And Artifacts

Representative small configs, datasets, tables, and plots were copied when they support setup, reproduction, or paper documentation. Large generated inference outputs and logs were not copied.

| Source | Release path | Category |
| --- | --- | --- |
| `RAG/RAG_multiturn.csv` | `artifacts/rag/RAG_multiturn.csv` | artifact |
| `RAG/RAG_multiturn2.csv` | `artifacts/rag/RAG_multiturn2.csv` | artifact |
| `RAG/RAG_singleturn.csv` | `artifacts/rag/RAG_singleturn.csv` | artifact |
| `_paper/ablation/prefine_preference_shift_conflict_rebuttal.md` | `artifacts/paper_tables/ablation/prefine_preference_shift_conflict_rebuttal.md` | artifact |
| `_paper/appendix_table3_combined_with_sources.csv` | `artifacts/paper_tables/appendix_table3_combined_with_sources.csv` | artifact |
| `_paper/table11_context_guided_base_prefine_gemma_gpt4o.csv` | `artifacts/paper_tables/table11_context_guided_base_prefine_gemma_gpt4o.csv` | artifact |
| `_paper/table12_context_guided_prefine_reasoning.csv` | `artifacts/paper_tables/table12_context_guided_prefine_reasoning.csv` | artifact |
| `_paper/table13_context_free_prefine.csv` | `artifacts/paper_tables/table13_context_free_prefine.csv` | artifact |
| `_paper/table3_recomputed_from_appendix.csv` | `artifacts/paper_tables/table3_recomputed_from_appendix.csv` | artifact |
| `_paper/table3_recomputed_from_appendix.md` | `artifacts/paper_tables/table3_recomputed_from_appendix.md` | artifact |
| `data/1229_dev_6.json` | `data/1229_dev_6.json` | data |
| `data/1229_dev_6_SAMPLED_100.json` | `data/1229_dev_6_SAMPLED_100.json` | data |
| `data/1229_query_all.json` | `data/1229_query_all.json` | data |
| `data/DEPRE-query_multiturn.json` | `data/DEPRE-query_multiturn.json` | data |
| `data/dev_3.json` | `data/dev_3.json` | data |
| `data/dev_3_api_calls_pref_value_groups.tsv` | `data/dev_3_api_calls_pref_value_groups.tsv` | data |
| `data/dev_4.json` | `data/dev_4.json` | data |
| `data/dev_5.json` | `data/dev_5.json` | data |
| `data/domain_slot_values.csv` | `data/domain_slot_values.csv` | data |
| `data/fixed_multiturn_example_query_pairs_400.json` | `data/fixed_multiturn_example_query_pairs_400.json` | data |
| `data/fixed_singleturn_example_query_pairs_400.json` | `data/fixed_singleturn_example_query_pairs_400.json` | data |
| `data/high_regroup_dev_6.json` | `data/high_regroup_dev_6.json` | data |
| `data/query_easy.json` | `data/query_easy.json` | data |
| `data/query_medium.json` | `data/query_medium.json` | data |
| `data/re_dev_6.json` | `data/re_dev_6.json` | data |
| `data/sampled_dataset_100.json` | `data/sampled_dataset_100.json` | data |
| `data/sgd_converted_dev_mapped_grouped_with_pref_with_constraints.json` | `data/sgd_converted_dev_mapped_grouped_with_pref_with_constraints.json` | data |
| `extended_schema/README.md` | `src/extended_schema/README.md` | artifact |
| `extended_schema/langmem/multiturn_comparison.csv` | `artifacts/extended_schema/langmem/multiturn_comparison.csv` | artifact |
| `extended_schema/langmem/singleturn_comparison.csv` | `artifacts/extended_schema/langmem/singleturn_comparison.csv` | artifact |
| `extended_schema/multiturn_comparison.csv` | `artifacts/extended_schema/multiturn_comparison.csv` | artifact |
| `extended_schema/singleturn_comparison.csv` | `artifacts/extended_schema/singleturn_comparison.csv` | artifact |
| `langmem/README.md` | `src/baselines/langmem/README.md` | artifact |
| `langmem/requirements.txt` | `src/baselines/langmem/requirements.txt` | artifact |
| `mem0/0323_memory-tokens-265/latest_session_memory_token_deltas_1229_dev_6.txt` | `artifacts/mem0/session_memory_tokens/265_examples/latest_session_memory_token_deltas_1229_dev_6.txt` | artifact |
| `mem0/0323_memory-tokens-265/mem0_265_examples.merged.csv` | `artifacts/mem0/session_memory_tokens/265_examples/mem0_265_examples.merged.csv` | artifact |
| `mem0/0323_memory-tokens-265/mem0_265_examples.merged.summary.json` | `artifacts/mem0/session_memory_tokens/265_examples/mem0_265_examples.merged.summary.json` | artifact |
| `mem0/0323_memory-tokens-265/mem0_265_examples_token_growth.png` | `artifacts/mem0/session_memory_tokens/265_examples/mem0_265_examples_token_growth.png` | artifact |
| `mem0/0323_memory-tokens-265/mem0_265_examples_token_growth.summary.csv` | `artifacts/mem0/session_memory_tokens/265_examples/mem0_265_examples_token_growth.summary.csv` | artifact |
| `mem0/0323_memory-tokens_111/latest_session_memory_token_deltas_1229_dev_6.txt` | `artifacts/mem0/session_memory_tokens/111_examples/latest_session_memory_token_deltas_1229_dev_6.txt` | artifact |
| `mem0/0323_memory-tokens_111/mem0_111_examples.merged.csv` | `artifacts/mem0/session_memory_tokens/111_examples/mem0_111_examples.merged.csv` | artifact |
| `mem0/0323_memory-tokens_111/mem0_111_examples.merged.summary.json` | `artifacts/mem0/session_memory_tokens/111_examples/mem0_111_examples.merged.summary.json` | artifact |
| `mem0/0323_memory-tokens_111/mem0_111_examples_token_growth.png` | `artifacts/mem0/session_memory_tokens/111_examples/mem0_111_examples_token_growth.png` | artifact |
| `mem0/0323_memory-tokens_111/mem0_111_examples_token_growth.summary.csv` | `artifacts/mem0/session_memory_tokens/111_examples/mem0_111_examples_token_growth.summary.csv` | artifact |
| `mem0/0323_memory-tokens_111/mem0_session_memory_token_growth.png` | `artifacts/mem0/session_memory_tokens/111_examples/mem0_session_memory_token_growth.png` | artifact |
| `mem0/0323_memory-tokens_111/mem0_session_memory_token_growth.summary.csv` | `artifacts/mem0/session_memory_tokens/111_examples/mem0_session_memory_token_growth.summary.csv` | artifact |
| `memory-dev/.gitignore` | `src/dataset_tools/.gitignore` | data |
| `memory-dev/LICENSE` | `src/dataset_tools/LICENSE` | data |
| `memory-dev/README.md` | `src/dataset_tools/README.md` | data |
| `memory-dev/datasets/sgd2ours/.gitignore` | `src/dataset_tools/datasets/sgd2ours/.gitignore` | data |
| `memory-dev/datasets/sgd2ours/common-domain-slots.tsv` | `src/dataset_tools/datasets/sgd2ours/common-domain-slots.tsv` | data |
| `memory-dev/datasets/sgd2ours/common_pref.csv` | `src/dataset_tools/datasets/sgd2ours/common_pref.csv` | data |
| `memory-dev/datasets/sgd2ours/query/dev_5_domain_slot_values.csv` | `src/dataset_tools/datasets/sgd2ours/query/dev_5_domain_slot_values.csv` | data |
| `memory-dev/datasets/sgd2ours/query/pref-slot-list.json` | `src/dataset_tools/datasets/sgd2ours/query/pref-slot-list.json` | data |
| `memory-dev/datasets/sgd2ours/query/query_all.json` | `src/dataset_tools/datasets/sgd2ours/query/query_all.json` | data |
| `memory-dev/datasets/sgd2ours/query/query_group.json` | `src/dataset_tools/datasets/sgd2ours/query/query_group.json` | data |
| `memory-dev/datasets/sgd2ours/query/query_slot.json` | `src/dataset_tools/datasets/sgd2ours/query/query_slot.json` | data |
| `memory-dev/methods/ReMem/.gitignore` | `src/dataset_tools/methods/ReMem/.gitignore` | data |
| `memory-dev/methods/ReMem/datasets/.gitkeep` | `src/dataset_tools/methods/ReMem/datasets/.gitkeep` | data |
| `memory-dev/methods/ReMem/domain_queries.json` | `src/dataset_tools/methods/ReMem/domain_queries.json` | data |
| `memory-dev/methods/ReMem/evaluate/pref_group.json` | `src/dataset_tools/methods/ReMem/evaluate/pref_group.json` | data |
| `memory-dev/methods/ReMem/evaluate/pref_list.json` | `src/dataset_tools/methods/ReMem/evaluate/pref_list.json` | data |
| `memory-dev/methods/ReMem/remem.prompt` | `src/dataset_tools/methods/ReMem/remem.prompt` | data |
| `memory-dev/methods/valid/.gitignore` | `src/dataset_tools/methods/valid/.gitignore` | data |
| `ours_memory/iteration_cap_comparison/0216_MEMORY_retry_validity.csv` | `artifacts/preference_memory/iteration_cap_comparison/0216_MEMORY_retry_validity.csv` | artifact |
| `ours_memory/iteration_cap_comparison/0216_MEMORY_retry_validity.png` | `artifacts/preference_memory/iteration_cap_comparison/0216_MEMORY_retry_validity.png` | artifact |
| `ours_memory/iteration_cap_comparison/0312_MEMORY1_max10_first_valid_steps.csv` | `artifacts/preference_memory/iteration_cap_comparison/0312_MEMORY1_max10_first_valid_steps.csv` | artifact |
| `ours_memory/iteration_cap_comparison/0312_MEMORY1_max10_last_step_distribution.csv` | `artifacts/preference_memory/iteration_cap_comparison/0312_MEMORY1_max10_last_step_distribution.csv` | artifact |
| `ours_memory/iteration_cap_comparison/0312_MEMORY1_max10_refinement_tail.png` | `artifacts/preference_memory/iteration_cap_comparison/0312_MEMORY1_max10_refinement_tail.png` | artifact |
| `ours_memory/iteration_cap_comparison/0312_MEMORY1_max10_step_bucket_summary.csv` | `artifacts/preference_memory/iteration_cap_comparison/0312_MEMORY1_max10_step_bucket_summary.csv` | artifact |
| `ours_memory/iteration_cap_comparison/0312_MEMORY1_max10_step_summary.csv` | `artifacts/preference_memory/iteration_cap_comparison/0312_MEMORY1_max10_step_summary.csv` | artifact |
| `ours_memory/iteration_cap_comparison/0312_vs_1231_performance_gap.png` | `artifacts/preference_memory/iteration_cap_comparison/0312_vs_1231_performance_gap.png` | artifact |
| `ours_memory/iteration_cap_comparison/1231_MEMORY3_max3_cumulative_validity.png` | `artifacts/preference_memory/iteration_cap_comparison/1231_MEMORY3_max3_cumulative_validity.png` | artifact |
| `ours_memory/iteration_cap_comparison/1231_MEMORY3_max3_first_valid_steps.csv` | `artifacts/preference_memory/iteration_cap_comparison/1231_MEMORY3_max3_first_valid_steps.csv` | artifact |
| `ours_memory/iteration_cap_comparison/1231_MEMORY3_max3_last_step_distribution.csv` | `artifacts/preference_memory/iteration_cap_comparison/1231_MEMORY3_max3_last_step_distribution.csv` | artifact |
| `ours_memory/iteration_cap_comparison/1231_MEMORY3_max3_step_bucket_summary.csv` | `artifacts/preference_memory/iteration_cap_comparison/1231_MEMORY3_max3_step_bucket_summary.csv` | artifact |
| `ours_memory/iteration_cap_comparison/1231_MEMORY3_max3_step_summary.csv` | `artifacts/preference_memory/iteration_cap_comparison/1231_MEMORY3_max3_step_summary.csv` | artifact |
| `ours_memory/iteration_cap_comparison/multiturn_by_pref.csv` | `artifacts/preference_memory/iteration_cap_comparison/multiturn_by_pref.csv` | artifact |
| `ours_memory/iteration_cap_comparison/multiturn_common_subset.csv` | `artifacts/preference_memory/iteration_cap_comparison/multiturn_common_subset.csv` | artifact |
| `ours_memory/iteration_cap_comparison/performance_gap_delta_stats.csv` | `artifacts/preference_memory/iteration_cap_comparison/performance_gap_delta_stats.csv` | artifact |
| `ours_memory/iteration_cap_comparison/performance_gap_summary.csv` | `artifacts/preference_memory/iteration_cap_comparison/performance_gap_summary.csv` | artifact |
| `ours_memory/iteration_cap_comparison/refinement_by_model.csv` | `artifacts/preference_memory/iteration_cap_comparison/refinement_by_model.csv` | artifact |
| `ours_memory/iteration_cap_comparison/refinement_totals.csv` | `artifacts/preference_memory/iteration_cap_comparison/refinement_totals.csv` | artifact |
| `ours_memory/iteration_cap_comparison/singleturn_by_pref.csv` | `artifacts/preference_memory/iteration_cap_comparison/singleturn_by_pref.csv` | artifact |
| `ours_memory/iteration_cap_comparison/singleturn_common_subset.csv` | `artifacts/preference_memory/iteration_cap_comparison/singleturn_common_subset.csv` | artifact |
| `ours_memory/iteration_cap_comparison/summary.md` | `artifacts/preference_memory/iteration_cap_comparison/summary.md` | artifact |
| `ours_memory/memory3_easy_gpt4omini.json` | `artifacts/preference_memory/bootstrap_memories/memory3_easy_gpt4omini.json` | artifact |
| `ours_memory/memory3_hard_gpt4omini.json` | `artifacts/preference_memory/bootstrap_memories/memory3_hard_gpt4omini.json` | artifact |
| `ours_memory/memory3_medium_gpt4omini.json` | `artifacts/preference_memory/bootstrap_memories/memory3_medium_gpt4omini.json` | artifact |
| `ours_memory/ours_implicit_pref_token_flow_by_dialogue_non_deprecated.csv` | `src/preference_memory/ours_implicit_pref_token_flow_by_dialogue_non_deprecated.csv` | artifact |
| `ours_memory/session_memory_eval_0312/README.md` | `src/preference_memory/session_memory_eval/README.md` | artifact |
| `pref_group.json` | `configs/pref_group.json` | config |
| `pref_group_extended.json` | `configs/pref_group_extended.json` | config |
| `pref_list.json` | `configs/pref_list.json` | config |
| `pref_list_extended.json` | `configs/pref_list_extended.json` | config |
| `query_multiturn-domain.json` | `configs/query_multiturn-domain.json` | config |
| `query_multiturn.json` | `configs/query_multiturn.json` | config |
| `query_multiturn_extended.json` | `configs/query_multiturn_extended.json` | config |
| `query_new_multi.json` | `configs/query_new_multi.json` | config |
| `query_new_single.json` | `configs/query_new_single.json` | config |
| `query_singleturn.json` | `configs/query_singleturn.json` | config |
| `query_singleturn_extended.json` | `configs/query_singleturn_extended.json` | config |
| `rebuttal_ours_memory_no_reasoning_token_cost.csv` | `artifacts/token_cost/rebuttal_ours_memory_no_reasoning_token_cost.csv` | artifact |
| `rebuttal_token_cost_construction_detail.csv` | `artifacts/token_cost/rebuttal_token_cost_construction_detail.csv` | artifact |
| `rebuttal_token_cost_summary.csv` | `artifacts/token_cost/rebuttal_token_cost_summary.csv` | artifact |
| `rebuttal_token_cost_summary.md` | `artifacts/token_cost/rebuttal_token_cost_summary.md` | artifact |
| `rebuttal_token_evolution_per-model.png` | `artifacts/token_cost/rebuttal_token_evolution_per-model.png` | artifact |
| `rebuttal_token_growth_comparison_combined.png` | `artifacts/token_cost/rebuttal_token_growth_comparison_combined.png` | artifact |
| `requirements.txt` | `docs/original_requirements_from_experiments4.txt` | artifact |
| `schema_all.json` | `configs/schema_all.json` | config |
| `schema_all_extended.json` | `configs/schema_all_extended.json` | config |
| `schema_all_extended_complete.json` | `configs/schema_all_extended_complete.json` | config |
| `schema_easy.json` | `configs/schema_easy.json` | config |
| `schema_easy_extended.json` | `configs/schema_easy_extended.json` | config |
| `schema_medium.json` | `configs/schema_medium.json` | config |
| `token_evolution_plot.png` | `artifacts/token_cost/token_evolution_plot.png` | artifact |
| `token_growth_comparison_combined0.png` | `artifacts/token_cost/token_growth_comparison_combined0.png` | artifact |
| `token_growth_comparison_combined1.png` | `artifacts/token_cost/token_growth_comparison_combined1.png` | artifact |
| `token_growth_comparison_combined_new.png` | `artifacts/token_cost/token_growth_comparison_combined_new.png` | artifact |
