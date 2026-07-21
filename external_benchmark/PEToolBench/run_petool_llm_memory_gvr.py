"""PEToolBench-local LLM-memory GVR runner.

This runner keeps the experiment inside PEToolBench and does not import or
modify experiments5/methods/our_memory. Unlike run_petool_memory_gvr.py, the
memory generation step is an explicit LLM prompt. The verifier/refiner uses
only PEToolBench interaction history, never baseline model outputs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from run_petool_memory_gvr import (
    HISTORY_KEYS,
    choose_memory_tool,
    combine_summaries,
    evaluate,
    first_json_object,
    generate_memory,
    normalize_rows,
    operation_family,
    parse_tool_name,
    read_json,
    semantic_score,
    verify_and_refine,
    write_json,
)


MEMORY_GENERATOR_SYSTEM = """You are the memory generation step of a Generate-Verify-Refine preference-following pipeline.
Extract durable user API preferences from the provided PEToolBench interaction history.
Return JSON only. Do not answer the final user request and do not choose a tool from the candidate list.

Preference rules:
- preferred history: previous tool calls are positive evidence.
- ratings history: rating 1 means prefer that provider namespace; rating 0 means avoid it.
- chronological history: later tool calls are stronger evidence than earlier tool calls.
- Prefer provider namespaces such as <Category>.<Provider>, not only individual operation names.
- Keep operation-family hints when the same provider is preferred for similar operations.
"""

MEMORY_SCHEMA = {
    "preferred_namespaces": [
        {
            "namespace": "<Category>.<Provider>",
            "score": "number",
            "operation_families": ["short family labels"],
            "evidence_indices": ["history turn indices"],
            "reason": "short reason",
        }
    ],
    "avoid_namespaces": [
        {
            "namespace": "<Category>.<Provider>",
            "score": "number",
            "operation_families": ["short family labels"],
            "evidence_indices": ["history turn indices"],
            "reason": "short reason",
        }
    ],
    "chronological_rule": "short rule for later-over-earlier evidence",
}

FINAL_INFERENCE_SYSTEM = """Your task is to generate one API tool call that satisfies the current user instruction and follows the user's preferences.
You are given:
1. Preference memory generated from the user's interaction history by a previous GVR memory step.
2. The original interaction API history.
3. The available candidate tools.

Use the memory as a preference guide:
- First match the current request to the correct operation.
- If multiple candidate tools satisfy the operation, choose the provider namespace preferred by memory.
- Avoid namespaces marked negative by memory.
- Do not choose a remembered provider if its operation does not satisfy the current request.
- Output only a single JSON object with tool_name and parameters.
"""

FINAL_INFERENCE_STRONG = """The verified memory includes a memory_matched_candidate_tool.
If that tool is present in the candidate tools and satisfies the current request's operation, use that exact tool_name.
Only reject it when its operation clearly does not answer the current request.
"""


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def build_memory_generation_prompt(record: Dict[str, Any]) -> str:
    history_type = record["history_type"]
    history = record.get("history", [])
    if history_type == "p":
        semantics = "Every historical tool call is positive preference evidence."
    elif history_type == "r":
        semantics = "Use rating=1 as positive preference evidence and rating=0 as negative avoid evidence."
    else:
        semantics = "The history is chronological; later turns are stronger preference evidence than earlier turns."
    return "\n".join(
        [
            f"History type: {history_type}",
            f"Semantics: {semantics}",
            "Interaction history JSON:",
            compact_json(history),
            "Return JSON with this schema:",
            compact_json(MEMORY_SCHEMA),
        ]
    )


def summarize_llm_memory(memory: Dict[str, Any], max_items: int = 8) -> str:
    if not memory:
        return "LLM memory was empty or unparsable."
    parts = []
    for key in ["preferred_namespaces", "avoid_namespaces"]:
        values = memory.get(key, [])
        if not isinstance(values, list):
            continue
        rendered = []
        for item in values[:max_items]:
            if isinstance(item, dict):
                namespace = item.get("namespace", "")
                reason = item.get("reason", "")
                families = item.get("operation_families", [])
                rendered.append(
                    {
                        "namespace": namespace,
                        "operation_families": families,
                        "reason": reason,
                    }
                )
        if rendered:
            parts.append(f"{key}: {compact_json(rendered)}")
    if memory.get("chronological_rule"):
        parts.append(f"chronological_rule: {memory.get('chronological_rule')}")
    return "\n".join(parts) if parts else compact_json(memory)


def render_verified_memory(
    record: Dict[str, Any],
    verified_memory: Dict[str, Any],
    args: argparse.Namespace,
) -> str:
    provider_preferences = verified_memory.get("provider_preferences", {})
    operation_preferences = verified_memory.get("operation_preferences", {})
    memory_choice = choose_memory_tool(record, verified_memory)
    query_family = memory_choice.get("_operation_family") or operation_family(record["query"])

    provider_rows = []
    for namespace, stats in sorted(
        provider_preferences.items(),
        key=lambda item: (float(item[1].get("score", 0.0) or 0.0), int(item[1].get("latest", -1) or -1)),
        reverse=True,
    )[:8]:
        provider_rows.append(
            {
                "namespace": namespace,
                "score": stats.get("score", 0.0),
                "positive": stats.get("positive", 0),
                "negative": stats.get("negative", 0),
                "latest": stats.get("latest", -1),
                "families": stats.get("operation_families", []),
            }
        )

    focused_preferences = operation_preferences.get(query_family, {})
    allow_override = record["history_type"] in set(args.memory_override_history_types)
    focused = {
        "current_request_operation_family": query_family,
        "memory_override_allowed": allow_override,
        "memory_matched_candidate_tool": memory_choice.get("tool_name", "") if allow_override else "",
        "memory_matched_namespace": memory_choice.get("_namespace", "") if allow_override else "",
        "preferred_namespaces_for_family": focused_preferences.get("preferred_namespaces", []),
        "avoid_namespaces_for_family": focused_preferences.get("avoid_namespaces", []),
        "provider_preference_ranking": provider_rows,
    }
    return compact_json(focused)


def build_memory_content(
    record: Dict[str, Any],
    llm_memory: Dict[str, Any],
    verified_memory: Dict[str, Any],
    args: argparse.Namespace,
) -> str:
    if args.memory_content_style == "recommendation":
        memory_choice = choose_memory_tool(record, verified_memory)
        allow_override = record["history_type"] in set(args.memory_override_history_types)
        recommendation = {
            "override_allowed": allow_override,
            "recommended_tool_name": memory_choice.get("tool_name", "") if allow_override else "",
            "recommended_namespace": memory_choice.get("_namespace", "") if allow_override else "",
            "operation_family": memory_choice.get("_operation_family", ""),
            "instruction": (
                "Use the recommended_tool_name exactly when override_allowed is true "
                "and it matches the current request. Infer parameters from the current user instruction."
            ),
        }
        return "\n".join(
            [
                "GVR verified recommendation:",
                compact_json(recommendation),
                "LLM-generated memory summary:",
                summarize_llm_memory(llm_memory, max_items=4),
            ]
        )
    return "\n".join(
        [
            "LLM-generated memory:",
            summarize_llm_memory(llm_memory),
            "Verified/refined memory from the same interaction history:",
            render_verified_memory(record, verified_memory, args),
        ]
    )


def choose_latest_namespace_tool(record: Dict[str, Any]) -> Dict[str, Any]:
    """Choose the current-operation tool from the most recent usable namespace."""
    recent_namespaces = []
    for turn in reversed(record.get("history", []) or []):
        call = turn.get("tool_call", {}) or {}
        namespace = parse_tool_name(str(call.get("tool_name", ""))).get("namespace", "")
        if namespace and namespace not in recent_namespaces:
            recent_namespaces.append(namespace)

    for namespace in recent_namespaces:
        candidates = []
        for index, tool in enumerate(record.get("candidate_tools", []) or []):
            parsed = parse_tool_name(str(tool.get("tool_name", "")))
            if parsed["namespace"] == namespace:
                candidates.append(
                    (
                        semantic_score(record["query"], tool),
                        -index,
                        tool,
                        parsed,
                    )
                )
        if candidates:
            score, _, tool, parsed = sorted(candidates, reverse=True)[0]
            return {
                "tool_name": tool.get("tool_name", ""),
                "_score": score,
                "_namespace": parsed["namespace"],
                "_operation_family": operation_family(
                    parsed["operation"],
                    str(tool.get("tool_description", "")),
                ),
            }
    return {"tool_name": ""}


def refine_with_policy(
    record: Dict[str, Any],
    generated: Dict[str, Any],
    memory: Dict[str, Any],
    policy: str,
) -> Dict[str, Any]:
    if policy != "p_r_memory_c_latest_namespace":
        return verify_and_refine(record, generated, memory, policy)
    if record["history_type"] in {"p", "r"}:
        return verify_and_refine(record, generated, memory, "p_r_memory_c_generator")

    generated_tool = generated.get("tool_name", "") if isinstance(generated, dict) else ""
    generated_params = generated.get("parameters", {}) if isinstance(generated, dict) else {}
    memory_choice = choose_latest_namespace_tool(record)
    final_tool_name = memory_choice.get("tool_name") or generated_tool
    decision = (
        "chronological_latest_namespace_refine"
        if memory_choice.get("tool_name")
        else "chronological_generator_kept"
    )
    return {
        "tool_name": final_tool_name,
        "parameters": generated_params if isinstance(generated_params, dict) else {},
        "_verifier": {
            "policy": policy,
            "decision": decision,
            "generated_tool_name": generated_tool,
            "memory_tool_name": final_tool_name,
            "memory_score": memory_choice.get("_score"),
            "memory_operation_family": memory_choice.get("_operation_family"),
            "memory_namespace": memory_choice.get("_namespace"),
            "verify_refine_prompt": (
                "For chronological histories, scan interaction API history from latest to "
                "earliest, choose the most recent namespace that has a candidate tool for "
                "the current request, and preserve generated parameters."
            ),
        },
    }


def build_final_inference_prompt(
    record: Dict[str, Any],
    memory_content: str,
    include_history: bool,
    args: argparse.Namespace,
) -> str:
    if args.prompt_layout == "original_memory":
        return "\n".join(
            [
                "Additional GVR preference memory generated from the interaction history:",
                memory_content,
                "",
                "Use the GVR memory together with the original PEToolBench instruction below.",
                "When memory_override_allowed is true and the recommendation matches the request, use the recommended tool_name.",
                "",
                "Original PEToolBench instruction:",
                record["system_prompt"],
                "",
                "Current user instruction:",
                record["query"],
                "",
                "Remember: output only the JSON tool call.",
            ]
        )
    history = record.get("history", []) if include_history else []
    tools = record.get("candidate_tools", [])
    history_description = {
        "p": "The history contains positive preference examples.",
        "r": "The history contains binary ratings; rating 1 is preferred and rating 0 should be avoided.",
        "c": "The history is chronological; later calls indicate stronger preferences than earlier calls.",
    }[record["history_type"]]
    sections = [
        FINAL_INFERENCE_SYSTEM,
        FINAL_INFERENCE_STRONG if args.inference_strength == "strong" else "",
        "",
        "Preference memory content:",
        memory_content,
    ]
    if include_history:
        sections.extend(
            [
                "",
                history_description,
                "Interaction API history:",
                compact_json(history),
            ]
        )
    sections.extend(
        [
            "",
            "Current user instruction:",
            record["query"],
            "",
            "Available tools you can call with input parameters are listed here:",
            compact_json(tools),
            "",
            'Generate your tool call in this exact JSON format: {"tool_name": "...", "parameters": {...}}',
            "Remember: output only JSON without additional text.",
        ]
    )
    return "\n".join(sections)


class VllmRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        from transformers import AutoTokenizer
        from vllm import LLM

        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        self.llm = LLM(
            model=args.model,
            dtype=args.dtype,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_input_tokens + max(args.memory_max_new_tokens, args.inference_max_new_tokens),
            trust_remote_code=True,
        )

    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int,
        temperature: float = 0.0,
        label: str = "generation",
    ) -> List[Dict[str, Any]]:
        from vllm import SamplingParams

        sampling = SamplingParams(temperature=temperature, max_tokens=max_new_tokens)
        outputs: List[Dict[str, Any]] = []
        for start in range(0, len(prompts), self.args.batch_size):
            batch = prompts[start : start + self.args.batch_size]
            encoded_prompts = []
            for prompt in batch:
                messages = [{"role": "user", "content": prompt}]
                text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                encoded = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.args.max_input_tokens,
                    add_special_tokens=False,
                )
                encoded_prompts.append(self.tokenizer.decode(encoded["input_ids"], skip_special_tokens=False))
            started = time.time()
            generations = self.llm.generate(encoded_prompts, sampling)
            elapsed = time.time() - started
            for generation in generations:
                text = generation.outputs[0].text.strip() if generation.outputs else ""
                outputs.append({"response": text, "elapsed_seconds": elapsed / max(len(batch), 1)})
            print(f"{label} generated {len(outputs)}/{len(prompts)}")
        return outputs


def load_baseline_summaries(
    cache_dir: Path,
    history_types: Iterable[str],
    limit: Optional[int],
) -> Dict[str, Dict[str, Any]]:
    summaries = {}
    for history_type in history_types:
        path = cache_dir / f"predictions_{history_type}.json"
        rows = read_json(path)
        if limit is not None:
            rows = rows[:limit]
        summaries[history_type] = evaluate(
            [
                {
                    "api_call_ground_truth": row.get("api_call_ground_truth", {}),
                    "parsed_response": row.get("parsed_response", {}),
                    "verifier": {"decision": "generator_baseline"},
                }
                for row in rows
            ]
        )
    return summaries


def load_existing_memory(memory_dir: Path, history_type: str) -> Optional[List[Dict[str, Any]]]:
    path = memory_dir / f"memory_{history_type}.json"
    if path.exists():
        return read_json(path)
    return None


def generate_memory_records(
    records: List[Dict[str, Any]],
    args: argparse.Namespace,
    runner: VllmRunner,
) -> List[Dict[str, Any]]:
    prompts = [build_memory_generation_prompt(record) for record in records]
    outputs = runner.generate(
        prompts,
        args.memory_max_new_tokens,
        args.memory_temperature,
        label="memory",
    )
    memory_records = []
    for record, output in zip(records, outputs):
        generated_memory = first_json_object(output["response"])
        verified_memory = generate_memory(record)
        memory_content = build_memory_content(record, generated_memory, verified_memory, args)
        memory_records.append(
            {
                "example_id": record["example_id"],
                "source_index": record["source_index"],
                "history_type": record["history_type"],
                "memory_generation_prompt": prompts[len(memory_records)],
                "memory_response": output["response"],
                "generated_memory": generated_memory,
                "verified_memory": verified_memory,
                "memory_content": memory_content,
                "memory_elapsed_seconds": output["elapsed_seconds"],
            }
        )
    return memory_records


def refresh_memory_records(
    records: List[Dict[str, Any]],
    memory_records: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    memory_by_id = {row["example_id"]: row for row in memory_records}
    refreshed = []
    for record in records:
        row = dict(memory_by_id[record["example_id"]])
        generated_memory = row.get("generated_memory", {})
        verified_memory = row.get("verified_memory") or generate_memory(record)
        row["verified_memory"] = verified_memory
        row["memory_content"] = build_memory_content(record, generated_memory, verified_memory, args)
        refreshed.append(row)
    return refreshed


def generate_final_predictions(
    records: List[Dict[str, Any]],
    memory_records: List[Dict[str, Any]],
    args: argparse.Namespace,
    runner: VllmRunner,
) -> List[Dict[str, Any]]:
    memory_by_id = {row["example_id"]: row for row in memory_records}
    prompts = [
        build_final_inference_prompt(
            record,
            memory_by_id[record["example_id"]]["memory_content"],
            include_history=args.include_history,
            args=args,
        )
        for record in records
    ]
    outputs = runner.generate(
        prompts,
        args.inference_max_new_tokens,
        args.inference_temperature,
        label="inference",
    )
    predictions = []
    for record, prompt, output in zip(records, prompts, outputs):
        memory_record = memory_by_id[record["example_id"]]
        parsed = first_json_object(output["response"])
        verifier = {"decision": "post_refine_disabled"}
        final_parsed = parsed
        if args.post_refine_policy != "none":
            refined = refine_with_policy(
                record,
                parsed,
                memory_record.get("verified_memory", {}),
                args.post_refine_policy,
            )
            verifier = refined.pop("_verifier")
            final_parsed = refined
        predictions.append(
            {
                "example_id": record["example_id"],
                "source_index": record["source_index"],
                "history_type": record["history_type"],
                "query": record["query"],
                "api_call_ground_truth": record["api_call_ground_truth"],
                "memory_content": memory_record["memory_content"],
                "memory_response": memory_record["memory_response"],
                "final_system_prompt": prompt,
                "response": output["response"],
                "generated_parsed_response": parsed,
                "parsed_response": final_parsed,
                "verifier": verifier,
                "elapsed_seconds": output["elapsed_seconds"],
            }
        )
    return predictions


def run(args: argparse.Namespace) -> None:
    output_dir = args.output_dir / args.run_name
    memory_dir = output_dir / "memory"
    output_dir.mkdir(parents=True, exist_ok=True)
    memory_dir.mkdir(parents=True, exist_ok=True)

    runner = VllmRunner(args)
    summaries: Dict[str, Dict[str, Any]] = {}
    for history_type in args.history_types:
        records = normalize_rows(args.dataset_dir, history_type, args.limit)
        memory_records = None if args.force_memory else load_existing_memory(memory_dir, history_type)
        if memory_records is None:
            print(f"generating memory {history_type} n={len(records)}")
            memory_records = generate_memory_records(records, args, runner)
            write_json(memory_dir / f"memory_{history_type}.json", memory_records)
        else:
            print(f"using cached memory {history_type} n={len(memory_records)}")
            memory_records = refresh_memory_records(records, memory_records, args)
        print(f"generating final inference {history_type} n={len(records)}")
        predictions = generate_final_predictions(records, memory_records, args, runner)
        summary = evaluate(predictions)
        summaries[history_type] = summary
        write_json(output_dir / f"predictions_{history_type}.json", predictions)
        write_json(output_dir / f"summary_{history_type}.json", summary)
        print(history_type, json.dumps(summary, ensure_ascii=False))

    baseline_summaries = {}
    if args.baseline_cache:
        baseline_summaries = load_baseline_summaries(args.baseline_cache, args.history_types, args.limit)
    aggregate = {
        "model": args.model,
        "method": "petool_llm_memory_generate_verify_refine_inference",
        "run_name": args.run_name,
        "limit": args.limit,
        "history_types": args.history_types,
        "include_history": args.include_history,
        "memory_override_history_types": args.memory_override_history_types,
        "inference_strength": args.inference_strength,
        "memory_content_style": args.memory_content_style,
        "post_refine_policy": args.post_refine_policy,
        "prompt_layout": args.prompt_layout,
        "memory_generator_system": MEMORY_GENERATOR_SYSTEM,
        "memory_schema": MEMORY_SCHEMA,
        "final_inference_system": FINAL_INFERENCE_SYSTEM,
        "baseline_cache": str(args.baseline_cache) if args.baseline_cache else None,
        "baseline_summaries_from_generator_cache": baseline_summaries,
        "baseline_overall_from_generator_cache": combine_summaries(baseline_summaries)
        if baseline_summaries
        else {},
        "summaries": summaries,
        "overall": combine_summaries(summaries),
    }
    write_json(output_dir / "summary_all.json", aggregate)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PEToolBench LLM-memory GVR.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset_test"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--run-name", default="petool_llm_memory_gvr_qwen25_full")
    parser.add_argument("--history-types", nargs="+", choices=["p", "r", "c"], default=["p", "r", "c"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--baseline-cache", type=Path, default=Path("results/qwen25_7b_instruct_vllm_full"))
    parser.add_argument("--include-history", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--memory-override-history-types", nargs="+", choices=["p", "r", "c"], default=["p", "r"])
    parser.add_argument("--inference-strength", choices=["soft", "strong"], default="strong")
    parser.add_argument("--memory-content-style", choices=["full", "recommendation"], default="recommendation")
    parser.add_argument("--prompt-layout", choices=["custom", "original_memory"], default="custom")
    parser.add_argument(
        "--post-refine-policy",
        choices=[
            "none",
            "verify",
            "p_r_memory_c_generator",
            "always_memory",
            "p_r_memory_c_latest_namespace",
        ],
        default="p_r_memory_c_generator",
    )
    parser.add_argument("--force-memory", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-input-tokens", type=int, default=5000)
    parser.add_argument("--memory-max-new-tokens", type=int, default=320)
    parser.add_argument("--inference-max-new-tokens", type=int, default=128)
    parser.add_argument("--memory-temperature", type=float, default=0.0)
    parser.add_argument("--inference-temperature", type=float, default=0.0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
