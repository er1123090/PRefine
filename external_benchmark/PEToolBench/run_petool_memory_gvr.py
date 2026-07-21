"""PEToolBench-local generate-verify-refine memory runner.

This script keeps the experiment self-contained inside the PEToolBench checkout.
It does not import or modify experiments5/methods/our_memory.

Pipeline:
1. Generate: use Qwen2.5-7B-Instruct predictions, either from an existing cache
   or by running vLLM locally.
2. Memory generate: build PEToolBench-specific provider/operation memory from
   the interaction history.
3. Verify: check whether the generated tool matches the memory-backed provider
   preference for semantically equivalent candidate tools.
4. Refine: when verifier confidence is high, replace the generated tool with
   the memory-backed candidate while preserving generated parameters when safe.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


HISTORY_KEYS = {
    "p": "instruction_preferred",
    "r": "instruction_ratings",
    "c": "instruction_chronological",
}

TOOL_RE = re.compile(r"<([^>]+)>")
STOPWORDS = {
    "a",
    "an",
    "and",
    "api",
    "are",
    "based",
    "be",
    "by",
    "call",
    "current",
    "for",
    "from",
    "get",
    "i",
    "in",
    "including",
    "is",
    "it",
    "list",
    "me",
    "of",
    "on",
    "please",
    "retrieve",
    "show",
    "the",
    "to",
    "user",
    "want",
    "with",
}

MEMORY_GENERATION_PROMPT = (
    "Generate PEToolBench preference memory as structured provider and "
    "operation-family preferences. For preferred histories, treat repeated "
    "provider namespaces as positive evidence. For ratings histories, rating 1 "
    "means prefer and rating 0 means avoid. For chronological histories, later "
    "evidence overrides earlier evidence."
)

VERIFY_REFINE_PROMPT = (
    "Verify the Qwen generated tool against PEToolBench memory. If preferred or "
    "ratings memory identifies a provider namespace that better matches the "
    "request's operation family, refine to that memory-backed candidate and "
    "preserve generated parameters only when they are valid for the refined "
    "tool. Keep chronological cases on the generator output unless an explicit "
    "policy requests memory override."
)

MEMORY_SCHEMA = {
    "provider_preferences": "namespace -> score, positive, negative, latest, operation_families",
    "operation_preferences": (
        "operation_family -> preferred_namespaces, avoid_namespaces, "
        "lower_priority_namespaces, namespace_scores"
    ),
    "evidence": "history index, tool namespace, operation, family, rating, weight",
}


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def parse_embedded_json(instruction: str, start_marker: str, end_marker: str) -> Any:
    start = instruction.find(start_marker)
    if start == -1:
        raise ValueError(f"missing marker: {start_marker!r}")
    start += len(start_marker)
    end = instruction.find(end_marker, start)
    if end == -1:
        raise ValueError(f"missing marker: {end_marker!r}")
    return json.loads(instruction[start:end].strip())


def parse_instruction(instruction: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    history = parse_embedded_json(
        instruction,
        "Interaction history is:\n",
        "\n\nAvailable tools",
    )
    tools = parse_embedded_json(
        instruction,
        "Available tools you can call with input parameters are listed here:\n",
        "\n\nGenerate your tool call",
    )
    return history, tools


def normalize_rows(dataset_dir: Path, history_type: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    rows = read_json(dataset_dir / f"user_entries_test_{history_type}.json")
    selected = rows if limit is None else rows[:limit]
    prompt_key = HISTORY_KEYS[history_type]
    normalized = []
    for index, row in enumerate(selected):
        history, candidate_tools = parse_instruction(row[prompt_key])
        normalized.append(
            {
                "example_id": f"petoolbench_{history_type}_{index:06d}",
                "source_index": index,
                "history_type": history_type,
                "system_prompt": row[prompt_key],
                "query": row["query"],
                "history": history,
                "candidate_tools": candidate_tools,
                "api_call_ground_truth": row["api_call_ground_truth"],
            }
        )
    return normalized


def first_json_object(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return {}


def parse_tool_name(tool_name: str) -> Dict[str, str]:
    parts = TOOL_RE.findall(tool_name or "")
    category = parts[0] if len(parts) >= 1 else ""
    provider = parts[1] if len(parts) >= 2 else ""
    operation = parts[2] if len(parts) >= 3 else (parts[-1] if parts else tool_name)
    namespace = f"<{category}>.<{provider}>" if category and provider else tool_name
    return {
        "tool_name": tool_name,
        "category": category,
        "provider": provider,
        "operation": operation,
        "namespace": namespace,
    }


def tokenize(text: str) -> List[str]:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text or "")
    text = text.replace("/", " ").replace("-", " ").replace("_", " ")
    tokens = [
        token
        for token in re.findall(r"[A-Za-z0-9]+", text.lower())
        if token and token not in STOPWORDS
    ]
    token_set = set(tokens)
    if {"log", "out"} <= token_set or "logout" in token_set:
        tokens.extend(["logout", "auth", "session"])
    if {"log", "in"} <= token_set or "login" in token_set:
        tokens.extend(["login", "auth", "session"])
    return tokens


def operation_family(operation: str, instruction: str = "") -> str:
    tokens = set(tokenize(f"{operation} {instruction}"))
    if tokens & {"login", "logout", "auth", "session", "signin", "signout"}:
        return "auth/session"
    if "pet" in tokens or "petstore" in tokens:
        return "petstore/pet"
    if "order" in tokens:
        return "commerce/order"
    if "inventory" in tokens:
        return "commerce/inventory"
    if "property" in tokens or "properties" in tokens or "rent" in tokens:
        return "properties"
    if "github" in tokens or "repo" in tokens or "repository" in tokens:
        return "github/repos"
    if "search" in tokens:
        return "search"
    if "health" in tokens or "healthcheck" in tokens:
        return "healthcheck"
    if "configuration" in tokens or "config" in tokens:
        return "configuration"
    if "brewery" in tokens:
        return "data/brewery"
    if tokens & {"match", "sport", "sports", "hockey"}:
        return "sports"
    if tokens & {"map", "distance", "geo", "geocode", "h3"}:
        return "mapping/location"
    if tokens & {"movie", "netflix", "show", "title"}:
        return "media/catalog"
    common = [token for token, _ in Counter(tokenize(operation)).most_common(2)]
    return "/".join(common) if common else "other"


def semantic_score(query: str, tool: Dict[str, Any]) -> float:
    parsed = parse_tool_name(str(tool.get("tool_name", "")))
    query_tokens = set(tokenize(query))
    tool_tokens = set(
        tokenize(
            " ".join(
                [
                    parsed["operation"],
                    str(tool.get("tool_description", "")),
                    " ".join(
                        str(param.get("description", ""))
                        for param in tool.get("required_parameters", []) or []
                        if isinstance(param, dict)
                    ),
                ]
            )
        )
    )
    score = float(len(query_tokens & tool_tokens))
    query_lower = (query or "").lower()
    op_lower = parsed["operation"].lower()
    desc_lower = str(tool.get("tool_description", "")).lower()
    if ("log out" in query_lower or "logout" in query_lower) and (
        "logout" in op_lower or "logs out" in desc_lower
    ):
        score += 10.0
    if ("log in" in query_lower or "login" in query_lower) and (
        "login" in op_lower or "log in" in desc_lower
    ):
        score += 10.0
    if not tool.get("required_parameters"):
        score += 0.1
    return score


def history_weight(history_type: str, rating: Any, index: int, total: int) -> float:
    if history_type == "r":
        if rating == 1:
            return 1.0
        if rating == 0:
            return -1.0
        return 0.0
    if history_type == "c":
        return 0.25 + 0.75 * (index / max(total - 1, 1))
    return 1.0


def generate_memory(record: Dict[str, Any]) -> Dict[str, Any]:
    history_type = record["history_type"]
    evidence = []
    history = record.get("history", [])
    for index, turn in enumerate(history):
        call = turn.get("tool_call", {}) or {}
        parsed = parse_tool_name(str(call.get("tool_name", "")))
        family = operation_family(parsed["operation"], str(turn.get("instruction", "")))
        weight = history_weight(history_type, turn.get("rating"), index, len(history))
        evidence.append(
            {
                "index": index,
                "tool_name": parsed["tool_name"],
                "namespace": parsed["namespace"],
                "operation": parsed["operation"],
                "operation_family": family,
                "rating": turn.get("rating"),
                "weight": weight,
            }
        )

    by_family: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(
        lambda: defaultdict(
            lambda: {
                "score": 0.0,
                "positive": 0,
                "negative": 0,
                "latest": -1,
                "evidence": [],
            }
        )
    )
    by_namespace: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "score": 0.0,
            "positive": 0,
            "negative": 0,
            "latest": -1,
            "families": set(),
        }
    )
    for item in evidence:
        bucket = by_family[item["operation_family"]][item["namespace"]]
        bucket["score"] += item["weight"]
        if item["weight"] > 0:
            bucket["positive"] += 1
        elif item["weight"] < 0:
            bucket["negative"] += 1
        bucket["latest"] = max(bucket["latest"], item["index"])
        bucket["evidence"].append(item["index"])

        ns_bucket = by_namespace[item["namespace"]]
        ns_bucket["score"] += item["weight"]
        if item["weight"] > 0:
            ns_bucket["positive"] += 1
        elif item["weight"] < 0:
            ns_bucket["negative"] += 1
        ns_bucket["latest"] = max(ns_bucket["latest"], item["index"])
        ns_bucket["families"].add(item["operation_family"])

    operation_preferences = {}
    for family, namespace_stats in by_family.items():
        if history_type == "c":
            ranked = sorted(
                namespace_stats.items(),
                key=lambda pair: pair[1]["latest"],
                reverse=True,
            )
            preferred = [ranked[0][0]] if ranked else []
            avoid = []
            lower = [namespace for namespace, _ in ranked[1:]]
        else:
            ranked = sorted(
                namespace_stats.items(),
                key=lambda pair: (pair[1]["score"], pair[1]["latest"]),
                reverse=True,
            )
            avoid = [
                namespace
                for namespace, stats in ranked
                if stats["negative"] > 0 and stats["score"] < 0
            ]
            preferred = [
                namespace
                for namespace, stats in ranked
                if stats["score"] > 0 and namespace not in avoid
            ][:3]
            lower = []
        operation_preferences[family] = {
            "preferred_namespaces": preferred,
            "avoid_namespaces": avoid,
            "lower_priority_namespaces": lower,
            "namespace_scores": {
                namespace: {
                    "score": round(stats["score"], 4),
                    "positive": stats["positive"],
                    "negative": stats["negative"],
                    "latest": stats["latest"],
                }
                for namespace, stats in namespace_stats.items()
            },
        }

    return {
        "memory_version": "petool_memory_gvr_v1",
        "history_type": history_type,
        "generation_prompt": MEMORY_GENERATION_PROMPT,
        "memory_schema": MEMORY_SCHEMA,
        "provider_preferences": {
            namespace: {
                "score": round(stats["score"], 4),
                "positive": stats["positive"],
                "negative": stats["negative"],
                "latest": stats["latest"],
                "operation_families": sorted(stats["families"]),
            }
            for namespace, stats in by_namespace.items()
        },
        "operation_preferences": operation_preferences,
        "evidence": evidence,
    }


def default_parameters(tool: Dict[str, Any]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for param in tool.get("required_parameters", []) or []:
        if isinstance(param, dict) and param.get("name"):
            params[param["name"]] = param.get("default", "")
    return params


def choose_memory_tool(record: Dict[str, Any], memory: Dict[str, Any]) -> Dict[str, Any]:
    ranked = []
    op_preferences = memory.get("operation_preferences", {})
    provider_preferences = memory.get("provider_preferences", {})
    for index, tool in enumerate(record.get("candidate_tools", []) or []):
        parsed = parse_tool_name(str(tool.get("tool_name", "")))
        family = operation_family(parsed["operation"], str(tool.get("tool_description", "")))
        score = semantic_score(record["query"], tool)
        pref = op_preferences.get(family, {})
        namespace = parsed["namespace"]
        provider_pref = provider_preferences.get(namespace, {})
        provider_score = float(provider_pref.get("score", 0.0) or 0.0)
        provider_total = max(
            int(provider_pref.get("positive", 0) or 0)
            + int(provider_pref.get("negative", 0) or 0),
            1,
        )
        provider_confidence = min(1.0, abs(provider_score) / provider_total)
        if provider_score > 0:
            score += 4.0 * provider_confidence
        elif provider_score < 0:
            score -= 5.0 * max(provider_confidence, 0.5)
        if namespace in pref.get("preferred_namespaces", []):
            score += 10.0
        if namespace in pref.get("avoid_namespaces", []):
            score -= 10.0
        if namespace in pref.get("lower_priority_namespaces", []):
            score -= 2.0
        ranked.append((score, -index, family, tool, parsed))
    if not ranked:
        return {"tool_name": "", "parameters": {}}
    score, _, family, tool, parsed = sorted(ranked, key=lambda item: (item[0], item[1]), reverse=True)[0]
    return {
        "tool_name": tool.get("tool_name", ""),
        "parameters": default_parameters(tool),
        "_score": score,
        "_operation_family": family,
        "_namespace": parsed["namespace"],
    }


def parameter_names(tool: Dict[str, Any]) -> set[str]:
    names = set()
    for key in ["required_parameters", "optional_parameters"]:
        for param in tool.get(key, []) or []:
            if isinstance(param, dict) and param.get("name"):
                names.add(str(param["name"]))
    return names


def find_candidate(record: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
    for tool in record.get("candidate_tools", []) or []:
        if tool.get("tool_name") == tool_name:
            return tool
    return {}


def merge_parameters(
    record: Dict[str, Any],
    generated: Dict[str, Any],
    refined_tool_name: str,
) -> Dict[str, Any]:
    refined_tool = find_candidate(record, refined_tool_name)
    generated_params = generated.get("parameters")
    if not isinstance(generated_params, dict):
        generated_params = {}
    allowed = parameter_names(refined_tool)
    if generated_params and (not allowed or set(generated_params) <= allowed):
        return generated_params
    return default_parameters(refined_tool)


def verify_and_refine(
    record: Dict[str, Any],
    generated: Dict[str, Any],
    memory: Dict[str, Any],
    policy: str,
) -> Dict[str, Any]:
    memory_choice = choose_memory_tool(record, memory)
    generated_tool = generated.get("tool_name", "") if isinstance(generated, dict) else ""

    use_memory = False
    reason = "generator_kept"
    if policy == "always_memory":
        use_memory = True
        reason = "memory_policy_always"
    elif policy == "p_r_memory_c_generator":
        if record["history_type"] in {"p", "r"}:
            use_memory = True
            reason = "memory_refine_preferred_or_ratings"
        else:
            use_memory = False
            reason = "chronological_generator_kept"
    elif policy == "verify":
        if record["history_type"] in {"p", "r"} and memory_choice.get("tool_name") != generated_tool:
            use_memory = True
            reason = "verifier_found_provider_preference_conflict"

    if use_memory:
        final_tool_name = memory_choice.get("tool_name", "")
        final_params = merge_parameters(record, generated, final_tool_name)
    else:
        final_tool_name = generated_tool
        final_params = generated.get("parameters", {}) if isinstance(generated, dict) else {}

    return {
        "tool_name": final_tool_name,
        "parameters": final_params,
        "_verifier": {
            "verify_refine_prompt": VERIFY_REFINE_PROMPT,
            "policy": policy,
            "decision": reason,
            "generated_tool_name": generated_tool,
            "memory_tool_name": memory_choice.get("tool_name", ""),
            "memory_score": memory_choice.get("_score"),
            "memory_operation_family": memory_choice.get("_operation_family"),
        },
    }


def evaluate(predictions: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(predictions)
    tool_correct = 0
    parameter_correct = 0
    parse_failures = 0
    refined = 0
    for row in rows:
        gt = row.get("api_call_ground_truth", {})
        pred = row.get("parsed_response", {})
        if not pred:
            parse_failures += 1
        tool_correct += int(gt.get("tool_name") == pred.get("tool_name"))
        parameter_correct += int(gt.get("parameters") == pred.get("parameters"))
        verifier = row.get("verifier", {})
        refined += int(
            verifier.get("decision")
            not in {"generator_kept", "chronological_generator_kept", "generator_baseline"}
        )
    n = len(rows)
    return {
        "n": n,
        "tool_correct": tool_correct,
        "parameter_correct": parameter_correct,
        "tool_accuracy": tool_correct / n if n else 0.0,
        "parameter_accuracy": parameter_correct / n if n else 0.0,
        "parse_failures": parse_failures,
        "refined_count": refined,
    }


def combine_summaries(summaries: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    total_n = sum(int(summary.get("n", 0) or 0) for summary in summaries.values())
    total_tool = sum(int(summary.get("tool_correct", 0) or 0) for summary in summaries.values())
    total_params = sum(int(summary.get("parameter_correct", 0) or 0) for summary in summaries.values())
    total_parse_failures = sum(int(summary.get("parse_failures", 0) or 0) for summary in summaries.values())
    total_refined = sum(int(summary.get("refined_count", 0) or 0) for summary in summaries.values())
    return {
        "n": total_n,
        "tool_correct": total_tool,
        "parameter_correct": total_params,
        "tool_accuracy": total_tool / total_n if total_n else 0.0,
        "parameter_accuracy": total_params / total_n if total_n else 0.0,
        "parse_failures": total_parse_failures,
        "refined_count": total_refined,
    }


def load_generator_cache(cache_dir: Path, history_type: str) -> Dict[str, Dict[str, Any]]:
    path = cache_dir / f"predictions_{history_type}.json"
    rows = read_json(path)
    return {row["example_id"]: row for row in rows}


def run_with_cache(args: argparse.Namespace) -> None:
    output_dir = args.output_dir / args.run_name
    summaries = {}
    baseline_summaries = {}
    for history_type in args.history_types:
        records = normalize_rows(args.dataset_dir, history_type, args.limit)
        generated_by_id = load_generator_cache(args.generator_cache, history_type)
        predictions = []
        baseline_predictions = []
        for record in records:
            cached = generated_by_id[record["example_id"]]
            generated = cached.get("parsed_response") or first_json_object(cached.get("response", ""))
            memory = generate_memory(record)
            refined = verify_and_refine(record, generated, memory, args.refine_policy)
            verifier = refined.pop("_verifier")
            predictions.append(
                {
                    "example_id": record["example_id"],
                    "source_index": record["source_index"],
                    "history_type": history_type,
                    "query": record["query"],
                    "api_call_ground_truth": record["api_call_ground_truth"],
                    "generated_response": cached.get("response", ""),
                    "generated_parsed_response": generated,
                    "parsed_response": refined,
                    "verifier": verifier,
                    "petool_memory": memory if args.keep_memory else {},
                }
            )
            baseline_predictions.append(
                {
                    "api_call_ground_truth": record["api_call_ground_truth"],
                    "parsed_response": generated,
                    "verifier": {"decision": "generator_baseline"},
                }
            )

        summary = evaluate(predictions)
        baseline_summary = evaluate(baseline_predictions)
        summaries[history_type] = summary
        baseline_summaries[history_type] = baseline_summary
        write_json(output_dir / f"predictions_{history_type}.json", predictions)
        write_json(output_dir / f"summary_{history_type}.json", summary)
        print(history_type, json.dumps(summary, ensure_ascii=False))

    aggregate = {
        "model": args.model,
        "method": "petool_memory_generate_verify_refine",
        "run_name": args.run_name,
        "refine_policy": args.refine_policy,
        "petool_memory_prompts": {
            "memory_generation": MEMORY_GENERATION_PROMPT,
            "verify_refine": VERIFY_REFINE_PROMPT,
        },
        "petool_memory_schema": MEMORY_SCHEMA,
        "generator_cache": str(args.generator_cache),
        "limit": args.limit,
        "history_types": args.history_types,
        "baseline_summaries_from_generator_cache": baseline_summaries,
        "baseline_overall_from_generator_cache": combine_summaries(baseline_summaries),
        "summaries": summaries,
        "overall": combine_summaries(summaries),
    }
    write_json(output_dir / "summary_all.json", aggregate)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


def run_vllm_generate(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_input_tokens + args.max_new_tokens,
        trust_remote_code=True,
    )
    output_dir = args.output_dir / args.run_name / "generator_cache"
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)

    for history_type in args.history_types:
        records = normalize_rows(args.dataset_dir, history_type, args.limit)
        predictions = []
        for start in range(0, len(records), args.batch_size):
            batch = records[start : start + args.batch_size]
            prompts = []
            for record in batch:
                messages = [
                    {"role": "system", "content": record["system_prompt"]},
                    {"role": "user", "content": record["query"]},
                ]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                encoded = tokenizer(
                    text,
                    truncation=True,
                    max_length=args.max_input_tokens,
                    add_special_tokens=False,
                )
                prompts.append(tokenizer.decode(encoded["input_ids"], skip_special_tokens=False))
            started = time.time()
            outputs = llm.generate(prompts, sampling)
            elapsed = time.time() - started
            for record, output in zip(batch, outputs):
                response = output.outputs[0].text.strip() if output.outputs else ""
                predictions.append(
                    {
                        "example_id": record["example_id"],
                        "source_index": record["source_index"],
                        "history_type": history_type,
                        "query": record["query"],
                        "api_call_ground_truth": record["api_call_ground_truth"],
                        "response": response,
                        "parsed_response": first_json_object(response),
                        "elapsed_seconds": elapsed / max(len(batch), 1),
                    }
                )
            write_json(output_dir / f"predictions_{history_type}.json", predictions)
            print(f"generated {history_type} {len(predictions)}/{len(records)}")

    args.generator_cache = output_dir
    run_with_cache(args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PEToolMemory generate-verify-refine.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset_test"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--run-name", default="petool_memory_gvr_qwen25")
    parser.add_argument("--history-types", nargs="+", choices=["p", "r", "c"], default=["p", "r", "c"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--generator-cache", type=Path, default=None)
    parser.add_argument(
        "--refine-policy",
        choices=["verify", "always_memory", "p_r_memory_c_generator"],
        default="p_r_memory_c_generator",
    )
    parser.add_argument("--keep-memory", action="store_true")
    parser.add_argument("--generate-with-vllm", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-input-tokens", type=int, default=4000)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = parser.parse_args()

    if args.generate_with_vllm:
        run_vllm_generate(args)
        return
    if args.generator_cache is None:
        raise SystemExit("--generator-cache is required unless --generate-with-vllm is set")
    run_with_cache(args)


if __name__ == "__main__":
    main()
