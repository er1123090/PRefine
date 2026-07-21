"""Run PEToolBench ours-memory prompt variants.

This runner preserves the original experiments5 ours-memory memory generation
prompts and adds PEToolBench-facing final prompt variants plus an optional
OpenAI-compatible vLLM backend. It is intentionally separate from
run_petool_ours_memory_original.py so existing experiment artifacts remain
auditable.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from run_petool_memory_gvr import combine_summaries, normalize_rows, write_json
import run_petool_ours_memory_original as original


TOOL_CALL_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "tool_name": {"type": "string"},
        "parameters": {
            "type": "object",
            "additionalProperties": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "number"},
                    {"type": "integer"},
                    {"type": "boolean"},
                    {"type": "array", "items": {"type": "string"}, "maxItems": 8},
                ]
            },
        },
    },
    "required": ["tool_name", "parameters"],
    "additionalProperties": False,
}


PARAMETER_VALUE_SCHEMA: Dict[str, Any] = {
    "anyOf": [
        {"type": "string"},
        {"type": "number"},
        {"type": "integer"},
        {"type": "boolean"},
        {"type": "array", "items": {"type": "string"}, "maxItems": 8},
    ]
}


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_sample_manifest(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    data = read_json(path)
    rows = data.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError(f"sample manifest rows must be a list: {path}")
    return data


def select_records(
    dataset_dir: Path,
    history_type: str,
    limit: Optional[int],
    manifest: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if manifest is None:
        return normalize_rows(dataset_dir, history_type, limit)
    all_records = normalize_rows(dataset_dir, history_type, None)
    by_source_index = {int(row["source_index"]): row for row in all_records}
    selected = []
    for row in manifest.get("rows", []):
        if row.get("history_type") != history_type:
            continue
        source_index = int(row["source_index"])
        if source_index not in by_source_index:
            raise KeyError(f"manifest source_index missing: {history_type} {source_index}")
        selected.append(by_source_index[source_index])
    return selected


def candidate_parameter_names(tool: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    for section in ("required_parameters", "optional_parameters"):
        params = tool.get(section, []) or []
        for param in params:
            if isinstance(param, dict) and param.get("name"):
                names.append(str(param["name"]))
    return names


def compact_candidate_schema(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    compact = []
    for tool in record.get("candidate_tools", []) or []:
        compact.append(
            {
                "tool_name": tool.get("tool_name", ""),
                "tool_description": tool.get("tool_description", ""),
                "required_parameters": tool.get("required_parameters", []) or [],
                "optional_parameters": tool.get("optional_parameters", []) or [],
                "allowed_parameter_names": candidate_parameter_names(tool),
            }
        )
    return compact


def tool_call_json_schema(record: Dict[str, Any]) -> Dict[str, Any]:
    schema = json.loads(json.dumps(TOOL_CALL_JSON_SCHEMA))
    candidates = [
        str(tool.get("tool_name", ""))
        for tool in record.get("candidate_tools", []) or []
        if tool.get("tool_name")
    ]
    if candidates:
        schema["properties"]["tool_name"]["enum"] = candidates
    parameter_names = sorted(
        {
            name
            for tool in record.get("candidate_tools", []) or []
            for name in candidate_parameter_names(tool)
        }
    )
    schema["properties"]["parameters"] = {
        "type": "object",
        "properties": {
            name: json.loads(json.dumps(PARAMETER_VALUE_SCHEMA))
            for name in parameter_names
        },
        "additionalProperties": False,
    }
    return schema


def build_retrieved_memory_text(memory: Dict[str, Any]) -> str:
    memory_text = memory.get("final_implicit_preference") or "None"
    api_history = memory.get("final_accumulated_api_calls") or []
    return (
        "[Long-term latent preference memory]\n"
        f"{memory_text}\n\n"
        "[Accumulated API-call history]\n"
        f"{json.dumps(api_history[-20:], ensure_ascii=False, indent=2)}"
    )


def parse_petool_tool_name(tool_name: str) -> tuple[str, str, str]:
    parts = str(tool_name or "").split(">.<")
    if len(parts) != 3:
        return "", "", str(tool_name or "")
    category = parts[0].lstrip("<")
    provider = parts[1]
    operation = parts[2].rstrip(">")
    return category, provider, operation


def build_candidate_rating_evidence_text(record: Dict[str, Any]) -> str:
    if record.get("history_type") != "r":
        return "No binary rating history for this example."

    history = record.get("history") or []
    exact_counts: Dict[str, Dict[str, int]] = {}
    provider_counts: Dict[str, Dict[str, int]] = {}
    for item in history:
        tool_call = item.get("tool_call") or {}
        tool_name = str(tool_call.get("tool_name") or "")
        _, provider, _ = parse_petool_tool_name(tool_name)
        if item.get("rating") == 1:
            key = "positive"
        elif item.get("rating") == 0:
            key = "negative"
        else:
            continue
        exact_counts.setdefault(tool_name, {"positive": 0, "negative": 0})[key] += 1
        if provider:
            provider_counts.setdefault(provider, {"positive": 0, "negative": 0})[key] += 1

    evidence = []
    for tool in record.get("candidate_tools", []) or []:
        tool_name = str(tool.get("tool_name") or "")
        _, provider, _ = parse_petool_tool_name(tool_name)
        exact = exact_counts.get(tool_name, {"positive": 0, "negative": 0})
        provider_total = provider_counts.get(provider, {"positive": 0, "negative": 0})
        if exact["positive"] or exact["negative"] or provider_total["positive"] or provider_total["negative"]:
            evidence.append(
                {
                    "tool_name": tool_name,
                    "exact_positive": exact["positive"],
                    "exact_negative": exact["negative"],
                    "provider_positive": provider_total["positive"],
                    "provider_negative": provider_total["negative"],
                }
            )

    if not evidence:
        return "No candidate/provider-level rating evidence in the interaction history."
    return json.dumps(evidence, ensure_ascii=False, indent=2)


def build_variant_final_prompt(
    record: Dict[str, Any],
    memory: Dict[str, Any],
    args: argparse.Namespace,
) -> str:
    if args.final_prompt_style == "original":
        return original.build_final_prompt(record, memory, args)
    if args.final_prompt_style == "baseline_system_tiebreak":
        retrieved = build_retrieved_memory_text(memory)
        return (
            "Use the original PEToolBench instruction below as the primary task definition.\n"
            "The current query and candidate schema in that instruction are authoritative.\n\n"
            "[Original PEToolBench instruction]\n"
            f"{record.get('system_prompt', '')}\n\n"
            "[Ours-memory tie-breaker]\n"
            f"{retrieved}\n\n"
            "[Decision Rules]\n"
            "1. First select candidate tools whose operation satisfies the current query.\n"
            "2. If memory suggests a provider or namespace, apply it only among candidates with the same operation.\n"
            "3. Ignore memory when it conflicts with the current query or candidate schema.\n"
            "4. Use memory for missing parameter values only when the current query leaves the value ambiguous.\n"
            "5. Return exactly one JSON object with keys tool_name and parameters. No markdown or explanations.\n\n"
            "Current user query:\n"
            f"{record.get('query', '')}\n\n"
            "Return exactly one JSON object and no extra text:\n"
            '{"tool_name": "<one candidate tool_name>", "parameters": {...}}\n'
        )

    history_note = {
        "p": "The history contains positive examples of the user's preferences.",
        "r": "The history contains ratings. Rating 1 is positive evidence; rating 0 is negative evidence.",
        "c": "The history is chronological. Later stable behavior is stronger evidence.",
    }.get(record.get("history_type"), "Use history as preference evidence.")
    schema = json.dumps(compact_candidate_schema(record), ensure_ascii=False, indent=2)
    retrieved = build_retrieved_memory_text(memory)
    query = record.get("query", "")
    base = (
        "You are solving PEToolBench, a personalized tool selection benchmark.\n"
        "Choose exactly one candidate tool for the current query by using the user's "
        "long-term latent preference memory and accumulated API-call history.\n\n"
        f"{history_note}\n\n"
        "[Task Definition]\n"
        "The current query is the primary requirement. Use memory/history only to resolve "
        "ambiguity about provider namespace, operation variants, or missing parameter values.\n"
        "Never choose a remembered provider when its operation does not satisfy the current query.\n\n"
        "[Candidate tools]\n"
        f"{schema}\n\n"
        "Relevant Memories (User Preferences & Constraints):\n"
        f"{retrieved}\n\n"
        "Current user query:\n"
        f"{query}\n\n"
    )
    structure_rules = (
        "[Output Structure Rules]\n"
        "1. Return exactly one JSON object with exactly these keys: tool_name, parameters.\n"
        "2. Copy tool_name exactly from one candidate tool_name string.\n"
        "3. parameters must be a JSON object. Include only parameter names allowed by the selected candidate.\n"
        "4. Prefer explicit values in the current query. Use memory/history only when the query leaves ambiguity.\n"
        "5. Do not output markdown, explanations, arrays, multiple tool calls, comments, or extra keys.\n"
        "6. Do not invent values beyond the schema; do not create empty values unless the schema/default/history supports them.\n"
    )
    provider_rules = (
        "\n[Provider Preference Rules]\n"
        "First match the current request to the correct operation. If multiple candidate tools satisfy the same operation, "
        "prefer the provider namespace supported by memory/history. Avoid provider namespaces with negative evidence. "
        "Reject a remembered provider if its operation does not answer the current query.\n"
    )
    rating_rules = (
        "\n[Rating Evidence Rules]\n"
        "For ratings histories, rating 1 means the selected tool/provider/parameter is preferred evidence. "
        "Rating 0 means the selected tool/provider/parameter should be avoided when a schema-valid alternative exists. "
        "Later rating evidence overrides earlier conflicting evidence.\n"
    )
    self_check = (
        "\n[Internal checklist - do not output]\n"
        "- Does the selected tool answer the current operation?\n"
        "- Is the selected provider consistent with memory/history when there is ambiguity?\n"
        "- Are all parameters valid for the selected candidate?\n"
        "- Is the final answer JSON only?\n"
    )

    if args.final_prompt_style == "variant1":
        extra = structure_rules
    elif args.final_prompt_style == "variant2":
        extra = structure_rules + provider_rules
    elif args.final_prompt_style == "variant3":
        extra = structure_rules + provider_rules + rating_rules
    elif args.final_prompt_style == "variant4":
        extra = structure_rules + provider_rules + rating_rules + self_check
    elif args.final_prompt_style == "variant5":
        extra = (
            structure_rules
            + provider_rules
            + rating_rules
            + "\n[Conflict Rule]\n"
            "If memory/history conflicts with the current query operation or candidate schema, ignore the conflicting memory.\n"
            + self_check
        )
    else:
        raise ValueError(f"unknown final_prompt_style: {args.final_prompt_style}")

    return (
        base
        + extra
        + "\nReturn exactly one JSON object and no extra text:\n"
        + '{"tool_name": "<one candidate tool_name>", "parameters": {...}}\n'
    )


def build_variant_messages(
    record: Dict[str, Any],
    memory: Dict[str, Any],
    args: argparse.Namespace,
) -> List[Dict[str, str]]:
    if args.final_prompt_style in {
        "baseline_chat_tiebreak",
        "baseline_chat_tiebreak_provider_guard",
        "baseline_chat_tiebreak_rating_guard",
    }:
        retrieved = build_retrieved_memory_text(memory)
        provider_guard = ""
        if args.final_prompt_style in {
            "baseline_chat_tiebreak_provider_guard",
            "baseline_chat_tiebreak_rating_guard",
        }:
            provider_guard = (
                " If the current query explicitly names a provider, source, product, API, brand, "
                "country/domain code, or service family and one candidate provider exactly contains that name, "
                "prefer that exact provider over a different provider suggested by memory. Do not let memory add "
                "provider suffixes such as Smartable/Data/API variants unless the current query is ambiguous and "
                "the remembered tool operation is the same operation."
            )
        rating_guard = ""
        if args.final_prompt_style == "baseline_chat_tiebreak_rating_guard" and record.get("history_type") == "r":
            rating_guard = (
                "\n\n[Candidate rating evidence]\n"
                f"{build_candidate_rating_evidence_text(record)}\n\n"
                "[Rating Use Rules]\n"
                "For rating histories, rating=1 is positive evidence and rating=0 is negative evidence. "
                "The accumulated API-call history above may list both satisfied and dissatisfied tool calls, "
                "so candidate rating evidence supersedes that API-call list when choosing among candidates. "
                "When multiple candidates satisfy the same operation, prefer exact tools or provider namespaces "
                "with positive evidence and avoid exact tools or provider namespaces with negative evidence. "
                "If a candidate has exact negative evidence and another operation-compatible candidate has no "
                "negative evidence, do not choose the negative-evidence candidate unless the current query names it explicitly."
            )
        system = (
            f"{record.get('system_prompt', '')}\n\n"
            "[Additional ours-memory evidence]\n"
            f"{retrieved}\n\n"
            "[Memory Use Rules]\n"
            "Use the original PEToolBench task definition and candidate schema above as authoritative. "
            "First select tools whose operation satisfies the current query. Use memory only as a tie-breaker "
            "among operation-compatible candidates, or for missing parameter values when the query is ambiguous. "
            "Ignore memory that conflicts with the current query or candidate schema. Return exactly one JSON object "
            "with keys tool_name and parameters, and no extra text."
            f"{provider_guard}"
            f"{rating_guard}"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": str(record.get("query", ""))},
        ]
    if args.final_prompt_style == "baseline_chat_strict":
        system = (
            f"{record.get('system_prompt', '')}\n\n"
            "[Strict Output Rules]\n"
            "Return exactly one JSON object with exactly these keys: tool_name, parameters. "
            "The tool_name value must be copied exactly from one candidate tool_name string. "
            "The parameters value must be a JSON object containing only valid parameter names for that selected tool. "
            "Do not output a parameter schema, explanations, markdown, comments, arrays, or multiple tool calls."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": str(record.get("query", ""))},
        ]
    return [
        {
            "role": "user",
            "content": build_variant_final_prompt(record, memory, args),
        }
    ]


class ExternalVllmCompletionRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        from transformers import AutoTokenizer

        self.args = args
        self.api_base = args.api_base.rstrip("/")
        self.endpoint = f"{self.api_base}/completions"
        tokenizer_model = args.tokenizer_model or args.model
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_model,
            trust_remote_code=True,
            local_files_only=args.local_tokenizer_only,
        )
        self.tokenizer.truncation_side = args.truncation_side
        self._seed_supported = True
        self._seed_lock = threading.Lock()

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        if self.args.prompt_format == "plain":
            parts = []
            for message in messages:
                role = str(message.get("role", "user")).strip().lower()
                title = {
                    "system": "System",
                    "user": "User",
                    "assistant": "Assistant",
                }.get(role, role.title() or "User")
                parts.append(f"{title}:\n{message.get('content', '')}")
            text = "\n\n".join(parts).rstrip() + "\n\nAssistant:\n"
            encoded = self.tokenizer(
                text,
                truncation=True,
                max_length=self.args.max_input_tokens,
                add_special_tokens=False,
            )
            return self.tokenizer.decode(encoded["input_ids"], skip_special_tokens=False)

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.args.max_input_tokens,
            add_special_tokens=False,
        )
        return self.tokenizer.decode(encoded["input_ids"], skip_special_tokens=False)

    def _post_json(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.args.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.args.request_timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _one_completion(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        guided_schema: Optional[Dict[str, Any]],
    ) -> str:
        payload = {
            "model": self.args.model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_new_tokens,
        }
        if guided_schema is not None:
            payload["guided_json"] = guided_schema
        with self._seed_lock:
            seed_supported = self._seed_supported
        if self.args.seed is not None and seed_supported:
            payload["seed"] = self.args.seed

        last_error: Optional[BaseException] = None
        for attempt in range(self.args.request_retries + 1):
            try:
                data = self._post_json(payload)
                choices = data.get("choices", [])
                return str(choices[0].get("text", "")).strip() if choices else ""
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                if "seed" in payload and exc.code in {400, 422} and "seed" in body.lower():
                    with self._seed_lock:
                        self._seed_supported = False
                    payload.pop("seed", None)
                    continue
                last_error = RuntimeError(f"HTTP {exc.code}: {body[:500]}")
            except Exception as exc:  # noqa: BLE001 - retry remote serving failures
                last_error = exc
            if attempt < self.args.request_retries:
                time.sleep(min(2**attempt, 10))
        raise RuntimeError(f"vLLM completion failed after retries: {last_error}")

    def generate_chat(
        self,
        messages_list: List[List[Dict[str, str]]],
        max_new_tokens: int,
        temperature: float,
        label: str,
        guided_schemas: Optional[List[Optional[Dict[str, Any]]]] = None,
    ) -> List[Dict[str, Any]]:
        prompts = [self._messages_to_prompt(messages) for messages in messages_list]
        if guided_schemas is None:
            guided_schemas = [None] * len(prompts)
        if len(guided_schemas) != len(prompts):
            raise ValueError("guided_schemas length must match messages_list length")
        outputs: List[Optional[Dict[str, Any]]] = [None] * len(prompts)
        total = len(prompts)
        for start in range(0, total, self.args.batch_size):
            batch = prompts[start : start + self.args.batch_size]
            schema_batch = guided_schemas[start : start + self.args.batch_size]
            started = time.time()
            with ThreadPoolExecutor(max_workers=self.args.max_concurrency) as executor:
                futures = {
                    executor.submit(
                        self._one_completion,
                        prompt,
                        max_new_tokens,
                        temperature,
                        schema,
                    ): offset
                    for offset, (prompt, schema) in enumerate(zip(batch, schema_batch))
                }
                for future in as_completed(futures):
                    offset = futures[future]
                    response = future.result()
                    outputs[start + offset] = {
                        "response": response,
                        "elapsed_seconds": (time.time() - started) / max(len(batch), 1),
                    }
            print(f"{label} generated {min(start + len(batch), total)}/{total}", flush=True)
        return [output or {"response": "", "elapsed_seconds": 0.0} for output in outputs]


def make_runner(args: argparse.Namespace) -> Any:
    if args.backend == "local_vllm":
        return original.VllmBatchRunner(args)
    if args.backend == "openai_compatible":
        if not args.api_base:
            raise SystemExit("--api-base is required for --backend openai_compatible")
        return ExternalVllmCompletionRunner(args)
    raise ValueError(f"unknown backend: {args.backend}")


def generate_variant_predictions(
    records: List[Dict[str, Any]],
    memory_records: List[Dict[str, Any]],
    args: argparse.Namespace,
    runner: Any,
) -> List[Dict[str, Any]]:
    memory_by_id = {row["example_id"]: row for row in memory_records}
    messages = [
        build_variant_messages(record, memory_by_id[record["example_id"]], args)
        for record in records
    ]
    generate_kwargs = {}
    if args.guided_json_output:
        generate_kwargs["guided_schemas"] = [tool_call_json_schema(record) for record in records]
    outputs = runner.generate_chat(
        messages,
        args.inference_max_new_tokens,
        args.inference_temperature,
        label=f"{args.final_prompt_style} inference",
        **generate_kwargs,
    )
    predictions = []
    for record, output in zip(records, outputs):
        memory = memory_by_id[record["example_id"]]
        parsed = original.parse_json_object(output["response"])
        predictions.append(
            {
                "example_id": record["example_id"],
                "source_index": record["source_index"],
                "history_type": record["history_type"],
                "query": record["query"],
                "api_call_ground_truth": record["api_call_ground_truth"],
                "memory_mode": memory.get("memory_mode"),
                "final_implicit_preference": memory.get("final_implicit_preference"),
                "response": output["response"],
                "parsed_response": parsed,
                "elapsed_seconds": output["elapsed_seconds"],
                "final_prompt_style": args.final_prompt_style,
                "variant_name": args.variant_name,
            }
        )
    return predictions


def generate_baseline_predictions(
    records: List[Dict[str, Any]],
    args: argparse.Namespace,
    runner: Any,
) -> List[Dict[str, Any]]:
    messages = [
        [
            {"role": "system", "content": record["system_prompt"]},
            {"role": "user", "content": record["query"]},
        ]
        for record in records
    ]
    generate_kwargs = {}
    if args.guided_json_output:
        generate_kwargs["guided_schemas"] = [tool_call_json_schema(record) for record in records]
    outputs = runner.generate_chat(
        messages,
        args.inference_max_new_tokens,
        args.inference_temperature,
        label="baseline inference",
        **generate_kwargs,
    )
    predictions = []
    for record, output in zip(records, outputs):
        predictions.append(
            {
                "example_id": record["example_id"],
                "source_index": record["source_index"],
                "history_type": record["history_type"],
                "query": record["query"],
                "api_call_ground_truth": record["api_call_ground_truth"],
                "response": output["response"],
                "parsed_response": original.parse_json_object(output["response"]),
                "elapsed_seconds": output["elapsed_seconds"],
            }
        )
    return predictions


def load_existing_memory(memory_dir: Path, history_type: str, expected_ids: Iterable[str]) -> Optional[List[Dict[str, Any]]]:
    rows = original.load_existing_memory(memory_dir, history_type)
    if rows is None:
        return None
    by_id = {row.get("example_id"): row for row in rows}
    expected = list(expected_ids)
    if all(example_id in by_id for example_id in expected):
        return [by_id[example_id] for example_id in expected]
    return None


def run_baseline(args: argparse.Namespace, runner: Any, manifest: Optional[Dict[str, Any]]) -> None:
    output_dir = args.output_dir / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: Dict[str, Dict[str, Any]] = {}
    for history_type in args.history_types:
        records = select_records(args.dataset_dir, history_type, args.limit, manifest)
        predictions = generate_baseline_predictions(records, args, runner)
        summary = original.evaluate_predictions(predictions)
        summaries[history_type] = summary
        write_json(output_dir / f"predictions_{history_type}.json", predictions)
        write_json(output_dir / f"summary_{history_type}.json", summary)
        print(history_type, json.dumps(summary, ensure_ascii=False), flush=True)

    aggregate = {
        "model": args.model,
        "tokenizer_model": args.tokenizer_model or args.model,
        "backend": args.backend,
        "api_base": args.api_base,
        "prompt_format": args.prompt_format,
        "guided_json_output": args.guided_json_output,
        "run_name": args.run_name,
        "mode": "baseline",
        "limit": args.limit,
        "sample_manifest": str(args.sample_manifest) if args.sample_manifest else None,
        "seed": args.seed,
        "history_types": args.history_types,
        "summaries": summaries,
        "overall": combine_summaries(summaries),
    }
    write_json(output_dir / "summary_all.json", aggregate)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2), flush=True)


def run_variant(args: argparse.Namespace, runner: Any, manifest: Optional[Dict[str, Any]]) -> None:
    original.configure_method_style(args)
    output_dir = args.output_dir / args.run_name
    memory_dir = output_dir / "memory"
    output_dir.mkdir(parents=True, exist_ok=True)
    memory_dir.mkdir(parents=True, exist_ok=True)

    summaries: Dict[str, Dict[str, Any]] = {}
    for history_type in args.history_types:
        records = select_records(args.dataset_dir, history_type, args.limit, manifest)
        expected_ids = [record["example_id"] for record in records]
        memory_records = None if args.force_memory else load_existing_memory(memory_dir, history_type, expected_ids)
        if memory_records is None:
            print(f"generating ours-memory {history_type} n={len(records)}", flush=True)
            memory_records = original.generate_memory_records(records, args, runner)
            write_json(memory_dir / f"memory_{history_type}.json", memory_records)
        else:
            print(f"using cached memory {history_type} n={len(memory_records)}", flush=True)

        predictions = generate_variant_predictions(records, memory_records, args, runner)
        summary = original.evaluate_predictions(predictions)
        summaries[history_type] = summary
        write_json(output_dir / f"predictions_{history_type}.json", predictions)
        write_json(output_dir / f"summary_{history_type}.json", summary)
        print(history_type, json.dumps(summary, ensure_ascii=False), flush=True)

    aggregate = {
        "model": args.model,
        "method": "petool_experiments5_ours_memory_variant",
        "run_name": args.run_name,
        "variant_name": args.variant_name,
        "variant_hypothesis": args.variant_hypothesis,
        "final_prompt_style": args.final_prompt_style,
        "tokenizer_model": args.tokenizer_model or args.model,
        "backend": args.backend,
        "api_base": args.api_base,
        "prompt_format": args.prompt_format,
        "guided_json_output": args.guided_json_output,
        "sample_manifest": str(args.sample_manifest) if args.sample_manifest else None,
        "seed": args.seed,
        "limit": args.limit,
        "history_types": args.history_types,
        "memory_mode": "verified_refine",
        "method_style": args.method_style,
        "memory_prompt_source": "experiments5/src/prompts.py LATENT_PREF_*",
        "inference_prompt_source": f"PEToolBench ours-memory {args.final_prompt_style}",
        "max_retries": args.max_retries,
        "memory_max_new_tokens": args.memory_max_new_tokens,
        "verifier_max_new_tokens": args.verifier_max_new_tokens,
        "summaries": summaries,
        "overall": combine_summaries(summaries),
    }
    write_json(output_dir / "summary_all.json", aggregate)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PEToolBench ours-memory variants.")
    parser.add_argument("--mode", choices=["baseline", "variant"], default="variant")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--tokenizer-model", default="")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset_test"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--run-name", default="petool_ours_memory_variant")
    parser.add_argument("--sample-manifest", type=Path, default=None)
    parser.add_argument("--variant-name", default="petool_ours_memory_variant1")
    parser.add_argument("--variant-hypothesis", default="")
    parser.add_argument(
        "--final-prompt-style",
        choices=[
            "original",
            "variant1",
            "variant2",
            "variant3",
            "variant4",
            "variant5",
            "baseline_system_tiebreak",
            "baseline_chat_tiebreak",
            "baseline_chat_tiebreak_provider_guard",
            "baseline_chat_tiebreak_rating_guard",
            "baseline_chat_strict",
        ],
        default="variant1",
    )
    parser.add_argument(
        "--method-style",
        choices=["experiment5_original", "experiment4_style"],
        default="experiment5_original",
    )
    parser.add_argument("--history-types", nargs="+", choices=["p", "r", "c"], default=["p", "r", "c"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force-memory", action="store_true")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-input-tokens", type=int, default=5000)
    parser.add_argument("--truncation-side", choices=["left", "right"], default="right")
    parser.add_argument("--prompt-format", choices=["chat", "plain"], default="chat")
    parser.add_argument("--guided-json-output", action="store_true")
    parser.add_argument("--memory-max-new-tokens", type=int, default=512)
    parser.add_argument("--verifier-max-new-tokens", type=int, default=256)
    parser.add_argument("--inference-max-new-tokens", type=int, default=160)
    parser.add_argument("--memory-temperature", type=float, default=0.4)
    parser.add_argument("--verifier-temperature", type=float, default=0.0)
    parser.add_argument("--inference-temperature", type=float, default=0.0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--backend", choices=["local_vllm", "openai_compatible"], default="local_vllm")
    parser.add_argument("--api-base", default="")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--request-retries", type=int, default=3)
    parser.add_argument("--max-concurrency", type=int, default=16)
    parser.add_argument("--local-tokenizer-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    manifest = load_sample_manifest(args.sample_manifest)
    runner = make_runner(args)
    if args.mode == "baseline":
        run_baseline(args, runner, manifest)
    else:
        run_variant(args, runner, manifest)


if __name__ == "__main__":
    main()
