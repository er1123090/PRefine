"""Run a reproducible vanilla-LLM sample through provider Batch APIs.

The script keeps Experiment8's experiment4 prompt/data preparation and writes
predictions in the existing experiment5-compatible output contract.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import httpx
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import (  # noqa: E402
    is_parsing_failed,
    load_pref_list,
    micro_f1_slot_and_value_or,
    micro_f1_with_filter,
    pref_em_rate,
)
from src.exp4_prompts import IMPLICIT_ZS_PROMPT_TEMPLATE  # noqa: E402
from src.exp4_runtime.inference import common, prepare_items  # noqa: E402


DEFAULT_OUTPUT_DIR = (
    ROOT / "outputs" / "vanilla_llm" / "batch_sample_1000_seed_20260723"
)
DEFAULT_SAMPLE_SIZE = 1000
DEFAULT_SEED = 20260723
DEFAULT_OPENAI_MODEL = "gpt-5"
DEFAULT_OPENAI_REASONING_EFFORT = "minimal"
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5"
DEFAULT_ANTHROPIC_MAX_TOKENS = 1024
ANTHROPIC_API_URL = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"

CONDITIONS = (
    {"turn": "single", "pref_type": "easy", "schema": "easy"},
    {"turn": "single", "pref_type": "medium", "schema": "easy"},
    {"turn": "single", "pref_type": "hard", "schema": "easy"},
    {"turn": "multi", "pref_type": "easy", "schema": "all"},
    {"turn": "multi", "pref_type": "medium", "schema": "all"},
    {"turn": "multi", "pref_type": "hard", "schema": "all"},
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def condition_name(row: Mapping[str, Any]) -> str:
    return f"{row['turn']}_{row['pref_type']}"


def provider_dir(output_dir: Path, provider: str) -> Path:
    if provider == "openai":
        return output_dir / "openai_gpt-5_minimal"
    if provider == "anthropic":
        return output_dir / "anthropic_claude-haiku-4-5_non_reasoning"
    raise ValueError(f"Unsupported provider: {provider}")


def resolve_providers(provider: str) -> Sequence[str]:
    return ("openai", "anthropic") if provider == "both" else (provider,)


def is_conflict_example(example: Mapping[str, Any]) -> bool:
    meta = example.get("meta")
    if not isinstance(meta, Mapping):
        return False
    return str(meta.get("subset", "")).startswith("conflict_")


def build_population(
    query: str,
    context_type: str,
    input_path: str | Path | None = None,
    exclude_easy_conflict: bool = False,
) -> List[Dict[str, Any]]:
    resolved_input_path = (
        Path(input_path).resolve()
        if input_path is not None
        else ROOT / "data" / "MPT_v2_mix600.json"
    )
    population: List[Dict[str, Any]] = []
    population_index = 0
    for condition in CONDITIONS:
        turn = condition["turn"]
        pref_type = condition["pref_type"]
        schema = condition["schema"]
        items = prepare_items(
            turn=turn,
            input_path=str(resolved_input_path),
            query_path=str(ROOT / "config" / f"query_{turn}turn_{query}.json"),
            pref_list_path=str(ROOT / "config" / "pref_list.json"),
            pref_group_path=str(ROOT / "config" / "pref_group.json"),
            pref_type=pref_type,
            max_queries=None,
        )
        tools_schema = common.load_tools_from_file(
            str(ROOT / "config" / f"schema_{schema}.json")
        )
        for item in items:
            example = item["original_ex"]
            if (
                exclude_easy_conflict
                and pref_type == "easy"
                and is_conflict_example(example)
            ):
                continue
            example_id = str(example.get("example_id", "unknown_user"))
            sub_idx = int(item["sub_idx"])
            prompt = common.build_memory_prompt(
                example=example,
                retrieved_memories_text="",
                current_user_utterance=item["utterance"],
                template=IMPLICIT_ZS_PROMPT_TEMPLATE,
                context_type=context_type,
                tools_schema=tools_schema,
            )
            population.append(
                {
                    "sample_id": f"v8-{population_index:05d}",
                    "population_index": population_index,
                    "turn": turn,
                    "query": query,
                    "schema": schema,
                    "pref_type": pref_type,
                    "condition": f"{turn}_{pref_type}",
                    "example_id": example_id,
                    "example_id_sub": f"{example_id}_{sub_idx}",
                    "sub_idx": sub_idx,
                    "utterance": item["utterance"],
                    "reference_ground_truth": item["ground_truth"],
                    "prompt": prompt,
                    "original_ex": example,
                }
            )
            population_index += 1
    return population


def sample_population(
    population: Sequence[Dict[str, Any]], sample_size: int, seed: int
) -> List[Dict[str, Any]]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if sample_size > len(population):
        raise ValueError(
            f"sample_size={sample_size} exceeds population={len(population)}"
        )
    selected_indices = sorted(
        random.Random(seed).sample(range(len(population)), sample_size)
    )
    return [copy.deepcopy(population[index]) for index in selected_indices]


def openai_request(
    row: Mapping[str, Any], model: str, reasoning_effort: str
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": row["prompt"]}],
    }
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    return {
        "custom_id": row["sample_id"],
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }


def anthropic_request(
    row: Mapping[str, Any], model: str, max_tokens: int
) -> Dict[str, Any]:
    return {
        "custom_id": row["sample_id"],
        "params": {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": row["prompt"]}],
        },
    }


def prepare_command(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    manifest_path = output_dir / "sample_manifest.jsonl"
    summary_path = output_dir / "sample_summary.json"
    if (manifest_path.exists() or summary_path.exists()) and not args.force:
        raise FileExistsError(
            f"Sample already exists under {output_dir}; use --force to replace it"
        )

    population = build_population(query=args.query, context_type=args.context_type)
    sample = sample_population(population, args.sample_size, args.seed)
    write_jsonl(manifest_path, sample)

    openai_dir = provider_dir(output_dir, "openai")
    anthropic_dir = provider_dir(output_dir, "anthropic")
    write_jsonl(
        openai_dir / "requests.jsonl",
        (
            openai_request(row, args.openai_model, args.openai_reasoning_effort)
            for row in sample
        ),
    )
    write_jsonl(
        anthropic_dir / "requests.jsonl",
        (
            anthropic_request(
                row, args.anthropic_model, args.anthropic_max_tokens
            )
            for row in sample
        ),
    )

    population_counts = Counter(condition_name(row) for row in population)
    sample_counts = Counter(condition_name(row) for row in sample)
    summary = {
        "created_at": now_iso(),
        "dataset": str(ROOT / "data" / "MPT_v2_mix600.json"),
        "query": args.query,
        "context_type": args.context_type,
        "sampling": "simple_random_without_replacement_over_prepared_requests",
        "seed": args.seed,
        "population_count": len(population),
        "sample_size": len(sample),
        "population_condition_counts": dict(sorted(population_counts.items())),
        "sample_condition_counts": dict(sorted(sample_counts.items())),
        "manifest_sha256": sha256_file(manifest_path),
        "providers": {
            "openai": {
                "model": args.openai_model,
                "reasoning_effort": args.openai_reasoning_effort,
                "endpoint": "/v1/chat/completions",
                "requests_sha256": sha256_file(openai_dir / "requests.jsonl"),
                "request_bytes": (openai_dir / "requests.jsonl").stat().st_size,
            },
            "anthropic": {
                "model": args.anthropic_model,
                "thinking": "disabled",
                "max_tokens": args.anthropic_max_tokens,
                "endpoint": "/v1/messages/batches",
                "requests_sha256": sha256_file(anthropic_dir / "requests.jsonl"),
                "request_bytes": (anthropic_dir / "requests.jsonl").stat().st_size,
            },
        },
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def require_key(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set")
    return value


def load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Batch state not found: {path}")
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"Batch state must be an object: {path}")
    return value


def save_state(
    path: Path, provider: str, batch_id: str, response: Mapping[str, Any], **extra: Any
) -> None:
    current = read_json(path) if path.exists() else {}
    if not isinstance(current, dict):
        current = {}
    current.update(
        {
            "provider": provider,
            "batch_id": batch_id,
            "updated_at": now_iso(),
            "response": dict(response),
            **extra,
        }
    )
    current.setdefault("created_at", now_iso())
    write_json(path, current)


def openai_client() -> OpenAI:
    return OpenAI(api_key=require_key("OPENAI_API_KEY"), timeout=120.0)


def anthropic_headers() -> Dict[str, str]:
    return {
        "x-api-key": require_key("ANTHROPIC_API_KEY"),
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }


def submit_openai(directory: Path, force_resubmit: bool) -> None:
    state_path = directory / "batch_state.json"
    if state_path.exists() and not force_resubmit:
        state = load_state(state_path)
        raise RuntimeError(
            f"OpenAI batch already recorded as {state.get('batch_id')}; "
            "refusing a duplicate charge without --force-resubmit"
        )
    requests_path = directory / "requests.jsonl"
    client = openai_client()
    with requests_path.open("rb") as handle:
        input_file = client.files.create(file=handle, purpose="batch")
    batch = client.batches.create(
        input_file_id=input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"experiment": "experiment8", "method": "vanilla_llm"},
    )
    response = batch.model_dump(mode="json")
    save_state(
        state_path,
        "openai",
        batch.id,
        response,
        input_file_id=input_file.id,
        request_sha256=sha256_file(requests_path),
    )
    print(f"OPENAI_BATCH_ID={batch.id}")


def submit_anthropic(directory: Path, force_resubmit: bool) -> None:
    state_path = directory / "batch_state.json"
    if state_path.exists() and not force_resubmit:
        state = load_state(state_path)
        raise RuntimeError(
            f"Anthropic batch already recorded as {state.get('batch_id')}; "
            "refusing a duplicate charge without --force-resubmit"
        )
    requests_path = directory / "requests.jsonl"
    requests = read_jsonl(requests_path)
    with httpx.Client(timeout=180.0) as client:
        response = client.post(
            f"{ANTHROPIC_API_URL}/messages/batches",
            headers=anthropic_headers(),
            json={"requests": requests},
        )
    response.raise_for_status()
    body = response.json()
    batch_id = str(body["id"])
    save_state(
        state_path,
        "anthropic",
        batch_id,
        body,
        request_sha256=sha256_file(requests_path),
    )
    print(f"ANTHROPIC_BATCH_ID={batch_id}")


def submit_command(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    summary = read_json(output_dir / "sample_summary.json")
    manifest_path = output_dir / "sample_manifest.jsonl"
    if sha256_file(manifest_path) != summary["manifest_sha256"]:
        raise RuntimeError("Sample manifest checksum differs from sample_summary.json")
    for provider in resolve_providers(args.provider):
        directory = provider_dir(output_dir, provider)
        if provider == "openai":
            submit_openai(directory, args.force_resubmit)
        else:
            submit_anthropic(directory, args.force_resubmit)


def fetch_openai_status(directory: Path) -> Dict[str, Any]:
    state_path = directory / "batch_state.json"
    state = load_state(state_path)
    batch = openai_client().batches.retrieve(state["batch_id"])
    body = batch.model_dump(mode="json")
    save_state(
        state_path,
        "openai",
        batch.id,
        body,
        input_file_id=state.get("input_file_id"),
        request_sha256=state.get("request_sha256"),
    )
    return body


def fetch_anthropic_status(directory: Path) -> Dict[str, Any]:
    state_path = directory / "batch_state.json"
    state = load_state(state_path)
    with httpx.Client(timeout=60.0) as client:
        response = client.get(
            f"{ANTHROPIC_API_URL}/messages/batches/{state['batch_id']}",
            headers=anthropic_headers(),
        )
    response.raise_for_status()
    body = response.json()
    save_state(
        state_path,
        "anthropic",
        str(body["id"]),
        body,
        request_sha256=state.get("request_sha256"),
    )
    return body


def status_command(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    for provider in resolve_providers(args.provider):
        directory = provider_dir(output_dir, provider)
        body = (
            fetch_openai_status(directory)
            if provider == "openai"
            else fetch_anthropic_status(directory)
        )
        if provider == "openai":
            counts = body.get("request_counts") or {}
            print(
                f"openai status={body.get('status')} "
                f"completed={counts.get('completed', 0)} "
                f"failed={counts.get('failed', 0)} "
                f"total={counts.get('total', 0)}"
            )
        else:
            counts = body.get("request_counts") or {}
            print(
                f"anthropic status={body.get('processing_status')} "
                f"succeeded={counts.get('succeeded', 0)} "
                f"errored={counts.get('errored', 0)} "
                f"processing={counts.get('processing', 0)}"
            )


def download_openai_results(directory: Path) -> Path:
    body = fetch_openai_status(directory)
    if body.get("status") != "completed":
        raise RuntimeError(f"OpenAI batch is not completed: {body.get('status')}")
    output_file_id = body.get("output_file_id")
    if not output_file_id:
        raise RuntimeError("OpenAI completed batch has no output_file_id")
    content = openai_client().files.content(output_file_id)
    raw_path = directory / "raw_results.jsonl"
    raw_path.write_bytes(content.content)
    error_file_id = body.get("error_file_id")
    if error_file_id:
        errors = openai_client().files.content(error_file_id)
        (directory / "raw_errors.jsonl").write_bytes(errors.content)
    return raw_path


def download_anthropic_results(directory: Path) -> Path:
    body = fetch_anthropic_status(directory)
    if body.get("processing_status") != "ended":
        raise RuntimeError(
            f"Anthropic batch is not ended: {body.get('processing_status')}"
        )
    state = load_state(directory / "batch_state.json")
    with httpx.Client(timeout=180.0) as client:
        response = client.get(
            f"{ANTHROPIC_API_URL}/messages/batches/{state['batch_id']}/results",
            headers=anthropic_headers(),
        )
    response.raise_for_status()
    raw_path = directory / "raw_results.jsonl"
    raw_path.write_text(response.text, encoding="utf-8")
    return raw_path


def openai_result_payload(row: Mapping[str, Any]) -> Dict[str, Any]:
    error = row.get("error")
    response = row.get("response") or {}
    if error:
        return {"error": f"BATCH_ERROR: {error}", "content": "", "usage": {}}
    if int(response.get("status_code", 0)) != 200:
        return {
            "error": (
                f"HTTP_{response.get('status_code')}: "
                f"{json.dumps(response.get('body'), ensure_ascii=False)}"
            ),
            "content": "",
            "usage": {},
        }
    body = response.get("body") or {}
    choices = body.get("choices") or []
    if not choices:
        return {"error": "BATCH_ERROR: missing choices", "content": "", "usage": {}}
    message = choices[0].get("message") or {}
    usage = body.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    return {
        "error": None,
        "content": message.get("content") or "",
        "reasoning": message.get("reasoning_content") or "",
        "usage": {
            "total_tokens": usage.get("total_tokens", 0),
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "reasoning_tokens": details.get("reasoning_tokens", 0),
            "cached_input_tokens": prompt_details.get("cached_tokens", 0),
        },
    }


def anthropic_result_payload(row: Mapping[str, Any]) -> Dict[str, Any]:
    result = row.get("result") or {}
    if result.get("type") != "succeeded":
        return {
            "error": f"BATCH_{str(result.get('type', 'unknown')).upper()}: "
            f"{json.dumps(result.get('error') or result, ensure_ascii=False)}",
            "content": "",
            "usage": {},
        }
    message = result.get("message") or {}
    content_blocks = message.get("content") or []
    content = "\n".join(
        str(block.get("text", ""))
        for block in content_blocks
        if block.get("type") == "text" and block.get("text")
    ).strip()
    usage = message.get("usage") or {}
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    cache_creation = int(usage.get("cache_creation_input_tokens", 0) or 0)
    cache_read = int(usage.get("cache_read_input_tokens", 0) or 0)
    return {
        "error": None,
        "content": content,
        "reasoning": "",
        "usage": {
            "total_tokens": input_tokens + output_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": 0,
            "cache_creation_input_tokens": cache_creation,
            "cached_input_tokens": cache_read,
        },
    }


def build_prediction(
    manifest_row: Mapping[str, Any],
    payload: Mapping[str, Any],
    provider: str,
    model: str,
    batch_id: str,
    sample_seed: int,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    error = payload.get("error")
    record = {
        "example_id": manifest_row["example_id"],
        "example_id_sub": manifest_row["example_id_sub"],
        "method": "vanilla_llm",
        "model_name": model,
        "context_type": "diag-apilist",
        "test_utterance": manifest_row["utterance"],
        "reference_ground_truth": manifest_row["reference_ground_truth"],
        "retrieved_memories": [],
        "status": "ERROR" if error else "OK",
        "error": error,
        "model_input": manifest_row["prompt"],
        "llm_output": error or payload.get("content", ""),
        "reasoning_content": "" if error else payload.get("reasoning", ""),
        "token_counts": {} if error else payload.get("usage", {}),
        "sample_id": manifest_row["sample_id"],
        "sample_seed": sample_seed,
        "population_index": manifest_row["population_index"],
        "turn": manifest_row["turn"],
        "query": manifest_row["query"],
        "schema": manifest_row["schema"],
        "pref_type": manifest_row["pref_type"],
        "condition": manifest_row["condition"],
        "batch_provider": provider,
        "batch_id": batch_id,
    }
    prediction = copy.deepcopy(manifest_row["original_ex"])
    prediction.update(record)
    return prediction, record


def metric_row(rows: Sequence[Dict[str, Any]], pref_map: Any) -> Dict[str, Any]:
    overall = micro_f1_slot_and_value_or(list(rows))
    pref_em = pref_em_rate(list(rows), pref_map)
    pref_f1 = micro_f1_with_filter(list(rows), pref_map, want_pref=True)
    nonpref = micro_f1_with_filter(list(rows), pref_map, want_pref=False)
    parse_failures = sum(is_parsing_failed(row)[0] for row in rows)
    api_errors = sum(row.get("status") == "ERROR" for row in rows)
    return {
        "n": len(rows),
        "api_errors": api_errors,
        "parsing_failures": parse_failures,
        "parsing_failure_rate": parse_failures / len(rows) if rows else 0.0,
        "overall": {
            "precision": overall.precision,
            "recall": overall.recall,
            "f1": overall.f1,
        },
        "pref": {
            "precision": pref_f1.precision,
            "recall": pref_f1.recall,
            "f1": pref_f1.f1,
            "exact_match_rate": pref_em.em,
            "exact_match_count": pref_em.em_count,
            "total": pref_em.n,
        },
        "nonpref": {
            "precision": nonpref.precision,
            "recall": nonpref.recall,
            "f1": nonpref.f1,
        },
    }


def evaluation_report(predictions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    pref_map = load_pref_list(str(ROOT / "config" / "pref_list.json"))
    report = {
        "overall": metric_row(predictions, pref_map),
        "conditions": {},
        "condition_conflict": {},
    }
    condition_values = sorted({row["condition"] for row in predictions})
    for condition in condition_values:
        subset = [row for row in predictions if row["condition"] == condition]
        report["conditions"][condition] = metric_row(subset, pref_map)
        report["condition_conflict"][condition] = {
            label: metric_row(
                [
                    row
                    for row in subset
                    if is_conflict_example(row) is want_conflict
                ],
                pref_map,
            )
            for label, want_conflict in (
                ("non_conflict", False),
                ("conflict", True),
            )
        }
    usage_fields = (
        "total_tokens",
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "cached_input_tokens",
        "cache_creation_input_tokens",
    )
    report["token_usage"] = {
        field: sum(
            int((row.get("token_counts") or {}).get(field, 0) or 0)
            for row in predictions
        )
        for field in usage_fields
    }
    return report


def normalize_results(
    output_dir: Path, provider: str, raw_path: Path
) -> Dict[str, Any]:
    directory = provider_dir(output_dir, provider)
    manifest = read_jsonl(output_dir / "sample_manifest.jsonl")
    manifest_by_id = {row["sample_id"]: row for row in manifest}
    raw_rows = read_jsonl(raw_path)
    raw_by_id = {row["custom_id"]: row for row in raw_rows}
    state = load_state(directory / "batch_state.json")
    summary = read_json(output_dir / "sample_summary.json")
    model = summary["providers"][provider]["model"]
    predictions: List[Dict[str, Any]] = []
    logs: List[Dict[str, Any]] = []
    unknown_ids = sorted(set(raw_by_id) - set(manifest_by_id))
    if unknown_ids:
        raise RuntimeError(f"Results contain unknown custom IDs: {unknown_ids[:5]}")
    for manifest_row in manifest:
        raw = raw_by_id.get(manifest_row["sample_id"])
        if raw is None:
            payload = {
                "error": "BATCH_ERROR: missing result for custom_id",
                "content": "",
                "usage": {},
            }
        elif provider == "openai":
            payload = openai_result_payload(raw)
        else:
            payload = anthropic_result_payload(raw)
        prediction, log_record = build_prediction(
            manifest_row,
            payload,
            provider,
            model,
            state["batch_id"],
            int(summary["seed"]),
        )
        predictions.append(prediction)
        logs.append(log_record)
    predictions.sort(key=lambda row: row["population_index"])
    logs.sort(key=lambda row: row["population_index"])
    write_json(directory / "predictions.json", predictions)
    write_jsonl(directory / "inference.jsonl", logs)
    report = evaluation_report(predictions)
    report.update(
        {
            "provider": provider,
            "model": model,
            "batch_id": state["batch_id"],
            "sample_manifest_sha256": sha256_file(
                output_dir / "sample_manifest.jsonl"
            ),
            "raw_result_count": len(raw_rows),
            "prediction_count": len(predictions),
            "generated_at": now_iso(),
        }
    )
    write_json(directory / "evaluation.json", report)
    return report


def collect_command(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    for provider in resolve_providers(args.provider):
        directory = provider_dir(output_dir, provider)
        raw_path = (
            download_openai_results(directory)
            if provider == "openai"
            else download_anthropic_results(directory)
        )
        report = normalize_results(output_dir, provider, raw_path)
        print(json.dumps(report, ensure_ascii=False, indent=2))


def add_shared_output_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    add_shared_output_argument(prepare_parser)
    prepare_parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    prepare_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    prepare_parser.add_argument("--query", choices=["hint", "nohint"], default="hint")
    prepare_parser.add_argument("--context-type", default="diag-apilist")
    prepare_parser.add_argument("--openai-model", default=DEFAULT_OPENAI_MODEL)
    prepare_parser.add_argument(
        "--openai-reasoning-effort",
        choices=["minimal", "low", "medium", "high"],
        default=DEFAULT_OPENAI_REASONING_EFFORT,
    )
    prepare_parser.add_argument(
        "--anthropic-model", default=DEFAULT_ANTHROPIC_MODEL
    )
    prepare_parser.add_argument(
        "--anthropic-max-tokens",
        type=int,
        default=DEFAULT_ANTHROPIC_MAX_TOKENS,
    )
    prepare_parser.add_argument("--force", action="store_true")
    prepare_parser.set_defaults(func=prepare_command)

    for command, func in (
        ("submit", submit_command),
        ("status", status_command),
        ("collect", collect_command),
    ):
        command_parser = subparsers.add_parser(command)
        add_shared_output_argument(command_parser)
        command_parser.add_argument(
            "--provider",
            choices=["openai", "anthropic", "both"],
            default="both",
        )
        if command == "submit":
            command_parser.add_argument("--force-resubmit", action="store_true")
        command_parser.set_defaults(func=func)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
