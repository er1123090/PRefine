"""Run stratified ours-memory inference through OpenAI and Anthropic Batch APIs.

The runner reuses the exact ``model_input`` prompts already produced by the
three completed ours-memory pipelines.  It selects one shared, reproducible,
proportional sample over:

    turn in {single, multi}
    case in {easy, medium, hard, conflict-medium, conflict-hard}

The same sample IDs are used for every memory source and target model so the
resulting comparisons are paired.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import httpx


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_vanilla_batch_sample import (  # noqa: E402
    ANTHROPIC_API_URL,
    anthropic_headers,
    anthropic_result_payload,
    download_anthropic_results,
    download_openai_results,
    evaluation_report,
    fetch_anthropic_status,
    fetch_openai_status,
    load_state,
    now_iso,
    openai_client,
    openai_result_payload,
    read_json,
    read_jsonl,
    save_state,
    sha256_file,
    write_json,
    write_jsonl,
)


DATA_ROOT = (
    ROOT / "outputs" / "mpt_v2_0725_hint_no_easy_conflict"
)
DEFAULT_OUTPUT_DIR = (
    DATA_ROOT
    / "ours_memory_batch"
    / "stratified_sample_1000_seed_20260727"
)
POPULATION_PATH = DATA_ROOT / "population" / "population.jsonl"

DEFAULT_SAMPLE_SIZE = 1000
DEFAULT_SEED = 20260727

OPENAI_MODEL = "gpt-5"
OPENAI_REASONING_EFFORT = "medium"
OPENAI_MAX_COMPLETION_TOKENS = 4096

ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
ANTHROPIC_THINKING_BUDGET = 2048
ANTHROPIC_MAX_TOKENS = 3072

SOURCE_RUNS = {
    "qwen3_8b": {
        "inference": (
            DATA_ROOT
            / "ours_memory"
            / "qwen3_8b_memory__to__openrouter_gpt_oss_20b_high_reasoning"
            / "inference.jsonl"
        ),
        "predictions": (
            DATA_ROOT
            / "ours_memory"
            / "qwen3_8b_memory__to__openrouter_gpt_oss_20b_high_reasoning"
            / "predictions.json"
        ),
    },
    "gemma4_12b_it": {
        "inference": (
            DATA_ROOT
            / "ours_memory"
            / "gemma4_12b_it_memory__to__openrouter_gpt_oss_20b_high_reasoning"
            / "inference.jsonl"
        ),
        "predictions": (
            DATA_ROOT
            / "ours_memory"
            / "gemma4_12b_it_memory__to__openrouter_gpt_oss_20b_high_reasoning"
            / "predictions.json"
        ),
    },
    "gpt_oss_20b": {
        "inference": (
            DATA_ROOT
            / "ours_memory"
            / "gpt_oss_20b_memory__to__gpt_oss_20b_high_reasoning_optimized"
            / "inference.jsonl"
        ),
        "predictions": (
            DATA_ROOT
            / "ours_memory"
            / "gpt_oss_20b_memory__to__gpt_oss_20b_high_reasoning_optimized"
            / "predictions.json"
        ),
    },
}

PROVIDERS = ("openai", "anthropic")


def stratum_name(row: Mapping[str, Any]) -> str:
    case = str(row["pref_type"])
    if bool(row.get("conflict")):
        case = f"conflict-{case}"
    return f"{row['turn']}_{case}"


def _quota_tie_breaker(stratum: str) -> tuple[int, str]:
    # Keep the published 1,000-row allocation stable: a final tied remainder
    # goes to single before multi.
    return (0 if stratum.startswith("single_") else 1, stratum)


def proportional_quotas(
    counts: Mapping[str, int], sample_size: int
) -> Dict[str, int]:
    total = sum(int(value) for value in counts.values())
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if sample_size > total:
        raise ValueError(
            f"sample_size={sample_size} exceeds population={total}"
        )
    raw = {
        stratum: int(count) * sample_size / total
        for stratum, count in counts.items()
    }
    quotas = {stratum: math.floor(value) for stratum, value in raw.items()}
    remaining = sample_size - sum(quotas.values())
    ranked = sorted(
        counts,
        key=lambda stratum: (
            -(raw[stratum] - quotas[stratum]),
            *_quota_tie_breaker(stratum),
        ),
    )
    for stratum in ranked[:remaining]:
        quotas[stratum] += 1
    return dict(sorted(quotas.items()))


def selection_digest(sample_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()


def select_stratified_sample(
    population: Sequence[Dict[str, Any]], sample_size: int, seed: int
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in population:
        grouped[stratum_name(row)].append(row)
    counts = {stratum: len(rows) for stratum, rows in grouped.items()}
    quotas = proportional_quotas(counts, sample_size)

    selected: List[Dict[str, Any]] = []
    for stratum, rows in grouped.items():
        ranked = sorted(
            rows,
            key=lambda row: (
                selection_digest(str(row["sample_id"]), seed),
                int(row["population_index"]),
            ),
        )
        for row in ranked[: quotas[stratum]]:
            item = copy.deepcopy(row)
            item["stratum"] = stratum
            item["selection_sha256"] = selection_digest(
                str(item["sample_id"]), seed
            )
            selected.append(item)
    selected.sort(key=lambda row: int(row["population_index"]))
    if len(selected) != sample_size:
        raise RuntimeError(
            f"selected {len(selected)} rows, expected {sample_size}"
        )
    return selected, quotas


def resolve_memories(value: str) -> Sequence[str]:
    return tuple(SOURCE_RUNS) if value == "all" else (value,)


def resolve_providers(value: str) -> Sequence[str]:
    return PROVIDERS if value == "both" else (value,)


def target_name(provider: str) -> str:
    if provider == "openai":
        return "openai_gpt5_medium"
    if provider == "anthropic":
        return "anthropic_claude_haiku4_5_thinking2048"
    raise ValueError(f"unsupported provider: {provider}")


def job_dir(output_dir: Path, memory: str, provider: str) -> Path:
    return output_dir / "jobs" / f"{memory}_memory__to__{target_name(provider)}"


def load_selected_prompt_rows(
    path: Path, selected_ids: set[str]
) -> Dict[str, Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    prompt_hashes: Dict[str, set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id", ""))
            if sample_id not in selected_ids:
                continue
            prompt = str(row.get("model_input") or "")
            if not prompt:
                raise ValueError(
                    f"{path}:{line_number} has an empty model_input"
                )
            prompt_hashes[sample_id].add(
                hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            )
            latest[sample_id] = row
    missing = sorted(selected_ids - set(latest))
    if missing:
        raise RuntimeError(
            f"{path} is missing {len(missing)} selected prompts: {missing[:5]}"
        )
    variants = {
        sample_id: hashes
        for sample_id, hashes in prompt_hashes.items()
        if len(hashes) != 1
    }
    if variants:
        raise RuntimeError(
            f"{path} has prompt drift for {len(variants)} selected IDs"
        )
    return latest


def openai_request(sample_id: str, prompt: str) -> Dict[str, Any]:
    return {
        "custom_id": sample_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": OPENAI_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "reasoning_effort": OPENAI_REASONING_EFFORT,
            "max_completion_tokens": OPENAI_MAX_COMPLETION_TOKENS,
        },
    }


def anthropic_request(sample_id: str, prompt: str) -> Dict[str, Any]:
    return {
        "custom_id": sample_id,
        "params": {
            "model": ANTHROPIC_MODEL,
            "max_tokens": ANTHROPIC_MAX_TOKENS,
            "thinking": {
                "type": "enabled",
                "budget_tokens": ANTHROPIC_THINKING_BUDGET,
                "display": "omitted",
            },
            "messages": [{"role": "user", "content": prompt}],
        },
    }


def request_for(
    provider: str, sample_id: str, prompt: str
) -> Dict[str, Any]:
    if provider == "openai":
        return openai_request(sample_id, prompt)
    return anthropic_request(sample_id, prompt)


def estimated_job_cost(
    provider: str, prompt_rows: Iterable[Mapping[str, Any]]
) -> Dict[str, Any]:
    rows = list(prompt_rows)
    input_proxy = sum(
        int((row.get("token_counts") or {}).get("input_tokens", 0) or 0)
        for row in rows
    )
    missing_input_counts = sum(
        not int((row.get("token_counts") or {}).get("input_tokens", 0) or 0)
        for row in rows
    )
    if provider == "openai":
        expected_output_tokens = len(rows) * 1000
        input_cost = input_proxy / 1_000_000 * 0.625
        output_cost = expected_output_tokens / 1_000_000 * 5.0
    else:
        expected_output_tokens = len(rows) * (
            ANTHROPIC_THINKING_BUDGET + 150
        )
        input_cost = input_proxy / 1_000_000 * 0.50
        output_cost = expected_output_tokens / 1_000_000 * 2.50
    return {
        "input_token_proxy": input_proxy,
        "missing_input_token_counts": missing_input_counts,
        "assumed_output_tokens": expected_output_tokens,
        "estimated_input_usd": round(input_cost, 6),
        "estimated_output_usd": round(output_cost, 6),
        "estimated_total_usd": round(input_cost + output_cost, 6),
    }


def prepare_command(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    manifest_path = output_dir / "sample_manifest.jsonl"
    target_manifest_path = output_dir / "target_manifest.jsonl"
    summary_path = output_dir / "sample_summary.json"
    if (
        manifest_path.exists()
        or target_manifest_path.exists()
        or summary_path.exists()
    ) and not args.force:
        raise FileExistsError(
            f"sample already exists under {output_dir}; use --force to replace"
        )

    population = read_jsonl(POPULATION_PATH)
    target_sample, quotas = select_stratified_sample(
        population, args.sample_size, args.seed
    )
    reuse_manifest_path = (
        Path(args.reuse_manifest).resolve()
        if args.reuse_manifest
        else None
    )
    reused_rows = (
        read_jsonl(reuse_manifest_path)
        if reuse_manifest_path is not None
        else []
    )
    reused_ids = [str(row["sample_id"]) for row in reused_rows]
    if len(set(reused_ids)) != len(reused_ids):
        raise RuntimeError("reuse manifest has duplicate sample IDs")
    target_ids = {str(row["sample_id"]) for row in target_sample}
    if not set(reused_ids).issubset(target_ids):
        extra = sorted(set(reused_ids) - target_ids)
        raise RuntimeError(
            "reuse manifest is not a subset of the target sample: "
            f"{extra[:5]}"
        )
    sample = [
        row
        for row in target_sample
        if str(row["sample_id"]) not in set(reused_ids)
    ]
    if not sample:
        raise RuntimeError("incremental sample is empty")

    write_jsonl(manifest_path, sample)
    write_jsonl(target_manifest_path, target_sample)
    selected_ids = {str(row["sample_id"]) for row in sample}

    jobs: Dict[str, Any] = {}
    for memory in SOURCE_RUNS:
        source = SOURCE_RUNS[memory]
        prompt_rows = load_selected_prompt_rows(
            Path(source["inference"]), selected_ids
        )
        for provider in PROVIDERS:
            directory = job_dir(output_dir, memory, provider)
            requests_path = directory / "requests.jsonl"
            write_jsonl(
                requests_path,
                (
                    request_for(
                        provider,
                        str(row["sample_id"]),
                        str(prompt_rows[str(row["sample_id"])]["model_input"]),
                    )
                    for row in sample
                ),
            )
            estimate = estimated_job_cost(provider, prompt_rows.values())
            job_key = f"{memory}:{provider}"
            job_summary = {
                "memory_source": memory,
                "provider": provider,
                "model": (
                    OPENAI_MODEL
                    if provider == "openai"
                    else ANTHROPIC_MODEL
                ),
                "reasoning": (
                    {
                        "reasoning_effort": OPENAI_REASONING_EFFORT,
                        "max_completion_tokens": OPENAI_MAX_COMPLETION_TOKENS,
                    }
                    if provider == "openai"
                    else {
                        "thinking_type": "enabled",
                        "budget_tokens": ANTHROPIC_THINKING_BUDGET,
                        "max_tokens": ANTHROPIC_MAX_TOKENS,
                        "display": "omitted",
                    }
                ),
                "request_count": len(sample),
                "requests_path": str(requests_path),
                "requests_sha256": sha256_file(requests_path),
                "request_bytes": requests_path.stat().st_size,
                "source_inference": str(source["inference"]),
                "source_predictions": str(source["predictions"]),
                "cost_estimate": estimate,
            }
            write_json(directory / "job_summary.json", job_summary)
            jobs[job_key] = job_summary

    population_counts = Counter(stratum_name(row) for row in population)
    sample_counts = Counter(stratum_name(row) for row in sample)
    target_counts = Counter(stratum_name(row) for row in target_sample)
    reused_counts = Counter(stratum_name(row) for row in reused_rows)
    total_estimated_cost = sum(
        float(job["cost_estimate"]["estimated_total_usd"])
        for job in jobs.values()
    )
    summary = {
        "created_at": now_iso(),
        "population_path": str(POPULATION_PATH),
        "population_count": len(population),
        "sample_size_per_memory": len(sample),
        "target_sample_size_per_memory": len(target_sample),
        "reused_sample_size_per_memory": len(reused_rows),
        "memory_count": len(SOURCE_RUNS),
        "request_count_per_provider": len(sample) * len(SOURCE_RUNS),
        "request_count_all_providers": (
            len(sample) * len(SOURCE_RUNS) * len(PROVIDERS)
        ),
        "sampling": (
            "deterministic_sha256_within_proportional_turn_case_strata"
        ),
        "seed": args.seed,
        "population_stratum_counts": dict(sorted(population_counts.items())),
        "target_stratum_quotas": quotas,
        "target_stratum_counts": dict(sorted(target_counts.items())),
        "sample_stratum_counts": dict(sorted(sample_counts.items())),
        "reused_stratum_counts": dict(sorted(reused_counts.items())),
        "manifest_sha256": sha256_file(manifest_path),
        "target_manifest_sha256": sha256_file(target_manifest_path),
        "reuse_manifest": (
            str(reuse_manifest_path)
            if reuse_manifest_path is not None
            else None
        ),
        "reuse_manifest_sha256": (
            sha256_file(reuse_manifest_path)
            if reuse_manifest_path is not None
            else None
        ),
        "jobs": jobs,
        "estimated_total_usd": round(total_estimated_cost, 6),
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def validate_job(
    output_dir: Path, memory: str, provider: str, expected_ids: set[str]
) -> Dict[str, Any]:
    directory = job_dir(output_dir, memory, provider)
    summary = read_json(directory / "job_summary.json")
    requests_path = directory / "requests.jsonl"
    if sha256_file(requests_path) != summary["requests_sha256"]:
        raise RuntimeError(f"request checksum mismatch: {requests_path}")
    requests = read_jsonl(requests_path)
    custom_ids = [str(row.get("custom_id", "")) for row in requests]
    if len(requests) != len(expected_ids):
        raise RuntimeError(
            f"{memory}:{provider} has {len(requests)} requests, "
            f"expected {len(expected_ids)}"
        )
    if len(set(custom_ids)) != len(custom_ids):
        raise RuntimeError(f"{memory}:{provider} has duplicate custom IDs")
    if set(custom_ids) != expected_ids:
        raise RuntimeError(f"{memory}:{provider} custom ID set differs")

    for request in requests:
        if provider == "openai":
            body = request["body"]
            if body.get("model") != OPENAI_MODEL:
                raise RuntimeError("unexpected OpenAI model")
            if body.get("reasoning_effort") != OPENAI_REASONING_EFFORT:
                raise RuntimeError("unexpected OpenAI reasoning effort")
        else:
            params = request["params"]
            if params.get("model") != ANTHROPIC_MODEL:
                raise RuntimeError("unexpected Anthropic model")
            thinking = params.get("thinking") or {}
            if thinking.get("budget_tokens") != ANTHROPIC_THINKING_BUDGET:
                raise RuntimeError("unexpected Anthropic thinking budget")
    return {
        "memory": memory,
        "provider": provider,
        "request_count": len(requests),
        "request_bytes": requests_path.stat().st_size,
        "requests_sha256": summary["requests_sha256"],
        "estimated_total_usd": summary["cost_estimate"][
            "estimated_total_usd"
        ],
    }


def validate_command(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    summary = read_json(output_dir / "sample_summary.json")
    manifest_path = output_dir / "sample_manifest.jsonl"
    if sha256_file(manifest_path) != summary["manifest_sha256"]:
        raise RuntimeError("sample manifest checksum mismatch")
    manifest = read_jsonl(manifest_path)
    expected_ids = {str(row["sample_id"]) for row in manifest}
    results = [
        validate_job(output_dir, memory, provider, expected_ids)
        for memory in resolve_memories(args.memory)
        for provider in resolve_providers(args.provider)
    ]
    print(json.dumps(results, ensure_ascii=False, indent=2))


def submit_openai(directory: Path, memory: str) -> str:
    state_path = directory / "batch_state.json"
    if state_path.exists():
        state = load_state(state_path)
        print(f"SKIP openai {memory} existing_batch={state['batch_id']}")
        return str(state["batch_id"])
    requests_path = directory / "requests.jsonl"
    summary = read_json(directory / "job_summary.json")
    sample_summary = read_json(directory.parents[1] / "sample_summary.json")
    if sha256_file(requests_path) != summary["requests_sha256"]:
        raise RuntimeError(f"request checksum mismatch: {requests_path}")
    client = openai_client()
    with requests_path.open("rb") as handle:
        input_file = client.files.create(file=handle, purpose="batch")
    batch = client.batches.create(
        input_file_id=input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "experiment": "experiment8",
            "method": "ours_memory",
            "memory_source": memory,
            "sample": (
                f"stratified_incremental_"
                f"{sample_summary['sample_size_per_memory']}_"
                f"target_{sample_summary['target_sample_size_per_memory']}_"
                f"seed_{sample_summary['seed']}"
            ),
        },
    )
    response = batch.model_dump(mode="json")
    save_state(
        state_path,
        "openai",
        batch.id,
        response,
        input_file_id=input_file.id,
        request_sha256=summary["requests_sha256"],
    )
    print(f"OPENAI memory={memory} batch_id={batch.id}")
    return str(batch.id)


def submit_anthropic(directory: Path, memory: str) -> str:
    state_path = directory / "batch_state.json"
    if state_path.exists():
        state = load_state(state_path)
        print(f"SKIP anthropic {memory} existing_batch={state['batch_id']}")
        return str(state["batch_id"])
    requests_path = directory / "requests.jsonl"
    summary = read_json(directory / "job_summary.json")
    if sha256_file(requests_path) != summary["requests_sha256"]:
        raise RuntimeError(f"request checksum mismatch: {requests_path}")
    requests = read_jsonl(requests_path)
    timeout = httpx.Timeout(900.0, connect=60.0)
    with httpx.Client(timeout=timeout) as client:
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
        request_sha256=summary["requests_sha256"],
    )
    print(f"ANTHROPIC memory={memory} batch_id={batch_id}")
    return batch_id


def submit_command(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    validate_args = argparse.Namespace(
        output_dir=str(output_dir),
        memory=args.memory,
        provider=args.provider,
    )
    validate_command(validate_args)
    for memory in resolve_memories(args.memory):
        for provider in resolve_providers(args.provider):
            directory = job_dir(output_dir, memory, provider)
            if provider == "openai":
                submit_openai(directory, memory)
            else:
                submit_anthropic(directory, memory)


def fetch_status_rows(
    output_dir: Path,
    memories: Sequence[str],
    providers: Sequence[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for memory in memories:
        for provider in providers:
            directory = job_dir(output_dir, memory, provider)
            state_path = directory / "batch_state.json"
            if not state_path.exists():
                rows.append(
                    {
                        "memory": memory,
                        "provider": provider,
                        "status": "not_submitted",
                    }
                )
                continue
            if provider == "openai":
                body = fetch_openai_status(directory)
                rows.append(
                    {
                        "memory": memory,
                        "provider": provider,
                        "batch_id": body.get("id"),
                        "status": body.get("status"),
                        "request_counts": body.get("request_counts") or {},
                    }
                )
            else:
                body = fetch_anthropic_status(directory)
                rows.append(
                    {
                        "memory": memory,
                        "provider": provider,
                        "batch_id": body.get("id"),
                        "status": body.get("processing_status"),
                        "request_counts": body.get("request_counts") or {},
                    }
                )
    return rows


def status_command(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    rows = fetch_status_rows(
        output_dir,
        resolve_memories(args.memory),
        resolve_providers(args.provider),
    )
    print(json.dumps(rows, ensure_ascii=False, indent=2))


def load_selected_prediction_bases(
    path: Path, selected_ids: set[str]
) -> Dict[str, Dict[str, Any]]:
    values = read_json(path)
    if not isinstance(values, list):
        raise ValueError(f"{path} must contain a JSON array")
    selected = {
        str(row["sample_id"]): row
        for row in values
        if str(row.get("sample_id", "")) in selected_ids
    }
    missing = sorted(selected_ids - set(selected))
    if missing:
        raise RuntimeError(
            f"{path} is missing {len(missing)} selected rows: {missing[:5]}"
        )
    return selected


def normalize_job(
    output_dir: Path, memory: str, provider: str, raw_path: Path
) -> Dict[str, Any]:
    directory = job_dir(output_dir, memory, provider)
    manifest = read_jsonl(output_dir / "sample_manifest.jsonl")
    selected_ids = {str(row["sample_id"]) for row in manifest}
    source_path = Path(SOURCE_RUNS[memory]["predictions"])
    bases = load_selected_prediction_bases(source_path, selected_ids)
    raw_rows = read_jsonl(raw_path)
    raw_by_id = {str(row["custom_id"]): row for row in raw_rows}
    unknown = sorted(set(raw_by_id) - selected_ids)
    if unknown:
        raise RuntimeError(f"unknown result IDs: {unknown[:5]}")
    state = load_state(directory / "batch_state.json")
    model = OPENAI_MODEL if provider == "openai" else ANTHROPIC_MODEL
    records: List[Dict[str, Any]] = []
    for manifest_row in manifest:
        sample_id = str(manifest_row["sample_id"])
        raw = raw_by_id.get(sample_id)
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
        error = payload.get("error")
        record = copy.deepcopy(bases[sample_id])
        record.update(
            {
                "timestamp": now_iso(),
                "method": "ours_memory",
                "model_name": model,
                "request_model": model,
                "response_provider": provider,
                "api_provider": provider,
                "status": "ERROR" if error else "OK",
                "error": error,
                "llm_output": error or payload.get("content", ""),
                "reasoning_content": (
                    "" if error else payload.get("reasoning", "")
                ),
                "token_counts": {} if error else payload.get("usage", {}),
                "thinking": True,
                "reasoning_effort": (
                    OPENAI_REASONING_EFFORT
                    if provider == "openai"
                    else f"budget_tokens_{ANTHROPIC_THINKING_BUDGET}"
                ),
                "batch_provider": provider,
                "batch_id": state["batch_id"],
                "sample_seed": read_json(
                    output_dir / "sample_summary.json"
                )["seed"],
                "stratum": manifest_row["stratum"],
                "source_memory_model": memory,
                "source_prompt_predictions": str(source_path),
            }
        )
        records.append(record)
    records.sort(key=lambda row: int(row["population_index"]))
    write_json(directory / "predictions.json", records)
    write_jsonl(directory / "inference.jsonl", records)
    report = evaluation_report(records)
    report.update(
        {
            "provider": provider,
            "model": model,
            "memory_source": memory,
            "batch_id": state["batch_id"],
            "sample_manifest_sha256": sha256_file(
                output_dir / "sample_manifest.jsonl"
            ),
            "raw_result_count": len(raw_rows),
            "prediction_count": len(records),
            "generated_at": now_iso(),
        }
    )
    write_json(directory / "evaluation.json", report)
    return report


def collect_command(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    reports: List[Dict[str, Any]] = []
    for memory in resolve_memories(args.memory):
        for provider in resolve_providers(args.provider):
            directory = job_dir(output_dir, memory, provider)
            raw_path = (
                download_openai_results(directory)
                if provider == "openai"
                else download_anthropic_results(directory)
            )
            reports.append(
                normalize_job(output_dir, memory, provider, raw_path)
            )
    print(json.dumps(reports, ensure_ascii=False, indent=2))


def _rows_by_sample_id(
    rows: Sequence[Mapping[str, Any]], label: str
) -> Dict[str, Dict[str, Any]]:
    values: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id:
            raise RuntimeError(f"{label} has a row without sample_id")
        if sample_id in values:
            raise RuntimeError(f"{label} has duplicate sample_id={sample_id}")
        values[sample_id] = dict(row)
    return values


def merge_reused_command(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    reuse_root = Path(args.reuse_root).resolve()
    final_output_dir = Path(args.final_output_dir).resolve()

    incremental_manifest = read_jsonl(output_dir / "sample_manifest.jsonl")
    reused_manifest = read_jsonl(reuse_root / "sample_manifest.jsonl")
    target_manifest = read_jsonl(output_dir / "target_manifest.jsonl")
    incremental_ids = {
        str(row["sample_id"]) for row in incremental_manifest
    }
    reused_ids = {str(row["sample_id"]) for row in reused_manifest}
    target_ids = {str(row["sample_id"]) for row in target_manifest}
    if incremental_ids & reused_ids:
        overlap = sorted(incremental_ids & reused_ids)
        raise RuntimeError(
            f"incremental and reused manifests overlap: {overlap[:5]}"
        )
    if incremental_ids | reused_ids != target_ids:
        missing = sorted(target_ids - (incremental_ids | reused_ids))
        extra = sorted((incremental_ids | reused_ids) - target_ids)
        raise RuntimeError(
            f"merged manifest mismatch: missing={missing[:5]} "
            f"extra={extra[:5]}"
        )

    write_jsonl(final_output_dir / "sample_manifest.jsonl", target_manifest)
    reports: List[Dict[str, Any]] = []
    for memory in resolve_memories(args.memory):
        for provider in resolve_providers(args.provider):
            incremental_directory = job_dir(output_dir, memory, provider)
            reused_directory = job_dir(reuse_root, memory, provider)
            final_directory = job_dir(final_output_dir, memory, provider)
            incremental_rows = read_json(
                incremental_directory / "predictions.json"
            )
            reused_rows = read_json(reused_directory / "predictions.json")
            if not isinstance(incremental_rows, list):
                raise RuntimeError(
                    f"{incremental_directory}/predictions.json is not a list"
                )
            if not isinstance(reused_rows, list):
                raise RuntimeError(
                    f"{reused_directory}/predictions.json is not a list"
                )
            incremental_by_id = _rows_by_sample_id(
                incremental_rows, f"{memory}:{provider}:incremental"
            )
            reused_by_id = _rows_by_sample_id(
                reused_rows, f"{memory}:{provider}:reused"
            )
            if set(incremental_by_id) != incremental_ids:
                raise RuntimeError(
                    f"{memory}:{provider} incremental prediction IDs differ"
                )
            if set(reused_by_id) != reused_ids:
                raise RuntimeError(
                    f"{memory}:{provider} reused prediction IDs differ"
                )
            combined_by_id = {**reused_by_id, **incremental_by_id}
            if set(combined_by_id) != target_ids:
                raise RuntimeError(
                    f"{memory}:{provider} combined prediction IDs differ"
                )
            combined = sorted(
                combined_by_id.values(),
                key=lambda row: int(row["population_index"]),
            )
            write_json(final_directory / "predictions.json", combined)
            write_jsonl(final_directory / "inference.jsonl", combined)
            report = evaluation_report(combined)
            report.update(
                {
                    "provider": provider,
                    "model": (
                        OPENAI_MODEL
                        if provider == "openai"
                        else ANTHROPIC_MODEL
                    ),
                    "memory_source": memory,
                    "prediction_count": len(combined),
                    "reused_prediction_count": len(reused_rows),
                    "incremental_prediction_count": len(incremental_rows),
                    "reused_from": str(reused_directory),
                    "incremental_from": str(incremental_directory),
                    "generated_at": now_iso(),
                }
            )
            write_json(final_directory / "evaluation.json", report)
            merge_summary = {
                "memory_source": memory,
                "provider": provider,
                "target_count": len(combined),
                "reused_count": len(reused_rows),
                "incremental_count": len(incremental_rows),
                "sample_manifest_sha256": sha256_file(
                    final_output_dir / "sample_manifest.jsonl"
                ),
                "inference_sha256": sha256_file(
                    final_directory / "inference.jsonl"
                ),
            }
            write_json(final_directory / "merge_summary.json", merge_summary)
            reports.append(merge_summary)

    write_json(
        final_output_dir / "merge_summary.json",
        {
            "generated_at": now_iso(),
            "target_sample_size_per_memory": len(target_ids),
            "reused_sample_size_per_memory": len(reused_ids),
            "incremental_sample_size_per_memory": len(incremental_ids),
            "memory_count": len(resolve_memories(args.memory)),
            "provider_count": len(resolve_providers(args.provider)),
            "source_reuse_root": str(reuse_root),
            "source_incremental_root": str(output_dir),
            "jobs": reports,
        },
    )
    print(json.dumps(reports, ensure_ascii=False, indent=2))


def _all_batches_finished(rows: Sequence[Mapping[str, Any]]) -> bool:
    if not rows:
        return False
    return all(
        (
            row["status"] == "completed"
            if row["provider"] == "openai"
            else row["status"] == "ended"
        )
        for row in rows
    )


def _terminal_failure(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any] | None:
    openai_failures = {"failed", "expired", "cancelled", "canceled"}
    for row in rows:
        if row["provider"] == "openai" and row["status"] in openai_failures:
            return dict(row)
    return None


def monitor_command(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    memories = resolve_memories(args.memory)
    providers = resolve_providers(args.provider)
    deadline = time.monotonic() + args.timeout_hours * 3600
    monitor_path = output_dir / "monitor_status.json"
    while True:
        rows = fetch_status_rows(output_dir, memories, providers)
        snapshot = {"updated_at": now_iso(), "batches": rows}
        write_json(monitor_path, snapshot)
        print(json.dumps(snapshot, ensure_ascii=False), flush=True)
        failure = _terminal_failure(rows)
        if failure is not None:
            raise RuntimeError(f"terminal batch failure: {failure}")
        if _all_batches_finished(rows):
            collect_args = argparse.Namespace(
                output_dir=str(output_dir),
                memory=args.memory,
                provider=args.provider,
            )
            collect_command(collect_args)
            write_json(
                output_dir / "monitor_complete.json",
                {
                    "completed_at": now_iso(),
                    "status": "collected",
                    "batch_count": len(rows),
                },
            )
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"batches did not finish within {args.timeout_hours} hours"
            )
        time.sleep(args.interval_seconds)


def add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--memory",
        choices=("all", *SOURCE_RUNS.keys()),
        default="all",
    )
    parser.add_argument(
        "--provider",
        choices=("both", *PROVIDERS),
        default="both",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    prepare_parser.add_argument(
        "--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE
    )
    prepare_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    prepare_parser.add_argument(
        "--reuse-manifest",
        help=(
            "Existing sampled manifest to exclude from requests while keeping "
            "it inside the target sample."
        ),
    )
    prepare_parser.add_argument("--force", action="store_true")
    prepare_parser.set_defaults(func=prepare_command)

    for name, func in (
        ("validate", validate_command),
        ("submit", submit_command),
        ("status", status_command),
        ("collect", collect_command),
        ("monitor", monitor_command),
    ):
        command_parser = subparsers.add_parser(name)
        add_selection_arguments(command_parser)
        if name == "monitor":
            command_parser.add_argument(
                "--interval-seconds", type=int, default=120
            )
            command_parser.add_argument(
                "--timeout-hours", type=float, default=26.0
            )
        command_parser.set_defaults(func=func)

    merge_parser = subparsers.add_parser("merge-reused")
    add_selection_arguments(merge_parser)
    merge_parser.add_argument("--reuse-root", required=True)
    merge_parser.add_argument("--final-output-dir", required=True)
    merge_parser.set_defaults(func=merge_reused_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
