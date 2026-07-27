#!/usr/bin/env python3
"""Run a deterministic Experiment8 population against a local chat server."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from openai import AsyncOpenAI
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
OURS_DIR = ROOT / "methods/ours_memory"
for path in (ROOT, SCRIPT_DIR, OURS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_vanilla_batch_sample as population_runtime  # noqa: E402
import inference_multiturn as ours_multi  # type: ignore  # noqa: E402
import inference_singleturn as ours_single  # type: ignore  # noqa: E402
from src.exp4_prompts import (  # noqa: E402
    IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE,
    IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE_MULTITURN,
)
from src.exp4_runtime import common  # noqa: E402
from src.provider_config import (  # noqa: E402
    is_openrouter_endpoint,
    resolve_openai_compatible_endpoint,
)


class RequestRateLimiter:
    """Space request starts evenly to avoid upstream burst throttling."""

    def __init__(self, requests_per_second: float) -> None:
        self.interval_seconds = 1.0 / requests_per_second
        self.next_start = 0.0
        self.lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self.lock:
            now = time.monotonic()
            delay = self.next_start - now
            if delay > 0:
                await asyncio.sleep(delay)
                now = time.monotonic()
            self.next_start = max(self.next_start, now) + self.interval_seconds


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()


def read_checkpoint(path: Path) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid checkpoint JSON at {path}:{line_number}"
                ) from exc
            sample_id = str(record.get("sample_id", ""))
            if not sample_id:
                raise ValueError(f"Missing sample_id at {path}:{line_number}")
            records[sample_id] = record
    return records


def read_memory(path: Path) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid memory JSON at {path}:{line_number}") from exc
            example_id = str(record.get("example_id", ""))
            if not example_id:
                raise ValueError(f"Missing example_id at {path}:{line_number}")
            if example_id in records:
                raise ValueError(f"Duplicate memory example_id: {example_id}")
            records[example_id] = record
    return records


def prepare_population(args: argparse.Namespace) -> list[Dict[str, Any]]:
    rows = population_runtime.build_population(
        query=args.query,
        context_type="diag-apilist",
        input_path=getattr(
            args,
            "input_path",
            str(ROOT / "data" / "MPT_v2_mix600.json"),
        ),
        exclude_easy_conflict=bool(
            getattr(args, "exclude_easy_conflict", False)
        ),
    )
    if len(rows) != args.expected_count:
        raise RuntimeError(
            f"Population mismatch: expected={args.expected_count}, actual={len(rows)}"
        )
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise RuntimeError("Population contains duplicate sample_id values")

    if args.method == "vanilla_llm":
        return rows

    memory_path = Path(args.memory_path).resolve()
    memory = read_memory(memory_path)
    source_ids = {str(row["example_id"]) for row in rows}
    missing = sorted(source_ids - set(memory))
    if missing:
        raise RuntimeError(
            f"Memory is missing {len(missing)} population users: {missing[:5]}"
        )

    schema_cache: Dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        schema_name = str(row["schema"])
        if schema_name not in schema_cache:
            schema_cache[schema_name] = common.load_tools_from_file(
                str(ROOT / "config" / f"schema_{schema_name}.json")
            )
        example_id = str(row["example_id"])
        if row["turn"] == "single":
            prompt = ours_single.build_memory_input_prompt(
                example=row["original_ex"],
                user_memory=memory[example_id],
                current_user_utterance=row["utterance"],
                template=IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE,
                context_type="memory_api",
                tools_schema=schema_cache[schema_name],
            )
        else:
            prompt = ours_multi.build_memory_input_prompt(
                example=row["original_ex"],
                user_memory=memory[example_id],
                current_user_utterance=row["utterance"],
                template=IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE_MULTITURN,
                context_type="memory_api",
                tools_schema=schema_cache[schema_name],
            )
        row["prompt"] = prompt
        row["memory_path"] = str(memory_path)
    return rows


def usage_payload(response: Any) -> Dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    details = getattr(usage, "completion_tokens_details", None)
    if details is None:
        details = getattr(usage, "output_tokens_details", None)
    reasoning_tokens = getattr(details, "reasoning_tokens", 0) if details else 0
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    result: Dict[str, Any] = {
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "reasoning_tokens": int(reasoning_tokens or 0),
        "cached_input_tokens": int(
            getattr(prompt_details, "cached_tokens", 0) or 0
        ),
    }
    model_extra = getattr(usage, "model_extra", None) or {}
    cost = getattr(usage, "cost", None)
    if cost is None:
        cost = model_extra.get("cost")
    if cost is not None:
        result["cost_usd"] = float(cost)
    return result


def build_chat_completion_request(
    row: Mapping[str, Any],
    *,
    args: argparse.Namespace,
    max_tokens: int,
) -> Dict[str, Any]:
    openrouter = is_openrouter_endpoint(getattr(args, "api_base", None))
    request: Dict[str, Any] = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": common.sanitize_text_for_transport(
                    str(row["prompt"])
                ),
            }
        ],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "max_tokens": max_tokens,
    }
    if openrouter:
        extra_body: Dict[str, Any] = {}
        if args.reasoning_effort is not None:
            extra_body["reasoning"] = {
                "effort": args.reasoning_effort,
                "exclude": True,
            }
        if args.top_k >= 0:
            extra_body["top_k"] = args.top_k
        if args.min_p > 0:
            extra_body["min_p"] = args.min_p
        if extra_body:
            request["extra_body"] = extra_body
    else:
        request["extra_body"] = {
            "top_k": args.top_k,
            "min_p": args.min_p,
            "chat_template_kwargs": {
                "enable_thinking": bool(args.thinking),
            },
        }
    if args.reasoning_effort is not None and not openrouter:
        request["reasoning_effort"] = args.reasoning_effort
    return request


async def infer_one(
    row: Mapping[str, Any],
    *,
    args: argparse.Namespace,
    max_tokens: int,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    checkpoint_path: Path,
    checkpoint_lock: asyncio.Lock,
    progress: tqdm,
    attempt: int,
    rate_limiter: RequestRateLimiter | None,
) -> Dict[str, Any]:
    started = time.monotonic()
    error: str | None = None
    content = ""
    reasoning = ""
    token_counts: Dict[str, Any] = {}
    finish_reason: str | None = None
    response_model: str | None = None
    response_provider: str | None = None

    async with semaphore:
        if rate_limiter is not None:
            await rate_limiter.wait()
        try:
            response = await client.chat.completions.create(
                **build_chat_completion_request(
                    row,
                    args=args,
                    max_tokens=max_tokens,
                )
            )
            message = response.choices[0].message
            content = (getattr(message, "content", None) or "").strip()
            reasoning = (
                getattr(message, "reasoning_content", None)
                or getattr(message, "reasoning", None)
                or ""
            ).strip()
            finish_reason = str(response.choices[0].finish_reason or "")
            response_model = str(getattr(response, "model", "") or "")
            response_extra = getattr(response, "model_extra", None) or {}
            response_provider = response_extra.get("provider")
            token_counts = usage_payload(response)
            if not content:
                error = "API_ERROR: model returned empty final content"
            elif finish_reason == "length":
                error = "API_ERROR: model hit max_tokens before returning a final answer"
            elif args.thinking and content.lstrip().startswith("<think>"):
                error = "API_ERROR: model hit max_tokens before returning a final answer"
        except Exception as exc:
            error = f"API_ERROR: {exc}"

        record: Dict[str, Any] = {
            "timestamp": now_iso(),
            "sample_id": row["sample_id"],
            "population_index": row["population_index"],
            "example_id": row["example_id"],
            "example_id_sub": row["example_id_sub"],
            "method": args.method,
            "model_name": getattr(args, "record_model_name", None) or args.model,
            "request_model": args.model,
            "api_provider": (
                "openrouter"
                if is_openrouter_endpoint(args.api_base)
                else "openai_compatible"
            ),
            "response_model": response_model,
            "response_provider": response_provider,
            "context_type": (
                "diag-apilist" if args.method == "vanilla_llm" else "memory_api"
            ),
            "turn": row["turn"],
            "query": row["query"],
            "schema": row["schema"],
            "pref_type": row["pref_type"],
            "condition": row["condition"],
            "test_utterance": row["utterance"],
            "reference_ground_truth": row["reference_ground_truth"],
            "status": "ERROR" if error else "OK",
            "error": error,
            "model_input": row["prompt"],
            "llm_output": error or content,
            "reasoning_content": "" if error else reasoning,
            "token_counts": {} if error else token_counts,
            "finish_reason": finish_reason,
            "thinking": bool(args.thinking),
            "reasoning_effort": args.reasoning_effort,
            "max_tokens_requested": max_tokens,
            "attempt": attempt,
            "elapsed_seconds": time.monotonic() - started,
        }
        if args.method == "ours_memory":
            record["memory_path"] = row["memory_path"]

        async with checkpoint_lock:
            append_jsonl(checkpoint_path, record)
        progress.update(1)
        return record


def materialize_predictions(
    rows: Sequence[Mapping[str, Any]],
    records: Mapping[str, Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    predictions: list[Dict[str, Any]] = []
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id not in records:
            raise RuntimeError(f"Missing final record for {sample_id}")
        prediction = copy.deepcopy(row["original_ex"])
        prediction.update(records[sample_id])
        predictions.append(prediction)
    predictions.sort(key=lambda row: int(row["population_index"]))
    return predictions


def status_counts(records: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    return dict(sorted(Counter(str(row.get("status")) for row in records).items()))


def record_is_complete(record: Mapping[str, Any]) -> bool:
    output = str(record.get("llm_output", "")).lstrip()
    if record.get("status") != "OK":
        return False
    if record.get("finish_reason") == "length":
        return False
    if not bool(record.get("thinking")):
        return True
    return not output.startswith("<think>")


async def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "inference.jsonl"
    predictions_path = output_dir / "predictions.json"
    evaluation_path = output_dir / "evaluation.json"
    summary_path = output_dir / "run_summary.json"

    if not args.resume and checkpoint_path.exists():
        raise FileExistsError(
            f"{checkpoint_path} already exists; pass --resume to continue"
        )

    rows = prepare_population(args)
    records = read_checkpoint(checkpoint_path) if args.resume else {}
    known_ids = {str(row["sample_id"]) for row in rows}
    unexpected = sorted(set(records) - known_ids)
    if unexpected:
        raise RuntimeError(f"Checkpoint has unexpected sample IDs: {unexpected[:5]}")

    client = AsyncOpenAI(
        api_key=args.api_key,
        base_url=args.api_base,
        timeout=args.request_timeout_seconds,
        max_retries=args.client_max_retries,
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    rate_limiter = (
        RequestRateLimiter(args.requests_per_second)
        if args.requests_per_second > 0
        else None
    )
    checkpoint_lock = asyncio.Lock()
    started = time.monotonic()

    try:
        for attempt in range(1, args.retry_rounds + 1):
            round_max_tokens = min(
                args.max_tokens * (2 ** (attempt - 1)),
                args.max_retry_tokens,
            )
            pending = [
                row
                for row in rows
                if not record_is_complete(
                    records.get(str(row["sample_id"]), {})
                )
            ]
            print(
                f"method={args.method} attempt={attempt}/{args.retry_rounds} "
                f"completed={len(rows) - len(pending)} pending={len(pending)} "
                f"concurrency={args.concurrency} thinking={args.thinking} "
                f"max_tokens={round_max_tokens}",
                flush=True,
            )
            if not pending:
                break

            progress = tqdm(
                total=len(pending),
                desc=f"{args.method} round {attempt}",
                unit="case",
            )
            tasks = [
                asyncio.create_task(
                    infer_one(
                        row,
                        args=args,
                        max_tokens=round_max_tokens,
                        client=client,
                        semaphore=semaphore,
                        checkpoint_path=checkpoint_path,
                        checkpoint_lock=checkpoint_lock,
                        progress=progress,
                        attempt=attempt,
                        rate_limiter=rate_limiter,
                    )
                )
                for row in pending
            ]
            try:
                round_records = await asyncio.gather(*tasks)
            finally:
                progress.close()
            records.update(
                {str(record["sample_id"]): record for record in round_records}
            )
            if all(record_is_complete(record) for record in round_records):
                break
            if attempt < args.retry_rounds:
                await asyncio.sleep(args.retry_delay_seconds)
    finally:
        await client.close()

    predictions = materialize_predictions(rows, records)
    write_json(predictions_path, predictions)
    evaluation = population_runtime.evaluation_report(predictions)
    evaluation.update(
        {
            "method": args.method,
            "model": getattr(args, "record_model_name", None) or args.model,
            "request_model": args.model,
            "thinking": bool(args.thinking),
            "prediction_count": len(predictions),
            "generated_at": now_iso(),
        }
    )
    write_json(evaluation_path, evaluation)
    summary = {
        "method": args.method,
        "model": getattr(args, "record_model_name", None) or args.model,
        "request_model": args.model,
        "api_provider": (
            "openrouter"
            if is_openrouter_endpoint(args.api_base)
            else "openai_compatible"
        ),
        "input_path": str(Path(args.input_path).resolve()),
        "exclude_easy_conflict": bool(args.exclude_easy_conflict),
        "thinking": bool(args.thinking),
        "reasoning_effort": args.reasoning_effort,
        "max_tokens": args.max_tokens,
        "max_retry_tokens": args.max_retry_tokens,
        "temperature": args.temperature,
        "requests_per_second": args.requests_per_second,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "seed": args.seed,
        "expected": args.expected_count,
        "checkpoint_unique": len(records),
        "prediction_count": len(predictions),
        "status_counts": status_counts(records.values()),
        "token_usage": {
            field: sum(
                float((record.get("token_counts") or {}).get(field, 0) or 0)
                for record in records.values()
            )
            for field in (
                "total_tokens",
                "input_tokens",
                "output_tokens",
                "reasoning_tokens",
                "cached_input_tokens",
                "cost_usd",
            )
        },
        "condition_counts": dict(
            sorted(Counter(str(row["condition"]) for row in rows).items())
        ),
        "elapsed_seconds": time.monotonic() - started,
        "completed_at": now_iso(),
        "memory_path": (
            str(Path(args.memory_path).resolve())
            if args.method == "ours_memory"
            else None
        ),
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    if summary["status_counts"].get("ERROR", 0):
        raise RuntimeError(
            f"Inference completed with {summary['status_counts']['ERROR']} API errors"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=["vanilla_llm", "ours_memory"],
        required=True,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--record-model-name",
        help="Stable model label stored in outputs when it differs from --model.",
    )
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--api-key")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--memory-path")
    parser.add_argument("--query", choices=["hint", "nohint"], default="hint")
    parser.add_argument(
        "--input-path",
        default=str(ROOT / "data" / "MPT_v2_mix600.json"),
    )
    parser.add_argument("--exclude-easy-conflict", action="store_true")
    parser.add_argument("--expected-count", type=int, default=6508)
    parser.add_argument("--concurrency", type=int, default=256)
    parser.add_argument(
        "--requests-per-second",
        type=float,
        default=0.0,
        help="Optional global request-start rate; 0 disables pacing.",
    )
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--max-retry-tokens", type=int)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh", "max"],
        default=None,
    )
    parser.add_argument("--request-timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--client-max-retries", type=int, default=0)
    parser.add_argument("--retry-rounds", type=int, default=2)
    parser.add_argument("--retry-delay-seconds", type=float, default=5.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.method == "ours_memory" and not args.memory_path:
        parser.error("--memory-path is required for ours_memory")
    if args.max_retry_tokens is None:
        args.max_retry_tokens = args.max_tokens
    if args.max_retry_tokens < args.max_tokens:
        parser.error("--max-retry-tokens must be >= --max-tokens")
    if args.requests_per_second < 0:
        parser.error("--requests-per-second must be >= 0")
    args.api_base, args.api_key = resolve_openai_compatible_endpoint(
        provider="auto",
        base_url=args.api_base,
        api_key=args.api_key,
    )
    if not args.api_key:
        args.api_key = "EMPTY"
    return args


if __name__ == "__main__":
    asyncio.run(run(build_parser()))
