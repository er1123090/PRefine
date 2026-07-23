#!/usr/bin/env python3
"""Resume-safe high-reasoning memory construction for the mix600 3x3 run."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import Preference_Memory_step1_LATENTPREF_VLLM as legacy


class HighReasoningCompletions:
    def __init__(self, completions: Any, model: str) -> None:
        self._completions = completions
        self._model = model.lower()

    async def create(self, **kwargs: Any) -> Any:
        if "qwen3" in self._model:
            extra_body = dict(kwargs.pop("extra_body", {}) or {})
            chat_template_kwargs = dict(
                extra_body.get("chat_template_kwargs", {}) or {}
            )
            chat_template_kwargs["enable_thinking"] = True
            extra_body["chat_template_kwargs"] = chat_template_kwargs
            kwargs["extra_body"] = extra_body
        else:
            kwargs["reasoning_effort"] = "high"
        return await self._completions.create(**kwargs)


class HighReasoningChat:
    def __init__(self, chat: Any, model: str) -> None:
        self.completions = HighReasoningCompletions(chat.completions, model)


class HighReasoningClient:
    def __init__(self, client: AsyncOpenAI, model: str) -> None:
        self.chat = HighReasoningChat(client.chat, model)


def read_completed_ids(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL in {path} at line {line_number}"
                ) from exc
            example_id = str(record.get("example_id"))
            if example_id in completed:
                raise ValueError(f"Duplicate example_id in {path}: {example_id}")
            completed.add(example_id)
    return completed


def load_dataset_shard(
    input_path: Path,
    shard_index: int,
    num_shards: int,
) -> tuple[list[dict[str, Any]], int, int, int]:
    data = legacy.load_dataset(str(input_path))
    if num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must satisfy 0 <= index < num-shards")
    total = len(data)
    start = (total * shard_index) // num_shards
    end = (total * (shard_index + 1)) // num_shards
    return data[start:end], total, start, end


async def process_one(
    example: dict[str, Any],
    *,
    model: str,
    client: HighReasoningClient,
    semaphore: asyncio.Semaphore,
    output_path: Path,
    file_lock: asyncio.Lock,
    progress: tqdm,
) -> None:
    async with semaphore:
        aggregator = legacy.PreferenceAggregator(client, model=model)
        state = legacy.MemoryState()

        for session in example.get("sessions", []):
            state = await aggregator.update_memory(
                state,
                legacy.format_dialogue(session.get("dialogue", [])),
                session.get("api_call", []),
            )

        record = {
            "example_id": example.get("example_id", "unknown"),
            "reasoning_effort": "high",
            "final_implicit_preference": state.implicit_pref,
            "final_accumulated_api_calls": state.accumulated_api_calls,
            "total_sessions_processed": state.session_count,
            "preference_evolution_history": state.evolution_log,
        }

        async with file_lock:
            with output_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
        progress.update(1)


async def run_shard(args: argparse.Namespace) -> None:
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    shard, total, start, end = load_dataset_shard(
        input_path,
        args.shard_index,
        args.num_shards,
    )
    completed = read_completed_ids(output_path) if args.resume else set()
    if not args.resume:
        output_path.write_text("", encoding="utf-8")

    pending = [
        example
        for example in shard
        if str(example.get("example_id", "unknown")) not in completed
    ]

    print(f"model={args.model}")
    print("reasoning=high")
    print(f"shard={args.shard_index + 1}/{args.num_shards} rows={start}:{end} total={total}")
    print(f"checkpoint={output_path}")
    print(f"completed={len(completed)} pending={len(pending)}")

    if not pending:
        return

    raw_client = AsyncOpenAI(
        api_key=args.api_key,
        base_url=args.api_base,
        timeout=args.timeout,
        max_retries=args.client_retries,
    )
    client = HighReasoningClient(raw_client, args.model)
    semaphore = asyncio.Semaphore(args.concurrency)
    file_lock = asyncio.Lock()
    progress = tqdm(total=len(pending), desc="High-reasoning users", unit="user")

    tasks = [
        asyncio.create_task(
            process_one(
                example,
                model=args.model,
                client=client,
                semaphore=semaphore,
                output_path=output_path,
                file_lock=file_lock,
                progress=progress,
            )
        )
        for example in pending
    ]

    try:
        await asyncio.gather(*tasks)
    finally:
        progress.close()
        await raw_client.close()


def collect_records(shard_root: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(shard_root.glob("*/memory.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSONL in {path} at line {line_number}"
                    ) from exc
                example_id = str(record.get("example_id"))
                if example_id in records:
                    raise ValueError(
                        f"Duplicate example_id across shards: {example_id}"
                    )
                records[example_id] = record
    return records


def merge_shards(args: argparse.Namespace) -> None:
    input_path = Path(args.input).resolve()
    shard_root = Path(args.shard_root).resolve()
    output_path = Path(args.output).resolve()
    dataset = legacy.load_dataset(str(input_path))
    records = collect_records(shard_root)
    expected_ids = [str(example.get("example_id")) for example in dataset]
    missing = [example_id for example_id in expected_ids if example_id not in records]
    unexpected = sorted(set(records) - set(expected_ids))
    if missing or unexpected:
        raise RuntimeError(
            f"Cannot merge: records={len(records)} expected={len(expected_ids)} "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for example_id in expected_ids:
            handle.write(json.dumps(records[example_id], ensure_ascii=False) + "\n")
    os.replace(temporary, output_path)
    print(f"Merged {len(records)} records -> {output_path}")


def print_status(args: argparse.Namespace) -> None:
    input_path = Path(args.input).resolve()
    shard_root = Path(args.shard_root).resolve()
    dataset = legacy.load_dataset(str(input_path))
    expected = {str(example.get("example_id")) for example in dataset}
    records = collect_records(shard_root)
    print(
        json.dumps(
            {
                "expected": len(expected),
                "completed": len(records),
                "remaining": len(expected - set(records)),
                "unexpected": len(set(records) - expected),
                "complete": set(records) == expected,
            },
            sort_keys=True,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run-shard")
    run.add_argument("--input", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--api-base", required=True)
    run.add_argument("--api-key", default="EMPTY")
    run.add_argument("--num-shards", type=int, default=2)
    run.add_argument("--shard-index", type=int, required=True)
    run.add_argument("--concurrency", type=int, default=4)
    run.add_argument("--timeout", type=float, default=1800.0)
    run.add_argument("--client-retries", type=int, default=5)
    run.add_argument("--resume", action="store_true")

    merge = subparsers.add_parser("merge")
    merge.add_argument("--input", required=True)
    merge.add_argument("--shard-root", required=True)
    merge.add_argument("--output", required=True)

    status = subparsers.add_parser("status")
    status.add_argument("--input", required=True)
    status.add_argument("--shard-root", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run-shard":
        asyncio.run(run_shard(args))
    elif args.command == "merge":
        merge_shards(args)
    else:
        print_status(args)


if __name__ == "__main__":
    main()
