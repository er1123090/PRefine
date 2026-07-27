"""Add MPT sessions to managed Mem0 without memory reads or token counting."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
from typing import Any

import tqdm


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from mem0 import MemoryClient
except ImportError:
    MemoryClient = None

from methods.mem0.build_memory import (
    build_mem0_messages,
    call_with_retries,
    extract_event_ids,
    wait_for_events,
)
from methods.mem0.utils_mem0 import load_chains_dataset


DEFAULT_INPUT = ROOT / "data" / "MPT_v2_0725.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "mem0" / "MPT_v2_0725_add_only.jsonl"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def key_fingerprint(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]


def append_jsonl(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    encoded = json.dumps(row, ensure_ascii=False)
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
            handle.flush()


def load_checkpoint(path: Path) -> tuple[set[tuple[str, int]], set[str]]:
    completed_sessions: set[tuple[str, int]] = set()
    completed_examples: set[str] = set()
    if not path.exists():
        return completed_sessions, completed_examples

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") != "OK":
                continue
            example_id = row.get("example_id")
            if example_id is None:
                continue
            record_type = row.get("record_type")
            if record_type == "session":
                session_index = row.get("session_index")
                if isinstance(session_index, int):
                    completed_sessions.add((str(example_id), session_index))
            elif record_type == "example":
                completed_examples.add(str(example_id))
    return completed_sessions, completed_examples


def process_example(
    user_data: dict[str, Any],
    *,
    dataset_index: int,
    api_key: str,
    completed_sessions: set[tuple[str, int]],
    output: Path,
    output_lock: threading.Lock,
    retry_count: int,
    retry_base_sleep: float,
    event_timeout_seconds: float,
    poll_interval_seconds: float,
) -> dict[str, Any]:
    memory_client = MemoryClient(api_key=api_key)
    example_id = str(user_data.get("example_id", "unknown_user"))
    sessions = user_data.get("sessions", [])
    added_sessions = 0
    skipped_sessions = 0

    try:
        for session_index, session in enumerate(sessions, start=1):
            checkpoint_key = (example_id, session_index)
            if checkpoint_key in completed_sessions:
                skipped_sessions += 1
                continue

            messages = build_mem0_messages(session)
            event_ids: list[str] = []
            event_statuses: list[str] = []
            if messages:
                response = call_with_retries(
                    memory_client.add,
                    messages,
                    user_id=example_id,
                    retry_count=retry_count,
                    retry_base_sleep=retry_base_sleep,
                )
                event_ids = extract_event_ids(response)
                event_records = wait_for_events(
                    memory_client,
                    event_ids,
                    retry_count=retry_count,
                    retry_base_sleep=retry_base_sleep,
                    timeout_seconds=event_timeout_seconds,
                    poll_interval_seconds=poll_interval_seconds,
                )
                event_statuses = [
                    str(event.get("status", "")) for event in event_records
                ]

            append_jsonl(
                output,
                {
                    "record_type": "session",
                    "status": "OK",
                    "method": "mem0_managed_add_only",
                    "example_id": example_id,
                    "dataset_index": dataset_index,
                    "session_index": session_index,
                    "dialogue_id": session.get("dialogue_id"),
                    "message_count": len(messages),
                    "add_event_ids": event_ids,
                    "add_event_statuses": event_statuses,
                    "completed_at": utc_now(),
                },
                output_lock,
            )
            added_sessions += 1

        result = {
            "record_type": "example",
            "status": "OK",
            "method": "mem0_managed_add_only",
            "example_id": example_id,
            "dataset_index": dataset_index,
            "total_sessions": len(sessions),
            "added_sessions_this_run": added_sessions,
            "skipped_sessions_from_checkpoint": skipped_sessions,
            "completed_at": utc_now(),
        }
        append_jsonl(output, result, output_lock)
        return result
    finally:
        close = getattr(getattr(memory_client, "client", None), "close", None)
        if callable(close):
            close()


def run_ingestion(
    *,
    input_path: Path,
    output: Path,
    concurrency: int,
    resume: bool,
    start_example: int,
    end_example: int | None,
    max_examples: int | None,
    retry_count: int,
    retry_base_sleep: float,
    event_timeout_seconds: float,
    poll_interval_seconds: float,
) -> dict[str, Any]:
    if MemoryClient is None:
        raise RuntimeError("mem0ai is required for managed Mem0 construction")
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")

    api_key = os.environ.get("MEM0_API_KEY")
    if not api_key:
        raise RuntimeError("MEM0_API_KEY is required")

    input_path = input_path.resolve()
    output = output.resolve()
    errors_output = output.with_suffix(output.suffix + ".errors.jsonl")
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    output.parent.mkdir(parents=True, exist_ok=True)

    fingerprint = key_fingerprint(api_key)
    manifest = {
        "method": "mem0_managed_add_only",
        "input_path": str(input_path),
        "dataset_sha256": file_sha256(input_path),
        "output": str(output),
        "errors_output": str(errors_output),
        "api_key_fingerprint": fingerprint,
        "token_counting": False,
        "memory_reads": False,
        "started_at": utc_now(),
        "status": "RUNNING",
    }

    if resume and manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("api_key_fingerprint") != fingerprint:
            raise RuntimeError(
                "Refusing to resume add-only output with a different Mem0 API key"
            )
        if previous.get("dataset_sha256") != manifest["dataset_sha256"]:
            raise RuntimeError(
                "Refusing to resume add-only output with a different dataset"
            )
    elif not resume:
        output.write_text("", encoding="utf-8")
        errors_output.write_text("", encoding="utf-8")

    completed_sessions, completed_examples = (
        load_checkpoint(output) if resume else (set(), set())
    )
    dataframe = load_chains_dataset(str(input_path))
    indexed_rows = list(enumerate(dataframe.to_dict("records")))
    indexed_rows = indexed_rows[start_example:end_example]
    if max_examples is not None:
        indexed_rows = indexed_rows[:max_examples]
    pending = [
        (dataset_index, row)
        for dataset_index, row in indexed_rows
        if str(row.get("example_id", "unknown_user")) not in completed_examples
    ]

    manifest.update(
        {
            "selected_examples": len(indexed_rows),
            "completed_examples_before_run": len(indexed_rows) - len(pending),
            "pending_examples_before_run": len(pending),
            "completed_sessions_before_run": len(completed_sessions),
            "concurrency": concurrency,
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        "Starting managed Mem0 add-only ingestion: "
        f"selected={len(indexed_rows)} pending={len(pending)} "
        f"completed_sessions={len(completed_sessions)} "
        f"concurrency={concurrency}",
        flush=True,
    )

    output_lock = threading.Lock()
    succeeded = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(
                process_example,
                row,
                dataset_index=dataset_index,
                api_key=api_key,
                completed_sessions=completed_sessions,
                output=output,
                output_lock=output_lock,
                retry_count=retry_count,
                retry_base_sleep=retry_base_sleep,
                event_timeout_seconds=event_timeout_seconds,
                poll_interval_seconds=poll_interval_seconds,
            ): (dataset_index, str(row.get("example_id", "unknown_user")))
            for dataset_index, row in pending
        }
        for future in tqdm.tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Mem0 add-only",
        ):
            dataset_index, example_id = futures[future]
            try:
                future.result()
                succeeded += 1
            except Exception as exc:
                failed += 1
                append_jsonl(
                    errors_output,
                    {
                        "record_type": "example_error",
                        "status": "ERROR",
                        "method": "mem0_managed_add_only",
                        "example_id": example_id,
                        "dataset_index": dataset_index,
                        "error": f"{type(exc).__name__}: {exc}",
                        "failed_at": utc_now(),
                    },
                    output_lock,
                )

    manifest.update(
        {
            "status": "COMPLETE" if failed == 0 else "FAILED",
            "succeeded_examples_this_run": succeeded,
            "failed_examples_this_run": failed,
            "finished_at": utc_now(),
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if failed:
        raise RuntimeError(f"Managed Mem0 add-only run had {failed} failures")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--start_example", type=int, default=0)
    parser.add_argument("--end_example", type=int, default=None)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--retry_count", type=int, default=5)
    parser.add_argument("--retry_base_sleep", type=float, default=1.0)
    parser.add_argument("--event_timeout_seconds", type=float, default=300.0)
    parser.add_argument("--poll_interval_seconds", type=float, default=1.0)
    return parser


def main() -> None:
    manifest = run_ingestion(**vars(build_parser().parse_args()))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
