import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from mem0 import MemoryClient
except ImportError:
    MemoryClient = None

from methods.mem0.utils_mem0 import load_chains_dataset
from src.construction_usage import (
    begin_usage_collection,
    end_usage_collection,
    record_response_usage,
    set_usage_session,
)
from src.token_measurement import (
    count_texts_tokens,
    encoding_metadata,
    memory_construction_lower_bound,
)


def normalize_mem0_memories(response: Any) -> List[Dict[str, Any]]:
    if isinstance(response, list):
        return [item for item in response if isinstance(item, dict)]
    if isinstance(response, dict):
        for key in ("results", "memories", "data"):
            items = response.get(key)
            if isinstance(items, list):
                return [item for item in items if isinstance(item, dict)]
    return []


def memory_texts(memories: List[Dict[str, Any]]) -> List[str]:
    rendered = []
    for memory in memories:
        value = (
            memory.get("memory")
            or memory.get("content")
            or memory.get("text")
        )
        if value:
            rendered.append(str(value))
    return rendered


def build_mem0_messages(session: Dict[str, Any]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    for turn in session.get("dialogue", []):
        role = str(turn.get("role", "")).lower()
        content = turn.get("message", "")
        if role and content:
            messages.append({"role": role, "content": str(content)})

    api_calls = session.get("api_call", [])
    if api_calls:
        messages.append(
            {
                "role": "assistant",
                "content": (
                    "[System Summary] API Calls executed in this session: "
                    f"{str(api_calls)}"
                ),
            }
        )
    return messages


def call_with_retries(
    function,
    *args,
    retry_count: int,
    retry_base_sleep: float,
    **kwargs,
):
    for attempt in range(retry_count + 1):
        try:
            return function(*args, **kwargs)
        except Exception:
            if attempt >= retry_count:
                raise
            time.sleep(retry_base_sleep * (2**attempt))
    raise RuntimeError("unreachable retry state")


def extract_event_ids(response: Any) -> List[str]:
    event_ids: List[str] = []
    if isinstance(response, dict):
        top_level_event_id = response.get("event_id")
        if isinstance(top_level_event_id, str) and top_level_event_id:
            event_ids.append(top_level_event_id)
        results = response.get("results", [])
        if isinstance(results, list):
            for item in results:
                if not isinstance(item, dict):
                    continue
                event_id = item.get("event_id")
                if isinstance(event_id, str) and event_id:
                    event_ids.append(event_id)
    elif isinstance(response, list):
        for item in response:
            if not isinstance(item, dict):
                continue
            event_id = item.get("event_id")
            if isinstance(event_id, str) and event_id:
                event_ids.append(event_id)
    return list(dict.fromkeys(event_ids))


def wait_for_events(
    client: Any,
    event_ids: List[str],
    *,
    retry_count: int,
    retry_base_sleep: float,
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> List[Dict[str, Any]]:
    if not event_ids:
        return []

    deadline = time.time() + timeout_seconds
    completed: Dict[str, Dict[str, Any]] = {}
    while time.time() < deadline:
        pending = []
        for event_id in event_ids:
            if event_id in completed:
                continue
            response = call_with_retries(
                client.client.get,
                f"/v1/event/{event_id}/",
                retry_count=retry_count,
                retry_base_sleep=retry_base_sleep,
            )
            response.raise_for_status()
            event = response.json()
            status = str(event.get("status", "")).upper()
            if status in {"SUCCEEDED", "SUCCESS", "COMPLETED"}:
                completed[event_id] = event
            elif status in {"FAILED", "ERROR"}:
                raise RuntimeError(
                    f"Mem0 event failed: event_id={event_id} payload={event}"
                )
            else:
                pending.append(event_id)
        if not pending:
            return [completed[event_id] for event_id in event_ids]
        time.sleep(poll_interval_seconds)

    raise TimeoutError(
        f"Timed out waiting for Mem0 events: {', '.join(event_ids)}"
    )


def fetch_user_memories(
    client: Any,
    user_id: str,
    *,
    retry_count: int,
    retry_base_sleep: float,
    page_size: int = 100,
) -> List[Dict[str, Any]]:
    page = 1
    memories: List[Dict[str, Any]] = []
    while True:
        response = call_with_retries(
            client.get_all,
            filters={"user_id": user_id},
            page=page,
            page_size=page_size,
            retry_count=retry_count,
            retry_base_sleep=retry_base_sleep,
        )
        current = normalize_mem0_memories(response)
        memories.extend(current)
        if not isinstance(response, dict) or not response.get("next"):
            break
        page += 1
    return memories


def wait_until_empty(
    client: Any,
    user_id: str,
    *,
    retry_count: int,
    retry_base_sleep: float,
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        memories = fetch_user_memories(
            client,
            user_id,
            retry_count=retry_count,
            retry_base_sleep=retry_base_sleep,
        )
        if not memories:
            return
        time.sleep(poll_interval_seconds)
    raise TimeoutError(f"Timed out waiting for Mem0 delete_all: user_id={user_id}")


def completed_example_ids(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") == "OK" and row.get("example_id") is not None:
                completed.add(str(row["example_id"]))
    return completed


def process_example(
    user_data: Dict[str, Any],
    *,
    dataset_index: int,
    api_key: str,
    token_encoding: str,
    retry_count: int,
    retry_base_sleep: float,
    event_timeout_seconds: float,
    poll_interval_seconds: float,
) -> Dict[str, Any]:
    memory_client = MemoryClient(api_key=api_key)
    user_id = str(user_data.get("example_id", "unknown_user"))
    usage_token = begin_usage_collection()
    construction_token_usage: Dict[str, Any]
    session_exports: List[Dict[str, Any]] = []
    final_memories: List[Dict[str, Any]] = []

    try:
        call_with_retries(
            memory_client.delete_all,
            user_id=user_id,
            retry_count=retry_count,
            retry_base_sleep=retry_base_sleep,
        )
        wait_until_empty(
            memory_client,
            user_id,
            retry_count=retry_count,
            retry_base_sleep=retry_base_sleep,
            timeout_seconds=event_timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )

        previous_memory_tokens = 0
        sessions = user_data.get("sessions", [])
        for session_index, session in enumerate(sessions, start=1):
            set_usage_session(session_index)
            mem0_messages = build_mem0_messages(session)
            local_input_tokens = count_texts_tokens(
                (
                    f"{message.get('role', '')}: {message.get('content', '')}"
                    for message in mem0_messages
                ),
                token_encoding,
            )

            add_usage = None
            event_ids: List[str] = []
            event_statuses: List[str] = []
            if mem0_messages:
                try:
                    add_response = call_with_retries(
                        memory_client.add,
                        mem0_messages,
                        user_id=user_id,
                        retry_count=retry_count,
                        retry_base_sleep=retry_base_sleep,
                    )
                    add_usage = record_response_usage(
                        add_response,
                        component="mem0_add",
                        provider="mem0",
                        model="managed",
                    )
                    event_ids = extract_event_ids(add_response)
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
                except Exception as exc:
                    raise RuntimeError(
                        "Mem0 session construction failed: "
                        f"user_id={user_id} session_index={session_index} "
                        f"dialogue_id={session.get('dialogue_id')} "
                        f"event_ids={event_ids}: {exc}"
                    ) from exc

            final_memories = fetch_user_memories(
                memory_client,
                user_id,
                retry_count=retry_count,
                retry_base_sleep=retry_base_sleep,
            )
            after_memory_tokens = count_texts_tokens(
                memory_texts(final_memories),
                token_encoding,
            )
            lower_bound = memory_construction_lower_bound(
                local_input_tokens,
                previous_memory_tokens,
                after_memory_tokens,
            )
            session_exports.append(
                {
                    "session_index": session_index,
                    "dialogue_id": session.get("dialogue_id"),
                    "local_construction_input_tokens": local_input_tokens,
                    "memory_count_before_session": (
                        session_exports[-1]["memory_count_after_session"]
                        if session_exports
                        else 0
                    ),
                    "memory_count_after_session": len(final_memories),
                    "stored_memory_tokens_before_session": previous_memory_tokens,
                    "stored_memory_tokens_after_session": after_memory_tokens,
                    **lower_bound,
                    "mem0_add_provider_usage": add_usage,
                    "add_event_ids": event_ids,
                    "add_event_statuses": event_statuses,
                    "memory_snapshot_error": None,
                }
            )
            if after_memory_tokens is not None:
                previous_memory_tokens = after_memory_tokens
    finally:
        construction_token_usage = end_usage_collection(usage_token)
        close = getattr(getattr(memory_client, "client", None), "close", None)
        if callable(close):
            close()

    input_lower_bound = sum(
        int(item["construction_input_tokens_lower_bound"])
        for item in session_exports
        if item.get("construction_input_tokens_lower_bound") is not None
    )
    output_lower_bound = sum(
        int(item["construction_output_tokens_lower_bound"])
        for item in session_exports
        if item.get("construction_output_tokens_lower_bound") is not None
    )
    return {
        "example_id": user_id,
        "dataset_index": dataset_index,
        "method": "mem0",
        "status": "OK",
        "total_sessions_processed": len(user_data.get("sessions", [])),
        "session_exports": session_exports,
        "local_construction_input_tokens": input_lower_bound,
        "construction_input_tokens_lower_bound": input_lower_bound,
        "construction_output_tokens_lower_bound": output_lower_bound,
        "construction_total_tokens_lower_bound": (
            input_lower_bound + output_lower_bound
        ),
        "memory_count_final": len(final_memories),
        "stored_memory_tokens_final": (
            session_exports[-1]["stored_memory_tokens_after_session"]
            if session_exports
            else 0
        ),
        "memory_snapshot": final_memories,
        "construction_token_usage": construction_token_usage,
        "token_counts": construction_token_usage["summary"],
        **encoding_metadata(token_encoding),
    }


def run_ingestion(
    input_path: str,
    metrics_output: str,
    token_encoding: str = "cl100k_base",
    *,
    start_example: int = 0,
    end_example: int | None = None,
    max_examples: int | None = None,
    concurrency: int = 1,
    resume: bool = False,
    retry_count: int = 5,
    retry_base_sleep: float = 1.0,
    event_timeout_seconds: float = 180.0,
    poll_interval_seconds: float = 1.0,
):
    if MemoryClient is None:
        raise RuntimeError("mem0ai is required for Mem0 memory construction")
    api_key = os.environ.get("MEM0_API_KEY")
    if not api_key:
        raise RuntimeError("MEM0_API_KEY is required for Mem0 construction")
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")

    df = load_chains_dataset(input_path)
    output = Path(metrics_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    errors_output = output.with_suffix(output.suffix + ".errors.jsonl")
    if not resume:
        output.write_text("", encoding="utf-8")
        errors_output.write_text("", encoding="utf-8")

    completed = completed_example_ids(output) if resume else set()
    indexed_rows = list(enumerate(df.to_dict("records")))
    indexed_rows = indexed_rows[start_example:end_example]
    if max_examples is not None:
        indexed_rows = indexed_rows[:max_examples]
    pending = [
        (dataset_index, row)
        for dataset_index, row in indexed_rows
        if str(row.get("example_id", "unknown_user")) not in completed
    ]
    print(
        f"Starting Mem0 ingestion: selected={len(indexed_rows)} "
        f"completed={len(indexed_rows) - len(pending)} pending={len(pending)} "
        f"concurrency={concurrency}"
    )

    success_count = 0
    failure_count = 0
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(
                process_example,
                row,
                dataset_index=dataset_index,
                api_key=api_key,
                token_encoding=token_encoding,
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
            desc="Ingesting Memories",
        ):
            dataset_index, user_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                failure_count += 1
                error_record = {
                    "dataset_index": dataset_index,
                    "example_id": user_id,
                    "status": "ERROR",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                with errors_output.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(error_record, ensure_ascii=False) + "\n"
                    )
                continue

            success_count += 1
            with output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "selected": len(indexed_rows),
                "skipped_completed": len(indexed_rows) - len(pending),
                "succeeded": success_count,
                "failed": failure_count,
                "metrics_output": str(output),
                "errors_output": str(errors_output),
            },
            ensure_ascii=False,
        )
    )
    if failure_count:
        raise RuntimeError(
            f"Mem0 ingestion completed with {failure_count} failed examples"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="/data/minseo/experiment8/data/MPT_v2_mix600.json")
    parser.add_argument(
        "--metrics_output",
        default="/data/minseo/experiment8/outputs/mem0/construction_metrics.jsonl",
    )
    parser.add_argument("--token_encoding", default="cl100k_base")
    parser.add_argument("--start_example", type=int, default=0)
    parser.add_argument("--end_example", type=int, default=None)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry_count", type=int, default=5)
    parser.add_argument("--retry_base_sleep", type=float, default=1.0)
    parser.add_argument("--event_timeout_seconds", type=float, default=180.0)
    parser.add_argument("--poll_interval_seconds", type=float, default=1.0)
    args = parser.parse_args()

    run_ingestion(
        args.input_path,
        args.metrics_output,
        args.token_encoding,
        start_example=args.start_example,
        end_example=args.end_example,
        max_examples=args.max_examples,
        concurrency=args.concurrency,
        resume=args.resume,
        retry_count=args.retry_count,
        retry_base_sleep=args.retry_base_sleep,
        event_timeout_seconds=args.event_timeout_seconds,
        poll_interval_seconds=args.poll_interval_seconds,
    )
