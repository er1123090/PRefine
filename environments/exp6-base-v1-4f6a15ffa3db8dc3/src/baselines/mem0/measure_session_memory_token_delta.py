import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import tiktoken
from mem0 import MemoryClient

from utils_mem0 import load_chains_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure how much the total stored Mem0 memory token count changes "
            "after each session is added."
        )
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="/data/minseo/experiments6/data/1229_dev_6.json",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="/data/minseo/experiments6/mem0/session_memory_token_deltas_1229_dev_6.csv",
    )
    parser.add_argument(
        "--summary_json",
        type=str,
        default="/data/minseo/experiments6/mem0/session_memory_token_deltas_1229_dev_6.summary.json",
    )
    parser.add_argument(
        "--app_id",
        type=str,
        default="experiments6-session-token-delta-20260323",
        help=(
            "App scope used for Mem0 isolation so this measurement does not mix with "
            "other memories in the same project."
        ),
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="cl100k_base",
        help="tiktoken encoding used to count stored memory tokens.",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Optional limit for quick experiments.",
    )
    parser.add_argument(
        "--start_example",
        type=int,
        default=0,
        help="Start index in the dataset (inclusive).",
    )
    parser.add_argument(
        "--end_example",
        type=int,
        default=None,
        help="End index in the dataset (exclusive).",
    )
    parser.add_argument(
        "--max_sessions_per_example",
        type=int,
        default=None,
        help="Optional per-example session limit for quick experiments.",
    )
    parser.add_argument(
        "--retry_count",
        type=int,
        default=5,
        help="Retry count for Mem0 API calls.",
    )
    parser.add_argument(
        "--retry_base_sleep",
        type=float,
        default=1.5,
        help="Base sleep in seconds for exponential backoff.",
    )
    parser.add_argument(
        "--async_mode",
        action="store_true",
        help=(
            "Use Mem0 async ingestion. Disabled by default because exact before/after "
            "measurement is cleaner with synchronous processing."
        ),
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Record the error in CSV and continue with the next session.",
    )
    parser.add_argument(
        "--skip_delete_all",
        action="store_true",
        help=(
            "Skip delete_all at the start of each example. Recommended when using a "
            "fresh app_id for this measurement run."
        ),
    )
    return parser.parse_args()


def ensure_api_key() -> str:
    api_key = os.environ.get("MEM0_API_KEY")
    if not api_key:
        raise RuntimeError("MEM0_API_KEY environment variable is not set.")
    return api_key


def call_with_retries(
    fn,
    *args,
    retry_count: int,
    retry_base_sleep: float,
    **kwargs,
):
    last_error = None
    for attempt in range(1, retry_count + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_error = exc
            if attempt == retry_count:
                break
            sleep_s = retry_base_sleep * (2 ** (attempt - 1))
            print(
                f"[Retry] {fn.__name__} failed on attempt {attempt}/{retry_count}: {exc}. "
                f"Sleeping {sleep_s:.1f}s..."
            )
            time.sleep(sleep_s)
    raise last_error


def normalize_results(response: Any) -> List[Dict[str, Any]]:
    if response is None:
        return []
    if isinstance(response, dict):
        results = response.get("results", [])
        if isinstance(results, list):
            return results
        return []
    if isinstance(response, list):
        return response
    return []


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

    # Preserve order while removing duplicates.
    return list(dict.fromkeys(event_ids))


def get_event(
    client: MemoryClient,
    event_id: str,
    retry_count: int,
    retry_base_sleep: float,
) -> Dict[str, Any]:
    response = call_with_retries(
        client.client.get,
        f"/v1/event/{event_id}/",
        retry_count=retry_count,
        retry_base_sleep=retry_base_sleep,
    )
    response.raise_for_status()
    return response.json()


def wait_for_events(
    client: MemoryClient,
    event_ids: List[str],
    retry_count: int,
    retry_base_sleep: float,
    timeout_seconds: float = 180.0,
    poll_interval_seconds: float = 1.0,
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

            event = get_event(
                client,
                event_id,
                retry_count=retry_count,
                retry_base_sleep=retry_base_sleep,
            )
            status = event.get("status")
            if status in {"SUCCEEDED", "FAILED"}:
                completed[event_id] = event
            else:
                pending.append(event_id)

        if not pending:
            return [completed[event_id] for event_id in event_ids if event_id in completed]

        time.sleep(poll_interval_seconds)

    raise TimeoutError(
        f"Timed out waiting for Mem0 events to complete: {', '.join(event_ids)}"
    )


def build_mem0_messages(session: Dict[str, Any]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    for turn in session.get("dialogue", []):
        role = str(turn.get("role", "")).lower()
        content = turn.get("message", "")
        if not role or not content:
            continue
        messages.append({"role": role, "content": content})

    api_calls = session.get("api_call", [])
    if api_calls:
        messages.append(
            {
                "role": "assistant",
                "content": f"[System Summary] API Calls executed in this session: {str(api_calls)}",
            }
        )
    return messages


def count_text_tokens(strings: Iterable[str], encoding) -> int:
    total = 0
    for text in strings:
        if not text:
            continue
        total += len(encoding.encode(text))
    return total


def fetch_user_memories(
    client: MemoryClient,
    user_id: str,
    app_id: str,
    retry_count: int,
    retry_base_sleep: float,
) -> List[Dict[str, Any]]:
    page = 1
    page_size = 1000
    all_results: List[Dict[str, Any]] = []

    while True:
        response = call_with_retries(
            client.get_all,
            filters={"AND": [{"user_id": user_id}, {"app_id": app_id}]},
            page=page,
            page_size=page_size,
            retry_count=retry_count,
            retry_base_sleep=retry_base_sleep,
        )
        current_results = normalize_results(response)
        all_results.extend(current_results)

        if len(current_results) < page_size:
            break
        page += 1

    return all_results


def count_memory_tokens(memories: List[Dict[str, Any]], encoding) -> int:
    return count_text_tokens((m.get("memory", "") for m in memories), encoding)


def maybe_trim(items: List[Any], max_count: int | None) -> List[Any]:
    if max_count is None:
        return items
    return items[:max_count]


def main() -> None:
    args = parse_args()
    api_key = ensure_api_key()

    df = load_chains_dataset(args.input_path)
    all_rows = df.to_dict("records")
    total_available_examples = len(all_rows)
    rows = all_rows[args.start_example : args.end_example]
    rows = maybe_trim(rows, args.max_examples)

    encoding = tiktoken.get_encoding(args.encoding)
    client = MemoryClient(api_key=api_key)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json = Path(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "example_index",
        "example_id",
        "session_index",
        "dialogue_id",
        "dialogue_turns",
        "api_call_count",
        "session_input_tokens",
        "before_memory_count",
        "after_memory_count",
        "before_memory_tokens",
        "after_memory_tokens",
        "delta_memory_count",
        "delta_memory_tokens",
        "add_event_ids",
        "add_event_statuses",
        "status",
        "error",
    ]

    total_examples = len(rows)
    total_sessions = 0
    success_sessions = 0
    failed_sessions = 0
    delta_sum = 0
    delta_abs_sum = 0
    max_delta = None
    min_delta = None
    started_at = time.time()

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for example_index, row in enumerate(rows):
            example_id = str(row.get("example_id", f"example_{example_index}"))
            sessions = maybe_trim(row.get("sessions", []), args.max_sessions_per_example)

            print(
                f"[Example {example_index + 1}/{total_examples}] "
                f"user_id={example_id} sessions={len(sessions)}"
            )

            if not args.skip_delete_all:
                try:
                    delete_response = call_with_retries(
                        client.delete_all,
                        user_id=example_id,
                        app_id=args.app_id,
                        retry_count=args.retry_count,
                        retry_base_sleep=args.retry_base_sleep,
                    )
                    delete_event_ids = extract_event_ids(delete_response)
                    wait_for_events(
                        client,
                        delete_event_ids,
                        retry_count=args.retry_count,
                        retry_base_sleep=args.retry_base_sleep,
                    )
                except Exception as exc:
                    print(f"[Warning] delete_all failed for user_id={example_id}: {exc}")

            for session_index, session in enumerate(sessions):
                total_sessions += 1
                dialogue_id = session.get("dialogue_id", "")
                mem0_messages = build_mem0_messages(session)
                session_input_tokens = count_text_tokens(
                    (msg.get("content", "") for msg in mem0_messages),
                    encoding,
                )

                row_out: Dict[str, Any] = {
                    "example_index": example_index,
                    "example_id": example_id,
                    "session_index": session_index,
                    "dialogue_id": dialogue_id,
                    "dialogue_turns": len(session.get("dialogue", [])),
                    "api_call_count": len(session.get("api_call", [])),
                    "session_input_tokens": session_input_tokens,
                    "before_memory_count": "",
                    "after_memory_count": "",
                    "before_memory_tokens": "",
                    "after_memory_tokens": "",
                    "delta_memory_count": "",
                    "delta_memory_tokens": "",
                    "add_event_ids": "",
                    "add_event_statuses": "",
                    "status": "PENDING",
                    "error": "",
                }

                if not mem0_messages:
                    row_out["status"] = "SKIPPED_EMPTY_SESSION"
                    writer.writerow(row_out)
                    f.flush()
                    print(
                        f"  [Session {session_index + 1}/{len(sessions)}] "
                        "skipped because no messages were produced."
                    )
                    continue

                try:
                    before_memories = fetch_user_memories(
                        client,
                        example_id,
                        args.app_id,
                        retry_count=args.retry_count,
                        retry_base_sleep=args.retry_base_sleep,
                    )
                    before_count = len(before_memories)
                    before_tokens = count_memory_tokens(before_memories, encoding)

                    add_response = call_with_retries(
                        client.add,
                        mem0_messages,
                        user_id=example_id,
                        app_id=args.app_id,
                        async_mode=args.async_mode,
                        retry_count=args.retry_count,
                        retry_base_sleep=args.retry_base_sleep,
                    )
                    add_event_ids = extract_event_ids(add_response)
                    add_events = wait_for_events(
                        client,
                        add_event_ids,
                        retry_count=args.retry_count,
                        retry_base_sleep=args.retry_base_sleep,
                    )

                    after_memories = fetch_user_memories(
                        client,
                        example_id,
                        args.app_id,
                        retry_count=args.retry_count,
                        retry_base_sleep=args.retry_base_sleep,
                    )
                    after_count = len(after_memories)
                    after_tokens = count_memory_tokens(after_memories, encoding)

                    delta_count = after_count - before_count
                    delta_tokens = after_tokens - before_tokens

                    row_out.update(
                        {
                            "before_memory_count": before_count,
                            "after_memory_count": after_count,
                            "before_memory_tokens": before_tokens,
                            "after_memory_tokens": after_tokens,
                            "delta_memory_count": delta_count,
                            "delta_memory_tokens": delta_tokens,
                            "add_event_ids": json.dumps(add_event_ids, ensure_ascii=False),
                            "add_event_statuses": json.dumps(
                                [event.get("status") for event in add_events],
                                ensure_ascii=False,
                            ),
                            "status": "OK",
                        }
                    )

                    writer.writerow(row_out)
                    f.flush()

                    success_sessions += 1
                    delta_sum += delta_tokens
                    delta_abs_sum += abs(delta_tokens)
                    max_delta = delta_tokens if max_delta is None else max(max_delta, delta_tokens)
                    min_delta = delta_tokens if min_delta is None else min(min_delta, delta_tokens)

                    print(
                        f"  [Session {session_index + 1}/{len(sessions)}] "
                        f"delta_tokens={delta_tokens:+d} "
                        f"(before={before_tokens}, after={after_tokens}, "
                        f"delta_memories={delta_count:+d})"
                    )
                except Exception as exc:
                    failed_sessions += 1
                    row_out["status"] = "ERROR"
                    row_out["error"] = str(exc)
                    writer.writerow(row_out)
                    f.flush()
                    print(
                        f"  [Session {session_index + 1}/{len(sessions)}] ERROR: {exc}"
                    )
                    if not args.continue_on_error:
                        raise

    duration_s = time.time() - started_at
    summary = {
        "input_path": args.input_path,
        "output_csv": str(output_csv),
        "encoding": args.encoding,
        "async_mode": args.async_mode,
        "app_id": args.app_id,
        "start_example": args.start_example,
        "end_example": args.end_example,
        "total_available_examples": total_available_examples,
        "examples_processed": total_examples,
        "sessions_processed": total_sessions,
        "sessions_succeeded": success_sessions,
        "sessions_failed": failed_sessions,
        "delta_token_sum": delta_sum,
        "delta_token_abs_sum": delta_abs_sum,
        "avg_delta_tokens": (delta_sum / success_sessions) if success_sessions else None,
        "avg_abs_delta_tokens": (delta_abs_sum / success_sessions) if success_sessions else None,
        "max_delta_tokens": max_delta,
        "min_delta_tokens": min_delta,
        "duration_seconds": duration_s,
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nMeasurement complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
