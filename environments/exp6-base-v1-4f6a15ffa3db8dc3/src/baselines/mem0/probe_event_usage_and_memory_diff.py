from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import tiktoken
from mem0 import MemoryClient

from utils_mem0 import load_chains_dataset


USAGE_KEYWORDS = (
    "token",
    "usage",
    "completion",
    "prompt_tokens",
    "input_tokens",
    "output_tokens",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe whether Mem0 cloud event responses expose construction-token "
            "usage, and record per-session memory diffs."
        )
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="/data/minseo/experiments6/data/1229_dev_6.json",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="/data/minseo/experiments6/mem0/output/probe_event_usage_and_memory_diff.jsonl",
    )
    parser.add_argument(
        "--app_id",
        type=str,
        default=f"experiments6-mem0-usage-probe-{int(time.time())}",
    )
    parser.add_argument("--start_example", type=int, default=0)
    parser.add_argument("--max_examples", type=int, default=1)
    parser.add_argument("--max_sessions_per_example", type=int, default=2)
    parser.add_argument("--retry_count", type=int, default=5)
    parser.add_argument("--retry_base_sleep", type=float, default=1.5)
    parser.add_argument("--encoding", type=str, default="cl100k_base")
    parser.add_argument(
        "--async_mode",
        action="store_true",
        help="Use Mem0 async ingestion. Default is synchronous add.",
    )
    return parser.parse_args()


def call_with_retries(fn, *args, retry_count: int, retry_base_sleep: float, **kwargs):
    last_error = None
    for attempt in range(1, retry_count + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_error = exc
            if attempt == retry_count:
                break
            time.sleep(retry_base_sleep * (2 ** (attempt - 1)))
    raise last_error


def normalize_results(response: Any) -> List[Dict[str, Any]]:
    if response is None:
        return []
    if isinstance(response, dict):
        results = response.get("results", [])
        return results if isinstance(results, list) else []
    if isinstance(response, list):
        return response
    return []


def extract_event_ids(response: Any) -> List[str]:
    event_ids: List[str] = []
    if isinstance(response, dict):
        top_level = response.get("event_id")
        if isinstance(top_level, str) and top_level:
            event_ids.append(top_level)
        for item in response.get("results", []) or []:
            if isinstance(item, dict) and item.get("event_id"):
                event_ids.append(str(item["event_id"]))
    elif isinstance(response, list):
        for item in response:
            if isinstance(item, dict) and item.get("event_id"):
                event_ids.append(str(item["event_id"]))
    return list(dict.fromkeys(event_ids))


def get_event(client: MemoryClient, event_id: str, args: argparse.Namespace) -> Dict[str, Any]:
    response = call_with_retries(
        client.client.get,
        f"/v1/event/{event_id}/",
        retry_count=args.retry_count,
        retry_base_sleep=args.retry_base_sleep,
    )
    response.raise_for_status()
    return response.json()


def wait_for_events(
    client: MemoryClient,
    event_ids: List[str],
    args: argparse.Namespace,
    timeout_seconds: float = 180.0,
) -> List[Dict[str, Any]]:
    if not event_ids:
        return []
    deadline = time.time() + timeout_seconds
    completed: Dict[str, Dict[str, Any]] = {}
    while time.time() < deadline:
        for event_id in event_ids:
            if event_id in completed:
                continue
            event = get_event(client, event_id, args)
            if event.get("status") in {"SUCCEEDED", "FAILED"}:
                completed[event_id] = event
        if len(completed) == len(event_ids):
            return [completed[event_id] for event_id in event_ids]
        time.sleep(1.0)
    raise TimeoutError(f"Timed out waiting for Mem0 events: {event_ids}")


def fetch_user_memories(
    client: MemoryClient,
    user_id: str,
    app_id: str,
    args: argparse.Namespace,
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
            retry_count=args.retry_count,
            retry_base_sleep=args.retry_base_sleep,
        )
        current = normalize_results(response)
        all_results.extend(current)
        if len(current) < page_size:
            break
        page += 1
    return all_results


def build_mem0_messages(session: Dict[str, Any]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    for turn in session.get("dialogue", []):
        role = str(turn.get("role", "")).lower()
        content = turn.get("message", "")
        if role and content:
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


def count_text_tokens(strings: Iterable[str], encoding: Any) -> int:
    return sum(len(encoding.encode(text)) for text in strings if text)


def memory_text(memory: Dict[str, Any]) -> str:
    return str(
        memory.get("memory")
        or memory.get("text")
        or memory.get("content")
        or memory.get("value")
        or ""
    )


def memory_id(memory: Dict[str, Any]) -> str:
    return str(memory.get("id") or memory.get("memory_id") or memory.get("uuid") or "")


def diff_memories(before: List[Dict[str, Any]], after: List[Dict[str, Any]]) -> Dict[str, Any]:
    before_by_id = {memory_id(item): item for item in before if memory_id(item)}
    after_by_id = {memory_id(item): item for item in after if memory_id(item)}

    added_ids = sorted(set(after_by_id) - set(before_by_id))
    removed_ids = sorted(set(before_by_id) - set(after_by_id))
    common_ids = sorted(set(before_by_id) & set(after_by_id))
    changed_ids = [
        item_id
        for item_id in common_ids
        if memory_text(before_by_id[item_id]) != memory_text(after_by_id[item_id])
    ]

    return {
        "added": [
            {"id": item_id, "memory": memory_text(after_by_id[item_id])}
            for item_id in added_ids
        ],
        "removed": [
            {"id": item_id, "memory": memory_text(before_by_id[item_id])}
            for item_id in removed_ids
        ],
        "changed": [
            {
                "id": item_id,
                "before": memory_text(before_by_id[item_id]),
                "after": memory_text(after_by_id[item_id]),
            }
            for item_id in changed_ids
        ],
    }


def find_usage_like_fields(obj: Any, path: str = "$") -> List[Dict[str, Any]]:
    found: List[Dict[str, Any]] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_path = f"{path}.{key}"
            if any(keyword in str(key).lower() for keyword in USAGE_KEYWORDS):
                found.append({"path": key_path, "value": value})
            found.extend(find_usage_like_fields(value, key_path))
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            found.extend(find_usage_like_fields(value, f"{path}[{index}]"))
    return found


def safe_raw_shape(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: safe_raw_shape(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [safe_raw_shape(value) for value in obj[:5]]
    if isinstance(obj, str):
        return obj if len(obj) <= 500 else f"{obj[:500]}..."
    return obj


def main() -> None:
    if not os.environ.get("MEM0_API_KEY"):
        raise RuntimeError("MEM0_API_KEY is not set in the environment.")

    args = parse_args()
    encoding = tiktoken.get_encoding(args.encoding)
    client = MemoryClient(api_key=os.environ["MEM0_API_KEY"])
    df = load_chains_dataset(args.input_path)
    examples = df.to_dict("records")[
        args.start_example : args.start_example + args.max_examples
    ]

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out:
        for example_offset, example in enumerate(examples):
            example_index = args.start_example + example_offset
            example_id = str(example.get("example_id", f"example_{example_index}"))
            delete_response = call_with_retries(
                client.delete_all,
                user_id=example_id,
                app_id=args.app_id,
                retry_count=args.retry_count,
                retry_base_sleep=args.retry_base_sleep,
            )
            wait_for_events(client, extract_event_ids(delete_response), args)

            for session_index, session in enumerate(
                example.get("sessions", [])[: args.max_sessions_per_example]
            ):
                messages = build_mem0_messages(session)
                before = fetch_user_memories(client, example_id, args.app_id, args)
                add_response = call_with_retries(
                    client.add,
                    messages,
                    user_id=example_id,
                    app_id=args.app_id,
                    async_mode=args.async_mode,
                    retry_count=args.retry_count,
                    retry_base_sleep=args.retry_base_sleep,
                )
                event_ids = extract_event_ids(add_response)
                events = wait_for_events(client, event_ids, args)
                after = fetch_user_memories(client, example_id, args.app_id, args)

                record = {
                    "example_index": example_index,
                    "example_id": example_id,
                    "session_index": session_index,
                    "dialogue_id": session.get("dialogue_id"),
                    "app_id": args.app_id,
                    "session_input_tokens": count_text_tokens(
                        (message.get("content", "") for message in messages),
                        encoding,
                    ),
                    "before_memory_count": len(before),
                    "after_memory_count": len(after),
                    "before_memory_tokens": count_text_tokens(
                        (memory_text(item) for item in before),
                        encoding,
                    ),
                    "after_memory_tokens": count_text_tokens(
                        (memory_text(item) for item in after),
                        encoding,
                    ),
                    "event_ids": event_ids,
                    "event_statuses": [event.get("status") for event in events],
                    "usage_like_fields": {
                        "add_response": find_usage_like_fields(add_response),
                        "events": find_usage_like_fields(events),
                    },
                    "memory_diff": diff_memories(before, after),
                    "raw_shapes": {
                        "add_response": safe_raw_shape(add_response),
                        "events": safe_raw_shape(events),
                    },
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                print(
                    f"example={example_id} session={session_index} "
                    f"events={record['event_statuses']} "
                    f"usage_fields="
                    f"{len(record['usage_like_fields']['add_response']) + len(record['usage_like_fields']['events'])} "
                    f"diff_added={len(record['memory_diff']['added'])} "
                    f"diff_changed={len(record['memory_diff']['changed'])}"
                )

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
