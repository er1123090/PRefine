"""Build Experiment8 Mem0 artifacts with official OSS Mem0 and local vLLM."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable, Iterable

import tqdm


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.mem0_local.runtime import (
    DEFAULT_UPSTREAM_PATH,
    build_mem0_config,
    check_vllm_endpoint,
    create_memory,
    upstream_metadata,
)
from src.construction_usage import (
    begin_usage_collection,
    current_usage_report,
    end_usage_collection,
    set_usage_session,
)
from src.token_measurement import (
    count_texts_tokens,
    encoding_metadata,
    memory_construction_lower_bound,
)


DEFAULT_INPUT = ROOT / "data" / "MPT_v2_0725.json"
DEFAULT_OUTPUT = (
    ROOT / "outputs" / "mem0_local" / "MPT_v2_0725_construction.jsonl"
)
DEFAULT_STORE = ROOT / "outputs" / "mem0_local" / "MPT_v2_0725_qdrant"
DEFAULT_HISTORY = (
    ROOT / "outputs" / "mem0_local" / "MPT_v2_0725_history.sqlite"
)


def normalize_memories(response: Any) -> list[dict[str, Any]]:
    if isinstance(response, list):
        return [item for item in response if isinstance(item, dict)]
    if isinstance(response, dict):
        for key in ("results", "memories", "data"):
            items = response.get(key)
            if isinstance(items, list):
                return [item for item in items if isinstance(item, dict)]
    return []


def memory_texts(memories: Iterable[dict[str, Any]]) -> list[str]:
    rendered: list[str] = []
    for memory in memories:
        value = memory.get("memory") or memory.get("content") or memory.get("text")
        if value:
            rendered.append(str(value))
    return rendered


def build_messages(session: dict[str, Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
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


def load_dataset_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = [
            json.loads(line)
            for line in text.splitlines()
            if line.strip()
        ]
    if isinstance(payload, dict) and isinstance(payload.get("dataset"), list):
        payload = payload["dataset"]
    if not isinstance(payload, list):
        raise ValueError("Dataset must be a JSON array or JSONL records")
    invalid = [
        index
        for index, item in enumerate(payload)
        if not isinstance(item, dict)
    ]
    if invalid:
        raise ValueError(f"Dataset contains non-object rows: {invalid[:10]}")
    return payload


def call_with_retries(
    function: Callable[..., Any],
    *args: Any,
    retry_count: int,
    retry_base_sleep: float,
    **kwargs: Any,
) -> Any:
    for attempt in range(retry_count + 1):
        try:
            return function(*args, **kwargs)
        except Exception:
            if attempt >= retry_count:
                raise
            time.sleep(retry_base_sleep * (2**attempt))
    raise RuntimeError("unreachable retry state")


def fetch_user_memories(memory: Any, user_id: str) -> list[dict[str, Any]]:
    return normalize_memories(
        memory.get_all(filters={"user_id": user_id}, top_k=10_000)
    )


def clear_user_state(memory: Any, user_id: str) -> None:
    """Clear vectors and Mem0 v3's rolling-message context for a clean rerun."""
    if fetch_user_memories(memory, user_id):
        memory.delete_all(user_id=user_id)

    database = getattr(memory, "db", None)
    connection = getattr(database, "connection", None)
    lock = getattr(database, "_lock", None)
    if connection is None or lock is None:
        raise RuntimeError("Official Mem0 SQLite state is unavailable")
    session_scope = f"user_id={user_id}"
    with lock:
        connection.execute(
            "DELETE FROM messages WHERE session_scope = ?",
            (session_scope,),
        )
        connection.commit()


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


def _optional_sum(values: Iterable[int | None]) -> int | None:
    rows = list(values)
    if any(value is None for value in rows):
        return None
    return sum(int(value) for value in rows if value is not None)


def _session_usage(session_index: int) -> dict[str, Any]:
    report = current_usage_report()
    return report.get("by_session", {}).get(
        str(session_index),
        {
            "summary": {
                "input_tokens": 0,
                "cached_input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "total_tokens": 0,
                "call_count": 0,
                "usage_available_calls": 0,
                "usage_missing_calls": 0,
            },
            "by_component": {},
        },
    )


def process_example(
    user_data: dict[str, Any],
    *,
    dataset_index: int,
    memory: Any,
    token_encoding: str,
    retry_count: int,
    retry_base_sleep: float,
    snapshot_mode: str,
    configuration_fingerprint: str,
    upstream: dict[str, Any],
    max_sessions: int | None = None,
) -> dict[str, Any]:
    user_id = str(user_data.get("example_id", "unknown_user"))
    usage_token = begin_usage_collection()
    session_exports: list[dict[str, Any]] = []
    final_memories: list[dict[str, Any]] = []
    final_memory_tokens: int | None = None
    collect_session_state = snapshot_mode == "all"
    try:
        call_with_retries(
            clear_user_state,
            memory,
            user_id,
            retry_count=retry_count,
            retry_base_sleep=retry_base_sleep,
        )

        previous_memory_tokens = 0
        previous_memory_count = 0
        sessions = list(user_data.get("sessions", []))
        if max_sessions is not None:
            sessions = sessions[:max_sessions]
        for session_index, session in enumerate(sessions, start=1):
            set_usage_session(session_index)
            messages = build_messages(session)
            local_input_tokens = count_texts_tokens(
                (
                    f"{message.get('role', '')}: {message.get('content', '')}"
                    for message in messages
                ),
                token_encoding,
            )

            add_response: Any = {"results": []}
            if messages:
                add_response = call_with_retries(
                    memory.add,
                    messages,
                    user_id=user_id,
                    metadata={
                        "experiment": "experiment8",
                        "dataset_index": dataset_index,
                        "session_index": session_index,
                        "dialogue_id": str(session.get("dialogue_id", "")),
                    },
                    retry_count=retry_count,
                    retry_base_sleep=retry_base_sleep,
                )

            if collect_session_state:
                final_memories = call_with_retries(
                    fetch_user_memories,
                    memory,
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
                memory_count_before_session: int | None = previous_memory_count
                memory_count_after_session: int | None = len(final_memories)
                memory_count_delta: int | None = (
                    len(final_memories) - previous_memory_count
                )
                stored_memory_tokens_before_session: int | None = (
                    previous_memory_tokens
                )
                stored_memory_tokens_after_session: int | None = (
                    after_memory_tokens
                )
                session_state_accounting = "full_snapshot"
            else:
                after_memory_tokens = None
                lower_bound = {
                    "construction_input_tokens_lower_bound": local_input_tokens,
                    "construction_output_tokens_lower_bound": None,
                    "construction_total_tokens_lower_bound": None,
                }
                memory_count_before_session = None
                memory_count_after_session = None
                memory_count_delta = None
                stored_memory_tokens_before_session = None
                stored_memory_tokens_after_session = None
                session_state_accounting = "deferred_to_final_snapshot"

            export: dict[str, Any] = {
                "session_index": session_index,
                "dialogue_id": session.get("dialogue_id"),
                "local_construction_input_tokens": local_input_tokens,
                "memory_count_before_session": memory_count_before_session,
                "memory_count_after_session": memory_count_after_session,
                "memory_count_delta": memory_count_delta,
                "stored_memory_tokens_before_session": (
                    stored_memory_tokens_before_session
                ),
                "stored_memory_tokens_after_session": (
                    stored_memory_tokens_after_session
                ),
                **lower_bound,
                "mem0_add_results": normalize_memories(add_response),
                "llm_usage": _session_usage(session_index),
                "memory_snapshot_error": None,
                "session_state_accounting": session_state_accounting,
            }
            if snapshot_mode == "all":
                export["memory_snapshot_after_session"] = final_memories
            session_exports.append(export)
            if collect_session_state:
                previous_memory_count = len(final_memories)
                previous_memory_tokens = after_memory_tokens

        if snapshot_mode == "final":
            final_memories = call_with_retries(
                fetch_user_memories,
                memory,
                user_id,
                retry_count=retry_count,
                retry_base_sleep=retry_base_sleep,
            )
            final_memory_tokens = count_texts_tokens(
                memory_texts(final_memories),
                token_encoding,
            )
        elif snapshot_mode == "all":
            final_memory_tokens = previous_memory_tokens
    finally:
        construction_token_usage = end_usage_collection(usage_token)

    input_lower_bound = _optional_sum(
        item.get("construction_input_tokens_lower_bound")
        for item in session_exports
    )
    output_lower_bound = _optional_sum(
        item.get("construction_output_tokens_lower_bound")
        for item in session_exports
    )
    if snapshot_mode == "final":
        output_lower_bound = final_memory_tokens
    total_lower_bound = (
        input_lower_bound + output_lower_bound
        if input_lower_bound is not None and output_lower_bound is not None
        else None
    )
    result = {
        "example_id": user_id,
        "dataset_index": dataset_index,
        "method": "mem0_local",
        "backend": "official_mem0_oss_vllm",
        "status": "OK",
        "configuration_fingerprint": configuration_fingerprint,
        "upstream": upstream,
        "total_sessions_processed": len(session_exports),
        "session_exports": session_exports,
        "local_construction_input_tokens": input_lower_bound,
        "construction_input_tokens_lower_bound": input_lower_bound,
        "construction_output_tokens_lower_bound": output_lower_bound,
        "construction_total_tokens_lower_bound": total_lower_bound,
        "memory_count_final": (
            len(final_memories) if snapshot_mode != "none" else None
        ),
        "stored_memory_tokens_final": final_memory_tokens,
        "memory_snapshot": final_memories if snapshot_mode != "none" else None,
        "construction_token_usage": construction_token_usage,
        "token_counts": construction_token_usage["summary"],
        "construction_accounting": {
            "provider_usage": (
                "Raw vLLM usage from every official Mem0 LLM call."
            ),
            "lower_bound": (
                "session message tokens + stored memory growth from per-session "
                "snapshots, or final stored memory tokens in final-snapshot mode"
            ),
            "embedding_usage_included": False,
            "session_state_accounting": (
                "full_snapshot"
                if collect_session_state
                else "deferred_to_final_snapshot"
            ),
        },
        **encoding_metadata(token_encoding),
    }
    return result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _public_config(config: dict[str, Any]) -> dict[str, Any]:
    clean = json.loads(json.dumps(config))
    embedder = clean.get("embedder", {}).get("config", {})
    if "api_key" in embedder:
        embedder["api_key"] = "<redacted>"
    return clean


def close_memory(memory: Any) -> None:
    for candidate in (
        getattr(getattr(memory, "llm", None), "client", None),
        getattr(getattr(memory, "vector_store", None), "client", None),
        getattr(memory, "db", None),
    ):
        close = getattr(candidate, "close", None)
        if callable(close):
            close()


def run_ingestion(
    *,
    input_path: Path,
    metrics_output: Path,
    vector_store_path: Path,
    history_db_path: Path,
    upstream_repo_path: Path,
    base_url: str,
    model: str,
    reasoning_effort: str,
    disable_response_format: bool,
    max_tokens: int,
    embedding_provider: str,
    embedding_model: str,
    embedding_dims: int,
    embedding_api_key: str | None,
    embedding_base_url: str | None,
    collection_name: str,
    token_encoding: str,
    start_example: int,
    end_example: int | None,
    max_examples: int | None,
    max_sessions: int | None,
    concurrency: int,
    resume: bool,
    retry_count: int,
    retry_base_sleep: float,
    snapshot_mode: str,
    skip_preflight: bool,
    dry_run: bool,
) -> dict[str, Any]:
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    if max_sessions is not None and max_sessions < 1:
        raise ValueError("max_sessions must be positive")
    if snapshot_mode not in {"all", "final", "none"}:
        raise ValueError("snapshot_mode must be all, final, or none")

    input_path = input_path.resolve()
    metrics_output = metrics_output.resolve()
    vector_store_path = vector_store_path.resolve()
    history_db_path = history_db_path.resolve()
    upstream = upstream_metadata(upstream_repo_path)
    if not upstream["checkout_present"] or not upstream["matches_lock"]:
        raise RuntimeError(
            "Official Mem0 checkout is absent or does not match UPSTREAM.json. "
            "Run `python methods/mem0_local/bootstrap_upstream.py`."
        )

    config = build_mem0_config(
        model=model,
        base_url=base_url,
        vector_store_path=vector_store_path,
        history_db_path=history_db_path,
        collection_name=collection_name,
        max_tokens=max_tokens,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_dims=embedding_dims,
        embedding_api_key=embedding_api_key,
        embedding_base_url=embedding_base_url,
    )
    dataset_sha256 = file_sha256(input_path)
    experiment_config = {
        "dataset_sha256": dataset_sha256,
        "mem0_config": _public_config(config),
        "reasoning_effort": reasoning_effort,
        "disable_response_format": disable_response_format,
        "local_vector_store_locking": True,
        "max_sessions": max_sessions,
        "token_encoding": token_encoding,
        "snapshot_mode": snapshot_mode,
        "upstream_commit": upstream["checkout_commit"],
    }
    configuration_fingerprint = stable_fingerprint(experiment_config)
    manifest_path = metrics_output.with_suffix(
        metrics_output.suffix + ".manifest.json"
    )
    errors_output = metrics_output.with_suffix(
        metrics_output.suffix + ".errors.jsonl"
    )

    indexed_rows = list(enumerate(load_dataset_records(input_path)))
    indexed_rows = indexed_rows[start_example:end_example]
    if max_examples is not None:
        indexed_rows = indexed_rows[:max_examples]

    manifest = {
        "method": "mem0_local",
        "status": "DRY_RUN" if dry_run else "RUNNING",
        "input_path": str(input_path),
        "metrics_output": str(metrics_output),
        "errors_output": str(errors_output),
        "vector_store_path": str(vector_store_path),
        "history_db_path": str(history_db_path),
        "selected_examples": len(indexed_rows),
        "configuration_fingerprint": configuration_fingerprint,
        "experiment_config": experiment_config,
        "upstream": upstream,
    }
    if dry_run:
        return manifest

    if not skip_preflight:
        manifest["available_vllm_models"] = check_vllm_endpoint(
            base_url,
            expected_model=model,
        )

    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    vector_store_path.parent.mkdir(parents=True, exist_ok=True)
    history_db_path.parent.mkdir(parents=True, exist_ok=True)
    if resume and manifest_path.exists():
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        previous_fingerprint = previous_manifest.get("configuration_fingerprint")
        if previous_fingerprint != configuration_fingerprint:
            raise RuntimeError(
                "Resume configuration mismatch: "
                f"previous={previous_fingerprint} current={configuration_fingerprint}"
            )
    elif not resume:
        metrics_output.write_text("", encoding="utf-8")
        errors_output.write_text("", encoding="utf-8")

    completed = completed_example_ids(metrics_output) if resume else set()
    pending = [
        (dataset_index, row)
        for dataset_index, row in indexed_rows
        if str(row.get("example_id", "unknown_user")) not in completed
    ]
    manifest.update(
        {
            "completed_before_run": len(indexed_rows) - len(pending),
            "pending_before_run": len(pending),
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    memory = create_memory(
        config,
        repo_path=upstream_repo_path,
        reasoning_effort=reasoning_effort,
        disable_response_format=disable_response_format,
    )
    success_count = 0
    failure_count = 0
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(
                process_example,
                row,
                dataset_index=dataset_index,
                memory=memory,
                token_encoding=token_encoding,
                retry_count=retry_count,
                retry_base_sleep=retry_base_sleep,
                snapshot_mode=snapshot_mode,
                configuration_fingerprint=configuration_fingerprint,
                upstream=upstream,
                max_sessions=max_sessions,
            ): (dataset_index, str(row.get("example_id", "unknown_user")))
            for dataset_index, row in pending
        }
        for future in tqdm.tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Building local Mem0",
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
                    "configuration_fingerprint": configuration_fingerprint,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                with errors_output.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(error_record, ensure_ascii=False) + "\n"
                    )
                continue

            success_count += 1
            with metrics_output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    close_memory(memory)

    manifest.update(
        {
            "status": "COMPLETE" if failure_count == 0 else "FAILED",
            "succeeded_this_run": success_count,
            "failed_this_run": failure_count,
            "completed_total": len(completed) + success_count,
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if failure_count:
        raise RuntimeError(
            f"Local Mem0 construction completed with {failure_count} failures"
        )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--metrics_output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--vector_store_path", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--history_db_path", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument(
        "--upstream_repo_path",
        type=Path,
        default=DEFAULT_UPSTREAM_PATH,
    )
    parser.add_argument("--base_url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="gpt-oss-20b")
    parser.add_argument(
        "--reasoning_effort",
        choices=["low", "medium", "high"],
        default="low",
    )
    parser.add_argument(
        "--disable_response_format",
        action="store_true",
        help=(
            "Do not send Mem0's json_object constraint to local backends. "
            "Useful for GPT-OSS on vLLM versions with xgrammar incompatibilities."
        ),
    )
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument(
        "--embedding_provider",
        choices=["fastembed", "huggingface", "openai"],
        default="fastembed",
    )
    parser.add_argument(
        "--embedding_model",
        default="BAAI/bge-small-en-v1.5",
    )
    parser.add_argument("--embedding_dims", type=int, default=384)
    parser.add_argument("--embedding_api_key", default=None)
    parser.add_argument("--embedding_base_url", default=None)
    parser.add_argument("--collection_name", default="experiment8_mem0_local_0725")
    parser.add_argument("--token_encoding", default="cl100k_base")
    parser.add_argument("--start_example", type=int, default=0)
    parser.add_argument("--end_example", type=int, default=None)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--max_sessions", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry_count", type=int, default=3)
    parser.add_argument("--retry_base_sleep", type=float, default=1.0)
    parser.add_argument(
        "--snapshot_mode",
        choices=["all", "final", "none"],
        default="all",
    )
    parser.add_argument("--skip_preflight", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = run_ingestion(**vars(args))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
