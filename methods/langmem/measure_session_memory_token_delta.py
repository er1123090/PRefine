from __future__ import annotations

import argparse
import asyncio
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from common import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_ENCODING,
    DEFAULT_INPUT_PATH,
    build_session_messages,
    count_serialized_memory_tokens,
    count_string_tokens,
    create_langmem_runtime,
    default_run_log_path,
    format_retrieved_memories,
    get_encoding,
    last_user_utterance,
    list_namespace_items,
    load_chains_dataset,
    make_snapshot_item,
    make_runtime_config,
    materialize_namespace,
    now_iso,
    search_namespace_items,
    session_input_token_count,
    setup_logger,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure how much LangMem stored memory and retrieved memory token counts "
            "change after each session is added."
        )
    )
    parser.add_argument("--input_path", type=str, default=DEFAULT_INPUT_PATH)
    parser.add_argument(
        "--output_csv",
        type=str,
        default="/data/minseo/experiments5/methods/langmem/session_memory_token_deltas_1229_dev_6.csv",
    )
    parser.add_argument(
        "--summary_json",
        type=str,
        default="/data/minseo/experiments5/methods/langmem/session_memory_token_deltas_1229_dev_6.summary.json",
    )
    parser.add_argument("--memory_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--embedding_model", type=str, default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--memory_top_k", type=int, default=5)
    parser.add_argument("--encoding", type=str, default=DEFAULT_ENCODING)
    parser.add_argument("--start_example", type=int, default=0)
    parser.add_argument("--end_example", type=int, default=None)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--max_sessions_per_example", type=int, default=None)
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--embedding_base_url", type=str, default=None)
    parser.add_argument("--embedding_api_key", type=str, default=None)
    parser.add_argument("--run_log_path", type=str, default=None)
    parser.add_argument("--continue_on_error", action="store_true")
    return parser.parse_args()


def maybe_trim(items: List[Any], max_count: int | None) -> List[Any]:
    if max_count is None:
        return items
    return items[:max_count]


def memory_items_for_count(items: List[Any]) -> List[Dict[str, Any]]:
    return [make_snapshot_item(item) for item in items]


async def main() -> None:
    args = parse_args()
    run_log_path = args.run_log_path or default_run_log_path(args.output_csv)
    logger = setup_logger("langmem_measure", run_log_path)
    encoding = get_encoding(args.encoding)
    df = load_chains_dataset(args.input_path)
    rows = df.to_dict("records")[args.start_example : args.end_example]
    rows = maybe_trim(rows, args.max_examples)
    logger.info(
        "Starting session token delta measurement: dataset_rows=%d selected_examples=%d input=%s output_csv=%s summary_json=%s model=%s embedding_model=%s top_k=%d memory_base_url=%s embedding_base_url=%s",
        len(df),
        len(rows),
        args.input_path,
        args.output_csv,
        args.summary_json,
        args.memory_model,
        args.embedding_model,
        args.memory_top_k,
        args.base_url,
        args.embedding_base_url,
    )

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
        "retrieval_query",
        "before_memory_count",
        "after_memory_count",
        "before_stored_memory_tokens",
        "after_stored_memory_tokens",
        "delta_stored_memory_count",
        "delta_stored_memory_tokens",
        "before_retrieved_memory_count",
        "after_retrieved_memory_count",
        "before_retrieved_memory_tokens",
        "after_retrieved_memory_tokens",
        "delta_retrieved_memory_count",
        "delta_retrieved_memory_tokens",
        "status",
        "error",
    ]

    total_sessions = 0
    success_sessions = 0
    failed_sessions = 0
    stored_delta_sum = 0
    stored_delta_abs_sum = 0
    retrieved_delta_sum = 0
    retrieved_delta_abs_sum = 0
    stored_nonzero_sessions = 0
    stored_negative_sessions = 0
    retrieved_nonzero_sessions = 0
    retrieved_negative_sessions = 0
    max_stored_delta = None
    min_stored_delta = None
    max_retrieved_delta = None
    min_retrieved_delta = None
    started_at = time.time()
    progress_interval = 25

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for example_index, example in enumerate(rows):
            example_id = str(example.get("example_id", f"example_{example_index}"))
            sessions = maybe_trim(example.get("sessions", []), args.max_sessions_per_example)
            logger.info(
                "Processing example %d/%d: example_id=%s sessions=%d",
                example_index + 1,
                len(rows),
                example_id,
                len(sessions),
            )
            runtime = create_langmem_runtime(
                memory_model=args.memory_model,
                embedding_model=args.embedding_model,
                base_url=args.base_url,
                api_key=args.api_key,
                embedding_base_url=args.embedding_base_url,
                embedding_api_key=args.embedding_api_key,
                query_limit=args.memory_top_k,
                enable_deletes=True,
            )
            namespace = materialize_namespace(example_id)
            config = make_runtime_config(example_id)

            for session_index, session in enumerate(sessions, start=1):
                total_sessions += 1
                retrieval_query = last_user_utterance(session)
                row_out: Dict[str, Any] = {
                    "example_index": example_index,
                    "example_id": example_id,
                    "session_index": session_index,
                    "dialogue_id": session.get("dialogue_id", ""),
                    "dialogue_turns": len(session.get("dialogue", [])),
                    "api_call_count": len(session.get("api_call", [])),
                    "session_input_tokens": session_input_token_count(session, encoding),
                    "retrieval_query": retrieval_query,
                    "before_memory_count": "",
                    "after_memory_count": "",
                    "before_stored_memory_tokens": "",
                    "after_stored_memory_tokens": "",
                    "delta_stored_memory_count": "",
                    "delta_stored_memory_tokens": "",
                    "before_retrieved_memory_count": "",
                    "after_retrieved_memory_count": "",
                    "before_retrieved_memory_tokens": "",
                    "after_retrieved_memory_tokens": "",
                    "delta_retrieved_memory_count": "",
                    "delta_retrieved_memory_tokens": "",
                    "status": "PENDING",
                    "error": "",
                }

                messages = build_session_messages(session)
                if not messages:
                    row_out["status"] = "SKIPPED_EMPTY_SESSION"
                    writer.writerow(row_out)
                    f.flush()
                    if total_sessions % progress_interval == 0:
                        logger.info(
                            "Measurement progress: sessions=%d ok=%d error=%d",
                            total_sessions,
                            success_sessions,
                            failed_sessions,
                        )
                    continue

                try:
                    before_items = await asyncio.to_thread(
                        list_namespace_items, runtime.store, namespace
                    )
                    before_snapshot_items = memory_items_for_count(before_items)
                    before_stored_tokens = count_serialized_memory_tokens(
                        before_snapshot_items, encoding
                    )
                    before_retrieved_items = await asyncio.to_thread(
                        search_namespace_items,
                        runtime.store,
                        namespace,
                        retrieval_query,
                        args.memory_top_k,
                    )
                    before_retrieved_records, before_retrieved_text = format_retrieved_memories(
                        before_retrieved_items
                    )
                    before_retrieved_tokens = count_string_tokens(
                        before_retrieved_text, encoding
                    )

                    await runtime.manager.ainvoke({"messages": messages}, config=config)

                    after_items = await asyncio.to_thread(
                        list_namespace_items, runtime.store, namespace
                    )
                    after_snapshot_items = memory_items_for_count(after_items)
                    after_stored_tokens = count_serialized_memory_tokens(
                        after_snapshot_items, encoding
                    )
                    after_retrieved_items = await asyncio.to_thread(
                        search_namespace_items,
                        runtime.store,
                        namespace,
                        retrieval_query,
                        args.memory_top_k,
                    )
                    after_retrieved_records, after_retrieved_text = format_retrieved_memories(
                        after_retrieved_items
                    )
                    after_retrieved_tokens = count_string_tokens(
                        after_retrieved_text, encoding
                    )

                    delta_stored_count = len(after_items) - len(before_items)
                    delta_stored_tokens = after_stored_tokens - before_stored_tokens
                    delta_retrieved_count = len(after_retrieved_records) - len(
                        before_retrieved_records
                    )
                    delta_retrieved_tokens = after_retrieved_tokens - before_retrieved_tokens

                    row_out.update(
                        {
                            "before_memory_count": len(before_items),
                            "after_memory_count": len(after_items),
                            "before_stored_memory_tokens": before_stored_tokens,
                            "after_stored_memory_tokens": after_stored_tokens,
                            "delta_stored_memory_count": delta_stored_count,
                            "delta_stored_memory_tokens": delta_stored_tokens,
                            "before_retrieved_memory_count": len(before_retrieved_records),
                            "after_retrieved_memory_count": len(after_retrieved_records),
                            "before_retrieved_memory_tokens": before_retrieved_tokens,
                            "after_retrieved_memory_tokens": after_retrieved_tokens,
                            "delta_retrieved_memory_count": delta_retrieved_count,
                            "delta_retrieved_memory_tokens": delta_retrieved_tokens,
                            "status": "OK",
                        }
                    )
                    writer.writerow(row_out)
                    f.flush()

                    success_sessions += 1
                    stored_delta_sum += delta_stored_tokens
                    stored_delta_abs_sum += abs(delta_stored_tokens)
                    retrieved_delta_sum += delta_retrieved_tokens
                    retrieved_delta_abs_sum += abs(delta_retrieved_tokens)
                    if delta_stored_tokens != 0:
                        stored_nonzero_sessions += 1
                    if delta_stored_tokens < 0:
                        stored_negative_sessions += 1
                    if delta_retrieved_tokens != 0:
                        retrieved_nonzero_sessions += 1
                    if delta_retrieved_tokens < 0:
                        retrieved_negative_sessions += 1
                    max_stored_delta = (
                        delta_stored_tokens
                        if max_stored_delta is None
                        else max(max_stored_delta, delta_stored_tokens)
                    )
                    min_stored_delta = (
                        delta_stored_tokens
                        if min_stored_delta is None
                        else min(min_stored_delta, delta_stored_tokens)
                    )
                    max_retrieved_delta = (
                        delta_retrieved_tokens
                        if max_retrieved_delta is None
                        else max(max_retrieved_delta, delta_retrieved_tokens)
                    )
                    min_retrieved_delta = (
                        delta_retrieved_tokens
                        if min_retrieved_delta is None
                        else min(min_retrieved_delta, delta_retrieved_tokens)
                    )
                    if total_sessions % progress_interval == 0:
                        logger.info(
                            "Measurement progress: sessions=%d ok=%d error=%d latest_example=%s latest_session=%d stored_delta=%d retrieved_delta=%d",
                            total_sessions,
                            success_sessions,
                            failed_sessions,
                            example_id,
                            session_index,
                            delta_stored_tokens,
                            delta_retrieved_tokens,
                        )
                except Exception as exc:
                    failed_sessions += 1
                    row_out["status"] = "ERROR"
                    row_out["error"] = str(exc)
                    writer.writerow(row_out)
                    f.flush()
                    logger.exception(
                        "Measurement failed: example_id=%s session_index=%d",
                        example_id,
                        session_index,
                    )
                    if not args.continue_on_error:
                        raise

    duration_seconds = time.time() - started_at
    summary = {
        "created_at": now_iso(),
        "input_path": args.input_path,
        "output_csv": args.output_csv,
        "summary_json": args.summary_json,
        "run_log_path": run_log_path,
        "memory_model": args.memory_model,
        "embedding_model": args.embedding_model,
        "memory_base_url": args.base_url,
        "embedding_base_url": args.embedding_base_url,
        "memory_top_k": args.memory_top_k,
        "encoding": args.encoding,
        "examples_processed": len(rows),
        "sessions_processed": total_sessions,
        "sessions_succeeded": success_sessions,
        "sessions_failed": failed_sessions,
        "rows_ok": success_sessions,
        "rows_error": failed_sessions,
        "stored_delta_token_sum": stored_delta_sum,
        "stored_delta_token_abs_sum": stored_delta_abs_sum,
        "avg_stored_delta_tokens": (
            stored_delta_sum / success_sessions if success_sessions else None
        ),
        "avg_abs_stored_delta_tokens": (
            stored_delta_abs_sum / success_sessions if success_sessions else None
        ),
        "max_stored_delta_tokens": max_stored_delta,
        "min_stored_delta_tokens": min_stored_delta,
        "stored_nonzero_delta_sessions": stored_nonzero_sessions,
        "stored_negative_delta_sessions": stored_negative_sessions,
        "retrieved_delta_token_sum": retrieved_delta_sum,
        "retrieved_delta_token_abs_sum": retrieved_delta_abs_sum,
        "avg_retrieved_delta_tokens": (
            retrieved_delta_sum / success_sessions if success_sessions else None
        ),
        "avg_abs_retrieved_delta_tokens": (
            retrieved_delta_abs_sum / success_sessions if success_sessions else None
        ),
        "max_retrieved_delta_tokens": max_retrieved_delta,
        "min_retrieved_delta_tokens": min_retrieved_delta,
        "retrieved_nonzero_delta_sessions": retrieved_nonzero_sessions,
        "retrieved_negative_delta_sessions": retrieved_negative_sessions,
        "duration_seconds": duration_seconds,
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(
        "Measurement finished: sessions=%d ok=%d error=%d output_csv=%s summary_json=%s duration_seconds=%.2f",
        total_sessions,
        success_sessions,
        failed_sessions,
        args.output_csv,
        args.summary_json,
        duration_seconds,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
