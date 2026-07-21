from __future__ import annotations

import argparse
import csv
import pickle
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from common import (
    DEFAULT_EMEM_LLM_MODEL,
    DEFAULT_ENCODING,
    DEFAULT_INPUT_PATH,
    DEFAULT_MANIFEST_PATH,
    build_session_turns,
    count_string_tokens,
    default_run_log_path,
    get_encoding,
    load_chains_dataset,
    load_manifest_records,
    now_iso,
    render_edu_memory_line,
    sanitize_model_label,
    setup_logger,
    write_json,
)


DEFAULT_OUTPUT_CSV = (
    "/data/minseo/experiments5/methods/emem/output/"
    "emem_1229_dev_6.session_memory_token_growth.csv"
)
DEFAULT_SUMMARY_JSON = (
    "/data/minseo/experiments5/methods/emem/output/"
    "emem_1229_dev_6.session_memory_token_growth.summary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure how EMem stored memory grows after each session by reading "
            "saved EMem build artifacts."
        )
    )
    parser.add_argument("--manifest_path", type=str, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--input_path", type=str, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--llm_model", type=str, default=DEFAULT_EMEM_LLM_MODEL)
    parser.add_argument("--output_csv", type=str, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--summary_json", type=str, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--encoding", type=str, default=DEFAULT_ENCODING)
    parser.add_argument("--start_example", type=int, default=0)
    parser.add_argument("--end_example", type=int, default=None)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--run_log_path", type=str, default=None)
    return parser.parse_args()


def maybe_trim(items: Sequence[Any], max_count: int | None) -> List[Any]:
    if max_count is None:
        return list(items)
    return list(items[:max_count])


def normalize_example_record(record: Any) -> Dict[str, Any]:
    if isinstance(record, dict) and "example_id" in record:
        return record
    if isinstance(record, dict) and len(record) == 1:
        only_value = next(iter(record.values()))
        if isinstance(only_value, dict):
            return only_value
    return dict(record) if isinstance(record, dict) else {}


def build_session_input_text(session: Dict[str, Any], session_index: int) -> str:
    turns = build_session_turns(session, session_index)
    return "\n".join(f"{turn.speaker}: {turn.text}" for turn in turns)


def session_input_token_count(
    session: Dict[str, Any], session_index: int, encoding: Any
) -> int:
    return count_string_tokens(build_session_input_text(session, session_index), encoding)


def openie_pickle_path(workspace_dir: str, llm_model: str) -> Path:
    return Path(workspace_dir) / f"openie_results_ner_{sanitize_model_label(llm_model)}.pkl"


def load_openie_sessions(stats_path: Path) -> List[Dict[str, Any]]:
    with stats_path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected EMem pickle payload type: {type(data)!r}")
    sessions = data.get("sessions", [])
    if not isinstance(sessions, list):
        raise ValueError(f"Unexpected EMem sessions payload type: {type(sessions)!r}")
    return sessions


def sort_session_chunks(session_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    indexed_chunks = list(enumerate(session_chunks, start=1))

    def sort_key(item: Tuple[int, Dict[str, Any]]) -> Tuple[int, int, int]:
        original_order, chunk = item
        if not isinstance(chunk, dict):
            return (2, original_order, original_order)

        session_obj = chunk.get("session")
        session_id = getattr(session_obj, "session_id", None)
        if isinstance(session_id, int):
            return (0, session_id, original_order)

        idx = chunk.get("idx")
        if isinstance(idx, int):
            return (1, idx, original_order)
        if isinstance(idx, str) and idx.isdigit():
            return (1, int(idx), original_order)

        return (2, original_order, original_order)

    return [chunk for _, chunk in sorted(indexed_chunks, key=sort_key)]


def build_rendered_memory_text(edus: Iterable[Any]) -> str:
    return "\n".join(f"- {render_edu_memory_line(edu)}" for edu in edus)


def build_raw_edu_text(edus: Iterable[Any]) -> str:
    return "\n".join(f"- {getattr(edu, 'edu_text', str(edu))}" for edu in edus)


def safe_session_index(chunk: Dict[str, Any], fallback: int) -> int:
    if isinstance(chunk, dict):
        session_obj = chunk.get("session")
        session_id = getattr(session_obj, "session_id", None)
        if isinstance(session_id, int) and session_id > 0:
            return session_id
    return fallback


def summarize_numeric(values: Sequence[int]) -> Dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "total": 0,
        }
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "total": int(sum(values)),
    }


def main() -> None:
    args = parse_args()
    run_log_path = args.run_log_path or default_run_log_path(args.output_csv)
    logger = setup_logger("emem_session_token_growth", run_log_path)
    encoding = get_encoding(args.encoding)

    df = load_chains_dataset(args.input_path)
    rows = df.to_dict("records")[args.start_example : args.end_example]
    rows = maybe_trim(rows, args.max_examples)
    rows = [normalize_example_record(row) for row in rows]
    manifest_map = {
        str(record.get("example_id", "")): record
        for record in load_manifest_records(args.manifest_path)
    }

    logger.info(
        "Starting EMem session token growth measurement: dataset_rows=%d selected_examples=%d input=%s manifest=%s llm_model=%s output_csv=%s summary_json=%s",
        len(df),
        len(rows),
        args.input_path,
        args.manifest_path,
        args.llm_model,
        args.output_csv,
        args.summary_json,
    )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json = Path(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "example_index",
        "example_id",
        "workspace_dir",
        "session_index",
        "dialogue_id",
        "dialogue_turns",
        "api_call_count",
        "session_input_tokens",
        "before_memory_count",
        "after_memory_count",
        "delta_memory_count",
        "before_memory_tokens",
        "after_memory_tokens",
        "delta_memory_tokens",
        "before_raw_edu_tokens",
        "raw_edu_tokens_after_session",
        "delta_raw_edu_tokens",
        "session_edu_count",
        "status",
        "error",
    ]

    total_examples = len(rows)
    measured_examples = 0
    skipped_examples = 0
    failed_examples = 0
    chunk_count_mismatches = 0
    total_sessions = 0
    measured_sessions = 0
    negative_delta_sessions = 0
    monotonicity_violations = 0
    progress_interval = max(1, min(10, total_examples)) if total_examples else 1
    started_at = time.time()

    delta_memory_count_values: List[int] = []
    after_memory_count_values: List[int] = []
    delta_memory_token_values: List[int] = []
    after_memory_token_values: List[int] = []
    delta_raw_edu_token_values: List[int] = []
    after_raw_edu_token_values: List[int] = []

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for example_index, example in enumerate(rows, start=args.start_example):
            example_id = str(example.get("example_id", f"example_{example_index:04d}"))
            manifest_record = manifest_map.get(example_id)
            if not manifest_record or manifest_record.get("build_status") != "OK":
                skipped_examples += 1
                logger.warning(
                    "Skipping example_id=%s because no successful manifest record was found.",
                    example_id,
                )
                continue

            workspace_dir = str(manifest_record.get("workspace_dir", ""))
            stats_path = openie_pickle_path(workspace_dir, args.llm_model)
            if not stats_path.exists():
                failed_examples += 1
                logger.error(
                    "Skipping example_id=%s because EMem pickle is missing: %s",
                    example_id,
                    stats_path,
                )
                continue

            try:
                session_chunks = sort_session_chunks(load_openie_sessions(stats_path))
            except Exception as exc:
                failed_examples += 1
                logger.exception(
                    "Failed to load EMem sessions for example_id=%s from %s",
                    example_id,
                    stats_path,
                )
                continue

            sessions = example.get("sessions", [])
            if len(session_chunks) != len(sessions):
                chunk_count_mismatches += 1
                logger.warning(
                    "Session count mismatch for example_id=%s: dataset_sessions=%d emem_sessions=%d",
                    example_id,
                    len(sessions),
                    len(session_chunks),
                )

            measured_examples += 1
            cumulative_edus: List[Any] = []

            for chunk_position, chunk in enumerate(session_chunks, start=1):
                total_sessions += 1
                session_index = safe_session_index(chunk, chunk_position)
                dataset_session = (
                    sessions[session_index - 1]
                    if 0 <= session_index - 1 < len(sessions)
                    else {}
                )
                current_session_edus = []
                if isinstance(chunk, dict):
                    current_session_edus = list(chunk.get("edus") or [])

                before_memory_count = len(cumulative_edus)
                before_memory_tokens = count_string_tokens(
                    build_rendered_memory_text(cumulative_edus), encoding
                )
                before_raw_edu_tokens = count_string_tokens(
                    build_raw_edu_text(cumulative_edus), encoding
                )

                cumulative_edus.extend(current_session_edus)

                after_memory_count = len(cumulative_edus)
                after_memory_tokens = count_string_tokens(
                    build_rendered_memory_text(cumulative_edus), encoding
                )
                after_raw_edu_tokens = count_string_tokens(
                    build_raw_edu_text(cumulative_edus), encoding
                )

                delta_memory_count = after_memory_count - before_memory_count
                delta_memory_tokens = after_memory_tokens - before_memory_tokens
                delta_raw_edu_tokens = after_raw_edu_tokens - before_raw_edu_tokens

                if delta_memory_tokens < 0 or delta_raw_edu_tokens < 0:
                    negative_delta_sessions += 1
                    logger.warning(
                        "Negative memory token delta detected for example_id=%s session_index=%d: rendered_delta=%d raw_delta=%d",
                        example_id,
                        session_index,
                        delta_memory_tokens,
                        delta_raw_edu_tokens,
                    )
                if after_memory_tokens < before_memory_tokens:
                    monotonicity_violations += 1

                row_out: Dict[str, Any] = {
                    "example_index": example_index,
                    "example_id": example_id,
                    "workspace_dir": workspace_dir,
                    "session_index": session_index,
                    "dialogue_id": dataset_session.get("dialogue_id", ""),
                    "dialogue_turns": len(dataset_session.get("dialogue", [])),
                    "api_call_count": len(dataset_session.get("api_call", [])),
                    "session_input_tokens": (
                        session_input_token_count(dataset_session, session_index, encoding)
                        if dataset_session
                        else ""
                    ),
                    "before_memory_count": before_memory_count,
                    "after_memory_count": after_memory_count,
                    "delta_memory_count": delta_memory_count,
                    "before_memory_tokens": before_memory_tokens,
                    "after_memory_tokens": after_memory_tokens,
                    "delta_memory_tokens": delta_memory_tokens,
                    "before_raw_edu_tokens": before_raw_edu_tokens,
                    "raw_edu_tokens_after_session": after_raw_edu_tokens,
                    "delta_raw_edu_tokens": delta_raw_edu_tokens,
                    "session_edu_count": len(current_session_edus),
                    "status": "OK",
                    "error": "",
                }
                writer.writerow(row_out)

                measured_sessions += 1
                delta_memory_count_values.append(delta_memory_count)
                after_memory_count_values.append(after_memory_count)
                delta_memory_token_values.append(delta_memory_tokens)
                after_memory_token_values.append(after_memory_tokens)
                delta_raw_edu_token_values.append(delta_raw_edu_tokens)
                after_raw_edu_token_values.append(after_raw_edu_tokens)

            f.flush()
            if measured_examples == total_examples or measured_examples % progress_interval == 0:
                logger.info(
                    "Measurement progress: examples=%d/%d sessions=%d",
                    measured_examples,
                    total_examples,
                    measured_sessions,
                )

    duration_seconds = round(time.time() - started_at, 3)
    summary = {
        "created_at": now_iso(),
        "input_path": args.input_path,
        "manifest_path": args.manifest_path,
        "llm_model": args.llm_model,
        "output_csv": args.output_csv,
        "run_log_path": run_log_path,
        "examples_requested": total_examples,
        "examples_measured": measured_examples,
        "examples_skipped": skipped_examples,
        "examples_failed": failed_examples,
        "chunk_count_mismatches": chunk_count_mismatches,
        "sessions_measured": measured_sessions,
        "total_sessions_seen": total_sessions,
        "negative_delta_sessions": negative_delta_sessions,
        "monotonicity_violations": monotonicity_violations,
        "delta_memory_count": summarize_numeric(delta_memory_count_values),
        "after_memory_count": summarize_numeric(after_memory_count_values),
        "delta_memory_tokens": summarize_numeric(delta_memory_token_values),
        "after_memory_tokens": summarize_numeric(after_memory_token_values),
        "delta_raw_edu_tokens": summarize_numeric(delta_raw_edu_token_values),
        "after_raw_edu_tokens": summarize_numeric(after_raw_edu_token_values),
        "duration_seconds": duration_seconds,
    }
    write_json(args.summary_json, summary)
    logger.info(
        "Measurement finished: examples_measured=%d sessions_measured=%d output_csv=%s summary_json=%s",
        measured_examples,
        measured_sessions,
        args.output_csv,
        args.summary_json,
    )


if __name__ == "__main__":
    main()
