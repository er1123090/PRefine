"""Aggregate provider-reported and locally counted tokens from experiment outputs."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, Iterator, List, Sequence

import tiktoken


USAGE_ALIASES = {
    "input_tokens": ("input_tokens", "prompt_tokens", "prompt_token_count"),
    "cached_input_tokens": (
        "cached_input_tokens",
        "cached_tokens",
        "cached_content_token_count",
    ),
    "output_tokens": ("output_tokens", "completion_tokens", "candidates_token_count"),
    "reasoning_tokens": (
        "reasoning_tokens",
        "reasoning_token_count",
        "thoughts_token_count",
    ),
    "total_tokens": ("total_tokens", "total_token_count"),
}

CONSTRUCTION_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "total_tokens",
    "call_count",
    "usage_available_calls",
    "usage_missing_calls",
)


def load_rows(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
        return rows
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        for key in ("data", "examples", "items", "dataset"):
            rows = value.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
        return [value]
    return []


def nested_values(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from nested_values(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from nested_values(item)


def usage_value(row: Dict[str, Any], aliases: Sequence[str]) -> int | None:
    sources = [row]
    token_counts = row.get("token_counts")
    if isinstance(token_counts, dict):
        sources.insert(0, token_counts)
    for source in sources:
        for key in aliases:
            value = source.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return int(value)
    return None


def usage_evidence(usage: Dict[str, Any]) -> bool:
    alias_names = {
        alias for aliases in USAGE_ALIASES.values() for alias in aliases
    }
    if any(
        key in alias_names
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
        for key, value in usage.items()
    ):
        return True
    call_count = usage.get("call_count")
    return isinstance(call_count, (int, float)) and not isinstance(call_count, bool)


def summarize(values: Iterable[float | int | None]) -> Dict[str, float | int | None]:
    data = [value for value in values if value is not None]
    return {
        "sum": sum(data),
        "mean": mean(data) if data else 0.0,
        "min": min(data) if data else 0,
        "max": max(data) if data else 0,
        "available_rows": len(data),
    }


def calculate_cost_usd(
    *,
    input_tokens: int | float,
    cached_input_tokens: int | float,
    output_tokens: int | float,
    input_cost_per_million: float,
    output_cost_per_million: float,
    cached_input_cost_per_million: float | None = None,
) -> float:
    cached = min(float(cached_input_tokens), float(input_tokens))
    uncached = max(0.0, float(input_tokens) - cached)
    cached_rate = (
        input_cost_per_million
        if cached_input_cost_per_million is None
        else cached_input_cost_per_million
    )
    return (
        uncached * input_cost_per_million
        + cached * cached_rate
        + float(output_tokens) * output_cost_per_million
    ) / 1_000_000


def usage_cost(
    usage: Dict[str, Any],
    *,
    input_rate: float | None,
    output_rate: float | None,
    cached_rate: float | None,
) -> float | None:
    if input_rate is None or output_rate is None:
        return None
    return calculate_cost_usd(
        input_tokens=usage.get("input_tokens") or 0,
        cached_input_tokens=usage.get("cached_input_tokens") or 0,
        output_tokens=usage.get("output_tokens") or 0,
        input_cost_per_million=input_rate,
        output_cost_per_million=output_rate,
        cached_input_cost_per_million=cached_rate,
    )


def usage_coverage(usage: Dict[str, Any]) -> float | None:
    call_count = usage.get("call_count")
    available_calls = usage.get("usage_available_calls")
    if (
        isinstance(call_count, (int, float))
        and call_count
        and isinstance(available_calls, (int, float))
    ):
        return available_calls / call_count
    return None


def covered_usage_cost(
    usage: Dict[str, Any],
    *,
    input_rate: float | None,
    output_rate: float | None,
    cached_rate: float | None,
) -> float | None:
    if not usage_evidence(usage):
        return None
    if usage_coverage(usage) == 0:
        return None
    return usage_cost(
        usage,
        input_rate=input_rate,
        output_rate=output_rate,
        cached_rate=cached_rate,
    )


def safe_component_name(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_")


def construction_report(row: Dict[str, Any]) -> Dict[str, Any]:
    value = row.get("construction_token_usage")
    return value if isinstance(value, dict) else {}


def setup_report(row: Dict[str, Any]) -> Dict[str, Any]:
    value = row.get("setup_token_usage")
    return value if isinstance(value, dict) else {}


def session_sources(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ("session_exports", "preference_evolution_history"):
        value = row.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with output.open("w", newline="", encoding="utf-8") as handle:
        if fieldnames:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def summarize_metrics(
    rows: List[Dict[str, Any]], fields: Sequence[str]
) -> Dict[str, Dict[str, float | int | None]]:
    return {
        field: summarize(
            value if isinstance(value, (int, float)) else None
            for value in (row.get(field) for row in rows)
        )
        for field in fields
    }


def grouped_metrics(
    rows: List[Dict[str, Any]],
    group_key: str,
    fields: Sequence[str],
) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        value = row.get(group_key)
        if value is not None and value != "":
            groups.setdefault(str(value), []).append(row)
    return {
        key: {
            "rows": len(group_rows),
            "metrics": summarize_metrics(group_rows, fields),
        }
        for key, group_rows in sorted(groups.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="JSON/JSONL paths or glob patterns")
    parser.add_argument(
        "--text-fields",
        default="model_input,llm_output,retrieved_memories",
        help="Comma-separated fields to count locally with tiktoken.",
    )
    parser.add_argument("--encoding", default="cl100k_base")
    parser.add_argument(
        "--skip-local",
        action="store_true",
        help="Only aggregate provider-reported token usage.",
    )
    parser.add_argument("--csv_output", default=None)
    parser.add_argument("--session_csv_output", default=None)
    parser.add_argument("--json_output", default=None)
    parser.add_argument("--input-cost-per-million", type=float, default=None)
    parser.add_argument("--output-cost-per-million", type=float, default=None)
    parser.add_argument("--cached-input-cost-per-million", type=float, default=None)
    args = parser.parse_args()
    if (args.input_cost_per_million is None) != (
        args.output_cost_per_million is None
    ):
        parser.error(
            "--input-cost-per-million and --output-cost-per-million must be set together"
        )

    paths = sorted(
        {
            Path(match)
            for pattern in args.inputs
            for match in glob.glob(pattern, recursive=True)
        }
    )
    encoding = None
    tokenizer_error = None
    if not args.skip_local:
        try:
            encoding = tiktoken.get_encoding(args.encoding)
        except Exception as exc:
            tokenizer_error = f"{type(exc).__name__}: {exc}"
            print(
                "[Warning] Local tokenizer unavailable; provider usage will still be "
                "reported. Pre-cache the requested tiktoken encoding or use --skip-local.",
                file=sys.stderr,
            )
    fields = [field.strip() for field in args.text_fields.split(",") if field.strip()]
    detail: List[Dict[str, Any]] = []
    session_detail: List[Dict[str, Any]] = []

    for path in paths:
        for index, row in enumerate(load_rows(path)):
            report = construction_report(row)
            report_summary = (
                report.get("summary") if isinstance(report.get("summary"), dict) else {}
            )
            setup = setup_report(row)
            setup_summary = (
                setup.get("summary") if isinstance(setup.get("summary"), dict) else {}
            )
            item: Dict[str, Any] = {
                "file": str(path),
                "row": index,
                "example_id": row.get("example_id"),
                "example_id_sub": row.get("example_id_sub"),
                "method": row.get("method")
                or ("ours_memory" if row.get("memory_mode") else None),
                "memory_mode": row.get("memory_mode"),
                "model_name": row.get("model_name"),
                "retrieved_memory_tokens": row.get("retrieved_memory_tokens"),
                "retrieval_prompt_tokens": row.get("retrieval_prompt_tokens"),
                "local_construction_input_tokens": row.get(
                    "local_construction_input_tokens"
                ),
            }
            for canonical, aliases in USAGE_ALIASES.items():
                item[canonical] = usage_value(row, aliases)
            row_token_counts = row.get("token_counts")
            row_usage = (
                row_token_counts if isinstance(row_token_counts, dict) else row
            )
            item["provider_usage_coverage"] = usage_coverage(row_usage)
            item["construction_provider_usage_coverage"] = usage_coverage(
                report_summary
            )
            for field in CONSTRUCTION_FIELDS:
                item[f"construction_{field}"] = report_summary.get(field)
            total_cost = (
                usage_cost(
                    item,
                    input_rate=args.input_cost_per_million,
                    output_rate=args.output_cost_per_million,
                    cached_rate=args.cached_input_cost_per_million,
                )
                if usage_evidence(row_usage)
                and item["provider_usage_coverage"] != 0
                else None
            )
            item["estimated_cost_usd"] = total_cost
            item["construction_estimated_cost_usd"] = covered_usage_cost(
                report_summary,
                input_rate=args.input_cost_per_million,
                output_rate=args.output_cost_per_million,
                cached_rate=args.cached_input_cost_per_million,
            )
            item["setup_provider_usage_coverage"] = usage_coverage(setup_summary)
            for field in CONSTRUCTION_FIELDS:
                item[f"setup_{field}"] = setup_summary.get(field)
            item["setup_estimated_cost_usd"] = covered_usage_cost(
                setup_summary,
                input_rate=args.input_cost_per_million,
                output_rate=args.output_cost_per_million,
                cached_rate=args.cached_input_cost_per_million,
            )
            retrieved_tokens = item.get("retrieved_memory_tokens")
            item["retrieval_estimated_input_cost_usd"] = (
                float(retrieved_tokens)
                * args.input_cost_per_million
                / 1_000_000
                if isinstance(retrieved_tokens, (int, float))
                and args.input_cost_per_million is not None
                else None
            )
            local_construction_tokens = item.get("local_construction_input_tokens")
            item["local_construction_input_estimated_cost_usd"] = (
                float(local_construction_tokens)
                * args.input_cost_per_million
                / 1_000_000
                if isinstance(local_construction_tokens, (int, float))
                and args.input_cost_per_million is not None
                else None
            )

            by_component = report.get("by_component")
            if isinstance(by_component, dict):
                for component, component_usage in by_component.items():
                    if not isinstance(component_usage, dict):
                        continue
                    prefix = f"construction_{safe_component_name(str(component))}"
                    for field in CONSTRUCTION_FIELDS:
                        item[f"{prefix}_{field}"] = component_usage.get(field)
                    item[f"{prefix}_provider_usage_coverage"] = usage_coverage(
                        component_usage
                    )
                    item[f"{prefix}_estimated_cost_usd"] = covered_usage_cost(
                        component_usage,
                        input_rate=args.input_cost_per_million,
                        output_rate=args.output_cost_per_million,
                        cached_rate=args.cached_input_cost_per_million,
                    )

            for field in fields:
                text = "\n".join(nested_values(row.get(field)))
                item[f"{field}_local_tokens"] = (
                    len(encoding.encode(text))
                    if encoding is not None and text
                    else (0 if encoding is not None else None)
                )

            sessions = session_sources(row)
            stored_values = [
                session.get("stored_memory_tokens_after_session")
                for session in sessions
                if isinstance(session.get("stored_memory_tokens_after_session"), (int, float))
            ]
            item["final_stored_memory_tokens"] = (
                stored_values[-1] if stored_values else None
            )
            item["final_stored_memory_estimated_input_cost_usd"] = (
                float(item["final_stored_memory_tokens"])
                * args.input_cost_per_million
                / 1_000_000
                if isinstance(item["final_stored_memory_tokens"], (int, float))
                and args.input_cost_per_million is not None
                else None
            )
            detail.append(item)

            by_session = report.get("by_session")
            by_session = by_session if isinstance(by_session, dict) else {}
            for session in sessions:
                session_index = session.get("session_index")
                session_usage = session.get("construction_token_usage")
                if not isinstance(session_usage, dict):
                    session_usage = by_session.get(str(session_index), {})
                session_summary = (
                    session_usage.get("summary")
                    if isinstance(session_usage, dict)
                    and isinstance(session_usage.get("summary"), dict)
                    else {}
                )
                session_row = {
                    "file": str(path),
                    "row": index,
                    "example_id": row.get("example_id"),
                    "method": item["method"],
                    "memory_mode": item["memory_mode"],
                    "session_index": session_index,
                    "session_input_tokens": session.get("session_input_tokens"),
                    "local_construction_input_tokens": session.get(
                        "local_construction_input_tokens"
                    ),
                    "memory_count_after_session": session.get(
                        "memory_count_after_session"
                    ),
                    "stored_memory_tokens_after_session": session.get(
                        "stored_memory_tokens_after_session"
                    ),
                    **{
                        f"construction_{field}": session_summary.get(field)
                        for field in CONSTRUCTION_FIELDS
                    },
                }
                session_row["provider_usage_coverage"] = usage_coverage(
                    session_summary
                )
                session_row["construction_estimated_cost_usd"] = covered_usage_cost(
                    session_summary,
                    input_rate=args.input_cost_per_million,
                    output_rate=args.output_cost_per_million,
                    cached_rate=args.cached_input_cost_per_million,
                )
                stored_tokens = session_row.get("stored_memory_tokens_after_session")
                session_row["stored_memory_estimated_input_cost_usd"] = (
                    float(stored_tokens)
                    * args.input_cost_per_million
                    / 1_000_000
                    if isinstance(stored_tokens, (int, float))
                    and args.input_cost_per_million is not None
                    else None
                )
                local_input_tokens = session_row.get(
                    "local_construction_input_tokens"
                )
                session_row[
                    "local_construction_input_estimated_cost_usd"
                ] = (
                    float(local_input_tokens)
                    * args.input_cost_per_million
                    / 1_000_000
                    if isinstance(local_input_tokens, (int, float))
                    and args.input_cost_per_million is not None
                    else None
                )
                session_detail.append(session_row)

    numeric_fields: List[str] = [
        *USAGE_ALIASES,
        *(f"{field}_local_tokens" for field in fields),
        "retrieved_memory_tokens",
        "retrieval_prompt_tokens",
        "local_construction_input_tokens",
        "final_stored_memory_tokens",
        "provider_usage_coverage",
        "construction_provider_usage_coverage",
        "setup_provider_usage_coverage",
        "estimated_cost_usd",
        "construction_estimated_cost_usd",
        "setup_estimated_cost_usd",
        "retrieval_estimated_input_cost_usd",
        "local_construction_input_estimated_cost_usd",
        "final_stored_memory_estimated_input_cost_usd",
    ]
    for row in detail:
        for key, value in row.items():
            if key.startswith("construction_") and isinstance(value, (int, float)):
                if key not in numeric_fields:
                    numeric_fields.append(key)
    session_numeric_fields = [
        "session_input_tokens",
        "local_construction_input_tokens",
        "memory_count_after_session",
        "stored_memory_tokens_after_session",
        "provider_usage_coverage",
        *(f"construction_{field}" for field in CONSTRUCTION_FIELDS),
        "construction_estimated_cost_usd",
        "stored_memory_estimated_input_cost_usd",
        "local_construction_input_estimated_cost_usd",
    ]
    summary = {
        "files": len(paths),
        "rows": len(detail),
        "encoding": args.encoding,
        "local_tokenizer_available": encoding is not None,
        "local_tokenizer_error": tokenizer_error,
        "pricing_usd_per_million_tokens": {
            "input": args.input_cost_per_million,
            "cached_input": (
                args.cached_input_cost_per_million
                if args.cached_input_cost_per_million is not None
                else args.input_cost_per_million
            ),
            "output": args.output_cost_per_million,
        },
        "metrics": summarize_metrics(detail, numeric_fields),
        "by_file": grouped_metrics(detail, "file", numeric_fields),
        "by_method": grouped_metrics(detail, "method", numeric_fields),
        "by_memory_mode": grouped_metrics(detail, "memory_mode", numeric_fields),
        "sessions": {
            "rows": len(session_detail),
            "metrics": summarize_metrics(session_detail, session_numeric_fields),
            "by_method": grouped_metrics(
                session_detail, "method", session_numeric_fields
            ),
            "by_memory_mode": grouped_metrics(
                session_detail, "memory_mode", session_numeric_fields
            ),
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.csv_output:
        write_csv(args.csv_output, detail)
    if args.session_csv_output:
        write_csv(args.session_csv_output, session_detail)
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "summary": summary,
                    "rows": detail,
                    "sessions": session_detail,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
