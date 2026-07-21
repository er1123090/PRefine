#!/usr/bin/env python3
"""Build memory files whose implicit preference is the full session preference history.

The experiments4 inference code consumes ``final_implicit_preference`` as the
implicit memory input. This builder preserves the original memory record, but
replaces that field with every session-level ``final_preference_at_session``
from ``preference_evolution_history``.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/data/minseo/experiments5")
DEFAULT_SOURCE_ROOT = Path("/data/minseo/experiments4/ours_memory/inference/1231_MEMORY3")
DEFAULT_OUTPUT_RUN_ID = "1229_dev6_accum_history_memory_20260602"
DEFAULT_MEMORY_MODE = "accumulated_preference_history"
DEFAULT_MEMORY_MODELS = [
    "deepseek-ai_DeepSeek-R1-Distill-Llama-8B",
    "deepseek-ai_DeepSeek-R1-0528-Qwen3-8B",
    "google_gemma-3-12b-it",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object in {path}:{line_no}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def session_preferences(record: dict[str, Any]) -> list[dict[str, Any]]:
    history = record.get("preference_evolution_history") or []
    if not isinstance(history, list):
        return []

    preferences: list[dict[str, Any]] = []
    for fallback_index, entry in enumerate(history, start=1):
        if not isinstance(entry, dict):
            continue
        preference = entry.get("final_preference_at_session")
        if not isinstance(preference, dict) or not preference:
            continue
        item = {
            "session_index": entry.get("session_index", fallback_index),
            "final_preference_at_session": preference,
        }
        preferences.append(item)
    return preferences


def build_payload(preferences: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "memory_mode": DEFAULT_MEMORY_MODE,
        "source": "preference_evolution_history.final_preference_at_session",
        "aggregation_policy": "preserve_all_session_final_preferences_in_order",
        "usage_policy": (
            "Use these session-level finalized implicit preferences as accumulated "
            "memory evidence. Current user constraints and explicit preferences "
            "override older or conflicting memory."
        ),
        "implicit_preferences": preferences,
    }


def convert_record(record: dict[str, Any]) -> tuple[dict[str, Any], int, int]:
    preferences = session_preferences(record)
    payload = build_payload(preferences)
    converted = dict(record)
    converted["source_final_implicit_preference"] = record.get("final_implicit_preference")
    converted["final_implicit_preference"] = json.dumps(payload, ensure_ascii=False, indent=2)
    converted["final_accumulated_implicit_preferences"] = preferences
    converted["history_preference_count"] = len(preferences)
    converted["memory_mode"] = DEFAULT_MEMORY_MODE
    payload_chars = len(converted["final_implicit_preference"])
    return converted, len(preferences), payload_chars


def convert_memory_file(source_path: Path, output_path: Path) -> dict[str, Any]:
    source_rows = load_jsonl(source_path)
    converted_rows: list[dict[str, Any]] = []
    preference_counts: list[int] = []
    payload_chars: list[int] = []

    for record in source_rows:
        converted, preference_count, payload_len = convert_record(record)
        converted_rows.append(converted)
        preference_counts.append(preference_count)
        payload_chars.append(payload_len)

    write_jsonl(output_path, converted_rows)
    return {
        "source_path": str(source_path),
        "output_path": str(output_path),
        "records": len(converted_rows),
        "total_session_preferences": sum(preference_counts),
        "min_session_preferences": min(preference_counts) if preference_counts else 0,
        "max_session_preferences": max(preference_counts) if preference_counts else 0,
        "max_payload_chars": max(payload_chars) if payload_chars else 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-run-id", default=DEFAULT_OUTPUT_RUN_ID)
    parser.add_argument("--memory-mode", default=DEFAULT_MEMORY_MODE)
    parser.add_argument("--memory-model", action="append", dest="memory_models")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    memory_models = args.memory_models or DEFAULT_MEMORY_MODELS
    output_root = ROOT / "outputs/our_memory" / args.output_run_id / "memories" / args.memory_mode

    summaries: list[dict[str, Any]] = []
    for memory_safe in memory_models:
        source_path = args.source_root / memory_safe / "_memory1.jsonl"
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        output_path = output_root / memory_safe / "_memory1.jsonl"
        summary = convert_memory_file(source_path, output_path)
        summary["memory_model_safe"] = memory_safe
        summaries.append(summary)
        print(
            "wrote {output_path} records={records} session_preferences={total_session_preferences}".format(
                **summary
            )
        )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_run_id": args.output_run_id,
        "memory_mode": args.memory_mode,
        "source_root": str(args.source_root),
        "output_root": str(output_root),
        "memory_models": memory_models,
        "note": (
            "final_implicit_preference is a JSON string containing all "
            "preference_evolution_history.final_preference_at_session entries."
        ),
        "files": summaries,
    }
    manifest_path = output_root.parent / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
