#!/usr/bin/env python3
"""Build inference-only memories: true-blind preferences plus accumulated API history."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/minseo/experiments5")
DEFAULT_DATA = Path("/data/minseo/experiments4/data/1229_dev_6.json")
DEFAULT_SOURCE_RUN = "1229_dev6_memory_variants_true_blind_20260528"
DEFAULT_OUTPUT_RUN = "1229_dev6_memory_variants_true_blind_api_history_20260529"

MEMORY_MODES = [
    "generation_only",
    "generation_only_accum",
    "blind_refine_1",
    "blind_refine_2",
    "blind_refine_3",
]
MEMORY_MODELS = [
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "google/gemma-3-12b-it",
]


def model_safe(model: str) -> str:
    return model.replace("/", "_")


def load_api_history_by_example(data_path: Path) -> dict[str, list[str]]:
    data = json.loads(data_path.read_text(encoding="utf-8"))
    history_by_id: dict[str, list[str]] = {}
    for example in data:
        example_id = example["example_id"]
        history: list[str] = []
        for idx, session in enumerate(example.get("sessions", []), start=1):
            for api_call in session.get("api_call", []):
                history.append(f"[Session {idx}] {api_call}")
        history_by_id[example_id] = history
    return history_by_id


def copy_with_api_history(
    source_file: Path,
    output_file: Path,
    api_history: dict[str, list[str]],
    data_path: Path,
) -> dict[str, Any]:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    true_blind_rows = 0
    rows_with_api_history = 0
    missing_history: list[str] = []

    with source_file.open("r", encoding="utf-8") as src, output_file.open("w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            rows += 1
            if record.get("true_blind") is True:
                true_blind_rows += 1

            example_id = record.get("example_id")
            if example_id not in api_history:
                missing_history.append(f"{source_file}:{line_no}:{example_id}")
                restored_history: list[str] = []
            else:
                restored_history = api_history[example_id]

            record["final_accumulated_api_calls"] = restored_history
            record["api_history_restored_for_inference"] = True
            record["api_history_source"] = str(data_path)
            record["source_true_blind_memory_file"] = str(source_file)
            if restored_history:
                rows_with_api_history += 1

            dst.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "source_file": str(source_file),
        "output_file": str(output_file),
        "rows": rows,
        "true_blind_rows": true_blind_rows,
        "rows_with_api_history": rows_with_api_history,
        "missing_history": missing_history,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create true-blind preference memories with restored accumulated API history for inference."
    )
    parser.add_argument("--data_path", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--source_run_id", default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--output_run_id", default=DEFAULT_OUTPUT_RUN)
    parser.add_argument("--root", type=Path, default=ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.root / "outputs/our_memory" / args.source_run_id / "memories"
    output_root = args.root / "outputs/our_memory" / args.output_run_id / "memories"
    api_history = load_api_history_by_example(args.data_path)

    manifest: dict[str, Any] = {
        "source_run_id": args.source_run_id,
        "output_run_id": args.output_run_id,
        "data_path": str(args.data_path),
        "memory_modes": MEMORY_MODES,
        "memory_models": MEMORY_MODELS,
        "files": [],
    }

    for mode in MEMORY_MODES:
        for model in MEMORY_MODELS:
            source_file = source_root / mode / model_safe(model) / "_memory1.jsonl"
            output_file = output_root / mode / model_safe(model) / "_memory1.jsonl"
            if not source_file.exists():
                raise FileNotFoundError(source_file)
            manifest["files"].append(copy_with_api_history(source_file, output_file, api_history, args.data_path))

    manifest_path = output_root.parent / "manifest_true_blind_api_history.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_root": str(output_root),
        "manifest_path": str(manifest_path),
        "files": len(manifest["files"]),
        "rows": sum(item["rows"] for item in manifest["files"]),
        "rows_with_api_history": sum(item["rows_with_api_history"] for item in manifest["files"]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
