from __future__ import annotations

import argparse
import hashlib
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from transformers import AutoTokenizer

from core import ContractError, canonical_json, load_json, read_jsonl, sha256_file, write_json_atomic, write_jsonl_atomic


def percentile(sorted_values: Sequence[int], probability: float) -> int:
    if not sorted_values:
        raise ContractError("empty percentile input")
    return sorted_values[math.ceil(probability * len(sorted_values)) - 1]


def summarize(values: Sequence[int]) -> dict[str, int]:
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "median": percentile(ordered, 0.5),
        "p95": percentile(ordered, 0.95),
        "p99": percentile(ordered, 0.99),
        "max": ordered[-1],
    }


def _nearest_case(rows: Sequence[Mapping[str, Any]], target: int) -> Mapping[str, Any]:
    return min(rows, key=lambda row: (abs(int(row["input_tokens"]) - target), str(row["case_id"])))


def select_canaries(cases: Sequence[Mapping[str, Any]], census: Sequence[Mapping[str, Any]]) -> list[str]:
    case_by_id = {case["case_id"]: case for case in cases}
    selected: set[str] = set()
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in census:
        grouped[(row["mode"], row["difficulty"])].append(row)
    for key in sorted(grouped):
        rows = grouped[key]
        selected.add(_nearest_case(rows, summarize([int(row["input_tokens"]) for row in rows])["median"])["case_id"])
    global_stats = summarize([int(row["input_tokens"]) for row in census])
    for label in ("min", "median", "p95", "max"):
        selected.add(_nearest_case(census, global_stats[label])["case_id"])
    medium_or = [
        case for case in cases
        if case["difficulty"] == "medium"
        and any(
            len({canonical_json(call["arguments"].get(slot)) for call in case["reference_ground_truth_preference"]}) > 1
            for slot in set().union(*(call["arguments"].keys() for call in case["reference_ground_truth_preference"]))
        )
    ]
    if medium_or:
        selected.add(min(medium_or, key=lambda case: case["case_id"])["case_id"])
    hard = [case for case in cases if case["difficulty"] == "hard"]
    if hard:
        selected.add(min(hard, key=lambda case: case["case_id"])["case_id"])
    explicit_multi = [case for case in cases if case["mode"] == "multi" and case["explicit_base_arguments"]]
    if explicit_multi:
        selected.add(min(explicit_multi, key=lambda case: case["case_id"])["case_id"])
    if not selected <= set(case_by_id):
        raise ContractError("canary selection produced an unknown case")
    return sorted(selected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--chat-template", required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--output-allowance", type=int, default=256)
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    template_path = Path(args.chat_template).resolve()
    template = template_path.read_text(encoding="utf-8")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokenizer.chat_template = template
    cases = read_jsonl(run_root / "prepared/cases.jsonl")
    rows: list[dict[str, Any]] = []
    for case in cases:
        request = case["request"]
        rendered = tokenizer.apply_chat_template(
            request["messages"],
            tools=request["tools"],
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=False,
        )
        token_ids = rendered["input_ids"] if isinstance(rendered, Mapping) else rendered
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        rows.append({
            "case_id": case["case_id"],
            "mode": case["mode"],
            "difficulty": case["difficulty"],
            "input_tokens": len(token_ids),
        })
    rows.sort(key=lambda row: row["case_id"])
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        grouped[f"{row['mode']}:{row['difficulty']}"] .append(row["input_tokens"])
    global_stats = summarize([row["input_tokens"] for row in rows])
    fits = global_stats["max"] + args.output_allowance <= args.max_model_len
    summary = {
        "chat_template_sha256": sha256_file(template_path),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocab_sha256": hashlib.sha256(str(len(tokenizer)).encode()).hexdigest(),
        "max_model_len": args.max_model_len,
        "output_allowance": args.output_allowance,
        "global": global_stats,
        "cells": {key: summarize(values) for key, values in sorted(grouped.items())},
        "fits_without_truncation": fits,
    }
    write_jsonl_atomic(run_root / "runtime/token_census.jsonl", rows)
    write_json_atomic(run_root / "runtime/token_census_summary.json", summary)
    write_json_atomic(run_root / "runtime/canary_manifest.json", {"case_ids": select_canaries(cases, rows)})
    print(canonical_json(summary))
    if not fits:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
