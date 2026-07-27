#!/usr/bin/env python3
"""Validate and normalize a complete MPT_v2_0725 memory JSONL artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            example_id = str(row.get("example_id", ""))
            if not example_id:
                raise ValueError(f"Missing example_id at {path}:{line_number}")
            rows.append(row)
    return rows


def index_unique(
    rows: list[dict[str, Any]], path: Path
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        example_id = str(row["example_id"])
        if example_id in indexed:
            raise ValueError(f"Duplicate example_id in {path}: {example_id}")
        indexed[example_id] = row
    return indexed


def normalized_digest(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--memory", required=True)
    parser.add_argument("--reused-source", required=True)
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    memory_path = Path(args.memory).resolve()
    reused_path = Path(args.reused_source).resolve()

    dataset = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(dataset, list):
        raise TypeError(f"Expected a JSON list: {input_path}")

    expected_ids = [str(row.get("example_id", "")) for row in dataset]
    if any(not example_id for example_id in expected_ids):
        raise ValueError("Dataset contains an empty example_id")
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("Dataset contains duplicate example_id values")

    dataset_by_id = {str(row["example_id"]): row for row in dataset}
    memory_by_id = index_unique(read_jsonl(memory_path), memory_path)
    reused_by_id = index_unique(read_jsonl(reused_path), reused_path)

    expected = set(expected_ids)
    actual = set(memory_by_id)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise RuntimeError(
            f"Incomplete memory: expected={len(expected)} actual={len(actual)} "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )

    reused_expected = {
        example_id
        for example_id, row in dataset_by_id.items()
        if row.get("meta", {}).get("subset") == "mix"
    }
    if set(reused_by_id) != reused_expected:
        raise RuntimeError(
            "Reused source does not match the dataset mix subset: "
            f"source={len(reused_by_id)} expected={len(reused_expected)}"
        )

    changed_reused = [
        example_id
        for example_id in sorted(reused_expected)
        if memory_by_id[example_id] != reused_by_id[example_id]
    ]
    if changed_reused:
        raise RuntimeError(
            f"Reused memory records changed: {changed_reused[:5]}"
        )

    subset_counts: Counter[str] = Counter()
    reasoning_counts: Counter[str] = Counter()
    session_mismatches: list[str] = []
    empty_preferences: list[str] = []
    for example_id in expected_ids:
        dataset_row = dataset_by_id[example_id]
        memory_row = memory_by_id[example_id]
        subset_counts[str(dataset_row.get("meta", {}).get("subset"))] += 1
        reasoning_counts[str(memory_row.get("reasoning_effort"))] += 1
        if memory_row.get("total_sessions_processed") != len(
            dataset_row.get("sessions", [])
        ):
            session_mismatches.append(example_id)
        if not str(memory_row.get("final_implicit_preference", "")).strip():
            empty_preferences.append(example_id)

    if reasoning_counts != Counter({"high": len(expected_ids)}):
        raise RuntimeError(
            f"Unexpected reasoning_effort values: {dict(reasoning_counts)}"
        )
    if session_mismatches:
        raise RuntimeError(
            f"Session-count mismatches: {session_mismatches[:5]}"
        )
    if empty_preferences:
        raise RuntimeError(
            f"Empty final preferences: {empty_preferences[:5]}"
        )

    ordered_rows = [memory_by_id[example_id] for example_id in expected_ids]
    if args.normalize:
        temporary = memory_path.with_suffix(memory_path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in ordered_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, memory_path)

    print(
        json.dumps(
            {
                "memory": str(memory_path),
                "records": len(ordered_rows),
                "unique_example_ids": len(memory_by_id),
                "subsets": dict(sorted(subset_counts.items())),
                "reasoning_effort": dict(reasoning_counts),
                "reused_exact_records": len(reused_expected),
                "normalized_sha256": normalized_digest(ordered_rows),
                "normalized": args.normalize,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
