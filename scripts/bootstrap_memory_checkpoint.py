#!/usr/bin/env python3
"""Seed or normalize a resume-safe memory checkpoint from partial JSONL files."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not str(row.get("example_id", "")):
                raise ValueError(f"Missing example_id at {path}:{line_number}")
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--expected-count", type=int, default=600)
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    dataset = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(dataset, list):
        raise TypeError(f"Expected a JSON list: {input_path}")
    expected_ids = [str(row.get("example_id", "")) for row in dataset]
    if len(expected_ids) != args.expected_count:
        raise RuntimeError(
            f"Dataset count mismatch: expected={args.expected_count}, "
            f"actual={len(expected_ids)}"
        )
    if any(not example_id for example_id in expected_ids):
        raise RuntimeError("Dataset contains an empty example_id")
    if len(set(expected_ids)) != len(expected_ids):
        raise RuntimeError("Dataset contains duplicate example_id values")

    records: dict[str, dict[str, Any]] = {}
    paths = [Path(source).resolve() for source in args.source]
    if output_path.exists():
        paths.append(output_path)
    for path in paths:
        for row in read_jsonl(path):
            example_id = str(row["example_id"])
            existing = records.get(example_id)
            if existing is not None and existing != row:
                raise RuntimeError(
                    f"Conflicting checkpoint records for {example_id}: {path}"
                )
            records[example_id] = row

    unexpected = sorted(set(records) - set(expected_ids))
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint IDs: {unexpected[:5]}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for example_id in expected_ids:
            if example_id in records:
                handle.write(
                    json.dumps(records[example_id], ensure_ascii=False) + "\n"
                )
    os.replace(temporary, output_path)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "expected": len(expected_ids),
                "completed": len(records),
                "remaining": len(expected_ids) - len(records),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
