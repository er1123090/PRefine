"""Strict JSONL helpers used by the public-only CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .contracts import PublicInputError


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.is_file():
        raise PublicInputError(f"input file does not exist: {file_path}")
    rows: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PublicInputError(f"invalid JSONL at line {number}") from exc
            if not isinstance(value, dict):
                raise PublicInputError(f"JSONL line {number} must be an object")
            rows.append(value)
    if not rows:
        raise PublicInputError("input JSONL is empty")
    return rows


def write_jsonl_once(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if file_path.exists() or file_path.is_symlink():
        raise PublicInputError(f"output already exists: {file_path}")
    temporary = file_path.with_name(file_path.name + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
                handle.write("\n")
        temporary.replace(file_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
