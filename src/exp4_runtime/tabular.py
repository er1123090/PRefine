"""Tiny pandas-compatible surface for the JSON-only experiment4 loaders.

The original experiment4 scripts used pandas only for ``read_json``,
``DataFrame``, ``len`` and ``iterrows``.  Keeping that narrow interface here
lets the scripts run in the vLLM environment even when pandas is not installed,
without changing the dataset rows passed to the experiment logic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple


class Row(dict):
    def to_dict(self) -> Dict[str, Any]:
        return dict(self)


class DataFrame:
    def __init__(self, rows: Iterable[Dict[str, Any]] | Dict[str, Any]):
        if isinstance(rows, dict):
            if isinstance(rows.get("dataset"), list):
                rows = rows["dataset"]
            else:
                rows = [rows]
        self._rows: List[Row] = [
            Row(row) for row in rows if isinstance(row, dict)
        ]

    def __len__(self) -> int:
        return len(self._rows)

    def iterrows(self) -> Iterator[Tuple[int, Row]]:
        yield from enumerate(self._rows)

    def to_dict(self, orient: str = "dict"):
        if orient != "records":
            raise ValueError("The pandas-free compatibility layer supports records only")
        return [row.to_dict() for row in self._rows]


def read_json(path: str, lines: bool = False) -> DataFrame:
    source = Path(path)
    if not lines:
        return DataFrame(json.loads(source.read_text(encoding="utf-8")))

    rows: List[Dict[str, Any]] = []
    try:
        for line in source.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError("JSONL rows must be objects")
                rows.append(item)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"{source} is not JSONL") from exc
    return DataFrame(rows)
