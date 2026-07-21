#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence


PACKAGE_ROOT = Path(__file__).resolve().parent

DEFAULT_MEMORY_ROOT = Path("/data/minseo/experiments4/ours_memory/inference/0312_MEMORY1")
DEFAULT_DATASET_PATH = Path("/data/minseo/experiments4/data/1229_dev_6.json")
DEFAULT_STEP2_PYTHON = Path("/data/minseo/experiments4/ours_memory/Preference_Memory_step2_ACTION_singleturn_VLLM.py")
DEFAULT_QUERY_PATH = Path("/data/minseo/experiments4/query_singleturn.json")
DEFAULT_PREF_LIST_PATH = Path("/data/minseo/experiments4/pref_list.json")
DEFAULT_PREF_GROUP_PATH = Path("/data/minseo/experiments4/pref_group.json")
DEFAULT_TOOLS_SCHEMA_PATH = Path("/data/minseo/experiments4/schema_easy.json")

DEFAULT_OUTPUT_ROOT = PACKAGE_ROOT / "outputs"
DEFAULT_PREP_ROOT = DEFAULT_OUTPUT_ROOT / "prepared"
DEFAULT_RUN_ROOT = DEFAULT_OUTPUT_ROOT / "runs"
DEFAULT_EVAL_ROOT = DEFAULT_OUTPUT_ROOT / "eval"
DEFAULT_PLOT_ROOT = DEFAULT_OUTPUT_ROOT / "plots"

SESSION_PREFIX_RE = re.compile(r"^\[Session\s+(\d+)\]")


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def safe_name(value: Any) -> str:
    text = str(value).strip()
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return safe.strip("_") or "unnamed"


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def iter_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    return list(iter_jsonl(path))


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: str | Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_session_tag(text: Any) -> int | None:
    if text is None:
        return None
    match = SESSION_PREFIX_RE.match(str(text))
    if not match:
        return None
    return int(match.group(1))


def filter_api_calls_by_session(api_calls: Any, session_index: int) -> List[str]:
    if not isinstance(api_calls, list):
        return []
    filtered: List[str] = []
    for call in api_calls:
        tagged_session = parse_session_tag(call)
        if tagged_session is None or tagged_session <= session_index:
            filtered.append(str(call))
    return filtered


def stringify_implicit_pref(value: Any) -> str:
    if value is None or value == "" or value == {}:
        return "{}"
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else "{}"
    return json.dumps(value, ensure_ascii=False, indent=2)
