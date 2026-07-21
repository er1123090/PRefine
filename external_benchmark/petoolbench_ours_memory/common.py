"""Shared helpers for the PEToolBench ours_memory adapter."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


HISTORY_KEYS = {
    "p": "instruction_preferred",
    "r": "instruction_ratings",
    "c": "instruction_chronological",
}

HISTORY_LABELS = {
    "p": "preferred",
    "r": "ratings",
    "c": "chronological",
}


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def iter_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_embedded_json(instruction: str, start_marker: str, end_marker: str) -> Any:
    start = instruction.find(start_marker)
    if start == -1:
        raise ValueError(f"start marker not found: {start_marker!r}")
    start += len(start_marker)
    end = instruction.find(end_marker, start)
    if end == -1:
        raise ValueError(f"end marker not found: {end_marker!r}")
    raw = instruction[start:end].strip()
    return json.loads(raw)


def parse_petoolbench_instruction(instruction: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    history = parse_embedded_json(
        instruction,
        "Interaction history is:\n",
        "\n\nAvailable tools",
    )
    tools = parse_embedded_json(
        instruction,
        "Available tools you can call with input parameters are listed here:\n",
        "\n\nGenerate your tool call",
    )
    if not isinstance(history, list):
        raise ValueError("parsed interaction history is not a list")
    if not isinstance(tools, list):
        raise ValueError("parsed candidate tools are not a list")
    return history, tools


def load_petoolbench_rows(
    input_path: str | Path,
    history_type: str,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if history_type not in HISTORY_KEYS:
        raise ValueError(f"unsupported history_type={history_type!r}; choose one of {sorted(HISTORY_KEYS)}")

    rows = read_json(input_path)
    if not isinstance(rows, list):
        raise ValueError(f"PEToolBench input must be a list, got {type(rows).__name__}")

    key = HISTORY_KEYS[history_type]
    normalized: List[Dict[str, Any]] = []
    selected = rows if limit is None else rows[:limit]
    for idx, row in enumerate(selected):
        if key not in row:
            raise ValueError(f"row {idx} missing {key}")
        history, tools = parse_petoolbench_instruction(str(row[key]))
        gt = row.get("api_call_ground_truth")
        if not isinstance(gt, dict) or "tool_name" not in gt or "parameters" not in gt:
            raise ValueError(f"row {idx} has invalid api_call_ground_truth")
        normalized.append(
            {
                "example_id": f"petoolbench_{history_type}_{idx:06d}",
                "source_index": idx,
                "history_type": history_type,
                "history_label": HISTORY_LABELS[history_type],
                "query": row.get("query", ""),
                "history": history,
                "history_length": len(history),
                "candidate_tools": tools,
                "candidate_tool_count": len(tools),
                "api_call_ground_truth": gt,
            }
        )
    return normalized


def tool_call_to_text(tool_call: Dict[str, Any]) -> str:
    return json.dumps(
        {
            "tool_name": tool_call.get("tool_name", ""),
            "parameters": tool_call.get("parameters", {}),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def build_ours_sessions(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    sessions: List[Dict[str, Any]] = []
    for turn in record.get("history", []):
        instruction = str(turn.get("instruction", "")).strip()
        rating = turn.get("rating")
        if rating is not None:
            instruction = (
                f"{instruction}\nObserved user satisfaction rating for this tool call: {rating}."
            )
        sessions.append(
            {
                "dialogue": [{"role": "User", "message": instruction}],
                "api_call": [tool_call_to_text(turn.get("tool_call", {}))],
            }
        )
    return sessions


def tool_namespace(tool_name: str) -> str:
    parts = re.findall(r"<([^>]+)>", tool_name)
    if len(parts) >= 2:
        return f"{parts[0]}::{parts[1]}"
    if parts:
        return parts[0]
    return tool_name


def mock_preference(record: Dict[str, Any]) -> Dict[str, Any]:
    history = record.get("history", [])
    positive_calls = []
    negative_calls = []
    unrated_calls = []
    for item in history:
        call = item.get("tool_call", {})
        rating = item.get("rating")
        if rating == 1:
            positive_calls.append(call)
        elif rating == 0:
            negative_calls.append(call)
        else:
            unrated_calls.append(call)

    evidence_calls = positive_calls or unrated_calls or [item.get("tool_call", {}) for item in history]
    namespaces = Counter(tool_namespace(str(call.get("tool_name", ""))) for call in evidence_calls)
    top_namespaces = [name for name, _ in namespaces.most_common(5) if name]

    if record.get("history_type") == "r":
        reasoning = (
            "Mock memory uses positively rated PEToolBench tool calls as preference evidence "
            "and treats zero-rated calls as negative evidence."
        )
    elif record.get("history_type") == "c":
        reasoning = (
            "Mock memory treats the chronological history as preference evidence, with later "
            "tool calls considered more representative by the benchmark prompt."
        )
    else:
        reasoning = "Mock memory summarizes the preferred PEToolBench tool-use history."

    implicit = (
        "Select candidate tools whose provider/domain and parameter conventions are consistent "
        f"with the user's observed tool-use history; strongest observed namespaces: {top_namespaces}."
    )
    return {
        "reasoning": reasoning,
        "implicit_pref": implicit,
        "history_type": record.get("history_type"),
        "top_tool_namespaces": top_namespaces,
        "positive_evidence_count": len(positive_calls),
        "negative_evidence_count": len(negative_calls),
        "unrated_evidence_count": len(unrated_calls),
    }


def first_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None

