"""Gold firewall: history sanitization is the only input to memory construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .contracts import FORBIDDEN_KEYS
from .io import iter_jsonl, load_json, verify_sha256, write_jsonl


def _assert_no_forbidden_nested(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).casefold() in FORBIDDEN_KEYS:
                raise ValueError(f"forbidden key inside allowed history at {path}.{key}")
            _assert_no_forbidden_nested(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_forbidden_nested(child, f"{path}[{index}]")


def sanitize_example(raw: dict[str, Any]) -> dict[str, Any]:
    example_id = str(raw.get("example_id", "")).strip()
    if not example_id:
        raise ValueError("history example lacks example_id")
    raw_sessions = raw.get("sessions", [])
    if not isinstance(raw_sessions, list):
        raise ValueError(f"{example_id}: sessions must be a list")

    sessions: list[dict[str, Any]] = []
    for session_index, raw_session in enumerate(raw_sessions, start=1):
        if not isinstance(raw_session, dict):
            raise ValueError(f"{example_id}: session {session_index} must be an object")
        _assert_no_forbidden_nested(raw_session, f"{example_id}.sessions[{session_index - 1}]")
        raw_dialogue = raw_session.get("dialogue", [])
        if not isinstance(raw_dialogue, list):
            raise ValueError(f"{example_id}: dialogue must be a list")
        dialogue: list[dict[str, str]] = []
        for turn in raw_dialogue:
            if not isinstance(turn, dict):
                raise ValueError(f"{example_id}: dialogue turn must be an object")
            role = str(turn.get("role", "")).strip()
            message = turn.get("message", turn.get("content", ""))
            if role and isinstance(message, str) and message.strip():
                dialogue.append({"role": role, "message": message.strip()})

        raw_calls = raw_session.get("api_call", raw_session.get("api_calls", []))
        if isinstance(raw_calls, str):
            raw_calls = [raw_calls]
        if not isinstance(raw_calls, list) or any(not isinstance(call, str) for call in raw_calls):
            raise ValueError(f"{example_id}: session api calls must be strings")
        sessions.append(
            {
                "session_index": session_index,
                "dialogue": dialogue,
                "api_calls": [call.strip() for call in raw_calls if call.strip()],
            }
        )
    return {"example_id": example_id, "sessions": sessions}


def sanitize_dataset(source: str | Path, expected_sha256: str, output: str | Path) -> int:
    verify_sha256(source, expected_sha256)
    raw = load_json(source)
    if isinstance(raw, dict) and isinstance(raw.get("dataset"), list):
        raw = raw["dataset"]
    if not isinstance(raw, list):
        raise ValueError("history source must be a JSON array")
    rows = [sanitize_example(item) for item in raw if isinstance(item, dict)]
    ids = [row["example_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate example_id in history source")
    write_jsonl(output, rows)
    validate_sanitized_history(output)
    return len(rows)


def validate_sanitized_history(path: str | Path) -> int:
    count = 0
    seen: set[str] = set()
    for row in iter_jsonl(path):
        if set(row) != {"example_id", "sessions"}:
            raise ValueError(f"sanitized history has unexpected fields: {sorted(set(row))}")
        _assert_no_forbidden_nested(row)
        example_id = str(row["example_id"])
        if example_id in seen:
            raise ValueError(f"duplicate example_id: {example_id}")
        seen.add(example_id)
        for session in row["sessions"]:
            if set(session) != {"session_index", "dialogue", "api_calls"}:
                raise ValueError("sanitized session has unexpected fields")
            for turn in session["dialogue"]:
                if set(turn) != {"role", "message"}:
                    raise ValueError("sanitized dialogue turn has unexpected fields")
        count += 1
    return count
