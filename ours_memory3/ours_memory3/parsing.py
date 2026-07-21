"""Conservative parsing and public-schema helpers.

No function in this module reads files or calls a model.  It accepts only
history that the caller has already designated public.
"""

from __future__ import annotations

import ast
import json
import math
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .contracts import PublicInputError


_CALL_RE = re.compile(r"([A-Za-z_][\w.]*)\s*\(([^()]*)\)", re.DOTALL)
_KV_RE = re.compile(
    r"([A-Za-z_]\w*)\s*=\s*(\"(?:\\.|[^\"])*\"|'(?:\\.|[^'])*'|[^,]+)"
)
_TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_ROLE_RE = re.compile(r"^[ \t]*(User|Assistant|Tool|System)[ \t]*:[ \t]*(.*)$", re.I)


def normalize_text(value: str) -> str:
    if not isinstance(value, str):
        raise PublicInputError("text must be a string")
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def tokens(value: str) -> tuple[str, ...]:
    return tuple(_TOKEN_RE.findall(normalize_text(value)))


def phrase_present(text: str, phrase: str) -> bool:
    wanted = tokens(phrase)
    actual = tokens(text)
    width = len(wanted)
    return bool(width) and any(actual[index : index + width] == wanted for index in range(len(actual) - width + 1))


def split_identifier(value: str) -> tuple[str, ...]:
    expanded = _CAMEL_RE.sub(" ", value.replace("_", " ").replace("-", " "))
    return tokens(expanded)


def identifier_aliases(value: str) -> tuple[str, ...]:
    words = split_identifier(value)
    if not words:
        return ()
    phrases = {" ".join(words), "".join(words)}
    last = words[-1]
    if last.endswith("s") and len(last) > 3:
        phrases.add(" ".join((*words[:-1], last[:-1])))
    elif len(last) > 2:
        phrases.add(" ".join((*words[:-1], last + "s")))
    for prefix in ("get", "search", "find", "book", "reserve"):
        if words and words[0] == prefix and len(words) > 1:
            phrases.add(" ".join(words[1:]))
    return tuple(sorted(phrase for phrase in phrases if phrase.strip()))


def public_user_text(query: str, mode: str) -> str:
    if mode not in {"singleturn", "multiturn"}:
        raise PublicInputError("mode must be singleturn or multiturn")
    if mode == "singleturn":
        return query
    lines = query.splitlines()
    matched = [_ROLE_RE.match(line) for line in lines]
    if not any(matched):
        return query
    user_parts: list[str] = []
    role: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        if role == "user":
            text = "\n".join(buffer).strip()
            if text:
                user_parts.append(text)

    for line, match in zip(lines, matched):
        if match:
            flush()
            role = match.group(1).casefold()
            buffer = [match.group(2)]
        elif role is not None:
            buffer.append(line)
    flush()
    return "\n".join(user_parts)


def _literal(value: str) -> Any:
    try:
        return ast.literal_eval(value.strip())
    except (SyntaxError, ValueError):
        return value.strip().strip("\"'")


def _calls_from_json(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        result: list[dict[str, Any]] = []
        for item in value:
            result.extend(_calls_from_json(item))
        return result
    if not isinstance(value, Mapping):
        return []
    function = value.get("function") if isinstance(value.get("function"), Mapping) else value
    name = function.get("name") or function.get("function_name") or function.get("domain")
    arguments = function.get("arguments") or function.get("parameters") or function.get("args")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {key: _literal(raw) for key, raw in _KV_RE.findall(arguments)}
    if isinstance(name, str) and isinstance(arguments, Mapping):
        return [{"name": name, "arguments": dict(arguments)}]
    result: list[dict[str, Any]] = []
    for key in ("tool_calls", "calls", "api_calls", "output"):
        if key in value:
            result.extend(_calls_from_json(value[key]))
    return result


def extract_calls(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, (Mapping, list)):
        return _calls_from_json(value)
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        decoded = None
    if decoded is not None:
        calls = _calls_from_json(decoded)
        if calls:
            return calls
    return [
        {"name": match.group(1), "arguments": {key: _literal(raw) for key, raw in _KV_RE.findall(match.group(2))}}
        for match in _CALL_RE.finditer(value)
    ]


def safe_scalar(value: Any, maximum_characters: int) -> str | None:
    if isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, int):
        text = str(value)
    elif isinstance(value, float):
        if not math.isfinite(value):
            return None
        text = format(value, ".15g")
    elif isinstance(value, str):
        text = unicodedata.normalize("NFKC", value).strip()
    else:
        return None
    if not text or len(text) > maximum_characters or any(ord(char) < 32 for char in text):
        return None
    return text


def session_calls(history: object) -> tuple[tuple[int, list[dict[str, Any]]], ...]:
    if isinstance(history, Mapping):
        sessions = history.get("sessions")
    else:
        sessions = history
    if not isinstance(sessions, Sequence) or isinstance(sessions, (str, bytes)):
        raise PublicInputError("history.sessions must be a list")
    result: list[tuple[int, list[dict[str, Any]]]] = []
    for offset, session in enumerate(sessions):
        if not isinstance(session, Mapping):
            raise PublicInputError("history session must be an object")
        raw_index = session.get("session_index", offset)
        if not isinstance(raw_index, int):
            raise PublicInputError("session_index must be an integer")
        raw_calls = session.get("api_calls", session.get("api_call", []))
        if not isinstance(raw_calls, Sequence) or isinstance(raw_calls, (str, bytes)):
            raw_calls = [raw_calls]
        calls: list[dict[str, Any]] = []
        for raw_call in raw_calls:
            calls.extend(extract_calls(raw_call))
        result.append((raw_index, calls))
    return tuple(result)


def normalized_preference_slots(value: object) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, Mapping) or not value:
        raise PublicInputError("preference_slots must be a nonempty object")
    output: dict[str, tuple[str, ...]] = {}
    for raw_domain, raw_slots in value.items():
        if not isinstance(raw_domain, str) or not raw_domain.strip() or not isinstance(raw_slots, Sequence) or isinstance(raw_slots, (str, bytes)):
            raise PublicInputError("preference_slots has an invalid domain or slot list")
        slots = tuple(str(slot).strip() for slot in raw_slots if isinstance(slot, str) and slot.strip())
        if not slots or len(set(slots)) != len(slots):
            raise PublicInputError("preference_slots contains invalid or duplicate slots")
        output[raw_domain.strip()] = slots
    return output


def schema_enum_values(schema: object) -> dict[tuple[str, str], tuple[str, ...]]:
    tools: Iterable[object]
    if isinstance(schema, Mapping) and isinstance(schema.get("tools"), list):
        tools = schema["tools"]
    elif isinstance(schema, list):
        tools = schema
    else:
        tools = ()
    output: dict[tuple[str, str], tuple[str, ...]] = {}
    for raw_tool in tools:
        if not isinstance(raw_tool, Mapping):
            continue
        tool = raw_tool.get("function") if isinstance(raw_tool.get("function"), Mapping) else raw_tool
        name = tool.get("name")
        parameters = tool.get("parameters")
        properties = parameters.get("properties") if isinstance(parameters, Mapping) else None
        if not isinstance(name, str) or not isinstance(properties, Mapping):
            continue
        for slot, property_value in properties.items():
            if not isinstance(slot, str) or not isinstance(property_value, Mapping):
                continue
            values = property_value.get("enum")
            if not isinstance(values, list):
                continue
            safe = tuple(value for value in (safe_scalar(item, 80) for item in values) if value is not None)
            if safe:
                output[(name, slot)] = safe
    return output
