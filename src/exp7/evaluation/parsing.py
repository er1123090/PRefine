"""Shared, dependency-free parsing and normalization for every turn type."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import re
from typing import Any, Mapping


_CALL_RE = re.compile(r"([A-Za-z_]\w*)\s*\((.*?)\)", re.DOTALL)
_BRACED_CALL_RE = re.compile(r"\{([A-Za-z_]\w*)\}\s*\(")
_THINK_RE = re.compile(r"<think>.*?</think>|</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?|```", re.IGNORECASE)
_ISO_DATE_RE = re.compile(r"^(?:\d{4}-)?(\d{1,2})-(\d{1,2})$")
_SLASH_DATE_RE = re.compile(r"^(\d{1,2})/(\d{1,2})(?:/\d{2,4})?$")
_TIME_RE = re.compile(r"^(\d{1,2}):(\d{2})(?:\s*([ap])\.?m\.?)?$", re.I)
_HOUR_RE = re.compile(r"^(\d{1,2})\s*([ap])\.?m\.?$", re.I)


@dataclass(frozen=True)
class ParsedCalls:
    values: dict[tuple[str, str], set[str]]
    call_count: int


def _build_call(name: str, arguments: Mapping[str, Any]) -> str:
    parts = [
        f"{key}={json.dumps(value, ensure_ascii=False)}"
        for key, value in arguments.items()
        if value is not None
    ]
    return f"{name}({', '.join(parts)})"


def _json_to_calls(value: Any) -> list[str]:
    if isinstance(value, list):
        calls: list[str] = []
        for item in value:
            calls.extend(_extract_calls(item) if isinstance(item, str) else _json_to_calls(item))
        return calls
    if not isinstance(value, dict):
        return []
    if isinstance(value.get("calls"), list):
        return _json_to_calls(value["calls"])

    lowered = {str(key).lower(): key for key in value}
    name_key = next(
        (lowered[key] for key in ("name", "tool", "function", "domain") if key in lowered),
        None,
    )
    args_key = next(
        (lowered[key] for key in ("arguments", "args", "parameters") if key in lowered),
        None,
    )
    if name_key is not None:
        name = value[name_key]
        if not isinstance(name, str):
            return []
        if args_key is not None and isinstance(value[args_key], dict):
            arguments = value[args_key]
        else:
            ignored = {
                "arguments", "args", "domain", "function", "name", "parameters",
                "raw_content", "reasoning", "tool",
            }
            arguments = {
                key: item
                for key, item in value.items()
                if str(key).lower() not in ignored
            }
        return [_build_call(name, arguments)]

    ignored = {
        "arguments", "args", "evaluation_result", "model_name", "parameters",
        "raw_content", "reasoning", "reasoning_tokens",
    }
    calls = []
    for name, arguments in value.items():
        if not isinstance(name, str) or name.lower() in ignored:
            continue
        if isinstance(arguments, list) and arguments and all(
            isinstance(item, dict) for item in arguments
        ):
            calls.extend(_build_call(name, item) for item in arguments)
        elif isinstance(arguments, dict):
            calls.append(_build_call(name, arguments))
        elif arguments is None:
            calls.append(f"{name}()")
    return calls


def _extract_calls(value: Any) -> list[str]:
    if isinstance(value, (dict, list)):
        return _json_to_calls(value)
    if not isinstance(value, str) or not value.strip():
        return []
    text = _FENCE_RE.sub("", _THINK_RE.sub("", value)).strip()
    text = _BRACED_CALL_RE.sub(r"\1(", text)
    try:
        parsed_json = json.loads(text)
    except json.JSONDecodeError:
        parsed_json = None
    if parsed_json is not None:
        calls = _json_to_calls(parsed_json)
        if calls:
            return calls
    return [match.group(0) for match in _CALL_RE.finditer(text)]


def _split_arguments(value: str) -> list[str]:
    parts: list[str] = []
    buffer: list[str] = []
    quote: str | None = None
    escaped = False
    for character in value:
        if escaped:
            buffer.append(character)
            escaped = False
        elif character == "\\":
            buffer.append(character)
            escaped = True
        elif quote is not None:
            buffer.append(character)
            if character == quote:
                quote = None
        elif character in ("'", '"'):
            buffer.append(character)
            quote = character
        elif character == ",":
            item = "".join(buffer).strip()
            if item:
                parts.append(item)
            buffer = []
        else:
            buffer.append(character)
    item = "".join(buffer).strip()
    if item:
        parts.append(item)
    return parts


def _normalize_value(slot: str, raw: str) -> str:
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    if value.lower() == "true":
        return "True"
    if value.lower() == "false":
        return "False"

    lowered_slot = slot.lower()
    if "time" in lowered_slot:
        match = _TIME_RE.fullmatch(value)
        if match:
            hour, minute = int(match.group(1)), int(match.group(2))
            marker = match.group(3)
        else:
            match = _HOUR_RE.fullmatch(value)
            if match:
                hour, minute, marker = int(match.group(1)), 0, match.group(2)
        if match:
            if marker:
                hour %= 12
                if marker.lower() == "p":
                    hour += 12
            if 0 <= hour < 24 and 0 <= minute < 60:
                return f"{hour:02d}:{minute:02d}"
    if "date" in lowered_slot or "day" in lowered_slot:
        match = _ISO_DATE_RE.fullmatch(value) or _SLASH_DATE_RE.fullmatch(value)
        if match:
            month, day = (int(part) for part in match.groups()[:2])
            if 1 <= month <= 12 and 1 <= day <= 31:
                return f"{month:02d}-{day:02d}"
    return value


def parse_calls(value: Any) -> ParsedCalls:
    """Parse legacy function/JSON shapes into a normalized slot/value map."""

    calls = _extract_calls(value)
    parsed: dict[tuple[str, str], set[str]] = defaultdict(set)
    for call in calls:
        match = _CALL_RE.fullmatch(call.strip())
        if not match:
            continue
        domain, arguments = match.groups()
        for item in _split_arguments(arguments):
            if "=" not in item:
                continue
            slot, raw = item.split("=", 1)
            slot = slot.strip()
            if slot:
                parsed[(domain.strip(), slot)].add(_normalize_value(slot, raw))
    return ParsedCalls(dict(parsed), len(calls))
