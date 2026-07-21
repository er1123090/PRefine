"""Conservative tool-call parsing and schema introspection."""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Iterable
from typing import Any


_CALL_RE = re.compile(r"([A-Za-z_][\w.]*)\s*\(([^()]*)\)", re.DOTALL)
_KV_RE = re.compile(
    r"([A-Za-z_]\w*)\s*=\s*(\"(?:\\.|[^\"])*\"|'(?:\\.|[^'])*'|[^,]+)"
)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_SPACE_RE = re.compile(r"\s+")


def normalize_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    else:
        text = str(value)
    return _SPACE_RE.sub(" ", text.strip()).casefold()


def _literal(token: str) -> Any:
    token = token.strip()
    try:
        return ast.literal_eval(token)
    except (ValueError, SyntaxError):
        return token.strip("\"'")


def parse_call_string(text: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for match in _CALL_RE.finditer(_THINK_RE.sub("", text)):
        name = match.group(1)
        arguments = {key: _literal(raw) for key, raw in _KV_RE.findall(match.group(2))}
        calls.append({"name": name, "arguments": arguments})
    return calls


def _json_calls(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        result: list[dict[str, Any]] = []
        for item in value:
            result.extend(_json_calls(item))
        return result
    if not isinstance(value, dict):
        return []

    if isinstance(value.get("function"), dict):
        function = value["function"]
        name = function.get("name")
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {key: _literal(raw) for key, raw in _KV_RE.findall(arguments)}
        if isinstance(name, str) and isinstance(arguments, dict):
            return [{"name": name, "arguments": arguments}]

    name = value.get("name") or value.get("function_name") or value.get("domain")
    arguments = value.get("arguments") or value.get("parameters") or value.get("args")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {key: _literal(raw) for key, raw in _KV_RE.findall(arguments)}
    if isinstance(name, str) and isinstance(arguments, dict):
        return [{"name": name, "arguments": arguments}]

    result: list[dict[str, Any]] = []
    for key in ("tool_calls", "calls", "api_calls", "output"):
        if key in value:
            result.extend(_json_calls(value[key]))
    return result


def extract_calls(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, (dict, list)):
        calls = _json_calls(value)
        if calls:
            return calls
        if isinstance(value, list):
            result: list[dict[str, Any]] = []
            for item in value:
                result.extend(extract_calls(item))
            return result
        return []
    if not isinstance(value, str):
        return []
    text = _THINK_RE.sub("", value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if parsed is not None:
        calls = _json_calls(parsed)
        if calls:
            return calls
    return parse_call_string(text)


def call_to_string(name: str, arguments: dict[str, Any]) -> str:
    parts = [f"{key}={json.dumps(value, ensure_ascii=False)}" for key, value in sorted(arguments.items())]
    return f"{name}({', '.join(parts)})"


def slot_value_map(value: Any) -> dict[tuple[str, str], set[str]]:
    result: dict[tuple[str, str], set[str]] = {}
    for call in extract_calls(value):
        domain = str(call["name"])
        for slot, raw in call["arguments"].items():
            normalized = normalize_value(raw)
            if normalized:
                result.setdefault((domain, str(slot)), set()).add(normalized)
    return result


def schema_domain_slots(schema: Any) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    tools: Iterable[Any]
    if isinstance(schema, dict) and isinstance(schema.get("tools"), list):
        tools = schema["tools"]
    elif isinstance(schema, list):
        tools = schema
    else:
        tools = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = fn.get("name")
        parameters = fn.get("parameters", {})
        properties = parameters.get("properties", {}) if isinstance(parameters, dict) else {}
        if isinstance(name, str) and isinstance(properties, dict):
            result[name] = {str(slot) for slot in properties}
    return result

