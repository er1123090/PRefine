from __future__ import annotations

import ast
import hashlib
import itertools
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SEED = 2026072001
MODEL_ID = "google/gemma-4-12B-it"
MODEL_REVISION = "12ace6d648d72bd41519e140f1185f34d38c7e3d"
DIFFICULTIES = ("easy", "medium", "hard")
MODES = ("single", "multi")


class ContractError(ValueError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_bytes(value: Any) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_id(prefix: str, value: Any, length: int = 24) -> str:
    return f"{prefix}_{sha256_bytes(canonical_bytes(value))[:length]}"


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    with open(temp, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, target)
    directory_fd = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def write_jsonl_atomic(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    with open(temp, "wb") as handle:
        for row in rows:
            handle.write(canonical_bytes(row))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, target)
    directory_fd = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ContractError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def schema_map(tools: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for tool in tools:
        if tool.get("type") != "function" or not isinstance(tool.get("function"), Mapping):
            raise ContractError("every tool must be an OpenAI function declaration")
        function = dict(tool["function"])
        name = function.get("name")
        if not isinstance(name, str) or not name or name in result:
            raise ContractError(f"invalid or duplicate function name: {name!r}")
        result[name] = function
    return result


def parse_call_string(text: str) -> tuple[str, dict[str, Any]]:
    try:
        expression = ast.parse(text.strip(), mode="eval").body
    except SyntaxError as exc:
        raise ContractError(f"invalid call syntax: {text!r}") from exc
    if not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Name):
        raise ContractError(f"not a simple function call: {text!r}")
    if expression.args:
        raise ContractError(f"positional arguments are not allowed: {text!r}")
    arguments: dict[str, Any] = {}
    for keyword in expression.keywords:
        if keyword.arg is None or keyword.arg in arguments:
            raise ContractError(f"duplicate or expanded argument in {text!r}")
        try:
            value = ast.literal_eval(keyword.value)
        except (ValueError, TypeError) as exc:
            raise ContractError(f"non-literal argument {keyword.arg!r} in {text!r}") from exc
        arguments[keyword.arg] = value
    return expression.func.id, arguments


def _type_name(spec: Mapping[str, Any]) -> str:
    type_value = spec.get("type", "string")
    if isinstance(type_value, list):
        non_null = [item for item in type_value if item != "null"]
        if len(non_null) != 1:
            raise ContractError(f"unsupported union type: {type_value!r}")
        type_value = non_null[0]
    if not isinstance(type_value, str):
        raise ContractError(f"invalid schema type: {type_value!r}")
    return type_value


def normalize_scalar(value: Any, spec: Mapping[str, Any]) -> Any:
    expected = _type_name(spec)
    if expected == "boolean":
        if isinstance(value, bool):
            normalized = value
        elif isinstance(value, str) and value.strip().lower() in {"true", "false"}:
            normalized = value.strip().lower() == "true"
        else:
            raise ContractError(f"expected boolean, got {value!r}")
    elif expected == "integer":
        if isinstance(value, bool):
            raise ContractError(f"expected integer, got {value!r}")
        if isinstance(value, int):
            normalized = value
        elif isinstance(value, str) and value.strip().lstrip("-").isdigit():
            normalized = int(value)
        else:
            raise ContractError(f"expected integer, got {value!r}")
    elif expected == "number":
        if isinstance(value, bool):
            raise ContractError(f"expected number, got {value!r}")
        try:
            normalized = float(value)
        except (TypeError, ValueError) as exc:
            raise ContractError(f"expected number, got {value!r}") from exc
        if normalized.is_integer():
            normalized = int(normalized)
    elif expected == "string":
        if isinstance(value, (dict, list)):
            raise ContractError(f"expected string, got {value!r}")
        if isinstance(value, bool):
            normalized = "True" if value else "False"
        else:
            normalized = str(value)
    else:
        raise ContractError(f"unsupported scalar type: {expected}")

    if "enum" in spec:
        allowed = [normalize_scalar(item, {"type": expected}) for item in spec["enum"]]
        allowed_by_json = {canonical_json(item): item for item in allowed}
        key = canonical_json(normalized)
        if key not in allowed_by_json:
            raise ContractError(f"value {normalized!r} is outside enum {allowed!r}")
        normalized = allowed_by_json[key]
    return normalized


def normalize_arguments(
    function_name: str,
    arguments: Mapping[str, Any],
    schemas: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if function_name not in schemas:
        raise ContractError(f"unknown function: {function_name}")
    parameters = schemas[function_name].get("parameters") or {}
    properties = parameters.get("properties") or {}
    required = parameters.get("required") or []
    normalized: dict[str, Any] = {}
    for key in sorted(arguments):
        if key not in properties:
            raise ContractError(f"unknown slot {function_name}.{key}")
        normalized[key] = normalize_scalar(arguments[key], properties[key])
    missing = sorted(set(required) - set(normalized))
    if missing:
        raise ContractError(f"missing required slots for {function_name}: {missing}")
    return normalized


def normalize_call(
    function_name: str,
    arguments: Mapping[str, Any],
    schemas: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {"name": function_name, "arguments": normalize_arguments(function_name, arguments, schemas)}


def canonical_calls(calls: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    unique = {canonical_json(call): dict(call) for call in calls}
    return [unique[key] for key in sorted(unique)]


def calls_from_slot_values(
    domain: str,
    slot_values: Mapping[str, Sequence[Any]],
    schemas: Mapping[str, Mapping[str, Any]],
    base_arguments: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    normalized_values: dict[str, list[Any]] = {}
    properties = (schemas[domain].get("parameters") or {}).get("properties") or {}
    for slot in sorted(slot_values):
        if slot not in properties:
            raise ContractError(f"unknown slot {domain}.{slot}")
        values = {canonical_json(normalize_scalar(value, properties[slot])): normalize_scalar(value, properties[slot]) for value in slot_values[slot]}
        normalized_values[slot] = [values[key] for key in sorted(values)]
        if not normalized_values[slot]:
            raise ContractError(f"empty OR set for {domain}.{slot}")
    base = normalize_arguments(domain, base_arguments or {}, schemas)
    collisions = set(base) & set(normalized_values)
    for slot in sorted(collisions):
        values = normalized_values[slot]
        if len(values) != 1 or canonical_json(values[0]) != canonical_json(base[slot]):
            raise ContractError(f"conflicting explicit/preference slot {domain}.{slot}")
        del normalized_values[slot]
    if not normalized_values:
        return [normalize_call(domain, base, schemas)]
    keys = sorted(normalized_values)
    result = []
    for combination in itertools.product(*(normalized_values[key] for key in keys)):
        arguments = dict(base)
        arguments.update(dict(zip(keys, combination)))
        result.append(normalize_call(domain, arguments, schemas))
    return canonical_calls(result)


def call_to_string(call: Mapping[str, Any]) -> str:
    arguments = call.get("arguments") or {}
    pieces = [f"{key}={json.dumps(arguments[key], ensure_ascii=False)}" for key in sorted(arguments)]
    return f"{call['name']}({', '.join(pieces)})"


def flatten_multiturn(turns: Sequence[Mapping[str, Any]]) -> str:
    lines = []
    for turn in turns:
        role = str(turn.get("role") or "User")
        message = str(turn.get("message") or turn.get("content") or "")
        if not message:
            raise ContractError("multi-turn query contains an empty message")
        lines.append(f"{role}: {message}")
    if not lines:
        raise ContractError("multi-turn query is empty")
    return "\n".join(lines)


def dialogue_history(example: Mapping[str, Any]) -> str:
    blocks = []
    for index, session in enumerate(example.get("sessions") or [], 1):
        lines = [f"[Session {index}]"]
        for turn in session.get("dialogue") or []:
            role = str(turn.get("role") or "").capitalize()
            content = turn.get("message") or turn.get("content") or ""
            if role and content:
                lines.append(f"{role}: {content}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def past_api_calls(example: Mapping[str, Any]) -> str:
    result = []
    for index, session in enumerate(example.get("sessions") or [], 1):
        calls = session.get("api_call") or []
        if isinstance(calls, str):
            calls = [calls]
        for call in calls:
            if call:
                result.append(f"[Session {index}] {call}")
    return "\n".join(result)


def render_prompt(
    example: Mapping[str, Any],
    user_utterance: str,
    prompt_template: str,
    tools: Sequence[Mapping[str, Any]],
) -> str:
    context = (
        "\n--- Dialogue History ---\n"
        + dialogue_history(example)
        + "\n\n--- Past API Calls ---\n"
        + past_api_calls(example)
        + "\n"
    )
    return prompt_template.format(
        dialogue_history=context,
        user_utterance=user_utterance.strip(),
        preference_schema=json.dumps(tools, ensure_ascii=False, indent=2),
    )


def build_request(
    prompt: str,
    tools: Sequence[Mapping[str, Any]],
    max_tokens: int = 256,
    model_id: str = MODEL_ID,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "tools": list(tools),
        "tool_choice": "auto",
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "seed": SEED,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if reasoning_effort is not None:
        request["reasoning_effort"] = reasoning_effort
    return request


def current_dirty_snapshot(repo_root: str | Path) -> dict[str, Any]:
    repo = Path(repo_root)
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout
    paths: set[str] = set()
    for command in (["git", "diff", "--name-only"], ["git", "diff", "--cached", "--name-only"]):
        output = subprocess.run(command, cwd=repo, check=True, capture_output=True, text=True).stdout
        paths.update(line for line in output.splitlines() if line)
    tracked: dict[str, Any] = {}
    for relative in sorted(paths):
        path = repo / relative
        tracked[relative] = sha256_file(path) if path.is_file() else None
    return {
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "status_porcelain": status,
        "tracked_dirty_hashes": tracked,
    }

