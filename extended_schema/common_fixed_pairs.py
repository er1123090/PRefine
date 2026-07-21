from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def load_python_module(module_path: str, module_name: str, extra_sys_path: str | None = None) -> Any:
    if extra_sys_path and extra_sys_path not in sys.path:
        sys.path.insert(0, extra_sys_path)

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_fixed_pairs_items(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if isinstance(raw_data, dict):
        items = raw_data.get("items")
        if isinstance(items, list):
            raw_data = items

    if not isinstance(raw_data, list):
        raise ValueError(f"Unsupported fixed-pairs format in {path}")

    items: List[Dict[str, Any]] = []
    for item in raw_data:
        if isinstance(item, dict):
            items.append(item)
    return items


def build_example_lookup(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        example_id = str(row.get("example_id", "")).strip()
        if example_id:
            lookup[example_id] = row
    return lookup


def get_sub_idx(item: Dict[str, Any]) -> int:
    example_id_sub = str(item.get("example_id_sub", "")).strip()
    if example_id_sub and "_" in example_id_sub:
        tail = example_id_sub.rsplit("_", 1)[1]
        if tail.isdigit():
            return int(tail)
    raise ValueError(f"Unable to infer sub_idx from fixed-pair item: {item}")


def get_singleturn_utterance(item: Dict[str, Any]) -> str:
    utterance = item.get("user_utterance")
    if utterance:
        return str(utterance)

    query_turns = item.get("query") or []
    if isinstance(query_turns, list):
        for turn in query_turns:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("role", "")).lower()
            message = turn.get("message") or turn.get("content")
            if role == "user" and message:
                return str(message).strip()

    raise ValueError(f"Fixed single-turn pair is missing a usable user utterance: {item}")


def get_singleturn_ground_truth(item: Dict[str, Any]) -> List[str]:
    ground_truth = item.get("reference_ground_truth")
    if isinstance(ground_truth, list) and ground_truth:
        return [str(value) for value in ground_truth]
    if isinstance(ground_truth, str) and ground_truth.strip():
        return [ground_truth.strip()]
    raise ValueError(f"Fixed single-turn pair is missing reference_ground_truth: {item}")


def load_multiturn_template_lookup(path: str) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if isinstance(raw_data, dict) and isinstance(raw_data.get("items"), list):
        iterable = raw_data["items"]
    elif isinstance(raw_data, list):
        iterable = raw_data
    elif isinstance(raw_data, dict):
        iterable = []
        for value in raw_data.values():
            if isinstance(value, list):
                iterable.extend(item for item in value if isinstance(item, dict))
            elif isinstance(value, dict):
                iterable.append(value)
    else:
        raise ValueError(f"Unsupported multiturn template format in {path}")

    lookup: Dict[str, Dict[str, Any]] = {}
    for item in iterable:
        query_id = item.get("query_id")
        if query_id:
            lookup[str(query_id)] = item
    return lookup


def get_multiturn_utterance(module: Any, item: Dict[str, Any]) -> str:
    utterance = item.get("user_utterance")
    if utterance:
        return str(utterance)

    query_turns = item.get("query") or []
    if isinstance(query_turns, list):
        return module.format_multiturn_dialogue(query_turns)

    raise ValueError(f"Fixed multiturn pair is missing a usable dialogue: {item}")


def build_multiturn_ground_truth(
    module: Any,
    template_lookup: Dict[str, Dict[str, Any]],
    fixed_item: Dict[str, Any],
) -> List[str]:
    query_id = str(fixed_item.get("query_id", "")).strip()
    template = template_lookup.get(query_id)
    if template is None:
        raise KeyError(f"Unable to find multiturn template for query_id={query_id}")

    api_calls = template.get("api_call") or []
    if not isinstance(api_calls, list) or not api_calls:
        raise ValueError(f"Template query_id={query_id} is missing api_call")

    domain = str(fixed_item.get("domain") or template.get("domain") or "").strip()
    if not domain:
        raise ValueError(f"Unable to infer domain for query_id={query_id}")

    _domain, base_args = module.parse_api_call_to_dict(str(api_calls[0]))
    slot_values_map = fixed_item.get("slot_values_map") or {}
    if not isinstance(slot_values_map, dict) or not slot_values_map:
        raise ValueError(f"Fixed multiturn pair is missing slot_values_map: {fixed_item}")

    normalized_slot_values: Dict[str, List[str]] = {}
    for slot, values in slot_values_map.items():
        if isinstance(values, list):
            normalized_slot_values[str(slot)] = [str(value) for value in values]
        else:
            normalized_slot_values[str(slot)] = [str(values)]

    return module.merge_and_generate_api_strings(domain, base_args, normalized_slot_values)
