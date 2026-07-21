"""
Conflict-majority task construction for experiment 5.

This module keeps the new conflict case separate from the legacy easy/medium/
hard assignment helpers.  A task is evaluator-ready: it contains the injected
query and the OR-allowed reference API calls for one majority preference
domain-slot.
"""

from __future__ import annotations

import copy
import json
import os
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .data_utils import (
    format_multiturn_dialogue,
    generate_func_strings,
    merge_and_generate_api_strings,
    parse_api_call_to_dict,
)


DIFFICULTIES = ("easy", "medium", "hard")
TURN_TYPES = ("singleturn", "multiturn")


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_conflict_tasks(path: str) -> List[Dict[str, Any]]:
    """Load either a plain task list or a build bundle with a ``tasks`` field."""
    raw = load_json(path)
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and isinstance(raw.get("tasks"), list):
        return raw["tasks"]
    raise ValueError(f"Unsupported conflict task file format: {path}")


def filter_tasks_by_example_id_sub(
    tasks: List[Dict[str, Any]],
    filter_ids: Optional[Sequence[str]],
) -> List[Dict[str, Any]]:
    if not filter_ids:
        return tasks

    by_id = {str(task.get("example_id_sub")): task for task in tasks}
    by_task_id = {str(task.get("task_id")): task for task in tasks}
    filtered: List[Dict[str, Any]] = []
    missing: List[str] = []

    for raw_id in filter_ids:
        wanted = str(raw_id).strip()
        task = by_id.get(wanted) or by_task_id.get(wanted)
        if task is None:
            missing.append(wanted)
            continue
        filtered.append(task)

    if missing:
        preview = ", ".join(missing[:10])
        print(f"[Warning] {len(missing)} task filter ids were not found. First missing: {preview}")
    return filtered


def limit_tasks(tasks: List[Dict[str, Any]], max_queries: Optional[int]) -> List[Dict[str, Any]]:
    if max_queries is None:
        return tasks
    if max_queries <= 0:
        raise ValueError(f"max_queries must be positive, got {max_queries}")
    if len(tasks) <= max_queries:
        return tasks
    if max_queries == 1:
        indices = [0]
    else:
        indices = [((len(tasks) - 1) * idx) // (max_queries - 1) for idx in range(max_queries)]
    return [tasks[idx] for idx in indices]


def normalize_pref_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def build_group_rule_targets(
    pref_group_data: Dict[str, Any],
) -> Dict[str, Dict[Tuple[str, str], List[str]]]:
    """Return group -> (domain, slot) -> allowed values, preserving config order."""
    groups: Dict[str, Dict[Tuple[str, str], List[str]]] = {}
    for group_name, group_data in pref_group_data.items():
        target_map: Dict[Tuple[str, str], List[str]] = {}
        for rule in group_data.get("rules", []):
            domain = rule.get("domain")
            slot = rule.get("slot")
            if not domain or not slot or rule.get("value") is None:
                continue
            key = (str(domain), str(slot))
            value = normalize_pref_value(rule.get("value"))
            target_map.setdefault(key, [])
            if value not in target_map[key]:
                target_map[key].append(value)
        groups[str(group_name)] = target_map
    return groups


def evidence_slot_counts(pref: Dict[str, Any]) -> Counter:
    """Count how many times each domain-slot appears in the selected value group."""
    counts: Counter = Counter()
    for evidence in pref.get("evidence", []) or []:
        domain = evidence.get("domain")
        slot = evidence.get("slot")
        if not domain or not slot:
            continue

        values = evidence.get("values")
        count = 0
        if isinstance(values, list) and values:
            for value_record in values:
                if isinstance(value_record, dict):
                    meta = value_record.get("meta") if isinstance(value_record.get("meta"), dict) else {}
                    raw_count = meta.get("count", value_record.get("count", 1))
                else:
                    raw_count = 1
                try:
                    count += int(raw_count)
                except (TypeError, ValueError):
                    count += 1
        else:
            raw_count = evidence.get("count", 1)
            try:
                count = int(raw_count)
            except (TypeError, ValueError):
                count = 1

        counts[(str(domain), str(slot))] += count
    return counts


def pref_majority_count(pref: Dict[str, Any]) -> int:
    try:
        return int(pref.get("count"))
    except (TypeError, ValueError):
        return int(sum(evidence_slot_counts(pref).values()))


def select_majority_prefs(
    example: Dict[str, Any],
    include_ties: bool = False,
) -> Tuple[List[Dict[str, Any]], Optional[str], Optional[int]]:
    prefs = example.get("api_calls_pref", [])
    if not isinstance(prefs, list) or not prefs:
        return [], "missing_api_calls_pref", None

    max_count = max(pref_majority_count(pref) for pref in prefs)
    selected = [pref for pref in prefs if pref_majority_count(pref) == max_count]

    if len(selected) > 1 and not include_ties:
        return [], "tie", max_count
    return selected, None, max_count


def classify_conflict_difficulty(target_seen_count: int, same_group_other_count: int) -> Optional[str]:
    if target_seen_count >= 2:
        return "easy"
    if target_seen_count == 1 and same_group_other_count >= 1:
        return "medium"
    if target_seen_count == 0 and same_group_other_count >= 2:
        return "hard"
    return None


def _safe_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_") or "x"


def get_multiturn_target_slots(template_data: Dict[str, Any]) -> set:
    slots = set()
    for target in template_data.get("target", []) or []:
        if isinstance(target, dict) and target.get("slot"):
            slots.add(str(target["slot"]))
    return slots


def select_multiturn_template(
    templates: Sequence[Dict[str, Any]],
    pref_slot_map: Dict[str, List[str]],
) -> Optional[Dict[str, Any]]:
    if not templates:
        return None

    target_slots = set(pref_slot_map)
    if not target_slots:
        return templates[0]

    best_template: Optional[Dict[str, Any]] = None
    best_score = -1
    for template in templates:
        score = len(target_slots & get_multiturn_target_slots(template))
        if score > best_score:
            best_template = template
            best_score = score
    return best_template or templates[0]


def build_query_and_ground_truth(
    turn_type: str,
    domain: str,
    slot: str,
    values: List[str],
    query_catalog: Dict[str, Any],
) -> Tuple[Optional[str], List[str], Dict[str, Any]]:
    pref_slot_map = {slot: values}

    if turn_type == "singleturn":
        query = query_catalog.get(domain)
        if not isinstance(query, str) or not query.strip():
            return None, [], {"skip_reason": "missing_singleturn_query"}
        return query.strip(), generate_func_strings(domain, pref_slot_map), {}

    if turn_type != "multiturn":
        raise ValueError(f"Unsupported turn_type: {turn_type}")

    raw_templates = query_catalog.get(domain, [])
    templates = raw_templates if isinstance(raw_templates, list) else [raw_templates]
    templates = [template for template in templates if isinstance(template, dict)]
    template = select_multiturn_template(templates, pref_slot_map)
    if not template:
        return None, [], {"skip_reason": "missing_multiturn_template"}

    query = format_multiturn_dialogue(template.get("query", []))
    base_api_str = ""
    api_calls = template.get("api_call") or []
    if isinstance(api_calls, list) and api_calls:
        base_api_str = str(api_calls[0])
    _, base_args = parse_api_call_to_dict(base_api_str)
    ground_truth = merge_and_generate_api_strings(domain, base_args, pref_slot_map)

    return query, ground_truth, {
        "query_id": template.get("query_id"),
        "base_api_call": base_api_str,
    }


def task_metadata(task: Dict[str, Any]) -> Dict[str, Any]:
    """Return compact task metadata suitable for attaching to inference output."""
    excluded = {"source_example"}
    return {key: copy.deepcopy(value) for key, value in task.items() if key not in excluded}


def build_conflict_majority_tasks(
    examples: Iterable[Dict[str, Any]],
    pref_group_data: Dict[str, Any],
    query_catalog: Dict[str, Any],
    turn_type: str,
    include_ties: bool = False,
    difficulties: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if turn_type not in TURN_TYPES:
        raise ValueError(f"turn_type must be one of {TURN_TYPES}, got {turn_type}")

    allowed_difficulties = set(difficulties or DIFFICULTIES)
    invalid = allowed_difficulties - set(DIFFICULTIES)
    if invalid:
        raise ValueError(f"Unknown difficulties: {sorted(invalid)}")

    group_targets = build_group_rule_targets(pref_group_data)
    tasks: List[Dict[str, Any]] = []
    skipped_examples: Counter = Counter()
    skipped_targets: Counter = Counter()
    tasks_by_difficulty: Counter = Counter()
    tasks_by_group: Counter = Counter()
    tie_examples: List[Dict[str, Any]] = []
    total_examples = 0

    for example in examples:
        total_examples += 1
        example_id = str(example.get("example_id", f"row_{total_examples - 1}"))
        selected_prefs, skip_reason, max_count = select_majority_prefs(
            example,
            include_ties=include_ties,
        )

        if skip_reason == "tie":
            prefs = example.get("api_calls_pref", []) or []
            tie_examples.append({
                "example_id": example_id,
                "count": max_count,
                "value_groups": [
                    str(pref.get("value_group"))
                    for pref in prefs
                    if pref_majority_count(pref) == max_count
                ],
            })

        if skip_reason:
            skipped_examples[skip_reason] += 1
            continue

        for pref in selected_prefs:
            group_name = str(pref.get("value_group"))
            if group_name not in group_targets:
                skipped_targets["missing_pref_group"] += 1
                continue

            evidence_counts = evidence_slot_counts(pref)
            group_total = int(sum(evidence_counts.values()))

            for (domain, slot), values in group_targets[group_name].items():
                target_seen_count = int(evidence_counts.get((domain, slot), 0))
                same_group_other_count = group_total - target_seen_count
                difficulty = classify_conflict_difficulty(
                    target_seen_count,
                    same_group_other_count,
                )
                if difficulty is None:
                    skipped_targets["unclassified"] += 1
                    continue
                if difficulty not in allowed_difficulties:
                    continue

                query, ground_truth, query_meta = build_query_and_ground_truth(
                    turn_type=turn_type,
                    domain=domain,
                    slot=slot,
                    values=values,
                    query_catalog=query_catalog,
                )
                if not query or not ground_truth:
                    skipped_targets[query_meta.get("skip_reason", "missing_query")] += 1
                    continue

                suffix = (
                    f"cm_{turn_type}_{difficulty}_{_safe_token(group_name)}_"
                    f"{_safe_token(domain)}_{_safe_token(slot)}"
                )
                example_id_sub = f"{example_id}_{suffix}"
                task = {
                    "task_id": example_id_sub,
                    "example_id": example_id,
                    "example_id_sub": example_id_sub,
                    "example_id_sub_suffix": suffix,
                    "turn_type": turn_type,
                    "difficulty": difficulty,
                    "majority_group": group_name,
                    "majority_count": int(max_count or pref_majority_count(pref)),
                    "target_domain": domain,
                    "target_slot": slot,
                    "target_values": list(values),
                    "target_seen_count": target_seen_count,
                    "same_group_other_count": same_group_other_count,
                    "source_group_total_count": group_total,
                    "query": query,
                    "reference_ground_truth": ground_truth,
                    "source_example": copy.deepcopy(example),
                }
                task.update(query_meta)
                tasks.append(task)
                tasks_by_difficulty[difficulty] += 1
                tasks_by_group[group_name] += 1

    summary = {
        "turn_type": turn_type,
        "strict_majority": not include_ties,
        "include_ties": include_ties,
        "total_examples": total_examples,
        "total_tasks": len(tasks),
        "tasks_by_difficulty": {difficulty: int(tasks_by_difficulty[difficulty]) for difficulty in DIFFICULTIES},
        "tasks_by_group": dict(sorted((key, int(value)) for key, value in tasks_by_group.items())),
        "skipped_examples": dict(sorted((key, int(value)) for key, value in skipped_examples.items())),
        "skipped_targets": dict(sorted((key, int(value)) for key, value in skipped_targets.items())),
        "tie_examples": tie_examples,
    }
    return tasks, summary


def build_task_bundle(
    tasks: List[Dict[str, Any]],
    summary: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "metadata": metadata or {},
        "summary": summary,
        "tasks": tasks,
    }


def write_task_bundle(path: str, bundle: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, ensure_ascii=False)
