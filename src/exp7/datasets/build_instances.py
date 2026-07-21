"""Pure, deterministic mix600 query and ground-truth construction.

This module reimplements the frozen legacy semantics without importing any
historical runner.  Positional ordering accidents from legacy ``set`` usage
are deliberately normalized while query and GT semantics stay unchanged.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import itertools
import json
import re
from typing import Any, Iterable, Mapping, Sequence


TURNS = ("singleturn", "multiturn")
DIFFICULTIES = ("easy", "medium", "hard")
_API_ARGUMENT = re.compile(r"(\w+)=[\"']([^\"']+)[\"']")


class DatasetBuildError(ValueError):
    """Raised when an input or requested condition violates the contract."""


def canonical_json(value: Any) -> str:
    """Return the compact canonical JSON representation used by G001."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def lines_sha256(lines: Iterable[str]) -> str:
    payload = "\n".join(lines) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def semantic_multiset_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    """Hash the G001 semantic projection while retaining duplicate records."""

    values = []
    for record in records:
        values.append(
            canonical_json(
                {
                    "example_id": record["source_example_id"],
                    "ground_truth": sorted(record["ground_truth"]),
                    "query": record["query"],
                }
            )
        )
    return lines_sha256(sorted(values))


def _as_legacy_string(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def parse_api_call(api_call: str) -> tuple[str, dict[str, str]]:
    if "(" not in api_call:
        return api_call.strip(), {}
    domain = api_call.split("(", 1)[0].strip()
    try:
        arguments = api_call.split("(", 1)[1].rsplit(")", 1)[0]
    except IndexError:
        return domain, {}
    return domain, {key: value for key, value in _API_ARGUMENT.findall(arguments)}


def _format_api_call(domain: str, arguments: Mapping[str, str]) -> str:
    parts = [f'{key}="{arguments[key]}"' for key in sorted(arguments)]
    return f"{domain}({', '.join(parts)})"


def _generate_preference_calls(
    domain: str,
    slot_values: Mapping[str, Sequence[str]],
    base_arguments: Mapping[str, str] | None = None,
) -> list[str]:
    if not slot_values:
        return []
    slots = sorted(slot_values)
    values = [slot_values[slot] for slot in slots]
    calls = []
    for combination in itertools.product(*values):
        arguments = dict(base_arguments or {})
        arguments.update(zip(slots, combination))
        calls.append(_format_api_call(domain, arguments))
    return calls


def group_multiturn_templates(raw_templates: Any) -> dict[str, list[dict[str, Any]]]:
    """Normalize list/dict query JSON into the legacy domain mapping."""

    if isinstance(raw_templates, dict):
        grouped: dict[str, list[dict[str, Any]]] = {}
        for domain, templates in raw_templates.items():
            values = templates if isinstance(templates, list) else [templates]
            grouped[str(domain)] = [value for value in values if isinstance(value, dict)]
        return grouped
    if not isinstance(raw_templates, list):
        return {}

    grouped = {}
    for template in raw_templates:
        if not isinstance(template, dict):
            continue
        domain = None
        for target in template.get("target", []):
            if isinstance(target, dict) and target.get("domain"):
                domain = str(target["domain"])
                break
        if not domain:
            api_calls = template.get("api_call", [])
            if isinstance(api_calls, list) and api_calls and isinstance(api_calls[0], str):
                domain, _ = parse_api_call(api_calls[0])
        if domain:
            grouped.setdefault(domain, []).append(template)
    return grouped


def _target_slots(template: Mapping[str, Any]) -> set[str]:
    return {
        str(target["slot"])
        for target in template.get("target", [])
        if isinstance(target, dict) and target.get("slot")
    }


def _select_template(
    templates: Sequence[Mapping[str, Any]],
    preference_slots: set[str],
) -> Mapping[str, Any] | None:
    if not templates:
        return None
    if not preference_slots:
        return templates[0]
    best = None
    best_score = 0
    for template in templates:
        score = len(preference_slots & _target_slots(template))
        if score > best_score:
            best = template
            best_score = score
    return best or templates[0]


def _format_dialogue(turns: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(
        f"{turn.get('role', 'User')}: {turn.get('message', '')}"
        for turn in turns
    )


def _multiturn_pair(
    domain: str,
    slot_values: Mapping[str, Sequence[str]],
    templates_by_domain: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[str, list[str]] | None:
    templates = templates_by_domain.get(domain, [])
    template = _select_template(templates, set(slot_values))
    if template is None:
        return None
    api_calls = template.get("api_call", [""])
    base_call = api_calls[0] if isinstance(api_calls, list) and api_calls else ""
    _, base_arguments = parse_api_call(str(base_call))
    return (
        _format_dialogue(template.get("query", [])),
        _generate_preference_calls(domain, slot_values, base_arguments),
    )


def _easy_slot_maps(
    example: Mapping[str, Any],
    pref_list: Mapping[str, Sequence[str]],
    available_domains: set[str],
) -> list[tuple[str, dict[str, list[str]]]]:
    results = []
    api_calls = example.get("api_calls", [])
    if not isinstance(api_calls, list):
        return results
    for call in api_calls:
        if not isinstance(call, str):
            continue
        domain, arguments = parse_api_call(call)
        if domain not in available_domains or domain not in pref_list:
            continue
        slots = {
            slot: [_as_legacy_string(value)]
            for slot, value in arguments.items()
            if slot in pref_list[domain]
        }
        if slots:
            results.append((domain, slots))
    return results


def _medium_slot_maps(
    example: Mapping[str, Any],
    pref_groups: Mapping[str, Mapping[str, Any]],
    available_domains: set[str],
) -> list[tuple[str, dict[str, list[str]]]]:
    results = []
    preferences = example.get("api_calls_pref", [])
    if not isinstance(preferences, list):
        return results
    for preference in preferences:
        if not isinstance(preference, dict):
            continue
        group = pref_groups.get(str(preference.get("value_group")))
        if not group:
            continue
        rules = group.get("rules", [])
        domain_slots: dict[str, dict[str, set[str]]] = {}
        for evidence in preference.get("evidence", []):
            if not isinstance(evidence, dict):
                continue
            domain = evidence.get("domain")
            slot = evidence.get("slot")
            if not domain or not slot or domain not in available_domains:
                continue
            values = {
                _as_legacy_string(rule.get("value"))
                for rule in rules
                if isinstance(rule, dict)
                and rule.get("domain") == domain
                and rule.get("slot") == slot
            }
            if values:
                domain_slots.setdefault(str(domain), {}).setdefault(str(slot), set()).update(values)
        for domain, slots in domain_slots.items():
            results.append(
                (domain, {slot: sorted(values) for slot, values in slots.items()})
            )
    return results


def _hard_slot_maps(
    example: Mapping[str, Any],
    pref_groups: Mapping[str, Mapping[str, Any]],
    available_domains: set[str],
) -> list[tuple[str, dict[str, list[str]]]]:
    results = []
    preferences = example.get("api_calls_pref", [])
    if not isinstance(preferences, list):
        return results
    for preference in preferences:
        if not isinstance(preference, dict):
            continue
        group = pref_groups.get(str(preference.get("value_group")))
        if not group:
            continue
        rules = group.get("rules", [])
        used_domains = {
            evidence.get("domain")
            for evidence in preference.get("evidence", [])
            if isinstance(evidence, dict) and evidence.get("domain")
        }
        candidate_domains = sorted(
            {
                str(rule["domain"])
                for rule in rules
                if isinstance(rule, dict)
                and rule.get("domain") in available_domains
                and rule.get("domain") not in used_domains
            }
        )
        for domain in candidate_domains:
            slots: dict[str, list[str]] = {}
            for rule in rules:
                if not isinstance(rule, dict) or rule.get("domain") != domain:
                    continue
                slot = rule.get("slot")
                value = rule.get("value")
                if not slot or value is None:
                    continue
                normalized = _as_legacy_string(value)
                if normalized not in slots.setdefault(str(slot), []):
                    slots[str(slot)].append(normalized)
            if slots:
                results.append((domain, slots))
    return results


def build_pairs(
    example: Mapping[str, Any],
    *,
    turn: str,
    difficulty: str,
    query_singleturn: Mapping[str, str],
    query_multiturn: Mapping[str, Sequence[Mapping[str, Any]]],
    pref_list: Mapping[str, Sequence[str]],
    pref_groups: Mapping[str, Mapping[str, Any]],
) -> list[tuple[str, list[str]]]:
    """Build legacy-semantic query/GT pairs for one source example."""

    if turn not in TURNS:
        raise DatasetBuildError(f"unsupported turn: {turn!r}")
    if difficulty not in DIFFICULTIES:
        raise DatasetBuildError(f"unsupported difficulty: {difficulty!r}")

    available = set(query_singleturn if turn == "singleturn" else query_multiturn)
    if difficulty == "easy":
        slot_maps = _easy_slot_maps(example, pref_list, available)
    elif difficulty == "medium":
        slot_maps = _medium_slot_maps(example, pref_groups, available)
    else:
        slot_maps = _hard_slot_maps(example, pref_groups, available)

    pairs = []
    for domain, slots in slot_maps:
        if turn == "singleturn":
            calls = _generate_preference_calls(domain, slots)
            pairs.append((query_singleturn[domain], calls))
        else:
            pair = _multiturn_pair(domain, slots, query_multiturn)
            if pair is not None:
                pairs.append(pair)
    return pairs


def _stable_pair_key(pair: tuple[str, list[str]]) -> str:
    query, ground_truth = pair
    return canonical_json({"ground_truth": sorted(ground_truth), "query": query})


def build_scenario(
    rows: Sequence[Mapping[str, Any]],
    *,
    dataset_id: str,
    turn: str,
    difficulty: str,
    query_singleturn: Mapping[str, str],
    query_multiturn: Mapping[str, Sequence[Mapping[str, Any]]],
    pref_list: Mapping[str, Sequence[str]],
    pref_groups: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build one deterministic scenario with stable, condition-scoped IDs."""

    records = []
    canonical_ids: set[str] = set()
    for source_index, example in enumerate(rows):
        example_id = str(example.get("example_id", "unknown"))
        pairs = sorted(
            build_pairs(
                example,
                turn=turn,
                difficulty=difficulty,
                query_singleturn=query_singleturn,
                query_multiturn=query_multiturn,
                pref_list=pref_list,
                pref_groups=pref_groups,
            ),
            key=_stable_pair_key,
        )
        occurrences: Counter[str] = Counter()
        for sub_index, (query, ground_truth) in enumerate(pairs):
            ground_truth = sorted(ground_truth)
            semantic_payload = canonical_json(
                {"ground_truth": ground_truth, "query": query}
            )
            semantic_sha256 = hashlib.sha256(semantic_payload.encode("utf-8")).hexdigest()
            occurrence = occurrences[semantic_sha256]
            occurrences[semantic_sha256] += 1
            instance_id = (
                f"{dataset_id}:{turn}:{difficulty}:{example_id}:"
                f"{semantic_sha256}:{occurrence}"
            )
            if instance_id in canonical_ids:
                raise DatasetBuildError(f"duplicate canonical instance ID: {instance_id}")
            canonical_ids.add(instance_id)
            records.append(
                {
                    "dataset_id": dataset_id,
                    "difficulty": difficulty,
                    "ground_truth": ground_truth,
                    "instance_id": instance_id,
                    "legacy_example_id_sub": f"{example_id}_{sub_index}",
                    "query": query,
                    "semantic_sha256": semantic_sha256,
                    "source_example": dict(example),
                    "source_example_id": example_id,
                    "source_index": source_index,
                    "turn": turn,
                }
            )
    return records


def build_all_scenarios(
    rows: Sequence[Mapping[str, Any]],
    *,
    dataset_id: str,
    turns: Sequence[str],
    difficulties: Sequence[str],
    query_singleturn: Mapping[str, str],
    query_multiturn_raw: Any,
    pref_list: Mapping[str, Sequence[str]],
    pref_groups: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    query_multiturn = group_multiturn_templates(query_multiturn_raw)
    scenarios = {}
    all_ids: set[str] = set()
    for turn in turns:
        for difficulty in difficulties:
            name = f"{turn}.{difficulty}"
            records = build_scenario(
                rows,
                dataset_id=dataset_id,
                turn=turn,
                difficulty=difficulty,
                query_singleturn=query_singleturn,
                query_multiturn=query_multiturn,
                pref_list=pref_list,
                pref_groups=pref_groups,
            )
            ids = {record["instance_id"] for record in records}
            overlap = all_ids & ids
            if overlap:
                raise DatasetBuildError(f"cross-scenario duplicate IDs: {sorted(overlap)[:1]}")
            all_ids.update(ids)
            scenarios[name] = records
    return scenarios
