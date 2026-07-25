from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence


def _majority_value(example: Mapping[str, Any]) -> Optional[Any]:
    meta = example.get("meta")
    if isinstance(meta, Mapping) and meta.get("majority") is not None:
        return meta["majority"]
    if example.get("majority") is not None:
        return example["majority"]
    return None


def _same_value(left: Any, right: Any) -> bool:
    return str(left) == str(right)


def _evidence_contains_value(pref: Mapping[str, Any], target: Any) -> bool:
    for evidence in pref.get("evidence", []) or []:
        if not isinstance(evidence, Mapping):
            continue
        if evidence.get("value") is not None and _same_value(
            evidence["value"], target
        ):
            return True
        for value_record in evidence.get("values", []) or []:
            if (
                isinstance(value_record, Mapping)
                and value_record.get("value") is not None
                and _same_value(value_record["value"], target)
            ):
                return True
    return False


def has_query_majority(example: Mapping[str, Any]) -> bool:
    return _majority_value(example) is not None


def select_query_preferences(example: Mapping[str, Any]) -> List[Dict[str, Any]]:
    prefs = example.get("api_calls_pref", [])
    if not isinstance(prefs, list):
        return []

    valid_prefs = [pref for pref in prefs if isinstance(pref, dict)]
    majority = _majority_value(example)
    if majority is None:
        return valid_prefs

    return [
        pref
        for pref in valid_prefs
        if _same_value(pref.get("value_group"), majority)
        or _evidence_contains_value(pref, majority)
    ]


def select_query_rules(
    example: Mapping[str, Any],
    pref: Mapping[str, Any],
    rules: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    majority = _majority_value(example)
    if majority is None or _same_value(pref.get("value_group"), majority):
        return list(rules)

    if not _evidence_contains_value(pref, majority):
        return []

    return [
        rule
        for rule in rules
        if isinstance(rule, dict)
        and rule.get("value") is not None
        and _same_value(rule["value"], majority)
    ]
