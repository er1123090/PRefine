"""Validation reports for deterministic prepared datasets."""

from __future__ import annotations

from collections import Counter
import re
from typing import Any, Mapping, Sequence

from .build_instances import parse_api_call, semantic_multiset_sha256


_ARGUMENT_NAME = re.compile(r"(\w+)=")


class DatasetContractError(ValueError):
    """Raised when generated data differs from the frozen contract."""


def _schema_slots(schema: Sequence[Mapping[str, Any]]) -> dict[str, set[str]]:
    result = {}
    for tool in schema:
        function = tool.get("function", {})
        name = function.get("name")
        if name:
            result[str(name)] = set(
                function.get("parameters", {}).get("properties", {})
            )
    return result


def _record_missing_slots(
    record: Mapping[str, Any],
    allowed: Mapping[str, set[str]],
) -> set[str]:
    missing = set()
    for api_call in record["ground_truth"]:
        domain, _ = parse_api_call(api_call)
        slots = set(_ARGUMENT_NAME.findall(api_call))
        missing.update(slots - allowed.get(domain, set()))
    return missing


def _preference_coverage(
    rows: Sequence[Mapping[str, Any]],
    pref_groups: Mapping[str, Any],
) -> dict[str, Any]:
    supported = set(pref_groups)
    annotation_counts = Counter(
        preference.get("value_group")
        for row in rows
        for preference in row.get("api_calls_pref", [])
        if isinstance(preference, dict) and preference.get("value_group")
    )
    supported_sources = 0
    for row in rows:
        row_groups = {
            preference.get("value_group")
            for preference in row.get("api_calls_pref", [])
            if isinstance(preference, dict)
        }
        if row_groups & supported:
            supported_sources += 1
    return {
        "annotation_counts": dict(sorted(annotation_counts.items())),
        "ignored_groups": {
            group: annotation_counts[group]
            for group in sorted(set(annotation_counts) - supported)
        },
        "source_examples": {
            "supported": supported_sources,
            "unsupported_only": len(rows) - supported_sources,
        },
        "supported_groups": {
            group: annotation_counts[group] for group in sorted(supported)
        },
    }


def _multiturn_conflicts(templates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    conflict_ids = []
    for index, template in enumerate(templates):
        preference_slots = {
            target.get("slot")
            for target in template.get("target", [])
            if isinstance(target, dict) and target.get("slot")
        }
        base_slots = set()
        for api_call in template.get("api_call", []):
            if isinstance(api_call, str):
                base_slots.update(_ARGUMENT_NAME.findall(api_call))
        if base_slots & preference_slots:
            conflict_ids.append(str(template.get("query_id", index)))
    return {
        "conflict_count": len(conflict_ids),
        "conflicting_query_ids": conflict_ids,
        "template_count": len(templates),
    }


def build_validation_report(
    *,
    rows: Sequence[Mapping[str, Any]],
    scenarios: Mapping[str, Sequence[Mapping[str, Any]]],
    pref_groups: Mapping[str, Any],
    query_multiturn: Sequence[Mapping[str, Any]],
    schemas: Mapping[str, Sequence[Mapping[str, Any]]],
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate exact parity and return warnings that must remain visible."""

    actual_scenarios = {}
    schema_counts = {}
    missing_slots = {"singleturn": set(), "multiturn": set()}
    all_ids = []

    for name, records in scenarios.items():
        turn = name.split(".", 1)[0]
        source_examples = len({record["source_example_id"] for record in records})
        actual_scenarios[name] = {
            "count": len(records),
            "semantic_multiset_sha256": semantic_multiset_sha256(records),
            "source_examples": source_examples,
        }
        allowed = _schema_slots(schemas[turn])
        warning_count = 0
        for record in records:
            all_ids.append(record["instance_id"])
            record_missing = _record_missing_slots(record, allowed)
            if record_missing:
                warning_count += 1
                missing_slots[turn].update(record_missing)
        schema_counts[name] = warning_count

    total_instances = sum(value["count"] for value in actual_scenarios.values())
    conflicts = _multiturn_conflicts(query_multiturn)
    schema_total = sum(schema_counts.values())
    expected_scenarios = expected["scenarios"]
    parity_errors = []
    if len(rows) != expected["source_example_count"]:
        parity_errors.append(
            f"source examples: expected {expected['source_example_count']}, got {len(rows)}"
        )
    if total_instances != expected["total_instances"]:
        parity_errors.append(
            f"total instances: expected {expected['total_instances']}, got {total_instances}"
        )
    if len(all_ids) != len(set(all_ids)):
        parity_errors.append("canonical instance IDs are not globally unique")
    for name, wanted in expected_scenarios.items():
        actual = actual_scenarios.get(name)
        if actual != wanted:
            parity_errors.append(f"scenario {name}: expected {wanted!r}, got {actual!r}")
    expected_schema = expected["schema_missing_slot_instances"]
    if schema_counts != expected_schema["scenarios"] or schema_total != expected_schema["total"]:
        parity_errors.append(
            "schema warning count mismatch: "
            f"expected {expected_schema!r}, got total={schema_total}, scenarios={schema_counts!r}"
        )
    if conflicts["conflict_count"] != expected["multiturn_base_preference_conflicts"]:
        parity_errors.append(
            "multiturn base/preference conflict mismatch: "
            f"expected {expected['multiturn_base_preference_conflicts']}, "
            f"got {conflicts['conflict_count']}"
        )

    report = {
        "contract_parity": {
            "errors": parity_errors,
            "expected_total_instances": expected["total_instances"],
            "status": "ok" if not parity_errors else "error",
            "total_instances": total_instances,
        },
        "multiturn_base_preference_conflicts": conflicts,
        "preference_groups": _preference_coverage(rows, pref_groups),
        "scenarios": actual_scenarios,
        "schema_warnings": {
            "missing_slots": {
                turn: sorted(values) for turn, values in missing_slots.items()
            },
            "scenarios": schema_counts,
            "total": schema_total,
        },
    }
    if parity_errors:
        raise DatasetContractError("; ".join(parity_errors))
    return report
