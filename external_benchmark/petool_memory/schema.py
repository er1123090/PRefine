"""PEToolBench-specific memory schema and deterministic builder."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

from tool_utils import (
    aggregate_by_namespace,
    history_signal,
    operation_family,
    parse_tool_name,
    shorten,
)


PETOOL_MEMORY_SCHEMA_VERSION = "petool_memory_v1"

PETOOL_MEMORY_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": [
        "memory_version",
        "history_type",
        "preference_summary",
        "provider_preferences",
        "operation_preferences",
        "candidate_selection_rules",
    ],
    "properties": {
        "memory_version": {"const": PETOOL_MEMORY_SCHEMA_VERSION},
        "history_type": {"enum": ["p", "r", "c"]},
        "preference_summary": {
            "type": "object",
            "required": ["primary_rule", "history_interpretation"],
        },
        "provider_preferences": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "namespace",
                    "polarity",
                    "confidence",
                    "support_count",
                    "negative_count",
                    "operation_families",
                ],
            },
        },
        "operation_preferences": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "operation_family",
                    "preferred_namespaces",
                    "avoid_namespaces",
                    "lower_priority_namespaces",
                    "selection_rule",
                ],
            },
        },
        "negative_evidence": {"type": "array"},
        "tool_call_patterns": {"type": "array"},
        "candidate_selection_rules": {"type": "array"},
    },
}


def history_interpretation(history_type: str) -> str:
    if history_type == "r":
        return (
            "Binary ratings define preference polarity: rating 1 is positive "
            "provider/tool evidence and rating 0 is negative evidence."
        )
    if history_type == "c":
        return (
            "History is chronological and later calls are stronger preference "
            "evidence than earlier calls, especially within the same operation family."
        )
    return "Observed history calls are treated as preferred successful tool-use evidence."


def _provider_polarity(history_type: str, score: float, support: int, negative: int) -> str:
    if negative and score < 0:
        return "avoid"
    if history_type == "c":
        return "recency_prefer" if support else "unknown"
    if support and negative:
        return "weak_prefer" if score >= 0 else "avoid"
    if support:
        return "prefer"
    return "unknown"


def _confidence(score: float, support: int, negative: int) -> float:
    total = max(support + negative, 1)
    raw = abs(score) / total
    return round(max(0.1, min(1.0, raw)), 3)


def _provider_preferences(history_type: str, evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    aggregate = aggregate_by_namespace(evidence)
    rows = []
    for namespace, item in aggregate.items():
        support = int(item["support_count"])
        negative = int(item["negative_count"])
        score = float(item["score"])
        rows.append(
            {
                "namespace": namespace,
                "category": item["category"],
                "provider": item["provider"],
                "polarity": _provider_polarity(history_type, score, support, negative),
                "confidence": _confidence(score, support, negative),
                "support_count": support,
                "negative_count": negative,
                "score": round(score, 4),
                "latest_history_index": item["latest_history_index"],
                "operation_families": sorted(item["operation_families"]),
                "evidence_indices": item["evidence_indices"],
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row["polarity"] in {"prefer", "recency_prefer", "weak_prefer"},
            row["score"],
            row["latest_history_index"],
        ),
        reverse=True,
    )


def _operation_preferences(history_type: str, evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_family: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(
        lambda: defaultdict(
            lambda: {
                "score": 0.0,
                "support_count": 0,
                "negative_count": 0,
                "latest_history_index": -1,
                "evidence_indices": [],
            }
        )
    )
    for item in evidence:
        family_bucket = by_family[item["operation_family"]][item["namespace"]]
        family_bucket["score"] += item["weight"]
        if item["weight"] > 0:
            family_bucket["support_count"] += 1
        elif item["weight"] < 0:
            family_bucket["negative_count"] += 1
        family_bucket["latest_history_index"] = max(
            family_bucket["latest_history_index"], item["history_index"]
        )
        family_bucket["evidence_indices"].append(item["history_index"])

    rows = []
    for family, namespaces in by_family.items():
        ranked = sorted(
            namespaces.items(),
            key=lambda pair: (pair[1]["score"], pair[1]["latest_history_index"]),
            reverse=True,
        )
        avoid = [
            namespace
            for namespace, stats in ranked
            if stats["negative_count"] > 0 and stats["score"] < 0
        ]
        if history_type == "c":
            recency_ranked = sorted(
                namespaces.items(),
                key=lambda pair: pair[1]["latest_history_index"],
                reverse=True,
            )
            preferred = [recency_ranked[0][0]] if recency_ranked else []
            lower_priority = [namespace for namespace, _ in recency_ranked[1:]]
            rule = (
                "For this operation family, prefer the namespace with the latest "
                "chronological evidence; earlier namespaces are lower priority."
            )
        else:
            preferred = [
                namespace
                for namespace, stats in ranked
                if stats["score"] > 0 and namespace not in avoid
            ][:3]
            lower_priority = []
            rule = (
                "Among semantically equivalent candidate tools in this operation family, "
                "prefer positive namespaces and avoid negative namespaces."
            )
        rows.append(
            {
                "operation_family": family,
                "preferred_namespaces": preferred,
                "avoid_namespaces": avoid,
                "lower_priority_namespaces": lower_priority,
                "namespace_scores": {
                    namespace: {
                        "score": round(stats["score"], 4),
                        "support_count": stats["support_count"],
                        "negative_count": stats["negative_count"],
                        "latest_history_index": stats["latest_history_index"],
                    }
                    for namespace, stats in namespaces.items()
                },
                "selection_rule": rule,
            }
        )
    return sorted(rows, key=lambda row: row["operation_family"])


def build_deterministic_petool_memory(record: Dict[str, Any]) -> Dict[str, Any]:
    history = record.get("history", []) or []
    history_type = record.get("history_type", "p")
    evidence = []
    for index, turn in enumerate(history):
        call = turn.get("tool_call", {}) or {}
        parsed = parse_tool_name(str(call.get("tool_name", "")))
        family = operation_family(parsed["operation"], str(turn.get("instruction", "")))
        signal, weight = history_signal(history_type, turn.get("rating"), index, len(history))
        evidence.append(
            {
                "history_index": index,
                "instruction": shorten(str(turn.get("instruction", ""))),
                "tool_name": parsed["tool_name"],
                "namespace": parsed["namespace"],
                "category": parsed["category"],
                "provider": parsed["provider"],
                "operation": parsed["operation"],
                "operation_family": family,
                "parameters": call.get("parameters", {}),
                "rating": turn.get("rating"),
                "signal": signal,
                "weight": weight,
            }
        )

    negative_evidence = [
        {
            "history_index": item["history_index"],
            "tool_name": item["tool_name"],
            "namespace": item["namespace"],
            "operation_family": item["operation_family"],
            "rating": item["rating"],
            "reason": "rating 0 marks this tool/provider as dispreferred",
        }
        for item in evidence
        if item["weight"] < 0
    ]

    return {
        "memory_version": PETOOL_MEMORY_SCHEMA_VERSION,
        "history_type": history_type,
        "preference_summary": {
            "primary_rule": (
                "First satisfy the current query semantics. When candidate tools are "
                "semantically equivalent, choose the provider/namespace supported by "
                "PEToolBench preference evidence."
            ),
            "history_interpretation": history_interpretation(history_type),
        },
        "provider_preferences": _provider_preferences(history_type, evidence),
        "operation_preferences": _operation_preferences(history_type, evidence),
        "negative_evidence": negative_evidence,
        "tool_call_patterns": evidence,
        "candidate_selection_rules": [
            "Reject tools that do not match the current user query even if their provider is preferred.",
            "For same-operation candidates, prefer namespaces listed in operation_preferences.preferred_namespaces.",
            "Avoid namespaces listed in avoid_namespaces when an equivalent non-avoided candidate exists.",
            "For chronological histories, later evidence in the same operation family overrides earlier evidence.",
            "Current explicit query arguments override historical parameter values.",
        ],
        "generation_metadata": {
            "builder": "deterministic_petool_memory",
            "history_length": len(history),
            "source_index": record.get("source_index"),
        },
    }


def validate_petool_memory(memory: Dict[str, Any]) -> List[str]:
    errors = []
    if not isinstance(memory, dict):
        return ["memory must be a JSON object"]
    if memory.get("memory_version") != PETOOL_MEMORY_SCHEMA_VERSION:
        errors.append(f"memory_version must be {PETOOL_MEMORY_SCHEMA_VERSION}")
    for key in PETOOL_MEMORY_JSON_SCHEMA["required"]:
        if key not in memory:
            errors.append(f"missing required key: {key}")
    if not isinstance(memory.get("provider_preferences", []), list):
        errors.append("provider_preferences must be a list")
    if not isinstance(memory.get("operation_preferences", []), list):
        errors.append("operation_preferences must be a list")
    if not isinstance(memory.get("candidate_selection_rules", []), list):
        errors.append("candidate_selection_rules must be a list")
    return errors


def render_memory_for_prompt(memory: Dict[str, Any]) -> str:
    preferred = []
    avoid = []
    for item in memory.get("provider_preferences", []) or []:
        line = (
            f"{item.get('namespace')} polarity={item.get('polarity')} "
            f"confidence={item.get('confidence')} families={item.get('operation_families')}"
        )
        if item.get("polarity") == "avoid":
            avoid.append(line)
        else:
            preferred.append(line)

    operation_lines = []
    for item in memory.get("operation_preferences", []) or []:
        operation_lines.append(
            f"{item.get('operation_family')}: prefer={item.get('preferred_namespaces', [])}; "
            f"avoid={item.get('avoid_namespaces', [])}; "
            f"lower_priority={item.get('lower_priority_namespaces', [])}"
        )

    sections = [
        f"Schema: {memory.get('memory_version')}",
        f"History interpretation: {memory.get('preference_summary', {}).get('history_interpretation', '')}",
        "Provider preferences:\n" + "\n".join(preferred[:12]),
        "Provider avoid evidence:\n" + "\n".join(avoid[:8]),
        "Operation-family preferences:\n" + "\n".join(operation_lines[:12]),
        "Selection rules:\n" + "\n".join(memory.get("candidate_selection_rules", [])),
    ]
    return "\n\n".join(section for section in sections if section.strip())

