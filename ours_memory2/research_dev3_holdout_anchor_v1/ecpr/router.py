"""Schema-, domain-, and confidence-gated typed evidence routing."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from .contracts import ALLOWED_ABLATIONS, TASK_FIELDS, CandidatePolicy
from .parsing import schema_domain_slots
from .prompts import lexical_count
from .query_gate import GateDecision, gate_public_query


def route_hypotheses(
    memory: dict[str, Any],
    task: dict[str, Any],
    schema: Any,
    preference_slots: dict[str, list[str]],
    policy: CandidatePolicy,
    ontology: dict[str, Any] | None = None,
    ablations: set[str] | None = None,
    *,
    decision: GateDecision | None = None,
) -> list[dict[str, Any]]:
    if set(task) != TASK_FIELDS:
        raise ValueError(
            f"router task field contract violation: {sorted(set(task))}"
        )
    ablations = set(ablations or ())
    if ontology is None:
        raise ValueError("public domain ontology is required")
    unknown = ablations - ALLOWED_ABLATIONS
    if unknown:
        raise ValueError(f"unknown ablation flags: {sorted(unknown)}")
    computed_decision = gate_public_query(
        str(task["query"]), str(task["mode"]), schema, ontology
    )
    if decision is not None and decision != computed_decision:
        raise ValueError("caller-supplied query-gate decision mismatch")
    decision = computed_decision
    if not decision.selected:
        return []
    if "no_typed" in ablations:
        return []

    schema_slots_by_domain = schema_domain_slots(schema)
    candidates: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for hypothesis in memory.get("typed_hypotheses", []):
        if not isinstance(hypothesis, dict):
            continue
        domain = str(hypothesis.get("domain", ""))
        slot = str(hypothesis.get("slot", ""))
        if domain != decision.schema_domain:
            continue
        schema_slots = schema_slots_by_domain.get(domain, set())
        pref_slots = {
            str(value) for value in preference_slots.get(domain, [])
        }
        if slot not in schema_slots or slot not in pref_slots:
            continue
        support = int(hypothesis.get("support", 0))
        counter = int(hypothesis.get("counterevidence", 0))
        confidence = float(hypothesis.get("confidence", 0.0))
        conflict_ratio = counter / (support + counter) if support + counter else 1.0
        if "no_routing" not in ablations:
            if support < policy.minimum_support or confidence < policy.minimum_confidence:
                continue
            if "no_counterevidence" not in ablations and conflict_ratio > policy.maximum_conflict_ratio:
                continue
        candidates[(domain, slot)].append(hypothesis)

    selected: list[dict[str, Any]] = []
    for domain_slot in sorted(candidates):
        items = sorted(
            candidates[domain_slot],
            key=lambda item: (
                -float(item.get("confidence", 0.0)),
                -int(item.get("support", 0)),
                -int(item.get("last_seen", 0)),
                str(item.get("value", "")),
            ),
        )
        if "no_routing" not in ablations and len(items) > 1:
            top, second = items[0], items[1]
            if float(second.get("confidence", 0.0)) >= policy.minimum_confidence:
                continue
        selected.append(items[0])

    selected.sort(
        key=lambda item: (
            -float(item.get("confidence", 0.0)),
            -int(item.get("support", 0)),
            -int(item.get("last_seen", 0)),
            str(item.get("domain", "")),
            str(item.get("slot", "")),
            str(item.get("value", "")),
        )
    )
    selected = selected[: policy.maximum_routed_hypotheses]
    while selected:
        prompt_projection = [
            {
                "domain": item["domain"],
                "slot": item["slot"],
                "value": item["value"],
                "support": item["support"],
                "counterevidence": item["counterevidence"],
                "confidence": item["confidence"],
                "last_seen": item["last_seen"],
            }
            for item in selected
        ]
        projected_text = json.dumps(
            prompt_projection, ensure_ascii=False, sort_keys=True
        )
        if lexical_count(projected_text) <= policy.overlay_lexical_token_cap:
            break
        selected.pop()
    return selected
