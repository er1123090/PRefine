"""Append-only, PReFine-anchored typed-evidence overlay.

The baseline PReFine memory is always constructed first.  When the public
query gate cannot authorize safe, repeated typed evidence, this module returns
that baseline block byte-for-byte.  It never serializes a target, a gold row,
or free-form latent text beyond the baseline's own PReFine projection.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .contracts import CandidatePolicy
from .io import canonical_json, sha256_text
from .latent_firewall import QueryConstraintMask, build_query_constraint_mask
from .prompts import baseline_memory_block, build_action_prompt, lexical_count
from .query_gate import GateDecision, gate_public_query
from .router import route_hypotheses


@dataclass(frozen=True)
class AnchoredActionCase:
    """One public-task action prompt and raw-free audit metadata."""

    memory_block: str
    prompt: str
    routing_decision: GateDecision
    query_constraint_mask: QueryConstraintMask
    routed_count: int
    routed_constraint_filtered_count: int
    fallback_to_baseline: bool
    audit_artifact: dict[str, Any]


def _safe_scalar_literal(item: Mapping[str, Any]) -> str | None:
    """Render a conservative API scalar without letting it become prompt syntax."""
    source = item.get("source_literal")
    if not isinstance(source, dict) or set(source) != {"type", "value"}:
        return None
    value_type = source.get("type")
    value = source.get("value")
    if value_type == "string" and isinstance(value, str):
        normalized = unicodedata.normalize("NFKC", value)
        if (
            not normalized
            or len(normalized) > 64
            or any(
                character in "\r\n`\"'{}[]<>"
                or unicodedata.category(character).startswith("C")
                for character in normalized
            )
            or not all(
                character.isalnum() or character in " -_/.,"
                for character in normalized
            )
        ):
            return None
        return json.dumps(normalized, ensure_ascii=False)
    if value_type == "boolean" and type(value) is bool:
        return "true" if value else "false"
    if value_type == "integer" and type(value) is int:
        return str(value)
    if value_type == "number" and type(value) in {int, float}:
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    return None


def compact_typed_overlay(
    routed: Sequence[Mapping[str, Any]],
    *,
    selected_domain: str,
    cap: int,
) -> str | None:
    """Return a concise factual overlay, or ``None`` when none is safe.

    The renderer intentionally avoids JSON lists and routing jargon: both made
    the prior candidate prompt easier for a small tool-calling model to mimic.
    """
    if cap <= 0:
        raise ValueError("anchored overlay cap must be positive")
    facts: list[str] = []
    for item in routed:
        if str(item.get("domain", "")) != selected_domain:
            raise ValueError("routed typed evidence escapes selected domain")
        slot = str(item.get("slot", ""))
        literal = _safe_scalar_literal(item)
        try:
            support = int(item.get("support", 0))
            counterevidence = int(item.get("counterevidence", 0))
        except (TypeError, ValueError):
            continue
        if not slot or literal is None or support < 1 or counterevidence < 0:
            continue
        facts.append(
            f"- {slot} = {literal} (repeated support: {support}; contrary observations: {counterevidence})"
        )

    if not facts:
        return None
    # Facts are already ranked by the router; dropping from the end preserves
    # its deterministic confidence/support/recency ordering.
    while facts:
        overlay = (
            "Validated recurring preference facts for the selected tool domain:\n"
            + "\n".join(facts)
        )
        if lexical_count(overlay) <= cap:
            return overlay
        facts.pop()
    return None


def _anchored_memory_block(
    *,
    memory: Mapping[str, Any],
    history: Mapping[str, Any],
    overlay: str | None,
    memory_cap: int,
) -> tuple[str, bool]:
    baseline = baseline_memory_block(dict(memory), dict(history), memory_cap)
    if overlay is None:
        # This exact identity is a safety and regression contract.
        return baseline, True
    available_baseline_cap = memory_cap - lexical_count(overlay)
    if available_baseline_cap <= 0:
        raise ValueError("anchored overlay leaves no room for PReFine baseline")
    anchored = (
        baseline_memory_block(dict(memory), dict(history), available_baseline_cap)
        + "\n\n"
        + overlay
    )
    if lexical_count(anchored) > memory_cap:
        raise AssertionError("anchored memory exceeded frozen lexical cap")
    return anchored, False


def build_anchored_action_case(
    *,
    task: dict[str, Any],
    history: dict[str, Any],
    memory: dict[str, Any],
    schema: Any,
    preference_slots: Mapping[str, Sequence[str]],
    policy: CandidatePolicy,
    public_domain_ontology: dict[str, Any],
    constraint_ontology: dict[str, Any],
) -> AnchoredActionCase:
    """Construct the candidate prompt without reading evaluator-only fields."""
    normalized_slots = {
        str(domain): [str(slot) for slot in slots]
        for domain, slots in preference_slots.items()
    }
    routing = gate_public_query(
        str(task["query"]), str(task["mode"]), schema, public_domain_ontology
    )
    constraint_mask = build_query_constraint_mask(
        current_query=str(task["query"]),
        query_mode=str(task["mode"]),
        selected_domain=routing.schema_domain,
        schema_key=str(task["schema_key"]),
        schema=schema,
        ontology=constraint_ontology,
    )
    routed_before_mask = route_hypotheses(
        memory,
        task,
        schema,
        normalized_slots,
        policy,
        public_domain_ontology,
        decision=routing,
    )
    masked_slots = set(constraint_mask.constrained_slots)
    routed = [
        item
        for item in routed_before_mask
        if not (
            str(item.get("domain", "")) == routing.schema_domain
            and str(item.get("slot", "")) in masked_slots
        )
    ]
    overlay = (
        compact_typed_overlay(
            routed,
            selected_domain=str(routing.schema_domain),
            cap=policy.overlay_lexical_token_cap,
        )
        if routing.schema_domain is not None
        else None
    )
    block, fallback_to_baseline = _anchored_memory_block(
        memory=memory,
        history=history,
        overlay=overlay,
        memory_cap=policy.memory_lexical_token_cap,
    )
    prompt = build_action_prompt(task, schema, block)
    routing_artifact = {
        "status": routing.status,
        "canonical_domain": routing.canonical_domain,
        "schema_domain": routing.schema_domain,
        "top_score": routing.top_score,
        "second_score": routing.second_score,
        "scores_sha256": sha256_text(canonical_json(routing.scores)),
    }
    artifact = {
        "schema_version": 1,
        "kind": "prefine_anchored_typed_overlay_action_case_v1",
        "case_key_sha256": sha256_text(str(task["case_key"])),
        "example_id_sha256": sha256_text(str(task["example_id"])),
        "query_sha256": sha256_text(str(task["query"])),
        "schema_key": str(task["schema_key"]),
        "routing": routing_artifact,
        "query_constraint_mask": constraint_mask.artifact(),
        "routed_before_constraint_count": len(routed_before_mask),
        "routed_constraint_filtered_count": len(routed_before_mask) - len(routed),
        "routed_count": len(routed),
        "routed_sha256": sha256_text(canonical_json(routed)),
        "overlay_present": overlay is not None,
        "overlay_sha256": sha256_text(overlay) if overlay is not None else None,
        "fallback_to_baseline": fallback_to_baseline,
        "memory_block_sha256": sha256_text(block),
        "prompt_sha256": sha256_text(prompt),
    }
    artifact["decision_sha256"] = sha256_text(canonical_json(artifact))
    return AnchoredActionCase(
        memory_block=block,
        prompt=prompt,
        routing_decision=routing,
        query_constraint_mask=constraint_mask,
        routed_count=len(routed),
        routed_constraint_filtered_count=len(routed_before_mask) - len(routed),
        fallback_to_baseline=fallback_to_baseline,
        audit_artifact=artifact,
    )
