"""Query-conditioned, evidence-calibrated append-only memory overlay."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .contracts import EvidenceFact, OverlayDecision, OverlayPolicy, PublicInputError
from .parsing import (
    identifier_aliases,
    normalized_preference_slots,
    phrase_present,
    public_user_text,
    safe_scalar,
    schema_enum_values,
    session_calls,
)


def _public_mapping(value: object, field: str) -> Mapping[str, Sequence[str]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise PublicInputError(f"{field} must be an object")
    result: dict[str, tuple[str, ...]] = {}
    for key, aliases in value.items():
        if not isinstance(key, str) or not isinstance(aliases, Sequence) or isinstance(aliases, (str, bytes)):
            raise PublicInputError(f"{field} has invalid aliases")
        result[key] = tuple(alias for alias in aliases if isinstance(alias, str) and alias.strip())
    return result


def select_public_domain(
    *,
    query: str,
    mode: str,
    preference_slots: Mapping[str, tuple[str, ...]],
    schema_domain: str | None,
    domain_aliases: object = None,
) -> str | None:
    if schema_domain is not None:
        if not isinstance(schema_domain, str) or schema_domain not in preference_slots:
            raise PublicInputError("schema_domain must be one declared preference domain")
        return schema_domain
    text = public_user_text(query, mode)
    supplied = _public_mapping(domain_aliases, "domain_aliases")
    matched: list[str] = []
    for domain in preference_slots:
        aliases = set(identifier_aliases(domain))
        aliases.update(supplied.get(domain, ()))
        if any(phrase_present(text, alias) for alias in aliases):
            matched.append(domain)
    return matched[0] if len(matched) == 1 else None


def explicit_public_slots(
    *,
    query: str,
    mode: str,
    domain: str,
    slots: Sequence[str],
    slot_aliases: object,
    schema: object,
    observed_values: Mapping[str, set[str]],
) -> tuple[str, ...]:
    text = public_user_text(query, mode)
    supplied = _public_mapping(slot_aliases, "slot_aliases")
    enum_values = schema_enum_values(schema)
    explicit: list[str] = []
    for slot in slots:
        aliases = set(identifier_aliases(slot))
        aliases.update(supplied.get(slot, ()))
        values = set(enum_values.get((domain, slot), ()))
        values.update(observed_values.get(slot, set()))
        if any(phrase_present(text, alias) for alias in aliases) or any(phrase_present(text, value) for value in values if len(value) >= 2):
            explicit.append(slot)
    return tuple(explicit)


def _observed_values(
    history: object,
    preference_slots: Mapping[str, tuple[str, ...]],
    policy: OverlayPolicy,
) -> tuple[dict[tuple[str, str, str], set[int]], dict[tuple[str, str, str], int], dict[str, set[str]]]:
    supports: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    latest: dict[tuple[str, str, str], int] = {}
    values_by_slot: dict[str, set[str]] = defaultdict(set)
    for session_index, calls in session_calls(history):
        per_session: set[tuple[str, str, str]] = set()
        for call in calls:
            domain = call.get("name")
            arguments = call.get("arguments")
            if not isinstance(domain, str) or domain not in preference_slots or not isinstance(arguments, Mapping):
                continue
            allowed = set(preference_slots[domain])
            for slot, raw_value in arguments.items():
                if not isinstance(slot, str) or slot not in allowed:
                    continue
                value = safe_scalar(raw_value, policy.maximum_value_characters)
                if value is not None:
                    per_session.add((domain, slot, value))
        for key in per_session:
            supports[key].add(session_index)
            latest[key] = max(session_index, latest.get(key, session_index))
            values_by_slot[key[1]].add(key[2])
    return supports, latest, values_by_slot


def _eligible_facts(
    *,
    domain: str,
    slots: Sequence[str],
    supports: Mapping[tuple[str, str, str], set[int]],
    latest: Mapping[tuple[str, str, str], int],
    explicit_slots: set[str],
    policy: OverlayPolicy,
) -> tuple[EvidenceFact, ...]:
    facts: list[EvidenceFact] = []
    for slot in slots:
        if slot in explicit_slots:
            continue
        candidates = [
            (value, session_ids)
            for (fact_domain, fact_slot, value), session_ids in supports.items()
            if fact_domain == domain and fact_slot == slot
        ]
        if not candidates:
            continue
        all_sessions = set().union(*(session_ids for _value, session_ids in candidates))
        ranked = sorted(candidates, key=lambda pair: (-len(pair[1]), pair[0]))
        if len(ranked) > 1 and len(ranked[0][1]) == len(ranked[1][1]):
            continue
        value, support_ids = ranked[0]
        support = len(support_ids)
        total = len(all_sessions)
        competing = total - support
        consensus = support / total if total else 0.0
        if (
            support < policy.minimum_independent_sessions
            or consensus < policy.minimum_consensus
            or competing > policy.maximum_competing_sessions
            or support - competing < policy.minimum_support_margin
        ):
            continue
        facts.append(
            EvidenceFact(
                domain=domain,
                slot=slot,
                value=value,
                support_sessions=support,
                competing_sessions=competing,
                consensus=round(consensus, 6),
                last_seen_session=latest[(domain, slot, value)],
            )
        )
    facts.sort(key=lambda item: (-item.consensus, -item.support_sessions, -item.last_seen_session, item.slot, item.value))
    return tuple(facts[: policy.maximum_facts])


def render_overlay(facts: Sequence[EvidenceFact], policy: OverlayPolicy) -> str:
    if not facts:
        return ""
    lines = [
        "[Evidence-Calibrated Preference Overlay]",
        "Use only for missing slots. Current user constraints override every item.",
    ]
    for fact in facts:
        lines.append(
            f'- {fact.slot}="{fact.value}" (repeated in {fact.support_sessions} independent sessions; consensus {fact.consensus:.2f})'
        )
    overlay = "\n".join(lines)
    if len(overlay) > policy.maximum_overlay_characters:
        raise PublicInputError("eligible overlay exceeds the fixed character cap")
    return overlay


def build_overlay(
    *,
    baseline_memory: str,
    history: object,
    query: str,
    mode: str,
    preference_slots: object,
    schema: object = None,
    schema_domain: str | None = None,
    domain_aliases: object = None,
    slot_aliases: object = None,
    policy: OverlayPolicy | None = None,
) -> OverlayDecision:
    """Return a baseline-preserving candidate memory using public data only."""
    if not isinstance(baseline_memory, str) or not baseline_memory:
        raise PublicInputError("baseline_memory must be a nonempty string")
    if not isinstance(query, str) or not query.strip():
        raise PublicInputError("query must be a nonempty string")
    active_policy = policy or OverlayPolicy()
    slots_by_domain = normalized_preference_slots(preference_slots)
    selected = select_public_domain(
        query=query,
        mode=mode,
        preference_slots=slots_by_domain,
        schema_domain=schema_domain,
        domain_aliases=domain_aliases,
    )
    if selected is None:
        return OverlayDecision(baseline_memory, None, (), (), "public_query_domain_ambiguous_or_absent")
    supports, latest, values_by_slot = _observed_values(history, slots_by_domain, active_policy)
    explicit = explicit_public_slots(
        query=query,
        mode=mode,
        domain=selected,
        slots=slots_by_domain[selected],
        slot_aliases=slot_aliases,
        schema=schema,
        observed_values=values_by_slot,
    )
    facts = _eligible_facts(
        domain=selected,
        slots=slots_by_domain[selected],
        supports=supports,
        latest=latest,
        explicit_slots=set(explicit),
        policy=active_policy,
    )
    if not facts:
        reason = "explicit_current_constraint_or_no_high_consensus_repeated_fact"
        return OverlayDecision(baseline_memory, selected, (), explicit, reason)
    overlay = render_overlay(facts, active_policy)
    return OverlayDecision(
        baseline_memory + "\n\n" + overlay,
        selected,
        facts,
        explicit,
        "append_only_repeated_query_conditioned_facts",
    )
