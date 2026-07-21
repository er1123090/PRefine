"""Frozen prompt construction; both experimental arms share the action prompt."""

from __future__ import annotations

import json
import re
from typing import Any

from .contracts import TASK_FIELDS
from .io import canonical_json
from .latent_firewall import (
    LatentFirewallDecision,
    QueryConstraintMask,
    canonical_vlt_prompt_tuple,
)


_LEXICAL_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


# Source-local copies of the standalone PReFine inference templates. The
# source method's schema filtering, repetition, cross-domain consistency, and
# mode-specific reasoning order are retained; the common policy below narrows
# memory use before any target action call.
ACTION_MODE_SCHEMA_KEYS = {
    "singleturn": "single",
    "multiturn": "multi",
}

ACTION_COMMON_SAFETY_RULES = """- The CURRENT DIALOGUE and CURRENT TOOL SCHEMA are authoritative.
- Explicit values and constraints stated by the user in CURRENT DIALOGUE always override memory.
- Memory may fill only otherwise-missing preference slots.
- Never change the requested operation, invent a function, invent a slot, or emit a slot outside the current schema.
- Respect schema-defined scalar types and enum values.
- If memory is weak, conflicting, ambiguous, irrelevant, or insufficient, omit the memory-derived slot.
- Never copy or replay an entire historical API call.
- Never carry forward transient historical details such as dates, times, locations, addresses, names, identifiers, or itinerary details.
- A stable preference value may be reused only when it is relevant to an otherwise-missing preference slot in the current request."""

SINGLE_ACTION_INFERENCE_TEMPLATE = """You are a Personalized Preference Reasoning Agent.
Your task is to generate the most appropriate Service API call for the current user utterance
by integrating the current request with retrieved long-term memory, strictly following the
current tool schema.

[Task Definition: SINGLE-TURN]
The user may not explicitly state every preference in the current utterance. Fill only a
missing preference slot when stable memory supplies reasonable support.
- Repetitiveness: a repeatedly chosen stable value can support a preference.
- Cross-domain Consistency: a stable behavioral constraint may support a schema-valid value
  in the requested domain, but only when the provided memory block authorizes that transfer.

[Reasoning Steps]
1. Schema Filtering (Slot Scope Control):
   - Identify the requested operation from CURRENT DIALOGUE.
   - Consider only functions and slots defined in CURRENT TOOL SCHEMA.
2. Relevant Memories:
   - Apply stable preferences only to otherwise-missing preference slots.
   - Map an authorized abstract constraint only to a schema-valid value.
3. Formulate Output:
   - Create a slot only when it is explicit in the current utterance or reasonably supported
     by the provided memory block.
   - Do not create empty slots.
   - Do not hallucinate values or infer beyond the schema.

[Non-negotiable Safety and Precedence Rules]
{safety_rules}

CURRENT TOOL SCHEMA:
{schema}

MEMORY BLOCK:
{memory}

CURRENT DIALOGUE (single current user utterance):
{query}

Output Format:
FunctionName(slot="value", ...)

Now produce exactly one final Service API call. Output no explanation, JSON wrapper, list,
markdown, or code fence.
"""

MULTI_ACTION_INFERENCE_TEMPLATE = """You are a Personalized Preference Reasoning Agent.
Your task is to generate the most appropriate Service API call for the current request by
integrating retrieved long-term memory with the current multi-turn dialogue context, strictly
following the current tool schema.

[Task Definition: MULTI-TURN]
Select the single requested Service API operation from the current dialogue. Resolve explicit
user constraints across the current dialogue before considering memory. Fill only a missing
preference slot when stable memory supplies reasonable support.
- Repetitiveness: a repeatedly chosen stable value can support a preference.
- Cross-domain Consistency: a stable behavioral constraint may support a schema-valid value
  in the requested domain, but only when the provided memory block authorizes that transfer.

[Reasoning Steps]
1. Schema Filtering (Slot Scope Control):
   - Identify the requested operation from CURRENT DIALOGUE.
   - Consider only functions and slots defined in CURRENT TOOL SCHEMA.
2. Current Dialogue Context:
   - Resolve the latest user request and all explicit user-provided constraints in this
     current dialogue.
   - Treat assistant and tool text as context, not as user preference evidence unless the
     user explicitly confirms it.
3. Relevant Memories:
   - Apply stable preferences only to otherwise-missing preference slots.
   - Map an authorized abstract constraint only to a schema-valid value.
4. Formulate Output:
   - Create a slot only when it is explicit in the current dialogue or reasonably supported
     by the provided memory block.
   - Do not create empty slots.
   - Do not hallucinate values or infer beyond the schema.

[Non-negotiable Safety and Precedence Rules]
{safety_rules}

CURRENT TOOL SCHEMA:
{schema}

MEMORY BLOCK:
{memory}

CURRENT DIALOGUE (multi-turn context):
{query}

Output Format:
FunctionName(slot="value", ...)

Now produce exactly one final Service API call. Output no explanation, JSON wrapper, list,
markdown, or code fence.
"""

LATENT_SYSTEM = """You are a Preference Abstraction Module for an agentic tool-calling system.
Do not summarize dialogue history or enumerate past actions. Infer stable latent preferences
that constrain future API argument selection. Every claim must be supported by repeated or
consistent evidence. State uncertainty when evidence is insufficient.

Previous belief:
{previous}

Accumulated dialogue:
{dialogue}

Accumulated API calls:
{api_calls}
"""

LATENT_INITIAL = """Infer one compact, decision-level latent constraint. Prefer a higher-level
abstraction explaining repeated choices; reflect the latest stable pattern when behavior changed.
Return JSON only: {"reasoning":"...", "implicit_pref":"..."}. If evidence is insufficient,
set implicit_pref to "insufficient evidence"."""

LATENT_REFINE = """The previous abstraction was rejected. Remove unsupported claims, preserve
only evidence-tight action constraints, and return the same JSON shape.
Previous draft: {draft}
Verifier feedback: {feedback}"""

LATENT_VERIFY = """Judge whether the candidate is supported, abstract (not a slot-value list),
actionable, and temporally consistent with the logs. Return JSON only:
{{"valid":true_or_false,"feedback":"..."}}.

Dialogue:
{dialogue}

API calls:
{api_calls}

Candidate:
{candidate}
"""


def lexical_tokens(text: str) -> list[str]:
    return _LEXICAL_TOKEN_RE.findall(text)


def lexical_count(text: str) -> int:
    return len(lexical_tokens(text))


def cap_text(text: str, cap: int) -> str:
    if cap <= 0:
        return ""
    matches = list(_LEXICAL_TOKEN_RE.finditer(text))
    if len(matches) <= cap:
        return text
    return text[: matches[cap - 1].end()] + "\n[TRUNCATED_AT_FROZEN_LEXICAL_CAP]"


def latent_text(value: Any) -> str:
    if not value:
        return "insufficient evidence"
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value.strip()
        value = parsed
    if isinstance(value, dict):
        candidate = value.get("implicit_pref", value)
        return candidate if isinstance(candidate, str) else canonical_json(candidate)
    return str(value)


def baseline_memory_block(memory: dict[str, Any], history: dict[str, Any], cap: int) -> str:
    calls: list[str] = []
    for session in history.get("sessions", []):
        index = int(session.get("session_index", 0))
        calls.extend(f"[session {index}] {call}" for call in session.get("api_calls", []))
    block = (
        "Latent preference abstraction:\n"
        + latent_text(memory.get("latent_abstraction"))
        + "\n\nAccumulated historical API calls (historical evidence only):\n"
        + ("\n".join(calls) if calls else "none")
    )
    return cap_text(block, cap)


def candidate_memory_block(
    routed: list[dict[str, Any]],
    cap: int,
    *,
    selected_domain: str | None = None,
    vlt_decision: LatentFirewallDecision | None = None,
    latent_trait_ontology: Any = None,
    schema: Any = None,
    preference_slots: dict[str, list[str]] | None = None,
    latent_trait_ontology_sha256: str | None = None,
    schema_key: str | None = None,
    current_query: str | None = None,
    query_mode: str | None = None,
    query_constraint_mask: QueryConstraintMask | None = None,
) -> str:
    if cap <= 0:
        raise ValueError("candidate memory cap must be positive")
    if selected_domain is None:
        if vlt_decision is not None and vlt_decision.authorized:
            raise ValueError("ABSTAIN cannot carry an authorized VLT tuple")
        block = (
            "Public query relevance gate: ABSTAIN. No typed or latent memory "
            "evidence is authorized for this query."
        )
        if lexical_count(block) > cap:
            raise ValueError("candidate ABSTAIN block exceeds lexical cap")
        return block
    if any(str(item.get("domain", "")) != selected_domain for item in routed):
        raise ValueError("routed evidence escapes selected public query domain")
    vlt_tuple = (
        canonical_vlt_prompt_tuple(
            vlt_decision,
            ontology=latent_trait_ontology,
            schema=schema,
            preference_slots=preference_slots,
            selected_domain=selected_domain,
            ontology_sha256=latent_trait_ontology_sha256,
            schema_key=schema_key,
            current_query=current_query,
            query_mode=query_mode,
            query_constraint_mask=query_constraint_mask,
        )
        if vlt_decision is not None
        else None
    )
    if vlt_tuple is not None and (
        vlt_decision is None
        or vlt_decision.selected_domain != selected_domain
        or vlt_decision.domain != selected_domain
    ):
        raise ValueError("authorized VLT tuple escapes selected public query domain")

    selected = list(routed)
    include_vlt = vlt_tuple is not None
    while True:
        parts = [
            "Public query relevance gate selected schema domain: "
            + selected_domain
            + ". This scopes memory relevance only; it does not imply that the "
            "user requested this domain or operation."
        ]
        if selected:
            safe = [
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
            parts.append(
                "Typed, current-task-routed evidence:\n"
                + json.dumps(safe, ensure_ascii=False, sort_keys=True)
            )
        else:
            parts.append(
                "No typed evidence survived routing in the selected domain."
            )
        if include_vlt:
            assert vlt_tuple is not None
            parts.append(
                "Verified latent transfer (closed tuple):\n" + vlt_tuple
            )
        elif not selected:
            parts.append("No memory evidence is authorized.")
        block = "\n\n".join(parts)
        if lexical_count(block) <= cap:
            return block
        if selected:
            selected.pop()
            continue
        if include_vlt:
            include_vlt = False
            continue
        raise ValueError("candidate memory block cannot fit lexical cap")


def build_action_prompt(
    task: dict[str, Any],
    schema: Any,
    memory_block: str,
) -> str:
    """The sole mode-aware action prompt constructor used by both arms."""
    if set(task) != TASK_FIELDS:
        raise ValueError(
            f"action task field contract violation: {sorted(set(task))}"
        )
    mode = task["mode"]
    schema_key = task["schema_key"]
    expected_schema_key = ACTION_MODE_SCHEMA_KEYS.get(mode)
    if expected_schema_key is None:
        raise ValueError(f"invalid action task mode: {mode}")
    if schema_key != expected_schema_key:
        raise ValueError("action task mode/schema_key mismatch")
    template = (
        SINGLE_ACTION_INFERENCE_TEMPLATE
        if mode == "singleturn"
        else MULTI_ACTION_INFERENCE_TEMPLATE
    )
    return template.format(
        safety_rules=ACTION_COMMON_SAFETY_RULES,
        query=task["query"],
        schema=json.dumps(
            schema, ensure_ascii=False, sort_keys=True, indent=2
        ),
        memory=memory_block,
    )
