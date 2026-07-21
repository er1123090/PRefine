"""Self-contained PReFine-style latent-memory state machine.

It is intentionally shared by both paired arms.  The candidate extension is
never consulted here; it is appended only after the final baseline block is
built.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

from .contracts import PublicInputError


Completion = Callable[[list[dict[str, str]], int, int, float, bool], str]

_GENERATION_SYSTEM = """You are a Preference Abstraction Module for an agentic tool-calling system.

Your role is NOT to summarize dialogue history. Your role is to infer stable,
latent user preferences that constrain future API argument selection.

Key Principles:
- Preference reasoning is holistic and non-decompositional.
- Do NOT enumerate slots or list past actions.
- Infer abstract constraints that explain multiple past decisions.
- A valid preference must be actionable: it should rule in or rule out future API arguments.
- If evidence is insufficient, state uncertainty explicitly.

### Context (Accumulated Data)
**Previous Belief**:
{previous}

**Full Dialogue History**:
{dialogue}

**Full API Calls**:
{api_calls}
"""

_INITIAL_TASK = """### Task: Infer Latent Preference as an Action Constraint

Given the accumulated interaction history, infer the user's latent preference
as a compact decision-level constraint.

Guidelines:
1. Do NOT describe individual turns or list slot values.
2. Identify a unifying preference that explains repeated or consistent choices.
3. If multiple signals exist, abstract them into a higher-level constraint.
4. If contradictions exist, prioritize the most recent stable pattern.
5. If no stable preference can be inferred, explicitly say "insufficient evidence".

### Output Format (JSON)
{{
  "reasoning": "One-paragraph explanation of the inferred abstraction and its supporting evidence",
  "implicit_pref": "A single-sentence constraint describing how future API arguments should be biased or restricted"
}}
"""

_REFINE_TASK = """### Task: Refine Preference Based on Evidence Gaps

Your previous preference abstraction was rejected.

You must:
- Remove any claim not directly supported by the logs.
- Increase abstraction if details are over-specified.
- Preserve only what consistently constrains action selection.

Do NOT add new information.

### Input
Draft Preference:
{draft}

Verifier Feedback:
{feedback}

### Output Format (JSON)
{{
  "reasoning": "How unsupported details were removed or abstracted",
  "implicit_pref": "The revised, evidence-tight preference constraint"
}}
"""

_VERIFY_TASK = """You are a Preference Verification Module.

Your task is to judge whether the candidate preference is a valid latent constraint
derived from the interaction logs.

Evaluation Criteria:
1. Evidence Support: every claim must be supported by multiple or consistent signals.
2. Abstraction Quality: reject preferences that merely restate slot values or actions.
3. Actionability: the preference must constrain or bias future API argument selection.
4. Temporal Consistency: if behavior changed, ensure the preference reflects the latest stable pattern.

### Evidence (Logs)
**Dialogue**:
{dialogue}

**API Calls**:
{api_calls}

### Candidate Preference to Verify
{candidate}

### Output Format (JSON)
{{
  "valid": true/false,
  "feedback": "If false, specify whether the issue is over-specificity, hallucination, lack of abstraction, or non-actionability."
}}
"""


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def parse_json_object(text: str) -> dict[str, Any]:
    if not isinstance(text, str):
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def _history_text(history: object, through: int) -> tuple[str, str]:
    if not isinstance(history, Mapping) or not isinstance(history.get("sessions"), list):
        raise PublicInputError("history must have a sessions list")
    dialogue_parts: list[str] = []
    api_parts: list[str] = []
    for offset, session in enumerate(history["sessions"][:through]):
        if not isinstance(session, Mapping):
            raise PublicInputError("history session must be an object")
        index = session.get("session_index", offset)
        turns = session.get("dialogue", [])
        if isinstance(turns, list):
            dialogue = "\n".join(
                f"{turn.get('role', 'unknown')}: {turn.get('message', '')}"
                for turn in turns
                if isinstance(turn, Mapping)
            )
        else:
            dialogue = ""
        dialogue_parts.append(f"[Session {index}]\n{dialogue}")
        calls = session.get("api_calls", session.get("api_call", []))
        if not isinstance(calls, list):
            calls = [calls]
        api_parts.extend(f"[Session {index}] {call}" for call in calls if isinstance(call, str))
    return "\n\n".join(dialogue_parts), "\n".join(api_parts) or "No API calls recorded."


def accumulated_api_history(history: object) -> tuple[str, ...]:
    _dialogue, api_calls = _history_text(history, len(history.get("sessions", [])) if isinstance(history, Mapping) else 0)
    return tuple(line for line in api_calls.splitlines() if line and line != "No API calls recorded.")


def build_prefine_latent(
    *,
    history: object,
    complete: Completion,
    seed: int,
    maximum_attempts: int = 10,
) -> dict[str, Any]:
    """Run the source-shaped generate/verify/refine transition graph."""
    if maximum_attempts < 1:
        raise PublicInputError("maximum_attempts must be positive")
    if not isinstance(history, Mapping) or not isinstance(history.get("sessions"), list):
        raise PublicInputError("history must have a sessions list")
    previous: dict[str, Any] = {}
    for through in range(1, len(history["sessions"]) + 1):
        dialogue, api_calls = _history_text(history, through)
        draft: dict[str, Any] = previous
        feedback = ""
        for attempt in range(maximum_attempts):
            task = _INITIAL_TASK if attempt == 0 else _REFINE_TASK.format(draft=_json(draft), feedback=feedback)
            generated = complete(
                [
                    {
                        "role": "system",
                        "content": _GENERATION_SYSTEM.format(previous=_json(previous) if previous else "None", dialogue=dialogue, api_calls=api_calls),
                    },
                    {"role": "user", "content": task},
                ],
                seed + through * 100 + attempt * 2,
                2048,
                0.4,
                True,
            )
            draft = parse_json_object(generated)
            verified = complete(
                [
                    {"role": "system", "content": "You are a Preference Verification Module. Output JSON only."},
                    {"role": "user", "content": _VERIFY_TASK.format(dialogue=dialogue, api_calls=api_calls, candidate=_json(draft))},
                ],
                seed + through * 100 + attempt * 2 + 1,
                512,
                0.0,
                True,
            )
            verdict = parse_json_object(verified)
            if verdict.get("valid") is True:
                break
            feedback = str(verdict.get("feedback", "verification failed"))
        previous = draft if draft else {"implicit_pref": "insufficient evidence"}
    return previous or {"implicit_pref": "insufficient evidence"}


def baseline_memory_block(*, latent: object, history: object) -> str:
    calls = accumulated_api_history(history)
    return (
        "[Implicit Preferences]:\n"
        + _json(latent)
        + "\n\n[Past API History]:\n"
        + ("\n".join(calls) if calls else "None")
    )
