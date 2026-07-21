"""Pure package-local prompt builders with frozen semantic roles."""

from __future__ import annotations

import json
from typing import Sequence

from .contracts import ChatMessage, ContextMode


SINGLE_ACTION_INFERENCE_TEMPLATE = """You are a Personalized Preference Reasoning Agent.
Your task is to generate the most appropriate Service API call for the current user utterance
by integrating (1) Retrieved Long-term Memories and (2) Accumulated API Call History,
strictly following the given schema.


[Task Definition]
The user may not explicitly state all information in the current turn. You must deduce missing information by analyzing inferred preferences in the Retrieved Memories and accumulated API call history.
- **Repetitiveness**: If a user frequently chose a specific value in the past, assume this is their preference.
- **Cross-domain Consistency**: Identify the user's **universal behavioral patterns or constraints** (e.g., cost sensitivity, service level, risk aversion) exhibited in previous interactions. If a direct preference is missing, **deduce** the current slot's value by applying these established patterns.

[Reasoning Steps]
1. **Schema Filtering (Slot Scope Control)**:
   - Identify the target domain from the `Schema`.
   - Consider ONLY the slots defined in the schema.
2. **Relevant Memories**: 
    - Infer the value by applying the user’s stable behavioral patterns
    - Map these abstract constraints to the most appropriate schema-valid value.
3. **Formulate Output**:
   - Create a slot ONLY when there is reasonable support from API history or memories.
   - Do NOT create empty slots
   - Do NOT hallucinate values or infer beyond the schema.

 Schema (Consider valid slots for the domain):
{preference_schema}

Relevant Memories (User Preferences & Constraints):
{retrieved_memories}

Current User Utterance:
{user_utterance}

Output Format:
Get~(slot_name="value", ...)

Now produce ONLY the final Service API call:
"""


MULTI_ACTION_INFERENCE_TEMPLATE = """You are a Personalized Preference Reasoning Agent.
Your task is to generate the most appropriate Service API call for the current user utterance
by integrating (1) Retrieved Long-term Memories, (2) Accumulated API Call History, and (3) current user's dialogue context,
strictly following the given schema.


[Task Definition]
1. Select the most appropriate Service API call for the current user utterance.
2. The user may not explicitly state all information in the current turn. You must deduce missing information by analyzing inferred preferences in the Retrieved Memories and accumulated API call history.
   - **Repetitiveness**: If a user frequently chose a specific value in the past, assume this is their preference.
   - **Cross-domain Consistency**: Identify the user's **universal behavioral patterns or constraints** (e.g., cost sensitivity, service level, risk aversion) exhibited in previous interactions. If a direct preference is missing, **deduce** the current slot's value by applying these established patterns.

[Reasoning Steps]
1. **Schema Filtering (Slot Scope Control)**:
   - Identify the target domain from the `Schema`.
   - Consider ONLY the slots defined in the schema.

2. **Current Dialogue Context**:
   - Analyze the current user's dialogue context and create the appropriate slots based on explicitly mentioned information.

3. **Relevant Memories**: 
   - Infer missing values by applying the user’s stable behavioral patterns
   - Map these abstract constraints to the most appropriate schema-valid value.

4. **Formulate Output**:
   - Create a slot ONLY when there is reasonable support from API history or memories.
   - Do NOT create empty slots
   - Do NOT hallucinate values or infer beyond the schema.

 Schema (Consider valid slots for the domain):
{preference_schema}

Relevant Memories (User Preferences & Constraints):
{retrieved_memories}

Current User Utterance:
{user_utterance}

Output Format:
Get~(slot_name="value", ...)

Now produce ONLY the final Service API call:
"""


def build_generation_messages(
    *, previous_preference: object, dialogue: str, api_history: Sequence[str]
) -> tuple[ChatMessage, ...]:
    system = _generation_system(previous_preference, dialogue, api_history)
    user = """
### Task: Infer Latent Preference as an Action Constraint

Given the accumulated interaction history, infer the user's latent preference
as a compact decision-level constraint.

Guidelines:
1. Do NOT describe individual turns or list slot values.
2. Identify a unifying preference that explains repeated or consistent choices.
3. If multiple signals exist, abstract them into a higher-level constraint (e.g., cost sensitivity, convenience-seeking).
4. If contradictions exist, prioritize the most recent stable pattern.
5. If no stable preference can be inferred, explicitly say "insufficient evidence".

### Output Format (JSON)
{{
  "reasoning": "One-paragraph explanation of the inferred abstraction and its supporting evidence",
  "implicit_pref": "A single-sentence constraint describing how future API arguments should be biased or restricted"
}}
"""
    return (ChatMessage("system", system), ChatMessage("user", user))


def build_refinement_messages(
    *,
    previous_preference: object,
    dialogue: str,
    api_history: Sequence[str],
    previous_draft: object,
    feedback: str,
) -> tuple[ChatMessage, ...]:
    system = _generation_system(previous_preference, dialogue, api_history)
    user = f"""
### Task: Refine Preference Based on Evidence Gaps

Your previous preference abstraction was rejected.

You must:
- Remove any claim not directly supported by the logs.
- Increase abstraction if details are over-specified.
- Preserve only what consistently constrains action selection.

Do NOT add new information.

### Input
Draft Preference:
"{_pretty(previous_draft)}"

Verifier Feedback:
"{feedback}"

### Output Format (JSON)
{{
  "reasoning": "How unsupported details were removed or abstracted",
  "implicit_pref": "The revised, evidence-tight preference constraint"
}}
"""
    return (ChatMessage("system", system), ChatMessage("user", user))


def build_verifier_messages(
    *, candidate: object, dialogue: str, api_history: Sequence[str]
) -> tuple[ChatMessage, ...]:
    system = "You are a Preference Verification Module. Output JSON only."
    user = f"""
You are a Preference Verification Module.

Your task is to judge whether the candidate preference is a valid latent constraint
derived from the interaction logs.

Evaluation Criteria:
1. Evidence Support:
   - Every claim must be supported by multiple or consistent signals.
2. Abstraction Quality:
   - Reject preferences that merely restate slot values or actions.
   - Prefer abstract constraints that generalize across domains.
3. Actionability:
   - The preference must constrain or bias future API argument selection.
   - If it cannot affect future actions, it is invalid.
4. Temporal Consistency:
   - If behavior changed, ensure the preference reflects the latest stable pattern.

### Evidence (Logs)
**Dialogue**:
{dialogue}

**API Calls**:
{_history(api_history)}

### Candidate Preference to Verify
{json.dumps(candidate, ensure_ascii=False, indent=2)}


### Output Format (JSON)
{{
  "valid": true/false,
  "feedback": "If false, specify whether the issue is over-specificity, hallucination, lack of abstraction, or non-actionability."
}}
"""
    return (ChatMessage("system", system), ChatMessage("user", user))


def build_inference_messages(
    *,
    context_mode: ContextMode,
    multi_turn: bool = False,
    current_utterance: str,
    tool_schema: object,
    preference: object | None = None,
    api_history: Sequence[str] = (),
    prior_dialogue: str | None = None,
) -> tuple[ChatMessage, ...]:
    """Build the source-faithful inference prompt for one normalized context."""

    memory_parts: list[str] = []
    if context_mode in {
        ContextMode.MEMORY_ONLY,
        ContextMode.MEMORY_API,
        ContextMode.MEMORY_DIAG,
    }:
        memory_parts.append(
            "[Implicit Preferences]:\n"
            + _format_inference_value(preference)
        )
    if context_mode in {ContextMode.MEMORY_API, ContextMode.API_ONLY}:
        memory_parts.append(
            f"[Past API History]:\n{_format_inference_api_history(api_history)}"
        )
    schema = (
        json.dumps(tool_schema, ensure_ascii=False, indent=2)
        if tool_schema
        else "No specific schema provided."
    )
    template = (
        MULTI_ACTION_INFERENCE_TEMPLATE
        if multi_turn
        else SINGLE_ACTION_INFERENCE_TEMPLATE
    )
    prompt = template.format(
        preference_schema=schema,
        retrieved_memories="\n\n".join(memory_parts),
        user_utterance=current_utterance.strip(),
    )
    if context_mode is ContextMode.MEMORY_DIAG:
        dialogue = prior_dialogue or "None"
        marker = "\nCurrent User Utterance:\n"
        dialogue_section = f"\nCurrent Dialogue Context:\n{dialogue}\n"
        prompt = prompt.replace(marker, dialogue_section + marker, 1)
    return (ChatMessage("user", prompt),)


def _generation_system(
    previous_preference: object, dialogue: str, api_history: Sequence[str]
) -> str:
    return f"""
You are a Preference Abstraction Module for an agentic tool-calling system.

Your role is NOT to summarize dialogue history.
Your role is to infer stable, latent user preferences that constrain future API argument selection.

Key Principles:
- Preference reasoning is holistic and non-decompositional.
- Do NOT enumerate slots or list past actions.
- Infer abstract constraints that explain multiple past decisions.
- A valid preference must be actionable: it should rule in or rule out future API arguments.

If evidence is insufficient, state uncertainty explicitly.

### Context (Accumulated Data)
**Previous Belief**: 
{_pretty(previous_preference)}

**Full Dialogue History**:
{dialogue}

**Full API Calls**:
{_history(api_history)}

"""


def _history(values: Sequence[str]) -> str:
    return "\n".join(values) if values else "No API calls recorded."


def _pretty(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _format_inference_value(value: object) -> str:
    if value is None:
        return "None"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value).strip()


def _format_inference_api_history(values: Sequence[str]) -> str:
    return "\n".join(values) if values else "None"
