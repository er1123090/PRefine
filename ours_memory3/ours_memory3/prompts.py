"""Frozen PReFine-compatible action-prompt boundary for paired evaluation."""

from __future__ import annotations

import hashlib
import json

from .contracts import PublicInputError


SINGLE_ACTION_TEMPLATE = """You are a Personalized Preference Reasoning Agent.
Your task is to generate the most appropriate Service API call for the current user utterance
by integrating (1) Retrieved Long-term Memories and (2) Accumulated API Call History,
strictly following the given schema.

[Task Definition]
The user may not explicitly state all information in the current turn. You must deduce missing information by analyzing inferred preferences in the Retrieved Memories and accumulated API call history.
- Repetitiveness: If a user frequently chose a specific value in the past, assume this is their preference.
- Cross-domain Consistency: Identify universal behavioral patterns or constraints from previous interactions. If a direct preference is missing, map the established pattern only to a schema-valid value.

[Reasoning Steps]
1. Schema Filtering (Slot Scope Control):
   - Identify the target domain from the Schema.
   - Consider ONLY the slots defined in the schema.
2. Relevant Memories:
   - Infer a value only from stable behavioral patterns or repeated API history.
   - Do not infer beyond the schema.
3. Formulate Output:
   - Create a slot ONLY when there is reasonable support from API history or memories.
   - Do NOT create empty slots or hallucinate values.

Schema (Consider valid slots for the domain):
{schema}

Relevant Memories (User Preferences & Constraints):
{memory}

Current User Utterance:
{query}

Output Format:
Get~(slot_name="value", ...)

Now produce ONLY the final Service API call:
"""


MULTI_ACTION_TEMPLATE = """You are a Personalized Preference Reasoning Agent.
Your task is to generate the most appropriate Service API call for the current user utterance
by integrating (1) Retrieved Long-term Memories, (2) Accumulated API Call History, and (3) current user's dialogue context,
strictly following the given schema.

[Task Definition]
1. Select the most appropriate Service API call for the current user utterance.
2. The user may not explicitly state all information in the current turn. You must deduce missing information by analyzing inferred preferences in the Retrieved Memories and accumulated API call history.
   - Repetitiveness: If a user frequently chose a specific value in the past, assume this is their preference.
   - Cross-domain Consistency: Identify universal behavioral patterns or constraints from previous interactions. If a direct preference is missing, map the established pattern only to a schema-valid value.

[Reasoning Steps]
1. Schema Filtering (Slot Scope Control):
   - Identify the target domain from the Schema.
   - Consider ONLY the slots defined in the schema.
2. Current Dialogue Context:
   - Analyze the current user's dialogue context and create appropriate slots based on explicitly mentioned information.
3. Relevant Memories:
   - Infer missing values only from stable behavioral patterns or repeated API history.
   - Do not infer beyond the schema.
4. Formulate Output:
   - Create a slot ONLY when there is reasonable support from API history or memories.
   - Do NOT create empty slots or hallucinate values.

Schema (Consider valid slots for the domain):
{schema}

Relevant Memories (User Preferences & Constraints):
{memory}

Current User Utterance:
{query}

Output Format:
Get~(slot_name="value", ...)

Now produce ONLY the final Service API call:
"""


def action_template(mode: str) -> str:
    if mode == "singleturn":
        return SINGLE_ACTION_TEMPLATE
    if mode == "multiturn":
        return MULTI_ACTION_TEMPLATE
    raise PublicInputError("mode must be singleturn or multiturn")


def action_template_sha256(mode: str) -> str:
    return hashlib.sha256(action_template(mode).encode("utf-8")).hexdigest()


def build_action_prompt(*, mode: str, schema: object, memory: str, query: str) -> str:
    if not isinstance(memory, str) or not memory:
        raise PublicInputError("memory must be a nonempty string")
    if not isinstance(query, str) or not query.strip():
        raise PublicInputError("query must be a nonempty string")
    schema_text = json.dumps(schema, ensure_ascii=False, indent=2) if schema else "No specific schema provided."
    return action_template(mode).format(schema=schema_text, memory=memory, query=query.strip())
