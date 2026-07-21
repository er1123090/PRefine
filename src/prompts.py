"""Shared prompt templates for experiments5 public method runners."""

IMPLICIT_ZS_PROMPT_TEMPLATE = """You are a Personalized Preference Extraction Specialist.
Your goal is to infer the user's "Preferences" based on their dialogue history and current utterance, strictly following the provided preference schema.

[Task Definition]
The user may not explicitly state all information in the current turn. You must deduce missing information by analyzing patterns in the Dialogue History.
- **Repetitiveness**: If a user frequently chose a specific value in the past, assume this is their preference.
- **Cross-domain Consistency**: Identify the user's **universal behavioral patterns or constraints** (e.g., cost sensitivity, service level, risk aversion) exhibited in previous interactions. If a direct preference is missing, **deduce** the current slot's value by applying these established patterns.

[Reasoning Steps]
1. **Filter Context**: Focus ONLY on the slots listed in the `Target Preference Schema`. Ignore transient slots like specific dates or times.
2. **Identify Patterns in History**:
   - Scan the `Dialogue History` for the target slots.
   - Determine the most likely preference value based on frequency and similarity.
3. **Formulate Output**:
   - Generate the final function call using the deduced information.

Target Preference Schema (Consider valid slots for the domain):
{preference_schema}

Dialogue History:
{dialogue_history}

Current User Utterance:
{user_utterance}

Output Format:
{{Domain}}({{slot_name}}="{{value}}", ...)

Now produce the final Service API call:
"""


LATENT_PREF_SYSTEM_PROMPT = """
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
{prev_implicit}

**Full Dialogue History**:
{full_dialogue}

**Full API Calls**:
{full_api_calls}

"""

LATENT_PREF_INITIAL_PROMPT = """
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

LATENT_PREF_REFINEMENT_PROMPT = """
### Task: Refine Preference Based on Evidence Gaps

Your previous preference abstraction was rejected.

You must:
- Remove any claim not directly supported by the logs.
- Increase abstraction if details are over-specified.
- Preserve only what consistently constrains action selection.

Do NOT add new information.

### Input
Draft Preference:
"{previous_draft}"

Verifier Feedback:
"{feedback}"

### Output Format (JSON)
{{
  "reasoning": "How unsupported details were removed or abstracted",
  "implicit_pref": "The revised, evidence-tight preference constraint"
}}
"""

LATENT_PREF_VERIFIER_PROMPT = """
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
{full_dialogue}

**API Calls**:
{full_api_calls}

### Candidate Preference to Verify
{candidate_pref}


### Output Format (JSON)
{{
  "valid": true/false,
  "feedback": "If false, specify whether the issue is over-specificity, hallucination, lack of abstraction, or non-actionability."
}}
"""


IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE = """You are a Personalized Preference Reasoning Agent.
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
   - Infer the value by applying the user's stable behavioral patterns
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

IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE_MULTITURN = """You are a Personalized Preference Reasoning Agent.
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
   - Infer missing values by applying the user's stable behavioral patterns
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
