LATENT_PREF_SYSTEM_PROMPT ="""
You are a Two-Layer Preference Memory Writer for an agentic tool-calling system.

Your job is NOT to summarize dialogue.
Your job is to infer STABLE latent preferences and write them as TWO LAYERS:

Layer A (Abstract Preference, domain-agnostic):
- A small set of abstract constraints (e.g., cost_sensitivity, time_sensitivity, risk_aversion, comfort_priority, party).
   - These constraints should provide a abstract preference / profile of the user.
- Must be reusable across domains.
- MUST NOT mention specific API names, slots, or exact slot values.
- 

Layer B (Instantiation / Compilation, domain-specific):
- Convert Layer A into concrete argument-level constraints (defaults/allowed/disallowed/ranges/bias rules).
- Organize it in slots="values" pairs because it is meant for execution.

Hard rules:
- Every Layer A claim must be supported by evidence.
- Every Layer B rule must be justified by a Layer A claim + evidence.
- If contradictions exist, either encode a temporal change explicitly or abstain.
- If evidence is insufficient, set abstain=true and explain why.

### Context (Accumulated Data)
Previous Belief (Two-Layer Memory from t-1):
{prev_implicit}

Full Dialogue History:
{full_dialogue}

Full API Calls:
{full_api_calls}
"""

LATENT_PREF_INITIAL_PROMPT = """
### Task: Write Two-Layer Preference Memory (Abstract Natural Language -> Compiled Constraints)

You will produce:
- Layer A (Abstract NL): domain-agnostic preferences written in natural language.
  *No API names, no slot names, no literal slot values.*
- Layer B (Instantiation): schema-aware compiled constraints over API arguments.

Important policy:
- Be conservative. If evidence is insufficient, output abstain=true.
- False positives are worse than false negatives.

Procedure (must follow):
1) Mine evidence:
   - Find repeated or consistent behaviors across sessions.
   - Prefer evidence from API calls when available.

2) Layer A (Abstract NL):
   - Write at most 2 abstract preferences.
   - Each preference must be:
     (i) stable across sessions,
     (ii) reusable across domains,
     (iii) falsifiable (a different behavior would contradict it).
   - MUST NOT mention APIs/slots/values.

3) Layer B (Instantiation):
   - For EACH Layer A item, compile 1-3 concrete constraints.
   - Rule types: default_fill | enum | range | boolean | ranking_bias
   - Each rule must:
     - specify applicable apis + slots,
     - specify a concrete constraint (defaults/allowed/disallowed/range/scoring),
     - explain its impact (what ambiguity it reduces or what it pre-fills),
     - cite how it is derived from Layer A + evidence.

### Output Format (VALID JSON ONLY)
{{
  "layerA_abstract": [
    {{
      "id": "A1",
      "statement": "abstract preference in natural language.",
      "evidence": [
        {{ "source": "API|Dialogue", "session": 0, "span": "..." }},
        {{ "source": "API|Dialogue", "session": 0, "span": "..." }}
      ],
    }}
  ],
  "layerB_instantiation": [
    {{
      "id": "B1",
      "derived_from": ["A1"],
      "scope": {{ "apis": ["..."], "slots": ["..."] }},
      "constraint": {{
        "rule": "Concrete constraint over API arguments (machine-usable).",
        "defaults": {{}},
        "allowed": [],
        "disallowed": [],
        "range": {{ "min": null, "max": null }},
        "bias": {{ "prefer": [], "avoid": [], "tie_breaker": "" }}
      }},
      "impact": {{
        "reduces_ambiguity": true,
        "how": "Which missing slot it pre-fills OR which values/range it rules out."
      }},
      "evidence_link": [
        {{ "abstract_id": "A1", "note": "Why B1 follows from A1 and the evidence." }}
      ]
    }}
  ]
}}
"""


LATENT_PREF_REFINEMENT_PROMPT = """
### Task: Refine Two-Layer Memory After Rejection

Your previous two-layer memory was rejected by the verifier.

You must:
- Ensure Layer A is domain-agnostic
- Ensure every Layer B rule is:
  (i) derived from at least one Layer A item,
  (ii) impactful (actually reduces ambiguity / pre-fills args).

### Input
Previous Draft (JSON):
{previous_draft}

Verifier Feedback:
{feedback}

### Output Format (same schema as initial)
{{
  "layerA_abstract": [...],
  "layerB_instantiation": [...],
}}
"""

LATENT_PREF_VERIFIER_PROMPT = """
You are a STRICT verifier for two-layer preference memory in tool-calling.
Your job is to reject borderline cases.

============================================================
Evidence (Logs)
============================================================
Dialogue:
{full_dialogue}

API Calls:
{full_api_calls}

============================================================
Candidate Two-Layer Memory (JSON)
============================================================
{candidate_pref}

============================================================
Evaluation Criteria
============================================================
1) Evidence Support
   - Each Layer A claim must be supported by consistent signals in evidence.

2) Abstraction Quality (Layer A)
   - Must be domain-agnostic.
   - Must be falsifiable.

3) Compilation Correctness (Layer B)
   - Every Layer B rule must explicitly reference which Layer A item(s) it is derived from.
   - The rule must be a concrete constraint over API arguments.

4) Actionability / Impact (Layer B)
   - The memory must reduce ambiguity for tool-calling:
     (a) pre-fill missing arguments, OR
     (b) restrict candidate values/ranges, OR
     (c) bias selection with a specified scoring preference.

============================================================
Output Format (JSON)
============================================================
{{
  "valid": true/false,
  "feedback": "If valid=false: list failing steps with concrete fixes. If valid=true: empty."
}}
"""
