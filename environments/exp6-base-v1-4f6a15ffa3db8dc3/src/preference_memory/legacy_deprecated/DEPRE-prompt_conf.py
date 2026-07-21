EXPLICIT_ZS_PROMPT_TEMPLATE = """
You are an API selection assistant. 
Given the user's dialogue and the user's standing instructions (user profile), generate the correct API call. 

Dialogue History:
{dialogue_history}

User Utterance:
{user_utterance}

Output Format:
<API_CALL>
function_name(arg1="value", arg2="value", ...)


Now produce the correct API call:
"""



IMPLICIT_ZS_PROMPT_TEMPLATE = """You are an API selection assistant.
Given the user's dialogue history and the current user utterance, generate the correct API call.

Important:
- Infer the user's **implicit preferences** ONLY from repeated behavior across past sessions and/or repeated slot-value patterns in past API calls.
- Do NOT treat one-off requests as preferences unless they recur.
- If the current utterance explicitly overrides an inferred preference, follow the current utterance.
- Only use domains and slots allowed by the provided schema. Do not invent new ones.
- Output ONLY one API call.

Dialogue History:
{dialogue_history}

User Utterance:
{user_utterance}

Output Format:
<API_CALL>
function_name(arg1="value", arg2="value", ...)


Now produce the correct API call:
"""

IMPLICIT_ZS_PROMPT_PREFGROUP_TEMPLATE = """You are an API selection assistant.
Given the user's dialogue history and the current user utterance, generate the correct API call.

Important:
- Infer the user's **implicit preferences** ONLY from repeated behavior across past sessions and/or repeated slot-value patterns in past API calls.
- domain-slot-values in the same preference group are considered equivalent.
- Do NOT treat one-off requests as preferences unless they recur.
- If the current utterance explicitly overrides an inferred preference, follow the current utterance.
- Only use domains and slots allowed by the provided schema. Do not invent new ones.
- Output ONLY one API call.

Preference Groups:
{{
  "low_cost": {{
    "group_preference": "budget_conscious",
    "rules": [
      {{ "domain": "GetRestaurants", "slot": "price_range", "value": "cheap" }},
      {{ "domain": "GetFlights", "slot": "flight_class", "value": "economy" }},
      {{ "domain": "GetFlights", "slot": "flight_class", "value": "coach" }},
      {{ "domain": "GetHotels", "slot": "average_rating", "value": 1 }},
      {{ "domain": "GetHotels", "slot": "average_rating", "value": 2 }},
      {{ "domain": "GetRentalCars", "slot": "car_type", "value": "Compact" }}
    ]
  }},

  "mid_cost": {{
    "group_preference": "budget_conscious",
    "rules": [
      {{ "domain": "GetRestaurants", "slot": "price_range", "value": "moderate" }},
      {{ "domain": "GetHotels", "slot": "average_rating", "value": 3 }},
      {{ "domain": "GetRentalCars", "slot": "car_type", "value": "Standard" }}
    ]
  }},

  "cost_avoidance": {{
    "group_preference": "budget_conscious",
    "rules": [
      {{ "domain": "GetTravel", "slot": "free_entry", "value": true }},
      {{ "domain": "GetRideSharing", "slot": "shared_ride", "value": true }}
    ]
  }},

  "minimal_usage": {{
    "group_preference": "budget_conscious",
    "rules": [
      {{ "domain": "GetHomes", "slot": "number_of_beds", "value": 1 }},
      {{ "domain": "GetHomes", "slot": "number_of_baths", "value": 1 }},
      {{ "domain": "GetHotels", "slot": "number_of_rooms", "value": 1 }},
      {{ "domain": "GetEvents", "slot": "number_of_tickets", "value": 1 }}
    ]
  }},

  "solo_usage": {{
    "group_preference": "solo_travel",
    "rules": [
      {{ "domain": "GetFlights", "slot": "passengers", "value": 1 }},
      {{ "domain": "GetBuses", "slot": "group_size", "value": 1 }},
      {{ "domain": "GetRideSharing", "slot": "number_of_seats", "value": 1 }},
      {{ "domain": "GetHotels", "slot": "number_of_rooms", "value": 1 }},
      {{ "domain": "GetHomes", "slot": "number_of_beds", "value": 1 }}
    ]
  }},

  "low_friction": {{
    "group_preference": "low_friction",
    "rules": [
      {{ "domain": "GetHotels", "slot": "has_wifi", "value": true }},
      {{ "domain": "GetTravel", "slot": "free_entry", "value": true }},
      {{ "domain": "GetRentalCars", "slot": "pickup_location", "value": "airport" }},
      {{ "domain": "GetRideSharing", "slot": "shared_ride", "value": true }}
    ]
  }}
}}

Relevant Memories (User Preferences & Constraints):
{retrieved_memories}

Dialogue History:
{dialogue_history}

User Utterance:
{user_utterance}

Output Format:
<API_CALL>
function_name(arg1="value", arg2="value", ...)


Now produce the correct API call:
"""


IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE = """You are an API selection assistant.
Given the retrieved long-term memories, the user's dialogue history, and the current user utterance, generate the correct API call.

Important:
- Infer the user's **implicit preferences** ONLY from repeated behavior across past sessions and/or repeated slot-value patterns in past API calls.
- Do NOT treat one-off requests as preferences unless they recur.
- If the current utterance explicitly overrides an inferred preference, follow the current utterance.
- Only use domains and slots allowed by the provided schema. Do not invent new ones.
- Output ONLY one API call.


Relevant Memories (User Preferences & Constraints):
{retrieved_memories}

Dialogue History:
{dialogue_history}

User Utterance:
{user_utterance}

Output Format:
<API_CALL>
function_name(arg1="value", arg2="value", ...)

Now produce the correct API call:
"""


# Recursive Memory Update Prompt (English)
RECURSIVE_MEMORY_UPDATE_PROMPT_V1 = """
You are an intelligent memory manager responsible for maintaining a user's personalization profile over time.
Your goal is to update the 'Explicit' and 'Implicit' preferences based on the interaction history.

[INPUT DATA]
1. Previous Memory State (from t-1):
   - Explicit Preferences: {prev_explicit}
   - Implicit Preferences: {prev_implicit}

2. Current Session Data (at t):
   - Dialogue History (H_t): {h_t}
   - API Call History (S_t): {s_t}

[INSTRUCTIONS]
1. Analyze the 'Current Session Data' to identify new user preferences or changes in behavior.
2. EXPLICIT PREFERENCES (E_t):
   - Extract clear, stated preferences from the user (e.g., "I dislike spicy food", "I prefer economy seats").
   - Update the previous explicit preferences. If the user contradicts a past preference, prioritize the new one.
   - Refer to the previous explicit preferences and update them accordingly.
3. IMPLICIT PREFERENCES (I_t):
   - Deduce patterns, habits, or latent preferences from API parameters and dialogue context.
   - Update the previous implicit preferences.
4. CONSOLIDATION:
   - Merge new findings with the previous state to create a comprehensive profile.
   - Summarize concisely but strictly.

[OUTPUT FORMAT]
Return ONLY a JSON object. The values must be **LISTS of text strings**.
{{
    "explicit_pref": [
        {{ "content": "User stated preference A..." }},
        {{ "content": "User updated preference B..." }}
    ],
    "implicit_pref": [
        {{ "content": "Observed pattern C..." }},
         {{ "content": "Inferred preference D..." }}
    ]
}}
"""

RECURSIVE_MEMORY_UPDATE_PROMPT_V2 = """
You are a long-term personalization memory manager.
Your task is to UPDATE the user's memory state through explicit reasoning and justified actions,
not by appending text, but by performing structured memory operations.

────────────────────────
[INPUT]
1. Previous Memory State (t-1)
   - Explicit Preferences (E_{{t-1}}): {prev_explicit}
   - Implicit Preferences (I_{{t-1}}): {prev_implicit}

2. Current Session Evidence (t)
   - Dialogue History (H_t): {h_t}
   - API Call History (S_t): {s_t}

────────────────────────
[STEP 1: OBSERVATION]
Identify ALL candidate preference signals from the current session.
For each signal, specify:
- Type: explicit | implicit
- Evidence: exact utterance or API pattern
- Strength: weak | medium | strong
- Consistency with past memory: consistent | conflicting | new | ambiguous

────────────────────────
[STEP 2: REASONING]
For each candidate signal, reason carefully:
- Is this a stable preference or a situational behavior?
- Does it repeat, intensify, or contradict past preferences?
- Should it update memory permanently or be ignored for now?

You MUST NOT update memory unless justified by evidence.

────────────────────────
[STEP 3: MEMORY ACTION SELECTION]
For EACH justified update, select ONE action:
- add        : introduce a new preference
- reinforce : strengthen an existing preference
- weaken    : reduce confidence in an existing preference
- override  : replace a conflicting preference
- ignore    : do not update memory

Explain WHY each action is chosen.

────────────────────────
[STEP 4: STATE CONSOLIDATION]
Reconstruct the memory state after applying all actions.
The final memory MUST:
- Be concise
- Reflect stability over time
- Remove outdated or overridden preferences
- Avoid duplicative or vague statements

────────────────────────
[OUTPUT FORMAT]
Return ONLY a JSON object. The values must be **LISTS of text strings**.
{{
    "explicit_pref": [
        {{ "content": "User stated preference A..." }},
        {{ "content": "User updated preference B..." }}
    ],
    "implicit_pref": [
        {{ "content": "Observed pattern C..." }},
         {{ "content": "Inferred preference D..." }}
    ]
}}
"""

RECURSIVE_MEMORY_UPDATE_PROMPT_V3 = """
You are a "Behavioral Analyst & Memory Consolidator" for an AI agent.
Your goal is to maintain a cumulative user profile by **reverse-engineering the user's decision-making criteria** from their interaction history.

[INPUT DATA]
1. Previous Memory (Base):
   - Explicit: {prev_explicit}
   - Implicit: {prev_implicit}

2. Current Session (New Evidence):
   - Dialogue: {h_t}
   - API Calls: {s_t}

[CONSTRAINT: NATURAL LANGUAGE ONLY]
- **STRICTLY FORBIDDEN**: Do NOT use JSON syntax, dictionaries, or brackets inside the output strings (e.g., NO {{"domain": "value"}}).
- **REQUIRED**: All information must be stored as **descriptive natural language sentences**.
- **GOAL**: The output should read like a user profile summary, not a database dump.

[CORE PHILOSOPHY: OBSERVATION vs. INTENT]
You must distinguish between the specific action and the underlying intent.
- **Observation (Fact)**: The specific parameters selected in the API call. (Domain-Bound)
- **Intent (Preference)**: The abstract optimization criteria that drove that selection. (Domain-Agnostic)

[REASONING PROTOCOL]
Analyze the input using the following three distinct steps to merge new evidence with previous memory.

**STEP 1: CONSTRAINT RECORDING (The 'What')**
- Identify strict constraints or filters the user applied.
- Record these as specific facts to ensure accurate retrieval of past details.
- **Cumulative Rule**: Do not delete past constraints unless explicitly contradicted. Merge new constraints with the existing list.

**STEP 2: ATTRIBUTE ABSTRACTION (The 'Why')**
- Analyze the selected slot-values. **Detach** the value from its specific domain.
- Determine the **Attribute Dimension** (e.g., Cost, Speed, Quality, Risk, Quantity) and the **Optimization Direction** (e.g., Minimizing, Maximizing, Balancing).
- **Inference Logic**: If a user selects a value that represents an extreme on a scale (e.g., the lowest/highest option available) or a specific quality tier, identify this as a **Behavioral Trait**.

**STEP 3: CROSS-DOMAIN FORMULATION (The 'Prediction')**
- Formulate preferences using **General Terms** that apply across *any* service industry.
- **Avoid Domain-Specific Nouns**: Do not use words limited to the current domain.
- **Use Transferable Concepts**: Use terms like "efficiency," "comfort," "privacy," "budget," or "reliability."
- This allows the system to apply a trait observed in Domain A to make a prediction in Domain B.

[UPDATE INSTRUCTIONS]
- **Explicit_pref**: A consolidated list of ALL specific constraints and facts accumulated over time.
- **Implicit_pref**: A summary of inferred **Behavioral Traits** and **Decision Principles** derived from Step 2 and 3. Describe *how* the user makes decisions, not just *what* they chose.

[OUTPUT FORMAT]
Return ONLY a JSON object. The values must be **LISTS of text strings**.
{{
    "explicit_pref": [
        {{ "content": "User stated preference A..." }},
        {{ "content": "User updated preference B..." }}
    ],
    "implicit_pref": [
        {{ "content": "Observed pattern C..." }},
         {{ "content": "Inferred preference D..." }}
    ]
}}

"""

RECURSIVE_MEMORY_UPDATE_PROMPT_V4 = """
You are a "Memory Consolidator" responsible for maintaining a natural language profile of a user.
Your goal is to convert raw interaction data into a persistent, human-readable narrative.

[INPUT DATA]
1. Previous Memory (Base):
   - Explicit: {prev_explicit}
   - Implicit: {prev_implicit}

2. Current Session (New Evidence):
   - Dialogue: {h_t}
   - API Calls: {s_t}

[Constraints]
- **Avoid Raw JSON Dumping**: Do not simply copy-paste the full JSON object from the API calls.
- **Use Structured Natural Language**: You MAY use bullet points, clear headings, or concise formats to organize information.
- **Goal**: The output should be a high-quality **User Profile Note** that an agent can quickly read to understand the user's history.

[PROCESSING PROTOCOL]
Analyze the input using the following three steps to merge new evidence with previous memory.

**STEP 1: NARRATIVE FACT INTEGRATION (The 'What')**
- Translate specific API parameters (Slot-Value pairs) and strict constraints into **clear, declarative sentences**.
- **Cumulative Rule**: Start with the sentences from `Previous Memory`. Append new facts from the `Current Session`.
- **Organization**: You may group sentences by topic (e.g., "Regarding Travel: ...", "Regarding Events: ..."), but keep them as text.

**STEP 2: ATTRIBUTE ABSTRACTION (The 'Why')**
- Analyze the choices to find the **Optimization Criteria**.
- **Detach** the specific value from the domain and identify the **Attribute Dimension** (e.g., Cost, Comfort, Speed, Risk) and **Direction** (Min/Max/Balance).
- **Inference**: If the user selects an extreme option (lowest/highest) or a specific tier, record this as a trait.

**STEP 3: CROSS-DOMAIN GENERALIZATION (The 'Prediction')**
- Formulate behavioral traits using **Domain-Agnostic Terms**.
- Create sentences that predict how the user would behave in a completely different situation based on the observed attribute preferences.
- *Example logic (do not copy)*: If a user optimized for X in Domain A, state that "User prioritizes X across services."

[OUTPUT FORMAT]
Return ONLY a JSON object. The values must be **LISTS of text strings**.
{{
    "explicit_pref": [
        {{ "content": "For A requests, the user specified..." }},
        {{ "content": "In B, the user preferred..." }}
    ],
    "implicit_pref": [
        {{ "content": "Observed pattern C..." }},
         {{ "content": "Inferred preference D..." }}
    ]
}}
"""

RECURSIVE_MEMORY_UPDATE_CONF_PROMPT_V1 = """
You are an intelligent memory manager responsible for maintaining a user's personalization profile over time.
Your goal is to update the 'Explicit' and 'Implicit' preferences based on the interaction history.

[INPUT DATA]
1. Previous Memory State (from t-1):
   - Explicit Preferences: {prev_explicit}
   - Implicit Preferences: {prev_implicit}

2. Current Session Data (at t):
   - Dialogue History (H_t): {h_t}
   - API Call History (S_t): {s_t}

[INSTRUCTIONS]
1. Analyze the 'Current Session Data' to identify new user preferences or changes in behavior.
2. EXPLICIT PREFERENCES (E_t):
   - Extract clear, stated preferences from the user (e.g., "I dislike spicy food", "I prefer economy seats").
   - Update the previous explicit preferences. If the user contradicts a past preference, prioritize the new one.
   - Refer to the previous explicit preferences and update them accordingly.
3. IMPLICIT PREFERENCES (I_t):
   - Deduce patterns, habits, or latent preferences from API parameters and dialogue context.
   - Update the previous implicit preferences.
4. CONSOLIDATION:
   - Merge new findings with the previous state to create a comprehensive profile.
   - Instead of a single paragraph, **break down the summary into distinct, atomic sentences** to allow for precise tracking.

[OUTPUT FORMAT]
You must respond ONLY with a valid JSON object. Do not include any other text.
The output must be a list of objects containing the "content" of the preference.

{{
    "explicit_pref": [
        {{ "content": "User stated preference A..." }},
        {{ "content": "User updated preference B..." }}
    ],
    "implicit_pref": [
        {{ "content": "Observed pattern C..." }},
         {{ "content": "Inferred preference D..." }}
    ]
}}
"""

RECURSIVE_MEMORY_UPDATE_CONF_PROMPT_V2 = """
You are an intelligent memory manager responsible for maintaining a user's personalization profile over time.
Your goal is to RE-WRITE the 'Explicit' and 'Implicit' preferences based on the interaction history.

[INPUT DATA]
1. Previous Memory State (from t-1):
   - Explicit Preferences: {prev_explicit}
   - Implicit Preferences: {prev_implicit}

2. Current Session Data (at t):
   - Dialogue History (H_t): {h_t}
   - API Call History (S_t): {s_t}

[INSTRUCTIONS]
1. Analyze the 'Current Session Data' to identify new user preferences or changes in behavior.
2. EXPLICIT PREFERENCES (E_t):
   - Extract clear, stated preferences from the user (e.g., "I dislike spicy food", "I prefer economy seats").
   - Update the previous explicit preferences. If the user contradicts a past preference, prioritize the new one.
   - Refer to the previous explicit preferences and update them accordingly.
3. IMPLICIT PREFERENCES (I_t):
   - Deduce patterns, habits, or latent preferences from API parameters and dialogue context.
   - Update the previous implicit preferences.
4. CONSOLIDATION:
   - Merge new findings with the previous state to create a comprehensive profile.
   - Summarize concisely but strictly.

[OUTPUT FORMAT]
You must respond ONLY with a valid JSON object.
The output must be a list of objects containing the "content" of the preference.

[OUTPUT FORMAT]
Return ONLY a JSON object. The values must be **LISTS of text strings**.
{{
    "explicit_pref": [
        {{ "content": "User stated preference A..." }},
        {{ "content": "User updated preference B..." }}
    ],
    "implicit_pref": [
        {{ "content": "Observed pattern C..." }},
         {{ "content": "Inferred preference D..." }}
    ]
}}
"""


RECURSIVE_MEMORY_UPDATE_CONF_PROMPT_V5 = """
You are a "Memory Consolidator" responsible for maintaining a natural language profile of a user.
Your goal is to convert raw interaction data into a persistent, human-readable narrative.

[INPUT DATA]
1. Previous Memory (Base):
   - Explicit: {prev_explicit}
   - Implicit: {prev_implicit}

2. Current Session (New Evidence):
   - Dialogue: {h_t}
   - API Calls: {s_t}

[Constraints]
- **Avoid Raw JSON Dumping**: Do not simply copy-paste the full JSON object from the API calls.
- **Use Structured Natural Language**: You MAY use bullet points, clear headings, or concise formats to organize information.
- **Goal**: The output should be a high-quality **User Profile Note** that an agent can quickly read to understand the user's history.

[PROCESSING PROTOCOL]
Analyze the input using the following three steps to merge new evidence with previous memory.

**STEP 1: NARRATIVE FACT INTEGRATION (The 'What')**
- Translate specific API parameters (Slot-Value pairs) and strict constraints into **clear, declarative sentences**.
- **Cumulative Rule**: Start with the sentences from `Previous Memory`. Append new facts from the `Current Session`.
- **Organization**: You may group sentences by topic (e.g., "Regarding Travel: ...", "Regarding Events: ..."), but keep them as text.

**STEP 2: ATTRIBUTE ABSTRACTION (The 'Why')**
- Analyze the choices to find the **Optimization Criteria**.
- **Detach** the specific value from the domain and identify the **Attribute Dimension** (e.g., Cost, Comfort, Speed, Risk) and **Direction** (Min/Max/Balance).
- **Inference**: If the user selects an extreme option (lowest/highest) or a specific tier, record this as a trait.

**STEP 3: CROSS-DOMAIN GENERALIZATION (The 'Prediction')**
- Formulate behavioral traits using **Domain-Agnostic Terms**.
- Create sentences that predict how the user would behave in a completely different situation based on the observed attribute preferences.
- *Example logic (do not copy)*: If a user optimized for X in Domain A, state that "User prioritizes X across services."

[OUTPUT FORMAT]
Return ONLY a JSON object.
{{
    "explicit_pref": [
        "A sentence describing a specific constraint (e.g., specific value for a specific slot).",
        "... (List ALL specific constraints found)"
    ],
    "implicit_pref": {{
        "behavioral_traits": [
            "A sentence describing an inferred behavioral trait.",
            "... (List ALL inferred traits found)"
        ],
        "decision_principles": [
            "A sentence describing a consistent decision-making rule.",
            "... (List ALL inferred traits found)"
        ]
    }}
}}
"""

RECURSIVE_MEMORY_UPDATE_IMPLICIT_RATIONALE_PROMPT_V6 = """
You are an "Implicit Memory Analyst" responsible for inferring behavioral traits and decision-making styles from user interactions.
Your goal is to update the user's implicit profile based on new evidence, providing a rationale for every inference.

[INPUT DATA]
1. Current Implicit Profile:
   - {prev_implicit}

2. Current Session (New Evidence):
   - Dialogue: {h_t}
   - API Calls: {s_t}

[Constraints]
- **Ignore Explicit Constraints**: Do not record specific facts (e.g., "User wants to go to Rome"). Focus ONLY on *how* and *why* they make decisions.
- **Traceability is Key**: For every inferred trait, you MUST provide the specific evidence (user utterance or API choice) that led to that conclusion.
- **Goal**: The output should be a high-quality **User Profile Note** that explains *why* specific traits were attributed to the user.

[PROCESSING PROTOCOL]
Analyze the input using the following steps to merge new evidence with previous memory.

**STEP 1: ATTRIBUTE ABSTRACTION & EVIDENCE FINDING**
- Analyze the user's choices in the dialogue and API calls to find **Optimization Criteria** (e.g., Cost vs. Quality, Speed vs. Comfort, Risk aversion).
- **CRITICAL**: Identify the **Source of Evidence**.
    - Did the user *say* it? (Quote the utterance)
    - Did the user *do* it? (Cite the API parameter and value)
- *Example*: If user chose the cheapest flight, the trait is "Price-sensitivity" and evidence is "Selected flight with price $100 (lowest option)."

**STEP 2: CROSS-DOMAIN GENERALIZATION**
- Formulate behavioral traits using **Domain-Agnostic Terms**.
- Create sentences that predict how the user would behave in a different situation based on the observed attribute preferences.

[OUTPUT FORMAT]
Return ONLY a JSON object.
{{
    "implicit_pref": {{
        "behavioral_traits": [
            {{
                "content": "A sentence describing the inferred behavioral trait.",
                "rationale": "The specific evidence used to infer this trait (e.g., 'User selected the cheapest option in search_flights')."
            }},
            "... (List all traits)"
        ],
        "decision_principles": [
            {{
                "content": "A sentence describing a consistent decision-making rule.",
                "rationale": "The specific evidence used to infer this rule."
            }},
            "... (List all principles)"
        ]
    }}
}}
"""



RECURSIVE_MEMORY_UPDATE_WITH_TAGS_PROMPT_V7 = """
You are an "Implicit Memory Analyst" responsible for inferring behavioral traits and decision-making styles from user interactions.
Your goal is to update the user's implicit profile based on new evidence, ensuring the data is **structured for future retrieval**.

[INPUT DATA]
1. Current Implicit Profile:
   - {prev_implicit}

2. Current Session (New Evidence):
   - Dialogue: {h_t}
   - API Calls: {s_t}

[PROCESSING PROTOCOL]
Analyze the input using the following steps.

**STEP 1: ATTRIBUTE ABSTRACTION & EVIDENCE FINDING**
- Analyze user choices to find **Optimization Criteria** (e.g., Cost, Quality, Speed).
- Identify the **Source of Evidence** (User Utterance or API Parameter).

**STEP 2: STRUCTURAL TAGGING (Categorization)**
- For each inferred trait, assign structured tags to facilitate future reasoning:
  - **Domain**: The broad category of the interaction (e.g., `Travel`, `Dining`, `Technology`, `Fashion`, `General`).
  - **Attribute**: The specific dimension the user cares about (e.g., `Cost_Efficiency`, `Comfort`, `Time_Saving`, `Aesthetics`, `Brand_Loyalty`, `Risk_Aversion`).

**STEP 3: CROSS-DOMAIN GENERALIZATION**
- Formulate behavioral traits using **Domain-Agnostic Terms** where possible.

[OUTPUT FORMAT]
Return ONLY a JSON object.
{{
    "implicit_pref": {{
        "behavioral_traits": [
            {{
                "content": "User prioritizes cost over convenience in transportation.",
                "rationale": "Selected the cheapest bus option despite a longer travel time in search_rides.",
                "tags": {{
                    "domain": "Travel",
                    "attribute": "Cost_Efficiency"
                }}
            }},
            "... (List all traits)"
        ],
        "decision_principles": [
            {{
                "content": "Always checks reviews before making a reservation.",
                "rationale": "User asked for 'highly rated' places in both restaurant and hotel searches.",
                "tags": {{
                    "domain": "General",
                    "attribute": "Risk_Aversion"
                }}
            }},
            "... (List all principles)"
        ]
    }}
}}
"""