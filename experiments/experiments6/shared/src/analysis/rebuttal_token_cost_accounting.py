from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import tiktoken


ROOT = Path("/data/minseo/experiments6")
DATA_PATH = ROOT / "data/1229_dev_6.json"
SCHEMA_EASY_PATH = ROOT / "schema_easy.json"
SCHEMA_ALL_PATH = ROOT / "schema_all.json"

OURS_MEMORY_ROOT = ROOT / "ours_memory/inference/1231_MEMORY3"
OURS_SINGLE_ROOTS = [
    ROOT / "ours_memory/inference/1231_MEMORY3_0317_single_api-only",
    ROOT / "ours_memory/inference/1231_MEMORY3_0317_single_mem-only",
]
OURS_MULTI_ROOTS = [
    ROOT / "ours_memory/inference/1231_MEMORY3_0317_multi_api-only",
    ROOT / "ours_memory/inference/1231_MEMORY3_0317_multi_mem-only",
]

MEM0_CONSTRUCTION_CSV = ROOT / "mem0/0323_memory-tokens-265/mem0_265_examples.merged.csv"
MEM0_INFERENCE_ROOT = ROOT / "mem0/inference_0309"

LANGMEM_SNAPSHOT_ROOT = ROOT / "langmem/memory_snapshots/semantic-custom"
LANGMEM_INFERENCE_ROOTS = [
    ROOT / "langmem/inference_single",
    ROOT / "langmem/inference_multi_minimal",
]

OUT_CONSTRUCTION = ROOT / "rebuttal_token_cost_construction_detail.csv"
OUT_INFERENCE = ROOT / "rebuttal_token_cost_inference_detail.csv"
OUT_SUMMARY = ROOT / "rebuttal_token_cost_summary.csv"
OUT_MD = ROOT / "rebuttal_token_cost_summary.md"


OURS_SINGLE_TEMPLATE = """You are a Personalized Preference Reasoning Agent.
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


OURS_MULTI_TEMPLATE = """You are a Personalized Preference Reasoning Agent.
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


def encoding() -> Any:
    return tiktoken.get_encoding("cl100k_base")


ENC = encoding()


def token_count(text: Any) -> int:
    if text is None:
        return 0
    if not isinstance(text, str):
        text = json.dumps(text, ensure_ascii=False, sort_keys=True)
    if not text:
        return 0
    return token_count_str(text)


@lru_cache(maxsize=200_000)
def token_count_str(text: str) -> int:
    return len(ENC.encode(text))


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def load_dataset_map() -> Dict[str, Dict[str, Any]]:
    data = read_json(DATA_PATH)
    return {str(item.get("example_id")): item for item in data}


def load_tools_schema(path: Path) -> List[Dict[str, Any]]:
    return read_json(path)


def format_dialogue(dialogue_list: Sequence[Dict[str, Any]]) -> str:
    return "\n".join(
        f"{turn.get('role', '')}: {turn.get('message') or turn.get('content') or ''}"
        for turn in dialogue_list
    )


def format_sessions_dialogue(example: Dict[str, Any]) -> str:
    sessions = example.get("sessions", [])
    rendered = []
    for idx, session in enumerate(sessions, start=1):
        lines = [f"[Session {idx}]"]
        for turn in session.get("dialogue", []):
            role = str(turn.get("role", "")).capitalize()
            content = turn.get("message") or turn.get("content") or ""
            if role and content:
                lines.append(f"{role}: {content}")
        rendered.append("\n".join(lines))
    return "\n\n".join(rendered) if rendered else "None"


def format_memory_content(content: Any) -> str:
    if content is None:
        return "None"
    if isinstance(content, (dict, list)):
        return json.dumps(content, indent=2, ensure_ascii=False)
    return str(content).strip()


def format_api_calls(api_calls: Any) -> str:
    if not api_calls or not isinstance(api_calls, list):
        return "None"
    return "\n".join(str(item) for item in api_calls)


def build_ours_memory_blocks(
    example: Dict[str, Any],
    user_memory: Dict[str, Any],
    context_type: str,
) -> Tuple[str, str, int, int]:
    raw_implicit_pref = (
        user_memory.get("final_implicit_preference")
        or user_memory.get("final_implicit_pref")
    )
    implicit_pref_str = format_memory_content(raw_implicit_pref)
    raw_api_calls = (
        user_memory.get("final_accumulated_api_calls")
        or user_memory.get("final_api_list")
    )
    api_history_str = format_api_calls(raw_api_calls)
    raw_explicit_pref = (
        user_memory.get("final_explicit_preference")
        or user_memory.get("final_explicit_pref")
    )
    explicit_pref_str = format_memory_content(raw_explicit_pref)

    memory_block_parts = []
    if raw_explicit_pref:
        memory_block_parts.append(f"[Explicit Preferences]:\n{explicit_pref_str}")
    memory_block_parts.append(f"[Implicit Preferences]:\n{implicit_pref_str}")

    if context_type == "memory_only":
        retrieved = "\n\n".join(memory_block_parts)
        dialogue = "None"
    elif context_type == "memory_api":
        memory_block_parts.append(f"[Past API History]:\n{api_history_str}")
        retrieved = "\n\n".join(memory_block_parts)
        dialogue = "None"
    elif context_type == "memory_diag":
        retrieved = "\n\n".join(memory_block_parts)
        dialogue = format_sessions_dialogue(example)
    else:
        retrieved = "\n\n".join(memory_block_parts)
        dialogue = format_sessions_dialogue(example)
    return retrieved, dialogue, token_count(implicit_pref_str), token_count(retrieved)


def estimate_ours_prompt_tokens(
    example: Dict[str, Any],
    user_memory: Dict[str, Any],
    utterance: str,
    context_type: str,
    turn_type: str,
    tools_schema: List[Dict[str, Any]],
) -> Tuple[int, int, int]:
    retrieved, dialogue, memory_payload_tokens, retrieved_context_tokens = build_ours_memory_blocks(
        example,
        user_memory,
        context_type,
    )
    template = OURS_MULTI_TEMPLATE if turn_type == "multi" else OURS_SINGLE_TEMPLATE
    prompt_tokens = estimate_template_prompt_tokens(
        template=template,
        schema=json.dumps(tools_schema, indent=2, ensure_ascii=False),
        retrieved=retrieved,
        dialogue=dialogue,
        utterance=str(utterance).strip(),
    )
    return prompt_tokens, memory_payload_tokens, retrieved_context_tokens


PROMPT_TOKEN_CACHE: Dict[Tuple[str, int, int, int, int], int] = {}


def estimate_template_prompt_tokens(
    template: str,
    schema: str,
    retrieved: str,
    dialogue: str,
    utterance: str,
) -> int:
    # Fast approximation for repeated large prompts. Full BPE tokenization of the
    # complete rendered prompt is exact but prohibitively slow across all runs.
    key = (
        "multi" if template is OURS_MULTI_TEMPLATE else "single",
        token_count(schema),
        token_count(retrieved),
        token_count(dialogue),
        token_count(utterance),
    )
    if key not in PROMPT_TOKEN_CACHE:
        skeleton = template.format(
            preference_schema="",
            retrieved_memories="",
            dialogue_history="",
            user_utterance="",
        )
        PROMPT_TOKEN_CACHE[key] = (
            token_count(skeleton)
            + key[1]
            + key[2]
            + key[3]
            + key[4]
        )
    return PROMPT_TOKEN_CACHE[key]


def load_ours_memory_maps() -> Dict[str, Dict[str, Dict[str, Any]]]:
    result: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for path in OURS_MEMORY_ROOT.glob("*/_memory1.jsonl"):
        model = path.parent.name
        model_alias = model.removeprefix("_deprecated_")
        rows = {}
        for obj in iter_jsonl(path):
            rows[str(obj.get("example_id"))] = obj
        result[model] = rows
        result.setdefault(model_alias, rows)
    return result


def infer_ours_context_type(path: Path) -> str:
    text = str(path)
    if "api-only" in text:
        return "memory_api"
    if "mem-only" in text:
        return "memory_only"
    if "MEM-DIAG" in text or "mem_diag" in text:
        return "memory_diag"
    return "memory_api"


def parse_ours_path(path: Path, turn_type: str) -> Tuple[str, str, str, str]:
    parts = path.parts
    root_name = (
        "1231_MEMORY3_0317_multi_api-only"
        if "1231_MEMORY3_0317_multi_api-only" in parts
        else "1231_MEMORY3_0317_multi_mem-only"
        if "1231_MEMORY3_0317_multi_mem-only" in parts
        else "1231_MEMORY3_0317_single_api-only"
        if "1231_MEMORY3_0317_single_api-only" in parts
        else "1231_MEMORY3_0317_single_mem-only"
    )
    idx = parts.index(root_name)
    memory_model = parts[idx + 1]
    context_label = parts[idx + 2] if len(parts) > idx + 2 else ""
    pref_type = parts[idx + 3] if len(parts) > idx + 3 else ""
    action_model = "/".join(parts[idx + 4 : -2]) if len(parts) > idx + 6 else ""
    return memory_model, context_label, pref_type, action_model


def collect_ours_inference_rows() -> List[Dict[str, Any]]:
    dataset_map = load_dataset_map()
    memory_maps = load_ours_memory_maps()
    schema_easy = load_tools_schema(SCHEMA_EASY_PATH)
    schema_all = load_tools_schema(SCHEMA_ALL_PATH)
    rows: List[Dict[str, Any]] = []

    for turn_type, roots in (("single", OURS_SINGLE_ROOTS), ("multi", OURS_MULTI_ROOTS)):
        tools_schema = schema_all if turn_type == "multi" else schema_easy
        for root in roots:
            if not root.exists():
                continue
            for path in sorted(root.rglob("*.json")):
                try:
                    payload = read_json(path)
                except Exception:
                    continue
                if not isinstance(payload, list):
                    continue
                memory_model, context_label, pref_type, action_model = parse_ours_path(
                    path,
                    turn_type,
                )
                memory_map = memory_maps.get(memory_model) or memory_maps.get(
                    memory_model.removeprefix("_deprecated_")
                )
                context_type = infer_ours_context_type(path)
                for obj in payload:
                    if not isinstance(obj, dict):
                        continue
                    example_id = str(obj.get("example_id"))
                    memory = memory_map.get(example_id) if memory_map else None
                    example = dataset_map.get(example_id) or obj
                    if not memory:
                        continue
                    prompt_tokens, memory_payload_tokens, retrieved_context_tokens = estimate_ours_prompt_tokens(
                        example=example,
                        user_memory=memory,
                        utterance=obj.get("test_utterance", ""),
                        context_type=context_type,
                        turn_type=turn_type,
                        tools_schema=tools_schema,
                    )
                    rows.append(
                        {
                            "method": "ours_memory",
                            "source": str(path),
                            "turn_type": turn_type,
                            "context_type": context_type,
                            "context_label": context_label,
                            "pref_type": pref_type,
                            "memory_model": memory_model,
                            "action_model": action_model,
                            "example_id": example_id,
                            "example_id_sub": obj.get("example_id_sub", ""),
                            "model_input_tokens_provider": "",
                            "model_output_tokens_provider": "",
                            "model_total_tokens_provider": "",
                            "model_input_tokens_est": prompt_tokens,
                            "model_output_tokens_est": token_count(obj.get("llm_output", "")),
                            "model_total_tokens_est": prompt_tokens
                            + token_count(obj.get("llm_output", "")),
                            "retrieved_memory_tokens": retrieved_context_tokens,
                            "memory_payload_tokens": memory_payload_tokens,
                            "token_source": "reconstructed_prompt_fast_tiktoken_estimate",
                        }
                    )
    return rows


def json_record_stream(paths: Iterable[Path]) -> Iterable[Tuple[Path, Dict[str, Any]]]:
    for path in paths:
        if path.suffix == ".json":
            try:
                payload = read_json(path)
            except Exception:
                continue
            if isinstance(payload, list):
                for obj in payload:
                    if isinstance(obj, dict):
                        yield path, obj
            elif isinstance(payload, dict):
                yield path, payload
        else:
            for obj in iter_jsonl(path):
                yield path, obj


def clean_number(value: Any) -> Any:
    if value in (None, ""):
        return ""
    return value


def sum_memory_text_tokens(memories: Any) -> int:
    if memories is None:
        return 0
    if isinstance(memories, str):
        return token_count(memories)
    total = 0
    if isinstance(memories, list):
        for item in memories:
            if isinstance(item, dict):
                total += token_count(
                    item.get("memory")
                    or item.get("display_text")
                    or item.get("content")
                    or item.get("value")
                    or item
                )
            else:
                total += token_count(item)
    return total


def collect_mem0_inference_rows() -> List[Dict[str, Any]]:
    paths = sorted(MEM0_INFERENCE_ROOT.rglob("*.log"))
    rows: List[Dict[str, Any]] = []
    for path, obj in json_record_stream(paths):
        token_counts = obj.get("token_counts") or {}
        model_input = obj.get("model_input", "")
        input_provider = clean_number(token_counts.get("input_tokens"))
        output_provider = clean_number(token_counts.get("output_tokens"))
        total_provider = clean_number(token_counts.get("total_tokens"))
        input_est = input_provider if input_provider != "" else token_count(model_input)
        output_est = (
            output_provider
            if output_provider != ""
            else token_count(obj.get("model_output") or obj.get("llm_output", ""))
        )
        rows.append(
            {
                "method": "mem0",
                "source": str(path),
                "turn_type": "multi" if "/multiturn/" in str(path) else "single",
                "context_type": obj.get("context_type", ""),
                "context_label": obj.get("context_type", ""),
                "pref_type": infer_pref_type_from_path(path),
                "memory_model": "mem0_service",
                "action_model": obj.get("model_name", ""),
                "example_id": obj.get("example_id", ""),
                "example_id_sub": obj.get("example_id_sub", ""),
                "model_input_tokens_provider": input_provider,
                "model_output_tokens_provider": output_provider,
                "model_total_tokens_provider": total_provider,
                "model_input_tokens_est": input_est,
                "model_output_tokens_est": output_est,
                "model_total_tokens_est": input_est + output_est,
                "retrieved_memory_tokens": sum_memory_text_tokens(obj.get("retrieved_memories")),
                "memory_payload_tokens": sum_memory_text_tokens(obj.get("retrieved_memories")),
                "token_source": "provider_usage_if_present_else_model_input_tiktoken",
            }
        )
    return rows


def infer_pref_type_from_path(path: Path) -> str:
    parts = set(path.parts)
    for pref in ("easy", "medium", "hard"):
        if pref in parts:
            return pref
    return ""


def collect_langmem_inference_rows() -> List[Dict[str, Any]]:
    paths: List[Path] = []
    for root in LANGMEM_INFERENCE_ROOTS:
        if root.exists():
            paths.extend(
                p for p in root.rglob("*.jsonl") if not p.name.startswith(".")
            )
    rows: List[Dict[str, Any]] = []
    for path, obj in json_record_stream(sorted(paths)):
        if "model_input" not in obj and "token_counts" not in obj:
            continue
        token_counts = obj.get("token_counts") or {}
        model_input = obj.get("model_input", "")
        input_provider = clean_number(token_counts.get("input_tokens"))
        output_provider = clean_number(token_counts.get("output_tokens"))
        total_provider = clean_number(token_counts.get("total_tokens"))
        input_est = input_provider if input_provider != "" else token_count(model_input)
        output_est = (
            output_provider
            if output_provider != ""
            else token_count(obj.get("model_output") or obj.get("llm_output", ""))
        )
        rows.append(
            {
                "method": "langmem",
                "source": str(path),
                "turn_type": "multi" if "inference_multi" in str(path) else "single",
                "context_type": obj.get("context_type", ""),
                "context_label": obj.get("context_type", ""),
                "pref_type": infer_pref_type_from_path(path),
                "memory_model": infer_langmem_memory_model(path),
                "action_model": obj.get("model_name", ""),
                "example_id": obj.get("example_id", ""),
                "example_id_sub": obj.get("example_id_sub", ""),
                "model_input_tokens_provider": input_provider,
                "model_output_tokens_provider": output_provider,
                "model_total_tokens_provider": total_provider,
                "model_input_tokens_est": input_est,
                "model_output_tokens_est": output_est,
                "model_total_tokens_est": input_est + output_est,
                "retrieved_memory_tokens": obj.get("retrieved_memory_tokens")
                if obj.get("retrieved_memory_tokens") not in (None, "")
                else sum_memory_text_tokens(obj.get("retrieved_memories")),
                "memory_payload_tokens": obj.get("retrieved_memory_tokens")
                if obj.get("retrieved_memory_tokens") not in (None, "")
                else sum_memory_text_tokens(obj.get("retrieved_memories")),
                "token_source": "provider_usage_if_present_else_model_input_tiktoken",
            }
        )
    return rows


def infer_langmem_memory_model(path: Path) -> str:
    parts = path.parts
    if "inference_single" in parts:
        idx = parts.index("inference_single")
        if len(parts) > idx + 1:
            return parts[idx + 1]
    if "inference_multi_minimal" in parts:
        idx = parts.index("inference_multi_minimal")
        if len(parts) > idx + 1:
            return parts[idx + 1]
    if "inference_multi" in parts:
        idx = parts.index("inference_multi")
        if len(parts) > idx + 1:
            return parts[idx + 1]
    return ""


def collect_inference_rows() -> List[Dict[str, Any]]:
    rows = []
    rows.extend(collect_ours_inference_rows())
    rows.extend(collect_mem0_inference_rows())
    rows.extend(collect_langmem_inference_rows())
    return rows


def collect_ours_construction_rows() -> List[Dict[str, Any]]:
    dataset_map = load_dataset_map()
    rows: List[Dict[str, Any]] = []
    for memory_path in sorted(OURS_MEMORY_ROOT.glob("*/_memory1.jsonl")):
        memory_model = memory_path.parent.name.removeprefix("_deprecated_")
        for record in iter_jsonl(memory_path):
            example_id = str(record.get("example_id"))
            example = dataset_map.get(example_id)
            if not example:
                continue
            sessions = example.get("sessions", [])
            accumulated_dialogue = ""
            accumulated_api_calls: List[str] = []
            prev_implicit = "{}"

            for session_idx, history_entry in enumerate(
                record.get("preference_evolution_history", []),
                start=1,
            ):
                session = sessions[session_idx - 1] if session_idx - 1 < len(sessions) else {}
                session_header = f"\n=== Session {session_idx} ===\n"
                session_dialogue = format_dialogue(session.get("dialogue", []))
                accumulated_dialogue = accumulated_dialogue + session_header + session_dialogue
                for api_call in session.get("api_call", []):
                    accumulated_api_calls.append(f"[Session {session_idx}] {api_call}")
                full_api_str = (
                    "\n".join(accumulated_api_calls)
                    if accumulated_api_calls
                    else "No API calls recorded."
                )
                safe_prev = prev_implicit if prev_implicit else "None"
                context_prompt = LATENT_PREF_SYSTEM_PROMPT.format(
                    prev_implicit=safe_prev,
                    full_dialogue=accumulated_dialogue,
                    full_api_calls=full_api_str,
                )
                previous_draft = None
                feedback = ""
                attempts = history_entry.get("refinement_process", [])
                session_gen_in = 0
                session_gen_out = 0
                session_verify_in = 0
                session_verify_out = 0

                for attempt in attempts:
                    step = int(attempt.get("step") or 0)
                    draft = attempt.get("draft_preference") or {}
                    if previous_draft is not None and feedback:
                        task_prompt = LATENT_PREF_REFINEMENT_PROMPT.format(
                            previous_draft=previous_draft,
                            feedback=feedback,
                        )
                    else:
                        task_prompt = LATENT_PREF_INITIAL_PROMPT
                    gen_input = token_count(context_prompt) + token_count(task_prompt)
                    gen_output = token_count(
                        json.dumps(draft, ensure_ascii=False, indent=2)
                    )
                    verify_input = token_count(attempt.get("verifier_input", ""))
                    verify_output = token_count(attempt.get("verifier_output") or {})

                    session_gen_in += gen_input
                    session_gen_out += gen_output
                    session_verify_in += verify_input
                    session_verify_out += verify_output

                    previous_draft = json.dumps(draft, ensure_ascii=False, indent=2)
                    feedback = str(attempt.get("verifier_feedback") or "")

                final_pref = history_entry.get("final_preference_at_session") or {}
                after_memory_tokens = token_count(
                    json.dumps(final_pref, ensure_ascii=False, indent=2)
                )
                implicit_pref_only_tokens = token_count(final_pref.get("implicit_pref", ""))
                rows.append(
                    {
                        "method": "ours_memory",
                        "memory_model": memory_model,
                        "example_id": example_id,
                        "session_index": session_idx,
                        "attempts": len(attempts),
                        "construction_input_tokens_est": session_gen_in
                        + session_verify_in,
                        "construction_output_tokens_est": session_gen_out
                        + session_verify_out,
                        "construction_total_tokens_est": session_gen_in
                        + session_verify_in
                        + session_gen_out
                        + session_verify_out,
                        "construction_input_tokens_lower_bound": session_gen_in
                        + session_verify_in,
                        "construction_output_tokens_lower_bound": session_gen_out
                        + session_verify_out,
                        "construction_total_tokens_lower_bound": session_gen_in
                        + session_verify_in
                        + session_gen_out
                        + session_verify_out,
                        "stored_memory_tokens_after_session": after_memory_tokens,
                        "implicit_pref_only_tokens_after_session": implicit_pref_only_tokens,
                        "stored_memory_delta_tokens": "",
                        "status": "OK",
                        "token_source": "reconstructed_generate_plus_logged_verify_tiktoken",
                    }
                )
                prev_implicit = (
                    json.dumps(final_pref, ensure_ascii=False, indent=2)
                    if final_pref
                    else "{}"
                )
    return rows


def collect_mem0_construction_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not MEM0_CONSTRUCTION_CSV.exists():
        return rows
    with MEM0_CONSTRUCTION_CSV.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("status") != "OK":
                continue
            session_input = int(float(r.get("session_input_tokens") or 0))
            delta = int(float(r.get("delta_memory_tokens") or 0))
            after = int(float(r.get("after_memory_tokens") or 0))
            output_lower = max(delta, 0)
            rows.append(
                {
                    "method": "mem0",
                    "memory_model": "mem0_service",
                    "example_id": r.get("example_id", ""),
                    "session_index": int(r.get("session_index") or 0) + 1,
                    "attempts": 1,
                    "construction_input_tokens_est": "",
                    "construction_output_tokens_est": "",
                    "construction_total_tokens_est": "",
                    "construction_input_tokens_lower_bound": session_input,
                    "construction_output_tokens_lower_bound": output_lower,
                    "construction_total_tokens_lower_bound": session_input
                    + output_lower,
                    "stored_memory_tokens_after_session": after,
                    "implicit_pref_only_tokens_after_session": "",
                    "stored_memory_delta_tokens": delta,
                    "status": "OK",
                    "token_source": "mem0_service_internal_usage_unlogged_lower_bound",
                }
            )
    return rows


def iter_langmem_snapshots() -> Iterable[Tuple[str, Path, Dict[str, Any]]]:
    for snapshot_path in sorted(LANGMEM_SNAPSHOT_ROOT.glob("*/langmem_1229_dev_6.jsonl")):
        model = snapshot_path.parent.name
        manifest_path = snapshot_path.with_suffix(".manifest.json")
        manifest: Dict[str, Any] = {}
        if manifest_path.exists():
            try:
                manifest = read_json(manifest_path)
            except Exception:
                manifest = {}
        yield model, snapshot_path, manifest


def collect_langmem_construction_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for memory_model, snapshot_path, manifest in iter_langmem_snapshots():
        backend = manifest.get("memory_backend", "")
        for record in iter_jsonl(snapshot_path):
            example_id = str(record.get("example_id", ""))
            prev_after = 0
            for export in record.get("session_exports", []):
                if export.get("status") != "OK":
                    continue
                session_input = int(export.get("session_input_tokens") or 0)
                after = int(export.get("stored_memory_tokens_after_session") or 0)
                delta = after - prev_after
                prev_after = after
                output_lower = max(delta, 0)
                rows.append(
                    {
                        "method": "langmem",
                        "memory_model": memory_model,
                        "example_id": example_id,
                        "session_index": int(export.get("session_index") or 0),
                        "attempts": 1,
                        "construction_input_tokens_est": "",
                        "construction_output_tokens_est": "",
                        "construction_total_tokens_est": "",
                        "construction_input_tokens_lower_bound": session_input,
                        "construction_output_tokens_lower_bound": output_lower,
                        "construction_total_tokens_lower_bound": session_input
                        + output_lower,
                        "stored_memory_tokens_after_session": after,
                        "implicit_pref_only_tokens_after_session": "",
                        "stored_memory_delta_tokens": delta,
                        "status": "OK",
                        "token_source": f"langmem_{backend or 'unknown'}_internal_usage_unlogged_lower_bound",
                    }
                )
    return rows


def collect_construction_rows() -> List[Dict[str, Any]]:
    rows = []
    rows.extend(collect_ours_construction_rows())
    rows.extend(collect_mem0_construction_rows())
    rows.extend(collect_langmem_construction_rows())
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def numeric(row: Dict[str, Any], key: str) -> Optional[float]:
    value = row.get(key)
    if value in (None, ""):
        return None
    try:
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    except Exception:
        return None


def summarize(rows: Sequence[Dict[str, Any]], group_keys: Sequence[str], value_keys: Sequence[str]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(key, "") for key in group_keys)].append(row)
    out: List[Dict[str, Any]] = []
    for key, bucket in sorted(buckets.items()):
        summary = {group_key: key[idx] for idx, group_key in enumerate(group_keys)}
        summary["n"] = len(bucket)
        for value_key in value_keys:
            vals = [numeric(row, value_key) for row in bucket]
            vals = [v for v in vals if v is not None]
            if vals:
                summary[f"avg_{value_key}"] = mean(vals)
                summary[f"median_{value_key}"] = median(vals)
            else:
                summary[f"avg_{value_key}"] = ""
                summary[f"median_{value_key}"] = ""
        out.append(summary)
    return out


def fmt(value: Any, digits: int = 1) -> str:
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def build_markdown(
    construction_rows: Sequence[Dict[str, Any]],
    inference_rows: Sequence[Dict[str, Any]],
    summary_rows: Sequence[Dict[str, Any]],
) -> str:
    construction_method = summarize(
        construction_rows,
        ["method"],
        [
            "construction_total_tokens_est",
            "construction_total_tokens_lower_bound",
            "construction_input_tokens_lower_bound",
            "construction_output_tokens_lower_bound",
            "stored_memory_tokens_after_session",
            "implicit_pref_only_tokens_after_session",
        ],
    )
    inference_method_context = summarize(
        inference_rows,
        ["method", "turn_type", "context_type"],
        [
            "model_input_tokens_provider",
            "model_input_tokens_est",
            "retrieved_memory_tokens",
            "memory_payload_tokens",
        ],
    )
    lines = [
        "# Rebuttal Token Cost Accounting",
        "",
        "All token estimates use `cl100k_base`. Provider usage is used when present; otherwise the stored prompt text is counted offline.",
        "",
        "Important caveat: Mem0 and LangMem construction logs do not expose their internal LLM/embedding token usage. Their construction values below are lower bounds from session text sent into the memory writer plus positive stored-memory growth. PREFINE construction is reconstructed from the paper code path: generator prompts are reconstructed, verifier prompts are logged.",
        "",
        "## Construction Cost Per Session Update",
        "",
        "| method | n updates | avg est total | avg lower-bound total | avg lower-bound input | avg lower-bound output | avg stored/injected memory after | avg compact pref only |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in construction_method:
        lines.append(
            "| {method} | {n} | {est} | {lb} | {inp} | {out} | {after} | {compact} |".format(
                method=row["method"],
                n=row["n"],
                est=fmt(row.get("avg_construction_total_tokens_est")),
                lb=fmt(row.get("avg_construction_total_tokens_lower_bound")),
                inp=fmt(row.get("avg_construction_input_tokens_lower_bound")),
                out=fmt(row.get("avg_construction_output_tokens_lower_bound")),
                after=fmt(row.get("avg_stored_memory_tokens_after_session")),
                compact=fmt(row.get("avg_implicit_pref_only_tokens_after_session")),
            )
        )
    lines.extend(
        [
            "",
            "## Inference Cost Per Test Query",
            "",
            "| method | turn | context | n queries | avg provider input | avg estimated input | avg retrieved memory/context | avg memory payload |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in inference_method_context:
        lines.append(
            "| {method} | {turn} | {context} | {n} | {provider} | {est} | {retrieved} | {payload} |".format(
                method=row["method"],
                turn=row["turn_type"],
                context=row["context_type"],
                n=row["n"],
                provider=fmt(row.get("avg_model_input_tokens_provider")),
                est=fmt(row.get("avg_model_input_tokens_est")),
                retrieved=fmt(row.get("avg_retrieved_memory_tokens")),
                payload=fmt(row.get("avg_memory_payload_tokens")),
            )
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Construction detail: `{OUT_CONSTRUCTION}`",
            f"- Inference detail: `{OUT_INFERENCE}`",
            f"- Summary CSV: `{OUT_SUMMARY}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    construction_rows = collect_construction_rows()
    inference_rows = collect_inference_rows()

    construction_fields = [
        "method",
        "memory_model",
        "example_id",
        "session_index",
        "attempts",
        "construction_input_tokens_est",
        "construction_output_tokens_est",
        "construction_total_tokens_est",
        "construction_input_tokens_lower_bound",
        "construction_output_tokens_lower_bound",
        "construction_total_tokens_lower_bound",
        "stored_memory_tokens_after_session",
        "implicit_pref_only_tokens_after_session",
        "stored_memory_delta_tokens",
        "status",
        "token_source",
    ]
    inference_fields = [
        "method",
        "source",
        "turn_type",
        "context_type",
        "context_label",
        "pref_type",
        "memory_model",
        "action_model",
        "example_id",
        "example_id_sub",
        "model_input_tokens_provider",
        "model_output_tokens_provider",
        "model_total_tokens_provider",
        "model_input_tokens_est",
        "model_output_tokens_est",
        "model_total_tokens_est",
        "retrieved_memory_tokens",
        "memory_payload_tokens",
        "token_source",
    ]
    write_csv(OUT_CONSTRUCTION, construction_rows, construction_fields)
    write_csv(OUT_INFERENCE, inference_rows, inference_fields)

    summary_rows = []
    summary_rows.extend(
        {
            "summary_type": "construction_by_method",
            **row,
        }
        for row in summarize(
            construction_rows,
            ["method"],
            [
                "construction_total_tokens_est",
                "construction_total_tokens_lower_bound",
                "stored_memory_tokens_after_session",
                "implicit_pref_only_tokens_after_session",
            ],
        )
    )
    summary_rows.extend(
        {
            "summary_type": "inference_by_method_turn_context",
            **row,
        }
        for row in summarize(
            inference_rows,
            ["method", "turn_type", "context_type"],
            [
                "model_input_tokens_provider",
                "model_input_tokens_est",
                "retrieved_memory_tokens",
                "memory_payload_tokens",
            ],
        )
    )
    summary_fields = sorted({key for row in summary_rows for key in row.keys()})
    write_csv(OUT_SUMMARY, summary_rows, summary_fields)
    OUT_MD.write_text(
        build_markdown(construction_rows, inference_rows, summary_rows),
        encoding="utf-8",
    )

    print(f"Wrote {OUT_CONSTRUCTION} ({len(construction_rows)} rows)")
    print(f"Wrote {OUT_INFERENCE} ({len(inference_rows)} rows)")
    print(f"Wrote {OUT_SUMMARY} ({len(summary_rows)} rows)")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
