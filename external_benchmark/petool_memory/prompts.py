"""PEToolBench-specific prompts for petool_memory generation and inference."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from schema import PETOOL_MEMORY_JSON_SCHEMA, render_memory_for_prompt


PETOOL_MEMORY_SYSTEM_PROMPT = """
You are PEToolMemory, a memory extractor for PEToolBench personalized tool selection.

PEToolBench tool names have the form <Category>.<Provider>.<Operation>. The memory must
capture provider/namespace preferences, operation-family preferences, negative rating
evidence, and chronological preference shifts. It must not merely summarize the dialogue.

Important constraints:
- Build memory only from the supplied historical interactions.
- Do not use the current test query, candidate tools, or ground-truth answer.
- Preserve negative evidence from rating=0.
- For chronological histories, later evidence is stronger than earlier evidence within
  the same operation family.
- Return one JSON object conforming to the schema.
""".strip()


def build_memory_extraction_prompt(
    record: Dict[str, Any],
    deterministic_draft: Optional[Dict[str, Any]] = None,
) -> str:
    payload = {
        "example_id": record.get("example_id"),
        "history_type": record.get("history_type"),
        "history_label": record.get("history_label"),
        "history": record.get("history", []),
    }
    draft_block = ""
    if deterministic_draft:
        draft_block = (
            "\n\nDeterministic draft you may revise if the evidence supports it:\n"
            f"{json.dumps(deterministic_draft, ensure_ascii=False, indent=2)}"
        )
    return (
        "Create PEToolBench long-term tool preference memory for the following history.\n\n"
        "Output schema:\n"
        f"{json.dumps(PETOOL_MEMORY_JSON_SCHEMA, ensure_ascii=False, indent=2)}\n\n"
        "Historical evidence only:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
        f"{draft_block}\n\n"
        "Return exactly one JSON object and no extra text."
    )


def build_inference_prompt(record: Dict[str, Any], memory: Dict[str, Any]) -> str:
    return (
        "You are solving PEToolBench with PEToolMemory.\n"
        "Choose one candidate tool that satisfies the current user query and aligns with "
        "the stored provider/namespace preferences.\n\n"
        "PEToolMemory:\n"
        f"{render_memory_for_prompt(memory)}\n\n"
        "Candidate tools:\n"
        f"{json.dumps(record.get('candidate_tools', []), ensure_ascii=False, indent=2)}\n\n"
        "Current user query:\n"
        f"{record.get('query', '')}\n\n"
        "Decision policy:\n"
        "1. First match the current query semantics and required parameters.\n"
        "2. If multiple candidates perform the same operation, prefer the namespace in "
        "operation-family memory.\n"
        "3. Respect rating-0 avoid evidence unless no valid equivalent candidate exists.\n"
        "4. For chronological memory, use the later namespace within the same operation family.\n"
        "5. Output only valid JSON using exactly one candidate tool_name.\n\n"
        "Return exactly one JSON object and no extra text:\n"
        '{"tool_name": "<one candidate tool_name>", "parameters": {...}}\n'
    )

