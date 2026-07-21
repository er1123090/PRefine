"""Build PEToolBench-specific petool_memory records."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from compat import first_json_object, iter_jsonl, tool_call_to_text, write_jsonl
from prompts import PETOOL_MEMORY_SYSTEM_PROMPT, build_memory_extraction_prompt
from schema import (
    build_deterministic_petool_memory,
    render_memory_for_prompt,
    validate_petool_memory,
)


async def call_live_model(prompt: str, model_name: str) -> Dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from src.llm_client import call_llm_api_async, make_anthropic_client, make_openai_client

    openai_client = make_openai_client()
    anthropic_client = make_anthropic_client() if "claude" in model_name.lower() else None
    result = await call_llm_api_async(
        prompt=f"{PETOOL_MEMORY_SYSTEM_PROMPT}\n\n{prompt}",
        model_name=model_name,
        openai_client=openai_client,
        anthropic_client=anthropic_client,
        tools_schema=None,
        temperature=0.0,
    )
    if openai_client:
        await openai_client.close()
    return result


def accumulated_api_calls(record: Dict[str, Any]) -> List[str]:
    calls = []
    for index, turn in enumerate(record.get("history", []) or [], start=1):
        calls.append(f"[Session {index}] {tool_call_to_text(turn.get('tool_call', {}))}")
    return calls


async def build_memory_record(record: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    deterministic_memory = build_deterministic_petool_memory(record)
    prompt = ""
    raw_response = ""
    error = None
    token_counts: Dict[str, Any] = {}
    memory = deterministic_memory
    builder = "deterministic"

    if args.mode in {"llm", "llm_with_draft"}:
        draft = deterministic_memory if args.mode == "llm_with_draft" else None
        prompt = build_memory_extraction_prompt(record, draft)
        llm_result = await call_live_model(prompt, args.model_name)
        raw_response = llm_result.get("output", "")
        token_counts = llm_result.get("token_counts", {})
        error = llm_result.get("error")
        parsed = first_json_object(raw_response) or {}
        validation_errors = validate_petool_memory(parsed)
        if parsed and not validation_errors:
            memory = parsed
            builder = args.mode
        elif args.fallback_deterministic:
            builder = f"{args.mode}_fallback_deterministic"
            error = error or f"invalid_petool_memory: {validation_errors}"
        else:
            raise ValueError(
                f"LLM did not produce valid petool_memory for {record.get('example_id')}: "
                f"{validation_errors}; error={error}"
            )

    validation_errors = validate_petool_memory(memory)
    if validation_errors:
        raise ValueError(f"invalid petool_memory for {record.get('example_id')}: {validation_errors}")

    return {
        "example_id": record["example_id"],
        "source_index": record.get("source_index"),
        "history_type": record.get("history_type"),
        "memory_mode": f"petool_memory_v1_{builder}",
        "petool_memory": memory,
        "petool_memory_text": render_memory_for_prompt(memory),
        "final_implicit_preference": json.dumps(memory, ensure_ascii=False, indent=2),
        "final_accumulated_api_calls": accumulated_api_calls(record),
        "total_sessions_processed": len(record.get("history", []) or []),
        "generation_prompt": prompt if args.keep_prompt else "",
        "raw_memory_response": raw_response,
        "error": error,
        "token_counts": token_counts,
    }


async def run(args: argparse.Namespace) -> None:
    rows = []
    for record in iter_jsonl(args.input):
        rows.append(await build_memory_record(record, args))
    write_jsonl(args.output, rows)
    print(f"petool_memory_records={len(rows)} output={args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PEToolBench-specific petool_memory JSONL.")
    parser.add_argument("--input", required=True, help="Normalized PEToolBench JSONL.")
    parser.add_argument("--output", required=True, help="petool_memory JSONL output.")
    parser.add_argument(
        "--mode",
        choices=["deterministic", "llm", "llm_with_draft"],
        default="deterministic",
        help="Memory generation mode. deterministic needs no API key.",
    )
    parser.add_argument("--model_name", default=os.environ.get("PETOOL_MEMORY_MODEL", "gpt-4o-mini"))
    parser.add_argument(
        "--fallback_deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fallback to deterministic memory if live LLM memory generation fails.",
    )
    parser.add_argument("--keep_prompt", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

