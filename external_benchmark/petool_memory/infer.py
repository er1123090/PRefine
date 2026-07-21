"""Run PEToolBench inference with petool_memory records."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from compat import first_json_object, iter_jsonl, write_json
from prompts import build_inference_prompt
from schema import validate_petool_memory
from tool_utils import default_parameters_for_tool, rank_candidate_tools


def load_memory(path: str) -> Dict[str, Dict[str, Any]]:
    return {row["example_id"]: row for row in iter_jsonl(path)}


def extract_petool_memory(memory_record: Dict[str, Any]) -> Dict[str, Any]:
    memory = memory_record.get("petool_memory")
    if isinstance(memory, dict):
        return memory
    raw = memory_record.get("final_implicit_preference", "")
    parsed = first_json_object(raw) or {}
    errors = validate_petool_memory(parsed)
    if errors:
        return {}
    return parsed


def rule_response(record: Dict[str, Any], memory: Dict[str, Any]) -> Dict[str, Any]:
    ranked = rank_candidate_tools(record, memory)
    if not ranked:
        return {"tool_name": "", "parameters": {}}
    _, _, tool, _ = ranked[0]
    return {
        "tool_name": tool.get("tool_name", ""),
        "parameters": default_parameters_for_tool(tool),
    }


async def call_live_model(prompt: str, model_name: str) -> Dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from src.llm_client import call_llm_api_async, make_anthropic_client, make_openai_client

    openai_client = make_openai_client()
    anthropic_client = make_anthropic_client() if "claude" in model_name.lower() else None
    result = await call_llm_api_async(
        prompt=prompt,
        model_name=model_name,
        openai_client=openai_client,
        anthropic_client=anthropic_client,
        tools_schema=None,
        temperature=0.0,
    )
    if openai_client:
        await openai_client.close()
    return result


async def run(args: argparse.Namespace) -> None:
    records = list(iter_jsonl(args.input))
    memory_by_id = load_memory(args.memory)
    predictions: List[Dict[str, Any]] = []

    for record in records:
        memory_record = memory_by_id.get(record["example_id"], {})
        petool_memory = extract_petool_memory(memory_record)
        prompt = build_inference_prompt(record, petool_memory)

        if args.mock:
            parsed = rule_response(record, petool_memory)
            response = json.dumps(parsed, ensure_ascii=False)
            error = None
            model_name = "petool-memory-rule"
            token_counts = {}
        else:
            llm_result = await call_live_model(prompt, args.model_name)
            response = llm_result.get("output", "")
            parsed = first_json_object(response) or {}
            error = llm_result.get("error")
            model_name = args.model_name
            token_counts = llm_result.get("token_counts", {})

        predictions.append(
            {
                "example_id": record["example_id"],
                "source_index": record.get("source_index"),
                "history_type": record.get("history_type"),
                "query": record.get("query"),
                "api_call_ground_truth": record.get("api_call_ground_truth"),
                "memory_mode": memory_record.get("memory_mode"),
                "model_name": model_name,
                "prompt": prompt if args.keep_prompt else "",
                "response": response,
                "parsed_response": parsed,
                "error": error,
                "token_counts": token_counts,
            }
        )

    write_json(args.output, predictions)
    print(f"predictions={len(predictions)} output={args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PEToolBench inference with petool_memory.")
    parser.add_argument("--input", required=True, help="Normalized PEToolBench JSONL.")
    parser.add_argument("--memory", required=True, help="petool_memory JSONL.")
    parser.add_argument("--output", required=True, help="Predictions JSON.")
    parser.add_argument("--mock", action="store_true", help="Use deterministic PEToolMemory rule scorer.")
    parser.add_argument("--model_name", default=os.environ.get("PETOOL_MEMORY_MODEL", "gpt-4o-mini"))
    parser.add_argument("--keep_prompt", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

