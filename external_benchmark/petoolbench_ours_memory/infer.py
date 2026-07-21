"""Run PEToolBench inference using ours_memory-style memory records."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from common import first_json_object, iter_jsonl, tool_call_to_text, write_json


def load_memory(path: str) -> Dict[str, Dict[str, Any]]:
    return {row["example_id"]: row for row in iter_jsonl(path)}


def build_prompt(record: Dict[str, Any], memory: Dict[str, Any]) -> str:
    memory_text = memory.get("final_implicit_preference") or "None"
    api_history = memory.get("final_accumulated_api_calls") or []
    return (
        "You are solving PEToolBench, a personalized tool selection benchmark.\n"
        "Use the user's long-term preference memory and past tool-call history to choose "
        "one candidate tool for the current query.\n\n"
        "Long-term preference memory:\n"
        f"{memory_text}\n\n"
        "Past tool-call evidence:\n"
        f"{json.dumps(api_history[-20:], ensure_ascii=False, indent=2)}\n\n"
        "Candidate tools:\n"
        f"{json.dumps(record.get('candidate_tools', []), ensure_ascii=False, indent=2)}\n\n"
        "Current user query:\n"
        f"{record.get('query', '')}\n\n"
        "Return exactly one JSON object and no extra text:\n"
        '{"tool_name": "<one candidate tool_name>", "parameters": {...}}\n'
    )


def mock_response(record: Dict[str, Any], strategy: str) -> Dict[str, Any]:
    if strategy == "first_candidate":
        tools = record.get("candidate_tools", [])
        if not tools:
            return {"tool_name": "", "parameters": {}}
        tool = tools[0]
        params = {}
        for item in tool.get("required_parameters", []) or []:
            if isinstance(item, dict) and item.get("name"):
                params[item["name"]] = item.get("default", "")
        return {"tool_name": tool.get("tool_name", ""), "parameters": params}
    return record.get("api_call_ground_truth", {"tool_name": "", "parameters": {}})


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
        memory = memory_by_id.get(record["example_id"], {})
        prompt = build_prompt(record, memory)
        if args.mock:
            parsed = mock_response(record, args.mock_strategy)
            response = json.dumps(parsed, ensure_ascii=False)
            error = None
            model_name = f"mock-{args.mock_strategy}"
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
                "memory_mode": memory.get("memory_mode"),
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
    parser = argparse.ArgumentParser(description="Run PEToolBench ours_memory inference.")
    parser.add_argument("--input", required=True, help="Normalized PEToolBench JSONL.")
    parser.add_argument("--memory", required=True, help="Memory JSONL.")
    parser.add_argument("--output", required=True, help="Predictions JSON.")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--mock_strategy", choices=["ground_truth", "first_candidate"], default="ground_truth")
    parser.add_argument("--model_name", default=os.environ.get("PETOOLBENCH_MODEL", "gpt-4o-mini"))
    parser.add_argument("--keep_prompt", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

