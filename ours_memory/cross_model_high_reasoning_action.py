#!/usr/bin/env python3
"""High-reasoning launcher for the single/multi ours_memory action stages."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


async def high_reasoning_call(
    prompt: str,
    model_name: str,
    openai_client: Any = None,
    tools_schema: list[dict[str, Any]] | None = None,
    reasoning_effort: str | None = None,
) -> str:
    del tools_schema
    if openai_client is None:
        return "API_KEY_MISSING_OPENAI"

    kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": int(os.environ.get("VLLM_MAX_TOKENS", "4096")),
    }
    model_lower = model_name.lower()
    if "qwen3" in model_lower:
        kwargs["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": True}
        }
    else:
        kwargs["reasoning_effort"] = "high"

    try:
        response = await openai_client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        content = (getattr(message, "content", None) or "").strip()
        reasoning = (getattr(message, "reasoning_content", None) or "").strip()
        if reasoning:
            return f"<think>\n{reasoning}\n</think>\n\n{content}"
        return content
    except Exception as exc:
        print(f"LLM API Error ({model_name}): {exc}")
        return f"API_ERROR: {exc}"


def complete_output(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, list) or not payload:
        return False
    serialized = json.dumps(payload, ensure_ascii=False)
    return "API_ERROR:" not in serialized and "API_KEY_MISSING" not in serialized


async def run(args: argparse.Namespace) -> None:
    output_path = Path(args.output_path).resolve()
    if args.resume and complete_output(output_path):
        print(f"Already complete; skipping {output_path}")
        return

    if args.turn == "single":
        import Preference_Memory_step2_ACTION_singleturn_VLLM as action
    else:
        import Preference_Memory_step2_ACTION_multiturn_VLLM as action

    action.call_llm_api_async = high_reasoning_call
    common = dict(
        input_path=args.input_path,
        memory_path=args.memory_path,
        output_path=str(output_path),
        log_path=args.log_path,
        pref_list_path=args.pref_list_path,
        pref_group_path=args.pref_group_path,
        tools_schema_path=args.tools_schema_path,
        context_type="memory_api",
        pref_type=args.pref_type,
        model_name=args.model_name,
        concurrency=args.concurrency,
        reasoning_effort="high",
        base_url=args.api_base,
        api_key="EMPTY",
        max_queries=None,
    )
    if args.turn == "single":
        common["query_map_path"] = args.query_path
        common["prompt_template"] = action.IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE
    else:
        common["multiturn_path"] = args.query_path
        common["prompt_template"] = action.IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE_MULTITURN

    await action.process_with_llm_async(**common)
    if not complete_output(output_path):
        raise RuntimeError(f"Output failed completeness check: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turn", choices=["single", "multi"], required=True)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--memory-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--query-path", required=True)
    parser.add_argument("--pref-list-path", required=True)
    parser.add_argument("--pref-group-path", required=True)
    parser.add_argument("--tools-schema-path", required=True)
    parser.add_argument("--pref-type", choices=["easy", "medium", "hard"], required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    return parser


if __name__ == "__main__":
    asyncio.run(run(build_parser().parse_args()))
