"""Thin CLI for the complete experiment4 LangMem inference runtime."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METHOD_DIR = Path(__file__).resolve().parent
for path in (ROOT, METHOD_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import prepare_multiturn_items, prepare_singleturn_items, run_inference_async
from src.exp4_prompts import (
    IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE,
    IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE_MULTITURN,
)
from src.provider_config import (
    is_openrouter_endpoint,
    resolve_embedding_endpoint,
    resolve_openai_compatible_endpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turn", choices=["single", "multi"], required=True)
    parser.add_argument("--memory_path", required=True)
    parser.add_argument("--input_path", default=str(ROOT / "data/MPT_v2_mix600.json"))
    parser.add_argument("--query_path", required=True)
    parser.add_argument("--pref_list_path", default=str(ROOT / "config/pref_list.json"))
    parser.add_argument("--pref_group_path", default=str(ROOT / "config/pref_group.json"))
    parser.add_argument("--tools_schema_path", required=True)
    parser.add_argument("--pref_type", choices=["easy", "medium", "hard"], required=True)
    parser.add_argument("--context_type", default="memory_api")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--log_path", required=True)
    parser.add_argument("--run_log_path", default=None)
    parser.add_argument("--memory_top_k", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--reasoning_effort", default=None)
    parser.add_argument("--provider", choices=["auto", "openrouter"], default="auto")
    parser.add_argument("--embedding_model", default="text-embedding-3-small")
    parser.add_argument("--embedding_base_url", default=None)
    parser.add_argument("--embedding_api_key", default=None)
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--max_queries", type=int, default=None)
    args = parser.parse_args()

    args.base_url, args.api_key = resolve_openai_compatible_endpoint(
        provider=args.provider,
        base_url=args.base_url,
        api_key=args.api_key,
    )
    embedding_base_url = args.embedding_base_url
    if embedding_base_url is None and is_openrouter_endpoint(args.base_url):
        embedding_base_url = args.base_url
    embedding_api_key = (
        args.embedding_api_key
        or (args.api_key if is_openrouter_endpoint(embedding_base_url) else None)
    )
    (
        args.embedding_model,
        args.embedding_base_url,
        args.embedding_api_key,
    ) = resolve_embedding_endpoint(
        provider=args.provider,
        embedding_model=args.embedding_model,
        base_url=embedding_base_url,
        api_key=embedding_api_key,
    )

    common_args = {
        "input_path": args.input_path,
        "pref_list_path": args.pref_list_path,
        "pref_group_path": args.pref_group_path,
        "pref_type": args.pref_type,
        "max_queries": args.max_queries,
    }
    if args.turn == "single":
        items = prepare_singleturn_items(query_path=args.query_path, **common_args)
        template = IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE
    else:
        items = prepare_multiturn_items(multiturn_path=args.query_path, **common_args)
        template = IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE_MULTITURN

    asyncio.run(
        run_inference_async(
            prepared_items=items,
            snapshot_path=args.memory_path,
            output_path=args.output_path,
            log_path=args.log_path,
            run_log_path=args.run_log_path,
            prompt_template=template,
            context_type=args.context_type,
            tools_schema_path=args.tools_schema_path,
            model_name=args.model_name,
            memory_top_k=args.memory_top_k,
            concurrency=args.concurrency,
            reasoning_effort=args.reasoning_effort,
            embedding_model=args.embedding_model,
            base_url=args.base_url,
            api_key=args.api_key,
            embedding_base_url=args.embedding_base_url,
            embedding_api_key=args.embedding_api_key,
        )
    )


if __name__ == "__main__":
    main()
