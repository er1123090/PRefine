from __future__ import annotations

import argparse
import asyncio

from common import (
    DEFAULT_INPUT_PATH,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_MULTITURN_QUERY_PATH,
    DEFAULT_PREF_GROUP_PATH,
    DEFAULT_PREF_LIST_PATH,
    DEFAULT_SCHEMA_PATH,
    default_run_log_path,
    prepare_multiturn_items,
    run_inference_async,
)
from prompt_inference import IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE_MULTITURN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_path", type=str, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--input_path", type=str, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output_path", type=str, default="output_emem_multi.json")
    parser.add_argument("--log_path", type=str, default="output_emem_multi.jsonl")
    parser.add_argument("--run_log_path", type=str, default=None)
    parser.add_argument("--multiturn_path", type=str, default=DEFAULT_MULTITURN_QUERY_PATH)
    parser.add_argument("--pref_list_path", type=str, default=DEFAULT_PREF_LIST_PATH)
    parser.add_argument("--pref_group_path", type=str, default=DEFAULT_PREF_GROUP_PATH)
    parser.add_argument("--tools_schema_path", type=str, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--pref_type", type=str, choices=["easy", "medium", "hard"], required=True)
    parser.add_argument(
        "--context_type",
        type=str,
        choices=["memory_only", "memory_diag", "memory_api"],
        required=True,
    )
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--reasoning_effort", type=str, choices=["minimal", "low", "medium", "high"], default=None)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--memory_top_k", type=int, default=5)
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--retrieval_api_key", type=str, default=None)
    parser.add_argument("--retrieval_embedding_api_key", type=str, default=None)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    run_log_path = args.run_log_path or default_run_log_path(args.output_path)
    prepared_items = prepare_multiturn_items(
        input_path=args.input_path,
        multiturn_path=args.multiturn_path,
        pref_list_path=args.pref_list_path,
        pref_group_path=args.pref_group_path,
        pref_type=args.pref_type,
    )
    await run_inference_async(
        prepared_items=prepared_items,
        manifest_path=args.manifest_path,
        output_path=args.output_path,
        log_path=args.log_path,
        run_log_path=run_log_path,
        prompt_template=IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE_MULTITURN,
        context_type=args.context_type,
        tools_schema_path=args.tools_schema_path,
        model_name=args.model_name,
        memory_top_k=args.memory_top_k,
        concurrency=args.concurrency,
        reasoning_effort=args.reasoning_effort,
        base_url=args.base_url,
        api_key=args.api_key,
        retrieval_api_key=args.retrieval_api_key,
        retrieval_embedding_api_key=args.retrieval_embedding_api_key,
    )


if __name__ == "__main__":
    asyncio.run(main())
