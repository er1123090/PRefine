"""Experiment4 vanilla inference over the MPT_v2 inputs."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.exp4_prompts import IMPLICIT_ZS_PROMPT_TEMPLATE
from src.exp4_runtime.inference import prepare_items, run_inference
from src.provider_config import resolve_openai_compatible_endpoint


def resolve_endpoint(
    provider: str,
    base_url: str | None,
    api_key: str | None,
) -> tuple[str | None, str | None]:
    return resolve_openai_compatible_endpoint(provider, base_url, api_key)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turn", choices=["single", "multi"], required=True)
    parser.add_argument("--input_path", default=str(ROOT / "data/MPT_v2_mix600.json"))
    parser.add_argument("--query_path", required=True)
    parser.add_argument("--pref_list_path", default=str(ROOT / "config/pref_list.json"))
    parser.add_argument("--pref_group_path", default=str(ROOT / "config/pref_group.json"))
    parser.add_argument("--tools_schema_path", required=True)
    parser.add_argument("--pref_type", choices=["easy", "medium", "hard"], required=True)
    parser.add_argument("--context_type", default="diag-apilist")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--log_path", required=True)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--reasoning_effort", default=None)
    parser.add_argument(
        "--provider",
        choices=["auto", "openrouter"],
        default="auto",
        help="Use 'openrouter' to select the OpenRouter endpoint and key automatically.",
    )
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--request_timeout_seconds", type=float, default=None)
    parser.add_argument("--client_max_retries", type=int, default=None)
    parser.add_argument("--max_queries", type=int, default=None)
    args = parser.parse_args()

    base_url, api_key = resolve_endpoint(
        provider=args.provider,
        base_url=args.base_url,
        api_key=args.api_key,
    )
    items = prepare_items(
        turn=args.turn,
        input_path=args.input_path,
        query_path=args.query_path,
        pref_list_path=args.pref_list_path,
        pref_group_path=args.pref_group_path,
        pref_type=args.pref_type,
        max_queries=args.max_queries,
    )
    asyncio.run(
        run_inference(
            prepared_items=items,
            output_path=args.output_path,
            log_path=args.log_path,
            prompt_template=IMPLICIT_ZS_PROMPT_TEMPLATE,
            context_type=args.context_type,
            tools_schema_path=args.tools_schema_path,
            model_name=args.model_name,
            concurrency=args.concurrency,
            reasoning_effort=args.reasoning_effort,
            method_name="vanilla_llm",
            base_url=base_url,
            api_key=api_key,
            request_timeout_seconds=args.request_timeout_seconds,
            client_max_retries=args.client_max_retries,
        )
    )


if __name__ == "__main__":
    main()
