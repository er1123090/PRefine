"""Direct inference adapter for an Experiment8 A-MEM artifact.

The requested experiment uses ``inference_batch.py``.  This direct adapter is
kept so A-MEM also participates in Experiment8's common ``run_inference.py``
interface alongside RAG, Mem0, and LangMem.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.amem.common import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    format_retrieved_notes,
    load_memory_artifact,
    make_embedder,
    public_note,
    retrieve_notes,
)
from src.exp4_prompts import (  # noqa: E402
    IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE,
    IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE_MULTITURN,
)
from src.exp4_runtime.inference import prepare_items, run_inference  # noqa: E402
from src.provider_config import resolve_openai_compatible_endpoint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turn", choices=["single", "multi"], required=True)
    parser.add_argument("--memory_path", required=True)
    parser.add_argument(
        "--input_path", default=str(ROOT / "data/MPT_v2_0725.json")
    )
    parser.add_argument("--query_path", required=True)
    parser.add_argument(
        "--pref_list_path", default=str(ROOT / "config/pref_list.json")
    )
    parser.add_argument(
        "--pref_group_path", default=str(ROOT / "config/pref_group.json")
    )
    parser.add_argument("--tools_schema_path", required=True)
    parser.add_argument(
        "--pref_type", choices=["easy", "medium", "hard"], required=True
    )
    parser.add_argument("--context_type", default="memory_api")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--log_path", required=True)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--reasoning_effort", default=None)
    parser.add_argument("--provider", choices=["auto", "openrouter"], default="auto")
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--embedding_model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--memory_top_k", type=int, default=10)
    parser.add_argument("--linked_neighbor_limit", type=int, default=10)
    parser.add_argument("--max_queries", type=int, default=None)
    args = parser.parse_args()

    base_url, api_key = resolve_openai_compatible_endpoint(
        provider=args.provider,
        base_url=args.base_url,
        api_key=args.api_key,
    )
    memory = load_memory_artifact(args.memory_path)
    embedder = make_embedder(args.embedding_model)

    async def retrieve(
        example: dict[str, Any], utterance: str
    ) -> tuple[list[dict[str, Any]], str]:
        example_id = str(example.get("example_id", "unknown_user"))
        notes = (memory.get(example_id) or {}).get("notes") or []
        selected = await asyncio.to_thread(
            retrieve_notes,
            notes,
            utterance,
            embedder=embedder,
            top_k=args.memory_top_k,
            linked_neighbor_limit=args.linked_neighbor_limit,
        )
        return (
            [public_note(note) for note in selected],
            format_retrieved_notes(selected),
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
    template = (
        IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE
        if args.turn == "single"
        else IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE_MULTITURN
    )
    asyncio.run(
        run_inference(
            prepared_items=items,
            output_path=args.output_path,
            log_path=args.log_path,
            prompt_template=template,
            context_type=args.context_type,
            tools_schema_path=args.tools_schema_path,
            model_name=args.model_name,
            concurrency=args.concurrency,
            reasoning_effort=args.reasoning_effort,
            method_name="amem",
            retriever=retrieve,
            base_url=base_url,
            api_key=api_key,
        )
    )


if __name__ == "__main__":
    main()
