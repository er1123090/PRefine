"""Experiment4 Chroma RAG inference over MPT_v2."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.exp4_prompts import (
    IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE,
    IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE_MULTITURN,
)
from src.exp4_runtime.inference import prepare_items, run_inference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turn", choices=["single", "multi"], required=True)
    parser.add_argument("--input_path", default=str(ROOT / "data/MPT_v2_mix600.json"))
    parser.add_argument("--query_path", required=True)
    parser.add_argument("--pref_list_path", default=str(ROOT / "config/pref_list.json"))
    parser.add_argument("--pref_group_path", default=str(ROOT / "config/pref_group.json"))
    parser.add_argument("--tools_schema_path", required=True)
    parser.add_argument("--db_path", default=str(ROOT / "outputs/rag/chroma"))
    parser.add_argument("--collection_name", default="user_memories")
    parser.add_argument("--rag_top_k", type=int, default=5)
    parser.add_argument("--pref_type", choices=["easy", "medium", "hard"], required=True)
    parser.add_argument("--context_type", default="memory_api")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--log_path", required=True)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--reasoning_effort", default=None)
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--embedding_base_url", default=None)
    parser.add_argument("--embedding_api_key", default=None)
    parser.add_argument("--max_queries", type=int, default=None)
    args = parser.parse_args()

    try:
        import chromadb
        from chromadb.utils import embedding_functions
    except ImportError as exc:
        raise RuntimeError("chromadb is required for RAG inference") from exc

    embedding_api_key = (
        args.embedding_api_key
        or args.api_key
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
    )
    if args.embedding_base_url and not embedding_api_key:
        raise RuntimeError("OPENAI_API_KEY is required when --embedding_base_url is set.")
    embedding = embedding_functions.OpenAIEmbeddingFunction(
        api_key=embedding_api_key,
        api_base=args.embedding_base_url,
        model_name="text-embedding-3-small",
    )
    client = chromadb.PersistentClient(path=args.db_path)
    collection = client.get_collection(
        name=args.collection_name,
        embedding_function=embedding,
    )

    async def retrieve(
        example: Dict[str, Any], utterance: str
    ) -> Tuple[List[Dict[str, Any]], str]:
        response = await asyncio.to_thread(
            collection.query,
            query_texts=[utterance],
            n_results=args.rag_top_k,
            where={"user_id": str(example.get("example_id", "unknown_user"))},
        )
        documents = (response.get("documents") or [[]])[0]
        metadatas = (response.get("metadatas") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]
        memories: List[Dict[str, Any]] = []
        for index, document in enumerate(documents):
            memories.append(
                {
                    "memory": document,
                    "metadata": metadatas[index] if index < len(metadatas) else {},
                    "distance": distances[index] if index < len(distances) else None,
                }
            )
        rendered = "\n".join(
            f"{index}. {item['memory']}" for index, item in enumerate(memories, 1)
        )
        return memories, rendered or "No relevant memories retrieved."

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
            method_name="rag",
            retriever=retrieve,
            base_url=args.base_url,
            api_key=args.api_key,
        )
    )


if __name__ == "__main__":
    main()
