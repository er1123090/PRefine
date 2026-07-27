"""Inspect or search the persistent local Mem0 store."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.mem0_local.build_memory import DEFAULT_HISTORY, DEFAULT_STORE
from methods.mem0_local.runtime import (
    DEFAULT_UPSTREAM_PATH,
    build_mem0_config,
    create_memory,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_id", required=True)
    parser.add_argument("--query", default=None)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--vector_store_path", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--history_db_path", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument(
        "--upstream_repo_path",
        type=Path,
        default=DEFAULT_UPSTREAM_PATH,
    )
    parser.add_argument("--base_url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="gpt-oss-20b")
    parser.add_argument(
        "--embedding_provider",
        choices=["fastembed", "huggingface", "openai"],
        default="fastembed",
    )
    parser.add_argument(
        "--embedding_model",
        default="BAAI/bge-small-en-v1.5",
    )
    parser.add_argument("--embedding_dims", type=int, default=384)
    parser.add_argument("--embedding_api_key", default=None)
    parser.add_argument("--embedding_base_url", default=None)
    parser.add_argument("--collection_name", default="experiment8_mem0_local_0725")
    args = parser.parse_args()

    config = build_mem0_config(
        model=args.model,
        base_url=args.base_url,
        vector_store_path=args.vector_store_path,
        history_db_path=args.history_db_path,
        collection_name=args.collection_name,
        max_tokens=2048,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        embedding_dims=args.embedding_dims,
        embedding_api_key=args.embedding_api_key,
        embedding_base_url=args.embedding_base_url,
    )
    memory = create_memory(
        config,
        repo_path=args.upstream_repo_path,
        reasoning_effort="low",
    )
    if args.query:
        result = memory.search(
            args.query,
            filters={"user_id": args.user_id},
            top_k=args.top_k,
        )
    else:
        result = memory.get_all(
            filters={"user_id": args.user_id},
            top_k=args.top_k,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
