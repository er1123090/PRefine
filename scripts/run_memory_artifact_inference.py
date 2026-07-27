#!/usr/bin/env python3
"""Prepare and run 0725 memory-artifact inference on Batch or local vLLM.

The prepared manifest is the source of truth for both providers so A-MEM,
LangMem, and RAG use byte-identical prompts across GPT-5-mini Batch and
GPT-OSS-20B vLLM inference.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from openai import AsyncOpenAI, OpenAI
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
LANGMEM_DIR = ROOT / "methods" / "langmem"
for path in (ROOT, LANGMEM_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from methods.amem.common import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl,
    index_unique_rows,
    now_iso,
    read_json,
    read_jsonl,
    sha256_file,
)
from scripts.run_vanilla_batch_sample import (  # noqa: E402
    build_population,
    evaluation_report,
    openai_result_payload,
)
from src.exp4_prompts import (  # noqa: E402
    IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE,
    IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE_MULTITURN,
)
from src.exp4_runtime.inference import common as runtime_common  # noqa: E402
import common as langmem_common  # type: ignore  # noqa: E402


DEFAULT_INPUT = ROOT / "data" / "MPT_v2_0725.json"
DEFAULT_LANGMEM = (
    ROOT
    / "outputs"
    / "langmem"
    / "MPT_v2_0725_gpt-5-mini_batch"
    / "memory.jsonl"
)
DEFAULT_RAG = ROOT / "outputs" / "rag" / "MPT_v2_0725_openai_chroma"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def output_paths(directory: Path) -> dict[str, Path]:
    return {
        "directory": directory,
        "manifest": directory / "manifest.jsonl",
        "requests": directory / "requests.jsonl",
        "summary": directory / "prepare_summary.json",
        "state": directory / "batch_state.json",
        "raw": directory / "raw_results.jsonl",
        "raw_errors": directory / "raw_errors.jsonl",
        "checkpoint": directory / "inference.jsonl",
        "predictions": directory / "predictions.json",
        "evaluation": directory / "evaluation.json",
        "run_summary": directory / "run_summary.json",
        "embedding_cache": directory / "retrieval_embeddings.npz",
        "embedding_meta": directory / "retrieval_embeddings.meta.json",
    }


def load_key_from_env_file(path: str | None, variable: str) -> str | None:
    if not path:
        return None
    env_path = Path(path).expanduser().resolve()
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() != variable:
            continue
        value = value.strip()
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {"'", '"'}
        ):
            value = value[1:-1]
        return value or None
    return None


def resolve_api_key(
    explicit: str | None,
    env_file: str | None,
    variable: str = "OPENAI_API_KEY",
) -> str:
    value = (
        explicit
        or os.environ.get(variable)
        or load_key_from_env_file(env_file, variable)
    )
    if not value:
        raise RuntimeError(
            f"{variable} is required. Set it in the environment or pass "
            "--api-key-env-file."
        )
    return value


def compact_population_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in row.items()
        if key not in {"original_ex", "prompt"}
    }


def memory_prompt(
    row: Mapping[str, Any],
    retrieved_text: str,
    tools_by_schema: Mapping[str, list[dict[str, Any]]],
) -> str:
    template = (
        IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE
        if row["turn"] == "single"
        else IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE_MULTITURN
    )
    return runtime_common.build_memory_prompt(
        example=row["original_ex"],
        retrieved_memories_text=retrieved_text,
        current_user_utterance=str(row["utterance"]),
        template=template,
        context_type="memory_api",
        tools_schema=tools_by_schema[str(row["schema"])],
    )


def batch_request(
    row: Mapping[str, Any],
    *,
    model: str,
    reasoning_effort: str,
    max_completion_tokens: int,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": row["prompt"]}],
        "max_completion_tokens": max_completion_tokens,
    }
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    return {
        "custom_id": row["sample_id"],
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }


def embedding_text(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def embed_texts(
    texts: Sequence[str],
    *,
    client: OpenAI,
    model: str,
    batch_size: int,
    retry_count: int,
) -> tuple[np.ndarray, dict[str, int]]:
    if not texts:
        return np.empty((0, 0), dtype=np.float32), {
            "input_tokens": 0,
            "total_tokens": 0,
        }
    vectors: list[list[float]] = []
    usage = {"input_tokens": 0, "total_tokens": 0}
    progress = tqdm(total=len(texts), desc=f"Embed {model}", unit="text")
    try:
        for start in range(0, len(texts), batch_size):
            chunk = list(texts[start : start + batch_size])
            last_error: Exception | None = None
            for attempt in range(retry_count + 1):
                try:
                    response = client.embeddings.create(
                        model=model,
                        input=chunk,
                        encoding_format="float",
                    )
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt >= retry_count:
                        raise
                    time.sleep(min(30.0, 2.0**attempt))
            else:
                raise RuntimeError(last_error)
            ordered = sorted(response.data, key=lambda item: int(item.index))
            if len(ordered) != len(chunk):
                raise RuntimeError(
                    f"Embedding count mismatch: expected={len(chunk)} "
                    f"actual={len(ordered)}"
                )
            vectors.extend(item.embedding for item in ordered)
            response_usage = getattr(response, "usage", None)
            if response_usage is not None:
                usage["input_tokens"] += int(
                    getattr(response_usage, "prompt_tokens", 0) or 0
                )
                usage["total_tokens"] += int(
                    getattr(response_usage, "total_tokens", 0) or 0
                )
            progress.update(len(chunk))
    finally:
        progress.close()
    return np.asarray(vectors, dtype=np.float32), usage


def normalized(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def load_or_create_embedding_cache(
    *,
    paths: Mapping[str, Path],
    labels: Sequence[str],
    texts: Sequence[str],
    query_texts: Sequence[str],
    model: str,
    client: OpenAI,
    batch_size: int,
    retry_count: int,
    source_sha256: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    combined_sha = hashlib.sha256()
    for label, text in zip(labels, texts, strict=True):
        combined_sha.update(label.encode("utf-8"))
        combined_sha.update(b"\0")
        combined_sha.update(text.encode("utf-8"))
        combined_sha.update(b"\0")
    for text in query_texts:
        combined_sha.update(b"query\0")
        combined_sha.update(text.encode("utf-8"))
        combined_sha.update(b"\0")
    cache_key = combined_sha.hexdigest()
    expected_meta = {
        "model": model,
        "source_sha256": source_sha256,
        "cache_key": cache_key,
        "document_count": len(texts),
        "query_count": len(query_texts),
    }
    if paths["embedding_cache"].exists() and paths["embedding_meta"].exists():
        recorded = read_json(paths["embedding_meta"])
        if all(recorded.get(key) == value for key, value in expected_meta.items()):
            cached = np.load(paths["embedding_cache"])
            return (
                np.asarray(cached["documents"], dtype=np.float32),
                np.asarray(cached["queries"], dtype=np.float32),
                recorded,
            )
    document_vectors, document_usage = embed_texts(
        texts,
        client=client,
        model=model,
        batch_size=batch_size,
        retry_count=retry_count,
    )
    query_vectors, query_usage = embed_texts(
        query_texts,
        client=client,
        model=model,
        batch_size=batch_size,
        retry_count=retry_count,
    )
    paths["embedding_cache"].parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        paths["embedding_cache"],
        documents=document_vectors,
        queries=query_vectors,
    )
    meta = {
        **expected_meta,
        "created_at": utc_now(),
        "dimensions": (
            int(document_vectors.shape[1]) if document_vectors.size else 0
        ),
        "usage": {
            "document": document_usage,
            "query": query_usage,
        },
    }
    atomic_write_json(paths["embedding_meta"], meta)
    return document_vectors, query_vectors, meta


def population_for(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = build_population(
        query=args.query,
        context_type="memory_api",
        input_path=args.input_path,
        exclude_easy_conflict=args.exclude_easy_conflict,
    )
    if args.max_queries is not None:
        if args.max_queries <= 0:
            raise ValueError("--max-queries must be positive")
        rows = rows[: args.max_queries]
    if args.expected_count is not None and len(rows) != args.expected_count:
        raise RuntimeError(
            f"Population mismatch: expected={args.expected_count}, actual={len(rows)}"
        )
    return rows


def prepare_langmem(
    args: argparse.Namespace,
    population: Sequence[dict[str, Any]],
    paths: Mapping[str, Path],
    client: OpenAI,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    memory_path = Path(args.memory_path).resolve()
    records = langmem_common.load_snapshot_records(str(memory_path))
    items: list[dict[str, Any]] = []
    labels: list[str] = []
    texts: list[str] = []
    user_indexes: dict[str, list[int]] = defaultdict(list)
    for record in records:
        user_id = str(record["example_id"])
        for item in record.get("memory_items", []):
            index = len(items)
            items.append(item)
            labels.append(f"{user_id}\0{item.get('key', '')}")
            texts.append(embedding_text(item.get("value", {})))
            user_indexes[user_id].append(index)
    query_texts = sorted({str(row["utterance"]) for row in population})
    document_vectors, query_vectors, embedding_meta = (
        load_or_create_embedding_cache(
            paths=paths,
            labels=labels,
            texts=texts,
            query_texts=query_texts,
            model=args.embedding_model,
            client=client,
            batch_size=args.embedding_batch_size,
            retry_count=args.retry_count,
            source_sha256=sha256_file(memory_path),
        )
    )
    document_vectors = normalized(document_vectors)
    query_vectors = normalized(query_vectors)
    query_by_text = {
        text: query_vectors[index] for index, text in enumerate(query_texts)
    }
    retrieval_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in tqdm(population, desc="Retrieve LangMem", unit="query"):
        user_id = str(row["example_id"])
        query = str(row["utterance"])
        cache_key = (user_id, query)
        if cache_key in retrieval_cache:
            continue
        indexes = user_indexes.get(user_id, [])
        if not indexes:
            retrieval_cache[cache_key] = []
            continue
        scores = document_vectors[indexes] @ query_by_text[query]
        order = np.argsort(-scores, kind="stable")[: args.top_k]
        retrieved: list[dict[str, Any]] = []
        for local_index in order:
            source_index = indexes[int(local_index)]
            item = items[source_index]
            value = item.get("value", {})
            retrieved.append(
                {
                    "namespace": item.get("namespace", []),
                    "key": item.get("key", ""),
                    "value": value,
                    "score": float(scores[int(local_index)]),
                    "display_text": langmem_common.render_memory_line(value),
                }
            )
        retrieval_cache[cache_key] = retrieved
    tools_by_schema = {
        schema: runtime_common.load_tools_from_file(
            str(ROOT / "config" / f"schema_{schema}.json")
        )
        for schema in {str(row["schema"]) for row in population}
    }
    manifest: list[dict[str, Any]] = []
    for row in population:
        retrieved = retrieval_cache[
            (str(row["example_id"]), str(row["utterance"]))
        ]
        rendered = (
            "\n".join(f"- {item['display_text']}" for item in retrieved)
            or "No relevant memories found."
        )
        manifest.append(
            {
                **compact_population_row(row),
                "sample_id": f"langmem-{int(row['population_index']):05d}",
                "method": "langmem",
                "context_type": "memory_api",
                "prompt": memory_prompt(row, rendered, tools_by_schema),
                "retrieved_memories": retrieved,
                "retrieved_memory_count": len(retrieved),
                "memory_top_k": args.top_k,
            }
        )
    return manifest, {
        "memory_path": str(memory_path),
        "memory_sha256": sha256_file(memory_path),
        "embedding_model": args.embedding_model,
        "embedding_cache": str(paths["embedding_cache"]),
        "embedding_cache_meta": embedding_meta,
        "memory_item_count": len(items),
    }


def prepare_rag(
    args: argparse.Namespace,
    population: Sequence[dict[str, Any]],
    paths: Mapping[str, Path],
    client: OpenAI,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError("chromadb is required for RAG preparation") from exc
    db_path = Path(args.db_path).resolve()
    chroma = chromadb.PersistentClient(path=str(db_path))
    collection = chroma.get_collection(args.collection_name)
    query_texts = sorted({str(row["utterance"]) for row in population})
    query_vectors, usage = embed_texts(
        query_texts,
        client=client,
        model=args.embedding_model,
        batch_size=args.embedding_batch_size,
        retry_count=args.retry_count,
    )
    query_by_text = {
        text: query_vectors[index].tolist()
        for index, text in enumerate(query_texts)
    }
    queries_by_user: dict[str, list[str]] = defaultdict(list)
    for row in population:
        user_id = str(row["example_id"])
        query = str(row["utterance"])
        if query not in queries_by_user[user_id]:
            queries_by_user[user_id].append(query)
    retrieval_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for user_id, queries in tqdm(
        sorted(queries_by_user.items()), desc="Retrieve RAG", unit="user"
    ):
        response = collection.query(
            query_embeddings=[query_by_text[query] for query in queries],
            n_results=args.top_k,
            where={"user_id": user_id},
            include=["documents", "metadatas", "distances"],
        )
        ids = response.get("ids") or [[] for _ in queries]
        documents = response.get("documents") or [[] for _ in queries]
        metadatas = response.get("metadatas") or [[] for _ in queries]
        distances = response.get("distances") or [[] for _ in queries]
        for query_index, query in enumerate(queries):
            retrieved: list[dict[str, Any]] = []
            for result_index, document in enumerate(documents[query_index]):
                retrieved.append(
                    {
                        "id": (
                            ids[query_index][result_index]
                            if result_index < len(ids[query_index])
                            else None
                        ),
                        "memory": document,
                        "metadata": (
                            metadatas[query_index][result_index]
                            if result_index < len(metadatas[query_index])
                            else {}
                        ),
                        "distance": (
                            distances[query_index][result_index]
                            if result_index < len(distances[query_index])
                            else None
                        ),
                    }
                )
            retrieval_cache[(user_id, query)] = retrieved
    tools_by_schema = {
        schema: runtime_common.load_tools_from_file(
            str(ROOT / "config" / f"schema_{schema}.json")
        )
        for schema in {str(row["schema"]) for row in population}
    }
    manifest: list[dict[str, Any]] = []
    for row in population:
        retrieved = retrieval_cache[
            (str(row["example_id"]), str(row["utterance"]))
        ]
        rendered = (
            "\n".join(
                f"{index}. {item['memory']}"
                for index, item in enumerate(retrieved, start=1)
            )
            or "No relevant memories retrieved."
        )
        manifest.append(
            {
                **compact_population_row(row),
                "sample_id": f"rag-{int(row['population_index']):05d}",
                "method": "rag",
                "context_type": "memory_api",
                "prompt": memory_prompt(row, rendered, tools_by_schema),
                "retrieved_memories": retrieved,
                "retrieved_memory_count": len(retrieved),
                "memory_top_k": args.top_k,
            }
        )
    return manifest, {
        "db_path": str(db_path),
        "collection_name": args.collection_name,
        "collection_count": collection.count(),
        "embedding_model": args.embedding_model,
        "query_embedding_usage": usage,
    }


def prepare_command(args: argparse.Namespace) -> None:
    if args.method not in {"langmem", "rag"}:
        raise ValueError(
            "A-MEM preparation is handled by methods/amem/inference_batch.py; "
            "this prepare command accepts langmem or rag."
        )
    directory = Path(args.output_dir).resolve()
    paths = output_paths(directory)
    immutable = (
        paths["state"],
        paths["raw"],
        paths["predictions"],
        paths["evaluation"],
    )
    if any(path.exists() for path in immutable):
        raise FileExistsError(
            f"Recorded inference state exists under {directory}; use a new directory."
        )
    prepared = (paths["manifest"], paths["requests"], paths["summary"])
    if any(path.exists() for path in prepared) and not args.force:
        raise FileExistsError(
            f"Prepared files exist under {directory}; pass --force only before submit."
        )
    directory.mkdir(parents=True, exist_ok=True)
    population = population_for(args)
    key = resolve_api_key(args.api_key, args.api_key_env_file)
    client = OpenAI(api_key=key, timeout=args.request_timeout_seconds)
    if args.method == "langmem":
        manifest, retrieval = prepare_langmem(args, population, paths, client)
    else:
        manifest, retrieval = prepare_rag(args, population, paths, client)
    requests = [
        batch_request(
            row,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            max_completion_tokens=args.max_completion_tokens,
        )
        for row in manifest
    ]
    atomic_write_jsonl(paths["manifest"], manifest)
    atomic_write_jsonl(paths["requests"], requests)
    summary = {
        "created_at": now_iso(),
        "method": args.method,
        "input_path": str(Path(args.input_path).resolve()),
        "input_sha256": sha256_file(Path(args.input_path).resolve()),
        "query": args.query,
        "context_type": "memory_api",
        "exclude_easy_conflict": args.exclude_easy_conflict,
        "population_count": len(population),
        "request_count": len(requests),
        "condition_counts": dict(
            sorted(Counter(str(row["condition"]) for row in manifest).items())
        ),
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "max_completion_tokens": args.max_completion_tokens,
        "top_k": args.top_k,
        "endpoint": "/v1/chat/completions",
        "manifest_sha256": sha256_file(paths["manifest"]),
        "requests_sha256": sha256_file(paths["requests"]),
        "request_bytes": paths["requests"].stat().st_size,
        "retrieval": retrieval,
    }
    atomic_write_json(paths["summary"], summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def batch_client(args: argparse.Namespace) -> OpenAI:
    return OpenAI(
        api_key=resolve_api_key(args.api_key, args.api_key_env_file),
        timeout=args.request_timeout_seconds,
    )


def submit_command(args: argparse.Namespace) -> None:
    directory = Path(args.prepared_dir).resolve()
    paths = output_paths(directory)
    if paths["state"].exists():
        state = read_json(paths["state"])
        raise RuntimeError(
            f"Batch already recorded as {state.get('batch_id')}; refusing duplicate charge."
        )
    summary = read_json(paths["summary"])
    requests_sha = sha256_file(paths["requests"])
    if summary.get("requests_sha256") != requests_sha:
        raise RuntimeError("requests.jsonl hash differs from prepare_summary.json")
    request_count = int(summary.get("request_count", 0))
    if not 0 < request_count <= 50_000:
        raise RuntimeError(
            f"Batch request count must be in [1, 50000], got {request_count}"
        )
    if paths["requests"].stat().st_size > 200_000_000:
        raise RuntimeError("Batch request file exceeds the 200 MB upload limit")
    client = batch_client(args)
    client.models.retrieve(str(summary["model"]))
    state = {
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "phase": "submission_pending",
        "input_file_id": None,
        "batch_id": None,
        "status": "submission_pending",
        "output_file_id": None,
        "error_file_id": None,
        "requests_sha256": requests_sha,
        "model": summary["model"],
        "reasoning_effort": summary["reasoning_effort"],
        "max_completion_tokens": summary["max_completion_tokens"],
        "method": summary.get("method") or args.method,
    }
    atomic_write_json(paths["state"], state)
    with paths["requests"].open("rb") as handle:
        uploaded = client.files.create(file=handle, purpose="batch")
    state.update(
        {
            "updated_at": now_iso(),
            "input_file_id": uploaded.id,
            "phase": "uploaded",
        }
    )
    atomic_write_json(paths["state"], state)
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "experiment": "experiment8",
            "dataset": "MPT_v2_0725",
            "method": str(state["method"]),
            "stage": "inference",
            "model": str(state["model"]),
            "reasoning_effort": str(state["reasoning_effort"]),
            "max_completion_tokens": str(state["max_completion_tokens"]),
        },
    )
    state.update(
        {
            "updated_at": now_iso(),
            "phase": "submitted",
            "batch_id": batch.id,
            "status": batch.status,
            "output_file_id": batch.output_file_id,
            "error_file_id": batch.error_file_id,
        }
    )
    atomic_write_json(paths["state"], state)
    print(json.dumps(state, ensure_ascii=False, indent=2))


def refresh_batch(
    args: argparse.Namespace, paths: Mapping[str, Path]
) -> dict[str, Any]:
    state = read_json(paths["state"])
    batch_id = state.get("batch_id")
    if not batch_id:
        raise RuntimeError(
            "Submission is pending without a batch_id; reconcile in the dashboard."
        )
    batch = batch_client(args).batches.retrieve(batch_id)
    state.update(
        {
            "updated_at": now_iso(),
            "status": batch.status,
            "output_file_id": batch.output_file_id,
            "error_file_id": batch.error_file_id,
            "request_counts": (
                batch.request_counts.model_dump()
                if batch.request_counts is not None
                else None
            ),
        }
    )
    atomic_write_json(paths["state"], state)
    return state


def status_command(args: argparse.Namespace) -> None:
    paths = output_paths(Path(args.prepared_dir).resolve())
    state = refresh_batch(args, paths)
    print(json.dumps(state, ensure_ascii=False, indent=2))


def population_by_index(summary: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    rows = build_population(
        query=str(summary.get("query", "hint")),
        context_type=str(summary.get("context_type", "memory_api")),
        input_path=str(summary["input_path"]),
        exclude_easy_conflict=bool(summary.get("exclude_easy_conflict", True)),
    )
    return {int(row["population_index"]): row for row in rows}


def materialize_prediction(
    manifest: Mapping[str, Any],
    record: Mapping[str, Any],
    population_index: Mapping[int, dict[str, Any]],
) -> dict[str, Any]:
    original = manifest.get("original_ex")
    if not isinstance(original, dict):
        original = population_index[int(manifest["population_index"])]["original_ex"]
    prediction = copy.deepcopy(original)
    prediction.update(record)
    return prediction


def batch_record(
    manifest: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    method: str,
    model: str,
    reasoning_effort: str,
    max_completion_tokens: int,
    batch_id: str,
) -> dict[str, Any]:
    error = payload.get("error")
    return {
        "example_id": manifest["example_id"],
        "example_id_sub": manifest["example_id_sub"],
        "sample_id": manifest["sample_id"],
        "method": method,
        "model_name": model,
        "context_type": manifest.get("context_type", "memory_api"),
        "test_utterance": manifest["utterance"],
        "reference_ground_truth": manifest["reference_ground_truth"],
        "retrieved_memories": manifest.get("retrieved_memories", []),
        "retrieved_memory_count": manifest.get("retrieved_memory_count", 0),
        "memory_top_k": manifest.get(
            "memory_top_k", manifest.get("top_k")
        ),
        "status": "ERROR" if error else "OK",
        "error": error,
        "model_input": manifest["prompt"],
        "llm_output": error or payload.get("content", ""),
        "reasoning_content": "" if error else payload.get("reasoning", ""),
        "token_counts": {} if error else payload.get("usage", {}),
        "population_index": manifest["population_index"],
        "turn": manifest["turn"],
        "query": manifest["query"],
        "schema": manifest["schema"],
        "pref_type": manifest["pref_type"],
        "condition": manifest["condition"],
        "reasoning_effort": reasoning_effort,
        "max_completion_tokens": max_completion_tokens,
        "batch_provider": "openai",
        "batch_id": batch_id,
    }


def collect_batch(args: argparse.Namespace) -> None:
    directory = Path(args.prepared_dir).resolve()
    paths = output_paths(directory)
    state = refresh_batch(args, paths)
    if state.get("status") != "completed":
        raise RuntimeError(f"Batch is not complete: {state.get('status')}")
    if not state.get("output_file_id"):
        raise RuntimeError("Completed batch has no output_file_id")
    client = batch_client(args)
    paths["raw"].write_bytes(
        client.files.content(state["output_file_id"]).content
    )
    if state.get("error_file_id"):
        paths["raw_errors"].write_bytes(
            client.files.content(state["error_file_id"]).content
        )
    manifest_rows = read_jsonl(paths["manifest"])
    raw_rows = read_jsonl(paths["raw"])
    raw_by_id = index_unique_rows(
        raw_rows, id_key="custom_id", label="memory inference Batch results"
    )
    summary = read_json(paths["summary"])
    population_index = population_by_index(summary)
    method = str(summary.get("method") or args.method)
    records: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for manifest in manifest_rows:
        raw = raw_by_id.get(str(manifest["sample_id"]))
        payload = (
            openai_result_payload(raw)
            if raw is not None
            else {
                "error": "BATCH_ERROR: missing result for custom_id",
                "content": "",
                "usage": {},
            }
        )
        record = batch_record(
            manifest,
            payload,
            method=method,
            model=str(state["model"]),
            reasoning_effort=str(state["reasoning_effort"]),
            max_completion_tokens=int(state["max_completion_tokens"]),
            batch_id=str(state["batch_id"]),
        )
        records.append(record)
        predictions.append(
            materialize_prediction(manifest, record, population_index)
        )
    records.sort(key=lambda row: int(row["population_index"]))
    predictions.sort(key=lambda row: int(row["population_index"]))
    atomic_write_jsonl(paths["checkpoint"], records)
    atomic_write_json(paths["predictions"], predictions)
    report = evaluation_report(predictions)
    report.update(
        {
            "method": method,
            "provider": "openai_batch",
            "model": state["model"],
            "reasoning_effort": state["reasoning_effort"],
            "max_completion_tokens": state["max_completion_tokens"],
            "batch_id": state["batch_id"],
            "prediction_count": len(predictions),
            "status_counts": dict(
                sorted(Counter(row["status"] for row in records).items())
            ),
            "token_usage": aggregate_usage(records),
            "generated_at": now_iso(),
        }
    )
    atomic_write_json(paths["evaluation"], report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def read_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    for row in read_jsonl(path):
        records[str(row["sample_id"])] = row
    return records


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()


def local_record_complete(row: Mapping[str, Any] | None) -> bool:
    if not row or row.get("status") != "OK":
        return False
    if row.get("finish_reason") == "length":
        return False
    return bool(str(row.get("llm_output", "")).strip())


def response_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    completion_details = getattr(usage, "completion_tokens_details", None)
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    return {
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "reasoning_tokens": int(
            getattr(completion_details, "reasoning_tokens", 0) or 0
        ),
        "cached_input_tokens": int(
            getattr(prompt_details, "cached_tokens", 0) or 0
        ),
    }


async def local_infer_one(
    manifest: Mapping[str, Any],
    *,
    args: argparse.Namespace,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    checkpoint: Path,
    lock: asyncio.Lock,
    progress: tqdm,
    attempt: int,
    max_tokens: int,
) -> dict[str, Any]:
    started = time.monotonic()
    error: str | None = None
    content = ""
    reasoning = ""
    token_counts: dict[str, Any] = {}
    finish_reason: str | None = None
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": manifest["prompt"]}],
                reasoning_effort=args.reasoning_effort,
                max_tokens=max_tokens,
            )
            message = response.choices[0].message
            content = (getattr(message, "content", None) or "").strip()
            reasoning = (
                getattr(message, "reasoning_content", None)
                or getattr(message, "reasoning", None)
                or ""
            ).strip()
            finish_reason = str(response.choices[0].finish_reason or "")
            token_counts = response_usage(response)
            if not content:
                error = "API_ERROR: model returned empty final content"
            elif finish_reason == "length":
                error = "API_ERROR: model hit max_tokens"
        except Exception as exc:
            error = f"API_ERROR: {exc}"
        record = {
            "timestamp": utc_now(),
            "sample_id": manifest["sample_id"],
            "population_index": manifest["population_index"],
            "example_id": manifest["example_id"],
            "example_id_sub": manifest["example_id_sub"],
            "method": args.method,
            "model_name": args.record_model_name or args.model,
            "request_model": args.model,
            "api_provider": "local_vllm",
            "context_type": manifest.get("context_type", "memory_api"),
            "turn": manifest["turn"],
            "query": manifest["query"],
            "schema": manifest["schema"],
            "pref_type": manifest["pref_type"],
            "condition": manifest["condition"],
            "test_utterance": manifest["utterance"],
            "reference_ground_truth": manifest["reference_ground_truth"],
            "retrieved_memories": manifest.get("retrieved_memories", []),
            "retrieved_memory_count": manifest.get("retrieved_memory_count", 0),
            "memory_top_k": manifest.get(
                "memory_top_k", manifest.get("top_k")
            ),
            "status": "ERROR" if error else "OK",
            "error": error,
            "model_input": manifest["prompt"],
            "llm_output": error or content,
            "reasoning_content": "" if error else reasoning,
            "token_counts": {} if error else token_counts,
            "finish_reason": finish_reason,
            "reasoning_effort": args.reasoning_effort,
            "max_tokens_requested": max_tokens,
            "attempt": attempt,
            "elapsed_seconds": time.monotonic() - started,
        }
        async with lock:
            append_jsonl(checkpoint, record)
        progress.update(1)
        return record


def aggregate_usage(records: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    fields = (
        "total_tokens",
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "cached_input_tokens",
        "cost_usd",
    )
    return {
        field: sum(
            float((record.get("token_counts") or {}).get(field, 0) or 0)
            for record in records
        )
        for field in fields
    }


async def local_command(args: argparse.Namespace) -> None:
    prepared = output_paths(Path(args.prepared_dir).resolve())
    output = output_paths(Path(args.output_dir).resolve())
    output["directory"].mkdir(parents=True, exist_ok=True)
    manifest_rows = read_jsonl(prepared["manifest"])
    summary = read_json(prepared["summary"])
    manifest_sha = sha256_file(prepared["manifest"])
    if (
        summary.get("manifest_sha256")
        and summary["manifest_sha256"] != manifest_sha
    ):
        raise RuntimeError(
            "manifest.jsonl hash differs from prepare_summary.json; "
            "refusing a non-comparable local run"
        )
    method = str(summary.get("method") or args.method)
    if method != args.method:
        raise RuntimeError(
            f"Prepared method mismatch: recorded={method} requested={args.method}"
        )
    if args.expected_count is not None and len(manifest_rows) != args.expected_count:
        raise RuntimeError(
            f"Manifest count mismatch: expected={args.expected_count}, "
            f"actual={len(manifest_rows)}"
        )
    if output["checkpoint"].exists() and not args.resume:
        raise FileExistsError(
            f"{output['checkpoint']} exists; pass --resume or use a new directory"
        )
    records = read_checkpoint(output["checkpoint"]) if args.resume else {}
    client = AsyncOpenAI(
        api_key=args.api_key or "EMPTY",
        base_url=args.api_base,
        timeout=args.request_timeout_seconds,
        max_retries=args.client_max_retries,
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    started = time.monotonic()
    try:
        for attempt in range(1, args.retry_rounds + 1):
            max_tokens = min(
                args.max_tokens * (2 ** (attempt - 1)),
                args.max_retry_tokens,
            )
            pending = [
                row
                for row in manifest_rows
                if not local_record_complete(records.get(str(row["sample_id"])))
            ]
            print(
                f"method={args.method} attempt={attempt}/{args.retry_rounds} "
                f"completed={len(manifest_rows) - len(pending)} "
                f"pending={len(pending)} concurrency={args.concurrency} "
                f"max_tokens={max_tokens}",
                flush=True,
            )
            if not pending:
                break
            progress = tqdm(
                total=len(pending),
                desc=f"{args.method} local round {attempt}",
                unit="case",
            )
            try:
                round_rows = await asyncio.gather(
                    *(
                        local_infer_one(
                            row,
                            args=args,
                            client=client,
                            semaphore=semaphore,
                            checkpoint=output["checkpoint"],
                            lock=lock,
                            progress=progress,
                            attempt=attempt,
                            max_tokens=max_tokens,
                        )
                        for row in pending
                    )
                )
            finally:
                progress.close()
            records.update(
                {str(row["sample_id"]): row for row in round_rows}
            )
            if all(local_record_complete(row) for row in round_rows):
                break
            if attempt < args.retry_rounds:
                await asyncio.sleep(args.retry_delay_seconds)
    finally:
        await client.close()
    population_index = population_by_index(summary)
    final_records = [
        records[str(row["sample_id"])] for row in manifest_rows
    ]
    predictions = [
        materialize_prediction(manifest, record, population_index)
        for manifest, record in zip(
            manifest_rows, final_records, strict=True
        )
    ]
    predictions.sort(key=lambda row: int(row["population_index"]))
    atomic_write_json(output["predictions"], predictions)
    report = evaluation_report(predictions)
    report.update(
        {
            "method": args.method,
            "provider": "local_vllm",
            "model": args.record_model_name or args.model,
            "request_model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "max_completion_tokens": args.max_retry_tokens,
            "prediction_count": len(predictions),
            "status_counts": dict(
                sorted(Counter(row["status"] for row in final_records).items())
            ),
            "token_usage": aggregate_usage(final_records),
            "generated_at": now_iso(),
        }
    )
    atomic_write_json(output["evaluation"], report)
    run_summary = {
        **report,
        "prepared_dir": str(prepared["directory"]),
        "manifest_sha256": manifest_sha,
        "elapsed_seconds": time.monotonic() - started,
        "completed_at": now_iso(),
    }
    atomic_write_json(output["run_summary"], run_summary)
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))
    if report["status_counts"].get("ERROR", 0):
        raise RuntimeError(
            f"Local inference finished with "
            f"{report['status_counts']['ERROR']} errors"
        )


def add_key_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--api-key")
    parser.add_argument("--api-key-env-file")
    parser.add_argument("--request-timeout-seconds", type=float, default=120.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--method", choices=["langmem", "rag"], required=True)
    prepare.add_argument("--input-path", default=str(DEFAULT_INPUT))
    prepare.add_argument("--memory-path", default=str(DEFAULT_LANGMEM))
    prepare.add_argument("--db-path", default=str(DEFAULT_RAG))
    prepare.add_argument("--collection-name", default="user_memories")
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--model", default="gpt-5-mini")
    prepare.add_argument("--reasoning-effort", default="minimal")
    prepare.add_argument("--max-completion-tokens", type=int, default=2048)
    prepare.add_argument("--query", choices=["hint", "nohint"], default="hint")
    prepare.add_argument("--exclude-easy-conflict", action="store_true")
    prepare.add_argument("--expected-count", type=int, default=4695)
    prepare.add_argument("--max-queries", type=int)
    prepare.add_argument("--top-k", type=int, default=5)
    prepare.add_argument(
        "--embedding-model", default="text-embedding-3-small"
    )
    prepare.add_argument("--embedding-batch-size", type=int, default=256)
    prepare.add_argument("--retry-count", type=int, default=5)
    prepare.add_argument("--force", action="store_true")
    add_key_args(prepare)

    for name in ("submit", "status", "collect"):
        command = subparsers.add_parser(name)
        command.add_argument("--prepared-dir", required=True)
        command.add_argument(
            "--method", choices=["amem", "langmem", "rag"], required=True
        )
        add_key_args(command)

    local = subparsers.add_parser("local")
    local.add_argument(
        "--method", choices=["amem", "langmem", "rag"], required=True
    )
    local.add_argument("--prepared-dir", required=True)
    local.add_argument("--output-dir", required=True)
    local.add_argument("--model", default="gpt-oss-20b")
    local.add_argument("--record-model-name")
    local.add_argument("--api-base", required=True)
    local.add_argument("--api-key", default="EMPTY")
    local.add_argument("--reasoning-effort", default="low")
    local.add_argument("--expected-count", type=int, default=4695)
    local.add_argument("--concurrency", type=int, default=256)
    local.add_argument("--max-tokens", type=int, default=2048)
    local.add_argument("--max-retry-tokens", type=int, default=2048)
    local.add_argument("--retry-rounds", type=int, default=2)
    local.add_argument("--retry-delay-seconds", type=float, default=5.0)
    local.add_argument("--request-timeout-seconds", type=float, default=7200.0)
    local.add_argument("--client-max-retries", type=int, default=0)
    local.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        prepare_command(args)
    elif args.command == "submit":
        submit_command(args)
    elif args.command == "status":
        status_command(args)
    elif args.command == "collect":
        collect_batch(args)
    elif args.command == "local":
        asyncio.run(local_command(args))
    else:
        raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
