"""Build Experiment8 A-MEM artifacts through a local OpenAI-compatible server.

This runner keeps the official A-MEM construction order inside each user:
metadata -> raw-content neighbor search -> evolution -> commit/re-embed.  Only
independent users are processed concurrently so a vLLM server can batch GPU
work without breaking causal memory evolution.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping
from urllib.request import Request, urlopen

import httpx
import tqdm
from openai import AsyncOpenAI


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.amem.build_memory_batch import (  # noqa: E402
    build_evolution_request,
    build_metadata_request,
    note_id_for,
)
from methods.amem.common import (  # noqa: E402
    DEFAULT_CONSTRUCTION_TOP_K,
    DEFAULT_EMBEDDING_MODEL,
    STATE_VERSION,
    TOKEN_FIELDS,
    UPSTREAM_COMMIT,
    UPSTREAM_REPOSITORY,
    apply_evolution_result,
    assign_embeddings,
    atomic_write_json,
    canonical_json,
    create_pending_note,
    load_dataset,
    make_embedder,
    nearest_notes,
    note_document,
    normalize_usage,
    now_iso,
    parse_message_json,
    read_json,
    read_jsonl,
    sha256_file,
    sum_usage,
    turn_note_units,
)


DEFAULT_INPUT_PATH = ROOT / "data" / "MPT_v2_0725.json"
DEFAULT_OUTPUT_DIR = (
    ROOT / "outputs" / "amem" / "MPT_v2_0725_gpt-oss-20b_local"
)
DEFAULT_MODEL = "gpt-oss-20b"
DEFAULT_REASONING_EFFORT = "low"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--reasoning_effort",
        choices=["low", "medium", "high", "minimal", ""],
        default=DEFAULT_REASONING_EFFORT,
    )
    parser.add_argument("--embedding_model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--top_k", type=int, default=DEFAULT_CONSTRUCTION_TOP_K)
    parser.add_argument("--max_completion_tokens", type=int, default=1536)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--max_sessions", type=int, default=None)
    parser.add_argument("--start_example", type=int, default=0)
    parser.add_argument("--end_example", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--embedding_concurrency", type=int, default=8)
    parser.add_argument("--request_timeout_seconds", type=float, default=7200.0)
    parser.add_argument("--timeout_seconds", type=float, default=None)
    parser.add_argument("--retry_count", type=int, default=3)
    parser.add_argument("--retry_base_sleep", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip_preflight", action="store_true")
    parser.add_argument(
        "--response_format",
        choices=["json_schema", "json_object", "none"],
        default="json_schema",
    )
    parser.add_argument("--dry_run", action="store_true")
    return parser


def paths_for(output_dir: Path) -> dict[str, Path]:
    output_dir = output_dir.resolve()
    return {
        "output_dir": output_dir,
        "artifact": output_dir / "memory.jsonl",
        "errors": output_dir / "errors.jsonl",
        "summary": output_dir / "construction_summary.json",
        "manifest": output_dir / "manifest.json",
    }


def validate_args(args: argparse.Namespace) -> None:
    if args.top_k <= 0:
        raise ValueError("--top_k must be positive")
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be positive")
    if args.embedding_concurrency <= 0:
        raise ValueError("--embedding_concurrency must be positive")
    if args.max_completion_tokens <= 0:
        raise ValueError("--max_completion_tokens must be positive")
    for name in ("max_examples", "max_sessions"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name} must be positive")


def completed_example_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    for row in read_jsonl(path):
        example_id = row.get("example_id")
        if example_id:
            completed.add(str(example_id))
    return completed


def _public_config(args: argparse.Namespace, *, input_sha256: str) -> dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "method": "amem",
        "backend": "openai_compatible_local",
        "input_path": str(args.input_path.resolve()),
        "input_sha256": input_sha256,
        "base_url": args.base_url,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "embedding_model": args.embedding_model,
        "top_k": args.top_k,
        "max_completion_tokens": args.max_completion_tokens,
        "max_examples": args.max_examples,
        "max_sessions": args.max_sessions,
        "start_example": args.start_example,
        "end_example": args.end_example,
        "embedding_concurrency": args.embedding_concurrency,
        "response_format": args.response_format,
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
    }


def check_endpoint(base_url: str, *, expected_model: str) -> list[str]:
    request = Request(
        f"{base_url.rstrip('/')}/models",
        headers={"Authorization": "Bearer EMPTY"},
    )
    with urlopen(request, timeout=10.0) as response:
        payload = json.loads(response.read().decode("utf-8"))
    models = [
        str(item.get("id"))
        for item in payload.get("data", [])
        if isinstance(item, dict) and item.get("id")
    ]
    if expected_model not in models:
        raise RuntimeError(
            f"Local model mismatch: expected={expected_model} available={models}"
        )
    return models


def _chat_body_from_batch_request(
    request: Mapping[str, Any],
    *,
    response_format: str,
) -> dict[str, Any]:
    body = dict(request["body"])
    if response_format == "json_object":
        body["response_format"] = {"type": "json_object"}
    elif response_format == "none":
        body.pop("response_format", None)
    return body


def _usage_from_response(response: Any) -> dict[str, int]:
    if hasattr(response, "model_dump"):
        return normalize_usage(response.model_dump())
    if isinstance(response, Mapping):
        return normalize_usage(response)
    raise TypeError(f"Unsupported OpenAI response type: {type(response)!r}")


async def close_client(client: Any) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        value = close()
        if hasattr(value, "__await__"):
            await value


async def encode_texts(
    embedder: Any,
    texts: list[str],
    *,
    embed_lock: asyncio.Semaphore | None = None,
) -> list[list[float]]:
    async_encode = getattr(embedder, "async_encode", None)
    if callable(async_encode):
        value = async_encode(texts)
        if hasattr(value, "__await__"):
            return await value
        return value
    if embed_lock is None:
        return await asyncio.to_thread(embedder.encode, texts)
    async with embed_lock:
        return await asyncio.to_thread(embedder.encode, texts)


async def assign_embeddings_async(
    user_memory: dict[str, Any],
    changed: set[str],
    *,
    embedder: Any,
    embed_lock: asyncio.Semaphore | None = None,
) -> None:
    if callable(getattr(embedder, "async_encode", None)):
        selected_ids = {str(note_id) for note_id in changed}
        targets = [
            note
            for note in user_memory.get("notes") or []
            if str(note.get("note_id")) in selected_ids
        ]
        embeddings = await encode_texts(
            embedder,
            [note_document(note) for note in targets],
            embed_lock=embed_lock,
        )
        if len(embeddings) != len(targets):
            raise RuntimeError("Embedding result count does not match note count")
        for note, embedding in zip(targets, embeddings):
            note["embedding"] = embedding
        return
    if embed_lock is None:
        await asyncio.to_thread(
            assign_embeddings,
            user_memory,
            changed,
            embedder=embedder,
        )
        return
    async with embed_lock:
        await asyncio.to_thread(
            assign_embeddings,
            user_memory,
            changed,
            embedder=embedder,
        )


def _body_from_response(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        value = response.model_dump()
    elif isinstance(response, Mapping):
        value = dict(response)
    else:
        raise TypeError(f"Unsupported OpenAI response type: {type(response)!r}")
    if not isinstance(value, dict):
        raise TypeError("OpenAI response did not dump to a JSON object")
    return value


def parse_local_message_json(body: Mapping[str, Any]) -> dict[str, Any]:
    """Parse local-model JSON while tolerating wrappers and raw controls."""

    try:
        return parse_message_json(body)
    except (json.JSONDecodeError, ValueError):
        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError("Local response has no choices")
        message = choices[0].get("message") or {}
        content = str(message.get("content") or "").strip()
        decoder = json.JSONDecoder(strict=False)
        for index, character in enumerate(content):
            if character != "{":
                continue
            try:
                value, _ = decoder.raw_decode(content[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        preview = repr(content[:500])
        raise ValueError(
            "Local A-MEM response did not contain a JSON object; "
            f"content_preview={preview}"
        )


async def call_json_with_retries(
    client: AsyncOpenAI,
    body: Mapping[str, Any],
    *,
    retry_count: int,
    retry_base_sleep: float,
) -> tuple[dict[str, Any], dict[str, int]]:
    last_error: Exception | None = None
    for attempt in range(retry_count + 1):
        try:
            response = await client.chat.completions.create(**dict(body))
            response_body = _body_from_response(response)
            return (
                parse_local_message_json(response_body),
                _usage_from_response(response),
            )
        except Exception as exc:  # pragma: no cover - exercised by real endpoints
            last_error = exc
            if attempt >= retry_count:
                break
            await asyncio.sleep(retry_base_sleep * (2**attempt))
    assert last_error is not None
    raise last_error


def _zero_usage() -> dict[str, int]:
    return {field: 0 for field in TOKEN_FIELDS}


async def process_user(
    *,
    client: AsyncOpenAI,
    example: Mapping[str, Any],
    dataset_index: int,
    args: argparse.Namespace,
    embedder: Any,
    embed_lock: asyncio.Semaphore | None = None,
) -> dict[str, Any]:
    example_id = str(example.get("example_id") or f"row-{dataset_index}")
    note_units = turn_note_units(example, max_sessions=args.max_sessions)
    user_memory: dict[str, Any] = {
        "dataset_index": dataset_index,
        "note_count": len(note_units),
        "notes": [],
    }
    usage_rows: list[dict[str, int]] = []

    for note_unit in note_units:
        note_index = int(note_unit["note_index"])
        note_id = note_id_for(dataset_index, note_unit)
        metadata_request = build_metadata_request(
            custom_id=f"amem-i{dataset_index:04d}-n{note_index:04d}-m",
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            max_completion_tokens=args.max_completion_tokens,
            content=str(note_unit["content"]),
        )
        metadata_body = _chat_body_from_batch_request(
            metadata_request,
            response_format=args.response_format,
        )
        metadata, metadata_usage = await call_json_with_retries(
            client,
            metadata_body,
            retry_count=args.retry_count,
            retry_base_sleep=args.retry_base_sleep,
        )
        pending_note = create_pending_note(
            note_unit,
            note_id=note_id,
            metadata_response=metadata,
            metadata_usage=metadata_usage,
            timestamp=now_iso(),
        )
        query_embeddings = await encode_texts(
            embedder,
            [str(pending_note["content"])],
            embed_lock=embed_lock,
        )
        query_embedding = query_embeddings[0]
        candidates = nearest_notes(
            user_memory.get("notes") or [],
            query_embedding,
            top_k=args.top_k,
        )
        evolution_request = build_evolution_request(
            custom_id=f"amem-i{dataset_index:04d}-n{note_index:04d}-e",
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            max_completion_tokens=args.max_completion_tokens,
            pending_note=pending_note,
            candidates=candidates,
        )
        evolution_body = _chat_body_from_batch_request(
            evolution_request,
            response_format=args.response_format,
        )
        evolution, evolution_usage = await call_json_with_retries(
            client,
            evolution_body,
            retry_count=args.retry_count,
            retry_base_sleep=args.retry_base_sleep,
        )
        changed = apply_evolution_result(
            user_memory,
            pending_note=pending_note,
            candidate_note_ids=[str(note["note_id"]) for note in candidates],
            response=evolution,
            evolution_usage=evolution_usage,
            timestamp=now_iso(),
        )
        await assign_embeddings_async(
            user_memory,
            changed,
            embedder=embedder,
            embed_lock=embed_lock,
        )
        usage_rows.extend([metadata_usage, evolution_usage])

    notes = user_memory.get("notes") or []
    return {
        "example_id": example_id,
        "dataset_index": dataset_index,
        "method": "amem",
        "backend": "openai_compatible_local",
        "memory_model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "embedding_model": args.embedding_model,
        "top_k": args.top_k,
        "note_unit": "dialogue_turn",
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        "notes": notes,
        "construction_usage": sum_usage(usage_rows),
    }


async def run_ingestion(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    args.input_path = args.input_path.resolve()
    paths = paths_for(args.output_dir)
    rows = load_dataset(args.input_path)
    indexed_rows = list(enumerate(rows))[args.start_example : args.end_example]
    if args.max_examples is not None:
        indexed_rows = indexed_rows[: args.max_examples]

    input_sha256 = sha256_file(args.input_path)
    config = _public_config(args, input_sha256=input_sha256)
    manifest = {
        **config,
        "status": "DRY_RUN" if args.dry_run else "RUNNING",
        "artifact_path": str(paths["artifact"]),
        "errors_path": str(paths["errors"]),
        "summary_path": str(paths["summary"]),
        "selected_examples": len(indexed_rows),
        "created_at": now_iso(),
    }
    if args.dry_run:
        return manifest

    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    if args.resume and paths["manifest"].exists():
        previous = read_json(paths["manifest"])
        previous_config = {key: previous.get(key) for key in config}
        if previous_config != config:
            raise RuntimeError(
                "Resume configuration mismatch: "
                f"{canonical_json({'previous': previous_config, 'current': config})}"
            )
    elif not args.resume:
        paths["artifact"].write_text("", encoding="utf-8")
        paths["errors"].write_text("", encoding="utf-8")

    if not args.skip_preflight:
        manifest["available_models"] = check_endpoint(
            args.base_url,
            expected_model=args.model,
        )
    atomic_write_json(paths["manifest"], manifest)

    completed = completed_example_ids(paths["artifact"]) if args.resume else set()
    pending = [
        (dataset_index, row)
        for dataset_index, row in indexed_rows
        if str(row.get("example_id") or f"row-{dataset_index}") not in completed
    ]
    embedder = make_embedder(args.embedding_model)
    await encode_texts(embedder, ["A-MEM embedding warmup"])
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=max(args.concurrency, 1),
            max_keepalive_connections=max(1, min(args.concurrency, 128)),
        ),
        timeout=args.timeout_seconds or args.request_timeout_seconds,
    )
    client = AsyncOpenAI(
        api_key=args.api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY",
        base_url=args.base_url,
        timeout=args.timeout_seconds or args.request_timeout_seconds,
        max_retries=0,
        http_client=http_client,
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    embed_lock = asyncio.Semaphore(args.embedding_concurrency)
    lock = asyncio.Lock()
    success_count = 0
    failure_count = 0

    async def one(dataset_index: int, row: Mapping[str, Any]) -> None:
        nonlocal success_count, failure_count
        example_id = str(row.get("example_id") or f"row-{dataset_index}")
        async with semaphore:
            try:
                result = await process_user(
                    client=client,
                    example=row,
                    dataset_index=dataset_index,
                    args=args,
                    embedder=embedder,
                    embed_lock=embed_lock,
                )
            except Exception as exc:
                failure_count += 1
                error_row = {
                    "example_id": example_id,
                    "dataset_index": dataset_index,
                    "status": "ERROR",
                    "error": f"{type(exc).__name__}: {exc}",
                    "updated_at": now_iso(),
                }
                async with lock:
                    with paths["errors"].open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(error_row, ensure_ascii=False) + "\n")
                return
            success_count += 1
            async with lock:
                with paths["artifact"].open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    try:
        tasks = [asyncio.create_task(one(index, row)) for index, row in pending]
        for task in tqdm.tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Building local A-MEM",
        ):
            await task
    finally:
        await close_client(client)
        await http_client.aclose()

    artifact_rows = read_jsonl(paths["artifact"]) if paths["artifact"].exists() else []
    all_usage = [
        row.get("construction_usage") or {}
        for row in artifact_rows
        if isinstance(row, Mapping)
    ]
    note_count = sum(len(row.get("notes") or []) for row in artifact_rows)
    link_count = sum(
        len(note.get("links") or [])
        for row in artifact_rows
        for note in row.get("notes") or []
    )
    summary = {
        **config,
        "status": "COMPLETE" if failure_count == 0 else "FAILED",
        "completed_at": now_iso(),
        "completed_before_run": len(indexed_rows) - len(pending),
        "succeeded_this_run": success_count,
        "failed_this_run": failure_count,
        "completed_total": len(artifact_rows),
        "example_count": len(artifact_rows),
        "note_count": note_count,
        "directed_link_count": link_count,
        "artifact_path": str(paths["artifact"]),
        "artifact_sha256": (
            sha256_file(paths["artifact"]) if paths["artifact"].exists() else None
        ),
        "construction_usage": sum_usage(all_usage),
        "official_semantics": {
            "turn_level_memory_note": True,
            "immutable_content": True,
            "separate_metadata_analysis": True,
            "raw_content_neighbor_query": True,
            "construction_neighbor_k": args.top_k,
            "strengthen_new_note": True,
            "update_neighbor_metadata": True,
            "directed_links_without_backlinks": True,
            "retriever_document_includes_metadata": True,
        },
        "adapter_boundaries": [
            "OpenAI-compatible chat completions replace OpenAI Batch.",
            "Causal order is preserved inside each Experiment8 example_id.",
            "Independent users are processed concurrently for vLLM batching.",
            "Each completed user is checkpointed immediately in memory.jsonl.",
        ],
    }
    atomic_write_json(paths["summary"], summary)
    manifest.update(summary)
    atomic_write_json(paths["manifest"], manifest)
    if failure_count:
        raise RuntimeError(
            f"Local A-MEM construction completed with {failure_count} failures"
        )
    return summary


def main() -> None:
    args = build_parser().parse_args()
    result = asyncio.run(run_ingestion(args))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
