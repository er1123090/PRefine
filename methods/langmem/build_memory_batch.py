"""Build Experiment8 LangMem artifacts with resumable OpenAI Batch rounds.

Each user's sessions are causally dependent: the memories produced for session N
must exist before session N+1 is prepared.  This runner therefore submits one
cross-user batch per session round while retaining durable state in SQLite.

The resulting artifact keeps provider-reported construction usage, per-session
memory accumulation metrics, and replayable insert/update/delete events.  The
same state database can later materialize session-count or construction-token
ablations without calling an LLM again.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sqlite3
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import jsonpatch
import numpy as np
from langchain_core.utils.function_calling import convert_to_openai_tool
from langmem import utils as langmem_utils
from openai import OpenAI
from trustcall._base import PatchDoc, _create_remove_doc_schema


ROOT = Path(__file__).resolve().parents[2]
LANGMEM_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(LANGMEM_DIR) not in sys.path:
    sys.path.insert(0, str(LANGMEM_DIR))

from common import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    MEMORY_INSTRUCTIONS,
    SemanticMemory,
    build_session_messages,
    count_serialized_memory_tokens,
    get_encoding,
    last_user_utterance,
    load_chains_dataset,
    materialize_namespace,
    now_iso,
    session_input_token_count,
)
from src.construction_usage import usage_from_response  # noqa: E402


DEFAULT_INPUT_PATH = ROOT / "data" / "MPT_v2_0725.json"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "langmem" / "MPT_v2_0725_gpt-5-mini_batch"
STATE_VERSION = 1
TOKEN_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "total_tokens",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resumable LangMem construction through OpenAI Batch."
    )
    parser.add_argument(
        "command",
        choices=(
            "run",
            "direct",
            "status",
            "prepare",
            "submit",
            "collect",
            "finalize",
            "materialize",
        ),
        nargs="?",
        default="run",
    )
    parser.add_argument("--input_path", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--reasoning_effort", default="minimal")
    parser.add_argument("--embedding_model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--poll_seconds", type=int, default=60)
    parser.add_argument("--embedding_chunk_size", type=int, default=128)
    parser.add_argument("--max_completion_tokens", type=int, default=2048)
    parser.add_argument(
        "--direct_concurrency",
        type=int,
        default=5,
        help="Concurrent ordinary API calls used by the direct command.",
    )
    parser.add_argument(
        "--direct_max_attempts",
        type=int,
        default=5,
        help="Maximum attempts for each ordinary API call.",
    )
    parser.add_argument(
        "--direct_retry_seconds",
        type=float,
        default=5.0,
        help="Initial retry delay for ordinary API calls.",
    )
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--force_init", action="store_true")
    parser.add_argument(
        "--materialized_output",
        default=None,
        help="Output JSONL for the materialize command.",
    )
    parser.add_argument("--max_sessions", type=int, default=None)
    parser.add_argument(
        "--max_cumulative_construction_tokens", type=int, default=None
    )
    parser.add_argument("--max_stored_memory_tokens", type=int, default=None)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def embedding_document_text(value: Mapping[str, Any]) -> str:
    """Match LangGraph InMemoryStore's default `$` index serialization."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain an object")
            rows.append(value)
    return rows


def require_api_key(args: argparse.Namespace) -> str:
    key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Pass --api_key or set OPENAI_API_KEY.")
    return key


def client_for(args: argparse.Namespace) -> OpenAI:
    return OpenAI(api_key=require_api_key(args), timeout=120.0)


def paths_for(args: argparse.Namespace) -> dict[str, Path]:
    output_dir = Path(args.output_dir).resolve()
    return {
        "output_dir": output_dir,
        "state": output_dir / "state.json",
        "db": output_dir / "construction.sqlite3",
        "artifact": output_dir / "memory.jsonl",
        "accumulation": output_dir / "memory_accumulation.jsonl",
        "manifest": output_dir / "memory.manifest.json",
    }


def connect_db(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            example_id TEXT PRIMARY KEY,
            dataset_index INTEGER NOT NULL,
            session_count INTEGER NOT NULL,
            next_session_index INTEGER NOT NULL DEFAULT 1,
            cumulative_input_tokens INTEGER NOT NULL DEFAULT 0,
            cumulative_cached_input_tokens INTEGER NOT NULL DEFAULT 0,
            cumulative_output_tokens INTEGER NOT NULL DEFAULT 0,
            cumulative_reasoning_tokens INTEGER NOT NULL DEFAULT 0,
            cumulative_total_tokens INTEGER NOT NULL DEFAULT 0,
            cumulative_memory_token_area INTEGER NOT NULL DEFAULT 0,
            peak_memory_count INTEGER NOT NULL DEFAULT 0,
            peak_stored_memory_tokens INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS memories (
            example_id TEXT NOT NULL,
            memory_key TEXT NOT NULL,
            value_json TEXT NOT NULL,
            embedding BLOB NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            created_session_index INTEGER NOT NULL,
            updated_session_index INTEGER NOT NULL,
            PRIMARY KEY (example_id, memory_key),
            FOREIGN KEY (example_id) REFERENCES users(example_id)
        );

        CREATE TABLE IF NOT EXISTS session_exports (
            example_id TEXT NOT NULL,
            session_index INTEGER NOT NULL,
            payload_json TEXT NOT NULL,
            PRIMARY KEY (example_id, session_index),
            FOREIGN KEY (example_id) REFERENCES users(example_id)
        );
        """
    )


def load_dataset(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = load_chains_dataset(args.input_path).to_dict("records")
    if args.max_examples is not None:
        rows = rows[: args.max_examples]
    return rows


def initial_state(args: argparse.Namespace, rows: Sequence[Mapping[str, Any]]) -> dict:
    input_path = Path(args.input_path).resolve()
    return {
        "version": STATE_VERSION,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "examples": len(rows),
        "sessions": sum(len(row.get("sessions", [])) for row in rows),
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "embedding_model": args.embedding_model,
        "max_completion_tokens": args.max_completion_tokens,
        "current_round": 1,
        "phase": "ready",
        "active_attempt": None,
        "batch_history": [],
        "embedding_api_usage": [],
    }


def initialize(args: argparse.Namespace) -> dict[str, Any]:
    paths = paths_for(args)
    output_dir = paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    if paths["state"].exists() and not args.force_init:
        state = read_json(paths["state"])
        validate_state(args, state)
        return state
    if paths["db"].exists() and args.force_init:
        paths["db"].unlink()
    rows = load_dataset(args)
    connection = connect_db(paths["db"])
    try:
        create_schema(connection)
        for dataset_index, row in enumerate(rows):
            example_id = str(row.get("example_id", f"row_{dataset_index:04d}"))
            connection.execute(
                """
                INSERT INTO users(example_id, dataset_index, session_count)
                VALUES (?, ?, ?)
                """,
                (example_id, dataset_index, len(row.get("sessions", []))),
            )
        connection.commit()
    finally:
        connection.close()
    state = initial_state(args, rows)
    atomic_write_json(paths["state"], state)
    print(
        f"Initialized {len(rows)} examples and {state['sessions']} sessions at "
        f"{output_dir}",
        flush=True,
    )
    return state


def validate_state(args: argparse.Namespace, state: Mapping[str, Any]) -> None:
    checks = {
        "version": STATE_VERSION,
        "input_path": str(Path(args.input_path).resolve()),
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "embedding_model": args.embedding_model,
        "max_completion_tokens": args.max_completion_tokens,
    }
    mismatches = {
        key: (state.get(key), expected)
        for key, expected in checks.items()
        if state.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(
            "Existing state does not match requested configuration: "
            + canonical_json(mismatches)
        )
    current_sha = sha256_file(Path(args.input_path).resolve())
    if state.get("input_sha256") != current_sha:
        raise RuntimeError("Input dataset checksum changed after initialization.")


def update_state(path: Path, state: dict[str, Any], **changes: Any) -> dict[str, Any]:
    state.update(changes)
    state["updated_at"] = utc_now()
    atomic_write_json(path, state)
    return state


def round_attempt_dir(
    output_dir: Path, round_index: int, attempt_index: int
) -> Path:
    return output_dir / "rounds" / f"round_{round_index:03d}" / (
        f"attempt_{attempt_index:02d}"
    )


def summarize_calls(calls: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    result = {
        field: sum(int(row.get(field) or 0) for row in calls)
        for field in TOKEN_FIELDS
    }
    result["call_count"] = len(calls)
    result["usage_available_calls"] = sum(
        int(bool(row.get("usage_available"))) for row in calls
    )
    result["usage_missing_calls"] = (
        result["call_count"] - result["usage_available_calls"]
    )
    return result


def build_usage_report(calls: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [dict(call) for call in calls]
    components = sorted(
        {str(call["component"]) for call in rows if call.get("component")}
    )
    sessions = sorted(
        {
            int(call["session_index"])
            for call in rows
            if isinstance(call.get("session_index"), int)
        }
    )
    return {
        "summary": summarize_calls(rows),
        "by_component": {
            component: summarize_calls(
                [call for call in rows if call.get("component") == component]
            )
            for component in components
        },
        "by_session": {
            str(session): {
                "summary": summarize_calls(
                    [
                        call
                        for call in rows
                        if call.get("session_index") == session
                    ]
                ),
                "by_component": {
                    component: summarize_calls(
                        [
                            call
                            for call in rows
                            if call.get("session_index") == session
                            and call.get("component") == component
                        ]
                    )
                    for component in components
                    if any(
                        call.get("session_index") == session
                        and call.get("component") == component
                        for call in rows
                    )
                },
            }
            for session in sessions
        },
        "calls": rows,
    }


def snapshot_item(
    example_id: str,
    memory_key: str,
    value: Mapping[str, Any],
    created_at: str,
    updated_at: str,
    created_session_index: int,
    updated_session_index: int,
) -> dict[str, Any]:
    return {
        "namespace": list(materialize_namespace(example_id)),
        "key": memory_key,
        "value": dict(value),
        "created_at": created_at,
        "updated_at": updated_at,
        "score": None,
        "construction_provenance": {
            "created_session_index": created_session_index,
            "updated_session_index": updated_session_index,
        },
    }


def memory_from_row(row: sqlite3.Row, include_embedding: bool = True) -> dict[str, Any]:
    item = snapshot_item(
        example_id=str(row["example_id"]),
        memory_key=str(row["memory_key"]),
        value=json.loads(row["value_json"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        created_session_index=int(row["created_session_index"]),
        updated_session_index=int(row["updated_session_index"]),
    )
    if include_embedding:
        item["_embedding"] = np.frombuffer(row["embedding"], dtype=np.float32)
    return item


def load_memories(
    connection: sqlite3.Connection, example_id: str, include_embedding: bool = True
) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT * FROM memories
        WHERE example_id = ?
        ORDER BY memory_key
        """,
        (example_id,),
    ).fetchall()
    return [memory_from_row(row, include_embedding=include_embedding) for row in rows]


def item_without_embedding(item: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in item.items() if key != "_embedding"}


def stored_memory_tokens(items: Sequence[Mapping[str, Any]], encoding: Any) -> int:
    return count_serialized_memory_tokens(
        [item_without_embedding(item) for item in items], encoding
    )


def allocate_integer(total: int, weights: Sequence[int]) -> list[int]:
    if not weights:
        return []
    normalized = [max(0, int(weight)) for weight in weights]
    weight_sum = sum(normalized)
    if weight_sum == 0:
        normalized = [1] * len(weights)
        weight_sum = len(weights)
    raw = [total * weight / weight_sum for weight in normalized]
    floors = [int(value) for value in raw]
    remaining = total - sum(floors)
    order = sorted(
        range(len(raw)),
        key=lambda index: (raw[index] - floors[index], -index),
        reverse=True,
    )
    for index in order[:remaining]:
        floors[index] += 1
    return floors


def embed_labeled_texts(
    client: OpenAI,
    model: str,
    labeled_texts: Sequence[tuple[str, str]],
    encoding: Any,
    chunk_size: int,
    component: str,
    session_indexes: Mapping[str, int],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    vectors: dict[str, np.ndarray] = {}
    allocated_usage: dict[str, dict[str, Any]] = {}
    api_usage: list[dict[str, Any]] = []
    for start in range(0, len(labeled_texts), chunk_size):
        chunk = list(labeled_texts[start : start + chunk_size])
        labels = [label for label, _ in chunk]
        texts = [text for _, text in chunk]
        response = client.embeddings.create(model=model, input=texts)
        usage = usage_from_response(response)
        api_usage.append(
            {
                "created_at": utc_now(),
                "component": component,
                "model": model,
                "inputs": len(chunk),
                **usage,
            }
        )
        weights = [max(1, len(encoding.encode(text))) for text in texts]
        input_allocations = allocate_integer(
            int(usage.get("input_tokens") or 0), weights
        )
        total_allocations = allocate_integer(
            int(usage.get("total_tokens") or 0), weights
        )
        indexed = sorted(response.data, key=lambda item: item.index)
        if len(indexed) != len(chunk):
            raise RuntimeError(
                f"Embedding response length mismatch: {len(indexed)} != {len(chunk)}"
            )
        for offset, (label, item) in enumerate(zip(labels, indexed, strict=True)):
            vectors[label] = np.asarray(item.embedding, dtype=np.float32)
            allocated_usage[label] = {
                "component": component,
                "session_index": session_indexes.get(label),
                "provider": "openai",
                "model": model,
                "usage_available": bool(usage.get("usage_available")),
                "input_tokens": input_allocations[offset],
                "cached_input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "total_tokens": total_allocations[offset],
                "allocation_method": "cl100k_proportional_to_api_chunk_total",
            }
    return vectors, allocated_usage, api_usage


def stable_memory_id(example_id: str, memory_key: str) -> str:
    namespace = materialize_namespace(example_id)
    return uuid.uuid5(
        uuid.NAMESPACE_DNS, str((*namespace, memory_key))
    ).hex


def retrieve_memories(
    items: Sequence[Mapping[str, Any]], query_vector: np.ndarray, limit: int = 5
) -> list[dict[str, Any]]:
    scored: list[tuple[float, Mapping[str, Any]]] = []
    query_norm = float(np.linalg.norm(query_vector))
    for item in items:
        vector = item.get("_embedding")
        if not isinstance(vector, np.ndarray):
            continue
        denominator = query_norm * float(np.linalg.norm(vector))
        score = (
            float(np.dot(query_vector, vector) / denominator)
            if denominator
            else 0.0
        )
        scored.append((score, item))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    result: list[dict[str, Any]] = []
    for score, item in scored[:limit]:
        clean = item_without_embedding(item)
        clean["score"] = score
        clean["stable_id"] = stable_memory_id(
            str(clean["namespace"][1]), str(clean["key"])
        )
        result.append(clean)
    return result


def manager_messages(
    messages: Sequence[Mapping[str, str]],
    retrieved: Sequence[Mapping[str, Any]],
    session_tag: str,
) -> list[dict[str, str]]:
    conversation = langmem_utils.get_conversation(list(messages))
    session = (
        f"\n\n<session_{session_tag}>\n{conversation}\n</session_{session_tag}>"
    )
    result = [
        {"role": "system", "content": "You are a memory subroutine for an AI."},
        {
            "role": "user",
            "content": (
                f"{MEMORY_INSTRUCTIONS}\n\nEnrich, prune, and organize memories based on any new information. "
                f"If an existing memory is incorrect or outdated, update it based on the new information. "
                f"All operations must be done in single parallel multi-tool call."
                f" Avoid duplicate extractions. {session}"
            ),
        },
    ]
    if retrieved:
        instances = "\n".join(
            (
                f'<instance id={item["stable_id"]} schema_type="SemanticMemory">\n'
                f'{item["value"]["content"]}\n</instance>'
            )
            for item in retrieved
        )
        existing_message = (
            "Generate JSONPatches to update the existing schema instances."
            " If you need to extract or insert *new* instances of the schemas"
            ", call the relevant function(s).\n"
            "<existing>\n"
            f"{instances}\n"
            "</existing>\n"
        )
        result[0]["content"] += "\n\n" + existing_message
    return result


def request_tools(retrieved: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    tools: list[Any] = []
    if retrieved:
        tools.append(PatchDoc)
    tools.append(SemanticMemory)
    if retrieved:
        allowed_ids = tuple(sorted(str(item["stable_id"]) for item in retrieved))
        tools.append(_create_remove_doc_schema(allowed_ids))
    return [convert_to_openai_tool(tool) for tool in tools]


def build_batch_request(
    custom_id: str,
    model: str,
    reasoning_effort: str,
    max_completion_tokens: int,
    messages: Sequence[Mapping[str, str]],
    retrieved: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "tools": request_tools(retrieved),
        "parallel_tool_calls": True,
        "max_completion_tokens": max_completion_tokens,
    }
    if retrieved:
        body["tool_choice"] = "required"
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }


def custom_id_for(dataset_index: int, session_index: int) -> str:
    return f"lm-i{dataset_index:04d}-s{session_index:02d}"


def prepare_round(args: argparse.Namespace, state: dict[str, Any]) -> dict[str, Any]:
    if state["phase"] != "ready":
        raise RuntimeError(f"Cannot prepare while phase={state['phase']}")
    paths = paths_for(args)
    rows = load_dataset(args)
    round_index = int(state["current_round"])
    connection = connect_db(paths["db"])
    encoding = get_encoding()
    candidates: list[dict[str, Any]] = []
    skipped: list[tuple[sqlite3.Row, Mapping[str, Any]]] = []
    try:
        user_rows = connection.execute(
            """
            SELECT * FROM users
            WHERE next_session_index = ? AND next_session_index <= session_count
            ORDER BY dataset_index
            """,
            (round_index,),
        ).fetchall()
        for user in user_rows:
            example = rows[int(user["dataset_index"])]
            session = example.get("sessions", [])[round_index - 1]
            messages = build_session_messages(session)
            if not messages:
                skipped.append((user, session))
                continue
            query_windows = langmem_utils.get_dialated_windows(messages, 1)
            if not query_windows:
                skipped.append((user, session))
                continue
            custom_id = custom_id_for(
                int(user["dataset_index"]), round_index
            )
            candidates.append(
                {
                    "custom_id": custom_id,
                    "example_id": str(user["example_id"]),
                    "dataset_index": int(user["dataset_index"]),
                    "session_index": round_index,
                    "session": session,
                    "messages": messages,
                    "query_text": query_windows[0],
                }
            )
        if skipped:
            with connection:
                for user, session in skipped:
                    payload = skipped_session_payload(
                        user=user,
                        session=session,
                        session_index=round_index,
                    )
                    connection.execute(
                        """
                        INSERT INTO session_exports(example_id, session_index, payload_json)
                        VALUES (?, ?, ?)
                        """,
                        (
                            str(user["example_id"]),
                            round_index,
                            canonical_json(payload),
                        ),
                    )
                    connection.execute(
                        """
                        UPDATE users SET next_session_index = next_session_index + 1
                        WHERE example_id = ?
                        """,
                        (str(user["example_id"]),),
                    )
        if not candidates:
            if user_rows:
                return update_state(
                    paths["state"], state, current_round=round_index + 1
                )
            finalize(args, state)
            return read_json(paths["state"])

        api_client = client_for(args)
        query_inputs = [
            (str(candidate["custom_id"]), str(candidate["query_text"]))
            for candidate in candidates
        ]
        session_indexes = {
            str(candidate["custom_id"]): round_index for candidate in candidates
        }
        query_vectors, query_usage, embedding_api_usage = embed_labeled_texts(
            client=api_client,
            model=args.embedding_model,
            labeled_texts=query_inputs,
            encoding=encoding,
            chunk_size=args.embedding_chunk_size,
            component="embedding_query",
            session_indexes=session_indexes,
        )

        attempt_index = 1
        attempt_dir = round_attempt_dir(
            paths["output_dir"], round_index, attempt_index
        )
        request_rows: list[dict[str, Any]] = []
        manifest_rows: list[dict[str, Any]] = []
        for candidate in candidates:
            custom_id = str(candidate["custom_id"])
            example_id = str(candidate["example_id"])
            memories = load_memories(connection, example_id)
            retrieved = retrieve_memories(
                memories, query_vectors[custom_id], limit=5
            )
            prepared_messages = manager_messages(
                candidate["messages"],
                retrieved,
                session_tag=custom_id,
            )
            request_rows.append(
                build_batch_request(
                    custom_id=custom_id,
                    model=args.model,
                    reasoning_effort=args.reasoning_effort,
                    max_completion_tokens=args.max_completion_tokens,
                    messages=prepared_messages,
                    retrieved=retrieved,
                )
            )
            manifest_rows.append(
                {
                    "custom_id": custom_id,
                    "example_id": example_id,
                    "dataset_index": int(candidate["dataset_index"]),
                    "session_index": round_index,
                    "dialogue_id": candidate["session"].get("dialogue_id"),
                    "last_user_utterance": last_user_utterance(
                        candidate["session"]
                    ),
                    "session_input_tokens": session_input_token_count(
                        candidate["session"], encoding
                    ),
                    "memory_count_before_session": len(memories),
                    "stored_memory_tokens_before_session": stored_memory_tokens(
                        memories, encoding
                    ),
                    "retrieved_memories": retrieved,
                    "retrieval_query_sha256": hashlib.sha256(
                        str(candidate["query_text"]).encode("utf-8")
                    ).hexdigest(),
                    "embedding_query_usage": query_usage[custom_id],
                    "request_messages_sha256": sha256_json(prepared_messages),
                }
            )
        requests_path = attempt_dir / "requests.jsonl"
        manifest_path = attempt_dir / "request_manifest.jsonl"
        write_jsonl(requests_path, request_rows)
        write_jsonl(manifest_path, manifest_rows)
        attempt = {
            "round": round_index,
            "attempt": attempt_index,
            "created_at": utc_now(),
            "request_count": len(request_rows),
            "requests_path": str(requests_path),
            "requests_sha256": sha256_file(requests_path),
            "request_manifest_path": str(manifest_path),
            "request_manifest_sha256": sha256_file(manifest_path),
            "batch_id": None,
            "batch_status": None,
            "input_file_id": None,
            "output_file_id": None,
            "error_file_id": None,
        }
        state["embedding_api_usage"].extend(embedding_api_usage)
        update_state(
            paths["state"],
            state,
            phase="prepared",
            active_attempt=attempt,
        )
        print(
            f"Prepared round {round_index}: {len(request_rows)} requests, "
            f"{len(skipped)} empty sessions skipped",
            flush=True,
        )
        return state
    finally:
        connection.close()


def skipped_session_payload(
    user: Mapping[str, Any],
    session: Mapping[str, Any],
    session_index: int,
) -> dict[str, Any]:
    cumulative = {
        "input_tokens": int(user["cumulative_input_tokens"]),
        "cached_input_tokens": int(user["cumulative_cached_input_tokens"]),
        "output_tokens": int(user["cumulative_output_tokens"]),
        "reasoning_tokens": int(user["cumulative_reasoning_tokens"]),
        "total_tokens": int(user["cumulative_total_tokens"]),
    }
    return {
        "session_index": session_index,
        "dialogue_id": session.get("dialogue_id"),
        "status": "SKIPPED_EMPTY_SESSION",
        "last_user_utterance": last_user_utterance(dict(session)),
        "session_input_tokens": 0,
        "memory_count_before_session": None,
        "memory_count_after_session": None,
        "stored_memory_tokens_before_session": None,
        "stored_memory_tokens_after_session": None,
        "stored_memory_delta_tokens": 0,
        "memory_operations": [],
        "construction_token_usage": build_usage_report([]),
        "cumulative_construction_token_usage": cumulative,
    }


def submit_round(args: argparse.Namespace, state: dict[str, Any]) -> dict[str, Any]:
    if state["phase"] != "prepared":
        raise RuntimeError(f"Cannot submit while phase={state['phase']}")
    paths = paths_for(args)
    attempt = dict(state["active_attempt"])
    requests_path = Path(attempt["requests_path"])
    if sha256_file(requests_path) != attempt["requests_sha256"]:
        raise RuntimeError("Prepared request checksum changed; refusing submission.")
    if attempt.get("batch_id"):
        raise RuntimeError(
            "This attempt already has a batch_id; refusing duplicate charges."
        )
    api_client = client_for(args)
    with requests_path.open("rb") as handle:
        uploaded = api_client.files.create(file=handle, purpose="batch")
    batch = api_client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "experiment": "experiment8_langmem",
            "model": args.model,
            "round": str(attempt["round"]),
            "reasoning_effort": args.reasoning_effort,
        },
    )
    attempt.update(
        {
            "submitted_at": utc_now(),
            "input_file_id": uploaded.id,
            "batch_id": batch.id,
            "batch_status": batch.status,
        }
    )
    history_entry = {
        "round": attempt["round"],
        "attempt": attempt["attempt"],
        "batch_id": batch.id,
        "input_file_id": uploaded.id,
        "request_count": attempt["request_count"],
        "submitted_at": attempt["submitted_at"],
        "status": batch.status,
    }
    state["batch_history"].append(history_entry)
    update_state(
        paths["state"],
        state,
        phase="submitted",
        active_attempt=attempt,
    )
    print(
        f"Submitted round {attempt['round']} batch {batch.id} "
        f"({attempt['request_count']} requests, status={batch.status})",
        flush=True,
    )
    return state


def _response_body(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        value = response.model_dump(mode="json")
    elif isinstance(response, Mapping):
        value = dict(response)
    else:
        raise TypeError(
            f"Unsupported ordinary API response type: {type(response).__name__}"
        )
    if not isinstance(value, dict):
        raise TypeError("Ordinary API response did not serialize to an object.")
    return value


def _direct_response_row(custom_id: str, response: Any) -> dict[str, Any]:
    body = _response_body(response)
    return {
        "id": f"direct_req_{uuid.uuid4().hex}",
        "custom_id": custom_id,
        "response": {
            "status_code": 200,
            "request_id": body.get("id"),
            "body": body,
        },
        "error": None,
    }


def _direct_history_entry(
    state: Mapping[str, Any], run_id: str
) -> dict[str, Any] | None:
    for entry in state.get("direct_history", []):
        if entry.get("run_id") == run_id:
            return entry
    return None


def execute_round_direct(
    args: argparse.Namespace, state: dict[str, Any]
) -> dict[str, Any]:
    if state["phase"] not in {"prepared", "direct_running"}:
        raise RuntimeError(
            f"Cannot execute ordinary API calls while phase={state['phase']}"
        )
    paths = paths_for(args)
    attempt = dict(state["active_attempt"])
    if state["phase"] == "prepared":
        run_id = (
            f"direct-r{int(attempt['round']):03d}-"
            f"a{int(attempt['attempt']):02d}-{uuid.uuid4().hex[:12]}"
        )
        attempt_dir = Path(attempt["requests_path"]).parent
        attempt.update(
            {
                "execution_mode": "direct",
                "direct_run_id": run_id,
                "direct_started_at": utc_now(),
                "direct_completed_at": None,
                "batch_id": None,
                "batch_status": "running",
                "output_path": str(attempt_dir / "output.jsonl"),
                "response_dir": str(attempt_dir / "direct_responses"),
                "request_counts": {
                    "completed": 0,
                    "failed": 0,
                    "total": int(attempt["request_count"]),
                },
            }
        )
        state.setdefault("direct_history", []).append(
            {
                "round": int(attempt["round"]),
                "attempt": int(attempt["attempt"]),
                "run_id": run_id,
                "request_count": int(attempt["request_count"]),
                "started_at": attempt["direct_started_at"],
                "status": "running",
                "request_counts": dict(attempt["request_counts"]),
            }
        )
        update_state(
            paths["state"],
            state,
            phase="direct_running",
            active_attempt=attempt,
        )
    elif attempt.get("execution_mode") != "direct":
        raise RuntimeError("direct_running state lacks a direct active attempt.")

    request_rows = read_jsonl(Path(attempt["requests_path"]))
    expected = {str(row["custom_id"]): row for row in request_rows}
    if len(expected) != len(request_rows):
        raise RuntimeError("Prepared requests contain duplicate custom_id values.")
    response_dir = Path(attempt["response_dir"])
    response_dir.mkdir(parents=True, exist_ok=True)

    completed: set[str] = set()
    for custom_id in expected:
        response_path = response_dir / f"{custom_id}.json"
        if not response_path.exists():
            continue
        cached = read_json(response_path)
        if (
            cached.get("custom_id") == custom_id
            and not cached.get("error")
            and int(cached.get("response", {}).get("status_code") or 0) == 200
        ):
            completed.add(custom_id)

    api_client = client_for(args)
    output_lock = threading.Lock()
    progress_count = len(completed)
    last_checkpoint = time.monotonic()

    def execute_one(custom_id: str, request: Mapping[str, Any]) -> str:
        body = request.get("body")
        if not isinstance(body, Mapping):
            raise ValueError(f"{custom_id} request body is not an object.")
        last_error: Exception | None = None
        max_attempts = max(1, int(args.direct_max_attempts))
        for attempt_index in range(1, max_attempts + 1):
            try:
                response = api_client.chat.completions.create(**dict(body))
                row = _direct_response_row(custom_id, response)
                atomic_write_json(response_dir / f"{custom_id}.json", row)
                return custom_id
            except Exception as exc:
                last_error = exc
                if attempt_index == max_attempts:
                    break
                delay = min(
                    60.0,
                    max(0.1, float(args.direct_retry_seconds))
                    * (2 ** (attempt_index - 1)),
                )
                time.sleep(delay)
        assert last_error is not None
        raise RuntimeError(
            f"{custom_id} failed after {max_attempts} attempts: {last_error}"
        ) from last_error

    pending = [
        (custom_id, request)
        for custom_id, request in expected.items()
        if custom_id not in completed
    ]
    failures: list[str] = []
    workers = max(1, int(args.direct_concurrency))
    if pending:
        print(
            f"Direct round {attempt['round']}: resuming with "
            f"{len(completed)}/{len(expected)} cached, concurrency={workers}",
            flush=True,
        )
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="langmem-direct",
        ) as executor:
            futures = {
                executor.submit(execute_one, custom_id, request): custom_id
                for custom_id, request in pending
            }
            for future in concurrent.futures.as_completed(futures):
                custom_id = futures[future]
                try:
                    future.result()
                    with output_lock:
                        completed.add(custom_id)
                        progress_count += 1
                except Exception as exc:
                    failures.append(f"{custom_id}: {exc}")
                now = time.monotonic()
                if (
                    progress_count % 10 == 0
                    or now - last_checkpoint >= 30
                    or failures
                ):
                    attempt["request_counts"] = {
                        "completed": len(completed),
                        "failed": len(failures),
                        "total": len(expected),
                    }
                    history = _direct_history_entry(
                        state, str(attempt["direct_run_id"])
                    )
                    if history is not None:
                        history["request_counts"] = dict(
                            attempt["request_counts"]
                        )
                    update_state(
                        paths["state"], state, active_attempt=attempt
                    )
                    last_checkpoint = now
                    print(
                        f"Direct round {attempt['round']}: "
                        f"{len(completed)}/{len(expected)} completed, "
                        f"{len(failures)} failed",
                        flush=True,
                    )

    if failures:
        attempt["request_counts"] = {
            "completed": len(completed),
            "failed": len(failures),
            "total": len(expected),
        }
        history = _direct_history_entry(state, str(attempt["direct_run_id"]))
        if history is not None:
            history["request_counts"] = dict(attempt["request_counts"])
            history["last_error"] = failures[0]
        update_state(paths["state"], state, active_attempt=attempt)
        raise RuntimeError(
            f"{len(failures)} ordinary API calls failed; rerun the direct "
            f"command to resume. First failure: {failures[0]}"
        )

    output_rows = [
        read_json(response_dir / f"{custom_id}.json")
        for custom_id in expected
    ]
    output_path = Path(attempt["output_path"])
    write_jsonl(output_path, output_rows)
    attempt.update(
        {
            "batch_status": "completed",
            "direct_completed_at": utc_now(),
            "output_sha256": sha256_file(output_path),
            "request_counts": {
                "completed": len(expected),
                "failed": 0,
                "total": len(expected),
            },
        }
    )
    history = _direct_history_entry(state, str(attempt["direct_run_id"]))
    if history is not None:
        history.update(
            {
                "status": "completed",
                "completed_at": attempt["direct_completed_at"],
                "output_sha256": attempt["output_sha256"],
                "request_counts": dict(attempt["request_counts"]),
            }
        )
    update_state(
        paths["state"],
        state,
        phase="direct_completed",
        active_attempt=attempt,
    )
    print(
        f"Direct round {attempt['round']}: {len(expected)} ordinary API "
        "responses checkpointed",
        flush=True,
    )
    return read_json(paths["state"])


def refresh_batch_status(
    args: argparse.Namespace, state: dict[str, Any]
) -> dict[str, Any]:
    if state["phase"] != "submitted":
        return state
    paths = paths_for(args)
    attempt = dict(state["active_attempt"])
    batch = client_for(args).batches.retrieve(attempt["batch_id"])
    attempt.update(
        {
            "batch_status": batch.status,
            "output_file_id": batch.output_file_id,
            "error_file_id": batch.error_file_id,
            "request_counts": (
                batch.request_counts.model_dump()
                if batch.request_counts is not None
                else None
            ),
        }
    )
    if state["batch_history"]:
        state["batch_history"][-1].update(
            {
                "status": batch.status,
                "request_counts": attempt["request_counts"],
                "output_file_id": batch.output_file_id,
                "error_file_id": batch.error_file_id,
            }
        )
    update_state(paths["state"], state, active_attempt=attempt)
    print(
        f"Batch {batch.id}: status={batch.status}, "
        f"counts={attempt['request_counts']}",
        flush=True,
    )
    return state


def download_file(client: OpenAI, file_id: str, path: Path) -> None:
    content = client.files.content(file_id).content
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def parse_tool_arguments(tool_call: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    function = tool_call.get("function")
    if not isinstance(function, Mapping):
        raise ValueError("Tool call is missing function.")
    name = str(function.get("name", ""))
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(arguments, dict):
        raise ValueError(f"{name} arguments must be an object.")
    return name, arguments


def normalize_semantic_memory(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized, _ = normalize_semantic_memory_with_repairs(payload)
    return normalized


def normalize_semantic_memory_with_repairs(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    value = dict(payload)
    repairs: list[dict[str, Any]] = []
    category = value.get("category")
    allowed_categories = {
        "explicit_preference",
        "implicit_preference",
        "behavior_pattern",
        "api_outcome",
        "profile_fact",
    }
    packed_delimiters = (
        ("domain", "','domain':'"),
        ("slot", "','slot':'"),
        ("value", "','value':'"),
        ("context", "','context':'"),
    )
    if (
        isinstance(category, str)
        and category not in allowed_categories
        and all(delimiter in category for _, delimiter in packed_delimiters)
    ):
        remainder = category
        unpacked: dict[str, Any] = {}
        first_delimiter = packed_delimiters[0][1]
        unpacked["category"], remainder = remainder.split(first_delimiter, 1)
        for index, (field, _) in enumerate(packed_delimiters):
            if index + 1 < len(packed_delimiters):
                next_delimiter = packed_delimiters[index + 1][1]
                unpacked[field], remainder = remainder.split(next_delimiter, 1)
            else:
                unpacked[field] = remainder[:-1] if remainder.endswith("'") else remainder
        if unpacked["category"] in allowed_categories:
            for field, repaired_value in unpacked.items():
                value[field] = repaired_value
            repairs.append(
                {
                    "type": "unpacked_concatenated_semantic_fields",
                    "source_field": "category",
                    "fields": list(unpacked),
                }
            )
    value.setdefault("domain", None)
    value.setdefault("slot", None)
    value.setdefault("value", None)
    value.setdefault("context", None)
    validated = SemanticMemory.model_validate(value)
    return validated.model_dump(mode="json"), repairs


def deterministic_memory_key(custom_id: str, tool_call: Mapping[str, Any], index: int) -> str:
    tool_call_id = str(tool_call.get("id") or index)
    return uuid.uuid5(
        uuid.NAMESPACE_URL, f"experiment8-langmem:{custom_id}:{tool_call_id}:{index}"
    ).hex


def parse_batch_body(row: Mapping[str, Any]) -> dict[str, Any]:
    error = row.get("error")
    if error:
        raise RuntimeError(f"Batch row error: {canonical_json(error)}")
    response = row.get("response")
    if not isinstance(response, Mapping):
        raise RuntimeError("Batch row has no response.")
    status_code = int(response.get("status_code") or 0)
    if status_code != 200:
        raise RuntimeError(
            f"Batch row HTTP {status_code}: {canonical_json(response.get('body'))}"
        )
    body = response.get("body")
    if not isinstance(body, dict):
        raise RuntimeError("Batch response body is not an object.")
    return body


def plan_memory_operations(
    custom_id: str,
    body: Mapping[str, Any],
    manifest: Mapping[str, Any],
    current_items: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Response has no choices.")
    message = choices[0].get("message", {})
    tool_calls = message.get("tool_calls") or []
    if not isinstance(tool_calls, list):
        raise ValueError("message.tool_calls must be a list.")
    by_key = {str(item["key"]): dict(item) for item in current_items}
    retrieved = manifest.get("retrieved_memories") or []
    stable_to_key = {
        str(item["stable_id"]): str(item["key"])
        for item in retrieved
        if isinstance(item, Mapping)
    }
    resolved: dict[str, tuple[Any, ...]] = {}
    inserts: list[tuple[str, dict[str, Any], list[dict[str, Any]]]] = []
    call_repairs: list[dict[str, Any]] = []

    def resolve_stable_id(candidate: str) -> str | None:
        if candidate in stable_to_key:
            return candidate
        prefix_matches = [
            stable_id
            for stable_id in stable_to_key
            if stable_id.startswith(candidate) or candidate.startswith(stable_id)
        ]
        if len(prefix_matches) == 1:
            call_repairs.append(
                {
                    "type": "repaired_truncated_memory_id",
                    "original_json_doc_id": candidate,
                    "resolved_json_doc_id": prefix_matches[0],
                }
            )
            return prefix_matches[0]
        return None

    for index, tool_call in enumerate(tool_calls):
        if not isinstance(tool_call, Mapping):
            call_repairs.append(
                {
                    "type": "ignored_non_object_tool_call",
                    "tool_call_index": index,
                }
            )
            continue
        try:
            name, arguments = parse_tool_arguments(tool_call)
        except Exception as exc:
            call_repairs.append(
                {
                    "type": "ignored_unparseable_tool_call",
                    "tool_call_id": tool_call.get("id"),
                    "error": str(exc),
                }
            )
            continue
        if name == "SemanticMemory" and not {
            "content",
            "category",
        }.issubset(arguments):
            if {
                "json_doc_id",
                "patches",
            }.issubset(arguments):
                call_repairs.append(
                    {
                        "type": "reclassified_semantic_memory_as_patch_doc",
                        "tool_call_id": tool_call.get("id"),
                    }
                )
                name = "PatchDoc"
            elif (
                str(arguments.get("recipient_name", "")).endswith("RemoveDoc")
                and isinstance(arguments.get("parameters"), Mapping)
            ):
                call_repairs.append(
                    {
                        "type": "reclassified_nested_remove_doc",
                        "tool_call_id": tool_call.get("id"),
                    }
                )
                name = "RemoveDoc"
                arguments = dict(arguments["parameters"])
            elif "json_doc_id" in arguments:
                call_repairs.append(
                    {
                        "type": "ignored_malformed_semantic_memory_reference",
                        "tool_call_id": tool_call.get("id"),
                        "json_doc_id": arguments.get("json_doc_id"),
                    }
                )
                continue
        if name == "SemanticMemory":
            try:
                normalized, repairs = normalize_semantic_memory_with_repairs(
                    arguments
                )
            except Exception as exc:
                call_repairs.append(
                    {
                        "type": "ignored_invalid_semantic_memory",
                        "tool_call_id": tool_call.get("id"),
                        "error": str(exc),
                    }
                )
                continue
            inserts.append(
                (
                    deterministic_memory_key(custom_id, tool_call, index),
                    normalized,
                    repairs,
                )
            )
        elif name == "PatchDoc":
            original_stable_id = str(arguments.get("json_doc_id", ""))
            stable_id = resolve_stable_id(original_stable_id)
            if stable_id is None:
                stable_id = original_stable_id
            memory_key = stable_to_key.get(stable_id)
            if memory_key is None and stable_id in {
                "existing",
                "memory",
                "memory_db",
                "memory_store",
            }:
                call_repairs.append(
                    {
                        "type": "ignored_aggregate_patch_unknown_target",
                        "tool_call_id": tool_call.get("id"),
                        "json_doc_id": stable_id,
                        "reason": (
                            "LangMem PatchDoc operates on one retrieved document; "
                            "aggregate container targets are invalid. Valid sibling "
                            "SemanticMemory/RemoveDoc calls remain applied."
                        ),
                    }
                )
                continue
            if memory_key is None or memory_key not in by_key:
                call_repairs.append(
                    {
                        "type": "ignored_patch_unknown_target",
                        "tool_call_id": tool_call.get("id"),
                        "json_doc_id": original_stable_id,
                    }
                )
                continue
            patches = arguments.get("patches")
            if not isinstance(patches, list):
                call_repairs.append(
                    {
                        "type": "ignored_invalid_patch",
                        "tool_call_id": tool_call.get("id"),
                        "json_doc_id": original_stable_id,
                        "error": "PatchDoc.patches must be a list.",
                    }
                )
                continue
            base_content = by_key[memory_key]["value"]["content"]
            try:
                patched = jsonpatch.JsonPatch(patches).apply(
                    base_content, in_place=False
                )
                normalized, repairs = normalize_semantic_memory_with_repairs(
                    patched
                )
            except Exception as exc:
                call_repairs.append(
                    {
                        "type": "ignored_invalid_patch",
                        "tool_call_id": tool_call.get("id"),
                        "json_doc_id": original_stable_id,
                        "error": str(exc),
                    }
                )
                continue
            resolved[stable_id] = (
                "update",
                normalized,
                repairs,
            )
        elif name == "RemoveDoc":
            original_stable_id = str(arguments.get("json_doc_id", ""))
            stable_id = resolve_stable_id(original_stable_id)
            if stable_id is None:
                call_repairs.append(
                    {
                        "type": "ignored_remove_unknown_target",
                        "tool_call_id": tool_call.get("id"),
                        "json_doc_id": original_stable_id,
                    }
                )
                continue
            resolved[stable_id] = ("delete", None)
        else:
            call_repairs.append(
                {
                    "type": "ignored_unexpected_tool",
                    "tool_call_id": tool_call.get("id"),
                    "tool_name": name,
                }
            )

    operations: list[dict[str, Any]] = []
    for stable_id, resolved_value in resolved.items():
        operation, content, *repair_values = resolved_value
        repairs = repair_values[0] if repair_values else []
        memory_key = stable_to_key[stable_id]
        before = item_without_embedding(by_key[memory_key])
        if operation == "delete":
            operations.append(
                {
                    "operation": "delete",
                    "memory_key": memory_key,
                    "before": before,
                    "after": None,
                }
            )
            continue
        assert content is not None
        after = dict(before)
        after["value"] = {"kind": "SemanticMemory", "content": content}
        if after["value"] != before["value"]:
            operations.append(
                {
                    "operation": "update",
                    "memory_key": memory_key,
                    "before": before,
                    "after": after,
                    **(
                        {"normalization_repairs": repairs}
                        if repairs
                        else {}
                    ),
                }
            )
    for memory_key, content, repairs in inserts:
        operations.append(
            {
                "operation": "insert",
                "memory_key": memory_key,
                "before": None,
                "after": {
                    "namespace": list(
                        materialize_namespace(str(manifest["example_id"]))
                    ),
                    "key": memory_key,
                    "value": {"kind": "SemanticMemory", "content": content},
                    "created_at": None,
                    "updated_at": None,
                    "score": None,
                    "construction_provenance": {
                        "created_session_index": int(
                            manifest["session_index"]
                        ),
                        "updated_session_index": int(
                            manifest["session_index"]
                        ),
                    },
                },
                **(
                    {"normalization_repairs": repairs}
                    if repairs
                    else {}
                ),
            }
        )
    if call_repairs:
        if operations:
            operations[0].setdefault("normalization_repairs", []).extend(
                call_repairs
            )
        else:
            operations.append(
                {
                    "operation": "noop",
                    "memory_key": "__normalization_only__",
                    "before": None,
                    "after": None,
                    "normalization_repairs": call_repairs,
                }
            )
    return operations


def apply_operations_to_items(
    current_items: Sequence[Mapping[str, Any]],
    operations: Sequence[Mapping[str, Any]],
    event_time: str,
    session_index: int,
) -> list[dict[str, Any]]:
    items = {
        str(item["key"]): item_without_embedding(item) for item in current_items
    }
    for operation in operations:
        memory_key = str(operation["memory_key"])
        op = operation["operation"]
        if op == "noop":
            continue
        if op == "delete":
            items.pop(memory_key, None)
            continue
        after = dict(operation["after"])
        existing = items.get(memory_key)
        if existing:
            after["created_at"] = existing["created_at"]
            after["construction_provenance"]["created_session_index"] = existing[
                "construction_provenance"
            ]["created_session_index"]
        else:
            after["created_at"] = event_time
        after["updated_at"] = event_time
        after["construction_provenance"][
            "updated_session_index"
        ] = session_index
        items[memory_key] = after
        operation["after"].update(after)
    return [items[key] for key in sorted(items)]


def usage_call_from_response(
    body: Mapping[str, Any],
    session_index: int,
    model: str,
    provider: str,
    request_group_id: str,
) -> dict[str, Any]:
    call = {
        "component": "memory_manager",
        "session_index": session_index,
        "provider": provider,
        "model": model,
        **usage_from_response(dict(body)),
    }
    if provider == "openai_batch":
        call["batch_id"] = request_group_id
    else:
        call["direct_run_id"] = request_group_id
    return call


def write_memory_to_db(
    connection: sqlite3.Connection,
    example_id: str,
    operation: Mapping[str, Any],
    embedding: np.ndarray | None,
    event_time: str,
    session_index: int,
) -> None:
    memory_key = str(operation["memory_key"])
    if operation["operation"] == "noop":
        return
    if operation["operation"] == "delete":
        connection.execute(
            "DELETE FROM memories WHERE example_id = ? AND memory_key = ?",
            (example_id, memory_key),
        )
        return
    if embedding is None:
        raise RuntimeError(f"Missing embedding for {example_id}/{memory_key}")
    after = operation["after"]
    provenance = after["construction_provenance"]
    connection.execute(
        """
        INSERT INTO memories(
            example_id, memory_key, value_json, embedding,
            created_at, updated_at, created_session_index, updated_session_index
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(example_id, memory_key) DO UPDATE SET
            value_json = excluded.value_json,
            embedding = excluded.embedding,
            updated_at = excluded.updated_at,
            updated_session_index = excluded.updated_session_index
        """,
        (
            example_id,
            memory_key,
            canonical_json(after["value"]),
            embedding.astype(np.float32).tobytes(),
            str(after.get("created_at") or event_time),
            str(after.get("updated_at") or event_time),
            int(provenance["created_session_index"]),
            int(provenance["updated_session_index"]),
        ),
    )


def collect_round(args: argparse.Namespace, state: dict[str, Any]) -> dict[str, Any]:
    if state["phase"] not in {"submitted", "direct_completed"}:
        raise RuntimeError(f"Cannot collect while phase={state['phase']}")
    paths = paths_for(args)
    attempt = dict(state["active_attempt"])
    execution_mode = str(attempt.get("execution_mode") or "batch")
    if execution_mode == "batch":
        state = refresh_batch_status(args, state)
        attempt = dict(state["active_attempt"])
    elif execution_mode != "direct":
        raise RuntimeError(f"Unsupported execution mode: {execution_mode}")
    status = attempt["batch_status"]
    if status != "completed":
        if status in {"failed", "expired", "cancelled", "cancelling"}:
            raise RuntimeError(f"Batch ended without completion: {status}")
        return state
    if execution_mode == "batch" and not attempt.get("output_file_id"):
        raise RuntimeError("Completed batch has no output_file_id.")
    attempt_dir = Path(attempt["requests_path"]).parent
    output_path = Path(
        attempt.get("output_path") or (attempt_dir / "output.jsonl")
    )
    error_path = attempt_dir / "errors.jsonl"
    api_client = client_for(args)
    if execution_mode == "batch" and not output_path.exists():
        download_file(api_client, attempt["output_file_id"], output_path)
    if (
        execution_mode == "batch"
        and attempt.get("error_file_id")
        and not error_path.exists()
    ):
        download_file(api_client, attempt["error_file_id"], error_path)
    if not output_path.exists():
        raise RuntimeError(
            f"{execution_mode} response file does not exist: {output_path}"
        )
    output_rows = read_jsonl(output_path)
    output_by_id = {str(row["custom_id"]): row for row in output_rows}
    manifest_rows = read_jsonl(Path(attempt["request_manifest_path"]))
    expected_ids = {str(row["custom_id"]) for row in manifest_rows}
    missing = sorted(expected_ids - set(output_by_id))
    if missing:
        error_note = (
            f"; see {error_path}" if error_path.exists() else ""
        )
        raise RuntimeError(
            f"Batch output is missing {len(missing)} requests{error_note}: "
            f"{missing[:5]}"
        )

    connection = connect_db(paths["db"])
    encoding = get_encoding()
    parsed: list[dict[str, Any]] = []
    document_inputs: list[tuple[str, str]] = []
    document_sessions: dict[str, int] = {}
    try:
        for manifest in manifest_rows:
            custom_id = str(manifest["custom_id"])
            example_id = str(manifest["example_id"])
            body = parse_batch_body(output_by_id[custom_id])
            current_items = load_memories(connection, example_id)
            operations = plan_memory_operations(
                custom_id=custom_id,
                body=body,
                manifest=manifest,
                current_items=current_items,
            )
            event_time = utc_now()
            after_items = apply_operations_to_items(
                current_items=current_items,
                operations=operations,
                event_time=event_time,
                session_index=int(manifest["session_index"]),
            )
            for operation in operations:
                if operation["operation"] in {"delete", "noop"}:
                    continue
                label = f"{custom_id}::{operation['memory_key']}"
                document_inputs.append(
                    (
                        label,
                        embedding_document_text(operation["after"]["value"]),
                    )
                )
                document_sessions[label] = int(manifest["session_index"])
                operation["_embedding_label"] = label
            parsed.append(
                {
                    "manifest": manifest,
                    "body": body,
                    "operations": operations,
                    "event_time": event_time,
                    "before_items": current_items,
                    "after_items": after_items,
                }
            )

        document_vectors: dict[str, np.ndarray] = {}
        document_usage: dict[str, dict[str, Any]] = {}
        document_api_usage: list[dict[str, Any]] = []
        if document_inputs:
            (
                document_vectors,
                document_usage,
                document_api_usage,
            ) = embed_labeled_texts(
                client=api_client,
                model=args.embedding_model,
                labeled_texts=document_inputs,
                encoding=encoding,
                chunk_size=args.embedding_chunk_size,
                component="embedding_document",
                session_indexes=document_sessions,
            )
        state["embedding_api_usage"].extend(document_api_usage)

        with connection:
            for result in parsed:
                manifest = result["manifest"]
                example_id = str(manifest["example_id"])
                session_index = int(manifest["session_index"])
                user = connection.execute(
                    "SELECT * FROM users WHERE example_id = ?",
                    (example_id,),
                ).fetchone()
                if user is None:
                    raise RuntimeError(f"Unknown example_id {example_id}")
                if int(user["next_session_index"]) != session_index:
                    raise RuntimeError(
                        f"Checkpoint mismatch for {example_id}: "
                        f"next={user['next_session_index']} round={session_index}"
                    )
                doc_calls: list[dict[str, Any]] = []
                for operation in result["operations"]:
                    label = operation.pop("_embedding_label", None)
                    embedding = document_vectors.get(label) if label else None
                    if label:
                        doc_calls.append(document_usage[label])
                    write_memory_to_db(
                        connection=connection,
                        example_id=example_id,
                        operation=operation,
                        embedding=embedding,
                        event_time=result["event_time"],
                        session_index=session_index,
                    )

                request_group_id = (
                    str(attempt["batch_id"])
                    if execution_mode == "batch"
                    else str(attempt["direct_run_id"])
                )
                chat_call = usage_call_from_response(
                    body=result["body"],
                    session_index=session_index,
                    model=args.model,
                    provider=(
                        "openai_batch"
                        if execution_mode == "batch"
                        else "openai"
                    ),
                    request_group_id=request_group_id,
                )
                calls = [
                    dict(manifest["embedding_query_usage"]),
                    chat_call,
                    *doc_calls,
                ]
                usage_report = build_usage_report(calls)
                summary = usage_report["summary"]
                before_tokens = int(
                    manifest["stored_memory_tokens_before_session"]
                )
                after_tokens = stored_memory_tokens(
                    result["after_items"], encoding
                )
                after_count = len(result["after_items"])
                cumulative = {
                    field: int(user[f"cumulative_{field}"])
                    + int(summary[field])
                    for field in TOKEN_FIELDS
                }
                snapshot_hash = sha256_json(result["after_items"])
                normalization_repairs = [
                    {
                        "memory_key": operation["memory_key"],
                        **repair,
                    }
                    for operation in result["operations"]
                    for repair in operation.get("normalization_repairs", [])
                ]
                payload = {
                    "session_index": session_index,
                    "dialogue_id": manifest.get("dialogue_id"),
                    "status": "OK",
                    "last_user_utterance": manifest.get(
                        "last_user_utterance", ""
                    ),
                    "session_input_tokens": int(
                        manifest["session_input_tokens"]
                    ),
                    "memory_count_before_session": int(
                        manifest["memory_count_before_session"]
                    ),
                    "memory_count_after_session": after_count,
                    "memory_count_delta": after_count
                    - int(manifest["memory_count_before_session"]),
                    "stored_memory_tokens_before_session": before_tokens,
                    "stored_memory_tokens_after_session": after_tokens,
                    "stored_memory_delta_tokens": after_tokens - before_tokens,
                    "memory_operations": result["operations"],
                    "response_normalization_repairs": normalization_repairs,
                    "response_normalization_repair_count": len(
                        normalization_repairs
                    ),
                    "memory_snapshot_sha256": snapshot_hash,
                    "construction_token_usage": usage_report,
                    "cumulative_construction_token_usage": cumulative,
                    "cumulative_memory_token_area": int(
                        user["cumulative_memory_token_area"]
                    )
                    + after_tokens,
                    "construction_round": int(attempt["round"]),
                    "execution_mode": execution_mode,
                    "request_group_id": request_group_id,
                    "reasoning_effort": args.reasoning_effort,
                }
                if execution_mode == "batch":
                    payload["batch_round"] = int(attempt["round"])
                    payload["batch_id"] = str(attempt["batch_id"])
                else:
                    payload["direct_run_id"] = str(
                        attempt["direct_run_id"]
                    )
                connection.execute(
                    """
                    INSERT INTO session_exports(example_id, session_index, payload_json)
                    VALUES (?, ?, ?)
                    """,
                    (example_id, session_index, canonical_json(payload)),
                )
                connection.execute(
                    """
                    UPDATE users SET
                        next_session_index = next_session_index + 1,
                        cumulative_input_tokens = ?,
                        cumulative_cached_input_tokens = ?,
                        cumulative_output_tokens = ?,
                        cumulative_reasoning_tokens = ?,
                        cumulative_total_tokens = ?,
                        cumulative_memory_token_area = ?,
                        peak_memory_count = MAX(peak_memory_count, ?),
                        peak_stored_memory_tokens = MAX(peak_stored_memory_tokens, ?)
                    WHERE example_id = ?
                    """,
                    (
                        cumulative["input_tokens"],
                        cumulative["cached_input_tokens"],
                        cumulative["output_tokens"],
                        cumulative["reasoning_tokens"],
                        cumulative["total_tokens"],
                        payload["cumulative_memory_token_area"],
                        after_count,
                        after_tokens,
                        example_id,
                    ),
                )
    finally:
        connection.close()

    attempt["collected_at"] = utc_now()
    attempt["output_path"] = str(output_path)
    attempt["output_sha256"] = sha256_file(output_path)
    if execution_mode == "batch" and state["batch_history"]:
        state["batch_history"][-1]["collected_at"] = attempt["collected_at"]
        state["batch_history"][-1]["output_sha256"] = attempt["output_sha256"]
    elif execution_mode == "direct":
        history = _direct_history_entry(
            state, str(attempt["direct_run_id"])
        )
        if history is not None:
            history["collected_at"] = attempt["collected_at"]
            history["status"] = "collected"
    next_round = int(state["current_round"]) + 1
    update_state(
        paths["state"],
        state,
        phase="ready",
        current_round=next_round,
        active_attempt=None,
    )
    print(
        f"Collected round {attempt['round']}: "
        f"{len(manifest_rows)} sessions applied",
        flush=True,
    )
    connection = connect_db(paths["db"])
    try:
        remaining = connection.execute(
            """
            SELECT COUNT(*) FROM users
            WHERE next_session_index <= session_count
            """
        ).fetchone()[0]
    finally:
        connection.close()
    if not remaining:
        finalize(args, state)
    return read_json(paths["state"])


def session_rows(
    connection: sqlite3.Connection, example_id: str
) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT payload_json FROM session_exports
        WHERE example_id = ?
        ORDER BY session_index
        """,
        (example_id,),
    ).fetchall()
    return [json.loads(row["payload_json"]) for row in rows]


def memory_mode_from_calls(calls: Sequence[Mapping[str, Any]]) -> str:
    providers = {
        str(call.get("provider"))
        for call in calls
        if call.get("component") == "memory_manager" and call.get("provider")
    }
    if providers == {"openai_batch"}:
        return "langmem_openai_batch"
    if providers == {"openai"}:
        return "langmem_openai_direct"
    if providers:
        return "langmem_openai_mixed"
    return "langmem_openai"


def build_final_record(
    connection: sqlite3.Connection, user: sqlite3.Row
) -> dict[str, Any]:
    example_id = str(user["example_id"])
    memories = [
        item_without_embedding(item)
        for item in load_memories(connection, example_id)
    ]
    sessions = session_rows(connection, example_id)
    calls = [
        call
        for session in sessions
        for call in session.get("construction_token_usage", {}).get("calls", [])
    ]
    report = build_usage_report(calls)
    return {
        "example_id": example_id,
        "namespace": list(materialize_namespace(example_id)),
        "memory_items": memories,
        "session_exports": sessions,
        "method": "langmem",
        "memory_mode": memory_mode_from_calls(calls),
        "memory_model": "gpt-5-mini",
        "reasoning_effort": "minimal",
        "token_encoding": "cl100k_base",
        "construction_token_usage": report,
        "token_counts": report["summary"],
        "memory_accumulation": {
            "sessions_processed": len(sessions),
            "final_memory_count": len(memories),
            "peak_memory_count": int(user["peak_memory_count"]),
            "peak_stored_memory_tokens": int(
                user["peak_stored_memory_tokens"]
            ),
            "cumulative_memory_token_area": int(
                user["cumulative_memory_token_area"]
            ),
            "event_source": "session_exports[].memory_operations",
        },
    }


def finalize(args: argparse.Namespace, state: dict[str, Any]) -> dict[str, Any]:
    paths = paths_for(args)
    connection = connect_db(paths["db"])
    try:
        remaining = connection.execute(
            """
            SELECT COUNT(*) FROM users
            WHERE next_session_index <= session_count
            """
        ).fetchone()[0]
        if remaining:
            raise RuntimeError(
                f"Cannot finalize: {remaining} users still have pending sessions."
            )
        users = connection.execute(
            "SELECT * FROM users ORDER BY dataset_index"
        ).fetchall()
        records = [build_final_record(connection, user) for user in users]
        accumulation_rows = [
            {
                "example_id": record["example_id"],
                **session,
            }
            for record in records
            for session in record["session_exports"]
        ]
    finally:
        connection.close()
    write_jsonl(paths["artifact"], records)
    write_jsonl(paths["accumulation"], accumulation_rows)
    aggregate_calls = [
        call
        for record in records
        for call in record["construction_token_usage"]["calls"]
    ]
    aggregate_report = build_usage_report(aggregate_calls)
    aggregate_memory_mode = memory_mode_from_calls(aggregate_calls)
    provider = {
        "langmem_openai_batch": "openai_batch",
        "langmem_openai_direct": "openai",
        "langmem_openai_mixed": "openai_mixed_batch_direct",
    }.get(aggregate_memory_mode, "openai")
    manifest = {
        "created_at": now_iso(),
        "method": "langmem",
        "artifact_type": "construction_manifest",
        "artifact_path": str(paths["artifact"]),
        "artifact_sha256": sha256_file(paths["artifact"]),
        "memory_accumulation_path": str(paths["accumulation"]),
        "memory_accumulation_sha256": sha256_file(paths["accumulation"]),
        "state_db_path": str(paths["db"]),
        "input_path": state["input_path"],
        "input_sha256": state["input_sha256"],
        "examples_processed": len(records),
        "sessions_processed": len(accumulation_rows),
        "memory_model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "provider": provider,
        "embedding_model": args.embedding_model,
        "construction_token_usage": aggregate_report,
        "embedding_api_usage": state["embedding_api_usage"],
        "batch_history": state["batch_history"],
        "direct_history": state.get("direct_history", []),
        "ablation_contract": {
            "memory_accumulation": (
                "Replay session_exports[].memory_operations in session order."
            ),
            "construction_tokens": (
                "Use cumulative_construction_token_usage.total_tokens as a "
                "session-boundary budget."
            ),
            "materialize_command": (
                "python methods/langmem/build_memory_batch.py materialize "
                "--output_dir <dir> --materialized_output <path> "
                "[--max_sessions N] "
                "[--max_cumulative_construction_tokens N] "
                "[--max_stored_memory_tokens N]"
            ),
        },
    }
    atomic_write_json(paths["manifest"], manifest)
    update_state(
        paths["state"],
        state,
        phase="complete",
        active_attempt=None,
        artifact_path=str(paths["artifact"]),
        manifest_path=str(paths["manifest"]),
    )
    print(
        f"Finalized {len(records)} examples / {len(accumulation_rows)} sessions: "
        f"{paths['artifact']}",
        flush=True,
    )
    return read_json(paths["state"])


def replay_operations(
    memories: dict[str, dict[str, Any]],
    operations: Sequence[Mapping[str, Any]],
) -> None:
    for operation in operations:
        key = str(operation["memory_key"])
        if operation["operation"] == "noop":
            continue
        if operation["operation"] == "delete":
            memories.pop(key, None)
        else:
            memories[key] = dict(operation["after"])


def materialize_ablation(args: argparse.Namespace) -> None:
    if not args.materialized_output:
        raise RuntimeError("--materialized_output is required for materialize.")
    if (
        args.max_sessions is None
        and args.max_cumulative_construction_tokens is None
        and args.max_stored_memory_tokens is None
    ):
        raise RuntimeError("At least one ablation limit is required.")
    paths = paths_for(args)
    connection = connect_db(paths["db"])
    records: list[dict[str, Any]] = []
    try:
        users = connection.execute(
            "SELECT * FROM users ORDER BY dataset_index"
        ).fetchall()
        for user in users:
            example_id = str(user["example_id"])
            memories: dict[str, dict[str, Any]] = {}
            included_sessions: list[dict[str, Any]] = []
            for session in session_rows(connection, example_id):
                session_index = int(session["session_index"])
                cumulative_tokens = int(
                    session.get("cumulative_construction_token_usage", {}).get(
                        "total_tokens", 0
                    )
                )
                stored_tokens = session.get(
                    "stored_memory_tokens_after_session"
                )
                if (
                    args.max_sessions is not None
                    and session_index > args.max_sessions
                ):
                    break
                if (
                    args.max_cumulative_construction_tokens is not None
                    and cumulative_tokens
                    > args.max_cumulative_construction_tokens
                ):
                    break
                if (
                    args.max_stored_memory_tokens is not None
                    and isinstance(stored_tokens, int)
                    and stored_tokens > args.max_stored_memory_tokens
                ):
                    break
                replay_operations(
                    memories, session.get("memory_operations", [])
                )
                included_sessions.append(session)
            calls = [
                call
                for session in included_sessions
                for call in session.get("construction_token_usage", {}).get(
                    "calls", []
                )
            ]
            report = build_usage_report(calls)
            records.append(
                {
                    "example_id": example_id,
                    "namespace": list(materialize_namespace(example_id)),
                    "memory_items": [
                        memories[key] for key in sorted(memories)
                    ],
                    "session_exports": included_sessions,
                    "method": "langmem",
                    "memory_mode": (
                        f"{memory_mode_from_calls(calls)}_ablation"
                    ),
                    "memory_model": args.model,
                    "reasoning_effort": args.reasoning_effort,
                    "token_encoding": "cl100k_base",
                    "construction_token_usage": report,
                    "token_counts": report["summary"],
                    "ablation": {
                        "max_sessions": args.max_sessions,
                        "max_cumulative_construction_tokens": (
                            args.max_cumulative_construction_tokens
                        ),
                        "max_stored_memory_tokens": (
                            args.max_stored_memory_tokens
                        ),
                        "sessions_included": len(included_sessions),
                    },
                }
            )
    finally:
        connection.close()
    output_path = Path(args.materialized_output).resolve()
    write_jsonl(output_path, records)
    print(f"Materialized {len(records)} records at {output_path}", flush=True)


def local_status(args: argparse.Namespace, state: Mapping[str, Any]) -> dict[str, Any]:
    paths = paths_for(args)
    connection = connect_db(paths["db"])
    try:
        completed_sessions = connection.execute(
            "SELECT COUNT(*) FROM session_exports"
        ).fetchone()[0]
        completed_users = connection.execute(
            """
            SELECT COUNT(*) FROM users
            WHERE next_session_index > session_count
            """
        ).fetchone()[0]
        memories = connection.execute(
            "SELECT COUNT(*) FROM memories"
        ).fetchone()[0]
    finally:
        connection.close()
    return {
        "phase": state["phase"],
        "current_round": state["current_round"],
        "examples": state["examples"],
        "completed_users": completed_users,
        "sessions": state["sessions"],
        "completed_sessions": completed_sessions,
        "memories": memories,
        "active_attempt": state.get("active_attempt"),
    }


def run_loop(args: argparse.Namespace, state: dict[str, Any]) -> None:
    poll_seconds = max(5, min(60, int(args.poll_seconds)))
    while True:
        phase = state["phase"]
        if phase == "complete":
            print(canonical_json(local_status(args, state)), flush=True)
            return
        if phase == "ready":
            state = prepare_round(args, state)
            continue
        if phase == "prepared":
            state = submit_round(args, state)
            continue
        if phase == "submitted":
            state = refresh_batch_status(args, state)
            status = state["active_attempt"]["batch_status"]
            if status == "completed":
                state = collect_round(args, state)
                continue
            if status in {"failed", "expired", "cancelled", "cancelling"}:
                raise RuntimeError(f"Batch ended without completion: {status}")
            time.sleep(poll_seconds)
            continue
        raise RuntimeError(f"Unsupported state phase: {phase}")


def run_direct_loop(args: argparse.Namespace, state: dict[str, Any]) -> None:
    """Resume construction without creating any new OpenAI Batch jobs."""
    while True:
        phase = state["phase"]
        if phase == "complete":
            print(canonical_json(local_status(args, state)), flush=True)
            return
        if phase == "ready":
            state = prepare_round(args, state)
            continue
        if phase in {"prepared", "direct_running"}:
            state = execute_round_direct(args, state)
            continue
        if phase == "direct_completed":
            state = collect_round(args, state)
            continue
        if phase == "submitted":
            # A Batch job submitted before direct mode was requested may still
            # be collected, but this path never creates another Batch job.
            state = collect_round(args, state)
            continue
        raise RuntimeError(f"Unsupported direct state phase: {phase}")


def main() -> None:
    args = parse_args()
    if args.command == "materialize":
        materialize_ablation(args)
        return
    state = initialize(args)
    paths = paths_for(args)
    if args.command == "run":
        run_loop(args, state)
    elif args.command == "direct":
        run_direct_loop(args, state)
    elif args.command == "status":
        if state["phase"] == "submitted":
            state = refresh_batch_status(args, state)
        print(json.dumps(local_status(args, state), indent=2), flush=True)
    elif args.command == "prepare":
        prepare_round(args, state)
    elif args.command == "submit":
        submit_round(args, state)
    elif args.command == "collect":
        collect_round(args, state)
    elif args.command == "finalize":
        finalize(args, state)


if __name__ == "__main__":
    main()
