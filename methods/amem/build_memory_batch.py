"""Build Experiment8 A-MEM artifacts with causal OpenAI Batch stages.

This is an Experiment8 adapter around the official ``WujiangXu/A-mem`` flow:

1. each dialogue turn becomes one immutable MemoryNote;
2. ``analyze_content`` generates keywords, context, and tags;
3. the raw turn retrieves five older notes;
4. ``process_memory`` strengthens the new note and/or updates neighbors;
5. the committed note and evolved neighbors are embedded again.

Requests at the same causal note index are batched across users.  The two LLM
calls used by the official implementation remain separate Batch jobs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from openai import OpenAI


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.amem.common import (  # noqa: E402
    DEFAULT_CONSTRUCTION_TOP_K,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    STATE_VERSION,
    TOKEN_FIELDS,
    UPSTREAM_COMMIT,
    UPSTREAM_REPOSITORY,
    apply_evolution_result,
    assign_embeddings,
    atomic_write_json,
    atomic_write_jsonl,
    canonical_json,
    create_pending_note,
    extract_batch_body,
    index_unique_rows,
    load_dataset,
    make_embedder,
    nearest_notes,
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
    ROOT / "outputs" / "amem" / "MPT_v2_0725_gpt-5-mini_batch"
)
TERMINAL_BATCH_STATUSES = {
    "failed",
    "expired",
    "cancelled",
}

METADATA_RESPONSE_SCHEMA = {
    "name": "amem_note_metadata",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "keywords": {
                "type": "array",
                "items": {"type": "string"},
            },
            "context": {"type": "string"},
            "tags": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["keywords", "context", "tags"],
        "additionalProperties": False,
    },
}

EVOLUTION_RESPONSE_SCHEMA = {
    "name": "amem_note_evolution",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "should_evolve": {"type": "boolean"},
            "actions": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["strengthen", "update_neighbor"],
                },
            },
            "suggested_connections": {
                "type": "array",
                "items": {"type": "string"},
            },
            "tags_to_update": {
                "type": "array",
                "items": {"type": "string"},
            },
            "new_context_neighborhood": {
                "type": "array",
                "items": {"type": "string"},
            },
            "new_tags_neighborhood": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        },
        "required": [
            "should_evolve",
            "actions",
            "suggested_connections",
            "tags_to_update",
            "new_context_neighborhood",
            "new_tags_neighborhood",
        ],
        "additionalProperties": False,
    },
}

METADATA_SYSTEM_PROMPT = """Generate a structured analysis of one memory note.

1. Identify the most salient keywords. Focus on nouns, verbs, and key
   concepts. Do not use the speaker name or causal index as a keyword.
2. Write one concise context sentence covering the main topic, key point,
   and future purpose of the note.
3. Create several broad categorical tags covering domain and memory type.

Ground every field in the immutable note content. Do not infer an unstated
preference or API argument. Return the exact JSON schema requested."""

EVOLUTION_SYSTEM_PROMPT = """You are an A-MEM memory evolution agent.

Analyze the new memory note and its nearest older notes. Decide whether the
new note should evolve. Allowed actions:

* strengthen: connect the new note to meaningful candidate note IDs and
  return the complete updated tag list for the new note.
* update_neighbor: return context and tag lists for every candidate in the
  exact candidate order. If a neighbor should not change, copy its original
  context and tags unchanged.

Only candidate note IDs may appear in suggested_connections. Connections are
directed from the new note to older notes. Do not rewrite source content,
invent facts, or erase contradictions. If should_evolve is false, return empty
actions/connections/tags_to_update and copy candidate metadata into the two
neighborhood arrays. Return the exact JSON schema requested."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("run", "prepare", "submit", "status", "collect", "finalize"),
        nargs="?",
        default="run",
    )
    parser.add_argument("--input_path", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--reasoning_effort", default=DEFAULT_REASONING_EFFORT
    )
    parser.add_argument(
        "--embedding_model", default=DEFAULT_EMBEDDING_MODEL
    )
    parser.add_argument(
        "--top_k", type=int, default=DEFAULT_CONSTRUCTION_TOP_K
    )
    parser.add_argument("--max_completion_tokens", type=int, default=1536)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--max_sessions", type=int, default=None)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--poll_seconds", type=int, default=60)
    parser.add_argument(
        "--raw_results",
        default=None,
        help="Collect the active stage from a local Batch-result JSONL file.",
    )
    parser.add_argument("--force_init", action="store_true")
    return parser.parse_args()


def paths_for(args: argparse.Namespace) -> dict[str, Path]:
    output_dir = Path(args.output_dir).resolve()
    return {
        "output_dir": output_dir,
        "state": output_dir / "state.json",
        "working": output_dir / "working_memory.json",
        "artifact": output_dir / "memory.jsonl",
        "summary": output_dir / "construction_summary.json",
    }


def stage_dir(output_dir: Path, note_index: int, stage: str) -> Path:
    return (
        output_dir
        / "rounds"
        / f"note_{note_index:04d}"
        / stage
    )


def require_api_key(args: argparse.Namespace) -> str:
    value = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not value:
        raise RuntimeError("Pass --api_key or set OPENAI_API_KEY.")
    return value


def client_for(args: argparse.Namespace) -> OpenAI:
    return OpenAI(api_key=require_api_key(args), timeout=120.0)


def custom_id_for(
    dataset_index: int, note_index: int, stage: str
) -> str:
    suffix = "m" if stage == "metadata" else "e"
    return f"amem-i{dataset_index:04d}-n{note_index:04d}-{suffix}"


def note_id_for(
    dataset_index: int, note_unit: Mapping[str, Any]
) -> str:
    return (
        f"amem-i{dataset_index:04d}"
        f"-s{int(note_unit['session_index']):03d}"
        f"-t{int(note_unit['turn_index']):03d}"
    )


def note_units_for_user(
    rows: Sequence[Mapping[str, Any]],
    user: Mapping[str, Any],
    *,
    max_sessions: int | None,
) -> list[dict[str, Any]]:
    return turn_note_units(
        rows[int(user["dataset_index"])],
        max_sessions=max_sessions,
    )


def initialize(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = paths_for(args)
    if paths["state"].exists() and not args.force_init:
        state = read_json(paths["state"])
        working = read_json(paths["working"])
        if not isinstance(state, dict) or not isinstance(working, dict):
            raise ValueError("A-MEM state files must contain JSON objects")
        return state, working
    if (
        args.force_init
        and paths["output_dir"].exists()
        and any(paths["output_dir"].iterdir())
    ):
        raise RuntimeError(
            "--force_init does not delete existing Batch state. "
            "Use a new --output_dir to avoid duplicate charges."
        )
    if args.top_k <= 0:
        raise ValueError("--top_k must be positive")
    if args.max_examples is not None and args.max_examples <= 0:
        raise ValueError("--max_examples must be positive")
    if args.max_sessions is not None and args.max_sessions <= 0:
        raise ValueError("--max_sessions must be positive")

    input_path = Path(args.input_path).resolve()
    rows = load_dataset(input_path)
    selected_rows = (
        rows[: args.max_examples]
        if args.max_examples is not None
        else rows
    )
    users: dict[str, Any] = {}
    for dataset_index, example in enumerate(selected_rows):
        example_id = str(example.get("example_id") or f"row-{dataset_index}")
        if example_id in users:
            raise ValueError(f"Duplicate example_id: {example_id}")
        units = turn_note_units(
            example,
            max_sessions=args.max_sessions,
        )
        users[example_id] = {
            "dataset_index": dataset_index,
            "note_count": len(units),
            "notes": [],
        }
    state = {
        "version": STATE_VERSION,
        "phase": "ready",
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "embedding_model": args.embedding_model,
        "top_k": args.top_k,
        "max_completion_tokens": args.max_completion_tokens,
        "max_examples": args.max_examples,
        "max_sessions": args.max_sessions,
        "current_note_index": 1,
        "active_attempt": None,
        "batch_history": [],
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
    }
    working = {
        "version": STATE_VERSION,
        "input_sha256": state["input_sha256"],
        "users": users,
    }
    atomic_write_json(paths["state"], state)
    atomic_write_json(paths["working"], working)
    return state, working


def validate_config(
    args: argparse.Namespace,
    state: Mapping[str, Any],
    working: Mapping[str, Any],
) -> None:
    input_path = Path(args.input_path).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"A-MEM input dataset not found: {input_path}")
    input_sha256 = sha256_file(input_path)
    expected = {
        "version": STATE_VERSION,
        "input_path": str(input_path),
        "input_sha256": input_sha256,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "embedding_model": args.embedding_model,
        "top_k": args.top_k,
        "max_completion_tokens": args.max_completion_tokens,
        "max_examples": args.max_examples,
        "max_sessions": args.max_sessions,
    }
    mismatches = {
        key: {"recorded": state.get(key), "requested": value}
        for key, value in expected.items()
        if state.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "A-MEM state/config mismatch; use the recorded flags or a new "
            f"output directory: {canonical_json(mismatches)}"
        )
    working_mismatches = {
        "version": {
            "recorded": working.get("version"),
            "expected": STATE_VERSION,
        },
        "input_sha256": {
            "recorded": working.get("input_sha256"),
            "expected": input_sha256,
        },
    }
    working_mismatches = {
        key: value
        for key, value in working_mismatches.items()
        if value["recorded"] != value["expected"]
    }
    if working_mismatches:
        raise RuntimeError(
            "A-MEM working memory identity mismatch: "
            f"{canonical_json(working_mismatches)}"
        )


def candidate_payload(
    notes: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "note_id": str(note.get("note_id")),
            "note_index": note.get("note_index"),
            "session_index": note.get("session_index"),
            "turn_index": note.get("turn_index"),
            "content": note.get("content"),
            "context": note.get("context"),
            "keywords": note.get("keywords") or [],
            "tags": note.get("tags") or [],
            "score": note.get("score"),
        }
        for note in notes
    ]


def _batch_request(
    *,
    custom_id: str,
    model: str,
    reasoning_effort: str,
    max_completion_tokens: int,
    system_prompt: str,
    user_payload: Mapping[str, Any],
    response_schema: Mapping[str, Any],
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(user_payload, ensure_ascii=False),
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": dict(response_schema),
        },
        "max_completion_tokens": max_completion_tokens,
    }
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }


def build_metadata_request(
    *,
    custom_id: str,
    model: str,
    reasoning_effort: str,
    max_completion_tokens: int,
    content: str,
) -> dict[str, Any]:
    return _batch_request(
        custom_id=custom_id,
        model=model,
        reasoning_effort=reasoning_effort,
        max_completion_tokens=max_completion_tokens,
        system_prompt=METADATA_SYSTEM_PROMPT,
        user_payload={"content": content},
        response_schema=METADATA_RESPONSE_SCHEMA,
    )


def build_evolution_request(
    *,
    custom_id: str,
    model: str,
    reasoning_effort: str,
    max_completion_tokens: int,
    pending_note: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return _batch_request(
        custom_id=custom_id,
        model=model,
        reasoning_effort=reasoning_effort,
        max_completion_tokens=max_completion_tokens,
        system_prompt=EVOLUTION_SYSTEM_PROMPT,
        user_payload={
            "new_memory": {
                "note_id": pending_note["note_id"],
                "content": pending_note["content"],
                "context": pending_note["context"],
                "keywords": pending_note["keywords"],
                "tags": pending_note["tags"],
            },
            "candidate_notes": candidate_payload(candidates),
            "candidate_count": len(candidates),
        },
        response_schema=EVOLUTION_RESPONSE_SCHEMA,
    )


def pending_users(
    working: Mapping[str, Any],
    note_index: int,
) -> list[tuple[str, Mapping[str, Any]]]:
    users = working.get("users") or {}
    return sorted(
        (
            (str(example_id), user)
            for example_id, user in users.items()
            if int(user.get("note_count", 0)) >= note_index
        ),
        key=lambda item: int(item[1]["dataset_index"]),
    )


def _write_attempt(
    args: argparse.Namespace,
    state: dict[str, Any],
    *,
    note_index: int,
    stage: str,
    manifests: Sequence[Mapping[str, Any]],
    requests: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    paths = paths_for(args)
    directory = stage_dir(paths["output_dir"], note_index, stage)
    manifest_path = directory / "manifest.jsonl"
    requests_path = directory / "requests.jsonl"
    if manifest_path.exists() or requests_path.exists():
        raise FileExistsError(
            f"A-MEM note {note_index} {stage} files already exist under "
            f"{directory}"
        )
    atomic_write_jsonl(manifest_path, manifests)
    atomic_write_jsonl(requests_path, requests)
    active_attempt = {
        "note_index": note_index,
        "stage": stage,
        "phase": f"{stage}_prepared",
        "prepared_at": now_iso(),
        "request_count": len(requests),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "requests_path": str(requests_path),
        "requests_sha256": sha256_file(requests_path),
        "batch_id": None,
        "batch_status": None,
        "input_file_id": None,
        "output_file_id": None,
        "error_file_id": None,
    }
    state.update(
        {
            "phase": f"{stage}_prepared",
            "updated_at": now_iso(),
            "active_attempt": active_attempt,
        }
    )
    atomic_write_json(paths["state"], state)
    print(
        f"Prepared A-MEM note {note_index} {stage}: "
        f"{len(requests)} requests -> {requests_path}"
    )
    return state


def prepare_round(
    args: argparse.Namespace,
    state: dict[str, Any],
    working: dict[str, Any],
    *,
    embedder: Any | None = None,
) -> dict[str, Any]:
    del embedder
    if state.get("phase") != "ready":
        raise RuntimeError(f"Cannot prepare while phase={state.get('phase')}")
    paths = paths_for(args)
    rows = load_dataset(state["input_path"])
    note_index = int(state["current_note_index"])
    pending = pending_users(working, note_index)
    if not pending:
        finalize(args, state, working)
        return read_json(paths["state"])

    manifests: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    for example_id, user in pending:
        dataset_index = int(user["dataset_index"])
        units = note_units_for_user(
            rows,
            user,
            max_sessions=state.get("max_sessions"),
        )
        note_unit = units[note_index - 1]
        custom_id = custom_id_for(dataset_index, note_index, "metadata")
        note_id = note_id_for(dataset_index, note_unit)
        manifests.append(
            {
                "custom_id": custom_id,
                "note_id": note_id,
                "example_id": example_id,
                "dataset_index": dataset_index,
                "note_unit": note_unit,
            }
        )
        requests.append(
            build_metadata_request(
                custom_id=custom_id,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                max_completion_tokens=args.max_completion_tokens,
                content=str(note_unit["content"]),
            )
        )
    return _write_attempt(
        args,
        state,
        note_index=note_index,
        stage="metadata",
        manifests=manifests,
        requests=requests,
    )


def submit_round(
    args: argparse.Namespace,
    state: dict[str, Any],
) -> dict[str, Any]:
    attempt = state.get("active_attempt") or {}
    stage = str(attempt.get("stage") or "")
    if state.get("phase") != f"{stage}_prepared":
        raise RuntimeError(f"Cannot submit while phase={state.get('phase')}")
    if attempt.get("batch_id"):
        raise RuntimeError(
            "This A-MEM stage already has a batch_id; refusing duplicate "
            "charges."
        )
    paths = paths_for(args)
    attempt.update(
        {
            "phase": "submission_pending",
            "submission_started_at": now_iso(),
        }
    )
    state.update(
        {
            "phase": "submission_pending",
            "updated_at": now_iso(),
            "active_attempt": attempt,
        }
    )
    atomic_write_json(paths["state"], state)

    client = client_for(args)
    requests_path = Path(attempt["requests_path"])
    with requests_path.open("rb") as handle:
        uploaded = client.files.create(file=handle, purpose="batch")
    attempt["input_file_id"] = uploaded.id
    state["updated_at"] = now_iso()
    atomic_write_json(paths["state"], state)
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "experiment": "experiment8",
            "method": "amem",
            "stage": stage,
            "note_index": str(attempt["note_index"]),
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "requests_sha256": str(attempt["requests_sha256"]),
        },
    )
    attempt.update(
        {
            "phase": f"{stage}_submitted",
            "submitted_at": now_iso(),
            "input_file_id": uploaded.id,
            "batch_id": batch.id,
            "batch_status": batch.status,
        }
    )
    state.update(
        {
            "phase": f"{stage}_submitted",
            "updated_at": now_iso(),
            "active_attempt": attempt,
        }
    )
    atomic_write_json(paths["state"], state)
    print(
        f"Submitted A-MEM note {attempt['note_index']} {stage} batch "
        f"{batch.id} (status={batch.status})"
    )
    return state


def refresh_status(
    args: argparse.Namespace,
    state: dict[str, Any],
) -> dict[str, Any]:
    attempt = state.get("active_attempt") or {}
    stage = str(attempt.get("stage") or "")
    if state.get("phase") not in {
        f"{stage}_submitted",
        f"{stage}_completed",
    }:
        raise RuntimeError(f"No submitted batch while phase={state.get('phase')}")
    batch_id = attempt.get("batch_id")
    if not batch_id:
        raise RuntimeError("Active attempt has no batch_id")
    batch = client_for(args).batches.retrieve(batch_id)
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
            "status_checked_at": now_iso(),
        }
    )
    state.update(
        {
            "phase": (
                f"{stage}_completed"
                if batch.status == "completed"
                else f"{stage}_submitted"
            ),
            "updated_at": now_iso(),
            "active_attempt": attempt,
        }
    )
    atomic_write_json(paths_for(args)["state"], state)
    print(
        f"A-MEM batch {batch_id}: stage={stage}, status={batch.status}, "
        f"counts={attempt.get('request_counts')}"
    )
    return state


def download_results(
    args: argparse.Namespace,
    state: dict[str, Any],
) -> Path:
    attempt = state.get("active_attempt") or {}
    stage = str(attempt["stage"])
    directory = stage_dir(
        paths_for(args)["output_dir"],
        int(attempt["note_index"]),
        stage,
    )
    raw_path = directory / "raw_results.jsonl"
    if args.raw_results:
        source = Path(args.raw_results).resolve()
        if source != raw_path:
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_bytes(source.read_bytes())
        return raw_path
    if attempt.get("batch_status") != "completed":
        state = refresh_status(args, state)
        attempt = state["active_attempt"]
    if attempt.get("batch_status") != "completed":
        raise RuntimeError(
            f"Batch is not completed: {attempt.get('batch_status')}"
        )
    output_file_id = attempt.get("output_file_id")
    if not output_file_id:
        raise RuntimeError("Completed batch has no output_file_id")
    client = client_for(args)
    raw_path.write_bytes(client.files.content(output_file_id).content)
    error_file_id = attempt.get("error_file_id")
    if error_file_id:
        (directory / "raw_errors.jsonl").write_bytes(
            client.files.content(error_file_id).content
        )
    return raw_path


def _batch_outputs(
    args: argparse.Namespace,
    state: dict[str, Any],
    *,
    label: str,
) -> tuple[list[dict[str, Any]], dict[str, Mapping[str, Any]], Path]:
    attempt = state.get("active_attempt") or {}
    manifests = read_jsonl(Path(attempt["manifest_path"]))
    raw_path = download_results(args, state)
    output_rows = read_jsonl(raw_path)
    output_by_id = index_unique_rows(
        output_rows,
        id_key="custom_id",
        label=label,
    )
    expected = {str(row["custom_id"]) for row in manifests}
    missing = expected - set(output_by_id)
    unknown = set(output_by_id) - expected
    if missing or unknown:
        raise RuntimeError(
            "Batch result ID mismatch: "
            f"missing={sorted(missing)[:5]} unknown={sorted(unknown)[:5]}"
        )
    return manifests, output_by_id, raw_path


def _history_entry(
    attempt: Mapping[str, Any],
    *,
    raw_path: Path,
    usage_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        **dict(attempt),
        "phase": f"{attempt['stage']}_collected",
        "collected_at": now_iso(),
        "raw_results_path": str(raw_path),
        "raw_results_sha256": sha256_file(raw_path),
        "usage": sum_usage(usage_rows),
    }


def collect_metadata(
    args: argparse.Namespace,
    state: dict[str, Any],
    working: dict[str, Any],
    *,
    embedder: Any | None = None,
) -> dict[str, Any]:
    attempt = state.get("active_attempt") or {}
    if attempt.get("stage") != "metadata":
        raise RuntimeError("Active A-MEM stage is not metadata")
    manifests, output_by_id, raw_path = _batch_outputs(
        args,
        state,
        label="A-MEM metadata Batch results",
    )
    users = working["users"]
    metadata_rows: list[dict[str, Any]] = []
    usage_rows: list[dict[str, Any]] = []
    for manifest in sorted(
        manifests, key=lambda item: int(item["dataset_index"])
    ):
        body = extract_batch_body(output_by_id[str(manifest["custom_id"])])
        metadata = parse_message_json(body)
        usage = normalize_usage(body)
        pending_note = create_pending_note(
            manifest["note_unit"],
            note_id=str(manifest["note_id"]),
            metadata_response=metadata,
            metadata_usage=usage,
            timestamp=now_iso(),
        )
        metadata_rows.append(
            {
                **manifest,
                "pending_note": pending_note,
            }
        )
        usage_rows.append(usage)

    embedder = embedder or make_embedder(args.embedding_model)
    query_embeddings = embedder.encode(
        [str(row["pending_note"]["content"]) for row in metadata_rows]
    )
    if len(query_embeddings) != len(metadata_rows):
        raise RuntimeError("Embedding result count does not match note count")

    evolution_manifests: list[dict[str, Any]] = []
    evolution_requests: list[dict[str, Any]] = []
    note_index = int(attempt["note_index"])
    for manifest, query_embedding in zip(metadata_rows, query_embeddings):
        example_id = str(manifest["example_id"])
        user_memory = users[example_id]
        candidates = nearest_notes(
            user_memory.get("notes") or [],
            query_embedding,
            top_k=args.top_k,
        )
        dataset_index = int(manifest["dataset_index"])
        custom_id = custom_id_for(dataset_index, note_index, "evolution")
        evolution_manifests.append(
            {
                "custom_id": custom_id,
                "note_id": manifest["note_id"],
                "example_id": example_id,
                "dataset_index": dataset_index,
                "pending_note": manifest["pending_note"],
                "candidate_notes": candidate_payload(candidates),
                "candidate_note_ids": [
                    str(note["note_id"]) for note in candidates
                ],
            }
        )
        evolution_requests.append(
            build_evolution_request(
                custom_id=custom_id,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                max_completion_tokens=args.max_completion_tokens,
                pending_note=manifest["pending_note"],
                candidates=candidates,
            )
        )

    state["batch_history"].append(
        _history_entry(
            attempt,
            raw_path=raw_path,
            usage_rows=usage_rows,
        )
    )
    return _write_attempt(
        args,
        state,
        note_index=note_index,
        stage="evolution",
        manifests=evolution_manifests,
        requests=evolution_requests,
    )


def collect_evolution(
    args: argparse.Namespace,
    state: dict[str, Any],
    working: dict[str, Any],
    *,
    embedder: Any | None = None,
) -> dict[str, Any]:
    attempt = state.get("active_attempt") or {}
    if attempt.get("stage") != "evolution":
        raise RuntimeError("Active A-MEM stage is not evolution")
    manifests, output_by_id, raw_path = _batch_outputs(
        args,
        state,
        label="A-MEM evolution Batch results",
    )
    embedder = embedder or make_embedder(args.embedding_model)
    users = working["users"]
    usage_rows: list[dict[str, Any]] = []
    for manifest in sorted(
        manifests, key=lambda item: int(item["dataset_index"])
    ):
        body = extract_batch_body(output_by_id[str(manifest["custom_id"])])
        decision = parse_message_json(body)
        usage = normalize_usage(body)
        user_memory = users[str(manifest["example_id"])]
        changed = apply_evolution_result(
            user_memory,
            pending_note=manifest["pending_note"],
            candidate_note_ids=manifest["candidate_note_ids"],
            response=decision,
            evolution_usage=usage,
            timestamp=now_iso(),
        )
        assign_embeddings(user_memory, changed, embedder=embedder)
        usage_rows.append(usage)

    state["batch_history"].append(
        _history_entry(
            attempt,
            raw_path=raw_path,
            usage_rows=usage_rows,
        )
    )
    collected_at = now_iso()
    state.update(
        {
            "phase": "ready",
            "updated_at": collected_at,
            "current_note_index": int(attempt["note_index"]) + 1,
            "active_attempt": None,
        }
    )
    paths = paths_for(args)
    atomic_write_json(paths["working"], working)
    atomic_write_json(paths["state"], state)
    print(
        f"Collected A-MEM note {attempt['note_index']} evolution: "
        f"{len(manifests)} committed notes, usage={sum_usage(usage_rows)}"
    )
    if not pending_users(working, int(state["current_note_index"])):
        finalize(args, state, working)
        return read_json(paths["state"])
    return state


def collect_round(
    args: argparse.Namespace,
    state: dict[str, Any],
    working: dict[str, Any],
    *,
    embedder: Any | None = None,
) -> dict[str, Any]:
    attempt = state.get("active_attempt") or {}
    stage = str(attempt.get("stage") or "")
    if state.get("phase") not in {
        f"{stage}_submitted",
        f"{stage}_completed",
        f"{stage}_prepared",
    }:
        raise RuntimeError(f"Cannot collect while phase={state.get('phase')}")
    if state.get("phase") == f"{stage}_prepared" and not args.raw_results:
        raise RuntimeError(
            "A prepared stage can only collect with --raw_results"
        )
    if stage == "metadata":
        return collect_metadata(
            args,
            state,
            working,
            embedder=embedder,
        )
    if stage == "evolution":
        return collect_evolution(
            args,
            state,
            working,
            embedder=embedder,
        )
    raise RuntimeError(f"Unknown active A-MEM stage: {stage}")


def _note_usage(note: Mapping[str, Any]) -> dict[str, int]:
    construction = note.get("construction_usage") or {}
    return sum_usage(
        [
            construction.get("metadata") or {},
            construction.get("evolution") or {},
        ]
    )


def finalize(
    args: argparse.Namespace,
    state: dict[str, Any],
    working: dict[str, Any],
) -> None:
    paths = paths_for(args)
    users = working.get("users") or {}
    incomplete = [
        {
            "example_id": example_id,
            "expected": int(user.get("note_count", 0)),
            "actual": len(user.get("notes") or []),
        }
        for example_id, user in users.items()
        if len(user.get("notes") or [])
        != int(user.get("note_count", 0))
    ]
    if incomplete:
        raise RuntimeError(
            "Cannot finalize incomplete A-MEM construction: "
            f"{canonical_json(incomplete[:5])}"
        )

    rows: list[dict[str, Any]] = []
    all_usage: list[Mapping[str, Any]] = []
    note_count = 0
    link_count = 0
    for example_id, user in sorted(
        users.items(), key=lambda item: int(item[1]["dataset_index"])
    ):
        notes = user.get("notes") or []
        note_count += len(notes)
        link_count += sum(len(note.get("links") or []) for note in notes)
        note_usage = [_note_usage(note) for note in notes]
        all_usage.extend(note_usage)
        rows.append(
            {
                "example_id": example_id,
                "dataset_index": int(user["dataset_index"]),
                "method": "amem",
                "memory_model": state["model"],
                "reasoning_effort": state["reasoning_effort"],
                "embedding_model": state["embedding_model"],
                "top_k": state["top_k"],
                "note_unit": "dialogue_turn",
                "upstream_repository": UPSTREAM_REPOSITORY,
                "upstream_commit": UPSTREAM_COMMIT,
                "notes": notes,
                "construction_usage": sum_usage(note_usage),
            }
        )
    atomic_write_jsonl(paths["artifact"], rows)
    summary = {
        "completed_at": now_iso(),
        "input_path": state["input_path"],
        "input_sha256": state["input_sha256"],
        "artifact_path": str(paths["artifact"]),
        "artifact_sha256": sha256_file(paths["artifact"]),
        "example_count": len(rows),
        "note_count": note_count,
        "directed_link_count": link_count,
        "model": state["model"],
        "reasoning_effort": state["reasoning_effort"],
        "embedding_model": state["embedding_model"],
        "top_k": state["top_k"],
        "causal_note_rounds": max(
            (int(user.get("note_count", 0)) for user in users.values()),
            default=0,
        ),
        "batch_jobs": len(state.get("batch_history") or []),
        "construction_usage": sum_usage(all_usage),
        "upstream": {
            "repository": UPSTREAM_REPOSITORY,
            "commit": UPSTREAM_COMMIT,
            "license": "MIT",
        },
        "official_semantics": {
            "turn_level_memory_note": True,
            "immutable_content": True,
            "separate_metadata_analysis": True,
            "raw_content_neighbor_query": True,
            "construction_neighbor_k": state["top_k"],
            "strengthen_new_note": True,
            "update_neighbor_metadata": True,
            "directed_links_without_backlinks": True,
            "retriever_document_includes_metadata": True,
        },
        "adapter_boundaries": [
            "OpenAI calls use asynchronous Batch jobs.",
            "Stable string note IDs replace in-memory integer indexes.",
            "Memory and retrieval remain isolated by Experiment8 example_id.",
            "Evolved neighbors are immediately re-embedded.",
        ],
    }
    atomic_write_json(paths["summary"], summary)
    state.update(
        {
            "phase": "finalized",
            "updated_at": now_iso(),
            "artifact_path": str(paths["artifact"]),
            "artifact_sha256": summary["artifact_sha256"],
        }
    )
    atomic_write_json(paths["state"], state)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def run_loop(
    args: argparse.Namespace,
    state: dict[str, Any],
    working: dict[str, Any],
) -> None:
    while state.get("phase") != "finalized":
        phase = str(state.get("phase") or "")
        if phase == "ready":
            state = prepare_round(args, state, working)
        elif phase.endswith("_prepared"):
            state = submit_round(args, state)
        elif phase == "submission_pending":
            attempt = state.get("active_attempt") or {}
            raise RuntimeError(
                "Submission was interrupted after local state was sealed. "
                "Refusing an automatic resubmit because it could create a "
                "duplicate Batch charge. Inspect the remote Batch list using "
                f"input_file_id={attempt.get('input_file_id')}."
            )
        elif phase.endswith("_submitted") or phase.endswith("_completed"):
            state = refresh_status(args, state)
            attempt = state.get("active_attempt") or {}
            status = attempt.get("batch_status")
            if status == "completed":
                state = collect_round(args, state, working)
            elif status in TERMINAL_BATCH_STATUSES:
                raise RuntimeError(f"Batch ended without completion: {status}")
            else:
                time.sleep(max(1, args.poll_seconds))
        else:
            raise RuntimeError(f"Unknown A-MEM phase: {phase}")


def main() -> None:
    args = parse_args()
    state, working = initialize(args)
    validate_config(args, state, working)
    if args.command == "prepare":
        prepare_round(args, state, working)
    elif args.command == "submit":
        submit_round(args, state)
    elif args.command == "status":
        refresh_status(args, state)
    elif args.command == "collect":
        collect_round(args, state, working)
    elif args.command == "finalize":
        finalize(args, state, working)
    else:
        run_loop(args, state, working)


if __name__ == "__main__":
    main()
