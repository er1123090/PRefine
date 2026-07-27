"""Shared primitives for the Experiment8 adaptation of official A-MEM.

The implementation is intentionally shaped after ``WujiangXu/A-mem``:

* every dialogue turn is an immutable ``MemoryNote`` content unit;
* note metadata is generated before neighbor retrieval;
* evolution may strengthen the new note and update retrieved neighbors;
* links are directed from the new note to older notes;
* retrieval expands semantic hits through their directed links.

Experiment8 keeps its own user isolation, local MiniLM embeddings, evaluator,
and OpenAI Batch orchestration around those A-MEM semantics.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence


DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_REASONING_EFFORT = "minimal"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CONSTRUCTION_TOP_K = 5
DEFAULT_RETRIEVAL_TOP_K = 10
STATE_VERSION = 2

UPSTREAM_REPOSITORY = "https://github.com/WujiangXu/A-mem"
UPSTREAM_COMMIT = "0c8039f28fdcc08189a23c07a3437d9d2482f9c2"

TOKEN_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "total_tokens",
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
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


def index_unique_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    id_key: str,
    label: str,
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row_index, row in enumerate(rows):
        raw_identifier = row.get(id_key)
        if raw_identifier is None or not str(raw_identifier).strip():
            raise RuntimeError(
                f"{label} row {row_index} has no non-empty {id_key}"
            )
        identifier = str(raw_identifier)
        if identifier in indexed:
            raise RuntimeError(
                f"{label} contains duplicate {id_key}: {identifier}"
            )
        indexed[identifier] = row
    return indexed


def atomic_write_jsonl(
    path: Path, rows: Iterable[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    value = read_json(Path(path))
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain a JSON array")
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{path}[{index}] must contain an object")
        rows.append(item)
    return rows


def render_turn(turn: Mapping[str, Any]) -> str:
    """Render one source turn using the official ``speaker: content`` shape."""
    role = str(turn.get("role") or "Unknown").strip() or "Unknown"
    message = str(turn.get("message") or turn.get("content") or "").strip()
    if not message:
        return ""
    lines = [f"{role}: {message}"]
    services = turn.get("service")
    if isinstance(services, list):
        for service in services:
            if str(service).strip():
                lines.append(f"Observed service call: {service}")
    return "\n".join(lines)


def turn_note_units(
    example: Mapping[str, Any],
    *,
    max_sessions: int | None = None,
) -> list[dict[str, Any]]:
    """Flatten an Experiment8 user's sessions into causal A-MEM note units."""
    sessions = list(example.get("sessions") or [])
    if max_sessions is not None:
        sessions = sessions[:max_sessions]
    units: list[dict[str, Any]] = []
    for session_index, session in enumerate(sessions, start=1):
        if not isinstance(session, Mapping):
            continue
        dialogue_id = str(session.get("dialogue_id") or "")
        for turn_index, turn in enumerate(
            session.get("dialogue") or [], start=1
        ):
            if not isinstance(turn, Mapping):
                continue
            content = render_turn(turn)
            if not content:
                continue
            units.append(
                {
                    "note_index": len(units) + 1,
                    "session_index": session_index,
                    "turn_index": turn_index,
                    "dialogue_id": dialogue_id,
                    "speaker": str(turn.get("role") or "Unknown"),
                    "content": content,
                }
            )
    return units


def note_document(note: Mapping[str, Any]) -> str:
    """Match the official retriever document: content plus generated metadata."""
    parts = [
        str(note.get("content") or ""),
        "Context: " + str(note.get("context") or ""),
        "Keywords: "
        + ", ".join(str(item) for item in note.get("keywords") or []),
        "Tags: " + ", ".join(str(item) for item in note.get("tags") or []),
    ]
    return "\n".join(part for part in parts if part.strip())


class LocalMiniLMEmbedder:
    """Persistent Chroma ONNX all-MiniLM-L6-v2 embedding function."""

    model_name = DEFAULT_EMBEDDING_MODEL

    def __init__(self) -> None:
        try:
            from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import (
                ONNXMiniLM_L6_V2,
            )
        except ImportError as exc:
            raise RuntimeError(
                "chromadb is required for A-MEM local embeddings"
            ) from exc
        # DefaultEmbeddingFunction creates ONNXMiniLM_L6_V2 on every call.
        # Keep one model/session alive for the full construction run instead.
        self._function = ONNXMiniLM_L6_V2()

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self._function(input=list(texts))
        return [[float(value) for value in row] for row in embeddings]


def make_embedder(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> LocalMiniLMEmbedder:
    if model_name != DEFAULT_EMBEDDING_MODEL:
        raise ValueError(
            "A-MEM currently fixes local embeddings to "
            f"{DEFAULT_EMBEDDING_MODEL}; received {model_name!r}"
        )
    return LocalMiniLMEmbedder()


def cosine_similarity(
    left: Sequence[float], right: Sequence[float]
) -> float:
    if len(left) != len(right) or not left:
        return -1.0
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return -1.0
    return dot / (left_norm * right_norm)


def nearest_notes(
    notes: Sequence[Mapping[str, Any]],
    query_embedding: Sequence[float],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    if top_k <= 0:
        return []
    ranked: list[tuple[float, Mapping[str, Any]]] = []
    for note in notes:
        embedding = note.get("embedding")
        if not isinstance(embedding, list):
            continue
        ranked.append((cosine_similarity(query_embedding, embedding), note))
    ranked.sort(
        key=lambda item: (-item[0], str(item[1].get("note_id", "")))
    )
    return [{**dict(note), "score": score} for score, note in ranked[:top_k]]


def retrieve_notes(
    notes: Sequence[Mapping[str, Any]],
    query: str,
    *,
    embedder: Any,
    top_k: int,
    linked_neighbor_limit: int,
) -> list[dict[str, Any]]:
    if not notes or top_k <= 0:
        return []
    query_embedding = embedder.encode([query])[0]
    return retrieve_notes_from_embedding(
        notes,
        query_embedding,
        top_k=top_k,
        linked_neighbor_limit=linked_neighbor_limit,
    )


def retrieve_notes_from_embedding(
    notes: Sequence[Mapping[str, Any]],
    query_embedding: Sequence[float],
    *,
    top_k: int,
    linked_neighbor_limit: int,
) -> list[dict[str, Any]]:
    """Implement official ``find_related_memories_raw`` one-hop expansion."""
    if not notes or top_k <= 0:
        return []
    seeds = nearest_notes(notes, query_embedding, top_k=top_k)
    note_by_id = {
        str(note.get("note_id")): note
        for note in notes
        if note.get("note_id") is not None
    }
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for seed in seeds:
        note_id = str(seed.get("note_id"))
        if note_id not in selected_ids:
            selected.append({**seed, "retrieval_source": "semantic"})
            selected_ids.add(note_id)
        added = 0
        for linked_id in seed.get("links") or []:
            linked_id = str(linked_id)
            if linked_id in selected_ids or linked_id not in note_by_id:
                continue
            selected.append(
                {
                    **dict(note_by_id[linked_id]),
                    "score": None,
                    "retrieval_source": f"linked:{note_id}",
                }
            )
            selected_ids.add(linked_id)
            added += 1
            if added >= linked_neighbor_limit:
                break
    return selected


def public_note(
    note: Mapping[str, Any], *, include_embedding: bool = False
) -> dict[str, Any]:
    return {
        key: value
        for key, value in note.items()
        if include_embedding or key != "embedding"
    }


def format_retrieved_notes(
    notes: Sequence[Mapping[str, Any]],
) -> str:
    if not notes:
        return "No relevant memories retrieved."
    rendered: list[str] = []
    for index, note in enumerate(notes, start=1):
        score = note.get("score")
        score_text = (
            f"{float(score):.4f}"
            if isinstance(score, (int, float))
            else "linked"
        )
        rendered.append(
            "\n".join(
                (
                    f"[Memory {index} | id={note.get('note_id')} | "
                    f"score={score_text}]",
                    f"Content: {note.get('content', '')}",
                    f"Context: {note.get('context', '')}",
                    "Keywords: "
                    + ", ".join(
                        str(item) for item in note.get("keywords") or []
                    ),
                    "Tags: "
                    + ", ".join(
                        str(item) for item in note.get("tags") or []
                    ),
                )
            )
        )
    return "\n\n".join(rendered)


def load_memory_artifact(
    path: str | Path,
) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(Path(path))
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        example_id = str(row.get("example_id", ""))
        if not example_id:
            raise ValueError("A-MEM artifact row is missing example_id")
        if example_id in result:
            raise ValueError(f"Duplicate A-MEM example_id: {example_id}")
        result[example_id] = row
    return result


def normalize_usage(body: Mapping[str, Any]) -> dict[str, int]:
    usage = body.get("usage") or {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    input_tokens = int(usage.get("prompt_tokens", 0) or 0)
    output_tokens = int(usage.get("completion_tokens", 0) or 0)
    total_tokens = int(
        usage.get("total_tokens", input_tokens + output_tokens) or 0
    )
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": int(
            prompt_details.get("cached_tokens", 0) or 0
        ),
        "output_tokens": output_tokens,
        "reasoning_tokens": int(
            completion_details.get("reasoning_tokens", 0) or 0
        ),
        "total_tokens": total_tokens,
    }


def sum_usage(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    values = list(rows)
    return {
        field: sum(int(row.get(field, 0) or 0) for row in values)
        for field in TOKEN_FIELDS
    }


def extract_batch_body(row: Mapping[str, Any]) -> dict[str, Any]:
    if row.get("error"):
        raise RuntimeError(f"Batch row error: {canonical_json(row['error'])}")
    response = row.get("response")
    if not isinstance(response, Mapping):
        raise RuntimeError("Batch row has no response")
    status_code = int(response.get("status_code", 0) or 0)
    if status_code != 200:
        raise RuntimeError(
            f"Batch row HTTP {status_code}: "
            f"{canonical_json(response.get('body'))}"
        )
    body = response.get("body")
    if not isinstance(body, dict):
        raise RuntimeError("Batch response body is not an object")
    return body


def parse_message_json(body: Mapping[str, Any]) -> dict[str, Any]:
    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError("Batch response has no choices")
    message = choices[0].get("message") or {}
    content = str(message.get("content") or "").strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines).strip()
    value = json.loads(content)
    if not isinstance(value, dict):
        raise ValueError("A-MEM response must be a JSON object")
    return value


def string_list(value: Any, *, limit: int = 20) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in result:
            result.append(text)
        if len(result) >= limit:
            break
    return result


def normalize_metadata_response(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "context": str(value.get("context") or "").strip(),
        "keywords": string_list(value.get("keywords")),
        "tags": string_list(value.get("tags")),
    }


def create_pending_note(
    note_unit: Mapping[str, Any],
    *,
    note_id: str,
    metadata_response: Mapping[str, Any],
    metadata_usage: Mapping[str, Any],
    timestamp: str,
) -> dict[str, Any]:
    metadata = normalize_metadata_response(metadata_response)
    return {
        "note_id": note_id,
        "note_index": int(note_unit["note_index"]),
        "session_index": int(note_unit["session_index"]),
        "turn_index": int(note_unit["turn_index"]),
        "dialogue_id": str(note_unit.get("dialogue_id") or ""),
        "speaker": str(note_unit.get("speaker") or "Unknown"),
        "content": str(note_unit["content"]),
        "context": metadata["context"],
        "keywords": metadata["keywords"],
        "tags": metadata["tags"],
        "links": [],
        "created_at": timestamp,
        "updated_at": timestamp,
        "construction_usage": {
            "metadata": dict(metadata_usage),
            "evolution": {field: 0 for field in TOKEN_FIELDS},
        },
    }


def normalize_evolution_response(
    value: Mapping[str, Any],
    *,
    candidate_note_ids: Sequence[str],
) -> dict[str, Any]:
    allowed = set(candidate_note_ids)
    actions = [
        action
        for action in string_list(value.get("actions"), limit=2)
        if action in {"strengthen", "update_neighbor"}
    ]
    connections = [
        note_id
        for note_id in string_list(value.get("suggested_connections"))
        if note_id in allowed
    ]
    contexts = (
        [
            str(item).strip()
            for item in value.get("new_context_neighborhood") or []
        ]
        if isinstance(value.get("new_context_neighborhood"), list)
        else []
    )
    raw_tags = value.get("new_tags_neighborhood")
    neighborhood_tags = (
        [string_list(item) for item in raw_tags]
        if isinstance(raw_tags, list)
        else []
    )
    return {
        "should_evolve": bool(value.get("should_evolve", False)),
        "actions": actions,
        "suggested_connections": connections,
        "tags_to_update": string_list(value.get("tags_to_update")),
        "new_context_neighborhood": contexts,
        "new_tags_neighborhood": neighborhood_tags,
    }


def apply_evolution_result(
    user_memory: MutableMapping[str, Any],
    *,
    pending_note: Mapping[str, Any],
    candidate_note_ids: Sequence[str],
    response: Mapping[str, Any],
    evolution_usage: Mapping[str, Any],
    timestamp: str,
) -> set[str]:
    """Apply official strengthen/update-neighbor semantics and commit the note."""
    notes = user_memory.setdefault("notes", [])
    note_id = str(pending_note["note_id"])
    if any(str(note.get("note_id")) == note_id for note in notes):
        raise ValueError(f"Duplicate note_id: {note_id}")
    note_by_id = {
        str(note.get("note_id")): note
        for note in notes
        if note.get("note_id") is not None
    }
    ordered_candidate_ids = [
        str(candidate_id)
        for candidate_id in candidate_note_ids
        if str(candidate_id) in note_by_id
    ]
    decision = normalize_evolution_response(
        response,
        candidate_note_ids=ordered_candidate_ids,
    )
    new_note = dict(pending_note)
    new_note["construction_usage"] = {
        **dict(new_note.get("construction_usage") or {}),
        "evolution": dict(evolution_usage),
    }
    changed = {note_id}

    if decision["should_evolve"]:
        if "strengthen" in decision["actions"]:
            new_note["links"] = decision["suggested_connections"]
            if decision["tags_to_update"]:
                new_note["tags"] = decision["tags_to_update"]
        if "update_neighbor" in decision["actions"]:
            contexts = decision["new_context_neighborhood"]
            tags = decision["new_tags_neighborhood"]
            for index, candidate_id in enumerate(ordered_candidate_ids):
                neighbor = note_by_id[candidate_id]
                updated = False
                if index < len(contexts) and contexts[index]:
                    neighbor["context"] = contexts[index]
                    updated = True
                if index < len(tags):
                    neighbor["tags"] = tags[index]
                    updated = True
                if updated:
                    neighbor["updated_at"] = timestamp
                    changed.add(candidate_id)

    new_note["updated_at"] = timestamp
    new_note["evolution"] = decision
    notes.append(new_note)
    return changed


def assign_embeddings(
    user_memory: MutableMapping[str, Any],
    note_ids: Iterable[str],
    *,
    embedder: Any,
) -> None:
    selected_ids = {str(note_id) for note_id in note_ids}
    targets = [
        note
        for note in user_memory.get("notes") or []
        if str(note.get("note_id")) in selected_ids
    ]
    if not targets:
        return
    embeddings = embedder.encode([note_document(note) for note in targets])
    if len(embeddings) != len(targets):
        raise RuntimeError("Embedding result count does not match note count")
    for note, embedding in zip(targets, embeddings):
        note["embedding"] = embedding
