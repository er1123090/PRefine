"""Canonical, provider-neutral RAG adapter with a verified index lifecycle."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
from typing import Any, Awaitable, Callable, Mapping, Protocol, Sequence

from ..contracts import (
    CanonicalPreparedRecord,
    MethodContractError,
    MethodRunContext,
    MethodStateHandle,
    await_if_needed,
    bind_method_state,
    canonical_prediction,
    method_state_path,
    validate_prepared_record,
    validate_prepared_records,
)
from ...provenance.admission import (
    FileAdmissionError,
    StrictJSONError,
    admit_directory,
    admit_regular_file,
    decode_strict_json,
)


METHOD_ID = "rag"
STATE_FORMAT_VERSION = 1
STATE_MANIFEST = "state_manifest.json"


@dataclass(frozen=True)
class IndexDocument:
    """One deterministic dialogue or API-history chunk sent to an index backend."""

    document_id: str
    source_example_id: str
    user_id: str
    kind: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_mapping(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "kind": self.kind,
            "metadata": copy.deepcopy(dict(self.metadata)),
            "source_example_id": self.source_example_id,
            "text": self.text,
            "user_id": self.user_id,
        }


@dataclass(frozen=True)
class RetrievalRequest:
    """A query that an index backend must scope to one canonical source/user."""

    query: str
    source_example_id: str
    user_id: str
    top_k: int


@dataclass(frozen=True)
class RetrievedContext:
    """One identity-scoped retrieved chunk returned by the backend."""

    text: str
    source_example_id: str
    user_id: str
    document_id: str = ""
    score: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_mapping(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "metadata": copy.deepcopy(dict(self.metadata)),
            "score": self.score,
            "source_example_id": self.source_example_id,
            "text": self.text,
            "user_id": self.user_id,
        }


@dataclass(frozen=True)
class RAGGenerationRequest:
    """Retrieved-only generation context passed to an injected model backend."""

    model_name: str
    tools_schema: tuple[Mapping[str, Any], ...]
    reasoning_effort: str | None
    retrieved: tuple[RetrievedContext, ...]
    retrieved_context: str


class RAGIndexBackend(Protocol):
    """Injected embedding/index/retrieval implementation."""

    def build_index(
        self,
        documents: Sequence[IndexDocument],
        state_dir: Path,
    ) -> Mapping[str, Any] | None | Awaitable[Mapping[str, Any] | None]: ...

    def retrieve(
        self,
        state_dir: Path,
        request: RetrievalRequest,
    ) -> Sequence[RetrievedContext | Mapping[str, Any]] | Awaitable[
        Sequence[RetrievedContext | Mapping[str, Any]]
    ]: ...


RAGGenerationValue = Mapping[str, Any] | str
RAGGenerator = Callable[
    [Mapping[str, Any], RAGGenerationRequest],
    RAGGenerationValue | Awaitable[RAGGenerationValue],
]


@dataclass(frozen=True)
class RAGRuntime:
    """All external RAG behavior is injected through this runtime bundle."""

    index_backend: RAGIndexBackend
    generator: RAGGenerator
    top_k: int = 5

    def __post_init__(self) -> None:
        if not callable(getattr(self.index_backend, "build_index", None)):
            raise MethodContractError("RAG backend must implement build_index()")
        if not callable(getattr(self.index_backend, "retrieve", None)):
            raise MethodContractError("RAG backend must implement retrieve()")
        if not callable(self.generator):
            raise MethodContractError("RAG generator must be callable")
        if not isinstance(self.top_k, int) or self.top_k <= 0:
            raise MethodContractError("RAG top_k must be a positive integer")


def _runtime(value: Any) -> RAGRuntime:
    if isinstance(value, RAGRuntime):
        return value
    if isinstance(value, Mapping):
        try:
            return RAGRuntime(**dict(value))
        except TypeError as exc:
            raise MethodContractError("RAG runtime mapping is malformed") from exc
    raise MethodContractError("RAG backend must be a RAGRuntime or runtime mapping")


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MethodContractError("RAG state metadata must be JSON serializable") from exc


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _source_payload(record: CanonicalPreparedRecord) -> tuple[dict[str, Any], str]:
    payload = record.as_mapping()
    source = payload.get("source_example")
    if not isinstance(source, Mapping):
        raise MethodContractError(
            f"RAG record {record.instance_id} lacks source_example"
        )
    source_value = copy.deepcopy(dict(source))
    embedded_id = source_value.get("example_id")
    if embedded_id is not None and str(embedded_id) != record.source_example_id:
        raise MethodContractError(
            f"RAG source identity drift for {record.instance_id}"
        )
    user_id = source_value.get("user_id", record.source_example_id)
    if not isinstance(user_id, str) or not user_id:
        raise MethodContractError(
            f"RAG source user_id is invalid for {record.instance_id}"
        )
    return source_value, user_id


def _index_documents(
    source_example_id: str,
    user_id: str,
    source: Mapping[str, Any],
) -> list[IndexDocument]:
    sessions = source.get("sessions", [])
    if not isinstance(sessions, list):
        raise MethodContractError(
            f"RAG source {source_example_id} sessions must be a list"
        )
    documents: list[IndexDocument] = []
    for session_index, session in enumerate(sessions):
        if not isinstance(session, Mapping):
            raise MethodContractError(
                f"RAG source {source_example_id} contains an invalid session"
            )
        dialogue = session.get("dialogue", [])
        if not isinstance(dialogue, list):
            raise MethodContractError(
                f"RAG source {source_example_id} session dialogue must be a list"
            )
        for turn_index, turn in enumerate(dialogue):
            if not isinstance(turn, Mapping):
                raise MethodContractError(
                    f"RAG source {source_example_id} contains an invalid dialogue turn"
                )
            role = str(turn.get("role", "")).strip().lower()
            message = turn.get("message", turn.get("content", ""))
            if not isinstance(message, str) or not message.strip():
                continue
            documents.append(
                IndexDocument(
                    document_id=(
                        f"{source_example_id}:dialogue:{session_index}:{turn_index}"
                    ),
                    source_example_id=source_example_id,
                    user_id=user_id,
                    kind="dialogue",
                    text=f"{role or 'unknown'}: {message}",
                    metadata={
                        "role": role,
                        "session_index": session_index,
                        "turn_index": turn_index,
                    },
                )
            )
        api_calls = session.get("api_call", [])
        if isinstance(api_calls, str):
            api_calls = [api_calls] if api_calls else []
        if not isinstance(api_calls, list) or not all(
            isinstance(call, str) for call in api_calls
        ):
            raise MethodContractError(
                f"RAG source {source_example_id} session api_call must be strings"
            )
        if api_calls:
            documents.append(
                IndexDocument(
                    document_id=f"{source_example_id}:api_history:{session_index}",
                    source_example_id=source_example_id,
                    user_id=user_id,
                    kind="api_history",
                    text=(
                        "[System Summary] API Calls executed in this session: "
                        + json.dumps(api_calls, ensure_ascii=False, separators=(",", ":"))
                    ),
                    metadata={"session_index": session_index},
                )
            )
    return documents


def _prepare_index(
    records: Sequence[CanonicalPreparedRecord],
) -> tuple[list[IndexDocument], dict[str, dict[str, Any]]]:
    sources: dict[str, dict[str, Any]] = {}
    documents: list[IndexDocument] = []
    for record in records:
        source, user_id = _source_payload(record)
        source_sha = _sha256(_canonical_bytes(source))
        existing = sources.get(record.source_example_id)
        if existing is not None:
            if existing["source_sha256"] != source_sha or existing["user_id"] != user_id:
                raise MethodContractError(
                    f"RAG source_example_id has conflicting payloads: {record.source_example_id}"
                )
            continue
        source_documents = _index_documents(record.source_example_id, user_id, source)
        sources[record.source_example_id] = {
            "document_count": len(source_documents),
            "source_sha256": source_sha,
            "user_id": user_id,
        }
        documents.extend(source_documents)
    return documents, sources


def _state_files(state_dir: Path) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(state_dir.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(state_dir).as_posix()
        try:
            info = os.lstat(path)
        except OSError as exc:
            raise MethodContractError(f"cannot inspect RAG state file: {path}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise MethodContractError(f"RAG backend state may not contain symlinks: {path}")
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode):
            raise MethodContractError(f"RAG backend state must contain regular files: {path}")
        if relative == STATE_MANIFEST:
            continue
        try:
            payload = admit_regular_file(
                path, label=f"RAG backend state file {relative}"
            ).payload
        except FileAdmissionError as exc:
            raise MethodContractError(str(exc)) from exc
        files[relative] = {"bytes": len(payload), "sha256": _sha256(payload)}
    return files


def _write_manifest_exclusive(path: Path, value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    return _sha256(payload)


def _read_state_manifest(
    context: MethodRunContext,
    state: MethodStateHandle,
    state_relative: PurePosixPath,
) -> dict[str, Any]:
    expected_path = method_state_path(context, state_relative)
    if state.method_id != METHOD_ID or state.path != expected_path:
        raise MethodContractError("RAG state handle does not identify canonical RAG state")
    if state.dataset_manifest_sha256 != context.dataset_manifest_sha256:
        raise MethodContractError("RAG state dataset manifest binding changed")
    manifest_path = state.path / STATE_MANIFEST
    try:
        payload = admit_regular_file(
            manifest_path, label="RAG state manifest"
        ).payload
    except FileAdmissionError as exc:
        raise MethodContractError(str(exc)) from exc
    expected_sha = state.metadata.get("state_manifest_sha256")
    if not isinstance(expected_sha, str) or _sha256(payload) != expected_sha:
        raise MethodContractError("RAG state manifest SHA-256 mismatch")
    try:
        manifest = decode_strict_json(payload, label="RAG state manifest")
    except StrictJSONError as exc:
        raise MethodContractError(str(exc)) from exc
    if not isinstance(manifest, dict):
        raise MethodContractError("RAG state manifest must be an object")
    if (
        manifest.get("format_version") != STATE_FORMAT_VERSION
        or manifest.get("method_id") != METHOD_ID
        or manifest.get("dataset_manifest_sha256") != context.dataset_manifest_sha256
        or manifest.get("records_sha256") != state.records_sha256
    ):
        raise MethodContractError("RAG state manifest binding changed")
    expected_files = manifest.get("backend_artifacts")
    if not isinstance(expected_files, dict) or _state_files(state.path) != expected_files:
        raise MethodContractError("RAG backend state files are missing or tampered")
    return manifest


def _normalize_retrieved(
    values: Any,
    request: RetrievalRequest,
) -> tuple[RetrievedContext, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise MethodContractError("RAG retrieve() must return a sequence of scoped contexts")
    contexts = []
    for value in values:
        if isinstance(value, RetrievedContext):
            context = value
        elif isinstance(value, Mapping):
            try:
                context = RetrievedContext(
                    text=value["text"],
                    source_example_id=value["source_example_id"],
                    user_id=value["user_id"],
                    document_id=str(value.get("document_id", "")),
                    score=value.get("score"),
                    metadata=copy.deepcopy(dict(value.get("metadata", {}))),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise MethodContractError("RAG retrieved context is malformed") from exc
        else:
            raise MethodContractError("RAG retrieved context must be a mapping")
        if not isinstance(context.text, str) or not context.text:
            raise MethodContractError("RAG retrieved context text must be non-empty")
        if (
            context.source_example_id != request.source_example_id
            or context.user_id != request.user_id
        ):
            raise MethodContractError("RAG backend returned cross-identity context")
        contexts.append(context)
    return tuple(contexts)


def _format_context(contexts: Sequence[RetrievedContext]) -> str:
    return "\n\n".join(
        f"[Retrieved Context {index}]\n{context.text}"
        for index, context in enumerate(contexts, start=1)
    )


class RAGAdapter:
    """Build a verified index and require identity-scoped retrieval for inference."""

    method_id = METHOD_ID

    def __init__(self, state_relative: str = "method_state/rag") -> None:
        self._state_relative = PurePosixPath(state_relative)

    async def build(
        self,
        records: Sequence[CanonicalPreparedRecord],
        context: MethodRunContext,
        backend: RAGRuntime,
    ) -> MethodStateHandle:
        runtime = _runtime(backend)
        validated = validate_prepared_records(records)
        if not validated:
            raise MethodContractError("RAG index build requires prepared records")
        state_dir = method_state_path(context, self._state_relative)
        if state_dir.exists() or state_dir.is_symlink():
            raise MethodContractError(f"refusing to overwrite RAG state: {state_dir}")
        state_dir.parent.mkdir(parents=True, exist_ok=True)
        state_dir.mkdir()
        try:
            documents, sources = _prepare_index(validated)
            backend_metadata = await await_if_needed(
                runtime.index_backend.build_index(tuple(documents), state_dir)
            )
            if backend_metadata is None:
                backend_metadata = {}
            if not isinstance(backend_metadata, Mapping):
                raise MethodContractError("RAG build_index metadata must be a mapping")
            _canonical_bytes(dict(backend_metadata))
            artifacts = _state_files(state_dir)
            if not artifacts:
                raise MethodContractError("RAG backend produced no persistent index state")
            preliminary = bind_method_state(
                method_id=METHOD_ID,
                context=context,
                relative_path=self._state_relative,
                records=validated,
            )
            manifest = {
                "backend_artifacts": artifacts,
                "backend_metadata": copy.deepcopy(dict(backend_metadata)),
                "dataset_manifest_sha256": context.dataset_manifest_sha256,
                "documents_sha256": _sha256(
                    _canonical_bytes([document.as_mapping() for document in documents])
                ),
                "format_version": STATE_FORMAT_VERSION,
                "instance_ids": [record.instance_id for record in validated],
                "method_id": METHOD_ID,
                "records_sha256": preliminary.records_sha256,
                "sources": sources,
            }
            manifest_sha = _write_manifest_exclusive(
                state_dir / STATE_MANIFEST, manifest
            )
            return bind_method_state(
                method_id=METHOD_ID,
                context=context,
                relative_path=self._state_relative,
                records=validated,
                metadata={
                    "document_count": len(documents),
                    "source_count": len(sources),
                    "state_manifest_sha256": manifest_sha,
                },
            )
        except BaseException:
            shutil.rmtree(state_dir)
            raise

    async def infer(
        self,
        record: CanonicalPreparedRecord,
        context: MethodRunContext,
        state: MethodStateHandle | None,
        backend: RAGRuntime,
    ) -> dict[str, Any]:
        runtime = _runtime(backend)
        canonical = validate_prepared_record(record.as_mapping())
        if state is None:
            raise MethodContractError("RAG inference requires built index state")
        try:
            with admit_directory(state.path, label="RAG state directory") as state_directory:
                manifest = _read_state_manifest(context, state, self._state_relative)
                if canonical.instance_id not in manifest.get("instance_ids", []):
                    raise MethodContractError(
                        "RAG record was not included in the built index state"
                    )
                sources = manifest.get("sources")
                source_spec = (
                    sources.get(canonical.source_example_id)
                    if isinstance(sources, dict)
                    else None
                )
                if not isinstance(source_spec, dict):
                    raise MethodContractError(
                        "RAG source was not included in the built index state"
                    )
                source, user_id = _source_payload(canonical)
                if (
                    _sha256(_canonical_bytes(source))
                    != source_spec.get("source_sha256")
                    or user_id != source_spec.get("user_id")
                ):
                    raise MethodContractError(
                        "RAG inference source payload differs from index state"
                    )
                retrieval_request = RetrievalRequest(
                    query=canonical.query,
                    source_example_id=canonical.source_example_id,
                    user_id=user_id,
                    top_k=runtime.top_k,
                )
                retrieved_value = await await_if_needed(
                    runtime.index_backend.retrieve(
                        state_directory.descriptor_path,
                        retrieval_request,
                    )
                )
        except FileAdmissionError as exc:
            raise MethodContractError(str(exc)) from exc
        retrieved = _normalize_retrieved(retrieved_value, retrieval_request)
        generation_request = RAGGenerationRequest(
            model_name=context.model_name,
            tools_schema=context.tools_schema,
            reasoning_effort=context.reasoning_effort,
            retrieved=retrieved,
            retrieved_context=_format_context(retrieved),
        )
        generated = await await_if_needed(
            runtime.generator(canonical.as_mapping(), generation_request)
        )
        if isinstance(generated, str):
            prediction: dict[str, Any] = {"prediction": generated}
        elif isinstance(generated, Mapping):
            prediction = copy.deepcopy(dict(generated))
        else:
            raise MethodContractError("RAG generator must return a mapping or string")
        retrieved_payload = [context.as_mapping() for context in retrieved]
        if "retrieved_context" in prediction and prediction["retrieved_context"] != retrieved_payload:
            raise MethodContractError("RAG generator changed retrieved_context")
        prediction.update(
            {
                "retrieval_query": canonical.query,
                "retrieval_scope": {
                    "source_example_id": canonical.source_example_id,
                    "user_id": user_id,
                },
                "retrieval_status": "complete",
                "retrieved_context": retrieved_payload,
            }
        )
        return canonical_prediction(
            canonical,
            method_id=METHOD_ID,
            prediction=prediction,
        )
