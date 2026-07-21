"""Canonical Mem0 adapter with explicit namespace and state bindings.

No provider client is created here.  A memory backend and a generator are
injected through :class:`Mem0Runtime`, which keeps imports and tests offline.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
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
    admit_regular_file,
    decode_strict_json,
)


METHOD_ID = "mem0"
STATE_FORMAT_VERSION = 1
STATE_MANIFEST = "state_manifest.json"
_NAMESPACE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


class Mem0MemoryBackend(Protocol):
    """Small injected surface compatible with the Mem0 add/search lifecycle."""

    def add(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        user_id: str,
        metadata: Mapping[str, Any],
    ) -> Any | Awaitable[Any]: ...

    def search(
        self,
        *,
        query: str,
        user_id: str,
        filters: Mapping[str, Any],
    ) -> Any | Awaitable[Any]: ...


@dataclass(frozen=True)
class Mem0GenerationRequest:
    """Identity-scoped retrieval context passed to the injected generator."""

    model_name: str
    tools_schema: tuple[Mapping[str, Any], ...]
    reasoning_effort: str | None
    namespace: str
    memory_user_id: str
    retrieval_query: str
    turn: str
    retrieved_memories: tuple[Mapping[str, Any], ...]
    retrieved_memory_text: str


Mem0GenerationValue = Mapping[str, Any] | str
Mem0Generator = Callable[
    [Mapping[str, Any], Mem0GenerationRequest],
    Mem0GenerationValue | Awaitable[Mem0GenerationValue],
]


@dataclass(frozen=True)
class Mem0Runtime:
    """All provider behavior required by Mem0 is supplied explicitly."""

    memory_backend: Mem0MemoryBackend
    generator: Mem0Generator

    def __post_init__(self) -> None:
        if not callable(getattr(self.memory_backend, "add", None)):
            raise MethodContractError("Mem0 memory backend must implement add()")
        if not callable(getattr(self.memory_backend, "search", None)):
            raise MethodContractError("Mem0 memory backend must implement search()")
        if not callable(self.generator):
            raise MethodContractError("Mem0 generator must be callable")


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MethodContractError(
            "Mem0 source and state values must be JSON serializable"
        ) from exc


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validate_namespace(namespace: str) -> str:
    if not isinstance(namespace, str) or _NAMESPACE.fullmatch(namespace) is None:
        raise MethodContractError(
            "Mem0 namespace must match [A-Za-z0-9][A-Za-z0-9._-]{0,127}"
        )
    return namespace


def _memory_user_id(namespace: str, source_example_id: str) -> str:
    return f"{namespace}::{source_example_id}"


def _source_payload(record: CanonicalPreparedRecord) -> dict[str, Any]:
    source = record.as_mapping().get("source_example")
    if not isinstance(source, Mapping):
        raise MethodContractError(
            f"Mem0 record {record.instance_id} lacks source_example"
        )
    source_value = copy.deepcopy(dict(source))
    embedded_id = source_value.get("example_id")
    if embedded_id is not None and str(embedded_id) != record.source_example_id:
        raise MethodContractError(
            f"Mem0 source identity drift for {record.instance_id}"
        )
    return source_value


def _session_batches(
    source_example_id: str,
    source: Mapping[str, Any],
) -> list[tuple[list[dict[str, str]], dict[str, Any]]]:
    sessions = source.get("sessions", [])
    if not isinstance(sessions, list):
        raise MethodContractError(
            f"Mem0 source {source_example_id} sessions must be a list"
        )
    batches: list[tuple[list[dict[str, str]], dict[str, Any]]] = []
    for session_index, session in enumerate(sessions):
        if not isinstance(session, Mapping):
            raise MethodContractError(
                f"Mem0 source {source_example_id} contains an invalid session"
            )
        dialogue = session.get("dialogue", [])
        if not isinstance(dialogue, list):
            raise MethodContractError(
                f"Mem0 source {source_example_id} session dialogue must be a list"
            )
        messages: list[dict[str, str]] = []
        for turn in dialogue:
            if not isinstance(turn, Mapping):
                raise MethodContractError(
                    f"Mem0 source {source_example_id} contains an invalid dialogue turn"
                )
            role = turn.get("role")
            content = turn.get("message", turn.get("content"))
            if not isinstance(role, str) or not role.strip():
                raise MethodContractError(
                    f"Mem0 source {source_example_id} dialogue role must be non-empty"
                )
            if not isinstance(content, str):
                raise MethodContractError(
                    f"Mem0 source {source_example_id} dialogue message must be a string"
                )
            messages.append({"role": role.lower(), "content": content})

        api_calls = session.get("api_call", [])
        if not isinstance(api_calls, list):
            raise MethodContractError(
                f"Mem0 source {source_example_id} session api_call must be a list"
            )
        _canonical_bytes(api_calls)
        if api_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": (
                        "[System Summary] API Calls executed in this session: "
                        f"{str(api_calls)}"
                    ),
                }
            )
        if messages:
            batches.append(
                (
                    messages,
                    {
                        "dialogue_id": session.get("dialogue_id"),
                        "session_index": session_index,
                    },
                )
            )
    return batches


def _prepare_sources(
    records: Sequence[CanonicalPreparedRecord],
) -> dict[str, dict[str, Any]]:
    sources: dict[str, dict[str, Any]] = {}
    for record in records:
        source = _source_payload(record)
        source_sha = _sha256(_canonical_bytes(source))
        existing = sources.get(record.source_example_id)
        if existing is not None:
            if existing["source_sha256"] != source_sha:
                raise MethodContractError(
                    "Mem0 source_example_id has conflicting payloads: "
                    f"{record.source_example_id}"
                )
            continue
        batches = _session_batches(record.source_example_id, source)
        sources[record.source_example_id] = {
            "batches": batches,
            "source_sha256": source_sha,
        }
    return sources


def _write_manifest_exclusive(path: Path, value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
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
    namespace: str,
) -> dict[str, Any]:
    expected_path = method_state_path(context, state_relative)
    if state.method_id != METHOD_ID or state.path != expected_path:
        raise MethodContractError(
            "Mem0 state handle does not identify canonical Mem0 state"
        )
    if state.dataset_manifest_sha256 != context.dataset_manifest_sha256:
        raise MethodContractError("Mem0 state dataset manifest binding changed")
    if state.metadata.get("namespace") != namespace:
        raise MethodContractError("Mem0 state namespace binding changed")
    manifest_path = state.path / STATE_MANIFEST
    try:
        payload = admit_regular_file(
            manifest_path, label="Mem0 state manifest"
        ).payload
    except FileAdmissionError as exc:
        raise MethodContractError(
            f"Mem0 state manifest is missing or unsafe: {exc}"
        ) from exc
    expected_sha = state.metadata.get("state_manifest_sha256")
    if not isinstance(expected_sha, str) or _sha256(payload) != expected_sha:
        raise MethodContractError("Mem0 state manifest SHA-256 mismatch")
    try:
        manifest = decode_strict_json(payload, label="Mem0 state manifest")
    except StrictJSONError as exc:
        raise MethodContractError(str(exc)) from exc
    if not isinstance(manifest, dict):
        raise MethodContractError("Mem0 state manifest must be an object")
    if (
        manifest.get("format_version") != STATE_FORMAT_VERSION
        or manifest.get("method_id") != METHOD_ID
        or manifest.get("namespace") != namespace
        or manifest.get("dataset_manifest_sha256")
        != context.dataset_manifest_sha256
        or manifest.get("records_sha256") != state.records_sha256
    ):
        raise MethodContractError("Mem0 state manifest binding changed")
    return manifest


def _normalize_search_response(
    response: Any,
    *,
    namespace: str,
    source_example_id: str,
    memory_user_id: str,
) -> tuple[dict[str, Any], ...]:
    if isinstance(response, Mapping):
        values = response.get("results")
    else:
        values = response
    if not isinstance(values, Sequence) or isinstance(
        values, (str, bytes, bytearray)
    ):
        raise MethodContractError(
            "Mem0 search() must return a result sequence or {'results': sequence}"
        )
    memories: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, Mapping):
            raise MethodContractError("Mem0 search result must be a mapping")
        memory = copy.deepcopy(dict(value))
        text = memory.get("memory")
        if not isinstance(text, str) or not text:
            raise MethodContractError(
                "Mem0 search result memory must be a non-empty string"
            )
        result_user_id = memory.get("user_id")
        if result_user_id is not None and result_user_id != memory_user_id:
            raise MethodContractError("Mem0 backend returned cross-identity memory")
        metadata = memory.get("metadata", {})
        if metadata is not None and not isinstance(metadata, Mapping):
            raise MethodContractError("Mem0 search result metadata must be a mapping")
        metadata = dict(metadata or {})
        if metadata.get("namespace", namespace) != namespace:
            raise MethodContractError("Mem0 backend returned cross-namespace memory")
        if (
            metadata.get("source_example_id", source_example_id)
            != source_example_id
        ):
            raise MethodContractError("Mem0 backend returned cross-source memory")
        memories.append(memory)
    return tuple(memories)


def _format_memories(memories: Sequence[Mapping[str, Any]]) -> str:
    if not memories:
        return "[No retrieved memories]"
    return "\n".join(
        f"{index}. {memory['memory']}"
        for index, memory in enumerate(memories, start=1)
    )


class Mem0Adapter:
    """Build namespaced remote memories and require verified state at inference."""

    method_id = METHOD_ID

    def __init__(
        self,
        *,
        namespace: str,
        state_root: str = "method_state/mem0",
    ) -> None:
        self._namespace = _validate_namespace(namespace)
        self._state_relative = PurePosixPath(state_root) / self._namespace

    @property
    def namespace(self) -> str:
        return self._namespace

    async def build(
        self,
        records: Sequence[CanonicalPreparedRecord],
        context: MethodRunContext,
        backend: Mem0Runtime,
    ) -> MethodStateHandle:
        runtime = (
            backend if isinstance(backend, Mem0Runtime) else Mem0Runtime(**backend)
        )
        validated = validate_prepared_records(records)
        if not validated:
            raise MethodContractError("Mem0 build requires prepared records")
        sources = _prepare_sources(validated)
        state_dir = method_state_path(context, self._state_relative)
        if state_dir.exists() or state_dir.is_symlink():
            raise MethodContractError(f"refusing to overwrite Mem0 state: {state_dir}")
        state_dir.parent.mkdir(parents=True, exist_ok=True)
        state_dir.mkdir()
        try:
            source_manifest: dict[str, dict[str, Any]] = {}
            batch_count = 0
            for source_example_id, source_spec in sources.items():
                memory_user_id = _memory_user_id(
                    self._namespace, source_example_id
                )
                batches = source_spec["batches"]
                for messages, session_metadata in batches:
                    metadata = {
                        "dataset_manifest_sha256": (
                            context.dataset_manifest_sha256
                        ),
                        "namespace": self._namespace,
                        "source_example_id": source_example_id,
                        **session_metadata,
                    }
                    await await_if_needed(
                        runtime.memory_backend.add(
                            tuple(copy.deepcopy(messages)),
                            user_id=memory_user_id,
                            metadata=metadata,
                        )
                    )
                    batch_count += 1
                source_manifest[source_example_id] = {
                    "memory_user_id": memory_user_id,
                    "session_batch_count": len(batches),
                    "source_sha256": source_spec["source_sha256"],
                }

            preliminary = bind_method_state(
                method_id=METHOD_ID,
                context=context,
                relative_path=self._state_relative,
                records=validated,
            )
            manifest = {
                "dataset_manifest_sha256": context.dataset_manifest_sha256,
                "format_version": STATE_FORMAT_VERSION,
                "instance_ids": [record.instance_id for record in validated],
                "method_id": METHOD_ID,
                "namespace": self._namespace,
                "records_sha256": preliminary.records_sha256,
                "sources": source_manifest,
            }
            manifest_sha = _write_manifest_exclusive(
                state_dir / STATE_MANIFEST,
                manifest,
            )
            return bind_method_state(
                method_id=METHOD_ID,
                context=context,
                relative_path=self._state_relative,
                records=validated,
                metadata={
                    "namespace": self._namespace,
                    "session_batch_count": batch_count,
                    "source_count": len(source_manifest),
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
        backend: Mem0Runtime,
    ) -> dict[str, Any]:
        runtime = (
            backend if isinstance(backend, Mem0Runtime) else Mem0Runtime(**backend)
        )
        canonical = validate_prepared_record(record.as_mapping())
        if state is None:
            raise MethodContractError("Mem0 inference requires built state")
        manifest = _read_state_manifest(
            context,
            state,
            self._state_relative,
            self._namespace,
        )
        if canonical.instance_id not in manifest.get("instance_ids", []):
            raise MethodContractError(
                "Mem0 record was not included in the built state"
            )
        sources = manifest.get("sources")
        source_spec = (
            sources.get(canonical.source_example_id)
            if isinstance(sources, dict)
            else None
        )
        if not isinstance(source_spec, dict):
            raise MethodContractError(
                "Mem0 source was not included in the built state"
            )
        source = _source_payload(canonical)
        if _sha256(_canonical_bytes(source)) != source_spec.get("source_sha256"):
            raise MethodContractError(
                "Mem0 inference source payload differs from built state"
            )
        memory_user_id = _memory_user_id(
            self._namespace,
            canonical.source_example_id,
        )
        if source_spec.get("memory_user_id") != memory_user_id:
            raise MethodContractError("Mem0 source namespace binding changed")

        response = await await_if_needed(
            runtime.memory_backend.search(
                query=canonical.query,
                user_id=memory_user_id,
                filters={"user_id": memory_user_id},
            )
        )
        memories = _normalize_search_response(
            response,
            namespace=self._namespace,
            source_example_id=canonical.source_example_id,
            memory_user_id=memory_user_id,
        )
        generation_request = Mem0GenerationRequest(
            model_name=context.model_name,
            tools_schema=context.tools_schema,
            reasoning_effort=context.reasoning_effort,
            namespace=self._namespace,
            memory_user_id=memory_user_id,
            retrieval_query=canonical.query,
            turn=canonical.turn,
            retrieved_memories=tuple(copy.deepcopy(memory) for memory in memories),
            retrieved_memory_text=_format_memories(memories),
        )
        generated = await await_if_needed(
            runtime.generator(canonical.as_mapping(), generation_request)
        )
        if isinstance(generated, str):
            prediction: dict[str, Any] = {"prediction": generated}
        elif isinstance(generated, Mapping):
            prediction = copy.deepcopy(dict(generated))
        else:
            raise MethodContractError(
                "Mem0 generator must return a mapping or string"
            )
        retrieved_payload = [copy.deepcopy(memory) for memory in memories]
        expected_scope = {
            "namespace": self._namespace,
            "source_example_id": canonical.source_example_id,
            "user_id": memory_user_id,
        }
        protected = {
            "retrieval_query": canonical.query,
            "retrieval_scope": expected_scope,
            "retrieved_memories": retrieved_payload,
        }
        for key, expected in protected.items():
            if key in prediction and prediction[key] != expected:
                raise MethodContractError(
                    f"Mem0 generator changed canonical retrieval field {key!r}"
                )
            prediction[key] = expected
        prediction["retrieval_status"] = "complete"
        return canonical_prediction(
            canonical,
            method_id=METHOD_ID,
            prediction=prediction,
        )
