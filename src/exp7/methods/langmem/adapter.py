"""Canonical, dependency-free LangMem adapter with injected backends."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Any, Awaitable, ClassVar, Mapping, Protocol, Sequence

from exp7.methods.contracts import (
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
from exp7.provenance.admission import (
    FileAdmissionError,
    StrictJSONError,
    admit_regular_file,
    decode_strict_json,
)


class LangMemAdapterError(MethodContractError):
    """Raised when LangMem state or an injected backend violates the adapter contract."""


@dataclass(frozen=True)
class LangMemBuildItem:
    """One unique source dialogue presented to the injected memory builder."""

    source_example_id: str
    instance_ids: tuple[str, ...]
    source_record: Mapping[str, Any]
    source_sha256: str
    text: str


@dataclass(frozen=True)
class LangMemInferenceRequest:
    """Provider-neutral model request augmented with retrieved memories."""

    record: Mapping[str, Any]
    turn: str
    query: str
    retrieved_memories: tuple[Mapping[str, Any], ...]
    model_name: str
    tools_schema: tuple[Mapping[str, Any], ...]
    reasoning_effort: str | None


class LangMemEmbedder(Protocol):
    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> Sequence[Sequence[float]] | Awaitable[Sequence[Sequence[float]]]:
        """Embed source documents for snapshot construction."""

    def embed_query(
        self,
        text: str,
    ) -> Sequence[float] | Awaitable[Sequence[float]]:
        """Embed one inference query."""


class LangMemStore(Protocol):
    def build(
        self,
        *,
        items: Sequence[LangMemBuildItem],
        embeddings: Sequence[Sequence[float]],
    ) -> Any | Awaitable[Any]:
        """Return a JSON-serializable snapshot."""

    def search(
        self,
        snapshot: Any,
        *,
        query: str,
        query_embedding: Sequence[float],
        limit: int,
    ) -> Sequence[Mapping[str, Any] | str] | Awaitable[
        Sequence[Mapping[str, Any] | str]
    ]:
        """Retrieve memories from a previously built snapshot."""


class LangMemModel(Protocol):
    def generate(
        self,
        request: LangMemInferenceRequest,
    ) -> Mapping[str, Any] | str | Awaitable[Mapping[str, Any] | str]:
        """Generate one service-API prediction."""


@dataclass(frozen=True)
class LangMemBackend:
    """All external LangMem behavior is injected explicitly."""

    store: LangMemStore
    embedder: LangMemEmbedder
    model: LangMemModel


@dataclass(frozen=True)
class _LoadedState:
    manifest: Mapping[str, Any]
    snapshot: Any


def _canonical_json_bytes(value: Any, *, label: str) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise LangMemAdapterError(f"{label} must be JSON serializable") from exc


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_sha256(value: Any, *, label: str) -> str:
    return _sha256(_canonical_json_bytes(value, label=label))


def _record_sha256(record: CanonicalPreparedRecord) -> str:
    return _json_sha256(record.as_mapping(), label="prepared record")


def _stable_fingerprint(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_regular_file(path: Path, *, label: str) -> bytes:
    try:
        return admit_regular_file(path, label=label).payload
    except FileAdmissionError as exc:
        raise LangMemAdapterError(str(exc)) from exc


def _write_exclusive(path: Path, payload: bytes) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _decode_json(payload: bytes, *, label: str) -> Any:
    try:
        return decode_strict_json(payload, label=label)
    except StrictJSONError as exc:
        raise LangMemAdapterError(str(exc)) from exc


def _state_path(context: MethodRunContext, relative_path: str) -> Path:
    try:
        return method_state_path(context, relative_path)
    except MethodContractError as exc:
        raise LangMemAdapterError(str(exc)) from exc


def _normalize_vector(value: Any, *, label: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise LangMemAdapterError(f"{label} must be a numeric sequence")
    vector: list[float] = []
    for component in value:
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            raise LangMemAdapterError(f"{label} must contain only numbers")
        number = float(component)
        if not math.isfinite(number):
            raise LangMemAdapterError(f"{label} must contain only finite numbers")
        vector.append(number)
    if not vector:
        raise LangMemAdapterError(f"{label} may not be empty")
    return tuple(vector)


def _build_items(
    records: Sequence[CanonicalPreparedRecord],
) -> tuple[LangMemBuildItem, ...]:
    grouped: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for record in records:
        payload = record.as_mapping()
        source = payload.get("source_example")
        if not isinstance(source, Mapping):
            raise LangMemAdapterError(
                "LangMem build requires a source_example mapping on every prepared record"
            )
        source_copy = copy.deepcopy(dict(source))
        source_sha256 = _json_sha256(source_copy, label="source example")
        existing = grouped.get(record.source_example_id)
        if existing is None:
            order.append(record.source_example_id)
            grouped[record.source_example_id] = {
                "instance_ids": [record.instance_id],
                "source": source_copy,
                "source_sha256": source_sha256,
            }
        else:
            if existing["source_sha256"] != source_sha256:
                raise LangMemAdapterError(
                    "prepared records disagree on source_example content for "
                    f"{record.source_example_id!r}"
                )
            existing["instance_ids"].append(record.instance_id)

    items: list[LangMemBuildItem] = []
    for source_example_id in order:
        value = grouped[source_example_id]
        source_record = value["source"]
        text = _canonical_json_bytes(
            source_record,
            label="source example",
        ).decode("utf-8")
        items.append(
            LangMemBuildItem(
                source_example_id=source_example_id,
                instance_ids=tuple(value["instance_ids"]),
                source_record=source_record,
                source_sha256=value["source_sha256"],
                text=text,
            )
        )
    return tuple(items)


def _manifest_binding(
    *,
    records: Sequence[CanonicalPreparedRecord],
    items: Sequence[LangMemBuildItem],
    dataset_manifest_sha256: str,
    contract_records_sha256: str,
    snapshot_payload: bytes,
) -> dict[str, Any]:
    instance_ids = [record.instance_id for record in records]
    record_sha256_by_instance_id = {
        record.instance_id: _record_sha256(record) for record in records
    }
    prepared_binding = [
        {
            "instance_id": instance_id,
            "sha256": record_sha256_by_instance_id[instance_id],
        }
        for instance_id in instance_ids
    ]
    source_binding = [
        {
            "instance_ids": list(item.instance_ids),
            "sha256": item.source_sha256,
            "source_example_id": item.source_example_id,
        }
        for item in items
    ]
    return {
        "contract_records_sha256": contract_records_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "format_version": 1,
        "instance_ids": instance_ids,
        "instance_ids_sha256": _json_sha256(instance_ids, label="instance IDs"),
        "method_id": "langmem",
        "prepared_records_sha256": _json_sha256(
            prepared_binding,
            label="prepared record binding",
        ),
        "record_sha256_by_instance_id": record_sha256_by_instance_id,
        "snapshot": {
            "bytes": len(snapshot_payload),
            "path": "snapshot.json",
            "sha256": _sha256(snapshot_payload),
        },
        "source_records": source_binding,
        "source_records_sha256": _json_sha256(
            source_binding,
            label="source record binding",
        ),
    }


def _normalize_memories(value: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise LangMemAdapterError("store search must return a sequence of memories")
    memories: list[Mapping[str, Any]] = []
    for item in value:
        if isinstance(item, str):
            normalized: dict[str, Any] = {"content": item}
        elif isinstance(item, Mapping):
            normalized = copy.deepcopy(dict(item))
            content = normalized.get("content", normalized.get("memory"))
            if not isinstance(content, str) or not content:
                raise LangMemAdapterError(
                    "retrieved memory mappings require non-empty content"
                )
            normalized["content"] = content
        else:
            raise LangMemAdapterError(
                "retrieved memories must be strings or mappings"
            )
        _canonical_json_bytes(normalized, label="retrieved memory")
        memories.append(normalized)
    return tuple(memories)


@dataclass(frozen=True)
class LangMemAdapter:
    """Canonical LangMem build/retrieve/infer lifecycle."""

    method_id: ClassVar[str] = "langmem"
    relative_state_path: str = "method_state/langmem"
    top_k: int = 5

    def __post_init__(self) -> None:
        if not isinstance(self.top_k, int) or isinstance(self.top_k, bool) or self.top_k <= 0:
            raise LangMemAdapterError("top_k must be a positive integer")

    async def build(
        self,
        records: Sequence[CanonicalPreparedRecord],
        context: MethodRunContext,
        backend: LangMemBackend,
    ) -> MethodStateHandle:
        validated = validate_prepared_records(records)
        if not validated:
            raise LangMemAdapterError("LangMem build requires at least one prepared record")
        final_path = _state_path(context, self.relative_state_path)
        if final_path.exists() or final_path.is_symlink():
            raise LangMemAdapterError(f"LangMem state already exists: {final_path}")
        final_path.parent.mkdir(parents=True, exist_ok=True)
        final_path = _state_path(context, self.relative_state_path)

        items = _build_items(validated)
        raw_embeddings = await await_if_needed(
            backend.embedder.embed_documents(tuple(item.text for item in items))
        )
        if isinstance(raw_embeddings, (str, bytes)) or not isinstance(
            raw_embeddings, Sequence
        ):
            raise LangMemAdapterError("embed_documents must return a sequence")
        if len(raw_embeddings) != len(items):
            raise LangMemAdapterError(
                "embed_documents result count does not match source record count"
            )
        embeddings = tuple(
            _normalize_vector(value, label=f"document embedding {index}")
            for index, value in enumerate(raw_embeddings)
        )
        snapshot = await await_if_needed(
            backend.store.build(items=items, embeddings=embeddings)
        )
        snapshot_payload = _canonical_json_bytes(snapshot, label="LangMem snapshot")

        temporary = Path(
            tempfile.mkdtemp(prefix=".langmem-build-", dir=final_path.parent)
        )
        committed = False
        try:
            temporary_relative = temporary.relative_to(context.run_dir).as_posix()
            temporary_handle = bind_method_state(
                method_id=self.method_id,
                context=context,
                relative_path=temporary_relative,
                records=validated,
            )
            state_manifest = _manifest_binding(
                records=validated,
                items=items,
                dataset_manifest_sha256=context.dataset_manifest_sha256,
                contract_records_sha256=temporary_handle.records_sha256,
                snapshot_payload=snapshot_payload,
            )
            manifest_payload = _canonical_json_bytes(
                state_manifest,
                label="LangMem state manifest",
            )
            _write_exclusive(temporary / "snapshot.json", snapshot_payload)
            _write_exclusive(temporary / "state_manifest.json", manifest_payload)
            temporary.rename(final_path)
            committed = True
        finally:
            if not committed and temporary.exists():
                shutil.rmtree(temporary)

        return bind_method_state(
            method_id=self.method_id,
            context=context,
            relative_path=self.relative_state_path,
            records=validated,
            metadata={
                "instance_count": len(validated),
                "prepared_records_sha256": state_manifest[
                    "prepared_records_sha256"
                ],
                "snapshot_sha256": state_manifest["snapshot"]["sha256"],
                "source_count": len(items),
                "state_manifest_sha256": _sha256(manifest_payload),
            },
        )

    def _load_state(
        self,
        record: CanonicalPreparedRecord,
        context: MethodRunContext,
        state: MethodStateHandle | None,
    ) -> _LoadedState:
        if state is None:
            raise LangMemAdapterError("LangMem inference requires built state")
        if state.method_id != self.method_id:
            raise LangMemAdapterError("method state is not LangMem state")
        if state.dataset_manifest_sha256 != context.dataset_manifest_sha256:
            raise LangMemAdapterError(
                "LangMem state is bound to a different dataset manifest"
            )
        try:
            relative = state.path.relative_to(context.run_dir).as_posix()
        except ValueError as exc:
            raise LangMemAdapterError("LangMem state path escapes run_dir") from exc
        verified_path = _state_path(context, relative)
        try:
            info = os.lstat(verified_path)
        except OSError as exc:
            raise LangMemAdapterError("LangMem state directory is missing") from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise LangMemAdapterError("LangMem state must be a non-symlink directory")
        if verified_path != state.path:
            raise LangMemAdapterError("LangMem state path changed after binding")

        manifest_payload = _read_regular_file(
            verified_path / "state_manifest.json",
            label="LangMem state manifest",
        )
        expected_manifest_sha256 = state.metadata.get("state_manifest_sha256")
        if (
            not isinstance(expected_manifest_sha256, str)
            or _sha256(manifest_payload) != expected_manifest_sha256
        ):
            raise LangMemAdapterError("LangMem state manifest SHA-256 mismatch")
        manifest = _decode_json(
            manifest_payload,
            label="LangMem state manifest",
        )
        if not isinstance(manifest, dict):
            raise LangMemAdapterError("LangMem state manifest must be an object")
        if manifest.get("format_version") != 1 or manifest.get("method_id") != self.method_id:
            raise LangMemAdapterError("unsupported LangMem state manifest")
        if manifest.get("dataset_manifest_sha256") != context.dataset_manifest_sha256:
            raise LangMemAdapterError("LangMem manifest dataset binding mismatch")
        if manifest.get("contract_records_sha256") != state.records_sha256:
            raise LangMemAdapterError("LangMem manifest record binding mismatch")

        instance_ids = manifest.get("instance_ids")
        record_digests = manifest.get("record_sha256_by_instance_id")
        if not isinstance(instance_ids, list) or not all(
            isinstance(value, str) for value in instance_ids
        ):
            raise LangMemAdapterError("LangMem manifest has invalid instance IDs")
        if manifest.get("instance_ids_sha256") != _json_sha256(
            instance_ids,
            label="instance IDs",
        ):
            raise LangMemAdapterError("LangMem instance ID binding mismatch")
        if not isinstance(record_digests, dict):
            raise LangMemAdapterError("LangMem manifest lacks record digests")
        if record.instance_id not in instance_ids:
            raise LangMemAdapterError("prepared record is not present in LangMem state")
        if record_digests.get(record.instance_id) != _record_sha256(record):
            raise LangMemAdapterError("prepared record content differs from LangMem state")

        prepared_binding = [
            {"instance_id": instance_id, "sha256": record_digests.get(instance_id)}
            for instance_id in instance_ids
        ]
        if manifest.get("prepared_records_sha256") != _json_sha256(
            prepared_binding,
            label="prepared record binding",
        ):
            raise LangMemAdapterError("LangMem prepared-record binding mismatch")
        source_records = manifest.get("source_records")
        if not isinstance(source_records, list) or manifest.get(
            "source_records_sha256"
        ) != _json_sha256(source_records, label="source record binding"):
            raise LangMemAdapterError("LangMem source-record binding mismatch")

        snapshot_spec = manifest.get("snapshot")
        if not isinstance(snapshot_spec, dict) or snapshot_spec.get("path") != "snapshot.json":
            raise LangMemAdapterError("LangMem manifest has an unsafe snapshot path")
        snapshot_payload = _read_regular_file(
            verified_path / "snapshot.json",
            label="LangMem snapshot",
        )
        if (
            snapshot_spec.get("bytes") != len(snapshot_payload)
            or snapshot_spec.get("sha256") != _sha256(snapshot_payload)
        ):
            raise LangMemAdapterError("LangMem snapshot SHA-256 mismatch")
        snapshot = _decode_json(snapshot_payload, label="LangMem snapshot")
        return _LoadedState(manifest=manifest, snapshot=snapshot)

    async def infer(
        self,
        record: CanonicalPreparedRecord,
        context: MethodRunContext,
        state: MethodStateHandle | None,
        backend: LangMemBackend,
    ) -> Mapping[str, Any]:
        canonical_record = (
            record
            if isinstance(record, CanonicalPreparedRecord)
            else validate_prepared_record(record)
        )
        if canonical_record.turn not in {"singleturn", "multiturn"}:
            raise LangMemAdapterError(
                f"unsupported canonical turn: {canonical_record.turn!r}"
            )
        loaded = self._load_state(canonical_record, context, state)
        query_embedding = _normalize_vector(
            await await_if_needed(
                backend.embedder.embed_query(canonical_record.query)
            ),
            label="query embedding",
        )
        raw_memories = await await_if_needed(
            backend.store.search(
                copy.deepcopy(loaded.snapshot),
                query=canonical_record.query,
                query_embedding=query_embedding,
                limit=self.top_k,
            )
        )
        memories = _normalize_memories(raw_memories)
        request = LangMemInferenceRequest(
            record=canonical_record.as_mapping(),
            turn=canonical_record.turn,
            query=canonical_record.query,
            retrieved_memories=memories,
            model_name=context.model_name,
            tools_schema=context.tools_schema,
            reasoning_effort=context.reasoning_effort,
        )
        raw_prediction = await await_if_needed(backend.model.generate(request))
        if isinstance(raw_prediction, str):
            prediction: dict[str, Any] = {"prediction": raw_prediction}
        elif isinstance(raw_prediction, Mapping):
            prediction = copy.deepcopy(dict(raw_prediction))
        else:
            raise LangMemAdapterError(
                "LangMem model must return a prediction string or mapping"
            )
        if (
            "retrieved_memories" in prediction
            and prediction["retrieved_memories"] != list(memories)
        ):
            raise LangMemAdapterError(
                "LangMem model changed the retrieved-memory evidence"
            )
        prediction["retrieved_memories"] = [
            copy.deepcopy(dict(memory)) for memory in memories
        ]
        prediction["memory_query"] = canonical_record.query
        return canonical_prediction(
            canonical_record,
            method_id=self.method_id,
            prediction=prediction,
        )


__all__ = [
    "LangMemAdapter",
    "LangMemAdapterError",
    "LangMemBackend",
    "LangMemBuildItem",
    "LangMemEmbedder",
    "LangMemInferenceRequest",
    "LangMemModel",
    "LangMemStore",
]
