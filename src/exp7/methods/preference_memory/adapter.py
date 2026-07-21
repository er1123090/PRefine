"""Canonical preference-memory build and inference lifecycle."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from typing import Any, Awaitable, Callable, Mapping, Sequence

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


METHOD_ID = "preference_memory"
STATE_FORMAT_VERSION = 1
MEMORY_FILE = "memory.jsonl"
STATE_MANIFEST = "state_manifest.json"


@dataclass(frozen=True)
class PreferenceRefinementRequest:
    """One sequential source-session update sent to an injected refiner."""

    source_example_id: str
    session_index: int
    session_dialogue: str
    accumulated_dialogue: str
    session_api_calls: tuple[str, ...]
    accumulated_api_calls: tuple[str, ...]
    previous_preference: Any


@dataclass(frozen=True)
class PreferenceRefinementResult:
    """Provider-neutral preference update and its auditable evidence."""

    preference: Any
    evidence: Any = field(default_factory=list)
    refinement_process: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreferenceGenerationRequest:
    """Verified memory context passed to the injected action generator."""

    model_name: str
    tools_schema: tuple[Mapping[str, Any], ...]
    reasoning_effort: str | None
    memory: Mapping[str, Any]
    memory_evidence: tuple[Mapping[str, Any], ...]


PreferenceRefinerValue = PreferenceRefinementResult | Mapping[str, Any] | str
PreferenceRefiner = Callable[
    [PreferenceRefinementRequest],
    PreferenceRefinerValue | Awaitable[PreferenceRefinerValue],
]
PreferenceGeneratorValue = Mapping[str, Any] | str
PreferenceGenerator = Callable[
    [Mapping[str, Any], PreferenceGenerationRequest],
    PreferenceGeneratorValue | Awaitable[PreferenceGeneratorValue],
]


@dataclass(frozen=True)
class PreferenceMemoryRuntime:
    """All preference extraction and generation behavior is injected."""

    refiner: PreferenceRefiner
    generator: PreferenceGenerator

    def __post_init__(self) -> None:
        if not callable(self.refiner):
            raise MethodContractError("preference-memory refiner must be callable")
        if not callable(self.generator):
            raise MethodContractError("preference-memory generator must be callable")


def _runtime(value: Any) -> PreferenceMemoryRuntime:
    if isinstance(value, PreferenceMemoryRuntime):
        return value
    if isinstance(value, Mapping):
        try:
            return PreferenceMemoryRuntime(**dict(value))
        except TypeError as exc:
            raise MethodContractError(
                "preference-memory runtime mapping is malformed"
            ) from exc
    raise MethodContractError(
        "preference-memory backend must be a runtime or runtime mapping"
    )


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
            "preference-memory state must be JSON serializable"
        ) from exc


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _source_payload(record: CanonicalPreparedRecord) -> dict[str, Any]:
    source = record.as_mapping().get("source_example")
    if not isinstance(source, Mapping):
        raise MethodContractError(
            f"preference-memory record {record.instance_id} lacks source_example"
        )
    value = copy.deepcopy(dict(source))
    embedded_id = value.get("example_id")
    if embedded_id is not None and str(embedded_id) != record.source_example_id:
        raise MethodContractError(
            f"preference-memory source identity drift for {record.instance_id}"
        )
    return value


def _source_sessions(
    source_example_id: str, source: Mapping[str, Any]
) -> list[tuple[str, tuple[str, ...]]]:
    sessions = source.get("sessions", [])
    if not isinstance(sessions, list):
        raise MethodContractError(
            f"preference-memory source {source_example_id} sessions must be a list"
        )
    normalized = []
    for session in sessions:
        if not isinstance(session, Mapping):
            raise MethodContractError(
                f"preference-memory source {source_example_id} has an invalid session"
            )
        dialogue = session.get("dialogue", [])
        if not isinstance(dialogue, list):
            raise MethodContractError(
                f"preference-memory source {source_example_id} dialogue must be a list"
            )
        lines = []
        for turn in dialogue:
            if not isinstance(turn, Mapping):
                raise MethodContractError(
                    f"preference-memory source {source_example_id} has an invalid turn"
                )
            role = str(turn.get("role", "")).strip() or "Unknown"
            message = turn.get("message", turn.get("content", ""))
            if isinstance(message, str) and message.strip():
                lines.append(f"{role}: {message}")
        api_calls = session.get("api_call", [])
        if isinstance(api_calls, str):
            api_calls = [api_calls] if api_calls else []
        if not isinstance(api_calls, list) or not all(
            isinstance(call, str) for call in api_calls
        ):
            raise MethodContractError(
                f"preference-memory source {source_example_id} API calls must be strings"
            )
        normalized.append(("\n".join(lines), tuple(api_calls)))
    return normalized


def _normalize_refinement(value: Any) -> PreferenceRefinementResult:
    if isinstance(value, PreferenceRefinementResult):
        result = value
    elif isinstance(value, str):
        result = PreferenceRefinementResult(preference=value)
    elif isinstance(value, Mapping):
        if "preference" in value:
            preference = value["preference"]
        elif "final_implicit_preference" in value:
            preference = value["final_implicit_preference"]
        else:
            raise MethodContractError(
                "preference-memory refiner result lacks preference"
            )
        process = value.get("refinement_process", ())
        if not isinstance(process, (list, tuple)) or not all(
            isinstance(step, Mapping) for step in process
        ):
            raise MethodContractError(
                "preference-memory refinement_process must contain mappings"
            )
        metadata = value.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise MethodContractError(
                "preference-memory refinement metadata must be a mapping"
            )
        result = PreferenceRefinementResult(
            preference=copy.deepcopy(preference),
            evidence=copy.deepcopy(value.get("evidence", [])),
            refinement_process=tuple(copy.deepcopy(dict(step)) for step in process),
            metadata=copy.deepcopy(dict(metadata)),
        )
    else:
        raise MethodContractError(
            "preference-memory refiner must return a result, mapping, or string"
        )
    _canonical_bytes(
        {
            "evidence": result.evidence,
            "metadata": dict(result.metadata),
            "preference": result.preference,
            "refinement_process": [dict(step) for step in result.refinement_process],
        }
    )
    return result


async def _build_memory_record(
    source_example_id: str,
    source: Mapping[str, Any],
    refiner: PreferenceRefiner,
) -> dict[str, Any]:
    previous: Any = {}
    accumulated_dialogue = ""
    accumulated_api_calls: list[str] = []
    evolution = []
    sessions = _source_sessions(source_example_id, source)
    for session_index, (dialogue, api_calls) in enumerate(sessions, start=1):
        header = f"=== Session {session_index} ==="
        accumulated_dialogue = "\n".join(
            part for part in (accumulated_dialogue, header, dialogue) if part
        )
        labeled_calls = [
            f"[Session {session_index}] {api_call}" for api_call in api_calls
        ]
        accumulated_api_calls.extend(labeled_calls)
        request = PreferenceRefinementRequest(
            source_example_id=source_example_id,
            session_index=session_index,
            session_dialogue=dialogue,
            accumulated_dialogue=accumulated_dialogue,
            session_api_calls=api_calls,
            accumulated_api_calls=tuple(accumulated_api_calls),
            previous_preference=copy.deepcopy(previous),
        )
        result = _normalize_refinement(await await_if_needed(refiner(request)))
        evolution.append(
            {
                "accumulated_api_calls": list(accumulated_api_calls),
                "evidence": copy.deepcopy(result.evidence),
                "final_preference_at_session": copy.deepcopy(result.preference),
                "metadata": copy.deepcopy(dict(result.metadata)),
                "previous_preference": copy.deepcopy(previous),
                "refinement_process": [
                    copy.deepcopy(dict(step)) for step in result.refinement_process
                ],
                "session_api_calls": list(api_calls),
                "session_dialogue": dialogue,
                "session_index": session_index,
            }
        )
        previous = copy.deepcopy(result.preference)
    record = {
        "example_id": source_example_id,
        "final_accumulated_api_calls": accumulated_api_calls,
        "final_accumulated_dialogue": accumulated_dialogue,
        "final_implicit_preference": previous,
        "preference_evolution_history": evolution,
        "source_example_id": source_example_id,
        "total_sessions_processed": len(sessions),
    }
    _canonical_bytes(record)
    return record


def _deduplicate_sources(
    records: Sequence[CanonicalPreparedRecord],
) -> dict[str, tuple[dict[str, Any], str]]:
    sources: dict[str, tuple[dict[str, Any], str]] = {}
    for record in records:
        source = _source_payload(record)
        digest = _sha256(_canonical_bytes(source))
        existing = sources.get(record.source_example_id)
        if existing is not None:
            if existing[1] != digest:
                raise MethodContractError(
                    "preference-memory source_example_id has conflicting payloads: "
                    f"{record.source_example_id}"
                )
            continue
        sources[record.source_example_id] = (source, digest)
    return sources


def _write_exclusive(path: Path, payload: bytes) -> None:
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


def _jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_bytes(record) + b"\n" for record in records)


def _admit_state_directory(path: Path) -> None:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise MethodContractError(
            f"preference-memory state directory is missing: {path}"
        ) from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise MethodContractError(
            f"preference-memory state must be a real directory: {path}"
        )
    children = list(path.iterdir())
    if {child.name for child in children} != {MEMORY_FILE, STATE_MANIFEST}:
        raise MethodContractError(
            "preference-memory state artifacts are missing or tampered"
        )
    for child in children:
        child_info = os.lstat(child)
        if stat.S_ISLNK(child_info.st_mode) or not stat.S_ISREG(child_info.st_mode):
            raise MethodContractError(
                f"preference-memory state contains an unsafe artifact: {child}"
            )


def _decode_json(payload: bytes, label: str) -> Any:
    try:
        return decode_strict_json(
            payload, label=f"preference-memory {label}"
        )
    except StrictJSONError as exc:
        raise MethodContractError(str(exc)) from exc


def _decode_memory_jsonl(payload: bytes) -> dict[str, dict[str, Any]]:
    memories: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line.strip():
            continue
        value = _decode_json(line, f"JSONL line {line_number}")
        if not isinstance(value, dict):
            raise MethodContractError(
                f"preference-memory JSONL line {line_number} must be an object"
            )
        source_id = value.get("source_example_id")
        if not isinstance(source_id, str) or not source_id:
            raise MethodContractError(
                f"preference-memory JSONL line {line_number} lacks source_example_id"
            )
        if source_id in memories:
            raise MethodContractError(
                f"duplicate preference-memory source_example_id: {source_id}"
            )
        if value.get("example_id") != source_id:
            raise MethodContractError(
                f"cross-source preference-memory record: {source_id}"
            )
        memories[source_id] = value
    return memories


def _load_state(
    context: MethodRunContext,
    state: MethodStateHandle,
    state_relative: PurePosixPath,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    expected_path = method_state_path(context, state_relative)
    if state.method_id != METHOD_ID or state.path != expected_path:
        raise MethodContractError(
            "preference-memory state handle does not identify canonical state"
        )
    if state.dataset_manifest_sha256 != context.dataset_manifest_sha256:
        raise MethodContractError(
            "preference-memory dataset manifest binding changed"
        )
    _admit_state_directory(state.path)
    manifest_path, memory_path = state.path / STATE_MANIFEST, state.path / MEMORY_FILE
    try:
        manifest_payload = admit_regular_file(
            manifest_path, label="preference-memory state manifest"
        ).payload
        memory_payload = admit_regular_file(
            memory_path, label="preference-memory memory JSONL"
        ).payload
    except FileAdmissionError as exc:
        raise MethodContractError(str(exc)) from exc
    _admit_state_directory(state.path)
    expected_manifest_sha = state.metadata.get("state_manifest_sha256")
    if not isinstance(expected_manifest_sha, str) or _sha256(manifest_payload) != expected_manifest_sha:
        raise MethodContractError("preference-memory state manifest SHA-256 mismatch")
    manifest = _decode_json(manifest_payload, "state manifest")
    if not isinstance(manifest, dict):
        raise MethodContractError("preference-memory state manifest must be an object")
    if (
        manifest.get("format_version") != STATE_FORMAT_VERSION
        or manifest.get("method_id") != METHOD_ID
        or manifest.get("dataset_manifest_sha256") != context.dataset_manifest_sha256
        or manifest.get("records_sha256") != state.records_sha256
    ):
        raise MethodContractError("preference-memory state manifest binding changed")
    memory_spec = manifest.get("memory")
    if not isinstance(memory_spec, dict) or (
        memory_spec.get("path") != MEMORY_FILE
        or memory_spec.get("bytes") != len(memory_payload)
        or memory_spec.get("sha256") != _sha256(memory_payload)
    ):
        raise MethodContractError("preference-memory JSONL is missing or tampered")
    memories = _decode_memory_jsonl(memory_payload)
    if memory_spec.get("count") != len(memories):
        raise MethodContractError("preference-memory JSONL count changed")
    sources = manifest.get("sources")
    if not isinstance(sources, dict) or set(sources) != set(memories):
        raise MethodContractError("preference-memory source membership changed")
    for source_id, memory in memories.items():
        source_spec = sources[source_id]
        if not isinstance(source_spec, dict) or source_spec.get("memory_sha256") != _sha256(
            _canonical_bytes(memory)
        ):
            raise MethodContractError(
                f"preference-memory row is tampered for source {source_id}"
            )
    return manifest, memories


class PreferenceMemoryAdapter:
    """Build one evolving memory per source and require it for inference."""

    method_id = METHOD_ID

    def __init__(self, state_relative: str = "method_state/preference_memory") -> None:
        self._state_relative = PurePosixPath(state_relative)

    async def build(
        self,
        records: Sequence[CanonicalPreparedRecord],
        context: MethodRunContext,
        backend: PreferenceMemoryRuntime,
    ) -> MethodStateHandle:
        runtime = _runtime(backend)
        validated = validate_prepared_records(records)
        if not validated:
            raise MethodContractError(
                "preference-memory build requires prepared records"
            )
        state_dir = method_state_path(context, self._state_relative)
        if state_dir.exists() or state_dir.is_symlink():
            raise MethodContractError(
                f"refusing to overwrite preference-memory state: {state_dir}"
            )
        state_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=".preference-memory-", dir=state_dir.parent)
        )
        try:
            sources = _deduplicate_sources(validated)
            memory_records = []
            source_manifest = {}
            for source_id in sorted(sources):
                source, source_sha = sources[source_id]
                memory = await _build_memory_record(
                    source_id, source, runtime.refiner
                )
                memory_records.append(memory)
                source_manifest[source_id] = {
                    "memory_sha256": _sha256(_canonical_bytes(memory)),
                    "source_sha256": source_sha,
                }
            memory_payload = _jsonl_bytes(memory_records)
            _write_exclusive(temporary / MEMORY_FILE, memory_payload)
            temporary_relative = temporary.relative_to(context.run_dir).as_posix()
            preliminary = bind_method_state(
                method_id=METHOD_ID,
                context=context,
                relative_path=temporary_relative,
                records=validated,
            )
            manifest = {
                "dataset_manifest_sha256": context.dataset_manifest_sha256,
                "format_version": STATE_FORMAT_VERSION,
                "instance_ids": [record.instance_id for record in validated],
                "memory": {
                    "bytes": len(memory_payload),
                    "count": len(memory_records),
                    "path": MEMORY_FILE,
                    "sha256": _sha256(memory_payload),
                },
                "method_id": METHOD_ID,
                "records_sha256": preliminary.records_sha256,
                "sources": source_manifest,
            }
            manifest_payload = json.dumps(
                manifest, ensure_ascii=False, indent=2, sort_keys=True
            ).encode("utf-8") + b"\n"
            _write_exclusive(temporary / STATE_MANIFEST, manifest_payload)
            directory = os.open(temporary, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            if state_dir.exists() or state_dir.is_symlink():
                raise MethodContractError(
                    f"refusing to overwrite preference-memory state: {state_dir}"
                )
            temporary.rename(state_dir)
            temporary = None
            return bind_method_state(
                method_id=METHOD_ID,
                context=context,
                relative_path=self._state_relative,
                records=validated,
                metadata={
                    "memory_count": len(memory_records),
                    "state_manifest_sha256": _sha256(manifest_payload),
                },
            )
        finally:
            if temporary is not None:
                shutil.rmtree(temporary)

    async def infer(
        self,
        record: CanonicalPreparedRecord,
        context: MethodRunContext,
        state: MethodStateHandle | None,
        backend: PreferenceMemoryRuntime,
    ) -> dict[str, Any]:
        runtime = _runtime(backend)
        canonical = validate_prepared_record(record.as_mapping())
        if state is None:
            raise MethodContractError(
                "preference-memory inference requires built state"
            )
        manifest, memories = _load_state(context, state, self._state_relative)
        if canonical.instance_id not in manifest.get("instance_ids", []):
            raise MethodContractError(
                "preference-memory record was not included in built state"
            )
        memory = memories.get(canonical.source_example_id)
        if memory is None:
            raise MethodContractError(
                f"missing preference memory for source {canonical.source_example_id}"
            )
        source = _source_payload(canonical)
        source_spec = manifest["sources"].get(canonical.source_example_id)
        if not isinstance(source_spec, dict) or source_spec.get("source_sha256") != _sha256(
            _canonical_bytes(source)
        ):
            raise MethodContractError(
                "preference-memory inference source differs from built state"
            )
        evidence = memory.get("preference_evolution_history")
        if not isinstance(evidence, list) or not all(
            isinstance(item, Mapping) for item in evidence
        ):
            raise MethodContractError(
                "preference-memory evidence is missing or malformed"
            )
        request = PreferenceGenerationRequest(
            model_name=context.model_name,
            tools_schema=context.tools_schema,
            reasoning_effort=context.reasoning_effort,
            memory=copy.deepcopy(memory),
            memory_evidence=tuple(copy.deepcopy(dict(item)) for item in evidence),
        )
        generated = await await_if_needed(
            runtime.generator(canonical.as_mapping(), request)
        )
        if isinstance(generated, str):
            prediction: dict[str, Any] = {"prediction": generated}
        elif isinstance(generated, Mapping):
            prediction = copy.deepcopy(dict(generated))
        else:
            raise MethodContractError(
                "preference-memory generator must return a mapping or string"
            )
        expected_memory = copy.deepcopy(memory)
        expected_evidence = copy.deepcopy(evidence)
        if "preference_memory" in prediction and prediction["preference_memory"] != expected_memory:
            raise MethodContractError(
                "preference-memory generator changed canonical memory"
            )
        if "memory_evidence" in prediction and prediction["memory_evidence"] != expected_evidence:
            raise MethodContractError(
                "preference-memory generator changed memory evidence"
            )
        prediction.update(
            {
                "memory_evidence": expected_evidence,
                "memory_source_example_id": canonical.source_example_id,
                "memory_status": "loaded",
                "preference_memory": expected_memory,
            }
        )
        return canonical_prediction(
            canonical,
            method_id=METHOD_ID,
            prediction=prediction,
        )
