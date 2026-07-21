"""Provider-neutral VanillaLLM execution for canonical prepared records."""

from __future__ import annotations

import asyncio
import copy
from contextlib import ExitStack
from dataclasses import dataclass, field
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Awaitable, Callable, Mapping, Sequence

from exp7.provenance.admission import (
    AdmittedFile,
    AdmittedDirectory,
    FileAdmissionError,
    StrictJSONError,
    admit_directory,
    admit_or_create_directory,
    admit_regular_file,
    decode_strict_json,
    lexical_absolute,
)


@dataclass(frozen=True)
class InferenceRequest:
    """Stable request passed to an injected inference callable."""

    model_name: str
    tools_schema: Sequence[Mapping[str, Any]]
    reasoning_effort: str | None = None


@dataclass(frozen=True)
class InferenceResult:
    """Provider-neutral response returned by an inference callable."""

    content: str
    reasoning_content: str = ""
    token_counts: Mapping[str, int] = field(default_factory=dict)
    response: Any = None


InferenceValue = InferenceResult | Mapping[str, Any] | str
InferenceCallable = Callable[
    [Mapping[str, Any], InferenceRequest],
    InferenceValue | Awaitable[InferenceValue],
]


class ManifestVerificationError(ValueError):
    """Raised before inference when a prepared artifact is not manifest-bound."""


@dataclass(frozen=True)
class AdmittedManifest:
    """One manifest object parsed and hashed from one admitted byte sequence."""

    file: AdmittedFile
    value: dict[str, Any]

    @property
    def path(self) -> Path:
        return self.file.path

    @property
    def payload(self) -> bytes:
        return self.file.payload

    @property
    def sha256(self) -> str:
        return self.file.sha256


@dataclass(frozen=True)
class RunSummary:
    input_count: int
    resumed_count: int
    inferred_count: int
    ok_count: int
    error_count: int
    output_path: Path
    checkpoint_path: Path
    status: str


def _admit_file(path: Path, *, label: str) -> AdmittedFile:
    try:
        return admit_regular_file(path, label=label)
    except FileAdmissionError as exc:
        raise ManifestVerificationError(str(exc)) from exc


def _decode_jsonl(file: AdmittedFile, label: str) -> list[dict[str, Any]]:
    try:
        text = file.payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ManifestVerificationError(f"cannot decode {label} {file.path}: {exc}") from exc
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ManifestVerificationError(
                f"cannot read {label} {file.path} line {line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise ManifestVerificationError(
                f"{label} line {line_number} is not a JSON object"
            )
        records.append(value)
    return records


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    return _decode_jsonl(_admit_file(path, label=label), label)


def admit_manifest(
    manifest_path: Path,
    *,
    expected_sha256: str | None = None,
) -> AdmittedManifest:
    file = _admit_file(manifest_path, label="manifest")
    if expected_sha256 is not None and file.sha256 != expected_sha256.lower():
        raise ManifestVerificationError(
            f"manifest SHA-256 mismatch: expected {expected_sha256.lower()}, "
            f"got {file.sha256}"
        )
    try:
        value = json.loads(file.payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestVerificationError(f"cannot read manifest {file.path}: {exc}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("outputs"), dict):
        raise ManifestVerificationError("manifest lacks an outputs mapping")
    return AdmittedManifest(file=file, value=value)


def verify_prepared_input(
    prepared_path: Path,
    manifest_path: Path,
    *,
    expected_manifest_sha256: str | None = None,
    admitted_manifest: AdmittedManifest | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Verify prepared JSONL membership, SHA-256, row count, and instance IDs."""

    expected_manifest_path = lexical_absolute(manifest_path)
    admitted = admitted_manifest or admit_manifest(
        expected_manifest_path,
        expected_sha256=expected_manifest_sha256,
    )
    if admitted.path != expected_manifest_path:
        raise ManifestVerificationError(
            "admitted manifest does not match the requested manifest path"
        )
    if (
        expected_manifest_sha256 is not None
        and admitted.sha256 != expected_manifest_sha256.lower()
    ):
        raise ManifestVerificationError(
            f"manifest SHA-256 mismatch: expected {expected_manifest_sha256.lower()}, "
            f"got {admitted.sha256}"
        )

    prepared = _admit_file(prepared_path, label="prepared JSONL")
    try:
        relative = prepared.path.relative_to(admitted.path.parent).as_posix()
    except ValueError as exc:
        raise ManifestVerificationError(
            "prepared input must be located beneath the manifest directory"
        ) from exc
    matches = [
        (scenario, spec)
        for scenario, spec in admitted.value["outputs"].items()
        if isinstance(spec, dict) and spec.get("path") == relative
    ]
    if len(matches) != 1:
        raise ManifestVerificationError(
            f"manifest must bind prepared path {relative!r} exactly once"
        )
    scenario, spec = matches[0]
    if prepared.sha256 != spec.get("sha256"):
        raise ManifestVerificationError(
            f"prepared SHA-256 mismatch for {scenario}: expected {spec.get('sha256')}, "
            f"got {prepared.sha256}"
        )
    records = _decode_jsonl(prepared, "prepared JSONL")
    if len(records) != spec.get("count"):
        raise ManifestVerificationError(
            f"prepared count mismatch for {scenario}: expected {spec.get('count')}, "
            f"got {len(records)}"
        )
    instance_ids = [str(record.get("instance_id", "")) for record in records]
    if any(not instance_id for instance_id in instance_ids):
        raise ManifestVerificationError("every prepared record must have an instance_id")
    if len(instance_ids) != len(set(instance_ids)):
        raise ManifestVerificationError("prepared instance_id values must be unique")
    return records, admitted.value


def _normalize_inference(value: InferenceValue) -> InferenceResult:
    if isinstance(value, InferenceResult):
        return value
    if isinstance(value, str):
        return InferenceResult(content=value, response=value)
    if isinstance(value, Mapping):
        content = value.get("content", value.get("output", value.get("llm_output", "")))
        reasoning = value.get("reasoning_content", value.get("reasoning", ""))
        token_counts = value.get("token_counts", {})
        return InferenceResult(
            content=str(content or ""),
            reasoning_content=str(reasoning or ""),
            token_counts=token_counts if isinstance(token_counts, Mapping) else {},
            response=value.get("response", content),
        )
    raise TypeError(f"unsupported inference result: {type(value).__name__}")


def _output_base(
    record: Mapping[str, Any],
    *,
    model_name: str,
    reasoning_effort: str | None,
) -> dict[str, Any]:
    source = record.get("source_example")
    source_copy = copy.deepcopy(source) if isinstance(source, Mapping) else {}
    output = dict(source_copy)
    output.update(
        {
            "dataset_id": record.get("dataset_id"),
            "difficulty": record.get("difficulty"),
            "example_id_sub": record.get("legacy_example_id_sub"),
            "ground_truth": copy.deepcopy(record.get("ground_truth", [])),
            "instance_id": record.get("instance_id"),
            "model_name": model_name,
            "query": record.get("query", ""),
            "reasoning_effort": reasoning_effort,
            "reference_ground_truth": copy.deepcopy(record.get("ground_truth", [])),
            "source_example": source_copy,
            "source_example_id": record.get("source_example_id"),
            "turn": record.get("turn"),
            "user_utterance": record.get("query", ""),
        }
    )
    return output


async def _infer(
    inference: InferenceCallable,
    record: Mapping[str, Any],
    request: InferenceRequest,
) -> InferenceResult:
    value = inference(record, request)
    if inspect.isawaitable(value):
        value = await value
    return _normalize_inference(value)


async def run_records(
    records: Sequence[Mapping[str, Any]],
    *,
    inference: InferenceCallable,
    model_name: str,
    tools_schema: Sequence[Mapping[str, Any]],
    reasoning_effort: str | None = None,
    completed: Mapping[str, Mapping[str, Any]] | None = None,
    on_result: Callable[[Mapping[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run in input order, accepting either a synchronous or async provider."""

    prior = dict(completed or {})
    request = InferenceRequest(model_name, tools_schema, reasoning_effort)
    results: list[dict[str, Any]] = []
    for record in records:
        instance_id = str(record["instance_id"])
        if instance_id in prior:
            results.append(copy.deepcopy(dict(prior[instance_id])))
            continue

        started = perf_counter()
        output = _output_base(
            record,
            model_name=model_name,
            reasoning_effort=reasoning_effort,
        )
        try:
            inference_result = await _infer(inference, record, request)
            content = inference_result.content
            reasoning = inference_result.reasoning_content
            token_counts = dict(inference_result.token_counts)
            status = "ok"
            error = None
            response = inference_result.response
            if response is None:
                response = content
        except Exception as exc:  # Preserve an evaluator-readable failed row.
            content = f"API_ERROR: {exc}"
            reasoning = ""
            token_counts = {}
            status = "error"
            error = str(exc)
            response = content

        latency = perf_counter() - started
        output.update(
            {
                "error": error,
                "latency": latency,
                "latency_seconds": latency,
                "llm_output": content,
                "prediction": content,
                "reasoning": reasoning,
                "reasoning_content": reasoning,
                "reasoning_token_count": int(
                    token_counts.get("reasoning_tokens", 0)
                    or token_counts.get("thinking_tokens", 0)
                    or token_counts.get("thoughts_tokens", 0)
                    or 0
                ),
                "response": response,
                "status": status,
                "token_counts": token_counts,
            }
        )
        results.append(output)
        if on_result is not None:
            on_result(output)
    return results


def _completed_records(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    for record in records:
        instance_id = str(record.get("instance_id", ""))
        if not instance_id or instance_id in completed:
            raise ManifestVerificationError(
                "checkpoint must contain unique non-empty instance_id values"
            )
        status = record.get("status")
        if status not in {"ok", "error"}:
            raise ManifestVerificationError(
                "checkpoint rows must have status 'ok' or 'error'"
            )
        error = record.get("error")
        if (status == "ok" and error is not None) or (
            status == "error" and not isinstance(error, str)
        ):
            raise ManifestVerificationError(
                "checkpoint status is inconsistent with its error field"
            )
        _validate_checkpoint_result(record)
        completed[instance_id] = dict(record)
    return completed


_CHECKPOINT_BINDING_FIELDS = (
    "dataset_id",
    "difficulty",
    "example_id_sub",
    "ground_truth",
    "instance_id",
    "model_name",
    "query",
    "reasoning_effort",
    "reference_ground_truth",
    "source_example",
    "source_example_id",
    "turn",
    "user_utterance",
)


_CHECKPOINT_RESULT_FIELDS = (
    "error",
    "latency",
    "latency_seconds",
    "llm_output",
    "prediction",
    "reasoning",
    "reasoning_content",
    "reasoning_token_count",
    "response",
    "status",
    "token_counts",
)

_CHECKPOINT_INTEGRITY_SCHEMA_VERSION = 1
_CHECKPOINT_INTEGRITY_SUFFIX = ".integrity.json"


def checkpoint_integrity_path(checkpoint_path: Path) -> Path:
    """Return the external integrity-seal path for a checkpoint path."""

    return checkpoint_path.with_name(
        checkpoint_path.name + _CHECKPOINT_INTEGRITY_SUFFIX
    )


def _checkpoint_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _checkpoint_integrity_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _committed_checkpoint_seal_bytes(payload: bytes) -> bytes:
    return _checkpoint_integrity_bytes(
        {
            "checkpoint_sha256": _checkpoint_sha256(payload),
            "checkpoint_size": len(payload),
            "schema_version": _CHECKPOINT_INTEGRITY_SCHEMA_VERSION,
            "state": "committed",
        }
    )


def _pending_checkpoint_seal_bytes(previous: bytes, appended: bytes) -> bytes:
    next_payload = previous + appended
    return _checkpoint_integrity_bytes(
        {
            "append_text": appended.decode("utf-8"),
            "next_sha256": _checkpoint_sha256(next_payload),
            "next_size": len(next_payload),
            "previous_sha256": _checkpoint_sha256(previous),
            "previous_size": len(previous),
            "schema_version": _CHECKPOINT_INTEGRITY_SCHEMA_VERSION,
            "state": "pending",
        }
    )


def _read_checkpoint_seal(file: AdmittedFile) -> dict[str, Any]:
    try:
        value = decode_strict_json(
            file.payload,
            label="checkpoint integrity seal",
        )
    except StrictJSONError as exc:
        raise ManifestVerificationError(str(exc)) from exc
    if not isinstance(value, dict):
        raise ManifestVerificationError(
            "checkpoint integrity seal must be a JSON object"
        )
    return value


def _valid_checkpoint_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _recover_checkpoint_integrity(
    directory: AdmittedDirectory,
    checkpoint_relative: Path,
    checkpoint_payload: bytes | None,
) -> bytes:
    seal_relative = checkpoint_integrity_path(checkpoint_relative)
    seal_file = directory.admit_file(
        seal_relative,
        label="checkpoint integrity seal",
    )
    seal = _read_checkpoint_seal(seal_file)
    state = seal.get("state")
    if seal.get("schema_version") != _CHECKPOINT_INTEGRITY_SCHEMA_VERSION:
        raise ManifestVerificationError(
            "checkpoint integrity seal has an unsupported schema version"
        )

    if state == "committed":
        expected_fields = {
            "checkpoint_sha256",
            "checkpoint_size",
            "schema_version",
            "state",
        }
        size = seal.get("checkpoint_size")
        digest = seal.get("checkpoint_sha256")
        if (
            set(seal) != expected_fields
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not _valid_checkpoint_digest(digest)
        ):
            raise ManifestVerificationError(
                "checkpoint integrity seal has invalid committed metadata"
            )
        if checkpoint_payload is None:
            if size != 0 or digest != _checkpoint_sha256(b""):
                raise ManifestVerificationError(
                    "checkpoint integrity mismatch: checkpoint file is missing"
                )
            directory.write_exclusive(
                checkpoint_relative,
                b"",
                label="checkpoint JSONL",
            )
            checkpoint_payload = b""
        if (
            len(checkpoint_payload) != size
            or _checkpoint_sha256(checkpoint_payload) != digest
        ):
            raise ManifestVerificationError(
                "checkpoint integrity mismatch with committed seal"
            )
        return checkpoint_payload

    if state != "pending":
        raise ManifestVerificationError(
            "checkpoint integrity seal state must be 'committed' or 'pending'"
        )
    expected_fields = {
        "append_text",
        "next_sha256",
        "next_size",
        "previous_sha256",
        "previous_size",
        "schema_version",
        "state",
    }
    append_text = seal.get("append_text")
    previous_size = seal.get("previous_size")
    next_size = seal.get("next_size")
    previous_digest = seal.get("previous_sha256")
    next_digest = seal.get("next_sha256")
    if (
        set(seal) != expected_fields
        or not isinstance(append_text, str)
        or not append_text.endswith("\n")
        or isinstance(previous_size, bool)
        or not isinstance(previous_size, int)
        or previous_size < 0
        or isinstance(next_size, bool)
        or not isinstance(next_size, int)
        or not _valid_checkpoint_digest(previous_digest)
        or not _valid_checkpoint_digest(next_digest)
    ):
        raise ManifestVerificationError(
            "checkpoint integrity seal has invalid pending metadata"
        )
    appended = append_text.encode("utf-8")
    if next_size != previous_size + len(appended):
        raise ManifestVerificationError(
            "checkpoint integrity pending size is inconsistent"
        )

    current = checkpoint_payload or b""
    if len(current) < previous_size or len(current) > next_size:
        raise ManifestVerificationError(
            "checkpoint integrity mismatch during pending recovery"
        )
    previous = current[:previous_size]
    suffix = current[previous_size:]
    if (
        _checkpoint_sha256(previous) != previous_digest
        or suffix != appended[: len(suffix)]
        or _checkpoint_sha256(previous + appended) != next_digest
    ):
        raise ManifestVerificationError(
            "checkpoint integrity mismatch during pending recovery"
        )

    remaining = appended[len(suffix):]
    if remaining:
        with directory.open_text(
            checkpoint_relative,
            label="checkpoint JSONL",
            append=True,
            exclusive=checkpoint_payload is None,
        ) as checkpoint_handle:
            checkpoint_handle.buffer.write(remaining)
            checkpoint_handle.flush()
            os.fsync(checkpoint_handle.fileno())
    recovered = previous + appended
    directory.atomic_write(
        seal_relative,
        _committed_checkpoint_seal_bytes(recovered),
        label="checkpoint integrity seal",
    )
    return recovered


@dataclass
class _CheckpointJournal:
    directory: AdmittedDirectory
    checkpoint_relative: Path
    payload: bytes

    @property
    def seal_relative(self) -> Path:
        return checkpoint_integrity_path(self.checkpoint_relative)

    def initialize(self) -> None:
        self.directory.atomic_write(
            self.seal_relative,
            _committed_checkpoint_seal_bytes(self.payload),
            label="checkpoint integrity seal",
        )

    def append(self, checkpoint_handle: Any, result: Mapping[str, Any]) -> None:
        _validate_checkpoint_result(result)
        appended = (
            json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n"
        ).encode("utf-8")
        self.directory.atomic_write(
            self.seal_relative,
            _pending_checkpoint_seal_bytes(self.payload, appended),
            label="checkpoint integrity seal",
        )
        checkpoint_handle.buffer.write(appended)
        checkpoint_handle.flush()
        os.fsync(checkpoint_handle.fileno())
        self.payload += appended
        self.directory.atomic_write(
            self.seal_relative,
            _committed_checkpoint_seal_bytes(self.payload),
            label="checkpoint integrity seal",
        )


def _validate_checkpoint_result(checkpoint: Mapping[str, Any]) -> None:
    missing = [
        field_name
        for field_name in _CHECKPOINT_RESULT_FIELDS
        if field_name not in checkpoint
    ]
    if missing:
        raise ManifestVerificationError(
            "checkpoint result is missing required field(s): " + ", ".join(missing)
        )

    llm_output = checkpoint["llm_output"]
    prediction = checkpoint["prediction"]
    if (
        not isinstance(llm_output, str)
        or not isinstance(prediction, str)
        or prediction != llm_output
    ):
        raise ManifestVerificationError(
            "checkpoint prediction is inconsistent with llm_output"
        )

    reasoning = checkpoint["reasoning"]
    reasoning_content = checkpoint["reasoning_content"]
    if (
        not isinstance(reasoning, str)
        or not isinstance(reasoning_content, str)
        or reasoning_content != reasoning
    ):
        raise ManifestVerificationError(
            "checkpoint reasoning_content is inconsistent with reasoning"
        )

    latency = checkpoint["latency"]
    latency_seconds = checkpoint["latency_seconds"]
    if (
        isinstance(latency, bool)
        or not isinstance(latency, (int, float))
        or not math.isfinite(latency)
        or latency < 0
        or isinstance(latency_seconds, bool)
        or not isinstance(latency_seconds, (int, float))
        or not math.isfinite(latency_seconds)
        or latency_seconds < 0
        or latency_seconds != latency
    ):
        raise ManifestVerificationError(
            "checkpoint latency_seconds is inconsistent with latency"
        )

    token_counts = checkpoint["token_counts"]
    if not isinstance(token_counts, Mapping):
        raise ManifestVerificationError("checkpoint token_counts must be an object")
    try:
        expected_reasoning_tokens = int(
            token_counts.get("reasoning_tokens", 0)
            or token_counts.get("thinking_tokens", 0)
            or token_counts.get("thoughts_tokens", 0)
            or 0
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ManifestVerificationError(
            "checkpoint token_counts contains an invalid reasoning token count"
        ) from exc
    reasoning_token_count = checkpoint["reasoning_token_count"]
    if (
        isinstance(reasoning_token_count, bool)
        or not isinstance(reasoning_token_count, int)
        or reasoning_token_count < 0
        or expected_reasoning_tokens < 0
        or reasoning_token_count != expected_reasoning_tokens
    ):
        raise ManifestVerificationError(
            "checkpoint reasoning_token_count is inconsistent with token_counts"
        )

    status = checkpoint["status"]
    error = checkpoint["error"]
    if status == "error":
        expected_error_output = f"API_ERROR: {error}"
        if (
            llm_output != expected_error_output
            or checkpoint["response"] != expected_error_output
            or reasoning
            or token_counts
            or reasoning_token_count != 0
        ):
            raise ManifestVerificationError(
                "checkpoint provider-error result is inconsistent with status/error"
            )
    elif llm_output.startswith("API_ERROR: "):
        raise ManifestVerificationError(
            "checkpoint successful result uses the reserved provider-error marker"
        )


def _validate_checkpoint_binding(
    checkpoint: Mapping[str, Any],
    prepared_record: Mapping[str, Any],
    *,
    model_name: str,
    reasoning_effort: str | None,
) -> None:
    expected = _output_base(
        prepared_record,
        model_name=model_name,
        reasoning_effort=reasoning_effort,
    )
    instance_id = str(prepared_record["instance_id"])
    for field_name in _CHECKPOINT_BINDING_FIELDS:
        if (
            field_name not in checkpoint
            or checkpoint[field_name] != expected[field_name]
        ):
            raise ManifestVerificationError(
                f"checkpoint field {field_name!r} does not match prepared input "
                f"for {instance_id}"
            )


def _json_array_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    payload = (json.dumps(records, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    return payload


def _output_directory(path: Path, *, label: str) -> AdmittedDirectory:
    try:
        return admit_or_create_directory(path, label=label)
    except FileAdmissionError as exc:
        raise ManifestVerificationError(str(exc)) from exc


def _bound_output_root(
    path: Path,
    *,
    expected_identity: tuple[int, int],
) -> AdmittedDirectory:
    try:
        directory = admit_directory(path, label="parent runner output root")
    except FileAdmissionError as exc:
        raise ManifestVerificationError(str(exc)) from exc
    if directory.identity[:2] != expected_identity:
        directory.close()
        raise ManifestVerificationError(
            "output root identity differs from parent runner"
        )
    return directory


def _completion_status(*, ok_count: int, error_count: int) -> str:
    if error_count == 0:
        return "complete"
    if ok_count == 0:
        return "all_provider_errors"
    return "complete_with_provider_errors"


def run_prepared_file(
    *,
    prepared_path: Path,
    manifest_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    tools_schema: Sequence[Mapping[str, Any]],
    model_name: str,
    inference: InferenceCallable,
    reasoning_effort: str | None = None,
    expected_manifest_sha256: str | None = None,
    resume: bool = False,
    output_root: Path | None = None,
    expected_output_root_identity: tuple[int, int] | None = None,
) -> RunSummary:
    """Verify, execute, checkpoint each row, then publish one JSON array."""

    records, _ = verify_prepared_input(
        prepared_path,
        manifest_path,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    expected_ids = [str(record["instance_id"]) for record in records]
    output_parent = lexical_absolute(output_path.parent)
    checkpoint_parent = lexical_absolute(checkpoint_path.parent)
    with ExitStack() as stack:
        if (output_root is None) != (expected_output_root_identity is None):
            raise ManifestVerificationError(
                "output root path and identity must be supplied together"
            )
        if output_root is not None:
            assert expected_output_root_identity is not None
            bound_output_root = stack.enter_context(
                _bound_output_root(
                    output_root,
                    expected_identity=expected_output_root_identity,
                )
            )
            try:
                bound_output_root.relative(output_path)
                bound_output_root.relative(checkpoint_path)
            except FileAdmissionError as exc:
                raise ManifestVerificationError(str(exc)) from exc
            if output_parent == bound_output_root.path:
                output_directory = bound_output_root
            else:
                output_directory = stack.enter_context(
                    _output_directory(
                        output_parent,
                        label="prediction output directory",
                    )
                )
            if checkpoint_parent == output_parent:
                checkpoint_directory = output_directory
            elif checkpoint_parent == bound_output_root.path:
                checkpoint_directory = bound_output_root
            else:
                checkpoint_directory = stack.enter_context(
                    _output_directory(
                        checkpoint_parent,
                        label="checkpoint output directory",
                    )
                )
            try:
                bound_output_root.verify()
            except FileAdmissionError as exc:
                raise ManifestVerificationError(str(exc)) from exc
        else:
            output_directory = stack.enter_context(
                _output_directory(
                    output_parent,
                    label="prediction output directory",
                )
            )
            if checkpoint_parent == output_parent:
                checkpoint_directory = output_directory
            else:
                checkpoint_directory = stack.enter_context(
                    _output_directory(
                        checkpoint_parent,
                        label="checkpoint output directory",
                    )
                )
        output_relative = output_directory.relative(output_path)
        checkpoint_relative = checkpoint_directory.relative(checkpoint_path)
        seal_relative = checkpoint_integrity_path(checkpoint_relative)
        try:
            output_exists = output_directory.exists(
                output_relative,
                label="prediction output",
            )
            checkpoint_exists = checkpoint_directory.exists(
                checkpoint_relative,
                label="checkpoint JSONL",
            )
            seal_exists = checkpoint_directory.exists(
                seal_relative,
                label="checkpoint integrity seal",
            )
        except FileAdmissionError as exc:
            raise ManifestVerificationError(str(exc)) from exc
        if output_exists:
            raise FileExistsError(f"refusing to overwrite output: {output_path}")
        if checkpoint_exists:
            if not resume:
                raise FileExistsError(
                    f"checkpoint exists; pass --resume to continue: {checkpoint_path}"
                )
            try:
                admitted_checkpoint = checkpoint_directory.admit_file(
                    checkpoint_relative,
                    label="checkpoint JSONL",
                )
            except FileAdmissionError as exc:
                raise ManifestVerificationError(str(exc)) from exc
            checkpoint_payload: bytes | None = admitted_checkpoint.payload
        else:
            checkpoint_payload = None
            if resume and not seal_exists:
                raise FileNotFoundError(
                    f"resume checkpoint does not exist: {checkpoint_path}"
                )

        if resume:
            if not seal_exists:
                raise ManifestVerificationError(
                    "checkpoint integrity seal is missing"
                )
            try:
                checkpoint_payload = _recover_checkpoint_integrity(
                    checkpoint_directory,
                    checkpoint_relative,
                    checkpoint_payload,
                )
            except FileAdmissionError as exc:
                raise ManifestVerificationError(str(exc)) from exc
            checkpoint_exists = True
            completed = _completed_records(
                _decode_jsonl(
                    AdmittedFile(
                        path=checkpoint_directory.path / checkpoint_relative,
                        payload=checkpoint_payload,
                    ),
                    "checkpoint JSONL",
                )
            )
            if list(completed) != expected_ids[: len(completed)]:
                raise ManifestVerificationError(
                    "checkpoint must be an ordered prefix of the prepared input"
                )
            for prepared_record, checkpoint_record in zip(records, completed.values()):
                _validate_checkpoint_binding(
                    checkpoint_record,
                    prepared_record,
                    model_name=model_name,
                    reasoning_effort=reasoning_effort,
                )
        else:
            completed = {}
            checkpoint_payload = b""

        try:
            journal = _CheckpointJournal(
                checkpoint_directory,
                checkpoint_relative,
                checkpoint_payload,
            )
            if not checkpoint_exists:
                journal.initialize()
            checkpoint_context = checkpoint_directory.open_text(
                checkpoint_relative,
                label="checkpoint JSONL",
                append=True,
                exclusive=not checkpoint_exists,
            )
            with checkpoint_context as checkpoint_handle:

                def append_checkpoint(result: Mapping[str, Any]) -> None:
                    journal.append(checkpoint_handle, result)

                results = asyncio.run(
                    run_records(
                        records,
                        inference=inference,
                        model_name=model_name,
                        tools_schema=tools_schema,
                        reasoning_effort=reasoning_effort,
                        completed=completed,
                        on_result=append_checkpoint,
                    )
                )
            output_directory.write_exclusive(
                output_relative,
                _json_array_bytes(results),
                label="prediction output",
            )
        except FileAdmissionError as exc:
            raise ManifestVerificationError(str(exc)) from exc
    ok_count = sum(record.get("status") == "ok" for record in results)
    error_count = sum(record.get("status") == "error" for record in results)
    return RunSummary(
        input_count=len(records),
        resumed_count=len(completed),
        inferred_count=len(records) - len(completed),
        ok_count=ok_count,
        error_count=error_count,
        output_path=output_path,
        checkpoint_path=checkpoint_path,
        status=_completion_status(ok_count=ok_count, error_count=error_count),
    )
