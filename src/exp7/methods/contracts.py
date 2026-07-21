"""Shared, provider-neutral contracts for canonical experiment methods."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import hashlib
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Awaitable, Mapping, Protocol, Sequence, TypeVar


_METHOD_ID = re.compile(r"[a-z][a-z0-9_]*\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class MethodContractError(ValueError):
    """Raised when prepared data or method state violates the shared contract."""


def validate_method_id(method_id: str) -> str:
    if not isinstance(method_id, str) or _METHOD_ID.fullmatch(method_id) is None:
        raise MethodContractError(
            "method_id must match [a-z][a-z0-9_]*: " f"{method_id!r}"
        )
    return method_id


def _required_string(
    record: Mapping[str, Any],
    key: str,
    *,
    allow_empty: bool = False,
) -> str:
    value = record.get(key)
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise MethodContractError(
            f"prepared record {key!r} must be a non-empty string"
        )
    return value


@dataclass(frozen=True)
class CanonicalPreparedRecord:
    """Validated immutable view over one canonical prepared-record payload."""

    dataset_id: str
    turn: str
    difficulty: str
    instance_id: str
    source_example_id: str
    query: str
    ground_truth: tuple[str, ...]
    _payload: Mapping[str, Any] = field(repr=False, compare=False)

    def as_mapping(self) -> dict[str, Any]:
        """Return an isolated payload copy suitable for a method backend."""

        return copy.deepcopy(dict(self._payload))


def validate_prepared_record(
    record: Mapping[str, Any],
) -> CanonicalPreparedRecord:
    """Validate identity/query/GT fields without rewriting their semantics."""

    if not isinstance(record, Mapping):
        raise MethodContractError("prepared record must be a mapping")
    ground_truth = record.get("ground_truth")
    if not isinstance(ground_truth, list) or not all(
        isinstance(value, str) and value for value in ground_truth
    ):
        raise MethodContractError(
            "prepared record 'ground_truth' must be a list of non-empty strings"
        )
    return CanonicalPreparedRecord(
        dataset_id=_required_string(record, "dataset_id"),
        turn=_required_string(record, "turn"),
        difficulty=_required_string(record, "difficulty"),
        instance_id=_required_string(record, "instance_id"),
        source_example_id=_required_string(record, "source_example_id"),
        query=_required_string(record, "query", allow_empty=True),
        ground_truth=tuple(ground_truth),
        _payload=copy.deepcopy(dict(record)),
    )


def validate_prepared_records(
    records: Sequence[Mapping[str, Any] | CanonicalPreparedRecord],
) -> tuple[CanonicalPreparedRecord, ...]:
    validated = tuple(
        record
        if isinstance(record, CanonicalPreparedRecord)
        else validate_prepared_record(record)
        for record in records
    )
    instance_ids = [record.instance_id for record in validated]
    if len(instance_ids) != len(set(instance_ids)):
        raise MethodContractError(
            "prepared records contain duplicate instance_id values"
        )
    return validated


@dataclass(frozen=True)
class MethodRunContext:
    """Manifest-bound inputs shared by method build and inference phases."""

    run_dir: Path
    dataset_manifest_sha256: str
    model_name: str = ""
    tools_schema: tuple[Mapping[str, Any], ...] = ()
    reasoning_effort: str | None = None

    def __post_init__(self) -> None:
        original = Path(self.run_dir)
        try:
            info = os.lstat(original)
        except OSError as exc:
            raise MethodContractError(
                f"run_dir is not readable: {original}"
            ) from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise MethodContractError(
                f"run_dir must be a real directory: {original}"
            )
        resolved = original.resolve(strict=True)
        digest = str(self.dataset_manifest_sha256).lower()
        if _SHA256.fullmatch(digest) is None:
            raise MethodContractError(
                "dataset_manifest_sha256 must be 64 lowercase hex digits"
            )
        if not isinstance(self.model_name, str):
            raise MethodContractError("model_name must be a string")
        if not all(isinstance(item, Mapping) for item in self.tools_schema):
            raise MethodContractError("tools_schema must contain mappings")
        object.__setattr__(self, "run_dir", resolved)
        object.__setattr__(self, "dataset_manifest_sha256", digest)
        object.__setattr__(
            self,
            "tools_schema",
            tuple(copy.deepcopy(dict(item)) for item in self.tools_schema),
        )


def _safe_relative_state_path(
    value: str | PurePosixPath,
) -> PurePosixPath:
    text = str(value)
    path = PurePosixPath(text)
    if (
        not text
        or not path.parts
        or "\\" in text
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise MethodContractError(f"unsafe method state path: {text!r}")
    return path


def method_state_path(
    context: MethodRunContext,
    relative_path: str | PurePosixPath,
) -> Path:
    """Resolve a lexical state path and reject every existing symlink component."""

    relative = _safe_relative_state_path(relative_path)
    current = context.run_dir
    for part in relative.parts:
        current = current / part
        if current.exists() or current.is_symlink():
            try:
                info = os.lstat(current)
            except OSError as exc:
                raise MethodContractError(
                    f"cannot inspect method state path: {current}"
                ) from exc
            if stat.S_ISLNK(info.st_mode):
                raise MethodContractError(
                    f"method state path may not contain symlinks: {current}"
                )
    try:
        current.absolute().relative_to(context.run_dir)
    except ValueError as exc:
        raise MethodContractError(
            f"method state path escapes run_dir: {relative}"
        ) from exc
    return current


def _records_sha256(
    records: Sequence[CanonicalPreparedRecord],
) -> str:
    digest = hashlib.sha256()
    for record in records:
        value = {
            "ground_truth": list(record.ground_truth),
            "instance_id": record.instance_id,
            "query": record.query,
            "source_example_id": record.source_example_id,
        }
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


@dataclass(frozen=True)
class MethodStateHandle:
    """A method state directory bound to one dataset manifest and record set."""

    method_id: str
    path: Path
    dataset_manifest_sha256: str
    records_sha256: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_manifest(self) -> dict[str, Any]:
        return {
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "metadata": copy.deepcopy(dict(self.metadata)),
            "method_id": self.method_id,
            "path": str(self.path),
            "records_sha256": self.records_sha256,
        }


def bind_method_state(
    *,
    method_id: str,
    context: MethodRunContext,
    relative_path: str | PurePosixPath,
    records: Sequence[CanonicalPreparedRecord],
    metadata: Mapping[str, Any] | None = None,
) -> MethodStateHandle:
    """Bind an existing real state directory to its exact prepared-record contract."""

    method_id = validate_method_id(method_id)
    validated = validate_prepared_records(records)
    path = method_state_path(context, relative_path)
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise MethodContractError(
            f"method state directory does not exist: {path}"
        ) from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise MethodContractError(
            f"method state must be a real directory: {path}"
        )
    metadata_value = copy.deepcopy(dict(metadata or {}))
    try:
        json.dumps(metadata_value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise MethodContractError(
            "method state metadata must be JSON serializable"
        ) from exc
    return MethodStateHandle(
        method_id=method_id,
        path=path,
        dataset_manifest_sha256=context.dataset_manifest_sha256,
        records_sha256=_records_sha256(validated),
        metadata=metadata_value,
    )


_CANONICAL_FIELDS = (
    "dataset_id",
    "turn",
    "difficulty",
    "instance_id",
    "source_example_id",
    "query",
    "ground_truth",
)


def canonical_prediction(
    record: CanonicalPreparedRecord,
    *,
    method_id: str,
    prediction: Mapping[str, Any] | str,
) -> dict[str, Any]:
    """Publish a backend result without allowing canonical identity or GT drift."""

    method_id = validate_method_id(method_id)
    if isinstance(prediction, str):
        result: dict[str, Any] = {
            "llm_output": prediction,
            "prediction": prediction,
            "response": prediction,
            "status": "ok",
        }
    elif isinstance(prediction, Mapping):
        result = copy.deepcopy(dict(prediction))
    else:
        raise MethodContractError(
            "method prediction must be a mapping or string"
        )

    canonical = record.as_mapping()
    canonical["ground_truth"] = list(record.ground_truth)
    for key in _CANONICAL_FIELDS:
        expected = canonical[key]
        if key in result and result[key] != expected:
            raise MethodContractError(
                f"method prediction changed canonical field {key!r}"
            )
        result[key] = copy.deepcopy(expected)
    if "method_id" in result and result["method_id"] != method_id:
        raise MethodContractError("method prediction changed method_id")
    result["method_id"] = method_id
    if "prediction" not in result:
        value = result.get(
            "llm_output",
            result.get("content", result.get("response")),
        )
        if not isinstance(value, str):
            raise MethodContractError(
                "method prediction lacks a string prediction value"
            )
        result["prediction"] = value
    if not isinstance(result["prediction"], str):
        raise MethodContractError(
            "method prediction value must be a string"
        )
    result.setdefault("llm_output", result["prediction"])
    result.setdefault("response", result["prediction"])
    result.setdefault("status", "ok")
    if result["status"] not in {"ok", "error"}:
        raise MethodContractError(
            "method prediction status must be 'ok' or 'error'"
        )
    return result


BackendT = TypeVar("BackendT")


class MethodAdapter(Protocol[BackendT]):
    """Lifecycle implemented by each real method adapter."""

    method_id: str

    def build(
        self,
        records: Sequence[CanonicalPreparedRecord],
        context: MethodRunContext,
        backend: BackendT,
    ) -> (
        MethodStateHandle
        | None
        | Awaitable[MethodStateHandle | None]
    ):
        """Optionally build/index method state bound to the prepared manifest."""

    def infer(
        self,
        record: CanonicalPreparedRecord,
        context: MethodRunContext,
        state: MethodStateHandle | None,
        backend: BackendT,
    ) -> Mapping[str, Any] | Awaitable[Mapping[str, Any]]:
        """Infer one canonical prediction through an injected backend."""


async def await_if_needed(value: Any) -> Any:
    """Await sync-or-async adapter/backend results without prescribing a runtime."""

    return await value if inspect.isawaitable(value) else value
