"""Strict, versioned experiment configuration for canonical suite entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Mapping, Sequence

from exp7.provenance.admission import (
    FileAdmissionError,
    admit_regular_file,
)


_SCHEMA_VERSION = 1
_EXPERIMENT_ID = re.compile(r"[a-z][a-z0-9_-]*\Z")
_RUN_PREFIX = re.compile(r"[a-z0-9][a-z0-9_-]*\Z")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_EXECUTABLE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*\Z")
_HISTORICAL_PARTS = frozenset(
    f"experiments{version}" for version in range(4, 7)
)
_CANONICAL_CONDITIONS = (
    ("singleturn", "easy"),
    ("singleturn", "medium"),
    ("singleturn", "hard"),
    ("multiturn", "easy"),
    ("multiturn", "medium"),
    ("multiturn", "hard"),
)
_SUPPORTED_EVALUATORS = frozenset({"exp6_slot_value_or_v1"})


class ExperimentConfigError(ValueError):
    """Raised when an experiment config fails closed."""


@dataclass(frozen=True)
class DatasetSettings:
    dataset_id: str
    format_version: int
    prepared_root: Path
    manifest: Path


@dataclass(frozen=True)
class ProviderSettings:
    provider_type: str
    model: str
    api_key_env: str
    base_url: str | None
    reasoning_effort: str | None
    temperature: float | None
    max_tokens: int | None
    timeout: float | None


@dataclass(frozen=True)
class OutputSettings:
    root: Path
    run_name_prefix: str


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int
    experiment_id: str
    dataset: DatasetSettings
    method_id: str
    provider: ProviderSettings
    conditions: tuple[tuple[str, str], ...]
    evaluator_variant: str
    output: OutputSettings
    python_bin: str
    config_path: Path
    config_relative_path: str
    config_sha256: str
    config_bytes: bytes

    def runner_extra_args(self) -> tuple[str, ...]:
        values = [
            "--experiment-config-path",
            self.config_relative_path,
            "--experiment-config-sha256",
            self.config_sha256,
            "--experiment-id",
            self.experiment_id,
            "--evaluator-variant",
            self.evaluator_variant,
        ]
        if self.provider.temperature is not None:
            values.extend(("--temperature", str(self.provider.temperature)))
        if self.provider.max_tokens is not None:
            values.extend(("--max-tokens", str(self.provider.max_tokens)))
        if self.provider.timeout is not None:
            values.extend(("--timeout", str(self.provider.timeout)))
        return tuple(values)


def _object(
    value: Any,
    *,
    label: str,
    keys: Sequence[str],
) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ExperimentConfigError(f"{label} must be a JSON object")
    expected = set(keys)
    actual = set(value)
    unknown = sorted(actual - expected)
    missing = sorted(expected - actual)
    if unknown:
        raise ExperimentConfigError(f"{label} has unknown keys: {unknown}")
    if missing:
        raise ExperimentConfigError(f"{label} is missing keys: {missing}")
    return value


def _string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentConfigError(f"{label} must be a non-empty string")
    return value


def _nullable_number(
    value: Any,
    *,
    label: str,
    minimum: float,
    maximum: float | None = None,
) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExperimentConfigError(f"{label} must be a number or null")
    number = float(value)
    if not math.isfinite(number):
        raise ExperimentConfigError(f"{label} must be finite")
    if number < minimum or (maximum is not None and number > maximum):
        raise ExperimentConfigError(f"{label} is outside the supported range")
    return number


def _repo_root(path: Path) -> Path:
    try:
        value = os.lstat(path)
    except OSError as exc:
        raise ExperimentConfigError(f"repository root is unreadable: {path}") from exc
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode):
        raise ExperimentConfigError(f"repository root must be a real directory: {path}")
    return path.resolve(strict=True)


def _repo_relative_path(
    root: Path,
    value: Any,
    *,
    label: str,
) -> tuple[str, Path]:
    text = _string(value, label=label)
    path = PurePosixPath(text)
    if (
        "\\" in text
        or path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ExperimentConfigError(
            f"{label} must be a safe repository-relative path"
        )
    if any(part.lower() in _HISTORICAL_PARTS for part in path.parts):
        raise ExperimentConfigError(f"{label} may not reference historical roots")
    current = root
    for part in path.parts:
        current = current / part
        if current.exists() or current.is_symlink():
            try:
                info = os.lstat(current)
            except OSError as exc:
                raise ExperimentConfigError(
                    f"cannot inspect {label}: {current}"
                ) from exc
            if stat.S_ISLNK(info.st_mode):
                raise ExperimentConfigError(
                    f"{label} may not traverse symlinks: {current}"
                )
    try:
        current.absolute().relative_to(root)
    except ValueError as exc:
        raise ExperimentConfigError(f"{label} escapes the repository") from exc
    return path.as_posix(), current


def _provider(value: Any) -> ProviderSettings:
    provider = _object(
        value,
        label="provider",
        keys=(
            "type",
            "model",
            "api_key_env",
            "base_url",
            "reasoning_effort",
            "temperature",
            "max_tokens",
            "timeout",
        ),
    )
    provider_type = _string(provider["type"], label="provider.type")
    if provider_type != "openai_compatible":
        raise ExperimentConfigError(
            "provider.type must be 'openai_compatible'"
        )
    model = _string(provider["model"], label="provider.model")
    api_key_env = _string(provider["api_key_env"], label="provider.api_key_env")
    if _ENV_NAME.fullmatch(api_key_env) is None:
        raise ExperimentConfigError("provider.api_key_env is invalid")
    base_url = provider["base_url"]
    if base_url is not None and (
        not isinstance(base_url, str)
        or not base_url.startswith(("http://", "https://"))
    ):
        raise ExperimentConfigError(
            "provider.base_url must be an HTTP(S) URL or null"
        )
    reasoning_effort = provider["reasoning_effort"]
    if reasoning_effort not in {None, "minimal", "low", "medium", "high"}:
        raise ExperimentConfigError("provider.reasoning_effort is invalid")
    temperature = _nullable_number(
        provider["temperature"],
        label="provider.temperature",
        minimum=0.0,
        maximum=2.0,
    )
    max_tokens = provider["max_tokens"]
    if max_tokens is not None and (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens <= 0
    ):
        raise ExperimentConfigError(
            "provider.max_tokens must be a positive integer or null"
        )
    timeout = _nullable_number(
        provider["timeout"],
        label="provider.timeout",
        minimum=0.000001,
    )
    return ProviderSettings(
        provider_type=provider_type,
        model=model,
        api_key_env=api_key_env,
        base_url=base_url,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )


def _conditions(value: Any) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list):
        raise ExperimentConfigError("conditions must be a JSON array")
    conditions: list[tuple[str, str]] = []
    for index, raw in enumerate(value):
        item = _object(
            raw,
            label=f"conditions[{index}]",
            keys=("turn", "difficulty"),
        )
        conditions.append(
            (
                _string(item["turn"], label=f"conditions[{index}].turn"),
                _string(
                    item["difficulty"],
                    label=f"conditions[{index}].difficulty",
                ),
            )
        )
    result = tuple(conditions)
    if result != _CANONICAL_CONDITIONS:
        raise ExperimentConfigError(
            "conditions must contain all six canonical conditions in order"
        )
    return result


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ExperimentConfigError(f"experiment config has duplicate key: {key!r}")
        value[key] = item
    return value


def _decode_config(payload: bytes) -> Any:
    def reject_constant(value: str) -> None:
        raise ExperimentConfigError(
            f"experiment config contains non-finite JSON number: {value}"
        )

    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExperimentConfigError(
            "experiment config is not valid strict UTF-8 JSON"
        ) from exc


def _experiment_config(
    repo_root: Path,
    config_path: str | Path,
    *,
    config_bytes: bytes | None = None,
    expected_sha256: str | None = None,
) -> ExperimentConfig:
    root = _repo_root(Path(repo_root))
    relative_config, resolved_config = _repo_relative_path(
        root,
        str(config_path),
        label="config path",
    )
    if config_bytes is None:
        try:
            admitted = admit_regular_file(
                resolved_config,
                label="experiment config",
            )
        except FileAdmissionError as exc:
            raise ExperimentConfigError(str(exc)) from exc
        payload = admitted.payload
        admitted_path = admitted.path
        sha256 = admitted.sha256
    else:
        if not isinstance(config_bytes, bytes):
            raise ExperimentConfigError("experiment config bytes must be bytes")
        payload = config_bytes
        admitted_path = resolved_config
        sha256 = hashlib.sha256(payload).hexdigest()
        if expected_sha256 is not None and sha256 != expected_sha256:
            raise ExperimentConfigError(
                "experiment config SHA-256 does not match its bytes"
            )
    raw = _decode_config(payload)
    document = _object(
        raw,
        label="experiment config",
        keys=(
            "schema_version",
            "experiment_id",
            "dataset",
            "method",
            "provider",
            "conditions",
            "evaluator",
            "output",
            "runtime",
        ),
    )
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != _SCHEMA_VERSION
    ):
        raise ExperimentConfigError(
            f"schema_version must be {_SCHEMA_VERSION}"
        )
    experiment_id = _string(
        document["experiment_id"],
        label="experiment_id",
    )
    if _EXPERIMENT_ID.fullmatch(experiment_id) is None:
        raise ExperimentConfigError("experiment_id is invalid")

    dataset_raw = _object(
        document["dataset"],
        label="dataset",
        keys=("dataset_id", "format_version", "prepared_root", "manifest"),
    )
    dataset_id = _string(dataset_raw["dataset_id"], label="dataset.dataset_id")
    if dataset_id != "mix600-v1":
        raise ExperimentConfigError("dataset.dataset_id must be 'mix600-v1'")
    if (
        type(dataset_raw["format_version"]) is not int
        or dataset_raw["format_version"] != 1
    ):
        raise ExperimentConfigError("dataset.format_version must be 1")
    _, prepared_root = _repo_relative_path(
        root,
        dataset_raw["prepared_root"],
        label="dataset.prepared_root",
    )
    _, manifest = _repo_relative_path(
        root,
        dataset_raw["manifest"],
        label="dataset.manifest",
    )
    if manifest != prepared_root / "manifest.json":
        raise ExperimentConfigError(
            "dataset.manifest must be manifest.json beneath dataset.prepared_root"
        )

    method_raw = _object(
        document["method"],
        label="method",
        keys=("method_id",),
    )
    method_id = _string(method_raw["method_id"], label="method.method_id")
    if method_id != "vanilla_llm":
        raise ExperimentConfigError(
            f"unknown method for this entrypoint: {method_id!r}"
        )

    evaluator_raw = _object(
        document["evaluator"],
        label="evaluator",
        keys=("variant",),
    )
    evaluator_variant = _string(
        evaluator_raw["variant"],
        label="evaluator.variant",
    )
    if evaluator_variant not in _SUPPORTED_EVALUATORS:
        raise ExperimentConfigError(
            f"unsupported evaluator variant: {evaluator_variant!r}"
        )

    output_raw = _object(
        document["output"],
        label="output",
        keys=("root", "run_name_prefix"),
    )
    _, output_root = _repo_relative_path(
        root,
        output_raw["root"],
        label="output.root",
    )
    run_name_prefix = _string(
        output_raw["run_name_prefix"],
        label="output.run_name_prefix",
    )
    if _RUN_PREFIX.fullmatch(run_name_prefix) is None:
        raise ExperimentConfigError("output.run_name_prefix is invalid")

    runtime_raw = _object(
        document["runtime"],
        label="runtime",
        keys=("python_bin",),
    )
    python_bin = _string(runtime_raw["python_bin"], label="runtime.python_bin")
    if _EXECUTABLE.fullmatch(python_bin) is None:
        raise ExperimentConfigError(
            "runtime.python_bin must be a simple executable name"
        )

    return ExperimentConfig(
        schema_version=_SCHEMA_VERSION,
        experiment_id=experiment_id,
        dataset=DatasetSettings(
            dataset_id=dataset_id,
            format_version=1,
            prepared_root=prepared_root,
            manifest=manifest,
        ),
        method_id=method_id,
        provider=_provider(document["provider"]),
        conditions=_conditions(document["conditions"]),
        evaluator_variant=evaluator_variant,
        output=OutputSettings(
            root=output_root,
            run_name_prefix=run_name_prefix,
        ),
        python_bin=python_bin,
        config_path=admitted_path,
        config_relative_path=relative_config,
        config_sha256=sha256,
        config_bytes=payload,
    )


def load_experiment_config(
    repo_root: Path,
    config_path: str | Path,
) -> ExperimentConfig:
    """Admit, hash, parse, and validate one canonical experiment config."""

    return _experiment_config(repo_root, config_path)


def parse_experiment_config_bytes(
    repo_root: Path,
    config_relative_path: str,
    payload: bytes,
    *,
    expected_sha256: str | None = None,
) -> ExperimentConfig:
    """Strictly parse already-admitted config bytes without reopening the source."""

    return _experiment_config(
        repo_root,
        config_relative_path,
        config_bytes=payload,
        expected_sha256=expected_sha256,
    )


__all__ = [
    "DatasetSettings",
    "ExperimentConfig",
    "ExperimentConfigError",
    "OutputSettings",
    "ProviderSettings",
    "load_experiment_config",
    "parse_experiment_config_bytes",
]
