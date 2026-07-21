"""Atomic, manifest-bound artifact management for six-condition runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import tempfile
from typing import Any, Callable, Mapping, Sequence, TextIO

from exp7.evaluation import EvaluationSummary, evaluate_run
from exp7.experiments import (
    ExperimentConfig,
    ExperimentConfigError,
    parse_experiment_config_bytes,
)
from exp7.methods.vanilla_llm.runner import (
    admit_manifest,
    checkpoint_integrity_path,
    verify_prepared_input,
)
from exp7.provenance.admission import (
    AdmittedDirectory,
    FileAdmissionError,
    StrictJSONError,
    admit_directory,
    admit_regular_file,
    create_directory,
    decode_strict_json,
    lexical_absolute,
)


CONDITIONS = tuple(
    (turn, difficulty)
    for turn in ("singleturn", "multiturn")
    for difficulty in ("easy", "medium", "hard")
)
TERMINAL_STATUSES = frozenset(
    {"complete", "complete_with_provider_errors", "all_provider_errors"}
)
TOOL_SCHEMA_FILES = {
    "singleturn": "schema_easy.json",
    "multiturn": "schema_all.json",
}
TOOL_SCHEMA_SNAPSHOTS = {
    "singleturn": "tool_schemas/singleturn.json",
    "multiturn": "tool_schemas/multiturn.json",
}


class RunArtifactError(RuntimeError):
    """Raised when a run cannot be created or safely resumed."""


@dataclass(frozen=True)
class ConditionInvocation:
    """One manifest-bound condition passed to an injectable runner."""

    turn: str
    difficulty: str
    prepared_path: Path
    output_path: Path
    checkpoint_path: Path
    command: tuple[str, ...]

    @property
    def label(self) -> str:
        return f"{self.turn}.{self.difficulty}"


@dataclass(frozen=True)
class ToolSchemaInput:
    turn: str
    source_path: Path
    payload: bytes
    sha256: str


@dataclass(frozen=True)
class SuiteSummary:
    status: str
    run_dir: Path
    completed_conditions: int
    prediction_count: int = 0
    provider_ok_count: int = 0
    provider_error_count: int = 0


ConditionRunner = Callable[[ConditionInvocation, TextIO], int]
EvaluationRunner = Callable[..., EvaluationSummary]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _resolve_python_runtime(requested: str) -> dict[str, Any]:
    located = shutil.which(requested)
    if located is None:
        raise RunArtifactError(f"cannot resolve Python executable: {requested!r}")
    executable = Path(located).resolve(strict=True)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise RunArtifactError(f"Python executable is not runnable: {executable}")
    probe = (
        "import json,os,platform,sys;"
        "print(json.dumps({"
        "'base_prefix':sys.base_prefix,"
        "'executable':sys.executable,"
        "'executable_realpath':os.path.realpath(sys.executable),"
        "'implementation':platform.python_implementation(),"
        "'prefix':sys.prefix,"
        "'version':platform.python_version()"
        "},sort_keys=True))"
    )
    result = subprocess.run(
        [str(executable), "-I", "-B", "-c", probe],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RunArtifactError(
            f"cannot fingerprint Python executable {executable}: {result.stderr.strip()}"
        )
    try:
        observed = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RunArtifactError(
            f"Python executable returned an invalid fingerprint: {executable}"
        ) from exc
    required = {
        "base_prefix",
        "executable",
        "executable_realpath",
        "implementation",
        "prefix",
        "version",
    }
    if not isinstance(observed, dict) or set(observed) != required or any(
        not isinstance(observed[key], str) or not observed[key]
        for key in required
    ):
        raise RunArtifactError(
            f"Python executable returned an incomplete fingerprint: {executable}"
        )
    value = {"requested": requested, **observed}
    value["fingerprint_sha256"] = _sha256(_canonical_json_bytes(value))
    return value


def _atomic_write(
    run_root: AdmittedDirectory,
    path: Path,
    payload: bytes,
) -> None:
    try:
        run_root.atomic_write(
            run_root.relative(path),
            payload,
            label=f"run artifact {path.name}",
        )
    except FileAdmissionError as exc:
        raise RunArtifactError(str(exc)) from exc


def _write_exclusive(
    run_root: AdmittedDirectory,
    path: Path,
    payload: bytes,
) -> None:
    try:
        run_root.write_exclusive(
            run_root.relative(path),
            payload,
            label=f"run artifact {path.name}",
        )
    except FileAdmissionError as exc:
        raise RunArtifactError(str(exc)) from exc


def _decode_json(payload: bytes, label: str) -> Any:
    try:
        return decode_strict_json(payload, label=label)
    except StrictJSONError as exc:
        raise RunArtifactError(str(exc)) from exc


def _load_json(path: Path, label: str) -> Any:
    try:
        admitted = admit_regular_file(path, label=label)
    except FileAdmissionError as exc:
        raise RunArtifactError(str(exc)) from exc
    return _decode_json(admitted.payload, label)


def verify_manifest_bundle(
    prepared_root: Path,
) -> tuple[dict[str, Any], bytes, str, dict[str, list[dict[str, Any]]]]:
    """Verify all canonical members before creating any run artifacts."""

    prepared_root = lexical_absolute(prepared_root)
    try:
        root_value = os.lstat(prepared_root)
    except OSError as exc:
        raise RunArtifactError(f"cannot inspect prepared root {prepared_root}: {exc}") from exc
    if stat.S_ISLNK(root_value.st_mode) or not stat.S_ISDIR(root_value.st_mode):
        raise RunArtifactError(
            f"prepared root must be a non-symlink directory: {prepared_root}"
        )
    manifest_path = prepared_root / "manifest.json"
    try:
        admitted_manifest = admit_manifest(manifest_path)
    except ValueError as exc:
        raise RunArtifactError(f"invalid dataset manifest: {exc}") from exc
    manifest = admitted_manifest.value

    records: dict[str, list[dict[str, Any]]] = {}
    for turn, difficulty in CONDITIONS:
        label = f"{turn}.{difficulty}"
        prepared_path = prepared_root / turn / f"{difficulty}.jsonl"
        try:
            condition_records, _ = verify_prepared_input(
                prepared_path,
                manifest_path,
                admitted_manifest=admitted_manifest,
            )
        except (OSError, ValueError) as exc:
            raise RunArtifactError(f"invalid prepared condition {label}: {exc}") from exc
        records[label] = condition_records
    return (
        manifest,
        admitted_manifest.payload,
        admitted_manifest.sha256,
        records,
    )


def _condition_statuses(run_dir: Path) -> dict[str, dict[str, Any]]:
    return {
        f"{turn}.{difficulty}": {
            "checkpoint": (
                f"predictions/{turn}/{difficulty}/checkpoint.jsonl"
            ),
            "prediction_count": 0,
            "predictions": (
                f"predictions/{turn}/{difficulty}/predictions.json"
            ),
            "provider_error_count": 0,
            "provider_ok_count": 0,
            "status": "pending",
        }
        for turn, difficulty in CONDITIONS
    }


def _aggregate_counts(
    conditions: Mapping[str, Mapping[str, Any]],
) -> dict[str, int]:
    return {
        field: sum(int(value.get(field, 0)) for value in conditions.values())
        for field in (
            "prediction_count",
            "provider_ok_count",
            "provider_error_count",
        )
    }


def _completion_status(
    *,
    prediction_count: int,
    provider_ok_count: int,
    provider_error_count: int,
) -> str:
    if provider_error_count == 0:
        return "complete"
    if prediction_count > 0 and provider_error_count == prediction_count:
        return "all_provider_errors"
    if provider_ok_count > 0:
        return "complete_with_provider_errors"
    raise RunArtifactError("provider counts do not describe a complete prediction set")


def _status_document(
    run_dir: Path,
    status: str,
    conditions: Mapping[str, Mapping[str, Any]],
    *,
    error: str | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "conditions": conditions,
        "format_version": 1,
        "run_id": run_dir.name,
        "status": status,
        "updated_at": _utc_now(),
    }
    value.update(_aggregate_counts(conditions))
    if error is not None:
        value["error"] = error
    return value


def _write_status(
    run_dir: Path,
    status: str,
    conditions: Mapping[str, Mapping[str, Any]],
    *,
    run_root: AdmittedDirectory,
    error: str | None = None,
) -> None:
    _atomic_write(
        run_root,
        run_dir / "status.json",
        _json_bytes(_status_document(run_dir, status, conditions, error=error)),
    )


def _write_metrics_status(
    run_dir: Path,
    status: str,
    *,
    run_root: AdmittedDirectory,
    variant: str | None = None,
    reason: str | None = None,
    error: str | None = None,
    summary: EvaluationSummary | None = None,
    metrics_sha256: str | None = None,
    provider_status: str | None = None,
) -> None:
    value: dict[str, Any] = {
        "format_version": 1,
        "status": status,
        "updated_at": _utc_now(),
    }
    if variant is not None:
        value["variant"] = variant
    if reason is not None:
        value["reason"] = reason
    if provider_status is not None:
        value["provider_status"] = provider_status
    if error is not None:
        value["error"] = error
    if summary is not None:
        value.update(
            {
                "condition_count": summary.condition_count,
                "metrics": "metrics.json",
                "metrics_sha256": metrics_sha256,
                "parse_failure_count": summary.parse_failure_count,
                "row_count": summary.row_count,
                "row_error_count": summary.row_error_count,
            }
        )
    _atomic_write(run_root, run_dir / "metrics/status.json", _json_bytes(value))


def _admit_tool_schemas(repo_root: Path) -> dict[str, ToolSchemaInput]:
    values: dict[str, ToolSchemaInput] = {}
    for turn, filename in TOOL_SCHEMA_FILES.items():
        source = repo_root / "configs/schemas/mix600-v1" / filename
        try:
            admitted = admit_regular_file(source, label=f"{turn} tool schema")
        except FileAdmissionError as exc:
            raise RunArtifactError(str(exc)) from exc
        document = _decode_json(admitted.payload, f"{turn} tool schema")
        if not isinstance(document, list) or not all(
            isinstance(item, dict) for item in document
        ):
            raise RunArtifactError(
                f"{turn} tool schema must be a JSON array of objects"
            )
        values[turn] = ToolSchemaInput(
            turn=turn,
            source_path=admitted.path,
            payload=admitted.payload,
            sha256=admitted.sha256,
        )
    return values


def _tool_schema_config(
    schemas: Mapping[str, ToolSchemaInput],
) -> dict[str, dict[str, str]]:
    return {
        turn: {
            "path": TOOL_SCHEMA_SNAPSHOTS[turn],
            "sha256": schemas[turn].sha256,
            "source": str(schemas[turn].source_path),
        }
        for turn in TOOL_SCHEMA_FILES
    }


def _validate_config_binding(
    *,
    repo_root: Path,
    prepared_root: Path,
    run_dir: Path,
    model: str,
    python_bin: str,
    reasoning_effort: str | None,
    base_url: str | None,
    api_key_env: str,
    extra_args: Sequence[str],
    experiment_config_bytes: bytes,
    experiment_config_sha256: str,
    experiment_config_relative_path: str,
    evaluator_variant: str,
    output_policy: str,
) -> ExperimentConfig:
    try:
        config = parse_experiment_config_bytes(
            repo_root,
            experiment_config_relative_path,
            experiment_config_bytes,
            expected_sha256=experiment_config_sha256,
        )
    except ExperimentConfigError as exc:
        raise RunArtifactError(f"invalid experiment config: {exc}") from exc

    actual_prepared = lexical_absolute(prepared_root)
    expected = {
        "api_key_env": config.provider.api_key_env,
        "base_url": config.provider.base_url,
        "evaluator_variant": config.evaluator_variant,
        "model": config.provider.model,
        "prepared_root": config.dataset.prepared_root,
        "python_bin": config.python_bin,
        "reasoning_effort": config.provider.reasoning_effort,
        "runner_extra_args": config.runner_extra_args(),
    }
    actual = {
        "api_key_env": api_key_env,
        "base_url": base_url,
        "evaluator_variant": evaluator_variant,
        "model": model,
        "prepared_root": actual_prepared,
        "python_bin": python_bin,
        "reasoning_effort": reasoning_effort,
        "runner_extra_args": tuple(extra_args),
    }
    mismatches = [key for key in expected if actual[key] != expected[key]]
    if config.dataset.manifest != actual_prepared / "manifest.json":
        mismatches.append("dataset_manifest")
    if mismatches:
        raise RunArtifactError(
            "experiment config differs from execution semantics: "
            + ", ".join(sorted(mismatches))
        )
    if output_policy not in {"config_root", "external_opt_in"}:
        raise RunArtifactError("configured runs require an explicit output policy")
    if output_policy == "config_root":
        try:
            lexical_absolute(run_dir).relative_to(config.output.root)
        except ValueError as exc:
            raise RunArtifactError(
                "config-root run directory escapes configured output.root"
            ) from exc
    return config


def _resolved_config(
    *,
    repo_root: Path,
    prepared_root: Path,
    run_dir: Path,
    model: str,
    python_bin: str,
    python_runtime: Mapping[str, Any],
    manifest_sha256: str,
    reasoning_effort: str | None,
    base_url: str | None,
    api_key_env: str,
    extra_args: Sequence[str],
    experiment_config_sha256: str | None,
    experiment_config_relative_path: str | None,
    evaluator_variant: str | None,
    output_policy: str,
    tool_schemas: Mapping[str, ToolSchemaInput],
) -> dict[str, Any]:
    value = {
        "api_key_env": api_key_env,
        "base_url": base_url,
        "conditions": [f"{turn}.{difficulty}" for turn, difficulty in CONDITIONS],
        "dataset_manifest": str(prepared_root / "manifest.json"),
        "dataset_manifest_sha256": manifest_sha256,
        "format_version": 1,
        "method": "vanilla_llm",
        "model": model,
        "prepared_root": str(prepared_root),
        "python": python_bin,
        "python_runtime": dict(python_runtime),
        "reasoning_effort": reasoning_effort,
        "repo_root": str(repo_root),
        "run_dir": str(run_dir),
        "runner_extra_args": list(extra_args),
        "output_policy": output_policy,
        "tool_schemas": _tool_schema_config(tool_schemas),
    }
    if experiment_config_sha256 is not None:
        value.update(
            {
                "evaluator_variant": evaluator_variant,
                "experiment_config": "experiment_config.json",
                "experiment_config_relative_path": experiment_config_relative_path,
                "experiment_config_sha256": experiment_config_sha256,
            }
        )
    return value


def _invocation(
    *,
    repo_root: Path,
    prepared_root: Path,
    run_dir: Path,
    manifest_sha256: str,
    turn: str,
    difficulty: str,
    python_bin: str,
    model: str,
    reasoning_effort: str | None,
    base_url: str | None,
    api_key_env: str,
    tool_schema_path: Path,
    tool_schema_sha256: str,
    extra_args: Sequence[str],
    resume: bool,
    output_root_identity: tuple[int, int] | None = None,
) -> ConditionInvocation:
    condition_dir = run_dir / "predictions" / turn / difficulty
    output_path = condition_dir / "predictions.json"
    checkpoint_path = condition_dir / "checkpoint.jsonl"
    command = [
        python_bin,
        "-B",
        str(repo_root / "scripts/run.py"),
        "--prepared-jsonl",
        str(prepared_root / turn / f"{difficulty}.jsonl"),
        "--manifest",
        str(prepared_root / "manifest.json"),
        "--manifest-sha256",
        manifest_sha256,
        "--tools-schema",
        str(tool_schema_path),
        "--tools-schema-sha256",
        tool_schema_sha256,
        "--output",
        str(output_path),
        "--checkpoint",
        str(checkpoint_path),
        "--model",
        model,
        "--api-key-env",
        api_key_env,
    ]
    if reasoning_effort is not None:
        command.extend(("--reasoning-effort", reasoning_effort))
    if base_url is not None:
        command.extend(("--base-url", base_url))
    if output_root_identity is not None:
        command.extend(
            (
                "--output-root",
                str(run_dir),
                "--output-root-device",
                str(output_root_identity[0]),
                "--output-root-inode",
                str(output_root_identity[1]),
            )
        )
    if resume:
        command.append("--resume")
    command.extend(extra_args)
    return ConditionInvocation(
        turn=turn,
        difficulty=difficulty,
        prepared_path=prepared_root / turn / f"{difficulty}.jsonl",
        output_path=output_path,
        checkpoint_path=checkpoint_path,
        command=tuple(command),
    )


def subprocess_condition_runner(invocation: ConditionInvocation, log: TextIO) -> int:
    """Run one condition while recording its combined provider output."""

    result = subprocess.run(
        invocation.command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log.write(result.stdout)
    log.flush()
    return result.returncode


def _verify_prediction(
    path: Path,
    expected: Sequence[Mapping[str, Any]],
    *,
    run_root: AdmittedDirectory,
) -> dict[str, int]:
    try:
        admitted = run_root.admit_file(
            run_root.relative(path),
            label="condition predictions",
        )
    except FileAdmissionError as exc:
        raise RunArtifactError(str(exc)) from exc
    value = _decode_json(admitted.payload, "condition predictions")
    if not isinstance(value, list):
        raise RunArtifactError(f"predictions must be a JSON array: {path}")
    if any(not isinstance(record, dict) for record in value):
        raise RunArtifactError(f"every prediction must be a JSON object: {path}")
    expected_ids = [str(record["instance_id"]) for record in expected]
    actual_ids = [str(record.get("instance_id", "")) for record in value]
    if actual_ids != expected_ids or len(value) != len(expected_ids):
        raise RunArtifactError(f"predictions do not match prepared instance order: {path}")
    statuses = [record.get("status") for record in value]
    if any(status not in {"ok", "error"} for status in statuses):
        raise RunArtifactError(
            f"prediction rows must have status 'ok' or 'error': {path}"
        )
    return {
        "prediction_count": len(value),
        "provider_ok_count": statuses.count("ok"),
        "provider_error_count": statuses.count("error"),
    }


def _resume_conditions(
    run_dir: Path,
    status: Mapping[str, Any],
    records: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    run_root: AdmittedDirectory,
) -> dict[str, dict[str, Any]]:
    raw = status.get("conditions")
    if not isinstance(raw, dict):
        raise RunArtifactError("run status lacks a conditions mapping")
    conditions = {label: dict(raw.get(label, {})) for label in records}
    if any(not value for value in conditions.values()):
        raise RunArtifactError("run status does not contain all six conditions")
    for label, expected in records.items():
        turn, difficulty = label.split(".", 1)
        condition_dir = run_dir / "predictions" / turn / difficulty
        output = condition_dir / "predictions.json"
        checkpoint = condition_dir / "checkpoint.jsonl"
        try:
            output_exists = run_root.exists(
                run_root.relative(output),
                label=f"predictions {label}",
            )
            checkpoint_exists = run_root.exists(
                run_root.relative(checkpoint),
                label=f"checkpoint {label}",
            )
        except FileAdmissionError as exc:
            raise RunArtifactError(str(exc)) from exc
        if output_exists:
            if not checkpoint_exists:
                raise RunArtifactError(f"completed predictions lack a safe checkpoint: {label}")
            try:
                run_root.admit_file(
                    run_root.relative(checkpoint),
                    label=f"checkpoint {label}",
                )
            except FileAdmissionError as exc:
                raise RunArtifactError(str(exc)) from exc
            counts = _verify_prediction(output, expected, run_root=run_root)
            conditions[label].update(counts)
            conditions[label]["status"] = _completion_status(**counts)
        elif conditions[label].get("status") in TERMINAL_STATUSES:
            raise RunArtifactError(f"terminal status lacks predictions: {label}")
    return conditions


def _initialize_run(
    *,
    run_dir: Path,
    run_root: AdmittedDirectory,
    config: Mapping[str, Any],
    manifest_bytes: bytes,
    experiment_config_bytes: bytes | None,
    evaluator_variant: str | None,
    tool_schemas: Mapping[str, ToolSchemaInput],
) -> dict[str, dict[str, Any]]:
    conditions = _condition_statuses(run_dir)
    _write_exclusive(run_root, run_dir / "resolved_config.json", _json_bytes(config))
    _write_exclusive(run_root, run_dir / "dataset_manifest.json", manifest_bytes)
    for turn, relative in TOOL_SCHEMA_SNAPSHOTS.items():
        _write_exclusive(run_root, run_dir / relative, tool_schemas[turn].payload)
    if experiment_config_bytes is not None:
        _write_exclusive(
            run_root,
            run_dir / "experiment_config.json",
            experiment_config_bytes,
        )
    _write_exclusive(run_root, run_dir / "run.log", b"")
    _write_metrics_status(
        run_dir,
        "pending",
        run_root=run_root,
        variant=evaluator_variant,
        reason="evaluation_not_run",
    )
    _write_status(run_dir, "planned", conditions, run_root=run_root)
    return conditions


def _load_resume(
    *,
    run_dir: Path,
    run_root: AdmittedDirectory,
    config: Mapping[str, Any],
    manifest_bytes: bytes,
    experiment_config_bytes: bytes | None,
    records: Mapping[str, Sequence[Mapping[str, Any]]],
    tool_schemas: Mapping[str, ToolSchemaInput],
) -> dict[str, dict[str, Any]]:
    required = [
        "resolved_config.json",
        "dataset_manifest.json",
        "run.log",
        "status.json",
        "metrics/status.json",
    ]
    if experiment_config_bytes is not None:
        required.append("experiment_config.json")
    required.extend(TOOL_SCHEMA_SNAPSHOTS.values())
    admitted_artifacts = {}
    for relative in required:
        try:
            admitted_artifacts[relative] = run_root.admit_file(
                relative,
                label=f"resume artifact {relative}",
            )
        except FileAdmissionError as exc:
            raise RunArtifactError(str(exc)) from exc
    if (
        _decode_json(
            admitted_artifacts["resolved_config.json"].payload,
            "resolved config",
        )
        != config
    ):
        raise RunArtifactError("resume config differs from the resolved run config")
    _decode_json(
        admitted_artifacts["dataset_manifest.json"].payload,
        "dataset manifest",
    )
    if admitted_artifacts["dataset_manifest.json"].payload != manifest_bytes:
        raise RunArtifactError("resume dataset manifest differs from the verified source")
    if experiment_config_bytes is not None:
        admitted_config = admitted_artifacts["experiment_config.json"]
        if admitted_config.payload != experiment_config_bytes:
            raise RunArtifactError(
                "resume experiment config differs from the admitted source"
            )
    for turn, relative in TOOL_SCHEMA_SNAPSHOTS.items():
        admitted_schema = admitted_artifacts[relative]
        if (
            admitted_schema.sha256 != tool_schemas[turn].sha256
            or admitted_schema.payload != tool_schemas[turn].payload
        ):
            raise RunArtifactError(
                f"resume {turn} tool schema snapshot differs from admitted source"
            )
    status = _decode_json(admitted_artifacts["status.json"].payload, "run status")
    if not isinstance(status, dict):
        raise RunArtifactError("run status must be a JSON object")
    if status.get("status") in TERMINAL_STATUSES:
        raise RunArtifactError("completed runs cannot be resumed")
    metrics_status = _decode_json(
        admitted_artifacts["metrics/status.json"].payload,
        "metrics status",
    )
    if not isinstance(metrics_status, dict):
        raise RunArtifactError("metrics status must be a JSON object")
    if metrics_status.get("status") != "pending":
        raise RunArtifactError("suite runner only resumes runs with pending evaluation")
    return _resume_conditions(
        run_dir,
        status,
        records,
        run_root=run_root,
    )


def _evaluate_terminal_run(
    *,
    run_dir: Path,
    run_root: AdmittedDirectory,
    variant: str,
    provider_status: str,
    evaluation_runner: EvaluationRunner,
) -> EvaluationSummary:
    _write_metrics_status(
        run_dir,
        "running",
        run_root=run_root,
        variant=variant,
        provider_status=provider_status,
    )
    try:
        with run_root.open_text(
            "run.log",
            label="run log",
            append=True,
        ) as log:
            log.write(f"[{_utc_now()}] START evaluation:{variant}\n")
            log.flush()
    except FileAdmissionError as exc:
        raise RunArtifactError(str(exc)) from exc
    try:
        summary = evaluation_runner(run_dir=run_dir, variant=variant)
        expected = run_dir / "metrics/metrics.json"
        if lexical_absolute(summary.output_path) != expected:
            raise RunArtifactError(
                f"evaluator published metrics outside the canonical path: "
                f"{summary.output_path}"
            )
        try:
            admitted_metrics = run_root.admit_file(
                "metrics/metrics.json",
                label="evaluation metrics",
            )
        except FileAdmissionError as exc:
            raise RunArtifactError(str(exc)) from exc
    except BaseException as exc:
        _write_metrics_status(
            run_dir,
            "failed",
            run_root=run_root,
            variant=variant,
            error=str(exc),
            provider_status=provider_status,
        )
        try:
            with run_root.open_text(
                "run.log",
                label="run log",
                append=True,
            ) as log:
                log.write(f"[{_utc_now()}] FAILED evaluation:{variant}: {exc}\n")
                log.flush()
        except FileAdmissionError as log_exc:
            raise RunArtifactError(str(log_exc)) from log_exc
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        if isinstance(exc, RunArtifactError):
            raise
        raise RunArtifactError(f"evaluation failed: {exc}") from exc
    _write_metrics_status(
        run_dir,
        "complete",
        run_root=run_root,
        variant=variant,
        summary=summary,
        metrics_sha256=admitted_metrics.sha256,
        provider_status=provider_status,
    )
    try:
        with run_root.open_text(
            "run.log",
            label="run log",
            append=True,
        ) as log:
            log.write(f"[{_utc_now()}] COMPLETE evaluation:{variant}\n")
            log.flush()
    except FileAdmissionError as exc:
        raise RunArtifactError(str(exc)) from exc
    return summary


def run_suite(
    *,
    repo_root: Path,
    prepared_root: Path,
    run_dir: Path,
    model: str,
    python_bin: str,
    reasoning_effort: str | None = None,
    base_url: str | None = None,
    api_key_env: str = "OPENAI_API_KEY",
    extra_args: Sequence[str] = (),
    experiment_config_bytes: bytes | None = None,
    experiment_config_sha256: str | None = None,
    experiment_config_relative_path: str | None = None,
    evaluator_variant: str | None = None,
    output_policy: str = "legacy",
    evaluation_runner: EvaluationRunner | None = None,
    resume: bool = False,
    dry_run: bool = False,
    condition_runner: ConditionRunner = subprocess_condition_runner,
    output: TextIO | None = None,
) -> SuiteSummary:
    """Verify and run all six conditions in canonical order."""

    integration_values = (
        experiment_config_bytes,
        experiment_config_sha256,
        experiment_config_relative_path,
        evaluator_variant,
    )
    if any(value is not None for value in integration_values) and not all(
        value is not None for value in integration_values
    ):
        raise RunArtifactError(
            "experiment config bytes, SHA-256, relative path, and evaluator variant "
            "must be supplied together"
        )
    if experiment_config_bytes is not None and (
        not isinstance(experiment_config_bytes, bytes)
        or _sha256(experiment_config_bytes) != experiment_config_sha256
    ):
        raise RunArtifactError("experiment config SHA-256 does not match its bytes")

    repo_root = repo_root.resolve(strict=True)
    run_dir = lexical_absolute(run_dir)
    stream = output
    configured = experiment_config_bytes is not None
    if configured:
        assert experiment_config_sha256 is not None
        assert experiment_config_relative_path is not None
        assert evaluator_variant is not None
        _validate_config_binding(
            repo_root=repo_root,
            prepared_root=prepared_root,
            run_dir=run_dir,
            model=model,
            python_bin=python_bin,
            reasoning_effort=reasoning_effort,
            base_url=base_url,
            api_key_env=api_key_env,
            extra_args=extra_args,
            experiment_config_bytes=experiment_config_bytes,
            experiment_config_sha256=experiment_config_sha256,
            experiment_config_relative_path=experiment_config_relative_path,
            evaluator_variant=evaluator_variant,
            output_policy=output_policy,
        )
    elif output_policy != "legacy":
        raise RunArtifactError("legacy runs require output_policy='legacy'")

    python_runtime = _resolve_python_runtime(python_bin)
    executable = str(python_runtime["executable"])
    tool_schemas = _admit_tool_schemas(repo_root)

    if dry_run:
        prepared_root = lexical_absolute(prepared_root)
        for turn, difficulty in CONDITIONS:
            invocation = _invocation(
                repo_root=repo_root,
                prepared_root=prepared_root,
                run_dir=run_dir,
                manifest_sha256="VERIFIED_AT_RUN_TIME",
                turn=turn,
                difficulty=difficulty,
                python_bin=executable,
                model=model,
                reasoning_effort=reasoning_effort,
                base_url=base_url,
                api_key_env=api_key_env,
                tool_schema_path=tool_schemas[turn].source_path,
                tool_schema_sha256=tool_schemas[turn].sha256,
                extra_args=extra_args,
                resume=False,
            )
            if stream is not None:
                import shlex

                stream.write(f"[{invocation.label}] {shlex.join(invocation.command)}\n")
        return SuiteSummary("dry-run", run_dir, 0)

    prepared_root = lexical_absolute(prepared_root)
    _, manifest_bytes, manifest_sha256, records = verify_manifest_bundle(prepared_root)
    config = _resolved_config(
        repo_root=repo_root,
        prepared_root=prepared_root,
        run_dir=run_dir,
        model=model,
        python_bin=python_bin,
        python_runtime=python_runtime,
        manifest_sha256=manifest_sha256,
        reasoning_effort=reasoning_effort,
        base_url=base_url,
        api_key_env=api_key_env,
        extra_args=extra_args,
        experiment_config_sha256=experiment_config_sha256,
        experiment_config_relative_path=experiment_config_relative_path,
        evaluator_variant=evaluator_variant,
        output_policy=output_policy,
        tool_schemas=tool_schemas,
    )

    try:
        run_root = (
            admit_directory(run_dir, label="resume run directory")
            if resume
            else create_directory(run_dir, label="run directory")
        )
    except FileAdmissionError as exc:
        raise RunArtifactError(str(exc)) from exc
    with run_root:
        return _run_bound_suite(
            run_root=run_root,
            run_dir=run_dir,
            config=config,
            manifest_bytes=manifest_bytes,
            manifest_sha256=manifest_sha256,
            experiment_config_bytes=experiment_config_bytes,
            records=records,
            tool_schemas=tool_schemas,
            resume=resume,
            repo_root=repo_root,
            prepared_root=prepared_root,
            executable=executable,
            model=model,
            reasoning_effort=reasoning_effort,
            base_url=base_url,
            api_key_env=api_key_env,
            extra_args=extra_args,
            evaluator_variant=evaluator_variant,
            evaluation_runner=evaluation_runner,
            condition_runner=condition_runner,
        )


def _run_bound_suite(
    *,
    run_root: AdmittedDirectory,
    run_dir: Path,
    config: Mapping[str, Any],
    manifest_bytes: bytes,
    manifest_sha256: str,
    experiment_config_bytes: bytes | None,
    records: Mapping[str, Sequence[Mapping[str, Any]]],
    tool_schemas: Mapping[str, ToolSchemaInput],
    resume: bool,
    repo_root: Path,
    prepared_root: Path,
    executable: str,
    model: str,
    reasoning_effort: str | None,
    base_url: str | None,
    api_key_env: str,
    extra_args: Sequence[str],
    evaluator_variant: str | None,
    evaluation_runner: EvaluationRunner | None,
    condition_runner: ConditionRunner,
) -> SuiteSummary:
    conditions = (
        _load_resume(
            run_dir=run_dir,
            run_root=run_root,
            config=config,
            manifest_bytes=manifest_bytes,
            experiment_config_bytes=experiment_config_bytes,
            records=records,
            tool_schemas=tool_schemas,
        )
        if resume
        else _initialize_run(
            run_dir=run_dir,
            run_root=run_root,
            config=config,
            manifest_bytes=manifest_bytes,
            experiment_config_bytes=experiment_config_bytes,
            evaluator_variant=evaluator_variant,
            tool_schemas=tool_schemas,
        )
    )
    _write_status(run_dir, "running", conditions, run_root=run_root)

    try:
        with run_root.open_text(
            "run.log",
            label="run log",
            append=True,
        ) as log:
            for turn, difficulty in CONDITIONS:
                label = f"{turn}.{difficulty}"
                if conditions[label].get("status") in TERMINAL_STATUSES:
                    continue
                condition_dir = run_dir / "predictions" / turn / difficulty
                checkpoint = condition_dir / "checkpoint.jsonl"
                output_path = condition_dir / "predictions.json"
                output_relative = run_root.relative(output_path)
                checkpoint_relative = run_root.relative(checkpoint)
                seal_relative = checkpoint_integrity_path(
                    checkpoint_relative
                )
                if run_root.exists(
                    output_relative,
                    label=f"predictions {label}",
                ):
                    raise RunArtifactError(f"refusing to overwrite predictions: {output_path}")
                checkpoint_exists = run_root.exists(
                    checkpoint_relative,
                    label=f"checkpoint {label}",
                )
                seal_exists = run_root.exists(
                    seal_relative,
                    label=f"checkpoint integrity seal {label}",
                )
                resume_condition = checkpoint_exists or (resume and seal_exists)
                if checkpoint_exists:
                    run_root.admit_file(
                        checkpoint_relative,
                        label=f"checkpoint {label}",
                    )
                if resume and seal_exists:
                    run_root.admit_file(
                        seal_relative,
                        label=f"checkpoint integrity seal {label}",
                    )
                run_root.ensure_directory(
                    run_root.relative(condition_dir),
                    label=f"condition directory {label}",
                )
                invocation = _invocation(
                    repo_root=repo_root,
                    prepared_root=prepared_root,
                    run_dir=run_dir,
                    manifest_sha256=manifest_sha256,
                    turn=turn,
                    difficulty=difficulty,
                    python_bin=executable,
                    model=model,
                    reasoning_effort=reasoning_effort,
                    base_url=base_url,
                    api_key_env=api_key_env,
                    tool_schema_path=run_dir / TOOL_SCHEMA_SNAPSHOTS[turn],
                    tool_schema_sha256=tool_schemas[turn].sha256,
                    extra_args=extra_args,
                    resume=resume_condition,
                    output_root_identity=run_root.identity[:2],
                )
                conditions[label]["status"] = "running"
                conditions[label]["started_at"] = _utc_now()
                _write_status(
                    run_dir,
                    "running",
                    conditions,
                    run_root=run_root,
                )
                log.write(f"[{_utc_now()}] START {label}\n")
                log.flush()
                returncode = condition_runner(invocation, log)
                if returncode != 0:
                    raise RunArtifactError(
                        f"condition runner failed for {label} with exit code {returncode}"
                    )
                counts = _verify_prediction(
                    invocation.output_path,
                    records[label],
                    run_root=run_root,
                )
                conditions[label].update(counts)
                conditions[label]["status"] = _completion_status(**counts)
                conditions[label]["completed_at"] = _utc_now()
                log.write(f"[{_utc_now()}] COMPLETE {label}\n")
                log.flush()
                _write_status(
                    run_dir,
                    "running",
                    conditions,
                    run_root=run_root,
                )
    except BaseException as exc:
        for value in conditions.values():
            if value.get("status") == "running":
                value["status"] = "failed"
        _write_status(
            run_dir,
            "failed",
            conditions,
            run_root=run_root,
            error=str(exc),
        )
        if isinstance(exc, FileAdmissionError):
            raise RunArtifactError(str(exc)) from exc
        raise

    counts = _aggregate_counts(conditions)
    suite_status = _completion_status(**counts)
    _write_status(run_dir, suite_status, conditions, run_root=run_root)
    if evaluator_variant is not None:
        _evaluate_terminal_run(
            run_dir=run_dir,
            run_root=run_root,
            variant=evaluator_variant,
            provider_status=suite_status,
            evaluation_runner=evaluation_runner or evaluate_run,
        )
    return SuiteSummary(
        suite_status,
        run_dir,
        len(CONDITIONS),
        **counts,
    )
