"""Filesystem pipeline for materializing the canonical mix600-v1 dataset."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Any, Mapping, Sequence

from .build_instances import build_all_scenarios, canonical_json
from .validation import build_validation_report
from exp7.provenance.admission import (
    AdmittedFile,
    FileAdmissionError,
    admit_regular_file,
    lexical_absolute,
)


class DatasetPreparationError(RuntimeError):
    """Raised when inputs or output boundaries fail closed."""


@dataclass(frozen=True)
class _AdmittedJson:
    file: AdmittedFile
    value: Any

    @property
    def path(self) -> Path:
        return self.file.path


def _admit_file(path: Path, *, label: str) -> AdmittedFile:
    try:
        return admit_regular_file(path, label=label)
    except FileAdmissionError as exc:
        raise DatasetPreparationError(str(exc)) from exc


def sha256_file(path: Path) -> str:
    return _admit_file(path, label="file").sha256


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _decode_json(file: AdmittedFile, *, label: str) -> Any:
    try:
        return json.loads(file.payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatasetPreparationError(
            f"cannot read {label} JSON {file.path}: {exc}"
        ) from exc


def _admit_json(path: Path, *, label: str) -> _AdmittedJson:
    file = _admit_file(path, label=label)
    return _AdmittedJson(file=file, value=_decode_json(file, label=label))


def _load_json(path: Path, *, label: str) -> Any:
    return _admit_json(path, label=label).value


def _resolve(root: Path, value: str | Path) -> Path:
    return lexical_absolute(Path(value), root=root)
def _output_path(root: Path, value: str | Path) -> Path:
    """Make an absolute path without resolving away hostile symlink components."""

    path = Path(value)
    return Path(os.path.abspath(path if path.is_absolute() else root / path))


def _display_path(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _admit_verified_json(
    root: Path, spec: Mapping[str, Any], *, label: str
) -> _AdmittedJson:
    admitted = _admit_json(_resolve(root, str(spec["path"])), label=label)
    actual_bytes = len(admitted.file.payload)
    actual_sha256 = admitted.file.sha256
    if actual_bytes != spec["expected_bytes"]:
        raise DatasetPreparationError(
            f"{label} byte-size mismatch: expected {spec['expected_bytes']}, got {actual_bytes}"
        )
    if actual_sha256 != spec["expected_sha256"]:
        raise DatasetPreparationError(
            f"{label} SHA-256 mismatch: expected {spec['expected_sha256']}, got {actual_sha256}"
        )
    return admitted


def _verified_input(root: Path, spec: Mapping[str, Any], *, label: str) -> Path:
    return _admit_verified_json(root, spec, label=label).path


def _admitted_bundle_sha256(root: Path, files: Sequence[AdmittedFile]) -> str:
    digest = hashlib.sha256()
    for file in sorted(files, key=lambda item: _display_path(root, item.path)):
        label = _display_path(root, file.path).encode("utf-8")
        digest.update(len(label).to_bytes(8, "big"))
        digest.update(label)
        digest.update(len(file.payload).to_bytes(8, "big"))
        digest.update(file.payload)
    return digest.hexdigest()


def _bundle_sha256(root: Path, paths: Sequence[Path]) -> str:
    files = [_admit_file(path, label="bundle member") for path in paths]
    return _admitted_bundle_sha256(root, files)


def _admitted_manifest(root: Path, file: AdmittedFile) -> dict[str, Any]:
    return {
        "bytes": len(file.payload),
        "path": _display_path(root, file.path),
        "sha256": file.sha256,
    }


def _input_manifest(root: Path, path: Path) -> dict[str, Any]:
    return _admitted_manifest(root, _admit_file(path, label="manifest input"))


def _validate_new_output_target(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise DatasetPreparationError(
            f"output already exists; pass --replace-existing for a verified replacement: {path}"
        )
    current = path.parent
    while not current.exists():
        if current.is_symlink():
            raise DatasetPreparationError(f"output parent is a symlink: {current}")
        if current == current.parent:
            break
        current = current.parent
    try:
        value = os.lstat(current)
    except OSError as exc:
        raise DatasetPreparationError(f"cannot inspect output parent {current}: {exc}") from exc
    if not stat.S_ISDIR(value.st_mode) or stat.S_ISLNK(value.st_mode):
        raise DatasetPreparationError(f"output parent is not a real directory: {current}")


def _validate_replace_target(path: Path, dataset_id: str) -> None:
    try:
        value = os.lstat(path)
    except OSError as exc:
        raise DatasetPreparationError(f"cannot inspect replacement target {path}: {exc}") from exc
    if not stat.S_ISDIR(value.st_mode) or stat.S_ISLNK(value.st_mode):
        raise DatasetPreparationError(f"replacement target must be a real directory: {path}")
    manifest_path = path / "manifest.json"
    try:
        manifest_value = os.lstat(manifest_path)
    except OSError as exc:
        raise DatasetPreparationError(
            f"replacement target lacks a readable manifest: {manifest_path}"
        ) from exc
    if not stat.S_ISREG(manifest_value.st_mode) or stat.S_ISLNK(manifest_value.st_mode):
        raise DatasetPreparationError(
            f"replacement manifest must be a regular file: {manifest_path}"
        )
    manifest = _load_json(manifest_path, label="existing manifest")
    if manifest.get("dataset_id") != dataset_id:
        raise DatasetPreparationError(
            f"refusing to replace a different dataset: {manifest.get('dataset_id')!r}"
        )


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for record in records:
            handle.write(canonical_json(record).encode("utf-8"))
            handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _commit_directory(
    temporary: Path,
    output: Path,
    *,
    dataset_id: str,
    replace_existing: bool,
) -> None:
    if not output.exists() and not output.is_symlink():
        temporary.rename(output)
        return
    if not replace_existing:
        raise DatasetPreparationError(f"output appeared during preparation: {output}")
    _validate_replace_target(output, dataset_id)
    backup = output.with_name(f".{output.name}.backup-{os.getpid()}")
    if backup.exists() or backup.is_symlink():
        raise DatasetPreparationError(f"replacement backup path already exists: {backup}")
    output.rename(backup)
    try:
        temporary.rename(output)
    except BaseException:
        backup.rename(output)
        raise
    shutil.rmtree(backup)


def prepare_dataset(
    *,
    repo_root: Path,
    config_path: Path,
    mix600_path: Path | None = None,
    output_dir: Path | None = None,
    replace_existing: bool = False,
) -> dict[str, Any]:
    """Materialize all six conditions and return the written manifest."""

    repo_root = repo_root.resolve(strict=True)
    config_path = _resolve(repo_root, config_path)
    dataset_file = _admit_json(config_path, label="dataset config")
    dataset_config = dataset_file.value
    dataset_id = str(dataset_config["dataset_id"])

    config_files = {
        "dataset": dataset_file,
        "queries": _admit_json(
            _resolve(repo_root, dataset_config["queries_config"]),
            label="query config",
        ),
        "preferences": _admit_json(
            _resolve(repo_root, dataset_config["preferences_config"]),
            label="preference config",
        ),
        "schemas": _admit_json(
            _resolve(repo_root, dataset_config["schemas_config"]),
            label="schema config",
        ),
    }
    query_config = config_files["queries"].value
    preference_config = config_files["preferences"].value
    schema_config = config_files["schemas"].value

    source_spec = dict(dataset_config["source"])
    source_spec["path"] = str(mix600_path or source_spec.pop("default_path"))
    input_files = {
        "mix600": _admit_verified_json(repo_root, source_spec, label="mix600"),
        "query_singleturn": _admit_verified_json(
            repo_root, query_config["inputs"]["singleturn"], label="query_singleturn"
        ),
        "query_multiturn": _admit_verified_json(
            repo_root, query_config["inputs"]["multiturn"], label="query_multiturn"
        ),
        "pref_list": _admit_verified_json(
            repo_root, preference_config["inputs"]["pref_list"], label="pref_list"
        ),
        "pref_group": _admit_verified_json(
            repo_root, preference_config["inputs"]["pref_group"], label="pref_group"
        ),
        "schema_singleturn": _admit_verified_json(
            repo_root, schema_config["inputs"]["singleturn"], label="schema_singleturn"
        ),
        "schema_multiturn": _admit_verified_json(
            repo_root, schema_config["inputs"]["multiturn"], label="schema_multiturn"
        ),
    }

    rows = input_files["mix600"].value
    query_singleturn = input_files["query_singleturn"].value
    query_multiturn = input_files["query_multiturn"].value
    pref_list = input_files["pref_list"].value
    pref_groups = input_files["pref_group"].value
    schemas = {
        "singleturn": input_files["schema_singleturn"].value,
        "multiturn": input_files["schema_multiturn"].value,
    }
    if not isinstance(rows, list):
        raise DatasetPreparationError("mix600 JSON must be a list")

    scenarios = build_all_scenarios(
        rows,
        dataset_id=dataset_id,
        turns=dataset_config["turns"],
        difficulties=dataset_config["difficulties"],
        query_singleturn=query_singleturn,
        query_multiturn_raw=query_multiturn,
        pref_list=pref_list,
        pref_groups=pref_groups,
    )
    validation = build_validation_report(
        rows=rows,
        scenarios=scenarios,
        pref_groups=pref_groups,
        query_multiturn=query_multiturn,
        schemas=schemas,
        expected=dataset_config["expected"],
    )

    configured_output = dataset_config["output"]["default_dir"]
    output = _output_path(repo_root, output_dir or configured_output)
    if output.exists() or output.is_symlink():
        if not replace_existing:
            _validate_new_output_target(output)
        _validate_replace_target(output, dataset_id)
    else:
        _validate_new_output_target(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent)
    )
    try:
        output_files = {}
        for scenario, records in scenarios.items():
            turn, difficulty = scenario.split(".", 1)
            relative_path = f"{turn}/{difficulty}.jsonl"
            path = temporary / relative_path
            _write_jsonl(path, records)
            output_files[scenario] = {
                "bytes": path.stat().st_size,
                "count": len(records),
                "path": relative_path,
                "sha256": sha256_file(path),
            }

        validation_path = temporary / "validation_report.json"
        validation_path.write_bytes(_json_bytes(validation))
        with validation_path.open("rb") as handle:
            os.fsync(handle.fileno())

        builder_files = [
            _admit_file(path, label="builder source")
            for path in sorted((repo_root / "src/exp7/datasets").glob("*.py"))
            if path.is_file() and not path.is_symlink()
        ]
        manifest = {
            "builder": {
                "files": [
                    _display_path(repo_root, file.path) for file in builder_files
                ],
                "sha256": _admitted_bundle_sha256(repo_root, builder_files),
            },
            "config": {
                "files": {
                    name: _admitted_manifest(repo_root, admitted.file)
                    for name, admitted in sorted(config_files.items())
                },
                "sha256": _admitted_bundle_sha256(
                    repo_root, [item.file for item in config_files.values()]
                ),
            },
            "dataset_id": dataset_id,
            "format_version": 1,
            "inputs": {
                name: _admitted_manifest(repo_root, admitted.file)
                for name, admitted in sorted(input_files.items())
            },
            "outputs": output_files,
            "summary": {
                "counts": {
                    scenario: len(records) for scenario, records in scenarios.items()
                },
                "total_instances": sum(len(records) for records in scenarios.values()),
            },
            "validation_report": {
                "bytes": validation_path.stat().st_size,
                "path": "validation_report.json",
                "sha256": sha256_file(validation_path),
            },
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_bytes(_json_bytes(manifest))
        with manifest_path.open("rb") as handle:
            os.fsync(handle.fileno())
        _commit_directory(
            temporary,
            output,
            dataset_id=dataset_id,
            replace_existing=replace_existing,
        )
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return manifest
