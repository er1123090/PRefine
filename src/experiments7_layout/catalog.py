"""Build and validate deterministic navigation catalogs.

The method, experiment, and source-data catalogs point to audited regular-file
copies in the readable tree. Canonical runtime and paper artifacts stay in the
sealed trees.
"""

from __future__ import annotations

import json
import os
import secrets
import stat
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from .readable_code import (
    MANIFEST_PATH as READABLE_CODE_MANIFEST,
    READABLE_VERSION_DIRECTORIES,
    ReadableCodeError,
    sync_readable_code,
)


class LayoutError(RuntimeError):
    """Raised when the navigation layer is incomplete or unsafe."""


PROTECTED_ROOTS = (
    "_paper",
    "configs/environment",
    "environments",
    "lineage",
    "manifests",
    "paper_outputs",
    "runs",
    "variants",
)

NAVIGATION_FILES: dict[str, tuple[str, ...]] = {
    "data": ("README.md", "datasets_index.json"),
    "methods": ("README.md", "methods_index.json"),
    "experiments": ("README.md", "index.json"),
    "outputs": ("README.md", "paper_outputs_index.json", "runs_index.json"),
    "results": ("README.md", "results_index.json"),
}

NAVIGATION_DIRECTORIES: dict[str, tuple[str, ...]] = {
    "data": ("sources",),
    "methods": (
        "emem",
        "langmem",
        "mem0",
        "our_memory",
        "rag",
        "remem",
        "self_refine",
        "vanilla_llm",
    ),
    "experiments": (*READABLE_VERSION_DIRECTORIES, "variants"),
    "outputs": (),
    "results": (),
}

CATALOG_PATHS = (
    "data/datasets_index.json",
    "methods/methods_index.json",
    "experiments/index.json",
    "outputs/paper_outputs_index.json",
    "outputs/runs_index.json",
    "results/results_index.json",
    "layout_manifest.json",
)

COUNT_CONTRACT = {
    "grounded_result_records": 1696,
    "result_records_total": 1908,
    "unresolved_result_records": 212,
    "variants": 84,
    "verified_raw_file_count": 521,
}

AUDITED_REGISTRY = "configs/environment/variant-registry-audited-v2.json"
AUDITED_PROFILES = "configs/environment/variant-profiles-audited-v2.json"
FINAL_SEAL = "paper_outputs/final/seal.json"
FINAL_MAPPING = "paper_outputs/final/mapping.jsonl"
FINAL_RAW_MANIFEST = "paper_outputs/final/raw-manifest.jsonl"
FINAL_SUMMARY = "paper_outputs/final/summary.json"
FINAL_UNRESOLVED = "paper_outputs/final/unresolved.jsonl"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LayoutError(f"cannot read canonical JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise LayoutError(f"canonical JSON must be an object: {path}")
    return value


def _line_count(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError as exc:
        raise LayoutError(f"cannot read canonical JSONL {path}: {exc}") from exc


def _sorted_unique_strings(values: Iterable[Any], *, field: str) -> list[str]:
    result: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value:
            raise LayoutError(f"{field} must contain non-empty strings")
        result.append(value)
    return sorted(set(result))


def validate_canonical_path(root: Path, relative_path: str) -> Path:
    """Return a contained existing target, rejecting traversal and escapes."""

    if not isinstance(relative_path, str) or not relative_path:
        raise LayoutError("canonical_path must be a non-empty string")
    if "\\" in relative_path or "//" in relative_path or relative_path.startswith("./"):
        raise LayoutError(f"canonical_path must use POSIX separators: {relative_path!r}")
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise LayoutError(f"unsafe canonical_path: {relative_path!r}")
    root_resolved = root.resolve(strict=True)
    candidate = root.joinpath(*pure.parts)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root_resolved)
    except (OSError, ValueError) as exc:
        raise LayoutError(f"canonical_path escapes or is missing: {relative_path!r}") from exc
    return resolved


def _source_origins(variant: Mapping[str, Any]) -> list[str]:
    evidence = variant.get("origin_evidence", [])
    if not isinstance(evidence, list):
        raise LayoutError("variant origin_evidence must be a list")
    return _sorted_unique_strings(
        (item.get("origin_root") for item in evidence if isinstance(item, dict)),
        field="source_origins",
    )


def _canonical_contract(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
    registry = _load_json(root / AUDITED_REGISTRY)
    profiles = _load_json(root / AUDITED_PROFILES)
    seal = _load_json(root / FINAL_SEAL)

    variants = registry.get("variants")
    if not isinstance(variants, list):
        raise LayoutError("audited registry variants must be a list")

    verified_dir = root / "paper_outputs/raw/verified"
    verified_files = sum(
        1
        for path in verified_dir.rglob("*")
        if path.is_file() and not path.is_symlink()
    )
    counts = {
        "grounded_result_records": seal.get("grounded_result_count"),
        "result_records_total": seal.get("inventory_result_count"),
        "unresolved_result_records": seal.get("unresolved_result_count"),
        "variants": registry.get("variant_count"),
        "verified_raw_file_count": seal.get("minimal_raw_union_count"),
    }
    if counts != COUNT_CONTRACT:
        raise LayoutError(f"canonical count contract mismatch: {counts!r}")
    if len(variants) != COUNT_CONTRACT["variants"]:
        raise LayoutError("audited variant list cardinality mismatch")
    if _line_count(root / FINAL_MAPPING) != COUNT_CONTRACT["result_records_total"]:
        raise LayoutError("final mapping cardinality mismatch")
    if _line_count(root / FINAL_UNRESOLVED) != COUNT_CONTRACT["unresolved_result_records"]:
        raise LayoutError("final unresolved cardinality mismatch")
    if _line_count(root / FINAL_RAW_MANIFEST) != COUNT_CONTRACT["verified_raw_file_count"]:
        raise LayoutError("final raw manifest cardinality mismatch")
    if verified_files != COUNT_CONTRACT["verified_raw_file_count"]:
        raise LayoutError(f"verified raw file cardinality mismatch: {verified_files}")
    return registry, profiles, counts


def _experiment_entries(
    root: Path,
    registry: Mapping[str, Any],
    readable_lookup: Mapping[tuple[str, str], str],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    variants = registry.get("variants", [])
    for variant in variants:
        if not isinstance(variant, dict):
            raise LayoutError("audited registry variants must be objects")
        variant_id = variant.get("variant_id")
        if not isinstance(variant_id, str) or not variant_id:
            raise LayoutError("audited variant_id must be a non-empty string")
        evidence = variant.get("origin_evidence")
        if not isinstance(evidence, list) or not evidence:
            raise LayoutError(f"audited variant has no origin evidence: {variant_id}")
        code_paths: list[str] = []
        for item in evidence:
            if not isinstance(item, dict):
                raise LayoutError(f"audited variant evidence is malformed: {variant_id}")
            key = (item.get("origin_root"), item.get("relative_path"))
            target = readable_lookup.get(key)
            if target is None:
                raise LayoutError(f"readable code is missing for {variant_id}: {key!r}")
            validate_canonical_path(root, target)
            code_paths.append(target)
        code_paths = sorted(set(code_paths))
        entry = {
            "affected_outputs": variant.get("affected_outputs", []),
            "canonical_path": code_paths[0],
            "code_paths": code_paths,
            "experiment_id": variant_id,
            "semantic_label": variant_id,
            "source_origins": _source_origins(variant),
            "stage": variant.get("stage"),
        }
        if "distinction" in variant:
            entry["distinction"] = variant["distinction"]
        if "paper_role" in variant:
            entry["paper_role"] = variant["paper_role"]
        entries.append(entry)
    entries.sort(key=lambda item: item["experiment_id"])
    if len(entries) != COUNT_CONTRACT["variants"]:
        raise LayoutError("experiment index must contain all 84 audited variants")
    return entries


def _readable_code_contract(root: Path) -> tuple[dict[str, Any], dict[tuple[str, str], str]]:
    manifest = _load_json(root / READABLE_CODE_MANIFEST)
    entries = manifest.get("entries")
    if not isinstance(entries, list) or manifest.get("entry_count") != len(entries):
        raise LayoutError("readable code manifest entry count is invalid")
    if manifest.get("variant_count") != COUNT_CONTRACT["variants"]:
        raise LayoutError("readable code manifest variant count is invalid")
    lookup: dict[tuple[str, str], str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise LayoutError("readable code manifest entries must be objects")
        origin = entry.get("source_origin")
        relative_path = entry.get("source_relative_path")
        target_path = entry.get("target_path")
        if not all(isinstance(value, str) and value for value in (origin, relative_path, target_path)):
            raise LayoutError("readable code manifest entry is incomplete")
        key = (origin, relative_path)
        if key in lookup or target_path in lookup.values():
            raise LayoutError(f"readable code manifest contains a duplicate mapping: {key!r}")
        validate_canonical_path(root, target_path)
        lookup[key] = target_path
    return manifest, lookup


def _dataset_catalog(root: Path, profiles: Mapping[str, Any]) -> dict[str, Any]:
    base = profiles.get("base")
    if not isinstance(base, dict):
        raise LayoutError("audited profile base must be an object")
    snapshot = base.get("physical_snapshot")
    semantic_label = base.get("semantic_label")
    release_alias = base.get("release_alias_of")
    if not all(isinstance(value, str) and value for value in (snapshot, semantic_label, release_alias)):
        raise LayoutError("audited base identity is incomplete")
    data_root_path = f"{snapshot}/data"
    data_root = validate_canonical_path(root, data_root_path)
    entries: list[dict[str, Any]] = []
    for path in sorted(data_root.iterdir(), key=lambda item: item.name):
        if path.suffix.lower() not in {".csv", ".json", ".tsv"}:
            continue
        if path.is_symlink() or not path.is_file():
            raise LayoutError(f"dataset navigation target must be a regular file: {path}")
        entries.append(
            {
                "canonical_path": f"{data_root_path}/{path.name}",
                "dataset_id": path.name,
                "semantic_label": semantic_label,
                "source_origins": sorted({release_alias, "experiments6"}),
            }
        )
    if not entries:
        raise LayoutError("no dataset files found in the audited base snapshot")
    return {
        "entries": entries,
        "entry_count": len(entries),
        "schema": "experiments7-datasets-index/v1",
    }


def _methods_catalog(experiments: list[dict[str, Any]], registry: Mapping[str, Any]) -> dict[str, Any]:
    by_id = {entry["experiment_id"]: entry for entry in experiments}
    entries: list[dict[str, Any]] = []
    for variant in registry.get("variants", []):
        if not isinstance(variant, dict) or not variant.get("paper_role"):
            continue
        variant_id = variant["variant_id"]
        experiment = by_id[variant_id]
        entries.append(
            {
                "canonical_path": experiment["canonical_path"],
                "code_paths": experiment["code_paths"],
                "method_name": variant["paper_role"],
                "semantic_label": variant_id,
                "source_origins": experiment["source_origins"],
                "stage": variant.get("stage"),
            }
        )
    entries.sort(key=lambda item: (str(item["method_name"]), item["semantic_label"]))
    if not entries:
        raise LayoutError("no paper-method variants found in the audited registry")
    return {
        "entries": entries,
        "entry_count": len(entries),
        "schema": "experiments7-methods-index/v1",
    }


def _paper_output_catalog(raw_origins: list[str]) -> dict[str, Any]:
    return {
        "entries": [
            {
                "canonical_path": "paper_outputs/final",
                "description": "Sealed paper result mapping, summary, raw manifest, and unresolved ledger.",
                "output_id": "paper_final",
                "semantic_label": "paper_final",
                "source_origins": raw_origins,
            },
            {
                "canonical_path": "paper_outputs/raw",
                "description": "Verified raw inference outputs referenced by the paper mapping.",
                "output_id": "paper_raw",
                "semantic_label": "paper_raw_verified",
                "source_origins": raw_origins,
            },
        ],
        "entry_count": 2,
        "schema": "experiments7-paper-outputs-index/v1",
    }


def _run_kind(name: str) -> str:
    if name == "fixtures":
        return "fixtures"
    if name.startswith("test-reference-external-"):
        return "test_reference_external"
    if name.startswith("test-external-"):
        return "test_external"
    if "smoke" in name.lower():
        return "smoke"
    if name.startswith(("g009-", "g010-", "G016_")):
        return "verification"
    return "other"


def _runs_catalog(root: Path) -> dict[str, Any]:
    runs = validate_canonical_path(root, "runs")
    entries: list[dict[str, Any]] = []
    for path in sorted(runs.iterdir(), key=lambda item: item.name):
        if path.is_symlink() or not path.is_dir():
            raise LayoutError(f"top-level run entry must be a real directory: {path}")
        entries.append(
            {
                "canonical_path": f"runs/{path.name}",
                "kind": _run_kind(path.name),
                "run_id": path.name,
                "semantic_label": path.name,
                "source_origins": [],
            }
        )
    return {
        "entries": entries,
        "entry_count": len(entries),
        "schema": "experiments7-runs-index/v1",
    }


def _results_catalog(raw_origins: list[str], counts: Mapping[str, int]) -> dict[str, Any]:
    entries = [
        {
            "canonical_path": FINAL_SUMMARY,
            "description": "Aggregate finalization and audit metrics.",
            "record_count": counts["result_records_total"],
            "result_id": "paper_summary",
            "semantic_label": "paper_results_final",
            "source_origins": raw_origins,
        },
        {
            "canonical_path": FINAL_MAPPING,
            "description": "One record per result reported in the paper.",
            "record_count": counts["result_records_total"],
            "result_id": "paper_result_mapping",
            "semantic_label": "paper_results_final",
            "source_origins": raw_origins,
        },
        {
            "canonical_path": FINAL_UNRESOLVED,
            "description": "Fail-closed ledger for paper values without confirmed raw provenance.",
            "record_count": counts["unresolved_result_records"],
            "result_id": "paper_unresolved",
            "semantic_label": "paper_results_unresolved",
            "source_origins": raw_origins,
        },
    ]
    return {
        "counts": dict(counts),
        "entries": entries,
        "entry_count": len(entries),
        "schema": "experiments7-results-index/v1",
    }


def _raw_origins(root: Path) -> list[str]:
    origins: list[str] = []
    try:
        with (root / FINAL_RAW_MANIFEST).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                origins.append(record["root_id"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise LayoutError(f"cannot derive raw source origins: {exc}") from exc
    return _sorted_unique_strings(origins, field="source_origins")


def build_catalogs(root: Path) -> dict[str, dict[str, Any]]:
    """Build all catalog documents from current canonical artifacts."""

    root = root.resolve(strict=True)
    registry, profiles, counts = _canonical_contract(root)
    readable_manifest, readable_lookup = _readable_code_contract(root)
    experiments = _experiment_entries(root, registry, readable_lookup)
    raw_origins = _raw_origins(root)
    documents: dict[str, dict[str, Any]] = {
        "data/datasets_index.json": _dataset_catalog(root, profiles),
        "experiments/index.json": {
            "entries": experiments,
            "entry_count": len(experiments),
            "schema": "experiments7-experiments-index/v1",
        },
        "methods/methods_index.json": _methods_catalog(experiments, registry),
        "outputs/paper_outputs_index.json": _paper_output_catalog(raw_origins),
        "outputs/runs_index.json": _runs_catalog(root),
        "results/results_index.json": _results_catalog(raw_origins, counts),
    }
    documents["layout_manifest.json"] = {
        "canonical_sources": [
            AUDITED_PROFILES,
            AUDITED_REGISTRY,
            FINAL_SEAL,
            READABLE_CODE_MANIFEST,
        ],
        "catalogs": list(CATALOG_PATHS[:-1]),
        "count_contract": dict(COUNT_CONTRACT),
        "navigation_files": {
            category: list(files) for category, files in sorted(NAVIGATION_FILES.items())
        },
        "navigation_directories": {
            category: list(directories)
            for category, directories in sorted(NAVIGATION_DIRECTORIES.items())
        },
        "policies": {
            "canonical_storage_is_not_replaced": True,
            "actual_code_copy_count": readable_manifest["entry_count"],
            "new_payload_copies": True,
            "new_symlinks": False,
            "outputs": "raw and run artifacts",
            "results": "metrics, aggregates, and evaluation records",
        },
        "protected_roots": list(PROTECTED_ROOTS),
        "schema": "experiments7-layout-manifest/v1",
    }
    return documents


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_READ_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
_TEMP_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC


def _identity(value: os.stat_result) -> tuple[int, int]:
    return value.st_dev, value.st_ino


def _best_effort_close(file_fd: int) -> None:
    try:
        os.close(file_fd)
    except OSError:
        pass


def _real_catalog_root(root: Path) -> tuple[Path, os.stat_result]:
    root_fd: int | None = None
    try:
        before = os.lstat(root)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
            raise LayoutError(f"catalog root is not a real directory: {root}")
        root_fd = os.open(root, _DIRECTORY_FLAGS)
        opened = os.fstat(root_fd)
        canonical = root.resolve(strict=True)
        after = os.lstat(root)
        canonical_stat = os.lstat(canonical)
        root_stats = (before, opened, after, canonical_stat)
        if (
            any(stat.S_ISLNK(value.st_mode) for value in root_stats)
            or any(not stat.S_ISDIR(value.st_mode) for value in root_stats)
            or any(_identity(value) != _identity(opened) for value in root_stats)
        ):
            raise LayoutError(f"catalog root binding changed: {root}")
        return canonical, opened
    except LayoutError:
        raise
    except (OSError, RuntimeError) as exc:
        raise LayoutError(f"catalog root is not a real directory: {root}") from exc
    finally:
        if root_fd is not None:
            _best_effort_close(root_fd)


def _catalog_parts(relative_path: str) -> tuple[str, str]:
    pure = PurePosixPath(relative_path)
    if relative_path == "layout_manifest.json" and pure.parts == (relative_path,):
        return "", relative_path
    if (
        len(pure.parts) == 2
        and pure.parts[0] in NAVIGATION_FILES
        and pure.parts[1] in NAVIGATION_FILES[pure.parts[0]]
    ):
        return pure.parts
    raise LayoutError(f"catalog destination is not declared: {relative_path}")


def _destination_stat(
    parent_fd: int,
    name: str,
    relative_path: str,
    *,
    allow_missing: bool,
) -> os.stat_result | None:
    try:
        value = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        if allow_missing:
            return None
        raise LayoutError(f"missing catalog {relative_path}") from None
    except OSError as exc:
        raise LayoutError(f"cannot inspect catalog destination: {relative_path}") from exc
    if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
        raise LayoutError(f"catalog destination is not a unique regular file: {relative_path}")
    return value


def _close_catalog_parents(parents: Mapping[str, tuple[int, os.stat_result]]) -> None:
    for parent_fd, _ in reversed(tuple(parents.values())):
        _best_effort_close(parent_fd)


def _preflight_catalog_destinations(
    root: Path,
    expected_root: os.stat_result,
    *,
    require_existing: bool,
) -> tuple[
    dict[str, tuple[int, os.stat_result]],
    list[tuple[str, str, str, os.stat_result | None]],
]:
    parents: dict[str, tuple[int, os.stat_result]] = {}
    destinations: list[tuple[str, str, str, os.stat_result | None]] = []
    try:
        root_path_stat = os.lstat(root)
        root_fd: int | None = None
        try:
            root_fd = os.open(root, _DIRECTORY_FLAGS)
            root_fd_stat = os.fstat(root_fd)
        except OSError as exc:
            if root_fd is not None:
                _best_effort_close(root_fd)
            raise LayoutError(f"catalog root is not a real directory: {root}") from exc
        if (
            stat.S_ISLNK(root_path_stat.st_mode)
            or not stat.S_ISDIR(root_path_stat.st_mode)
            or not stat.S_ISDIR(root_fd_stat.st_mode)
            or _identity(root_path_stat) != _identity(expected_root)
            or _identity(root_fd_stat) != _identity(expected_root)
        ):
            _best_effort_close(root_fd)
            raise LayoutError(f"catalog root is not a real directory: {root}")
        parents[""] = (root_fd, root_fd_stat)

        for relative_path in CATALOG_PATHS:
            parent_name, destination_name = _catalog_parts(relative_path)
            if parent_name not in parents:
                parent_fd: int | None = None
                try:
                    path_stat = os.stat(
                        parent_name,
                        dir_fd=root_fd,
                        follow_symlinks=False,
                    )
                    parent_fd = os.open(parent_name, _DIRECTORY_FLAGS, dir_fd=root_fd)
                    fd_stat = os.fstat(parent_fd)
                except OSError as exc:
                    if parent_fd is not None:
                        _best_effort_close(parent_fd)
                    raise LayoutError(f"catalog parent is unsafe: {relative_path}") from exc
                if (
                    stat.S_ISLNK(path_stat.st_mode)
                    or not stat.S_ISDIR(path_stat.st_mode)
                    or not stat.S_ISDIR(fd_stat.st_mode)
                    or _identity(path_stat) != _identity(fd_stat)
                ):
                    _best_effort_close(parent_fd)
                    raise LayoutError(f"catalog parent is unsafe: {relative_path}")
                parents[parent_name] = (parent_fd, fd_stat)

            parent_fd = parents[parent_name][0]
            destination_stat = _destination_stat(
                parent_fd,
                destination_name,
                relative_path,
                allow_missing=not require_existing,
            )
            destinations.append(
                (relative_path, parent_name, destination_name, destination_stat)
            )
    except Exception:
        _close_catalog_parents(parents)
        raise
    return parents, destinations


def _verify_parent_binding(
    root: Path,
    parents: Mapping[str, tuple[int, os.stat_result]],
    parent_name: str,
) -> None:
    root_fd, expected_root = parents[""]
    try:
        root_path_stat = os.lstat(root)
        root_fd_stat = os.fstat(root_fd)
    except OSError as exc:
        raise LayoutError("catalog root binding changed") from exc
    if (
        stat.S_ISLNK(root_path_stat.st_mode)
        or not stat.S_ISDIR(root_path_stat.st_mode)
        or not stat.S_ISDIR(root_fd_stat.st_mode)
        or _identity(root_path_stat) != _identity(expected_root)
        or _identity(root_fd_stat) != _identity(expected_root)
    ):
        raise LayoutError("catalog root binding changed")
    if not parent_name:
        return

    parent_fd, expected_parent = parents[parent_name]
    try:
        parent_path_stat = os.stat(
            parent_name,
            dir_fd=root_fd,
            follow_symlinks=False,
        )
        parent_fd_stat = os.fstat(parent_fd)
    except OSError as exc:
        raise LayoutError(f"catalog parent binding changed: {parent_name}") from exc
    if (
        stat.S_ISLNK(parent_path_stat.st_mode)
        or not stat.S_ISDIR(parent_path_stat.st_mode)
        or not stat.S_ISDIR(parent_fd_stat.st_mode)
        or _identity(parent_path_stat) != _identity(expected_parent)
        or _identity(parent_fd_stat) != _identity(expected_parent)
    ):
        raise LayoutError(f"catalog parent binding changed: {parent_name}")


def _verify_destination_binding(
    parent_fd: int,
    destination_name: str,
    relative_path: str,
    expected: os.stat_result | None,
) -> os.stat_result | None:
    current = _destination_stat(
        parent_fd,
        destination_name,
        relative_path,
        allow_missing=True,
    )
    if expected is None:
        if current is not None:
            raise LayoutError(f"catalog destination binding changed: {relative_path}")
    elif current is None or _identity(current) != _identity(expected):
        raise LayoutError(f"catalog destination binding changed: {relative_path}")
    return current


def _read_catalog_destination(
    root: Path,
    parents: Mapping[str, tuple[int, os.stat_result]],
    destination: tuple[str, str, str, os.stat_result | None],
) -> bytes:
    relative_path, parent_name, destination_name, expected_stat = destination
    parent_fd = parents[parent_name][0]
    _verify_parent_binding(root, parents, parent_name)
    current = _verify_destination_binding(
        parent_fd,
        destination_name,
        relative_path,
        expected_stat,
    )
    if current is None:
        raise LayoutError(f"missing catalog {relative_path}")
    try:
        file_fd = os.open(destination_name, _READ_FLAGS, dir_fd=parent_fd)
    except OSError as exc:
        raise LayoutError(f"cannot safely open catalog {relative_path}") from exc
    try:
        opened_stat = os.fstat(file_fd)
        if (
            not stat.S_ISREG(opened_stat.st_mode)
            or opened_stat.st_nlink != 1
            or _identity(opened_stat) != _identity(current)
        ):
            raise LayoutError(f"catalog destination binding changed: {relative_path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        final_stat = os.fstat(file_fd)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(
            getattr(opened_stat, field) != getattr(final_stat, field)
            for field in stable_fields
        ):
            raise LayoutError(f"catalog changed while reading: {relative_path}")
    finally:
        _best_effort_close(file_fd)
    _verify_parent_binding(root, parents, parent_name)
    _verify_destination_binding(parent_fd, destination_name, relative_path, current)
    return b"".join(chunks)


def _write_catalog_temp(
    parent_fd: int,
    destination_name: str,
    payload: bytes,
) -> tuple[str, os.stat_result]:
    for _ in range(100):
        temp_name = f".{destination_name}.tmp-{secrets.token_hex(16)}"
        try:
            temp_fd = os.open(temp_name, _TEMP_FLAGS, 0o666, dir_fd=parent_fd)
            break
        except FileExistsError:
            continue
        except OSError as exc:
            raise LayoutError(f"cannot create catalog temp for {destination_name}") from exc
    else:
        raise LayoutError(f"cannot allocate catalog temp for {destination_name}")

    keep_temp = False
    try:
        created_stat = os.fstat(temp_fd)
        if not stat.S_ISREG(created_stat.st_mode) or created_stat.st_nlink != 1:
            raise LayoutError(f"catalog temp is not a unique regular file: {destination_name}")
        remaining = memoryview(payload)
        while remaining:
            written = os.write(temp_fd, remaining)
            if written <= 0:
                raise LayoutError(f"short catalog write: {destination_name}")
            remaining = remaining[written:]
        os.fsync(temp_fd)
        final_stat = os.fstat(temp_fd)
        if (
            not stat.S_ISREG(final_stat.st_mode)
            or final_stat.st_nlink != 1
            or _identity(final_stat) != _identity(created_stat)
            or final_stat.st_size != len(payload)
        ):
            raise LayoutError(f"catalog temp changed while writing: {destination_name}")
        keep_temp = True
        return temp_name, final_stat
    finally:
        _best_effort_close(temp_fd)
        if not keep_temp:
            try:
                os.unlink(temp_name, dir_fd=parent_fd)
            except OSError:
                pass


def _verify_temp_binding(
    parent_fd: int,
    temp_name: str,
    expected: os.stat_result,
) -> None:
    temp_stat = _destination_stat(
        parent_fd,
        temp_name,
        temp_name,
        allow_missing=False,
    )
    if temp_stat is None or _identity(temp_stat) != _identity(expected):
        raise LayoutError(f"catalog temp binding changed: {temp_name}")


def _cleanup_catalog_temps(
    parents: Mapping[str, tuple[int, os.stat_result]],
    temps: Mapping[str, tuple[str, str, os.stat_result]],
) -> None:
    for parent_name, temp_name, _ in temps.values():
        try:
            os.unlink(temp_name, dir_fd=parents[parent_name][0])
        except OSError:
            pass


def write_catalogs(
    root: Path,
    *,
    check: bool = False,
    _expected_root: os.stat_result | None = None,
) -> None:
    """Write catalogs, or fail when checked-in bytes are stale."""

    root, root_stat = _real_catalog_root(root)
    if _expected_root is not None and _identity(root_stat) != _identity(_expected_root):
        raise LayoutError("catalog root binding changed")
    parents, destinations = _preflight_catalog_destinations(
        root,
        root_stat,
        require_existing=check,
    )
    temps: dict[str, tuple[str, str, os.stat_result]] = {}
    try:
        _verify_parent_binding(root, parents, "")
        documents = build_catalogs(root)
        writes = {
            relative_path: _json_bytes(documents[relative_path])
            for relative_path in CATALOG_PATHS
        }
        _verify_parent_binding(root, parents, "")
        if check:
            for destination in destinations:
                relative_path = destination[0]
                actual = _read_catalog_destination(root, parents, destination)
                if actual != writes[relative_path]:
                    raise LayoutError(
                        f"stale catalog {relative_path}; run scripts/layout.py build-indexes"
                    )
            return

        for relative_path, parent_name, destination_name, _ in destinations:
            parent_fd = parents[parent_name][0]
            temp_name, temp_stat = _write_catalog_temp(
                parent_fd,
                destination_name,
                writes[relative_path],
            )
            temps[relative_path] = (parent_name, temp_name, temp_stat)

        for relative_path, parent_name, destination_name, expected_stat in destinations:
            parent_fd = parents[parent_name][0]
            _verify_parent_binding(root, parents, parent_name)
            _verify_destination_binding(
                parent_fd,
                destination_name,
                relative_path,
                expected_stat,
            )
            _, temp_name, temp_stat = temps[relative_path]
            _verify_temp_binding(parent_fd, temp_name, temp_stat)

        for relative_path, parent_name, destination_name, expected_stat in destinations:
            parent_fd = parents[parent_name][0]
            _verify_parent_binding(root, parents, parent_name)
            _verify_destination_binding(
                parent_fd,
                destination_name,
                relative_path,
                expected_stat,
            )
            _, temp_name, temp_stat = temps[relative_path]
            _verify_temp_binding(parent_fd, temp_name, temp_stat)
            try:
                os.replace(
                    temp_name,
                    destination_name,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                )
                os.fsync(parent_fd)
            except OSError as exc:
                raise LayoutError(f"cannot durably commit catalog {relative_path}") from exc
            del temps[relative_path]
            committed = _destination_stat(
                parent_fd,
                destination_name,
                relative_path,
                allow_missing=False,
            )
            if committed is None or _identity(committed) != _identity(temp_stat):
                raise LayoutError(f"catalog commit binding changed: {relative_path}")
    finally:
        try:
            _cleanup_catalog_temps(parents, temps)
        finally:
            _close_catalog_parents(parents)


def _validate_navigation_files(root: Path) -> None:
    for category, declared_names in NAVIGATION_FILES.items():
        directory = root / category
        mode = os.lstat(directory).st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise LayoutError(f"navigation root is not a real directory: {category}")
        actual_files: list[str] = []
        actual_directories: list[str] = []
        for path in directory.iterdir():
            value = os.lstat(path)
            if stat.S_ISLNK(value.st_mode):
                raise LayoutError(f"navigation entry is a symlink: {category}/{path.name}")
            if stat.S_ISREG(value.st_mode):
                actual_files.append(path.name)
            elif stat.S_ISDIR(value.st_mode):
                actual_directories.append(path.name)
            else:
                raise LayoutError(f"navigation entry is special: {category}/{path.name}")
        expected_names = sorted(declared_names)
        expected_directories = sorted(NAVIGATION_DIRECTORIES[category])
        if sorted(actual_files) != expected_names or sorted(actual_directories) != expected_directories:
            raise LayoutError(
                f"undeclared or missing navigation entries in {category}: "
                f"expected_files={expected_names!r} actual_files={sorted(actual_files)!r} "
                f"expected_directories={expected_directories!r} "
                f"actual_directories={sorted(actual_directories)!r}"
            )
        for name in declared_names:
            path = directory / name
            mode = os.lstat(path).st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                raise LayoutError(f"navigation file is not regular: {category}/{name}")
    manifest = root / "layout_manifest.json"
    mode = os.lstat(manifest).st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise LayoutError("layout_manifest.json is not a regular file")
    readable_manifest = root / READABLE_CODE_MANIFEST
    mode = os.lstat(readable_manifest).st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise LayoutError(f"{READABLE_CODE_MANIFEST} is not a regular file")


def _validate_entries(root: Path, document: Mapping[str, Any], relative_path: str) -> None:
    entries = document.get("entries")
    if not isinstance(entries, list):
        raise LayoutError(f"catalog entries must be a list: {relative_path}")
    if document.get("entry_count") != len(entries):
        raise LayoutError(f"catalog entry_count mismatch: {relative_path}")
    for entry in entries:
        if not isinstance(entry, dict):
            raise LayoutError(f"catalog entry must be an object: {relative_path}")
        validate_canonical_path(root, entry.get("canonical_path"))
        origins = entry.get("source_origins")
        if not isinstance(origins, list) or origins != sorted(set(origins)):
            raise LayoutError(f"source_origins must be sorted and unique: {relative_path}")
        label = entry.get("semantic_label")
        if not isinstance(label, str) or not label:
            raise LayoutError(f"semantic_label must be preserved: {relative_path}")
        code_paths = entry.get("code_paths")
        if code_paths is not None:
            if (
                not isinstance(code_paths, list)
                or code_paths != sorted(set(code_paths))
                or not code_paths
            ):
                raise LayoutError(f"code_paths must be sorted and unique: {relative_path}")
            for code_path in code_paths:
                validate_canonical_path(root, code_path)


def validate_layout(root: Path) -> dict[str, Any]:
    """Fail closed unless checked-in navigation exactly matches canonical state."""

    root, root_stat = _real_catalog_root(root)
    try:
        readable_report = sync_readable_code(root, check=True)
    except ReadableCodeError as exc:
        raise LayoutError(f"readable code validation failed: {exc}") from exc
    write_catalogs(root, check=True, _expected_root=root_stat)
    validated_root, validated_stat = _real_catalog_root(root)
    if validated_root != root or _identity(validated_stat) != _identity(root_stat):
        raise LayoutError("catalog root binding changed")
    _validate_navigation_files(root)
    for relative_path in CATALOG_PATHS[:-1]:
        document = _load_json(root / relative_path)
        _validate_entries(root, document, relative_path)

    paper_paths = {
        entry["canonical_path"]
        for entry in _load_json(root / "outputs/paper_outputs_index.json")["entries"]
    }
    if paper_paths != {"paper_outputs/final", "paper_outputs/raw"}:
        raise LayoutError("public paper navigation must expose only final and raw")

    final_root, final_stat = _real_catalog_root(root)
    if final_root != root or _identity(final_stat) != _identity(root_stat):
        raise LayoutError("catalog root binding changed")

    return {
        "catalog_count": len(CATALOG_PATHS) - 1,
        "counts": dict(COUNT_CONTRACT),
        "navigation_roots": sorted(NAVIGATION_FILES),
        "readable_code_files": readable_report["entry_count"],
        "schema": "experiments7-layout-validation/v1",
        "status": "ok",
    }
