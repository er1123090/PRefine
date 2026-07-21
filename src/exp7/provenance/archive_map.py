"""Deterministic, non-destructive archive planning for experiments7."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
from typing import Any


SCHEMA = "experiments7-archive-map/v1"
DEFAULT_OUTPUT = "archive/archive-map.json"
DEFAULT_MARKDOWN = "archive/archive-map.md"


class ArchiveMapError(RuntimeError):
    pass


def _record(
    path: str,
    disposition: str,
    reason: str,
    evidence: tuple[tuple[str, str, str], ...],
    recovery: tuple[str, ...],
    prerequisites: tuple[str, ...],
    *,
    warning: str | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "path": path,
        "disposition": disposition,
        "reason": reason,
        "evidence_specs": evidence,
        "recovery_pointers": recovery,
        "cutover_prerequisites": prerequisites,
        "externalization_allowed": False,
    }
    if warning:
        value["warning"] = warning
    return value


READABLE = (
    "readable_code_manifest.json",
    "sha256_manifest",
    "Covers 443 payload copies: data/sources=41, methods=305, experiments=97.",
)
FINAL_SEAL = (
    "paper_outputs/final/seal.json",
    "sealed_summary",
    "Authenticates the admitted final set; not an off-host backup.",
)
RAW_MANIFEST = (
    "paper_outputs/final/raw-manifest.jsonl",
    "payload_sha256_manifest",
    "Covers 521 payload files and 7,686,987,770 bytes; README/schema are separate.",
)


TARGETS = (
    _record("data/sources", "archive_after_cutover", "Historical inputs are superseded by prepared datasets.", (READABLE,), ("readable_code_manifest.json", "manifests/copies.jsonl"), ("Prepared parity remains green.", "Active code no longer resolves historical copies.")),
    _record("methods", "archive_after_cutover", "Historical method copies become provenance after method parity.", (READABLE,), ("readable_code_manifest.json", "lineage/code.jsonl"), ("All retained methods consume shared IDs/GT.", "Method parity tests pass.")),
    _record("experiments", "archive_after_cutover", "Historical experiment/evaluation copies remain provenance after canonical cutover.", (READABLE, ("configs/environment/variant-registry-audited-v2.json", "audited_variant_registry", "Defines registered variants; not a payload backup.")), ("readable_code_manifest.json", "configs/environment/variant-registry-audited-v2.json"), ("Canonical configs cover retained variants.", "Evaluation parity passes.")),
    _record("environments", "keep_active_audit", "Sealed environments remain canonical runtime and rollback evidence.", (("configs/environment/catalog.json", "environment_catalog", "Points to environment manifests and seals."),), ("configs/environment/catalog.json", "docs/runnable-environment.md"), ("Replacement runtime is sealed.", "Rollback is demonstrated.")),
    _record("lineage", "keep_active_audit", "Lineage explains retained historical behavior.", (("lineage/code.jsonl", "snapshot_sha256", "Code lineage only."), ("lineage/config.jsonl", "snapshot_sha256", "Config lineage only.")), ("lineage/code.jsonl", "lineage/config.jsonl", "manifests/source-pre.jsonl"), ("Every retained lineage ID resolves.",)),
    _record("manifests", "keep_active_audit", "Seals and source manifests are the recovery basis.", (("manifests/cp0-seal.json", "seal_pointer", "Binds the CP0 set."),), ("manifests/cp0-seal.json", "manifests/source-pre.jsonl", "manifests/copies.jsonl"), ("Recovery pointers resolve from a verified copy.",)),
    _record("variants", "keep_active_audit", "Variant metadata preserves semantic distinctions.", (("configs/environment/variant-registry-audited-v2.json", "audited_variant_registry", "Variant meaning only."), ("configs/environment/variant-profiles-audited-v2.json", "audited_profile_registry", "Profile composition only.")), ("configs/environment/variant-registry-audited-v2.json", "configs/environment/variant-profiles-audited-v2.json"), ("Canonical configs preserve retained IDs.",)),
    _record("paper_outputs/final", "keep_active_audit", "The admitted final set anchors paper claims.", (FINAL_SEAL,), ("paper_outputs/final/seal.json", "paper_outputs/final/mapping.jsonl", "paper_outputs/final/raw-manifest.jsonl"), ("Replacement publication validates against the seal.",)),
    _record("paper_outputs/raw", "externalize_after_verified_backup", "Raw payload can leave only after independent recovery proof.", (RAW_MANIFEST, FINAL_SEAL), ("paper_outputs/final/raw-manifest.jsonl", "paper_outputs/final/seal.json", "paper_outputs/raw/README.md", "paper_outputs/raw/schema.json"), ("Record off-host immutable object versions.", "Verify all 521 hashes in a restore transcript.", "Preserve the two metadata files."), warning="No off-host backup, object version, or restore transcript is recorded; externalization is forbidden."),
    _record("paper_outputs/admission", "keep_active_audit", "Admission evidence explains the final set.", (("paper_outputs/admission/precopy/seal.json", "admission_seal", "Covers the precopy bundle."),), ("paper_outputs/admission/precopy/seal.json",), ("Equivalent admission evidence is retained.")),
    _record("paper_outputs/inventory", "keep_active_audit", "Inventory diagnostics explain artifact interpretation.", (("paper_outputs/inventory/task-evidence.json", "snapshot_sha256", "Task evidence only."),), ("paper_outputs/inventory/task-evidence.json",), ("Equivalent inventory evidence is sealed.")),
    _record("paper_outputs/provenance", "keep_active_audit", "Graph evidence preserves paper claim origins.", (("paper_outputs/provenance/README.md", "coverage_description_sha256", "Description, not a full payload seal."),), ("paper_outputs/provenance/nodes.jsonl", "paper_outputs/provenance/edges.jsonl"), ("Final mappings resolve through provenance.")),
    _record("paper_outputs/strict-runs", "keep_active_audit", "Frozen validators preserve fail-closed publication evidence.", (("paper_outputs/strict-runs/exp7-strict-v6-20260718T032509Z-11942ff2120f70e5fd0d18a08d70569d/checkpoints/cp0.json", "strict_run_checkpoint", "One sealed checkpoint."),), ("paper_outputs/final/seal.json",), ("The final seal identifies authoritative runs.")),
    _record("outputs", "generated_rebuildable", "Navigation indexes are deterministic and rebuildable.", (("outputs/runs_index.json", "navigation_index_sha256", "Navigation only; not payload backup."),), ("outputs/runs_index.json",), ("Catalog regeneration passes.",)),
    _record("results", "generated_rebuildable", "The result index is rebuildable from final records.", (("results/results_index.json", "navigation_index_sha256", "Navigation only."),), ("paper_outputs/final/mapping.jsonl", "paper_outputs/final/unresolved.jsonl"), ("Index regeneration passes.",)),
    _record("runs", "split_before_cleanup", "Runs mix durable evidence, smoke outputs, and fixtures.", (("outputs/runs_index.json", "navigation_index_sha256", "Does not identify disposable runs."),), ("outputs/runs_index.json", "runs/fixtures"), ("Classify durable runs and smoke outputs.", "Move fixtures before cleanup."), warning="Do not bulk-delete runs; preserve fixtures first."),
)


def _relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not value or "\\" in value or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ArchiveMapError(f"unsafe repository-relative path: {value!r}")
    return path


def _bound(root: Path, relative: str, *, directory: bool) -> Path:
    path = root.joinpath(*_relative(relative).parts)
    try:
        info = path.lstat()
    except OSError as exc:
        raise ArchiveMapError(f"required path is missing: {relative}") from exc
    if stat.S_ISLNK(info.st_mode):
        raise ArchiveMapError(f"archive root/evidence may not be a symlink: {relative}")
    try:
        path.resolve(strict=True).relative_to(root)
    except (OSError, ValueError) as exc:
        raise ArchiveMapError(f"path escapes repository root: {relative}") from exc
    if directory != stat.S_ISDIR(info.st_mode) or (not directory and not stat.S_ISREG(info.st_mode)):
        raise ArchiveMapError(f"wrong path type: {relative}")
    return path


def _validate_symlink(root: Path, path: Path) -> None:
    try:
        path.resolve(strict=True).relative_to(root)
    except (OSError, ValueError) as exc:
        raise ArchiveMapError(
            f"symlink escapes repository: {path.relative_to(root)}"
        ) from exc


def inventory_tree(root: Path, relative: str) -> dict[str, int]:
    root = root.resolve(strict=True)
    target = _bound(root, relative, directory=True)
    files = total = links = 0
    for directory, directories, names in os.walk(target, followlinks=False):
        base = Path(directory)
        for name in tuple(directories):
            path = base / name
            if path.is_symlink():
                _validate_symlink(root, path)
                directories.remove(name)
                links += 1
        for name in names:
            path = base / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                _validate_symlink(root, path)
                links += 1
            elif stat.S_ISREG(info.st_mode):
                files += 1
                total += info.st_size
            else:
                raise ArchiveMapError(f"unsupported entry: {path.relative_to(root)}")
    return {"bytes": total, "file_count": files, "symlink_count": links}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_archive_map(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    records = []
    for spec in TARGETS:
        row = {key: value for key, value in spec.items() if key != "evidence_specs"}
        row.update(inventory_tree(root, spec["path"]))
        row["evidence"] = []
        for relative, coverage_type, limits in spec["evidence_specs"]:
            path = _bound(root, relative, directory=False)
            row["evidence"].append({"path": relative, "sha256": _sha256(path), "coverage": {"type": coverage_type, "limits": limits}})
        if spec["path"] == "paper_outputs/raw":
            row["payload_coverage"] = {"file_count": 521, "bytes": 7_686_987_770, "metadata_file_count": 2, "metadata_bytes": 9_922}
        records.append(row)
    return {"schema": SCHEMA, "planned_only": True, "no_mutations": True, "inventory_method": "lstat totals; safe symlinks counted and not followed; payload content not rehashed", "record_count": len(records), "records": records}


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def render_markdown(value: dict[str, Any]) -> bytes:
    lines = ["# Non-destructive archive map", "", "Planning index only: it authorizes no move, deletion, or externalization.", "", "| Current path | Files | Bytes | Symlinks | Planned disposition |", "| --- | ---: | ---: | ---: | --- |"]
    for row in value["records"]:
        lines.append(f"| `{row['path']}` | {row['file_count']:,} | {row['bytes']:,} | {row['symlink_count']:,} | `{row['disposition']}` |")
    lines += ["", "`paper_outputs/raw` remains blocked from externalization until an off-host immutable backup and restore transcript exist.", "", "```bash", "python -B scripts/archive_map.py", "python -B scripts/archive_map.py --check", "```", ""]
    return "\n".join(lines).encode()


def _output(root: Path, relative: str) -> Path:
    path = root.joinpath(*_relative(relative).parts)
    try: path.parent.resolve(strict=True).relative_to(root)
    except (OSError, ValueError) as exc: raise ArchiveMapError(f"output escapes repository: {relative}") from exc
    if path.is_symlink(): raise ArchiveMapError(f"output may not be a symlink: {relative}")
    return path


def _write(path: Path, payload: bytes) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_archive_map(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True); value = build_archive_map(root)
    _write(_output(root, DEFAULT_OUTPUT), json_bytes(value)); _write(_output(root, DEFAULT_MARKDOWN), render_markdown(value))
    return value


def check_archive_map(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True); value = build_archive_map(root); stale = []
    for relative, expected in ((DEFAULT_OUTPUT, json_bytes(value)), (DEFAULT_MARKDOWN, render_markdown(value))):
        path = _output(root, relative)
        if not path.is_file() or path.read_bytes() != expected: stale.append(relative)
    if stale: raise ArchiveMapError("archive map is missing or stale: " + ", ".join(stale))
    return value


def target_paths() -> tuple[str, ...]:
    return tuple(spec["path"] for spec in TARGETS)
