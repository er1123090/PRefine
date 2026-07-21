"""Shared, target-blind contracts for the post-runtime NVML recovery."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
_NESTED_DEVELOPMENT_SOURCE = ROOT / "source_run"
SOURCE_ROOT = (
    _NESTED_DEVELOPMENT_SOURCE
    if _NESTED_DEVELOPMENT_SOURCE.is_dir()
    else ROOT.parents[1]
)
DEFAULT_AMENDMENT = ROOT / "RECOVERY2_AMENDMENT.json"
SOURCE_INVENTORY = ROOT / "SOURCE_RUN_INVENTORY.json"

GOLD_RELATIVE_PATH = "evaluator_vault/gold.jsonl"
SEALED_MANIFEST_RELATIVE_PATH = "evaluator_vault/sealed_manifest.json"
IGNORED_PARTS = {"__pycache__", ".pytest_cache"}
IGNORED_SUFFIXES = {".pyc", ".pyo"}

RECOVERY_OUTPUTS = {
    "attempt": ROOT / "artifacts/recovery2.attempt.json",
    "pre_gold": ROOT / "artifacts/recovery2.pre_gold.json",
    "gold_open": ROOT / "artifacts/recovery2.gold_open.json",
    "summary": ROOT / "reports/summary.json",
    "evidence": ROOT / "reports/final_evaluation.evidence.json",
    "result": ROOT / "artifacts/recovery2.result.json",
    "failure": ROOT / "artifacts/recovery2.failure.json",
    "critic": ROOT / "artifacts/recovery2.critic.json",
}

BOUND_SCRIPTS = (
    "recovery2_common.py",
    "recovery2_runner.py",
    "recovery2_critic.py",
    "seal_recovery2.py",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json_once(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        payload = (canonical_json(value) + "\n").encode("utf-8")
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(fd)


def _ignored(relative: Path) -> bool:
    return bool(set(relative.parts) & IGNORED_PARTS) or relative.suffix in IGNORED_SUFFIXES


def build_source_inventory(source_root: Path = SOURCE_ROOT) -> dict[str, Any]:
    """Hash every stable source-run file except the still-sealed gold bytes."""

    source_root = source_root.resolve()
    sealed_manifest = load_json(source_root / SEALED_MANIFEST_RELATIVE_PATH)
    declared_gold_sha256 = sealed_manifest.get("gold_sha256")
    if not isinstance(declared_gold_sha256, str) or len(declared_gold_sha256) != 64:
        raise ValueError("parent sealed manifest lacks a valid declared gold hash")

    rows: list[dict[str, Any]] = []
    gold_seen = False
    try:
        protocol_relative = ROOT.relative_to(source_root)
    except ValueError:
        protocol_relative = None
    for path in sorted(source_root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(source_root)
        if protocol_relative is not None and (
            relative == protocol_relative or protocol_relative in relative.parents
        ):
            continue
        if _ignored(relative) or path.is_dir():
            continue
        relative_text = relative.as_posix()
        metadata = path.lstat()
        if relative_text == GOLD_RELATIVE_PATH:
            if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
                raise ValueError("sealed gold path must be a non-symlink regular file")
            rows.append(
                {
                    "path": relative_text,
                    "kind": "sealed_target_declared_only_not_opened",
                    "size": int(metadata.st_size),
                    "declared_sha256": declared_gold_sha256,
                }
            )
            gold_seen = True
            continue
        if path.is_symlink():
            rows.append(
                {"path": relative_text, "kind": "symlink", "target": os.readlink(path)}
            )
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"unsupported source-run filesystem node: {relative_text}")
        rows.append(
            {
                "path": relative_text,
                "kind": "regular",
                "size": int(metadata.st_size),
                "sha256": sha256_file(path),
            }
        )
    if not gold_seen:
        raise ValueError("sealed gold path is missing from the source run")
    return {
        "schema_version": 1,
        "kind": "target_blind_source_run_inventory",
        "source_root_name": source_root.name,
        "gold_content_opened": False,
        "gold_binding_authority": SEALED_MANIFEST_RELATIVE_PATH,
        "rows": rows,
        "rows_sha256": sha256_bytes(canonical_json(rows).encode("utf-8")),
    }


def validate_source_inventory(amendment: dict[str, Any]) -> dict[str, Any]:
    if amendment.get("source_inventory_sha256") != sha256_file(SOURCE_INVENTORY):
        raise ValueError("source inventory file hash differs from recovery amendment")
    recorded = load_json(SOURCE_INVENTORY)
    current = build_source_inventory(SOURCE_ROOT)
    if recorded != current:
        raise ValueError("standalone source_run bytes or filesystem identities changed")
    if current.get("gold_content_opened") is not False:
        raise ValueError("target-blind source inventory contract changed")
    return current


def validate_amendment(path: str | Path = DEFAULT_AMENDMENT) -> tuple[dict[str, Any], str]:
    amendment_path = Path(path).resolve()
    amendment = load_json(amendment_path)
    if amendment.get("kind") != "post_runtime_nvml_recovery_amendment_v1":
        raise ValueError("wrong recovery amendment kind")
    if amendment.get("schema_version") != 1:
        raise ValueError("unsupported recovery amendment schema")
    if amendment.get("gold_rows_read_before_recovery") is not False:
        raise ValueError("recovery amendment does not preserve target blindness")
    if amendment.get("parent_performance_status") != "NOT_RUN":
        raise ValueError("parent attempt is not an evaluation-free runtime failure")
    scripts = amendment.get("bound_scripts")
    if not isinstance(scripts, dict) or set(scripts) != set(BOUND_SCRIPTS):
        raise ValueError("recovery amendment script inventory is incomplete")
    for name in BOUND_SCRIPTS:
        if scripts[name] != sha256_file(ROOT / name):
            raise ValueError(f"bound recovery script changed: {name}")
    validate_source_inventory(amendment)
    return amendment, sha256_file(amendment_path)


def stage_record(
    *,
    stage: str,
    sequence: int,
    payload: dict[str, Any],
    amendment_sha256: str,
    previous: dict[str, Any] | None,
) -> dict[str, Any]:
    record = {
        "schema_version": 1,
        "kind": "post_runtime_nvml_recovery_stage",
        "stage": stage,
        "sequence": sequence,
        "previous_stage": None if previous is None else previous["stage"],
        "previous_record_sha256": None if previous is None else previous["record_sha256"],
        "amendment_sha256": amendment_sha256,
        "payload": payload,
    }
    record["record_sha256"] = sha256_bytes(canonical_json(record).encode("utf-8"))
    return record


def validate_stage_record(
    record: dict[str, Any],
    *,
    stage: str,
    sequence: int,
    amendment_sha256: str,
    previous: dict[str, Any] | None,
) -> None:
    claimed = record.get("record_sha256")
    if not isinstance(claimed, str):
        raise ValueError(f"{stage} receipt lacks a record hash")
    unhashed = dict(record)
    del unhashed["record_sha256"]
    if sha256_bytes(canonical_json(unhashed).encode("utf-8")) != claimed:
        raise ValueError(f"{stage} receipt hash mismatch")
    if (
        record.get("kind") != "post_runtime_nvml_recovery_stage"
        or record.get("stage") != stage
        or record.get("sequence") != sequence
        or record.get("amendment_sha256") != amendment_sha256
        or record.get("previous_stage") != (None if previous is None else previous.get("stage"))
        or record.get("previous_record_sha256")
        != (None if previous is None else previous.get("record_sha256"))
    ):
        raise ValueError(f"{stage} receipt chain contract mismatch")


def assert_recovery_outputs_absent() -> None:
    existing = [str(path.relative_to(ROOT)) for path in RECOVERY_OUTPUTS.values() if path.exists()]
    if existing:
        raise FileExistsError(f"recovery outputs already exist: {existing}")
