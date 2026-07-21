"""Fail-closed validation of the durable physical-cutover receipt."""

from __future__ import annotations

import base64
import binascii
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from .admission import (
    FileAdmissionError,
    StrictJSONError,
    admit_directory,
    admit_regular_file,
    decode_strict_json,
    lexical_absolute,
)


RECEIPT_SCHEMA = "experiments7-cutover-receipt/v1"
VALIDATION_SCHEMA = "experiments7-cutover-receipt-validation/v1"
TRUST_SCHEMA = "experiments7-cutover-trust/v1"
SOURCE_INVENTORY_SCHEMA = "experiments7-cutover-source-inventory/v1"
DESTINATION_INVENTORY_SCHEMA = "experiments7-cutover-destination-inventory/v1"
RESTORE_REPORT_SCHEMA = "experiments7-cutover-restore-report/v1"
PROVIDER_PROOF_SCHEMA = "experiments7-provider-offhost-proof/v1"
ARCHIVE_MAP_SCHEMA = "experiments7-archive-map/v1"
POST_CUTOVER_STATE_SCHEMA = "experiments7-post-cutover-local-state/v1"
POST_CUTOVER_VALIDATION_SCHEMA = (
    "experiments7-post-cutover-local-state-validation/v1"
)

ARCHIVE_MAP_PATH = "archive/archive-map.json"
RECEIPT_PATH = "archive/cutover-receipt.json"
SOURCE_INVENTORY_PATH = "archive/cutover-source-inventory.jsonl"
DESTINATION_INVENTORY_PATH = "archive/cutover-destination-inventory.jsonl"
RESTORE_REPORT_PATH = "archive/cutover-restore-report.json"
PROVIDER_PROOF_PATH = "archive/provider-offhost-proof.json"

ABSENT_CUTOVER_PATHS = ("data/sources", "experiments", "methods")
RAW_METADATA_STUBS = ("README.md", "schema.json")
CUTOVER_PATHS = (*ABSENT_CUTOVER_PATHS, "paper_outputs/raw")

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:+@/-]{7,255}\Z")
_FAKE_WORDS = frozenset(
    {"example", "fake", "local", "none", "placeholder", "self", "test", "unknown"}
)
_OFF_HOST_SCHEMES = frozenset({"az", "gs", "r2", "s3"})
_SIGNATURE_ALGORITHM = "rsa-pkcs1v15-sha256"
_SHA256_DIGEST_INFO = bytes.fromhex("3031300d060960864801650304020105000420")


class CutoverReceiptError(RuntimeError):
    """Raised when any final-completion receipt invariant is unproven."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CutoverReceiptError(message)


def _strict_object(
    value: Any,
    *,
    label: str,
    fields: Sequence[str],
) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} must be an object")
    expected = set(fields)
    actual = set(value)
    _require(actual == expected, f"{label} fields differ: {sorted(actual ^ expected)}")
    return dict(value)


def _admitted_payload(path: Path, *, label: str) -> bytes:
    try:
        return admit_regular_file(path, label=label).payload
    except FileAdmissionError as exc:
        raise CutoverReceiptError(str(exc)) from exc


def _read(repo_root: Path, relative: str, *, label: str) -> bytes:
    root = lexical_absolute(repo_root)
    path = root.joinpath(*PurePosixPath(relative).parts)
    return _admitted_payload(path, label=label)


def _decode_json(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = decode_strict_json(payload, label=label)
    except StrictJSONError as exc:
        raise CutoverReceiptError(str(exc)) from exc
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def _decode_jsonl(payload: bytes, *, label: str) -> list[dict[str, Any]]:
    _require(payload.endswith(b"\n"), f"{label} must end with a newline")
    rows = []
    for number, line in enumerate(payload.splitlines(), start=1):
        _require(bool(line), f"{label} contains a blank row at line {number}")
        rows.append(_decode_json(line, label=f"{label} line {number}"))
    _require(bool(rows), f"{label} must contain at least one row")
    canonical = b"".join(_canonical_json(row) + b"\n" for row in rows)
    _require(payload == canonical, f"{label} is not canonical sorted JSONL")
    return rows


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CutoverReceiptError(f"value is not canonical JSON: {exc}") from exc


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha(value: Any, *, label: str) -> str:
    _require(isinstance(value, str) and _HEX64.fullmatch(value) is not None, f"invalid {label}")
    return value


def _count(value: Any, *, label: str, allow_zero: bool = False) -> int:
    _require(type(value) is int, f"{label} must be an integer")
    _require(value >= (0 if allow_zero else 1), f"{label} is out of range")
    return value


def _identifier(value: Any, *, label: str) -> str:
    _require(isinstance(value, str) and _IDENTIFIER.fullmatch(value) is not None, f"invalid {label}")
    lowered = value.lower()
    _require(not any(word in lowered for word in _FAKE_WORDS), f"{label} is fake/local-only")
    return value


def _timestamp(value: Any, *, label: str, now: datetime) -> datetime:
    _require(isinstance(value, str) and value.endswith("Z"), f"{label} must be UTC Z time")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise CutoverReceiptError(f"invalid {label}") from exc
    parsed = parsed.astimezone(timezone.utc)
    _require(parsed <= now, f"{label} is in the future")
    return parsed


def _off_host_uri(value: Any, *, label: str) -> str:
    _require(isinstance(value, str), f"{label} must be a string")
    parsed = urlsplit(value)
    _require(
        parsed.scheme in _OFF_HOST_SCHEMES and bool(parsed.netloc) and bool(parsed.path.strip("/")),
        f"{label} is not a supported off-host URI",
    )
    lowered = value.lower()
    _require(not any(word in lowered for word in _FAKE_WORDS), f"{label} is fake/local-only")
    return value


def _relative(value: Any, *, expected: str, label: str) -> str:
    _require(value == expected, f"{label} must be canonical path {expected!r}")
    return expected


def _signature(value: Any, *, label: str) -> bytes:
    _require(isinstance(value, str) and len(value) >= 128, f"invalid {label}")
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise CutoverReceiptError(f"invalid base64 {label}") from exc


def _trusted_key(
    trust: Mapping[str, Any],
    *,
    role: str,
    identity: str,
    key_id: str,
) -> tuple[int, int]:
    field = f"{role}_signers"
    values = trust[field]
    _require(isinstance(values, list), f"trust {field} must be a list")
    matches = []
    for index, item in enumerate(values):
        key = _strict_object(
            item,
            label=f"trust {field}[{index}]",
            fields=("algorithm", "exponent", "identity", "key_id", "modulus_hex"),
        )
        if key["identity"] == identity and key["key_id"] == key_id:
            matches.append(key)
    _require(len(matches) == 1, f"exactly one trusted {role} key must match")
    key = matches[0]
    _require(key["algorithm"] == _SIGNATURE_ALGORITHM, f"unsupported trusted {role} algorithm")
    _identifier(key["identity"], label=f"trusted {role} identity")
    _identifier(key["key_id"], label=f"trusted {role} key_id")
    _require(isinstance(key["modulus_hex"], str), f"trusted {role} modulus must be hex")
    try:
        modulus = int(key["modulus_hex"], 16)
    except ValueError as exc:
        raise CutoverReceiptError(f"trusted {role} modulus must be hex") from exc
    exponent = key["exponent"]
    _require(type(exponent) is int and exponent >= 3 and exponent % 2 == 1, f"invalid trusted {role} exponent")
    _require(modulus.bit_length() >= 2048, f"trusted {role} RSA key is below 2048 bits")
    return modulus, exponent


def _verify_rsa(payload: bytes, signature: bytes, key: tuple[int, int], *, label: str) -> None:
    modulus, exponent = key
    width = (modulus.bit_length() + 7) // 8
    _require(len(signature) == width, f"{label} RSA signature length differs")
    recovered = pow(int.from_bytes(signature, "big"), exponent, modulus).to_bytes(width, "big")
    expected_tail = _SHA256_DIGEST_INFO + hashlib.sha256(payload).digest()
    _require(recovered.startswith(b"\x00\x01"), f"{label} RSA signature is invalid")
    separator = recovered.find(b"\x00", 2)
    _require(separator >= 10, f"{label} RSA padding is invalid")
    _require(recovered[2:separator] == b"\xff" * (separator - 2), f"{label} RSA padding is invalid")
    _require(recovered[separator + 1 :] == expected_tail, f"{label} RSA signature is invalid")


def _load_trust(
    repo_root: Path,
    *,
    trust_bundle: Path,
    expected_sha256: str,
) -> tuple[dict[str, Any], str]:
    root = lexical_absolute(repo_root)
    trust_path = lexical_absolute(trust_bundle)
    try:
        trust_path.relative_to(root)
    except ValueError:
        pass
    else:
        raise CutoverReceiptError(
            "cutover trust bundle must be outside the repository"
        )
    expected = _sha(expected_sha256, label="expected trust bundle SHA-256")
    payload = _admitted_payload(trust_path, label="external cutover trust bundle")
    actual = _sha256(payload)
    _require(actual == expected, "external cutover trust bundle digest differs")
    trust = _strict_object(
        _decode_json(payload, label="external cutover trust bundle"),
        label="external cutover trust bundle",
        fields=("provider_signers", "reviewer_signers", "schema"),
    )
    _require(trust["schema"] == TRUST_SCHEMA, "cutover trust schema differs")
    return trust, actual


def _inventory_descriptor(value: Any, *, source: bool) -> dict[str, Any]:
    label = "source inventory" if source else "destination inventory"
    expected_path = SOURCE_INVENTORY_PATH if source else DESTINATION_INVENTORY_PATH
    expected_schema = SOURCE_INVENTORY_SCHEMA if source else DESTINATION_INVENTORY_SCHEMA
    descriptor = _strict_object(
        value,
        label=f"receipt {label}",
        fields=("bytes", "file_count", "path", "schema", "sha256"),
    )
    _relative(descriptor["path"], expected=expected_path, label=f"receipt {label} path")
    _require(descriptor["schema"] == expected_schema, f"receipt {label} schema differs")
    _sha(descriptor["sha256"], label=f"receipt {label} SHA-256")
    _count(descriptor["file_count"], label=f"receipt {label} file_count")
    _count(descriptor["bytes"], label=f"receipt {label} bytes", allow_zero=True)
    return descriptor


def _source_inventory(payload: bytes) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    rows = _decode_jsonl(payload, label="source inventory")
    seen: set[str] = set()
    totals = {path: {"bytes": 0, "file_count": 0} for path in CUTOVER_PATHS}
    for index, item in enumerate(rows):
        row = _strict_object(item, label=f"source inventory row {index}", fields=("path", "schema", "sha256", "size"))
        _require(row["schema"] == SOURCE_INVENTORY_SCHEMA, "source inventory row schema differs")
        path = row["path"]
        _require(isinstance(path, str) and path not in seen, "source inventory paths must be unique")
        pure = PurePosixPath(path)
        _require(not pure.is_absolute() and ".." not in pure.parts and "\\" not in path, "unsafe source inventory path")
        roots = [root for root in CUTOVER_PATHS if path.startswith(root + "/")]
        _require(len(roots) == 1, f"source inventory path is outside cutover scope: {path!r}")
        size = _count(row["size"], label="source inventory size", allow_zero=True)
        _sha(row["sha256"], label="source inventory SHA-256")
        seen.add(path)
        totals[roots[0]]["bytes"] += size
        totals[roots[0]]["file_count"] += 1
    return rows, totals


def _destination_inventory(payload: bytes) -> list[dict[str, Any]]:
    rows = _decode_jsonl(payload, label="destination inventory")
    seen_sources: set[str] = set()
    seen_objects: set[tuple[str, str]] = set()
    for index, item in enumerate(rows):
        row = _strict_object(
            item,
            label=f"destination inventory row {index}",
            fields=("object_uri", "schema", "sha256", "size", "source_path", "version_id"),
        )
        _require(row["schema"] == DESTINATION_INVENTORY_SCHEMA, "destination inventory row schema differs")
        source = row["source_path"]
        _require(isinstance(source, str) and source not in seen_sources, "destination source paths must be unique")
        uri = _off_host_uri(row["object_uri"], label="destination object URI")
        version = _identifier(row["version_id"], label="destination version_id")
        _require((uri, version) not in seen_objects, "destination object versions must be unique")
        _sha(row["sha256"], label="destination SHA-256")
        _count(row["size"], label="destination size", allow_zero=True)
        seen_sources.add(source)
        seen_objects.add((uri, version))
    return rows


def _validate_inventory_pair(source: list[dict[str, Any]], destination: list[dict[str, Any]]) -> None:
    by_source = {row["path"]: row for row in source}
    by_destination = {row["source_path"]: row for row in destination}
    _require(set(by_source) == set(by_destination), "source/destination inventory path sets differ")
    for path, source_row in by_source.items():
        destination_row = by_destination[path]
        _require(
            destination_row["size"] == source_row["size"] and destination_row["sha256"] == source_row["sha256"],
            f"source/destination size or hash differs for {path}",
        )


def inspect_post_cutover_local_state(repo_root: Path) -> dict[str, Any]:
    """Bind the actual local state after physical externalization."""

    root = lexical_absolute(repo_root)
    snapshot: dict[str, Any] = {
        "absent_paths": [],
        "raw": {"metadata_stubs": [], "status": "absent"},
        "schema": POST_CUTOVER_STATE_SCHEMA,
    }
    try:
        with admit_directory(root, label="cutover repository") as repository:
            for relative in ABSENT_CUTOVER_PATHS:
                if repository.exists(relative, label=f"post-cutover path {relative}"):
                    raise CutoverReceiptError(
                        f"post-cutover payload still exists: {relative}"
                    )
                snapshot["absent_paths"].append(relative)

            raw_relative = "paper_outputs/raw"
            if repository.exists(raw_relative, label="post-cutover raw path"):
                raw_path = root.joinpath(*PurePosixPath(raw_relative).parts)
                with admit_directory(raw_path, label="post-cutover raw metadata"):
                    try:
                        entries = sorted(
                            os.scandir(raw_path),
                            key=lambda entry: entry.name,
                        )
                    except OSError as exc:
                        raise CutoverReceiptError(
                            f"cannot inspect {raw_relative}: {exc}"
                        ) from exc
                    names = {entry.name for entry in entries}
                    allowed = set(RAW_METADATA_STUBS)
                    if names != allowed:
                        raise CutoverReceiptError(
                            f"post-cutover payload still exists or metadata stubs differ: "
                            f"{raw_relative}: {sorted(names ^ allowed)}"
                        )
                    stubs = []
                    for entry in entries:
                        if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                            raise CutoverReceiptError(
                                f"post-cutover metadata stub is unsafe: "
                                f"{raw_relative}/{entry.name}"
                            )
                        payload = _admitted_payload(
                            raw_path / entry.name,
                            label=f"post-cutover metadata stub {entry.name}",
                        )
                        stubs.append(
                            {
                                "bytes": len(payload),
                                "path": f"{raw_relative}/{entry.name}",
                                "sha256": _sha256(payload),
                            }
                        )
                    snapshot["raw"] = {
                        "metadata_stubs": stubs,
                        "status": "metadata_stubs_only",
                    }
    except FileAdmissionError as exc:
        raise CutoverReceiptError(str(exc)) from exc

    payload = _canonical_json(snapshot)
    return {
        "no_mutations": True,
        "schema": POST_CUTOVER_VALIDATION_SCHEMA,
        "sha256": _sha256(payload),
        "snapshot": snapshot,
        "status": "verified",
        "verified": True,
    }


def post_cutover_local_state_report(repo_root: Path) -> dict[str, Any]:
    try:
        return inspect_post_cutover_local_state(repo_root)
    except Exception as exc:
        return {
            "error": str(exc),
            "no_mutations": True,
            "schema": POST_CUTOVER_VALIDATION_SCHEMA,
            "status": "blocked",
            "verified": False,
        }


def validate_cutover_receipt(
    repo_root: Path,
    *,
    trust_bundle: Path,
    expected_trust_sha256: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Validate all durable receipt bindings without mutating repository state."""

    current_time = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    root = lexical_absolute(repo_root)
    receipt_payload = _read(root, RECEIPT_PATH, label="cutover receipt")
    receipt = _strict_object(
        _decode_json(receipt_payload, label="cutover receipt"),
        label="cutover receipt",
        fields=(
            "approval",
            "archive_map",
            "completed_at",
            "destination_inventory",
            "post_cutover_local_state",
            "provider_authentication",
            "receipt_id",
            "restore_evidence",
            "reviewer_signature",
            "rollback",
            "schema",
            "source_inventory",
        ),
    )
    _require(receipt["schema"] == RECEIPT_SCHEMA, "cutover receipt schema differs")
    _identifier(receipt["receipt_id"], label="receipt_id")
    completed_at = _timestamp(receipt["completed_at"], label="completed_at", now=current_time)
    trust, trust_sha256 = _load_trust(
        root,
        trust_bundle=trust_bundle,
        expected_sha256=expected_trust_sha256,
    )

    local_state_binding = _strict_object(
        receipt["post_cutover_local_state"],
        label="receipt post-cutover local state",
        fields=("schema", "sha256"),
    )
    _require(
        local_state_binding["schema"] == POST_CUTOVER_STATE_SCHEMA,
        "post-cutover local-state schema binding differs",
    )
    local_state = inspect_post_cutover_local_state(root)
    _require(
        local_state_binding["sha256"] == local_state["sha256"],
        "post-cutover local-state digest differs from signed receipt",
    )

    archive_binding = _strict_object(
        receipt["archive_map"],
        label="receipt archive map",
        fields=("path", "schema", "sha256"),
    )
    _relative(archive_binding["path"], expected=ARCHIVE_MAP_PATH, label="archive map path")
    _require(archive_binding["schema"] == ARCHIVE_MAP_SCHEMA, "receipt archive map schema differs")
    archive_payload = _read(root, ARCHIVE_MAP_PATH, label="archive map")
    _require(_sha256(archive_payload) == _sha(archive_binding["sha256"], label="archive map SHA-256"), "archive map digest differs from receipt")
    archive_map = _decode_json(archive_payload, label="archive map")
    _require(archive_map.get("schema") == ARCHIVE_MAP_SCHEMA, "archive map schema differs")
    _require(archive_map.get("planned_only") is False, "archive map remains planned_only")
    _require(archive_map.get("no_mutations") is True, "archive map no_mutations contract differs")
    records = archive_map.get("records")
    _require(isinstance(records, list), "archive map records must be a list")
    by_path = {row.get("path"): row for row in records if isinstance(row, dict)}
    _require(all(path in by_path for path in CUTOVER_PATHS), "archive map lacks a required cutover path")

    source_descriptor = _inventory_descriptor(receipt["source_inventory"], source=True)
    destination_descriptor = _inventory_descriptor(receipt["destination_inventory"], source=False)
    source_payload = _read(root, SOURCE_INVENTORY_PATH, label="source inventory")
    destination_payload = _read(root, DESTINATION_INVENTORY_PATH, label="destination inventory")
    _require(_sha256(source_payload) == source_descriptor["sha256"], "source inventory digest differs")
    _require(_sha256(destination_payload) == destination_descriptor["sha256"], "destination inventory digest differs")
    source_rows, source_totals = _source_inventory(source_payload)
    destination_rows = _destination_inventory(destination_payload)
    _validate_inventory_pair(source_rows, destination_rows)
    total_bytes = sum(row["size"] for row in source_rows)
    _require(source_descriptor["file_count"] == len(source_rows) and source_descriptor["bytes"] == total_bytes, "source inventory totals differ")
    _require(destination_descriptor["file_count"] == len(destination_rows) and destination_descriptor["bytes"] == total_bytes, "destination inventory totals differ")
    for path in CUTOVER_PATHS:
        record = by_path[path]
        _require(record.get("symlink_count") == 0, f"archive map symlinks are not receipted for {path}")
        _require(
            record.get("file_count") == source_totals[path]["file_count"] and record.get("bytes") == source_totals[path]["bytes"],
            f"archive map/source inventory totals differ for {path}",
        )

    restore_binding = _strict_object(
        receipt["restore_evidence"],
        label="receipt restore evidence",
        fields=("path", "schema", "sha256"),
    )
    _relative(restore_binding["path"], expected=RESTORE_REPORT_PATH, label="restore report path")
    _require(restore_binding["schema"] == RESTORE_REPORT_SCHEMA, "restore report schema binding differs")
    restore_payload = _read(root, RESTORE_REPORT_PATH, label="restore report")
    _require(_sha256(restore_payload) == _sha(restore_binding["sha256"], label="restore report SHA-256"), "restore report digest differs")
    restore = _strict_object(
        _decode_json(restore_payload, label="restore report"),
        label="restore report",
        fields=(
            "archive_map_sha256",
            "destination_inventory_sha256",
            "object_count",
            "object_manifest_sha256",
            "restore_id",
            "restored_at",
            "schema",
            "source_inventory_sha256",
            "status",
            "total_bytes",
        ),
    )
    _require(restore["schema"] == RESTORE_REPORT_SCHEMA and restore["status"] == "restored_verified", "restore report is not verified")
    _identifier(restore["restore_id"], label="restore_id")
    restored_at = _timestamp(restore["restored_at"], label="restored_at", now=current_time)
    _require(restored_at <= completed_at, "restore occurred after receipt completion")
    _require(restore["archive_map_sha256"] == archive_binding["sha256"], "restore/archive map binding differs")
    _require(restore["source_inventory_sha256"] == source_descriptor["sha256"], "restore/source inventory binding differs")
    _require(restore["destination_inventory_sha256"] == destination_descriptor["sha256"], "restore/destination inventory binding differs")
    _sha(restore["object_manifest_sha256"], label="restore object manifest SHA-256")
    _require(restore["object_count"] == len(destination_rows) and restore["total_bytes"] == total_bytes, "restore inventory totals differ")

    provider_binding = _strict_object(
        receipt["provider_authentication"],
        label="provider authentication",
        fields=("algorithm", "key_id", "path", "provider_id", "schema", "sha256", "signature", "signer_identity"),
    )
    _relative(provider_binding["path"], expected=PROVIDER_PROOF_PATH, label="provider proof path")
    _require(provider_binding["schema"] == PROVIDER_PROOF_SCHEMA, "provider proof schema binding differs")
    _require(provider_binding["algorithm"] == _SIGNATURE_ALGORITHM, "provider signature algorithm differs")
    provider_id = _identifier(provider_binding["provider_id"], label="provider_id")
    provider_identity = _identifier(provider_binding["signer_identity"], label="provider signer identity")
    provider_key_id = _identifier(provider_binding["key_id"], label="provider key_id")
    provider_payload = _read(root, PROVIDER_PROOF_PATH, label="provider proof")
    _require(_sha256(provider_payload) == _sha(provider_binding["sha256"], label="provider proof SHA-256"), "provider proof digest differs")
    provider = _strict_object(
        _decode_json(provider_payload, label="provider proof"),
        label="provider proof",
        fields=(
            "archive_map_sha256",
            "destination_inventory_sha256",
            "immutability_enabled",
            "immutability_mode",
            "issued_at",
            "object_count",
            "object_manifest_sha256",
            "provider_account_id",
            "provider_id",
            "restore_report_sha256",
            "schema",
            "source_inventory_sha256",
            "storage_location",
            "total_bytes",
            "version_pointer",
            "versioning_enabled",
        ),
    )
    _require(provider["schema"] == PROVIDER_PROOF_SCHEMA and provider["provider_id"] == provider_id, "provider proof identity differs")
    _identifier(provider["provider_account_id"], label="provider_account_id")
    issued_at = _timestamp(provider["issued_at"], label="provider issued_at", now=current_time)
    _require(issued_at <= completed_at, "provider proof was issued after receipt completion")
    _off_host_uri(provider["storage_location"], label="provider storage_location")
    _off_host_uri(provider["version_pointer"], label="provider version_pointer")
    _require(provider["versioning_enabled"] is True and provider["immutability_enabled"] is True, "provider versioning/immutability is not enabled")
    _require(provider["immutability_mode"] in {"bucket_lock", "immutable_blob", "object_lock_compliance"}, "provider immutability mode differs")
    _require(provider["archive_map_sha256"] == archive_binding["sha256"], "provider/archive map binding differs")
    _require(provider["source_inventory_sha256"] == source_descriptor["sha256"], "provider/source inventory binding differs")
    _require(provider["destination_inventory_sha256"] == destination_descriptor["sha256"], "provider/destination inventory binding differs")
    _require(provider["restore_report_sha256"] == restore_binding["sha256"], "provider/restore report binding differs")
    _require(provider["object_manifest_sha256"] == restore["object_manifest_sha256"], "provider/restore object manifest binding differs")
    _require(provider["object_count"] == len(destination_rows) and provider["total_bytes"] == total_bytes, "provider inventory totals differ")
    provider_key = _trusted_key(trust, role="provider", identity=provider_identity, key_id=provider_key_id)
    _verify_rsa(_canonical_json(provider), _signature(provider_binding["signature"], label="provider signature"), provider_key, label="provider")

    rollback = _strict_object(
        receipt["rollback"],
        label="receipt rollback",
        fields=("object_manifest_sha256", "restore_id", "storage_location", "version_pointer"),
    )
    _require(rollback["storage_location"] == provider["storage_location"], "rollback storage location differs")
    _require(rollback["version_pointer"] == provider["version_pointer"], "rollback version pointer differs")
    _require(rollback["object_manifest_sha256"] == provider["object_manifest_sha256"], "rollback object manifest differs")
    _require(rollback["restore_id"] == restore["restore_id"], "rollback restore_id differs")

    approval = _strict_object(
        receipt["approval"],
        label="receipt approval",
        fields=("approved_at", "decision", "independent_of", "reviewer_id", "reviewer_name"),
    )
    reviewer_name = _identifier(approval["reviewer_name"], label="reviewer_name")
    reviewer_id = _identifier(approval["reviewer_id"], label="reviewer_id")
    independent_of = _identifier(approval["independent_of"], label="approval independent_of")
    _require(approval["decision"] == "approved_physical_cutover", "reviewer decision does not approve physical cutover")
    approved_at = _timestamp(approval["approved_at"], label="approved_at", now=current_time)
    _require(restored_at <= approved_at <= completed_at, "approval chronology differs")
    _require(reviewer_id not in {provider_id, provider_identity, independent_of}, "reviewer is not independent")

    reviewer_signature = _strict_object(
        receipt["reviewer_signature"],
        label="reviewer signature",
        fields=("algorithm", "key_id", "signature", "signer_identity"),
    )
    _require(reviewer_signature["algorithm"] == _SIGNATURE_ALGORITHM, "reviewer signature algorithm differs")
    _require(reviewer_signature["signer_identity"] == reviewer_id, "reviewer signature identity differs")
    reviewer_key_id = _identifier(reviewer_signature["key_id"], label="reviewer key_id")
    reviewer_key = _trusted_key(trust, role="reviewer", identity=reviewer_id, key_id=reviewer_key_id)
    signed_receipt = dict(receipt)
    signed_receipt.pop("reviewer_signature")
    _verify_rsa(_canonical_json(signed_receipt), _signature(reviewer_signature["signature"], label="reviewer signature"), reviewer_key, label="reviewer")

    final_local_state = inspect_post_cutover_local_state(root)
    _require(
        final_local_state["sha256"] == local_state_binding["sha256"],
        "post-cutover local state changed during receipt validation",
    )
    local_state = final_local_state

    return {
        "archive_map_sha256": archive_binding["sha256"],
        "externalization_authorized": True,
        "independent_approval_verified": True,
        "no_mutations": True,
        "post_cutover_local_state": local_state,
        "physical_cutover_complete": True,
        "provider_authenticity_verified": True,
        "receipt_id": receipt["receipt_id"],
        "receipt_sha256": _sha256(receipt_payload),
        "reviewer_name": reviewer_name,
        "schema": VALIDATION_SCHEMA,
        "status": "verified",
        "trust_bundle_sha256": trust_sha256,
        "verified": True,
    }


def cutover_receipt_report(
    repo_root: Path,
    *,
    trust_bundle: Path | None = None,
    expected_trust_sha256: str | None = None,
) -> dict[str, Any]:
    """Return a stable blocked report instead of treating absence as success."""

    try:
        _require(trust_bundle is not None, "external cutover trust bundle is required")
        _require(
            expected_trust_sha256 is not None,
            "expected external trust bundle SHA-256 is required",
        )
        return validate_cutover_receipt(
            repo_root,
            trust_bundle=trust_bundle,
            expected_trust_sha256=expected_trust_sha256,
        )
    except Exception as exc:
        return {
            "error": str(exc),
            "externalization_authorized": False,
            "independent_approval_verified": False,
            "no_mutations": True,
            "physical_cutover_complete": False,
            "provider_authenticity_verified": False,
            "schema": VALIDATION_SCHEMA,
            "status": "blocked",
            "verified": False,
        }


__all__ = [
    "ARCHIVE_MAP_PATH",
    "ARCHIVE_MAP_SCHEMA",
    "ABSENT_CUTOVER_PATHS",
    "CUTOVER_PATHS",
    "CutoverReceiptError",
    "DESTINATION_INVENTORY_PATH",
    "DESTINATION_INVENTORY_SCHEMA",
    "PROVIDER_PROOF_PATH",
    "PROVIDER_PROOF_SCHEMA",
    "POST_CUTOVER_STATE_SCHEMA",
    "POST_CUTOVER_VALIDATION_SCHEMA",
    "RECEIPT_PATH",
    "RECEIPT_SCHEMA",
    "RAW_METADATA_STUBS",
    "RESTORE_REPORT_PATH",
    "RESTORE_REPORT_SCHEMA",
    "SOURCE_INVENTORY_PATH",
    "SOURCE_INVENTORY_SCHEMA",
    "TRUST_SCHEMA",
    "VALIDATION_SCHEMA",
    "cutover_receipt_report",
    "inspect_post_cutover_local_state",
    "post_cutover_local_state_report",
    "validate_cutover_receipt",
]
