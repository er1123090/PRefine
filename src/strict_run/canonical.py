"""Canonical encodings and fail-closed namespace validation."""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import PurePosixPath
from typing import NoReturn


STRICT_RUN_ID_RE = re.compile(
    r"exp7-strict-v[0-9]+-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{32}", re.ASCII
)
SHA256_RE = re.compile(r"[0-9a-f]{64}", re.ASCII)


class StrictRunError(RuntimeError):
    """A stable, non-secret-bearing strict-run contract failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def fail(code: str, message: str) -> NoReturn:
    raise StrictRunError(code, message)


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def strict_json_loads(raw: bytes, field: str) -> object:
    """Decode UTF-8 JSON while rejecting duplicate keys and non-finite numbers."""

    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                fail("DUPLICATE_JSON_KEY", f"{field} contains duplicate key {key!r}")
            result[key] = value
        return result

    def invalid_constant(value: str) -> object:
        fail("NONFINITE_JSON_NUMBER", f"{field} contains {value}")

    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=invalid_constant,
        )
    except StrictRunError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise StrictRunError("INVALID_JSON", f"{field} is not strict UTF-8 JSON") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        fail("INVALID_SHA256", f"{field} must be canonical lowercase SHA-256")
    return value


def validate_sealed_run_id(value: object) -> str:
    """Validate the ID before any filesystem access."""

    if not isinstance(value, str):
        fail("INVALID_RUN_ID", "sealed_run_id must be text")
    if "\x00" in value or len(value.encode("utf-8")) > 128:
        fail("INVALID_RUN_ID", "sealed_run_id contains forbidden bytes or is overlong")
    if STRICT_RUN_ID_RE.fullmatch(value) is None:
        fail("INVALID_RUN_ID", "sealed_run_id is not canonical")
    timestamp = value.split("-", 4)[3]
    try:
        parsed = datetime.strptime(timestamp, "%Y%m%dT%H%M%SZ")
    except ValueError as exc:
        raise StrictRunError("INVALID_RUN_ID", "timestamp is not a real UTC date") from exc
    if parsed.strftime("%Y%m%dT%H%M%SZ") != timestamp:
        fail("INVALID_RUN_ID", "timestamp spelling is not canonical")
    return value


def validate_absolute_path_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        fail("INVALID_PATH", f"{field} must be a nonempty absolute path")
    if "\x00" in value or not value.startswith("/"):
        fail("INVALID_PATH", f"{field} must be an absolute NUL-free path")
    if value != "/" and value.endswith("/"):
        fail("NONCANONICAL_PATH", f"{field} has a trailing slash")
    components = value.split("/")[1:]
    if any(component in ("", ".", "..") for component in components):
        fail("NONCANONICAL_PATH", f"{field} has a noncanonical component")
    if os.path.normpath(value) != value:
        fail("NONCANONICAL_PATH", f"{field} is not normalized")
    return value


def validate_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        fail("INVALID_RELATIVE_PATH", "relative path must be nonempty text")
    if "\x00" in value or value.startswith("/") or value.endswith("/"):
        fail("INVALID_RELATIVE_PATH", "relative path is not canonical")
    parts = value.split("/")
    if any(part in ("", ".", "..") for part in parts):
        fail("INVALID_RELATIVE_PATH", "relative path contains a forbidden component")
    if str(PurePosixPath(value)) != value:
        fail("INVALID_RELATIVE_PATH", "relative path spelling is not canonical")
    return value


def require_exact_keys(
    value: object, expected: set[str], schema: str
) -> dict[str, object]:
    if not isinstance(value, dict):
        fail("SCHEMA_INVALID", f"{schema} must be an object")
    actual = set(value)
    if actual != expected:
        fail(
            "SCHEMA_INVALID",
            f"{schema} keys differ; missing={sorted(expected - actual)!r} "
            f"extra={sorted(actual - expected)!r}",
        )
    return value
