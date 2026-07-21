from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


SECRET_RE = re.compile(r"(?:api[_-]?key|token|secret|password|credential)", re.IGNORECASE)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_config_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def clean_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or value == "." or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe relative path: {value!r}")
    return path


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def write_json_exclusive(path: Path, value: Any) -> None:
    data = canonical_json_bytes(value) + b"\n"
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def redact_argv(argv: Iterable[str]) -> list[str]:
    values = list(argv)
    result: list[str] = []
    redact_next = False
    for value in values:
        if redact_next:
            result.append("<redacted>")
            redact_next = False
            continue
        if value.startswith("--") and "=" in value:
            key, _ = value.split("=", 1)
            result.append(f"{key}=<redacted>" if SECRET_RE.search(key) else value)
            continue
        result.append(value)
        if value.startswith("-") and SECRET_RE.search(value):
            redact_next = True
    return result


def credential_presence(environment: dict[str, str]) -> dict[str, str]:
    return {
        key: "<redacted-present>"
        for key, value in sorted(environment.items())
        if value and SECRET_RE.search(key)
    }
