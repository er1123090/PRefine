#!/usr/bin/env python3
"""Append-only publication of the historical G0 ordering incident."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path, PurePosixPath
from typing import Mapping

RUN_ID = "exp7-g0-20260715T074803Z-bd1e85e8-f246-4261-b9e8-27d87e47540a"
INCIDENT_REL = f"manifests/sealed-runs/{RUN_ID}/incidents/0001-g0-ord-01.json"
TERMINAL_REL = f"manifests/sealed-runs/{RUN_ID}/terminal-invalidated.json"

CRITICAL_HASHES = {
    "manifests/cp0-seal.json": "127a791d91860feb01296396f246f511b198ed75b684f973f586066abeb3be1f",
    "manifests/source-pre.jsonl": "6ab9fd4b02a55b2c59889ee2034f42d8162bca99a6abbfc35a0d65ca02d63bbc",
    "variants/registry.json": "7126e93f2bfdfff49f250f5d1308df0de736fa10fe54959b49f3a4582ce24b38",
    "lineage/code.jsonl": "ce0d6a295fbc91f655b1a0abda439ec8d50e7645012b2ceb7a31089f6d44dc7a",
    "lineage/config.jsonl": "71d64df6953be1ef359c06be57d4e3f300e6003a5d31f22e34a9e4c1615433ec",
}

INVENTORY_ROOTS = ("manifests", "scripts/g0", "variants", "lineage", "paper_outputs/inventory")
EXCLUDED_COMPONENTS = {"sealed-runs", "v2", "runs", "attempts", "cache", "caches", "__pycache__"}
PROTECTED_ROOTS = (
    "/data/minseo/experiments4",
    "/data/minseo/experiments5",
    "/data/minseo/experiments6",
    "/data/minseo/experiments7/_paper",
)


class PublicationError(RuntimeError):
    """A safety or integrity condition prevented publication."""


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _safe_root(root: Path) -> Path:
    root = root.absolute()
    st = root.lstat()
    if not stat.S_ISDIR(st.st_mode) or stat.S_ISLNK(st.st_mode):
        raise PublicationError("unsafe root")
    return root


def _excluded(relative: PurePosixPath) -> bool:
    return any(part in EXCLUDED_COMPONENTS for part in relative.parts)


def _hash_regular(path: Path) -> tuple[str, int, int]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise PublicationError(f"cannot safely open {path}") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise PublicationError(f"not a regular file: {path}")
        digest = hashlib.sha256()
        size = 0
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            size += len(block)
        after = os.fstat(fd)
        identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        if identity_before != identity_after or size != before.st_size:
            raise PublicationError(f"file changed while hashing: {path}")
        return digest.hexdigest(), size, stat.S_IMODE(before.st_mode)
    finally:
        os.close(fd)


def build_legacy_baseline(root: Path) -> list[dict[str, object]]:
    root = _safe_root(root)
    rows: list[dict[str, object]] = []
    for prefix in INVENTORY_ROOTS:
        start = root.joinpath(*PurePosixPath(prefix).parts)
        try:
            start_st = start.lstat()
        except FileNotFoundError:
            continue
        if not stat.S_ISDIR(start_st.st_mode) or stat.S_ISLNK(start_st.st_mode):
            raise PublicationError(f"unsafe inventory root: {prefix}")
        stack = [(start, PurePosixPath(prefix))]
        while stack:
            directory, relative_dir = stack.pop()
            with os.scandir(directory) as entries:
                ordered = sorted(entries, key=lambda entry: os.fsencode(entry.name))
            for entry in ordered:
                relative = relative_dir / entry.name
                if _excluded(relative):
                    continue
                if entry.is_symlink():
                    raise PublicationError(f"symlink in managed inventory: {relative}")
                if entry.is_dir(follow_symlinks=False):
                    stack.append((Path(entry.path), relative))
                elif entry.is_file(follow_symlinks=False):
                    digest, size, mode = _hash_regular(Path(entry.path))
                    rows.append({"mode": mode, "path": relative.as_posix(), "sha256": digest, "size": size})
                else:
                    raise PublicationError(f"non-regular managed entry: {relative}")
    rows.sort(key=lambda row: os.fsencode(str(row["path"])))
    return rows


def _verify_critical(baseline: list[dict[str, object]], expected: Mapping[str, str]) -> None:
    actual = {str(row["path"]): str(row["sha256"]) for row in baseline}
    for path, digest in expected.items():
        if actual.get(path) != digest:
            raise PublicationError(f"critical hash drift: {path}")


def _ensure_parent(root: Path, relative: str) -> Path:
    current = root
    parts = PurePosixPath(relative).parts[:-1]
    for part in parts:
        current = current / part
        try:
            os.mkdir(current, 0o755)
            parent_fd = os.open(current.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        except FileExistsError:
            pass
        st = current.lstat()
        if not stat.S_ISDIR(st.st_mode) or stat.S_ISLNK(st.st_mode):
            raise PublicationError(f"unsafe publication parent: {current}")
    return root.joinpath(*PurePosixPath(relative).parts)


def _publish_exclusive(root: Path, relative: str, payload: bytes) -> None:
    final = _ensure_parent(root, relative)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(final, flags, 0o644)
    except OSError as exc:
        raise PublicationError(f"exclusive publication refused: {relative}") from exc
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise PublicationError(f"short write: {relative}")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    parent_fd = os.open(final.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def publish(root: Path, expected_hashes: Mapping[str, str] = CRITICAL_HASHES) -> tuple[dict, dict]:
    root = _safe_root(root)
    # Refuse before inventorying if either output already exists or has an unsafe parent.
    for relative in (INCIDENT_REL, TERMINAL_REL):
        final = root.joinpath(*PurePosixPath(relative).parts)
        if final.exists() or final.is_symlink():
            raise PublicationError(f"output already exists: {relative}")
    baseline = build_legacy_baseline(root)
    _verify_critical(baseline, expected_hashes)
    incident = {
        "actor": {"canonical_task": "/root/planner_g1_matrix", "role": "planner"},
        "argv": None,
        "argv_unavailable_reason": "Exact argv was not captured contemporaneously.",
        "content_bytes_read": 0,
        "decision": "INVALIDATED_POLICY_VIOLATION",
        "filenames_observed": True,
        "legacy_baseline": baseline,
        "observed_at": None,
        "observed_at_unavailable_reason": "Exact observation time was not captured contemporaneously.",
        "operation": "descendant_enumeration",
        "protected_root_strings": list(PROTECTED_ROOTS),
        "protected_writes": 0,
        "run_id": RUN_ID,
        "schema": "experiments7-g0-incident/v1",
        "violated": ["G0-ORD-01", "I-G0-01"],
    }
    incident_bytes = canonical_json(incident)
    incident_hash = hashlib.sha256(incident_bytes).hexdigest()
    terminal = {
        "acceptance_eligible": False,
        "incident": {"path": INCIDENT_REL, "sha256": incident_hash},
        "legacy_baseline": baseline,
        "run_id": RUN_ID,
        "schema": "experiments7-terminal-invalidation/v1",
        "terminal_state": "failed",
    }
    _publish_exclusive(root, INCIDENT_REL, incident_bytes)
    # Deliberately last: an incident without this commit marker remains ineligible.
    _publish_exclusive(root, TERMINAL_REL, canonical_json(terminal))
    return incident, terminal


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    try:
        incident, terminal = publish(args.root)
    except (OSError, PublicationError) as exc:
        print(str(exc), file=os.sys.stderr)
        return 2
    print(json.dumps({"incident": INCIDENT_REL, "records": len(incident["legacy_baseline"]), "terminal": TERMINAL_REL}, sort_keys=True))
    assert terminal["acceptance_eligible"] is False
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
