#!/usr/bin/env python3
"""Append-only correction for the legacy incident baseline scope omission."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path, PurePosixPath

RUN_ID = "exp7-g0-20260715T074803Z-bd1e85e8-f246-4261-b9e8-27d87e47540a"
INCIDENT_REL = f"manifests/sealed-runs/{RUN_ID}/incidents/0001-g0-ord-01.json"
TERMINAL_REL = f"manifests/sealed-runs/{RUN_ID}/terminal-invalidated.json"
OUTPUT_REL = f"manifests/sealed-runs/{RUN_ID}/incidents/0002-legacy-baseline-supplement.json"
INCIDENT_SHA256 = "6a9be3649482b4a7072ab235ee440883132273d9c582882b09442f0a89d4942b"
TERMINAL_SHA256 = "90ef8d22c21f33ec79547ac9041a74d98122f739991ab3d61d2f009aef39dc1c"
OMITTED = {
    ".experiments7-owner.json": "700a9e70a382d331fbc855e3976e151cccc75c79ed94ac1aa11446b75d5c27ec",
    "README.md": "3f32154f9bf480fcc077e94d26378a6997811bed4fc955160567a615dd4b32fc",
    "scripts/validate_g1_inventory.py": "1119da36837a15c9ab2deb4403f8cd21c30f77329a7d01b35c930412298d9abc",
}
REASON = "The original managed-root inventory omitted three repository singleton files from the legacy baseline scope."


class PublicationError(RuntimeError):
    pass


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _root(path: Path) -> Path:
    path = path.absolute()
    st = path.lstat()
    if not stat.S_ISDIR(st.st_mode) or stat.S_ISLNK(st.st_mode):
        raise PublicationError("unsafe root")
    return path


def _parts(relative: str) -> tuple[str, ...]:
    path = PurePosixPath(relative)
    if path.is_absolute() or not path.parts or any(p in ("", ".", "..") for p in path.parts):
        raise PublicationError(f"unsafe relative path: {relative}")
    return path.parts


def _read_under(root: Path, relative: str) -> tuple[bytes, dict[str, object]]:
    dirfd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0))
    try:
        parts = _parts(relative)
        for part in parts[:-1]:
            nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0), dir_fd=dirfd)
            os.close(dirfd)
            dirfd = nxt
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0), dir_fd=dirfd)
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode):
                raise PublicationError(f"not a regular file: {relative}")
            chunks = []
            digest = hashlib.sha256()
            while True:
                block = os.read(fd, 1024 * 1024)
                if not block:
                    break
                chunks.append(block)
                digest.update(block)
            after = os.fstat(fd)
            identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
            if identity(before) != identity(after):
                raise PublicationError(f"file changed while reading: {relative}")
            data = b"".join(chunks)
            return data, {"mode": stat.S_IMODE(before.st_mode), "path": relative, "sha256": digest.hexdigest(), "size": len(data)}
        finally:
            os.close(fd)
    except OSError as exc:
        raise PublicationError(f"safe read refused: {relative}") from exc
    finally:
        os.close(dirfd)


def _read_external(path: Path) -> tuple[bytes, str]:
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise PublicationError("snapshot safe read refused") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise PublicationError("snapshot is not regular")
        data = b""
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            data += block
        after = os.fstat(fd)
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if len(data) != before.st_size or identity(before) != identity(after):
            raise PublicationError("snapshot changed while reading")
        return data, hashlib.sha256(data).hexdigest()
    finally:
        os.close(fd)


def _json(data: bytes, label: str) -> dict:
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicationError(f"invalid {label} JSON") from exc
    if not isinstance(value, dict):
        raise PublicationError(f"invalid {label} object")
    return value


def _publish(root: Path, relative: str, payload: bytes) -> None:
    parts = _parts(relative)
    dirfd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0))
    try:
        for part in parts[:-1]:
            nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0), dir_fd=dirfd)
            os.close(dirfd)
            dirfd = nxt
        fd = os.open(parts[-1], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0), 0o644, dir_fd=dirfd)
        try:
            view = memoryview(payload)
            while view:
                count = os.write(fd, view)
                if count <= 0:
                    raise PublicationError("short write")
                view = view[count:]
            os.fsync(fd)
        finally:
            os.close(fd)
        os.fsync(dirfd)
    except OSError as exc:
        raise PublicationError("exclusive publication refused") from exc
    finally:
        os.close(dirfd)


def publish(root: Path, snapshot_path: Path, incident_sha256: str = INCIDENT_SHA256,
            terminal_sha256: str = TERMINAL_SHA256, omitted: dict[str, str] = OMITTED) -> dict:
    root = _root(root)
    incident_data, incident_row = _read_under(root, INCIDENT_REL)
    terminal_data, terminal_row = _read_under(root, TERMINAL_REL)
    if incident_row["sha256"] != incident_sha256 or terminal_row["sha256"] != terminal_sha256:
        raise PublicationError("original record hash drift")
    incident = _json(incident_data, "incident")
    terminal = _json(terminal_data, "terminal")
    baseline = incident.get("legacy_baseline")
    if not isinstance(baseline, list) or len(baseline) != 37 or terminal.get("legacy_baseline") != baseline:
        raise PublicationError("original baseline mismatch")
    if terminal.get("acceptance_eligible") is not False:
        raise PublicationError("old run eligibility drift")
    baseline_map = {row.get("path"): row.get("sha256") for row in baseline if isinstance(row, dict)}
    if len(baseline_map) != 37:
        raise PublicationError("invalid original baseline rows")
    snapshot_data, snapshot_sha256 = _read_external(snapshot_path)
    snapshot = _json(snapshot_data, "snapshot")
    if len(snapshot) != 40 or set(snapshot) - set(baseline_map) != set(omitted) or set(baseline_map) - set(snapshot):
        raise PublicationError("snapshot set mismatch")
    if any(snapshot[path] != digest for path, digest in baseline_map.items()):
        raise PublicationError("snapshot baseline hash mismatch")
    if any(snapshot.get(path) != digest for path, digest in omitted.items()):
        raise PublicationError("snapshot singleton hash mismatch")
    rows = []
    for path in sorted(omitted, key=os.fsencode):
        _, row = _read_under(root, path)
        if row["sha256"] != snapshot[path]:
            raise PublicationError(f"singleton drift: {path}")
        rows.append(row)
    supplement = {
        "acceptance_eligible": False,
        "binding": {
            "incident": {"path": INCIDENT_REL, "sha256": incident_sha256},
            "terminal": {"path": TERMINAL_REL, "sha256": terminal_sha256},
        },
        "complete_count": 40,
        "correction_reason": REASON,
        "effect": "Corrects only the legacy baseline-scope omission; it does not alter the incident facts, and old-run acceptance_eligible remains false.",
        "omitted_rows": rows,
        "run_id": RUN_ID,
        "schema": "experiments7-g0-incident-baseline-supplement/v1",
        "snapshot": {"path": str(snapshot_path), "sha256": snapshot_sha256},
    }
    _publish(root, OUTPUT_REL, canonical_json(supplement))
    return supplement


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = publish(args.root, args.snapshot)
    except (OSError, PublicationError) as exc:
        print(str(exc), file=os.sys.stderr)
        return 2
    print(json.dumps({"complete_count": result["complete_count"], "output": OUTPUT_REL}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
