#!/usr/bin/env python3
"""Generate or check deterministic G2 facade metadata without protected reads."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from facade.metadata import construct_metadata  # noqa: E402
from facade.selection import canonical_bytes, sha256_bytes  # noqa: E402


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short write")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    payloads = construct_metadata(ROOT)
    mismatches = []
    artifacts = []
    for relative, payload in sorted(payloads.items()):
        path = ROOT / relative
        if args.check:
            if not path.is_file() or path.is_symlink() or path.read_bytes() != payload:
                mismatches.append(relative)
        else:
            _write_new(path, payload)
        artifacts.append({"path": relative, "sha256": sha256_bytes(payload), "bytes": len(payload)})
    report = {
        "schema": "experiments7-g2-metadata-generation/v1",
        "state": "BLOCKED" if mismatches else "PASS",
        "mode": "check" if args.check else "create",
        "artifacts": artifacts,
        "mismatches": mismatches,
        "protected_reads": 0,
    }
    print(canonical_bytes(report).decode("utf-8"), end="")
    return 2 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
