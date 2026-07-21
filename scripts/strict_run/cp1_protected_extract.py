#!/usr/bin/env python3
"""Read a CP0-bound paper through the frozen readonly provider.

The program deliberately has no output path: it emits one canonical JSON
document to stdout after the Landlock/seccomp envelope is irreversible.  A
separate unconfined controller may persist that document only in its declared
external no-replace evidence location.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import importlib.util
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any


SCHEMA = "experiments7-cp1-protected-paper-extraction/v6"
MAX_PAGES = 256


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def require_hex_digest(value: str, label: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise RuntimeError(f"{label} must be a lowercase SHA-256 digest")
    return value


def require_frozen_invocation() -> None:
    if not sys.dont_write_bytecode or os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise RuntimeError("CP1 protected extractor requires python -B and PYTHONDONTWRITEBYTECODE=1")


def load_provider(path: Path, expected_sha256: str) -> Any:
    if sha256_file(path) != expected_sha256:
        raise RuntimeError("frozen provider bytes differ")
    specification = importlib.util.spec_from_file_location("exp7_cp1_frozen_provider", path)
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot import frozen provider")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def stable_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        stat.S_IMODE(value.st_mode),
        value.st_size,
        value.st_mtime_ns,
    )


def paper_identity(path: Path) -> dict[str, int | str]:
    parent = path.parent
    name = path.name
    parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        parent_before = os.fstat(parent_fd)
        first = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISREG(first.st_mode):
            raise RuntimeError("paper is not a regular file")
        paper_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent_fd)
        try:
            opened = os.fstat(paper_fd)
            if stable_identity(first) != stable_identity(opened):
                raise RuntimeError("paper changed before extraction")
            digest = hashlib.sha256()
            total = 0
            while True:
                block = os.read(paper_fd, 1024 * 1024)
                if not block:
                    break
                digest.update(block)
                total += len(block)
            final = os.fstat(paper_fd)
            if stable_identity(opened) != stable_identity(final) or total != final.st_size:
                raise RuntimeError("paper changed while hashing")
        finally:
            os.close(paper_fd)
        if stable_identity(parent_before) != stable_identity(os.fstat(parent_fd)):
            raise RuntimeError("paper parent changed during extraction")
    finally:
        os.close(parent_fd)
    return {
        "sha256": digest.hexdigest(),
        "bytes": total,
        "st_dev": final.st_dev,
        "st_ino": final.st_ino,
        "mode": stat.S_IMODE(final.st_mode),
        "mtime_ns": final.st_mtime_ns,
    }


def denial_probe(path: Path) -> dict[str, object]:
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError as exc:
        if exc.errno not in {errno.EACCES, errno.EPERM}:
            raise RuntimeError(f"paper write probe returned unexpected errno: {exc.errno}") from exc
        return {"operation": "open(O_WRONLY|O_NOFOLLOW)", "denied": True, "errno": exc.errno}
    else:
        os.close(descriptor)
        raise RuntimeError("paper write-capable open unexpectedly succeeded")


def extract_pages(path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    import pypdf

    package_path = Path(pypdf.__file__).resolve()
    parser = {
        "module": "pypdf",
        "version": str(pypdf.__version__),
        "path": str(package_path),
        "sha256": sha256_file(package_path),
    }
    reader = pypdf.PdfReader(str(path), strict=True)
    if not 0 < len(reader.pages) <= MAX_PAGES:
        raise RuntimeError("paper page count is outside the bounded CP1 contract")
    pages: list[dict[str, object]] = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text(extraction_mode="layout")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError(f"paper page {index} has no extractable layout text")
        pages.append(
            {
                "page": index,
                "layout_text": text,
                "layout_text_sha256": sha256_bytes(text.encode("utf-8")),
            }
        )
    return parser, pages


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--sealed-run-id", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--paper-path", required=True)
    parser.add_argument("--expected-paper-sha256", required=True)
    parser.add_argument("--provider-path", required=True)
    parser.add_argument("--provider-sha256", required=True)
    args = parser.parse_args()
    require_frozen_invocation()
    paper_path = Path(args.paper_path)
    provider_path = Path(args.provider_path)
    expected_paper_sha256 = require_hex_digest(args.expected_paper_sha256, "paper digest")
    provider_sha256 = require_hex_digest(args.provider_sha256, "provider digest")
    if not paper_path.is_absolute() or not provider_path.is_absolute():
        raise RuntimeError("protected paths must be absolute")
    provider = load_provider(provider_path, provider_sha256)
    provider_evidence = provider.apply_readonly_envelope(())
    write_probe = denial_probe(paper_path)
    paper = paper_identity(paper_path)
    if paper["sha256"] != expected_paper_sha256:
        raise RuntimeError("paper bytes differ from CP0 paper-pre binding")
    parser_evidence, pages = extract_pages(paper_path)
    sys.stdout.buffer.write(
        canonical_json(
            {
                "schema": SCHEMA,
                "sealed_run_id": args.sealed_run_id,
                "run_root": args.run_root,
                "paper": paper,
                "provider": provider_evidence,
                "paper_write_probe": write_probe,
                "parser": parser_evidence,
                "page_count": len(pages),
                "pages": pages,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
