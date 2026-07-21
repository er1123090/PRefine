"""CLI helpers; external transcripts are never written inside a strict run."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from strict_run import (  # noqa: E402
    ExternalDocument,
    ExternalOutputReservation,
    canonical_json,
    load_pre_reservation_document,
)
from strict_run.canonical import fail, validate_absolute_path_text  # noqa: E402
from strict_run.filesystem import open_absolute_directory  # noqa: E402


def load_canonical_json(path: str) -> object:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    if canonical_json(value) != raw:
        fail("NONCANONICAL_JSON", "external input must be canonical JSON")
    return value


def open_pre_reservation_json(
    path: str,
    *,
    strict_parent: str,
    sealed_run_id: str,
    run_root: str,
) -> ExternalDocument:
    return load_pre_reservation_document(
        path,
        strict_parent=strict_parent,
        sealed_run_id=sealed_run_id,
        run_root=run_root,
    )


def preflight_external_output(
    path: str,
    *,
    strict_parent: str,
    sealed_run_id: str,
    run_root: str,
) -> ExternalOutputReservation:
    return ExternalOutputReservation.create(
        path,
        strict_parent=strict_parent,
        sealed_run_id=sealed_run_id,
        run_root=run_root,
    )


def preflight_existing_external_output(
    path: str,
    *,
    strict_parent: str,
    sealed_run_id: str,
    run_root: str,
) -> ExternalOutputReservation:
    return ExternalOutputReservation.create_for_existing_run(
        path,
        strict_parent=strict_parent,
        sealed_run_id=sealed_run_id,
        run_root=run_root,
    )


def write_external_no_replace(path: str, value: object, run_root: str) -> None:
    path = validate_absolute_path_text(path, "external output")
    run_root = validate_absolute_path_text(run_root, "run_root")
    if os.path.commonpath((path, run_root)) == run_root:
        fail("TRANSCRIPT_IN_RUN_TREE", "external transcript cannot be written in strict root")
    directory, name = os.path.split(path)
    directory_fd = open_absolute_directory(directory)
    try:
        fd = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o444,
            dir_fd=directory_fd,
        )
        try:
            payload = canonical_json(value)
            view = memoryview(payload)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    fail("SHORT_WRITE", "external output write made no progress")
                view = view[written:]
            os.fsync(fd)
        finally:
            os.close(fd)
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def print_summary(value: object) -> None:
    sys.stdout.buffer.write(canonical_json(value))
