#!/usr/bin/env python3
"""Frozen V6 G0 child: the sole process that reads protected inputs.

This file is copied into a strict run before it is executed.  It intentionally
depends only on the adjacent frozen ``provider.py`` and the Python standard
library.  Its only output channel is stdout; the unconfined controller turns
that output into descriptor-bound V6 publications after the child exits.
"""
from __future__ import annotations

import argparse
import base64
import errno
import hashlib
import json
import os
import stat
import sys
import time
import types
from pathlib import Path
from typing import Any, Callable


SOURCE_SCHEMA = "experiments7-source-pre/v6"
PAPER_SCHEMA = "experiments7-paper-pre/v6"
RESULT_SCHEMA = "experiments7-g0-protected-result/v6"
CONFIG_SCHEMA = "experiments7-g0-protected-config/v6"
PRODUCER_BINDING_SCHEMA = "experiments7-cp0-producer-binding/v6"
PRODUCER_BINDING_MAX_BYTES = 1024 * 1024
PROTECTED_READ_ATTESTATION_SCHEMA = (
    "experiments7-cp0-protected-read-attestation/v6"
)
PUBLICATION_TRANSCRIPT_SCHEMA = "experiments7-publication-transcript/v6"
STAGE_ARTIFACT_SCHEMA = "experiments7-stage-artifact/v6"
ENVELOPE_EVIDENCE_SCHEMA = "experiments7-cp0-envelope-evidence/v6"
ACCEPTANCE_PATH = "reservation-acceptance.json"
ENVELOPE_EVIDENCE_PATH = "frozen/envelope_evidence/artifact.json"


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be an object")
    return value


def _exact_keys(
    value: object, expected: set[str], label: str
) -> dict[str, object]:
    row = _object(value, label)
    if set(row) != expected:
        raise RuntimeError(f"{label} schema is not exact")
    return row


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"{label} is not a sha256 digest")
    return value


def _transcript_time(transcript: dict[str, object], label: str) -> int:
    emitted = transcript.get("emitted_monotonic_ns")
    if type(emitted) is not int or emitted < 0:
        raise RuntimeError(f"{label} emitted time is invalid")
    return emitted


def _validate_producer_binding(
    producer_binding: object,
    config: dict[str, object],
    operation: str,
) -> tuple[dict[str, object], int, int]:
    if operation not in {"source-pre", "paper-pre"}:
        raise RuntimeError("protected read operation is invalid")
    binding = _exact_keys(
        producer_binding,
        {
            "schema",
            "acceptance_transcript",
            "envelope_artifact_evidence",
            "envelope_transcript",
            "envelope_payload",
        },
        "producer binding",
    )
    if binding["schema"] != PRODUCER_BINDING_SCHEMA:
        raise RuntimeError("producer binding schema differs")

    acceptance = _object(
        binding["acceptance_transcript"], "acceptance transcript"
    )
    envelope_evidence = _object(
        binding["envelope_artifact_evidence"], "envelope artifact evidence"
    )
    envelope_transcript = _object(
        binding["envelope_transcript"], "envelope transcript"
    )
    envelope_payload = _exact_keys(
        binding["envelope_payload"],
        {
            "schema",
            "sealed_run_id",
            "run_root",
            "acceptance_transcript_sha256",
        },
        "envelope payload",
    )

    run_id = config.get("sealed_run_id")
    run_root = config.get("run_root")
    acceptance_context = acceptance.get("context")
    envelope_context = envelope_transcript.get("context")
    if (
        acceptance.get("schema") != PUBLICATION_TRANSCRIPT_SCHEMA
        or acceptance.get("sealed_run_id") != run_id
        or acceptance.get("run_root") != run_root
        or acceptance.get("relative_path") != ACCEPTANCE_PATH
        or not isinstance(acceptance_context, dict)
    ):
        raise RuntimeError("acceptance transcript binding is invalid")
    if (
        envelope_transcript.get("schema") != PUBLICATION_TRANSCRIPT_SCHEMA
        or envelope_transcript.get("sealed_run_id") != run_id
        or envelope_transcript.get("run_root") != run_root
        or envelope_transcript.get("relative_path") != ENVELOPE_EVIDENCE_PATH
        or not isinstance(envelope_context, dict)
        or envelope_context != acceptance_context
    ):
        raise RuntimeError("envelope transcript context binding is invalid")

    acceptance_sha256 = sha256_bytes(canonical_json(acceptance))
    envelope_transcript_sha256 = sha256_bytes(
        canonical_json(envelope_transcript)
    )
    envelope_payload_raw = canonical_json(envelope_payload)
    if (
        envelope_payload["schema"] != ENVELOPE_EVIDENCE_SCHEMA
        or envelope_payload["sealed_run_id"] != run_id
        or envelope_payload["run_root"] != run_root
        or _sha256(
            envelope_payload["acceptance_transcript_sha256"],
            "envelope acceptance transcript",
        )
        != acceptance_sha256
    ):
        raise RuntimeError("envelope payload binding is invalid")
    if (
        envelope_evidence.get("schema") != STAGE_ARTIFACT_SCHEMA
        or envelope_evidence.get("relative_path") != ENVELOPE_EVIDENCE_PATH
        or envelope_evidence.get("artifact_type") != "regular"
        or _sha256(envelope_evidence.get("sha256"), "envelope artifact")
        != sha256_bytes(envelope_payload_raw)
        or envelope_evidence.get("bytes") != len(envelope_payload_raw)
        or _sha256(
            envelope_evidence.get("publication_transcript_sha256"),
            "envelope publication transcript",
        )
        != envelope_transcript_sha256
    ):
        raise RuntimeError("envelope artifact evidence binding is invalid")

    acceptance_emitted = _transcript_time(
        acceptance, "acceptance transcript"
    )
    envelope_emitted = _transcript_time(
        envelope_transcript, "envelope transcript"
    )
    return ({
        "sealed_run_id": run_id,
        "run_root": run_root,
        "acceptance_transcript_sha256": acceptance_sha256,
        "envelope_artifact_evidence_sha256": sha256_bytes(
            canonical_json(envelope_evidence)
        ),
        "envelope_transcript_sha256": envelope_transcript_sha256,
        "envelope_context": json.loads(canonical_json(envelope_context)),
    }, acceptance_emitted, envelope_emitted)


def _attest_protected_read(
    validated_binding: tuple[dict[str, object], int, int],
    operation: str,
) -> dict[str, object]:
    fields, acceptance_emitted, envelope_emitted = validated_binding
    started = time.monotonic_ns()
    if started <= acceptance_emitted or started <= envelope_emitted:
        raise RuntimeError("protected read did not start strictly after its gate")
    return {
        "schema": PROTECTED_READ_ATTESTATION_SCHEMA,
        **fields,
        "operation": operation,
        "started_monotonic_ns": started,
    }


def _begin_protected_read(
    producer_binding: object,
    config: dict[str, object],
    operation: str,
) -> dict[str, object]:
    return _attest_protected_read(
        _validate_producer_binding(producer_binding, config, operation),
        operation,
    )


def _sha256_path(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _identity(value: os.stat_result, *, require_regular: bool = False) -> dict[str, int | str]:
    if require_regular and not stat.S_ISREG(value.st_mode):
        raise RuntimeError("expected a regular file")
    return {
        "file_type": "regular" if stat.S_ISREG(value.st_mode) else "directory",
        "st_dev": value.st_dev,
        "st_ino": value.st_ino,
        "mode": stat.S_IMODE(value.st_mode),
        "size": value.st_size,
        "mtime_ns": value.st_mtime_ns,
    }


def _stable(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        stat.S_IMODE(value.st_mode),
        value.st_size,
        value.st_mtime_ns,
    )


def _hash_fd(fd: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    while True:
        block = os.read(fd, 1024 * 1024)
        if not block:
            return digest.hexdigest(), total
        digest.update(block)
        total += len(block)


def _load_canonical_json(path: str, label: str) -> object:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    if canonical_json(value) != raw:
        raise RuntimeError(f"{label} is not canonical JSON")
    return value


def _load_canonical_json_stdin(label: str) -> object:
    raw = sys.stdin.buffer.read(PRODUCER_BINDING_MAX_BYTES + 1)
    if not raw:
        raise RuntimeError(f"{label} is missing from stdin")
    if len(raw) > PRODUCER_BINDING_MAX_BYTES:
        raise RuntimeError(f"{label} exceeds the stdin size limit")
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"{label} is not valid JSON") from exc
    if canonical_json(value) != raw:
        raise RuntimeError(f"{label} is not canonical JSON")
    return value


def _load_config(path: str) -> dict[str, object]:
    value = _load_canonical_json(path, "frozen config")
    if not isinstance(value, dict):
        raise RuntimeError("frozen config must be an object")
    required = {
        "schema",
        "sealed_run_id",
        "run_root",
        "provider_path",
        "provider_sha256",
        "g0_executable_sha256",
        "source_roots",
        "paper_path",
    }
    if set(value) != required or value["schema"] != CONFIG_SCHEMA:
        raise RuntimeError("frozen config schema differs")
    if not isinstance(value["source_roots"], list) or len(value["source_roots"]) != 3:
        raise RuntimeError("frozen config source roots differ")
    root_ids = {row.get("root_id") for row in value["source_roots"] if isinstance(row, dict)}
    if root_ids != {"experiments4", "experiments5", "experiments6"}:
        raise RuntimeError("frozen config root IDs differ")
    if not all(
        isinstance(value[key], str) and value[key]
        for key in (
            "sealed_run_id",
            "run_root",
            "provider_path",
            "provider_sha256",
            "g0_executable_sha256",
            "paper_path",
        )
    ):
        raise RuntimeError("frozen config has invalid strings")
    for row in value["source_roots"]:
        if not isinstance(row, dict) or set(row) != {"root_id", "path"}:
            raise RuntimeError("frozen config root schema differs")
        if not isinstance(row["path"], str) or not os.path.isabs(row["path"]):
            raise RuntimeError("frozen config root path is invalid")
    return value


def _read_held_fd(fd: int, label: str, limit: int = 16 * 1024 * 1024) -> bytes:
    if type(fd) is not int or fd < 0:
        raise RuntimeError(f"{label} descriptor is invalid")
    before = os.fstat(fd)
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError(f"{label} descriptor is not regular")
    chunks: list[bytes] = []
    total = 0
    offset = 0
    while True:
        block = os.pread(fd, 1024 * 1024, offset)
        if not block:
            break
        offset += len(block)
        total += len(block)
        if total > limit:
            raise RuntimeError(f"{label} descriptor exceeds its limit")
        chunks.append(block)
    after = os.fstat(fd)
    if _identity(before) != _identity(after) or total != after.st_size:
        raise RuntimeError(f"{label} descriptor changed while loading")
    return b"".join(chunks)


def _load_provider(
    config: dict[str, object],
    provider_fd: int,
    *,
    race_hook: Callable[[], None] | None = None,
) -> Any:
    provider_path = str(config["provider_path"])
    expected_provider_path = (
        f"{config['run_root']}/frozen/provider/provider.py"
    )
    if provider_path != expected_provider_path:
        raise RuntimeError("frozen provider logical path differs")
    provider_bytes = _read_held_fd(provider_fd, "frozen provider")
    if sha256_bytes(provider_bytes) != config["provider_sha256"]:
        raise RuntimeError("frozen provider bytes differ")
    if _sha256_path(os.path.abspath(__file__)) != config["g0_executable_sha256"]:
        raise RuntimeError("frozen G0 executable bytes differ")
    if race_hook is not None:
        race_hook()
    try:
        source = provider_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("frozen provider is not UTF-8 Python") from exc
    module = types.ModuleType("exp7_frozen_provider")
    module.__file__ = provider_path
    code = compile(source, provider_path, "exec", dont_inherit=True)
    exec(code, module.__dict__)
    return module


def _require_frozen_invocation() -> None:
    if not sys.dont_write_bytecode or os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise RuntimeError("G0 protected child requires python -B and PYTHONDONTWRITEBYTECODE=1")


def _source_record(root_id: str, relative: bytes, value: os.stat_result, digest: str) -> dict[str, object]:
    try:
        relative.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("protected source path is not UTF-8") from exc
    record_id = hashlib.sha256(
        b"experiments7-source-pre/v6\0" + root_id.encode("utf-8") + b"\0" + relative
    ).hexdigest()
    return {
        "schema": SOURCE_SCHEMA,
        "record_id": f"source:{record_id}",
        "root_id": root_id,
        "relative_path_b64": base64.b64encode(relative).decode("ascii"),
        "type": "regular",
        "sha256": digest,
        "size": value.st_size,
        "descriptor_identity": _identity(value, require_regular=True),
    }


def _scan_directory(root_id: str, directory_fd: int, relative: bytes, records: list[dict[str, object]]) -> None:
    before = os.fstat(directory_fd)
    if not stat.S_ISDIR(before.st_mode):
        raise RuntimeError("protected scan expected a directory")
    for name in sorted(os.listdir(directory_fd), key=os.fsencode):
        name_bytes = os.fsencode(name)
        child_relative = name_bytes if not relative else relative + b"/" + name_bytes
        first = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISREG(first.st_mode):
            fd = os.open(
                name,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=directory_fd,
            )
            try:
                opened = os.fstat(fd)
                if _stable(first) != _stable(opened):
                    raise RuntimeError("protected regular file changed before hashing")
                digest, total = _hash_fd(fd)
                final = os.fstat(fd)
                if _stable(opened) != _stable(final) or total != final.st_size:
                    raise RuntimeError("protected regular file changed while hashing")
            finally:
                os.close(fd)
            records.append(_source_record(root_id, child_relative, final, digest))
        elif stat.S_ISDIR(first.st_mode):
            fd = os.open(
                name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=directory_fd,
            )
            try:
                opened = os.fstat(fd)
                if _stable(first) != _stable(opened):
                    raise RuntimeError("protected directory changed before enumeration")
                _scan_directory(root_id, fd, child_relative, records)
                if _stable(opened) != _stable(os.fstat(fd)):
                    raise RuntimeError("protected directory changed while enumerating")
            finally:
                os.close(fd)
    if _stable(before) != _stable(os.fstat(directory_fd)):
        raise RuntimeError("protected directory changed during enumeration")


def _build_source_records(config: dict[str, object]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for item in sorted(config["source_roots"], key=lambda row: str(row["root_id"])):
        assert isinstance(item, dict)
        root_id = str(item["root_id"])
        path = str(item["path"])
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            root_before = os.fstat(fd)
            _scan_directory(root_id, fd, b"", records)
            if _stable(root_before) != _stable(os.fstat(fd)):
                raise RuntimeError("protected source root changed during enumeration")
        finally:
            os.close(fd)
    records.sort(
        key=lambda row: (
            str(row["root_id"]),
            base64.b64decode(str(row["relative_path_b64"]), validate=True),
        )
    )
    if not records:
        raise RuntimeError("protected source roots contained no regular files")
    return records


def _build_paper_record(config: dict[str, object], provider: Any) -> dict[str, object]:
    path = str(config["paper_path"])
    parent = os.path.dirname(path)
    name = os.path.basename(path)
    parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        parent_before = os.fstat(parent_fd)
        first = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISREG(first.st_mode):
            raise RuntimeError("protected paper is not a regular file")
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent_fd)
        try:
            opened = os.fstat(fd)
            if _stable(first) != _stable(opened):
                raise RuntimeError("protected paper changed before hashing")
            digest, total = _hash_fd(fd)
            final = os.fstat(fd)
            if _stable(opened) != _stable(final) or total != final.st_size:
                raise RuntimeError("protected paper changed while hashing")
        finally:
            os.close(fd)
        if _stable(parent_before) != _stable(os.fstat(parent_fd)):
            raise RuntimeError("protected paper parent changed during hashing")
    finally:
        os.close(parent_fd)
    return {
        "schema": PAPER_SCHEMA,
        "sealed_run_id": config["sealed_run_id"],
        "run_root": config["run_root"],
        "paper_path": path,
        "sha256": digest,
        "size": total,
        "descriptor_identity": _identity(final, require_regular=True),
        "parent_identity": _identity(parent_before),
        "provider": provider.provider_identity(),
    }


def _denial_probe(path: str) -> dict[str, object]:
    flags = os.O_WRONLY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        if exc.errno not in {errno.EACCES, errno.EPERM}:
            raise RuntimeError(f"write probe returned unexpected errno: {exc.errno}") from exc
        return {
            "operation": "open(O_WRONLY|O_NOFOLLOW)",
            "target_path": path,
            "denied": True,
            "errno": exc.errno,
        }
    else:
        os.close(descriptor)
        raise RuntimeError("protected write-capable open unexpectedly succeeded")


def _emit(value: object) -> None:
    sys.stdout.buffer.write(canonical_json(value))


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", required=True)
    parser.add_argument("--provider-fd", required=True, type=int)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("verify-envelope", "source-pre", "paper-pre"),
    )
    parser.add_argument("--producer-binding-stdin", action="store_true")
    args = parser.parse_args()
    _require_frozen_invocation()
    config = _load_config(os.path.abspath(args.config))
    producer_binding: object | None = None
    validated_binding: tuple[dict[str, object], int, int] | None = None
    if args.mode == "verify-envelope":
        if args.producer_binding_stdin:
            raise RuntimeError(
                "verify-envelope must not accept producer binding stdin"
            )
    else:
        if not args.producer_binding_stdin:
            raise RuntimeError(
                "protected source/paper mode requires producer binding stdin"
            )
        producer_binding = _load_canonical_json_stdin("producer binding")
        validated_binding = _validate_producer_binding(
            producer_binding, config, str(args.mode)
        )
    provider = _load_provider(config, args.provider_fd)
    provider_evidence = provider.apply_readonly_envelope(())
    if args.mode == "verify-envelope":
        _emit(
            {
                "schema": RESULT_SCHEMA,
                "mode": args.mode,
                "sealed_run_id": config["sealed_run_id"],
                "run_root": config["run_root"],
                "python_dont_write_bytecode": sys.dont_write_bytecode,
                "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
                "provider_evidence": provider_evidence,
                "paper_write_probe": _denial_probe(str(config["paper_path"])),
            }
        )
        return 0

    operation = str(args.mode)
    if validated_binding is None:
        raise RuntimeError("protected producer binding was not validated")
    attestation = _attest_protected_read(
        validated_binding, operation
    )
    paper_write_probe = _denial_probe(str(config["paper_path"]))
    if args.mode == "source-pre":
        _emit(
            {
                "schema": RESULT_SCHEMA,
                "mode": args.mode,
                "sealed_run_id": config["sealed_run_id"],
                "run_root": config["run_root"],
                "provider_evidence": provider_evidence,
                "paper_write_probe": paper_write_probe,
                "producer_attestation": attestation,
                "records": _build_source_records(config),
            }
        )
        return 0
    _emit(
        {
            "schema": RESULT_SCHEMA,
            "mode": args.mode,
            "sealed_run_id": config["sealed_run_id"],
            "run_root": config["run_root"],
            "provider_evidence": provider_evidence,
            "paper_write_probe": paper_write_probe,
            "producer_attestation": attestation,
            "paper": _build_paper_record(config, provider),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
