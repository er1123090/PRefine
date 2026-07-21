#!/usr/bin/env python3
"""Canonical descriptor-based source and paper manifest builder."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
import sys
import time
from collections.abc import Iterable

SCHEMA = "experiments7-canonical-manifest/v1"
PAPER_SCHEMA = "experiments7-paper-baseline/v1"
CHUNK_SIZE = 8 * 1024 * 1024
PRODUCER_BINDING_SCHEMA = "experiments7-cp0-producer-binding/v6"
PROTECTED_READ_ATTESTATION_SCHEMA = (
    "experiments7-cp0-protected-read-attestation/v6"
)
PUBLICATION_TRANSCRIPT_SCHEMA = "experiments7-publication-transcript/v6"
STAGE_ARTIFACT_SCHEMA = "experiments7-stage-artifact/v6"
ENVELOPE_EVIDENCE_SCHEMA = "experiments7-cp0-envelope-evidence/v6"
ACCEPTANCE_PATH = "reservation-acceptance.json"
ENVELOPE_EVIDENCE_PATH = "frozen/envelope_evidence/artifact.json"


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


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


def validate_producer_binding(
    producer_binding: object, config: dict[str, object]
) -> dict[str, object]:
    run_id = config.get("sealed_run_id")
    run_root = config.get("run_root")
    if (
        not isinstance(run_id, str)
        or not run_id
        or not isinstance(run_root, str)
        or not os.path.isabs(run_root)
    ):
        raise RuntimeError("configured run identity is invalid")
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
    return {
        "sealed_run_id": run_id,
        "run_root": run_root,
        "acceptance_emitted_monotonic_ns": _transcript_time(
            acceptance, "acceptance transcript"
        ),
        "envelope_emitted_monotonic_ns": _transcript_time(
            envelope_transcript, "envelope transcript"
        ),
        "acceptance_transcript_sha256": acceptance_sha256,
        "envelope_artifact_evidence_sha256": sha256_bytes(
            canonical_json(envelope_evidence)
        ),
        "envelope_transcript_sha256": envelope_transcript_sha256,
        "envelope_context": json.loads(canonical_json(envelope_context)),
    }


def _begin_attested_protected_read(
    producer_binding: object,
    config: dict[str, object],
    operation: str,
) -> dict[str, object]:
    if operation not in {"source-pre", "paper-pre"}:
        raise RuntimeError("protected read operation is invalid")
    validated = validate_producer_binding(producer_binding, config)
    started = time.monotonic_ns()
    if (
        started <= validated["acceptance_emitted_monotonic_ns"]
        or started <= validated["envelope_emitted_monotonic_ns"]
    ):
        raise RuntimeError("protected read did not start strictly after its gate")
    return {
        "schema": PROTECTED_READ_ATTESTATION_SCHEMA,
        "sealed_run_id": validated["sealed_run_id"],
        "run_root": validated["run_root"],
        "operation": operation,
        "started_monotonic_ns": started,
        "acceptance_transcript_sha256": validated[
            "acceptance_transcript_sha256"
        ],
        "envelope_artifact_evidence_sha256": validated[
            "envelope_artifact_evidence_sha256"
        ],
        "envelope_transcript_sha256": validated[
            "envelope_transcript_sha256"
        ],
        "envelope_context": validated["envelope_context"],
    }


def _kind(mode: int) -> str:
    if stat.S_ISREG(mode): return "regular"
    if stat.S_ISDIR(mode): return "directory"
    if stat.S_ISLNK(mode): return "symlink"
    if stat.S_ISFIFO(mode): return "fifo"
    if stat.S_ISSOCK(mode): return "socket"
    if stat.S_ISCHR(mode): return "character_device"
    if stat.S_ISBLK(mode): return "block_device"
    return "unknown"


def selected_metadata(st: os.stat_result) -> dict[str, object]:
    return {
        "type": _kind(st.st_mode),
        "mode": stat.S_IMODE(st.st_mode),
        "uid": st.st_uid,
        "gid": st.st_gid,
        "nlink": st.st_nlink,
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "ctime_ns": st.st_ctime_ns,
        "dev": st.st_dev,
        "inode": st.st_ino,
    }


def stable_tuple(st: os.stat_result) -> tuple[object, ...]:
    metadata = selected_metadata(st)
    return tuple(metadata[key] for key in sorted(metadata))


def _read_hash(fd: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    while True:
        block = os.read(fd, CHUNK_SIZE)
        if not block:
            break
        digest.update(block)
        total += len(block)
    return digest.hexdigest(), total


def _record_id(root_id: str, relative: bytes) -> str:
    return hashlib.sha256(root_id.encode("utf-8") + b"\0" + relative).hexdigest()


def _record(root_id: str, relative: bytes, st: os.stat_result) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "record_id": _record_id(root_id, relative),
        "root_id": root_id,
        "relative_path_b64": base64.b64encode(relative).decode("ascii"),
        **selected_metadata(st),
    }


def _scan_directory(root_id: str, directory_fd: int, relative: bytes, records: list[dict[str, object]]) -> None:
    before = os.fstat(directory_fd)
    names = sorted(os.listdir(directory_fd), key=os.fsencode)
    for name in names:
        name_bytes = os.fsencode(name)
        child_relative = name_bytes if not relative else relative + b"/" + name_bytes
        first = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        item = _record(root_id, child_relative, first)
        item_type = item["type"]
        if item_type == "regular":
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory_fd)
            try:
                opened = os.fstat(fd)
                if stable_tuple(first) != stable_tuple(opened):
                    raise RuntimeError(f"regular-file substitution: {root_id}:{child_relative!r}")
                digest, total = _read_hash(fd)
                final = os.fstat(fd)
                if stable_tuple(opened) != stable_tuple(final) or total != final.st_size:
                    raise RuntimeError(f"regular-file changed while hashing: {root_id}:{child_relative!r}")
                item["sha256"] = digest
            finally:
                os.close(fd)
        elif item_type == "directory":
            fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory_fd)
            try:
                opened = os.fstat(fd)
                if stable_tuple(first) != stable_tuple(opened):
                    raise RuntimeError(f"directory substitution: {root_id}:{child_relative!r}")
                _scan_directory(root_id, fd, child_relative, records)
                final = os.fstat(fd)
                if stable_tuple(opened) != stable_tuple(final):
                    raise RuntimeError(f"directory changed while scanning: {root_id}:{child_relative!r}")
            finally:
                os.close(fd)
        elif item_type == "symlink":
            target = os.readlink(name, dir_fd=directory_fd)
            item["link_target_b64"] = base64.b64encode(os.fsencode(target)).decode("ascii")
            final = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if stable_tuple(first) != stable_tuple(final):
                raise RuntimeError(f"symlink changed while reading: {root_id}:{child_relative!r}")
        records.append(item)
    after = os.fstat(directory_fd)
    if stable_tuple(before) != stable_tuple(after):
        raise RuntimeError(f"directory changed during enumeration: {root_id}:{relative!r}")


def _build_source_records(roots: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    root_ids: set[str] = set()
    for root in sorted(roots, key=lambda value: str(value["root_id"])):
        root_id = str(root["root_id"])
        path = str(root["path"])
        if root_id in root_ids:
            raise RuntimeError(f"duplicate root id: {root_id}")
        root_ids.add(root_id)
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            root_st = os.fstat(fd)
            expected = root["binding"]
            if (root_st.st_dev, root_st.st_ino) != (expected["dev"], expected["inode"]):
                raise RuntimeError(f"root binding mismatch: {root_id}")
            records.append(_record(root_id, b"", root_st))
            _scan_directory(root_id, fd, b"", records)
            if stable_tuple(root_st) != stable_tuple(os.fstat(fd)):
                raise RuntimeError(f"root changed during scan: {root_id}")
        finally:
            os.close(fd)
    records.sort(key=lambda item: (str(item["root_id"]).encode("utf-8"), base64.b64decode(str(item["relative_path_b64"]))))
    return records


def build_attested_source_records(
    config: dict[str, object], producer_binding: object
) -> tuple[list[dict[str, object]], dict[str, object]]:
    attestation = _begin_attested_protected_read(
        producer_binding, config, "source-pre"
    )
    return _build_source_records(config["protected_sources"]), attestation


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        count = os.write(fd, view)
        if count <= 0:
            raise OSError("short write")
        view = view[count:]


def atomic_publish(path: str, chunks: Iterable[bytes]) -> dict[str, object]:
    directory = os.path.dirname(path)
    basename = os.path.basename(path)
    if not os.path.isabs(path) or not basename or basename in (".", ".."):
        raise RuntimeError(f"invalid publication path: {path}")
    directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    temp = f".{basename}.tmp.{os.getpid()}"
    fd = -1
    try:
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o644, dir_fd=directory_fd)
        digest = hashlib.sha256()
        size = 0
        for chunk in chunks:
            _write_all(fd, chunk)
            digest.update(chunk)
            size += len(chunk)
        os.fsync(fd)
        temp_st = os.fstat(fd)
        os.close(fd)
        fd = -1
        os.link(temp, basename, src_dir_fd=directory_fd, dst_dir_fd=directory_fd, follow_symlinks=False)
        os.unlink(temp, dir_fd=directory_fd)
        os.fsync(directory_fd)
        final_st = os.stat(basename, dir_fd=directory_fd, follow_symlinks=False)
        if (final_st.st_dev, final_st.st_ino) != (temp_st.st_dev, temp_st.st_ino):
            raise RuntimeError(f"publication identity mismatch: {path}")
        return {"path": path, "sha256": digest.hexdigest(), "size": size, "dev": final_st.st_dev, "inode": final_st.st_ino}
    except Exception:
        if fd >= 0:
            os.close(fd)
        try:
            os.unlink(temp, dir_fd=directory_fd)
        except OSError:
            pass
        raise
    finally:
        os.close(directory_fd)


def publish_source_manifest(
    config: dict[str, object],
    output: str,
    producer_binding: object,
) -> dict[str, object]:
    records, attestation = build_attested_source_records(
        config, producer_binding
    )
    result = atomic_publish(output, (canonical_json(record) for record in records))
    result["record_count"] = len(records)
    result["metadata_scope"] = config["metadata_scope"]
    result["producer_attestation"] = attestation
    return result


def _namespace_entries(directory_fd: int) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for name in sorted(os.listdir(directory_fd), key=os.fsencode):
        st = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        entries.append({"name_b64": base64.b64encode(os.fsencode(name)).decode("ascii"), **selected_metadata(st)})
    return entries


def _build_paper_record(config: dict[str, object], bundle_lock_sha256: str) -> dict[str, object]:
    paper = config["paper"]
    path = str(paper["path"])
    parent = os.path.dirname(path)
    basename = os.path.basename(path)
    parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        parent_st = os.fstat(parent_fd)
        if (parent_st.st_dev, parent_st.st_ino) != (paper["parent_binding"]["dev"], paper["parent_binding"]["inode"]):
            raise RuntimeError("_paper binding mismatch")
        namespace_before = _namespace_entries(parent_fd)
        first = os.stat(basename, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISREG(first.st_mode):
            raise RuntimeError("paper is not regular")
        if (first.st_dev, first.st_ino) != (paper["binding"]["dev"], paper["binding"]["inode"]):
            raise RuntimeError("paper binding mismatch")
        fd = os.open(basename, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent_fd)
        try:
            opened = os.fstat(fd)
            if stable_tuple(first) != stable_tuple(opened):
                raise RuntimeError("paper substitution before hashing")
            digest, total = _read_hash(fd)
            final = os.fstat(fd)
            if stable_tuple(opened) != stable_tuple(final) or total != final.st_size:
                raise RuntimeError("paper changed while hashing")
        finally:
            os.close(fd)
        namespace_after = _namespace_entries(parent_fd)
        if namespace_before != namespace_after or stable_tuple(parent_st) != stable_tuple(os.fstat(parent_fd)):
            raise RuntimeError("foreign _paper namespace changed during baseline")
        return {
            "schema": PAPER_SCHEMA,
            "path": path,
            "ownership": "preserved_foreign_readonly",
            "sha256": digest,
            "bytes_read": total,
            "metadata": selected_metadata(final),
            "paper_directory": {"path": parent, "metadata": selected_metadata(parent_st), "entries": namespace_after},
            "bundle_lock_sha256": bundle_lock_sha256,
            "metadata_scope": config["metadata_scope"],
        }
    finally:
        os.close(parent_fd)


def build_attested_paper_record(
    config: dict[str, object],
    bundle_lock_sha256: str,
    producer_binding: object,
) -> tuple[dict[str, object], dict[str, object]]:
    attestation = _begin_attested_protected_read(
        producer_binding, config, "paper-pre"
    )
    return _build_paper_record(config, bundle_lock_sha256), attestation


def publish_paper_manifest(
    config: dict[str, object],
    bundle_lock_sha256: str,
    output: str,
    producer_binding: object,
) -> dict[str, object]:
    record, attestation = build_attested_paper_record(
        config, bundle_lock_sha256, producer_binding
    )
    result = atomic_publish(output, [canonical_json(record)])
    result["paper_sha256"] = record["sha256"]
    result["producer_attestation"] = attestation
    return result


def load_json(path: str) -> object:
    with open(path, "rb") as handle:
        return json.load(handle)


def load_canonical_json(path: str, label: str) -> object:
    with open(path, "rb") as handle:
        raw = handle.read()
    value = json.loads(raw)
    if canonical_json(value) != raw:
        raise RuntimeError(f"{label} is not canonical JSON")
    return value


def main() -> None:
    print(
        json.dumps(
            {
                "status": "BLOCKED",
                "reason": (
                    "legacy manifest CLI is disabled; protected reads require "
                    "the strict V6 G0 child with stdin-bound producer evidence"
                ),
            },
            sort_keys=True,
        ),
        file=sys.stderr,
    )
    raise SystemExit(2)


if __name__ == "__main__":
    main()
