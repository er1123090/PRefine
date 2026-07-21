"""No-replace publication and completed out-of-tree transcript validation."""
from __future__ import annotations

import fcntl
import hashlib
import os
import re
import secrets
import stat
import time
from dataclasses import dataclass

from .canonical import (
    canonical_json,
    fail,
    require_exact_keys,
    require_sha256,
    sha256_bytes,
    validate_relative_path,
    validate_sealed_run_id,
)
from .filesystem import (
    RunDescriptorBinding,
    assert_path_matches_fd,
    descriptor_evidence,
    exists_relative,
    hash_regular_at,
    mount_identity,
    open_bound_run_handle,
    open_parent_directory,
    open_run_handle,
    stable_identity,
    validate_run_root_text,
)
from .writer_policy import WriterIdentity, WriterPolicy


TRANSCRIPT_SCHEMA = "experiments7-publication-transcript/v6"
EVIDENCE_SCHEMA = "experiments7-stage-artifact/v6"


@dataclass(frozen=True)
class ArtifactEvidence:
    relative_path: str
    artifact_type: str
    sha256: str | None
    size: int | None
    identity: dict[str, object]
    writer: WriterIdentity
    writer_rule_id: str
    publication_transcript_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": EVIDENCE_SCHEMA,
            "relative_path": self.relative_path,
            "artifact_type": self.artifact_type,
            "sha256": self.sha256,
            "bytes": self.size,
            "identity": self.identity,
            "writer": self.writer.to_dict(),
            "writer_rule_id": self.writer_rule_id,
            "publication_transcript_sha256": self.publication_transcript_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ArtifactEvidence":
        row = require_exact_keys(
            value,
            {
                "schema", "relative_path", "artifact_type", "sha256", "bytes", "identity",
                "writer", "writer_rule_id", "publication_transcript_sha256",
            },
            EVIDENCE_SCHEMA,
        )
        if row["schema"] != EVIDENCE_SCHEMA:
            fail("EVIDENCE_INVALID", "artifact evidence schema differs")
        relative = validate_relative_path(row["relative_path"])
        if row["artifact_type"] not in {"regular", "directory"}:
            fail("EVIDENCE_INVALID", "artifact type is invalid")
        if row["artifact_type"] == "regular":
            digest = require_sha256(row["sha256"], "artifact sha256")
            if type(row["bytes"]) is not int or row["bytes"] < 0:
                fail("EVIDENCE_INVALID", "regular artifact bytes are invalid")
            size: int | None = int(row["bytes"])
        else:
            if row["sha256"] is not None or row["bytes"] is not None:
                fail("EVIDENCE_INVALID", "directory cannot have byte fields")
            digest = None
            size = None
        identity = _validate_identity(row["identity"], "artifact identity")
        if identity["type"] != row["artifact_type"]:
            fail("EVIDENCE_INVALID", "artifact identity/type differs")
        if not isinstance(row["writer_rule_id"], str):
            fail("EVIDENCE_INVALID", "artifact writer rule is invalid")
        return cls(
            relative, str(row["artifact_type"]), digest, size, dict(identity),
            WriterIdentity.from_dict(row["writer"]), str(row["writer_rule_id"]),
            require_sha256(row["publication_transcript_sha256"], "transcript hash"),
        )


@dataclass(frozen=True)
class Publication:
    evidence: ArtifactEvidence
    transcript: dict[str, object]


def transcript_hash(transcript: dict[str, object]) -> str:
    return sha256_bytes(canonical_json(transcript))


WRITE_OPEN_FLAGS = (
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
)
READBACK_OPEN_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
)
WRITE_FLAG_NAMES = [
    "O_WRONLY", "O_CREAT", "O_EXCL", "O_NOFOLLOW", "O_CLOEXEC",
]
READ_FLAG_NAMES = ["O_RDONLY", "O_NOFOLLOW", "O_CLOEXEC"]
DIRECTORY_FLAG_NAMES = [
    "O_RDONLY", "O_DIRECTORY", "O_NOFOLLOW", "O_CLOEXEC",
]
EVENT_KEYS = {
    "syscall",
    "fd_role",
    "fd_number",
    "descriptor",
    "dirfd_number",
    "dirfd_descriptor",
    "arguments",
    "result",
    "publication_id",
    "effective_publisher",
    "completed_monotonic_ns",
}


def _effective_publisher() -> dict[str, object]:
    with open("/proc/self/stat", "r", encoding="utf-8") as stream:
        raw = stream.read().strip()
    _, separator, tail = raw.rpartition(") ")
    fields = tail.split()
    if not separator or len(fields) <= 19:
        fail("PUBLISHER_IDENTITY_UNAVAILABLE", "cannot parse process start time")
    return {
        "pid": os.getpid(),
        "process_start_time_ticks": int(fields[19]),
        "effective_uid": os.geteuid(),
        "effective_gid": os.getegid(),
        "supplementary_groups": sorted(os.getgroups()),
    }


def _event_time(events: list[dict[str, object]]) -> int:
    value = time.monotonic_ns()
    if events and value <= events[-1]["completed_monotonic_ns"]:
        value = int(events[-1]["completed_monotonic_ns"]) + 1
    return value


def _syscall_event(
    events: list[dict[str, object]],
    *,
    syscall: str,
    fd_role: str,
    fd: int,
    dirfd: int | None,
    arguments: dict[str, object],
    result: dict[str, object],
    publication_id: str,
    effective_publisher: dict[str, object],
) -> dict[str, object]:
    return {
        "syscall": syscall,
        "fd_role": fd_role,
        "fd_number": fd,
        "descriptor": descriptor_evidence(fd),
        "dirfd_number": dirfd,
        "dirfd_descriptor": None if dirfd is None else descriptor_evidence(dirfd),
        "arguments": arguments,
        "result": result,
        "publication_id": publication_id,
        "effective_publisher": effective_publisher,
        "completed_monotonic_ns": _event_time(events),
    }


def _make_transcript(
    *,
    publication_id: str,
    effective_publisher: dict[str, object],
    sealed_run_id: str,
    run_root: str,
    relative_path: str,
    artifact_type: str,
    digest: str | None,
    size: int | None,
    identity: dict[str, object],
    root_identity: dict[str, object],
    writer: WriterIdentity,
    events: list[dict[str, object]],
    context: dict[str, object],
) -> dict[str, object]:
    emitted = time.monotonic_ns()
    if events and emitted <= int(events[-1]["completed_monotonic_ns"]):
        emitted = int(events[-1]["completed_monotonic_ns"]) + 1
    return {
        "schema": TRANSCRIPT_SCHEMA,
        "publication_id": publication_id,
        "effective_publisher": effective_publisher,
        "sealed_run_id": sealed_run_id,
        "run_root": run_root,
        "relative_path": relative_path,
        "artifact_type": artifact_type,
        "sha256": digest,
        "bytes": size,
        "identity": identity,
        "root_identity": root_identity,
        "writer": writer.to_dict(),
        "events": events,
        "context": context,
        "emitted_monotonic_ns": emitted,
    }


def _exact_int(value: object, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        fail("TRANSCRIPT_INVALID", f"{field} is not an exact integer")
    return value


def _validate_identity(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict) or value.get("type") not in {"regular", "directory"}:
        fail("TRANSCRIPT_INVALID", f"{field} type is invalid")
    expected = {"type", "dev", "inode", "mode", "uid", "gid"}
    if value["type"] == "regular":
        expected.add("size")
    if set(value) != expected:
        fail("TRANSCRIPT_INVALID", f"{field} keys differ")
    for key in expected - {"type"}:
        _exact_int(value[key], f"{field}.{key}")
    return value


def _validate_publisher(value: object) -> dict[str, object]:
    row = require_exact_keys(
        value,
        {
            "pid",
            "process_start_time_ticks",
            "effective_uid",
            "effective_gid",
            "supplementary_groups",
        },
        "effective-publisher",
    )
    _exact_int(row["pid"], "publisher.pid", minimum=1)
    _exact_int(
        row["process_start_time_ticks"],
        "publisher.process_start_time_ticks",
        minimum=1,
    )
    _exact_int(row["effective_uid"], "publisher.effective_uid")
    _exact_int(row["effective_gid"], "publisher.effective_gid")
    groups = row["supplementary_groups"]
    if (
        not isinstance(groups, list)
        or any(type(group) is not int or group < 0 for group in groups)
        or groups != sorted(set(groups))
    ):
        fail("TRANSCRIPT_INVALID", "effective publisher groups are invalid")
    return row


def _validate_descriptor(value: object, field: str) -> dict[str, object]:
    row = require_exact_keys(
        value, {"identity", "status_flags", "descriptor_flags"}, field
    )
    _validate_identity(row["identity"], f"{field}.identity")
    _exact_int(row["status_flags"], f"{field}.status_flags")
    descriptor_flags = _exact_int(
        row["descriptor_flags"], f"{field}.descriptor_flags"
    )
    if not descriptor_flags & fcntl.FD_CLOEXEC:
        fail("TRANSCRIPT_INVALID", f"{field} is not close-on-exec")
    return row


def _object_key(descriptor: dict[str, object]) -> tuple[object, ...]:
    identity = descriptor["identity"]
    return tuple(
        identity[key] for key in ("type", "dev", "inode", "mode", "uid", "gid")
    )


def _validate_event(
    raw: object,
    *,
    publication_id: str,
    publisher: dict[str, object],
    previous: int,
) -> dict[str, object]:
    event = require_exact_keys(raw, EVENT_KEYS, "publication-event")
    if (
        not isinstance(event["syscall"], str)
        or not event["syscall"]
        or not isinstance(event["fd_role"], str)
        or not event["fd_role"]
    ):
        fail("TRANSCRIPT_INVALID", "event syscall/fd role is invalid")
    _exact_int(event["fd_number"], "event.fd_number")
    _validate_descriptor(event["descriptor"], "event.descriptor")
    if event["dirfd_number"] is None:
        if event["dirfd_descriptor"] is not None:
            fail("TRANSCRIPT_INVALID", "event dirfd evidence is asymmetric")
    else:
        _exact_int(event["dirfd_number"], "event.dirfd_number")
        _validate_descriptor(event["dirfd_descriptor"], "event.dirfd_descriptor")
    if not isinstance(event["arguments"], dict) or not isinstance(event["result"], dict):
        fail("TRANSCRIPT_INVALID", "event arguments/result must be objects")
    event_publisher = _validate_publisher(event["effective_publisher"])
    if (
        event["publication_id"] != publication_id
        or event_publisher != publisher
    ):
        fail("TRANSCRIPT_MISMATCH", "event publication/publisher binding differs")
    completed = _exact_int(
        event["completed_monotonic_ns"], "event.completed_monotonic_ns"
    )
    if completed <= previous:
        fail("TRANSCRIPT_EARLY", "event chronology is not strictly increasing")
    return event


def _require_event(
    event: dict[str, object],
    syscall: str,
    fd_role: str,
    arguments: dict[str, object],
) -> None:
    if (
        event["syscall"] != syscall
        or event["fd_role"] != fd_role
        or event["arguments"] != arguments
    ):
        fail("TRANSCRIPT_INVALID", f"{syscall} event differs from exact contract")


def _return_fd(event: dict[str, object], field: str) -> int:
    result = require_exact_keys(event["result"], {"return_fd"}, field)
    value = _exact_int(result["return_fd"], f"{field}.return_fd")
    if value != event["fd_number"]:
        fail("TRANSCRIPT_MISMATCH", f"{field} returned fd differs")
    return value


def _return_zero(event: dict[str, object], field: str) -> None:
    result = require_exact_keys(event["result"], {"return_code"}, field)
    if _exact_int(result["return_code"], f"{field}.return_code") != 0:
        fail("TRANSCRIPT_INVALID", f"{field} did not return zero")


def _verify_read_segments(
    parent_fd: int,
    basename: str,
    current_identity: dict[str, object],
    segments: list[tuple[int, str]],
    expected_size: int,
    expected_digest: str,
) -> None:
    fd = os.open(basename, READBACK_OPEN_FLAGS, dir_fd=parent_fd)
    try:
        initial_stat = os.fstat(fd)
        if stable_identity(initial_stat) != current_identity:
            fail("TRANSCRIPT_STALE", "segment verification descriptor differs")
        for length, expected_chunk_digest in segments:
            remaining = length
            digest = hashlib.sha256()
            while remaining:
                block = os.read(fd, remaining)
                if not block:
                    fail("TRANSCRIPT_EARLY", "segment verification reached early EOF")
                remaining -= len(block)
                digest.update(block)
            if digest.hexdigest() != expected_chunk_digest:
                fail("TRANSCRIPT_STALE", "recorded read chunk digest differs")
        if os.read(fd, 1) != b"":
            fail("TRANSCRIPT_INVALID", "recorded read segmentation did not reach EOF")
        os.lseek(fd, 0, os.SEEK_SET)
        replay_digest = hashlib.sha256()
        replay_size = 0
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            replay_digest.update(block)
            replay_size += len(block)
        final_stat = os.fstat(fd)
        if (
            replay_size != expected_size
            or replay_digest.hexdigest() != expected_digest
            or stable_identity(final_stat) != current_identity
            or initial_stat.st_mtime_ns != final_stat.st_mtime_ns
            or initial_stat.st_ctime_ns != final_stat.st_ctime_ns
        ):
            fail("TRANSCRIPT_STALE", "artifact changed during full replay verification")
    finally:
        os.close(fd)


def _validate_regular_transcript(
    root_fd: int,
    parent_fd: int,
    parent_descriptor: dict[str, object],
    basename: str,
    relative: str,
    row: dict[str, object],
    events: list[dict[str, object]],
) -> None:
    digest, size, current = hash_regular_at(root_fd, relative)
    expected_digest = require_sha256(row["sha256"], "transcript artifact sha256")
    expected_size = _exact_int(row["bytes"], "transcript bytes")
    current_identity = stable_identity(current)
    _validate_identity(row["identity"], "transcript.identity")
    if (
        expected_digest != digest
        or expected_size != size
        or row["identity"] != current_identity
        or current_identity["mode"] != 0o444
    ):
        fail("TRANSCRIPT_STALE", "regular artifact bytes/identity/mode differ")

    first = events[0]
    _require_event(
        first,
        "openat_exclusive",
        "artifact",
        {
            "name": basename,
            "flags": WRITE_OPEN_FLAGS,
            "flag_names": WRITE_FLAG_NAMES,
            "mode": 0o444,
        },
    )
    artifact_fd = _return_fd(first, "openat-result")
    if (
        first["dirfd_descriptor"] != parent_descriptor
        or first["descriptor"]["identity"]["size"] != 0
        or first["descriptor"]["status_flags"] & os.O_ACCMODE != os.O_WRONLY
    ):
        fail("TRANSCRIPT_STALE", "exclusive-open descriptor evidence differs")
    object_key = _object_key(first["descriptor"])
    if object_key != tuple(
        current_identity[key]
        for key in ("type", "dev", "inode", "mode", "uid", "gid")
    ):
        fail("TRANSCRIPT_STALE", "exclusive-open artifact object differs")

    remaining = size
    cumulative = 0
    cursor = 1
    while cursor < len(events) and events[cursor]["syscall"] == "write":
        event = events[cursor]
        arguments = require_exact_keys(
            event["arguments"], {"requested_bytes"}, "write-arguments"
        )
        requested = _exact_int(
            arguments["requested_bytes"],
            "write-arguments.requested_bytes",
            minimum=1,
        )
        _require_event(
            event, "write", "artifact", {"requested_bytes": requested}
        )
        result = require_exact_keys(
            event["result"], {"bytes_written"}, "write-result"
        )
        written = _exact_int(
            result["bytes_written"], "write-result.bytes_written", minimum=1
        )
        if (
            requested != remaining
            or written > remaining
            or event["fd_number"] != artifact_fd
            or event["dirfd_number"] is not None
            or _object_key(event["descriptor"]) != object_key
            or event["descriptor"]["status_flags"]
            != first["descriptor"]["status_flags"]
            or event["descriptor"]["descriptor_flags"]
            != first["descriptor"]["descriptor_flags"]
        ):
            fail("TRANSCRIPT_INVALID", "write descriptor/accounting differs")
        cumulative += written
        remaining -= written
        if event["descriptor"]["identity"]["size"] != cumulative:
            fail("TRANSCRIPT_INVALID", "write descriptor size is not cumulative")
        cursor += 1
    if remaining != 0:
        fail("TRANSCRIPT_EARLY", "write events do not cover exact payload bytes")
    if len(events) < cursor + 4:
        fail("TRANSCRIPT_EARLY", "regular syscall state machine is incomplete")

    file_sync, parent_sync, readback = events[cursor : cursor + 3]
    _require_event(file_sync, "fsync", "artifact", {})
    _return_zero(file_sync, "artifact-fsync-result")
    if (
        file_sync["fd_number"] != artifact_fd
        or file_sync["dirfd_number"] is not None
        or _object_key(file_sync["descriptor"]) != object_key
        or file_sync["descriptor"]["identity"] != current_identity
        or file_sync["descriptor"]["status_flags"]
        != first["descriptor"]["status_flags"]
        or file_sync["descriptor"]["descriptor_flags"]
        != first["descriptor"]["descriptor_flags"]
    ):
        fail("TRANSCRIPT_MISMATCH", "artifact fsync descriptor differs")

    _require_event(parent_sync, "fsync", "containing_directory", {})
    _return_zero(parent_sync, "parent-fsync-result")
    if (
        parent_sync["descriptor"] != parent_descriptor
        or parent_sync["dirfd_number"] is not None
        or first["dirfd_number"] != parent_sync["fd_number"]
    ):
        fail("TRANSCRIPT_MISMATCH", "parent fsync descriptor differs")

    _require_event(
        readback,
        "openat_readback",
        "readback_artifact",
        {
            "name": basename,
            "flags": READBACK_OPEN_FLAGS,
            "flag_names": READ_FLAG_NAMES,
        },
    )
    _return_fd(readback, "readback-open-result")
    if (
        readback["dirfd_descriptor"] != parent_descriptor
        or readback["dirfd_number"] != parent_sync["fd_number"]
        or readback["descriptor"]["identity"] != current_identity
        or _object_key(readback["descriptor"]) != object_key
        or readback["descriptor"]["status_flags"] & os.O_ACCMODE != os.O_RDONLY
    ):
        fail("TRANSCRIPT_STALE", "readback descriptor chain differs")

    read_events = events[cursor + 3 :]
    if not read_events:
        fail("TRANSCRIPT_EARLY", "readback has no read/EOF evidence")
    read_total = 0
    saw_eof = False
    segments: list[tuple[int, str]] = []
    for index, read_event in enumerate(read_events):
        if (
            read_event["syscall"] != "read"
            or read_event["fd_role"] != "readback_artifact"
            or read_event["fd_number"] != readback["fd_number"]
            or read_event["dirfd_number"] is not None
            or read_event["descriptor"] != readback["descriptor"]
        ):
            fail("TRANSCRIPT_MISMATCH", "readback read descriptor chain differs")
        arguments = require_exact_keys(
            read_event["arguments"], {"requested_bytes"}, "read-arguments"
        )
        requested = _exact_int(
            arguments["requested_bytes"], "read.requested_bytes", minimum=1
        )
        if requested != 1024 * 1024:
            fail("TRANSCRIPT_INVALID", "readback request size differs")
        result = read_event["result"]
        if set(result) == {"bytes_read", "chunk_sha256"}:
            if saw_eof:
                fail("TRANSCRIPT_INVALID", "read event follows EOF")
            count = _exact_int(
                result["bytes_read"], "read.bytes_read", minimum=1
            )
            if count > requested:
                fail("TRANSCRIPT_INVALID", "read exceeds requested bytes")
            chunk_digest = require_sha256(
                result["chunk_sha256"], "read chunk sha256"
            )
            segments.append((count, chunk_digest))
            read_total += count
        elif set(result) == {
            "bytes_read", "eof", "total_bytes", "readback_sha256"
        }:
            if index != len(read_events) - 1 or saw_eof:
                fail("TRANSCRIPT_INVALID", "EOF is not the unique final read")
            eof_count = _exact_int(result["bytes_read"], "read.eof_bytes")
            total_bytes = _exact_int(result["total_bytes"], "read.total_bytes")
            readback_sha256 = require_sha256(
                result["readback_sha256"], "readback sha256"
            )
            if (
                eof_count != 0
                or result["eof"] is not True
                or total_bytes != read_total
                or readback_sha256 != expected_digest
            ):
                fail("TRANSCRIPT_STALE", "readback EOF aggregate differs")
            saw_eof = True
        else:
            fail("SCHEMA_INVALID", "read result keys differ")
    if not saw_eof or read_total != size:
        fail("TRANSCRIPT_EARLY", "readback did not reach exact EOF/byte count")
    _verify_read_segments(
        parent_fd,
        basename,
        current_identity,
        segments,
        expected_size,
        expected_digest,
    )


def _validate_directory_transcript(
    parent_descriptor: dict[str, object],
    relative: str,
    row: dict[str, object],
    events: list[dict[str, object]],
    root_fd: int,
    basename: str,
) -> None:
    from .filesystem import lstat_relative

    current = lstat_relative(root_fd, relative)
    current_identity = stable_identity(current)
    _validate_identity(row["identity"], "transcript.identity")
    if (
        row["sha256"] is not None
        or row["bytes"] is not None
        or not stat.S_ISDIR(current.st_mode)
        or row["identity"] != current_identity
        or current_identity["mode"] != 0o755
    ):
        fail("TRANSCRIPT_STALE", "directory identity/mode differs")
    if len(events) != 3:
        fail("TRANSCRIPT_EARLY", "directory syscall state machine is incomplete")
    mkdir_event, parent_sync, readback = events
    _require_event(
        mkdir_event,
        "mkdirat_exclusive",
        "created_directory_observation",
        {"name": basename, "mode": 0o755},
    )
    _return_zero(mkdir_event, "mkdirat-result")
    _require_event(parent_sync, "fsync", "containing_directory", {})
    _return_zero(parent_sync, "directory-parent-fsync-result")
    _require_event(
        readback,
        "openat_readback",
        "created_directory",
        {
            "name": basename,
            "flags": DIRECTORY_OPEN_FLAGS,
            "flag_names": DIRECTORY_FLAG_NAMES,
        },
    )
    _return_fd(readback, "directory-readback-result")
    if (
        mkdir_event["dirfd_descriptor"] != parent_descriptor
        or mkdir_event["dirfd_number"] != parent_sync["fd_number"]
        or readback["dirfd_number"] != parent_sync["fd_number"]
        or parent_sync["descriptor"] != parent_descriptor
        or parent_sync["dirfd_number"] is not None
        or readback["dirfd_descriptor"] != parent_descriptor
        or mkdir_event["descriptor"]["identity"] != current_identity
        or readback["descriptor"]["identity"] != current_identity
        or mkdir_event["descriptor"]["status_flags"] & os.O_ACCMODE != os.O_RDONLY
        or readback["descriptor"]["status_flags"] & os.O_ACCMODE != os.O_RDONLY
        or not mkdir_event["descriptor"]["status_flags"] & os.O_DIRECTORY
        or not readback["descriptor"]["status_flags"] & os.O_DIRECTORY
    ):
        fail("TRANSCRIPT_MISMATCH", "directory descriptor chain differs")


def validate_publication_transcript(
    root_fd: int,
    sealed_run_id: str,
    run_root: str,
    transcript: object,
    *,
    expected_relative_path: str | None = None,
    expected_writer: WriterIdentity | None = None,
) -> dict[str, object]:
    row = require_exact_keys(
        transcript,
        {
            "schema",
            "publication_id",
            "effective_publisher",
            "sealed_run_id",
            "run_root",
            "relative_path",
            "artifact_type",
            "sha256",
            "bytes",
            "identity",
            "root_identity",
            "writer",
            "events",
            "context",
            "emitted_monotonic_ns",
        },
        TRANSCRIPT_SCHEMA,
    )
    if row["schema"] != TRANSCRIPT_SCHEMA or row["sealed_run_id"] != sealed_run_id:
        fail("TRANSCRIPT_MISMATCH", "transcript schema or run ID differs")
    if row["run_root"] != run_root:
        fail("TRANSCRIPT_MISMATCH", "transcript run root differs")
    publication_id = row["publication_id"]
    if (
        not isinstance(publication_id, str)
        or re.fullmatch(r"[0-9a-f]{32}", publication_id) is None
    ):
        fail("TRANSCRIPT_INVALID", "publication ID is invalid")
    publisher = _validate_publisher(row["effective_publisher"])
    relative = validate_relative_path(row["relative_path"])
    if expected_relative_path is not None and relative != expected_relative_path:
        fail("TRANSCRIPT_MISMATCH", "transcript artifact path differs")
    writer = WriterIdentity.from_dict(row["writer"])
    if expected_writer is not None and writer != expected_writer:
        fail("TRANSCRIPT_MISMATCH", "transcript writer differs")
    _validate_identity(row["root_identity"], "transcript.root_identity")
    if row["root_identity"] != stable_identity(os.fstat(root_fd)):
        fail("TRANSCRIPT_STALE", "transcript root identity differs")
    if not isinstance(row["events"], list) or not row["events"]:
        fail("TRANSCRIPT_EARLY", "transcript contains no completed syscall events")
    events: list[dict[str, object]] = []
    previous = -1
    for raw_event in row["events"]:
        event = _validate_event(
            raw_event,
            publication_id=publication_id,
            publisher=publisher,
            previous=previous,
        )
        previous = int(event["completed_monotonic_ns"])
        events.append(event)
    emitted = _exact_int(
        row["emitted_monotonic_ns"], "transcript.emitted_monotonic_ns"
    )
    if emitted <= previous:
        fail("TRANSCRIPT_EARLY", "transcript predates its final syscall event")
    if not isinstance(row["context"], dict):
        fail("TRANSCRIPT_INVALID", "transcript context must be an object")

    parent_fd, basename = open_parent_directory(root_fd, relative)
    try:
        parent_descriptor = descriptor_evidence(parent_fd)
        if row["artifact_type"] == "regular":
            _validate_regular_transcript(
                root_fd,
                parent_fd,
                parent_descriptor,
                basename,
                relative,
                row,
                events,
            )
        elif row["artifact_type"] == "directory":
            _validate_directory_transcript(
                parent_descriptor, relative, row, events, root_fd, basename
            )
        else:
            fail("TRANSCRIPT_INVALID", "artifact type is invalid")
    finally:
        os.close(parent_fd)
    return row


class StagePublisher:
    """Publish one writer's artifacts without replacement beneath an explicit run."""

    def __init__(
        self,
        strict_parent: str,
        sealed_run_id: str,
        run_root: str,
        writer: WriterIdentity,
        policy: WriterPolicy,
        *,
        allow_reserved_paths: bool = False,
        run_binding: RunDescriptorBinding | None = None,
    ) -> None:
        self.strict_parent = strict_parent
        self.sealed_run_id = validate_sealed_run_id(sealed_run_id)
        self.run_root = run_root
        self.writer = writer
        self.policy = policy
        self.allow_reserved_paths = allow_reserved_paths
        if run_binding is not None and not isinstance(
            run_binding, RunDescriptorBinding
        ):
            fail("RUN_BINDING_INVALID", "publisher run binding is not typed")
        self.run_binding = run_binding

    def _open_run_handle(self):
        if self.run_binding is None:
            return open_run_handle(
                self.strict_parent, self.sealed_run_id, self.run_root
            )
        return open_bound_run_handle(
            self.strict_parent,
            self.sealed_run_id,
            self.run_root,
            self.run_binding,
        )

    def _assert_bound_handle(self, parent_fd: int, root_fd: int) -> None:
        if self.run_binding is not None:
            self.run_binding.assert_fds(
                parent_fd,
                root_fd,
                strict_parent=self.strict_parent,
                run_root=self.run_root,
            )

    def _assert_publication_mount(self, fd: int) -> None:
        if self.run_binding is None:
            return
        current = mount_identity(fd)
        namespace = current.get("mount_namespace")
        expected = self.run_binding.root
        if (
            os.fstat(fd).st_dev != expected.dev
            or current.get("mount_id") != expected.mount_id
            or not isinstance(namespace, dict)
            or namespace.get("dev") != expected.mount_namespace_dev
            or namespace.get("inode") != expected.mount_namespace_inode
            or namespace.get("link") != expected.mount_namespace_link
        ):
            fail(
                "MOUNT_SUBSTITUTION",
                "publication directory crossed the reservation-created mount",
            )

    def _guard_reserved_path(self, relative: str) -> None:
        if not self.allow_reserved_paths and (
            relative == "checkpoints"
            or relative.startswith("checkpoints/")
            or relative == "terminal"
            or relative.startswith("terminal/")
        ):
            fail("RESERVED_CONTROLLER_PATH", "producer cannot publish controller paths")

    def _guard_terminal(self, root_fd: int) -> None:
        if exists_relative(root_fd, "terminal/blocked.json"):
            fail("RUN_TERMINAL_BLOCKED", "blocked run cannot be continued")
        if exists_relative(root_fd, "checkpoints/cp6.json"):
            fail("RUN_SUCCESS_TERMINAL", "CP6 run cannot receive later artifacts")

    def _directory_publication(
        self,
        root_fd: int,
        relative: str,
        directory_fd: int,
        events: list[dict[str, object]],
        publication_id: str,
        effective_publisher: dict[str, object],
        writer_rule_id: str,
    ) -> Publication:
        identity = stable_identity(os.fstat(directory_fd))
        transcript = _make_transcript(
            publication_id=publication_id,
            effective_publisher=effective_publisher,
            sealed_run_id=self.sealed_run_id,
            run_root=self.run_root,
            relative_path=relative,
            artifact_type="directory",
            digest=None,
            size=None,
            identity=identity,
            root_identity=stable_identity(os.fstat(root_fd)),
            writer=self.writer,
            events=events,
            context={},
        )
        evidence = ArtifactEvidence(
            relative,
            "directory",
            None,
            None,
            identity,
            self.writer,
            writer_rule_id,
            transcript_hash(transcript),
        )
        return Publication(evidence, transcript)

    def _ensure_directories(self, root_fd: int, relative: str) -> list[Publication]:
        parts = validate_relative_path(relative).split("/")[:-1]
        publications: list[Publication] = []
        fd = os.dup(root_fd)
        self._assert_publication_mount(fd)
        prefix: list[str] = []
        try:
            for component in parts:
                prefix.append(component)
                current = "/".join(prefix)
                try:
                    child_fd = os.open(component, DIRECTORY_OPEN_FLAGS, dir_fd=fd)
                except FileNotFoundError:
                    rule = self.policy.authorize(current, self.writer)
                    publication_id = secrets.token_hex(16)
                    publisher = _effective_publisher()
                    events: list[dict[str, object]] = []
                    os.mkdir(component, 0o755, dir_fd=fd)
                    observation_fd = os.open(
                        component, DIRECTORY_OPEN_FLAGS, dir_fd=fd
                    )
                    try:
                        self._assert_publication_mount(observation_fd)
                        events.append(
                            _syscall_event(
                                events,
                                syscall="mkdirat_exclusive",
                                fd_role="created_directory_observation",
                                fd=observation_fd,
                                dirfd=fd,
                                arguments={"name": component, "mode": 0o755},
                                result={"return_code": 0},
                                publication_id=publication_id,
                                effective_publisher=publisher,
                            )
                        )
                    finally:
                        os.close(observation_fd)
                    os.fsync(fd)
                    events.append(
                        _syscall_event(
                            events,
                            syscall="fsync",
                            fd_role="containing_directory",
                            fd=fd,
                            dirfd=None,
                            arguments={},
                            result={"return_code": 0},
                            publication_id=publication_id,
                            effective_publisher=publisher,
                        )
                    )
                    child_fd = os.open(component, DIRECTORY_OPEN_FLAGS, dir_fd=fd)
                    self._assert_publication_mount(child_fd)
                    events.append(
                        _syscall_event(
                            events,
                            syscall="openat_readback",
                            fd_role="created_directory",
                            fd=child_fd,
                            dirfd=fd,
                            arguments={
                                "name": component,
                                "flags": DIRECTORY_OPEN_FLAGS,
                                "flag_names": DIRECTORY_FLAG_NAMES,
                            },
                            result={"return_fd": child_fd},
                            publication_id=publication_id,
                            effective_publisher=publisher,
                        )
                    )
                    publications.append(
                        self._directory_publication(
                            root_fd,
                            current,
                            child_fd,
                            events,
                            publication_id,
                            publisher,
                            rule.rule_id,
                        )
                    )
                self._assert_publication_mount(child_fd)
                os.close(fd)
                fd = child_fd
        finally:
            os.close(fd)
        return publications

    def ensure_directory(self, relative: str) -> list[Publication]:
        relative = validate_relative_path(relative)
        self._guard_reserved_path(relative)
        self.policy.authorize(relative, self.writer)
        with self._open_run_handle() as handle:
            self._guard_terminal(handle.root_fd)
            result = self._ensure_directories(
                handle.root_fd, f"{relative}/directory-placeholder"
            )
            self._assert_bound_handle(handle.parent_fd, handle.root_fd)
            return result

    def _publish_bytes_at(
        self,
        root_fd: int,
        relative: str,
        payload: bytes,
        *,
        context: dict[str, object] | None,
        _failure_after: str | None,
    ) -> list[Publication]:
        relative = validate_relative_path(relative)
        self._assert_publication_mount(root_fd)
        self._guard_reserved_path(relative)
        rule = self.policy.authorize(relative, self.writer)
        self._guard_terminal(root_fd)
        publications = self._ensure_directories(root_fd, relative)
        publication_id = secrets.token_hex(16)
        publisher = _effective_publisher()
        events: list[dict[str, object]] = []
        parent_fd = os.dup(root_fd)
        try:
            for component in relative.split("/")[:-1]:
                next_fd = os.open(
                    component, DIRECTORY_OPEN_FLAGS, dir_fd=parent_fd
                )
                self._assert_publication_mount(next_fd)
                os.close(parent_fd)
                parent_fd = next_fd
            name = relative.split("/")[-1]
            self._assert_publication_mount(parent_fd)
            fd = os.open(name, WRITE_OPEN_FLAGS, 0o444, dir_fd=parent_fd)
            events.append(
                _syscall_event(
                    events,
                    syscall="openat_exclusive",
                    fd_role="artifact",
                    fd=fd,
                    dirfd=parent_fd,
                    arguments={
                        "name": name,
                        "flags": WRITE_OPEN_FLAGS,
                        "flag_names": WRITE_FLAG_NAMES,
                        "mode": 0o444,
                    },
                    result={"return_fd": fd},
                    publication_id=publication_id,
                    effective_publisher=publisher,
                )
            )
            try:
                if _failure_after == "exclusive_open":
                    fail("INJECTED_CRASH", "injected after exclusive final-path open")
                view = memoryview(payload)
                while view:
                    requested = len(view)
                    written = os.write(fd, view)
                    if written <= 0:
                        fail("SHORT_WRITE", "artifact write made no progress")
                    events.append(
                        _syscall_event(
                            events,
                            syscall="write",
                            fd_role="artifact",
                            fd=fd,
                            dirfd=None,
                            arguments={"requested_bytes": requested},
                            result={"bytes_written": written},
                            publication_id=publication_id,
                            effective_publisher=publisher,
                        )
                    )
                    view = view[written:]
                if _failure_after == "write_complete":
                    fail("INJECTED_CRASH", "injected after final bytes")
                os.fsync(fd)
                events.append(
                    _syscall_event(
                        events,
                        syscall="fsync",
                        fd_role="artifact",
                        fd=fd,
                        dirfd=None,
                        arguments={},
                        result={"return_code": 0},
                        publication_id=publication_id,
                        effective_publisher=publisher,
                    )
                )
                written_st = os.fstat(fd)
                if _failure_after == "file_sync":
                    fail("INJECTED_CRASH", "injected after file sync")
            finally:
                os.close(fd)
            os.fsync(parent_fd)
            events.append(
                _syscall_event(
                    events,
                    syscall="fsync",
                    fd_role="containing_directory",
                    fd=parent_fd,
                    dirfd=None,
                    arguments={},
                    result={"return_code": 0},
                    publication_id=publication_id,
                    effective_publisher=publisher,
                )
            )
            if _failure_after == "directory_sync":
                fail("INJECTED_CRASH", "injected after containing-directory sync")
            check_fd = os.open(name, READBACK_OPEN_FLAGS, dir_fd=parent_fd)
            events.append(
                _syscall_event(
                    events,
                    syscall="openat_readback",
                    fd_role="readback_artifact",
                    fd=check_fd,
                    dirfd=parent_fd,
                    arguments={
                        "name": name,
                        "flags": READBACK_OPEN_FLAGS,
                        "flag_names": READ_FLAG_NAMES,
                    },
                    result={"return_fd": check_fd},
                    publication_id=publication_id,
                    effective_publisher=publisher,
                )
            )
            try:
                check_st = os.fstat(check_fd)
                if (check_st.st_dev, check_st.st_ino) != (
                    written_st.st_dev,
                    written_st.st_ino,
                ):
                    fail("PUBLICATION_SUBSTITUTED", "publication inode changed")
                chunks: list[bytes] = []
                readback_digest = hashlib.sha256()
                readback_bytes = 0
                while True:
                    block = os.read(check_fd, 1024 * 1024)
                    if block:
                        chunks.append(block)
                        readback_digest.update(block)
                        readback_bytes += len(block)
                        read_result: dict[str, object] = {
                            "bytes_read": len(block),
                            "chunk_sha256": sha256_bytes(block),
                        }
                    else:
                        read_result = {
                            "bytes_read": 0,
                            "eof": True,
                            "total_bytes": readback_bytes,
                            "readback_sha256": readback_digest.hexdigest(),
                        }
                    events.append(
                        _syscall_event(
                            events,
                            syscall="read",
                            fd_role="readback_artifact",
                            fd=check_fd,
                            dirfd=None,
                            arguments={"requested_bytes": 1024 * 1024},
                            result=read_result,
                            publication_id=publication_id,
                            effective_publisher=publisher,
                        )
                    )
                    if not block:
                        break
                after_read = os.fstat(check_fd)
                if (
                    stable_identity(after_read) != stable_identity(check_st)
                    or readback_bytes != len(payload)
                    or readback_digest.hexdigest() != sha256_bytes(payload)
                    or b"".join(chunks) != payload
                ):
                    fail(
                        "PUBLICATION_SUBSTITUTED",
                        "publication read-back differs or mutated",
                    )
            finally:
                os.close(check_fd)
        finally:
            os.close(parent_fd)
        identity = stable_identity(check_st)
        digest = sha256_bytes(payload)
        transcript = _make_transcript(
            publication_id=publication_id,
            effective_publisher=publisher,
            sealed_run_id=self.sealed_run_id,
            run_root=self.run_root,
            relative_path=relative,
            artifact_type="regular",
            digest=digest,
            size=len(payload),
            identity=identity,
            root_identity=stable_identity(os.fstat(root_fd)),
            writer=self.writer,
            events=events,
            context={} if context is None else context,
        )
        publications.append(
            Publication(
                ArtifactEvidence(
                    relative,
                    "regular",
                    digest,
                    len(payload),
                    identity,
                    self.writer,
                    rule.rule_id,
                    transcript_hash(transcript),
                ),
                transcript,
            )
        )
        return publications

    def publish_bytes(
        self,
        relative: str,
        payload: bytes,
        *,
        context: dict[str, object] | None = None,
        _failure_after: str | None = None,
    ) -> list[Publication]:
        with self._open_run_handle() as handle:
            result = self._publish_bytes_at(
                handle.root_fd,
                relative,
                payload,
                context=context,
                _failure_after=_failure_after,
            )
            self._assert_bound_handle(handle.parent_fd, handle.root_fd)
            return result

    def publish_bytes_on_reserved_descriptors(
        self,
        parent_fd: int,
        root_fd: int,
        relative: str,
        payload: bytes,
        *,
        context: dict[str, object] | None = None,
        _failure_after: str | None = None,
    ) -> list[Publication]:
        strict_parent, run_root = validate_run_root_text(
            self.strict_parent, self.sealed_run_id, self.run_root
        )
        if self.run_binding is None:
            assert_path_matches_fd(strict_parent, parent_fd)
            assert_path_matches_fd(run_root, root_fd)
        else:
            self.run_binding.assert_fds(
                parent_fd,
                root_fd,
                strict_parent=strict_parent,
                run_root=run_root,
            )
        if os.fstat(parent_fd).st_dev != os.fstat(root_fd).st_dev:
            fail("MOUNT_SUBSTITUTION", "reserved child changed filesystem device")
        self._assert_publication_mount(root_fd)
        result = self._publish_bytes_at(
            root_fd,
            relative,
            payload,
            context=context,
            _failure_after=_failure_after,
        )
        if self.run_binding is None:
            assert_path_matches_fd(strict_parent, parent_fd)
            assert_path_matches_fd(run_root, root_fd)
        else:
            self.run_binding.assert_fds(
                parent_fd,
                root_fd,
                strict_parent=strict_parent,
                run_root=run_root,
            )
        if os.fstat(parent_fd).st_dev != os.fstat(root_fd).st_dev:
            fail("MOUNT_SUBSTITUTION", "reserved child changed filesystem device")
        return result

    def publish_json(
        self,
        relative: str,
        value: object,
        *,
        context: dict[str, object] | None = None,
        _failure_after: str | None = None,
    ) -> list[Publication]:
        return self.publish_bytes(
            relative,
            canonical_json(value),
            context=context,
            _failure_after=_failure_after,
        )

    def publish_json_on_reserved_descriptors(
        self,
        parent_fd: int,
        root_fd: int,
        relative: str,
        value: object,
        *,
        context: dict[str, object] | None = None,
        _failure_after: str | None = None,
    ) -> list[Publication]:
        return self.publish_bytes_on_reserved_descriptors(
            parent_fd,
            root_fd,
            relative,
            canonical_json(value),
            context=context,
            _failure_after=_failure_after,
        )
