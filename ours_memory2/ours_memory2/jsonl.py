"""Strict JSONL input validation and output-root safety."""

from __future__ import annotations

from dataclasses import dataclass
import json
import errno
import os
from pathlib import Path
from typing import Iterable, Mapping

from .contracts import (
    DialogueTurn,
    ExampleInput,
    InputContractError,
    OutputFilename,
    SessionInput,
    normalize_scalar_id,
)


@dataclass(frozen=True)
class _OpenedChild:
    filename: OutputFilename
    descriptor: int
    original_size: int
    created: bool
    device: int
    inode: int


def validate_output_root(raw_output_root: str | Path) -> Path:
    """Resolve and validate an output root without creating or opening anything."""

    try:
        resolved = Path(raw_output_root).expanduser().resolve(strict=False)
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        raise InputContractError("cannot be resolved", path="output_root") from exc
    if "ours_memory" in resolved.parts:
        raise InputContractError("contains a reserved path component", path="output_root")
    return resolved


def read_examples_jsonl(path: str | Path) -> tuple[ExampleInput, ...]:
    """Read strict UTF-8 JSONL examples with physical-line error paths."""

    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise InputContractError("cannot be read", path="input") from exc

    examples: list[ExampleInput] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(raw.splitlines(), start=1):
        line_path = f"input.line[{line_number}]"
        try:
            text = raw_line.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise InputContractError("is not valid UTF-8", path=line_path) from exc
        if not text.strip():
            continue
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise InputContractError("is not valid line-local JSON", path=line_path) from exc
        if not isinstance(value, dict):
            raise InputContractError("must be an object", path=line_path)
        example = _parse_example(value, line_path)
        if example.example_id in seen:
            raise InputContractError(
                "duplicates an earlier normalized identifier",
                path=f"{line_path}.example_id",
            )
        seen.add(example.example_id)
        examples.append(example)
    if not examples:
        raise InputContractError("must contain at least one example", path="input")
    return tuple(examples)


def _parse_example(value: dict[str, object], path: str) -> ExampleInput:
    example_id = normalize_scalar_id(value.get("example_id"), path=f"{path}.example_id")
    sessions_value = value.get("sessions")
    if not isinstance(sessions_value, list) or not sessions_value:
        raise InputContractError("must be a nonempty list", path=f"{path}.sessions")
    sessions = tuple(
        _parse_session(session, f"{path}.sessions[{index}]")
        for index, session in enumerate(sessions_value)
    )
    metadata = {
        key: item for key, item in value.items() if key not in {"example_id", "sessions"}
    }
    return ExampleInput(example_id=example_id, sessions=sessions, metadata=metadata)


def _parse_session(value: object, path: str) -> SessionInput:
    if not isinstance(value, dict):
        raise InputContractError("must be an object", path=path)
    dialogue_value = value.get("dialogue")
    if not isinstance(dialogue_value, list) or not dialogue_value:
        raise InputContractError("must be a nonempty list", path=f"{path}.dialogue")
    dialogue = tuple(
        _parse_turn(turn, f"{path}.dialogue[{index}]")
        for index, turn in enumerate(dialogue_value)
    )
    api_value = value.get("api_call")
    if not isinstance(api_value, list):
        raise InputContractError("must be a list", path=f"{path}.api_call")
    api_calls: list[str] = []
    for index, item in enumerate(api_value):
        if not isinstance(item, str):
            raise InputContractError("must be a string", path=f"{path}.api_call[{index}]")
        api_calls.append(item)
    return SessionInput(dialogue=dialogue, api_call=tuple(api_calls))


def _parse_turn(value: object, path: str) -> DialogueTurn:
    if not isinstance(value, dict):
        raise InputContractError("must be an object", path=path)
    role = value.get("role")
    if not isinstance(role, str) or not role.strip():
        raise InputContractError("must be a nonempty string", path=f"{path}.role")
    message = value.get("message")
    if not isinstance(message, str) or not message.strip():
        raise InputContractError("must be a nonempty string", path=f"{path}.message")
    return DialogueTurn(role=role, message=message)


def encode_jsonl(rows: Iterable[object]) -> bytes:
    """Serialize JSON-safe rows deterministically with trailing newlines."""

    return b"".join(
        (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        for row in rows
    )


def append_jsonl_streams(
    output_root: str | Path, streams: Mapping[OutputFilename, Iterable[object]]
) -> tuple[Path, ...]:
    """Append correlated direct-child streams as one rollback-capable transaction."""

    root = validate_output_root(output_root)
    payloads = _prepare_payloads(streams)
    if not payloads:
        return ()

    root_created = False
    directory_fd: int | None = None
    opened: list[_OpenedChild] = []
    failed = False
    try:
        try:
            root.mkdir(parents=True, exist_ok=False)
            root_created = True
        except FileExistsError:
            pass
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_fd = os.open(root, directory_flags)
        for filename, _payload in payloads:
            opened.append(_open_direct_child(directory_fd, filename))

        for (_filename, payload), child in zip(payloads, opened):
            _write_all(child.descriptor, payload)
            os.fsync(child.descriptor)
    except BaseException:
        failed = True
        if directory_fd is not None:
            _rollback_opened(directory_fd, opened)
        raise
    finally:
        for child in opened:
            try:
                os.close(child.descriptor)
            except OSError:
                pass
        if directory_fd is not None:
            try:
                os.close(directory_fd)
            except OSError:
                pass
        if failed and root_created:
            try:
                root.rmdir()
            except (FileNotFoundError, OSError):
                pass
    return tuple(root / filename.value for filename, _payload in payloads)


def _prepare_payloads(
    streams: Mapping[OutputFilename, Iterable[object]],
) -> tuple[tuple[OutputFilename, bytes], ...]:
    payloads: list[tuple[OutputFilename, bytes]] = []
    seen: set[OutputFilename] = set()
    for filename, rows in streams.items():
        if not isinstance(filename, OutputFilename):
            raise InputContractError(
                "must be a typed output filename", path="output.filename"
            )
        if filename in seen:
            raise InputContractError("must be unique", path="output.filename")
        seen.add(filename)
        payload = encode_jsonl(rows)
        if payload:
            payloads.append((filename, payload))
    return tuple(payloads)


def _open_direct_child(directory_fd: int, filename: OutputFilename) -> _OpenedChild:
    flags = os.O_WRONLY | os.O_APPEND | getattr(os, "O_NONBLOCK", 0)
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    created = False
    try:
        try:
            descriptor = os.open(
                filename.value,
                flags | os.O_CREAT | os.O_EXCL,
                0o666,
                dir_fd=directory_fd,
            )
            created = True
        except FileExistsError:
            descriptor = os.open(filename.value, flags, dir_fd=directory_fd)
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.EISDIR, errno.ENODEV, errno.ENXIO}:
            raise InputContractError(
                "must be a single-link regular direct-child file",
                path=f"output.{filename.value}",
            ) from None
        raise
    try:
        status = os.fstat(descriptor)
        if not _is_regular_file(status.st_mode) or status.st_nlink != 1:
            raise InputContractError(
                "must be a single-link regular direct-child file",
                path=f"output.{filename.value}",
            )
        return _OpenedChild(
            filename=filename,
            descriptor=descriptor,
            original_size=0 if created else status.st_size,
            created=created,
            device=status.st_dev,
            inode=status.st_ino,
        )
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        if created:
            try:
                os.unlink(filename.value, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        raise


def _is_regular_file(mode: int) -> bool:
    import stat

    return stat.S_ISREG(mode)


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("output write made no progress")
        view = view[written:]


def _rollback_opened(
    directory_fd: int,
    opened: Iterable[_OpenedChild],
) -> None:
    for child in reversed(tuple(opened)):
        if child.created:
            _unlink_created_child(directory_fd, child)
        else:
            try:
                os.ftruncate(child.descriptor, child.original_size)
            except OSError:
                pass


def _unlink_created_child(directory_fd: int, child: _OpenedChild) -> None:
    try:
        status = os.stat(
            child.filename.value, dir_fd=directory_fd, follow_symlinks=False
        )
    except FileNotFoundError:
        return
    if (status.st_dev, status.st_ino) != (child.device, child.inode):
        return
    try:
        os.unlink(child.filename.value, dir_fd=directory_fd)
    except FileNotFoundError:
        pass
