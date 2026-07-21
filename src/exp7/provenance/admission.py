"""Stable, symlink-safe admission of regular input files."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import stat
from typing import Any, Iterator, TextIO


class FileAdmissionError(RuntimeError):
    """Raised when an input cannot be bound to one stable byte sequence."""


class StrictJSONError(ValueError):
    """Raised when admitted bytes are not unambiguous strict JSON."""


@dataclass(frozen=True)
class AdmittedFile:
    """Bytes and identity captured from one stable file descriptor."""

    path: Path
    payload: bytes

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()


_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_FILE_READ_FLAGS = (
    os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
)


def _directory_identity(value: os.stat_result) -> tuple[int, int, int]:
    return (value.st_dev, value.st_ino, value.st_mode)


@dataclass
class _DirectoryChain:
    entries: list[tuple[Path, int, tuple[int, int, int]]]

    @property
    def path(self) -> Path:
        return self.entries[-1][0]

    @property
    def descriptor(self) -> int:
        return self.entries[-1][1]

    def verify(self, *, label: str) -> None:
        for path, _, expected in self.entries:
            try:
                current = os.lstat(path)
            except OSError as exc:
                raise FileAdmissionError(
                    f"{label} parent changed while being used: {path}: {exc}"
                ) from exc
            if _directory_identity(current) != expected:
                raise FileAdmissionError(
                    f"{label} parent changed while being used: {path}"
                )

    def close(self) -> None:
        for _, descriptor, _ in reversed(self.entries):
            os.close(descriptor)
        self.entries.clear()


def _open_directory_chain(
    path: Path,
    *,
    label: str,
    create: bool = False,
    exclusive_final: bool = False,
) -> _DirectoryChain:
    absolute = lexical_absolute(path)
    if not absolute.is_absolute() or not absolute.parts:
        raise FileAdmissionError(f"{label} must be an absolute directory path")
    entries: list[tuple[Path, int, tuple[int, int, int]]] = []
    try:
        current_path = Path(absolute.anchor)
        root_descriptor = os.open(absolute.anchor, _DIRECTORY_FLAGS)
        root_value = os.fstat(root_descriptor)
        entries.append(
            (current_path, root_descriptor, _directory_identity(root_value))
        )
        for index, part in enumerate(absolute.parts[1:], start=1):
            current_path = current_path / part
            final = index == len(absolute.parts) - 1
            try:
                before = os.stat(
                    part,
                    dir_fd=entries[-1][1],
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                if not create:
                    raise FileAdmissionError(
                        f"cannot inspect {label} directory {current_path}: missing"
                    )
                try:
                    os.mkdir(part, 0o755, dir_fd=entries[-1][1])
                except OSError as exc:
                    raise FileAdmissionError(
                        f"cannot create {label} directory {current_path}: {exc}"
                    ) from exc
                before = os.stat(
                    part,
                    dir_fd=entries[-1][1],
                    follow_symlinks=False,
                )
            else:
                if final and exclusive_final:
                    raise FileExistsError(
                        f"refusing to overwrite directory: {absolute}"
                    )
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                raise FileAdmissionError(
                    f"{label} parent must be a non-symlink directory: {current_path}"
                )
            try:
                descriptor = os.open(
                    part,
                    _DIRECTORY_FLAGS,
                    dir_fd=entries[-1][1],
                )
            except OSError as exc:
                raise FileAdmissionError(
                    f"cannot safely open {label} parent {current_path}: {exc}"
                ) from exc
            opened = os.fstat(descriptor)
            if _directory_identity(opened) != _directory_identity(before):
                os.close(descriptor)
                raise FileAdmissionError(
                    f"{label} parent changed while being opened: {current_path}"
                )
            entries.append(
                (current_path, descriptor, _directory_identity(opened))
            )
        chain = _DirectoryChain(entries)
        chain.verify(label=label)
        return chain
    except BaseException:
        for _, descriptor, _ in reversed(entries):
            os.close(descriptor)
        raise


def _safe_relative(value: str | Path, *, label: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or not relative.parts or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise FileAdmissionError(f"{label} must be a safe relative path: {value}")
    return relative


class AdmittedDirectory:
    """Descriptor-bound directory used for contained reads and writes."""

    def __init__(self, chain: _DirectoryChain, *, label: str):
        self._chain = chain
        self.label = label

    @property
    def path(self) -> Path:
        return self._chain.path

    @property
    def identity(self) -> tuple[int, int, int]:
        return self._chain.entries[-1][2]

    @property
    def descriptor_path(self) -> Path:
        """Return a Linux path view pinned to the admitted directory descriptor."""

        self.verify()
        path = Path("/proc/self/fd") / str(self._chain.descriptor)
        try:
            observed = os.stat(path)
        except OSError as exc:
            raise FileAdmissionError(
                f"cannot expose descriptor-bound {self.label}: {exc}"
            ) from exc
        if _directory_identity(observed) != self.identity:
            raise FileAdmissionError(
                f"descriptor-bound {self.label} identity changed: {self.path}"
            )
        return path

    def relative(self, path: Path) -> Path:
        absolute = lexical_absolute(path)
        try:
            return _safe_relative(
                absolute.relative_to(self.path),
                label=self.label,
            )
        except ValueError as exc:
            raise FileAdmissionError(
                f"path escapes admitted {self.label}: {absolute}"
            ) from exc

    def verify(self) -> None:
        self._chain.verify(label=self.label)

    def close(self) -> None:
        if self._chain.entries:
            self._chain.close()

    def __enter__(self) -> "AdmittedDirectory":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        verification_error: BaseException | None = None
        try:
            self.verify()
        except BaseException as caught:
            verification_error = caught
        finally:
            self.close()
        if exc_type is None and verification_error is not None:
            raise verification_error

    def _parent_chain(
        self,
        relative: str | Path,
        *,
        label: str,
        create: bool,
    ) -> tuple[Path, list[tuple[Path, int, tuple[int, int, int]]]]:
        self.verify()
        safe = _safe_relative(relative, label=label)
        entries: list[tuple[Path, int, tuple[int, int, int]]] = []
        descriptor = os.dup(self._chain.descriptor)
        entries.append((self.path, descriptor, self.identity))
        current_path = self.path
        try:
            for part in safe.parts[:-1]:
                current_path = current_path / part
                try:
                    before = os.stat(
                        part,
                        dir_fd=entries[-1][1],
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    if not create:
                        raise FileAdmissionError(
                            f"cannot inspect {label} parent {current_path}: missing"
                        )
                    os.mkdir(part, 0o755, dir_fd=entries[-1][1])
                    before = os.stat(
                        part,
                        dir_fd=entries[-1][1],
                        follow_symlinks=False,
                    )
                if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                    raise FileAdmissionError(
                        f"{label} parent must be a non-symlink directory: {current_path}"
                    )
                child = os.open(part, _DIRECTORY_FLAGS, dir_fd=entries[-1][1])
                opened = os.fstat(child)
                if _directory_identity(opened) != _directory_identity(before):
                    os.close(child)
                    raise FileAdmissionError(
                        f"{label} parent changed while being opened: {current_path}"
                    )
                entries.append(
                    (current_path, child, _directory_identity(opened))
                )
            return safe, entries
        except BaseException:
            for _, opened_descriptor, _ in reversed(entries):
                os.close(opened_descriptor)
            raise

    def ensure_directory(self, relative: str | Path, *, label: str) -> None:
        marker = _safe_relative(relative, label=label) / ".directory-marker"
        _, entries = self._parent_chain(marker, label=label, create=True)
        try:
            self._verify_relative_entries(entries, label=label)
            self.verify()
        finally:
            for _, descriptor, _ in reversed(entries):
                os.close(descriptor)

    def _verify_relative_entries(
        self,
        entries: list[tuple[Path, int, tuple[int, int, int]]],
        *,
        label: str,
    ) -> None:
        for index, (path, descriptor, expected) in enumerate(entries):
            current = os.fstat(descriptor)
            if _directory_identity(current) != expected:
                raise FileAdmissionError(
                    f"{label} parent changed while being used: {path}"
                )
            if index == 0:
                continue
            try:
                named = os.stat(
                    path.name,
                    dir_fd=entries[index - 1][1],
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise FileAdmissionError(
                    f"{label} parent changed while being used: {path}: {exc}"
                ) from exc
            if (
                stat.S_ISLNK(named.st_mode)
                or _directory_identity(named) != expected
            ):
                raise FileAdmissionError(
                    f"{label} parent changed while being used: {path}"
                )

    def _leaf_stat(
        self,
        relative: str | Path,
        *,
        label: str,
    ) -> tuple[os.stat_result | None, list[tuple[Path, int, tuple[int, int, int]]], Path]:
        safe, entries = self._parent_chain(relative, label=label, create=False)
        try:
            try:
                value = os.stat(
                    safe.name,
                    dir_fd=entries[-1][1],
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                value = None
            return value, entries, safe
        except BaseException:
            for _, descriptor, _ in reversed(entries):
                os.close(descriptor)
            raise

    def exists(self, relative: str | Path, *, label: str) -> bool:
        try:
            value, entries, _ = self._leaf_stat(relative, label=label)
        except FileAdmissionError as exc:
            if ": missing" in str(exc):
                return False
            raise
        try:
            return value is not None
        finally:
            for _, descriptor, _ in reversed(entries):
                os.close(descriptor)

    def admit_file(self, relative: str | Path, *, label: str) -> AdmittedFile:
        safe, entries = self._parent_chain(relative, label=label, create=False)
        try:
            return _read_leaf(
                self.path / safe,
                safe.name,
                entries,
                label=label,
            )
        finally:
            for _, descriptor, _ in reversed(entries):
                os.close(descriptor)

    def write_exclusive(
        self,
        relative: str | Path,
        payload: bytes,
        *,
        label: str,
    ) -> None:
        safe, entries = self._parent_chain(relative, label=label, create=True)
        descriptor: int | None = None
        try:
            flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            descriptor = os.open(
                safe.name,
                flags,
                0o644,
                dir_fd=entries[-1][1],
            )
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = None
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.fsync(entries[-1][1])
            self._verify_relative_entries(entries, label=label)
            self.verify()
        finally:
            if descriptor is not None:
                os.close(descriptor)
            for _, parent_descriptor, _ in reversed(entries):
                os.close(parent_descriptor)

    def atomic_write(
        self,
        relative: str | Path,
        payload: bytes,
        *,
        label: str,
    ) -> None:
        safe, entries = self._parent_chain(relative, label=label, create=True)
        temporary_name = f".{safe.name}.{secrets.token_hex(12)}"
        descriptor: int | None = None
        try:
            try:
                current = os.stat(
                    safe.name,
                    dir_fd=entries[-1][1],
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                current = None
            if current is not None and (
                stat.S_ISLNK(current.st_mode) or not stat.S_ISREG(current.st_mode)
            ):
                raise FileAdmissionError(
                    f"{label} must be a non-symlink regular file: {self.path / safe}"
                )
            descriptor = os.open(
                temporary_name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o644,
                dir_fd=entries[-1][1],
            )
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = None
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(
                temporary_name,
                safe.name,
                src_dir_fd=entries[-1][1],
                dst_dir_fd=entries[-1][1],
            )
            os.fsync(entries[-1][1])
            self._verify_relative_entries(entries, label=label)
            self.verify()
        except BaseException:
            try:
                os.unlink(temporary_name, dir_fd=entries[-1][1])
            except OSError:
                pass
            raise
        finally:
            if descriptor is not None:
                os.close(descriptor)
            for _, parent_descriptor, _ in reversed(entries):
                os.close(parent_descriptor)

    @contextmanager
    def open_text(
        self,
        relative: str | Path,
        *,
        label: str,
        append: bool,
        exclusive: bool = False,
    ) -> Iterator[TextIO]:
        safe, entries = self._parent_chain(relative, label=label, create=True)
        flags = os.O_WRONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        if append:
            flags |= os.O_APPEND
        if exclusive:
            flags |= os.O_CREAT | os.O_EXCL
        descriptor: int | None = None
        try:
            descriptor = os.open(
                safe.name,
                flags,
                0o644,
                dir_fd=entries[-1][1],
            )
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode):
                raise FileAdmissionError(
                    f"{label} must be a non-symlink regular file: {self.path / safe}"
                )
            with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
                descriptor = None
                yield handle
            current = os.stat(
                safe.name,
                dir_fd=entries[-1][1],
                follow_symlinks=False,
            )
            if (
                stat.S_ISLNK(current.st_mode)
                or (current.st_dev, current.st_ino, current.st_mode)
                != (opened.st_dev, opened.st_ino, opened.st_mode)
            ):
                raise FileAdmissionError(
                    f"{label} changed while being written: {self.path / safe}"
                )
            self._verify_relative_entries(entries, label=label)
            self.verify()
        finally:
            if descriptor is not None:
                os.close(descriptor)
            for _, parent_descriptor, _ in reversed(entries):
                os.close(parent_descriptor)


def admit_directory(path: Path, *, label: str) -> AdmittedDirectory:
    return AdmittedDirectory(
        _open_directory_chain(path, label=label),
        label=label,
    )


def create_directory(path: Path, *, label: str) -> AdmittedDirectory:
    return AdmittedDirectory(
        _open_directory_chain(
            path,
            label=label,
            create=True,
            exclusive_final=True,
        ),
        label=label,
    )


def admit_or_create_directory(path: Path, *, label: str) -> AdmittedDirectory:
    return AdmittedDirectory(
        _open_directory_chain(path, label=label, create=True),
        label=label,
    )


def lexical_absolute(path: Path, *, root: Path | None = None) -> Path:
    """Return an absolute lexical path without resolving symlinks."""

    candidate = path if path.is_absolute() else (root or Path.cwd()) / path
    return Path(os.path.abspath(candidate))


def _fingerprint(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_leaf(
    path: Path,
    leaf_name: str,
    parents: list[tuple[Path, int, tuple[int, int, int]]],
    *,
    label: str,
) -> AdmittedFile:
    try:
        before = os.stat(
            leaf_name,
            dir_fd=parents[-1][1],
            follow_symlinks=False,
        )
    except OSError as exc:
        raise FileAdmissionError(f"cannot inspect {label} {path}: {exc}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise FileAdmissionError(
            f"{label} must be a non-symlink regular file: {path}"
        )
    try:
        descriptor = os.open(
            leaf_name,
            _FILE_READ_FLAGS,
            dir_fd=parents[-1][1],
        )
    except OSError as exc:
        raise FileAdmissionError(f"cannot open {label} {path}: {exc}") from exc
    try:
        opened = os.fstat(descriptor)
        if _fingerprint(opened) != _fingerprint(before):
            raise FileAdmissionError(f"{label} changed while being opened: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        current = os.stat(
            leaf_name,
            dir_fd=parents[-1][1],
            follow_symlinks=False,
        )
    except OSError as exc:
        raise FileAdmissionError(f"cannot read {label} {path}: {exc}") from exc
    finally:
        os.close(descriptor)
    if (
        _fingerprint(after) != _fingerprint(opened)
        or _fingerprint(current) != _fingerprint(opened)
    ):
        raise FileAdmissionError(f"{label} changed while being read: {path}")
    for parent_path, _, expected in parents:
        try:
            observed = os.lstat(parent_path)
        except OSError as exc:
            raise FileAdmissionError(
                f"{label} parent changed while being read: {parent_path}: {exc}"
            ) from exc
        if _directory_identity(observed) != expected:
            raise FileAdmissionError(
                f"{label} parent changed while being read: {parent_path}"
            )
    return AdmittedFile(path=path, payload=b"".join(chunks))


def admit_regular_file(path: Path, *, label: str) -> AdmittedFile:
    """Read a non-symlink regular file once and reject identity/content drift."""

    admitted_path = lexical_absolute(path)
    parents = _open_directory_chain(admitted_path.parent, label=label)
    try:
        return _read_leaf(
            admitted_path,
            admitted_path.name,
            parents.entries,
            label=label,
        )
    finally:
        parents.close()


def decode_strict_json(payload: bytes, *, label: str) -> Any:
    """Decode UTF-8 JSON while rejecting duplicate keys and non-finite numbers."""

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise StrictJSONError(
                    f"{label} contains duplicate object key {key!r}"
                )
            value[key] = item
        return value

    def reject_constant(value: str) -> Any:
        raise StrictJSONError(f"{label} contains non-finite number {value}")

    def decode_float(value: str) -> float:
        decoded = float(value)
        if not math.isfinite(decoded):
            raise StrictJSONError(f"{label} contains non-finite number {value}")
        return decoded

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise StrictJSONError(f"{label} is not valid UTF-8 JSON") from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
            parse_float=decode_float,
        )
    except json.JSONDecodeError as exc:
        raise StrictJSONError(f"{label} is not valid JSON: {exc}") from exc
