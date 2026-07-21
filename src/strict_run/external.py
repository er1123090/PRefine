"""Descriptor-bound immutable registry for out-of-tree canonical JSON transcripts."""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .canonical import (
    StrictRunError,
    canonical_json,
    fail,
    require_sha256,
    sha256_bytes,
    validate_absolute_path_text,
    validate_sealed_run_id,
)
from .filesystem import (
    DIR_FLAGS,
    READ_FLAGS,
    RunDescriptorBinding,
    assert_path_matches_fd,
    mount_identity,
    open_absolute_directory,
    open_bound_run_handle,
    open_run_handle,
    stable_identity,
    validate_run_root_text,
)


DEFAULT_MAX_DOCUMENT_BYTES = 64 * 1024 * 1024
EXTERNAL_READ_FLAGS = READ_FLAGS | os.O_NONBLOCK
_ARRAY_INDEX = re.compile(r"0|[1-9][0-9]*", re.ASCII)


def _full_identity(st: os.stat_result) -> dict[str, object]:
    identity = stable_identity(st)
    identity.update(
        {
            "nlink": st.st_nlink,
            "mtime_ns": st.st_mtime_ns,
            "ctime_ns": st.st_ctime_ns,
        }
    )
    return identity


def _inode_key(st: os.stat_result) -> tuple[int, int]:
    return (st.st_dev, st.st_ino)


def _require_expected_mount_id(
    fd: int, expected_mount_id: int | None, label: str
) -> None:
    if expected_mount_id is None:
        return
    if type(expected_mount_id) is not int or expected_mount_id <= 0:
        fail("EXTERNAL_MOUNT_SUBSTITUTION", "expected mount ID is invalid")
    if mount_identity(fd)["mount_id"] != expected_mount_id:
        fail(
            "EXTERNAL_MOUNT_SUBSTITUTION",
            f"{label} descriptor mount differs from preflight",
        )


def _fd_path(fd: int) -> str:
    try:
        path = os.readlink(f"/proc/self/fd/{fd}")
    except OSError as exc:
        raise StrictRunError(
            "EXTERNAL_DESCRIPTOR_UNAVAILABLE",
            "cannot resolve an external document descriptor",
        ) from exc
    if path.endswith(" (deleted)"):
        fail("EXTERNAL_DOCUMENT_MUTATED", "external descriptor target was deleted")
    return path


def _assert_exact_fd_path(path: str, fd: int, field: str) -> None:
    if _fd_path(fd) != path or os.path.realpath(path) != path:
        fail("NONCANONICAL_EXTERNAL_PATH", f"{field} descriptor path differs")


def _inside(path: str, root: str) -> bool:
    try:
        return os.path.commonpath((path, root)) == root
    except ValueError:
        return False


def _validate_document_limit(value: object) -> int:
    if type(value) is not int or not 0 < value <= 1 << 30:
        fail("EXTERNAL_LIMIT_INVALID", "external document limit is invalid")
    return value


def _decode_canonical_json(raw: bytes, field: str) -> object:
    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                fail("DUPLICATE_JSON_KEY", f"{field} contains a duplicate object key")
            result[key] = value
        return result

    def invalid_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON number: {value}")

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=invalid_constant,
        )
    except StrictRunError:
        raise
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise StrictRunError(
            "EXTERNAL_JSON_INVALID", f"{field} is not strict UTF-8 JSON"
        ) from exc
    try:
        encoded = canonical_json(value)
    except (TypeError, ValueError, RecursionError) as exc:
        raise StrictRunError(
            "EXTERNAL_JSON_INVALID", f"{field} cannot be canonically encoded"
        ) from exc
    if encoded != raw:
        fail("NONCANONICAL_JSON", f"{field} bytes are not canonical JSON")
    return value


def _normalize_pointer(value: object) -> tuple[str | int, ...]:
    if isinstance(value, str):
        if value == "":
            return ()
        if not value.startswith("/"):
            fail("JSON_POINTER_INVALID", "string JSON pointer must begin with slash")
        tokens: list[str | int] = []
        for raw in value.split("/")[1:]:
            if re.search(r"~(?:[^01]|$)", raw):
                fail("JSON_POINTER_INVALID", "JSON pointer escape is invalid")
            tokens.append(raw.replace("~1", "/").replace("~0", "~"))
        normalized = tuple(tokens)
    elif isinstance(value, (tuple, list)):
        normalized = tuple(value)
    else:
        fail("JSON_POINTER_INVALID", "JSON pointer must be text or a segment sequence")
    if len(normalized) > 256:
        fail("JSON_POINTER_INVALID", "JSON pointer has too many segments")
    for segment in normalized:
        if type(segment) is int:
            if segment < 0:
                fail("JSON_POINTER_INVALID", "JSON pointer index is negative")
        elif isinstance(segment, str):
            if "\x00" in segment or len(segment.encode("utf-8")) > 4096:
                fail("JSON_POINTER_INVALID", "JSON pointer segment is invalid")
        else:
            fail(
                "JSON_POINTER_INVALID",
                "JSON pointer segments must be strings or exact integers",
            )
    return normalized


@dataclass(frozen=True)
class ExternalTranscriptRecord:
    path: str
    json_pointer: tuple[str | int, ...] | str | list[str | int] = ()
    expected_sha256: str | None = None
    expected_size: int | None = None
    expected_identity: Mapping[str, object] | None = None
    expected_parent_identity: Mapping[str, object] | None = None
    expected_mount_id: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "path", validate_absolute_path_text(self.path, "external document")
        )
        object.__setattr__(self, "json_pointer", _normalize_pointer(self.json_pointer))
        pins = (
            self.expected_sha256,
            self.expected_size,
            self.expected_identity,
            self.expected_parent_identity,
            self.expected_mount_id,
        )
        if all(value is None for value in pins):
            return
        if any(value is None for value in pins):
            fail(
                "EXTERNAL_EXPECTATION_INVALID",
                "external transcript publication evidence must be complete",
            )
        assert self.expected_sha256 is not None
        assert self.expected_size is not None
        assert self.expected_identity is not None
        assert self.expected_parent_identity is not None
        assert self.expected_mount_id is not None
        require_sha256(self.expected_sha256, "external expected sha256")
        if type(self.expected_size) is not int or self.expected_size < 0:
            fail("EXTERNAL_EXPECTATION_INVALID", "external expected size is invalid")
        if (
            not isinstance(self.expected_identity, Mapping)
            or set(self.expected_identity) != {
                "type",
                "dev",
                "inode",
                "mode",
                "uid",
                "gid",
                "size",
                "nlink",
                "mtime_ns",
                "ctime_ns",
            }
            or self.expected_identity.get("type") != "regular"
            or not isinstance(self.expected_parent_identity, Mapping)
            or set(self.expected_parent_identity) != {
                "type",
                "dev",
                "inode",
                "mode",
                "uid",
                "gid",
                "nlink",
                "mtime_ns",
                "ctime_ns",
            }
            or self.expected_parent_identity.get("type") != "directory"
            or type(self.expected_mount_id) is not int
            or self.expected_mount_id <= 0
        ):
            fail(
                "EXTERNAL_EXPECTATION_INVALID",
                "external expected descriptor identity is invalid",
            )
        object.__setattr__(
            self,
            "expected_identity",
            MappingProxyType(dict(self.expected_identity)),
        )
        object.__setattr__(
            self,
            "expected_parent_identity",
            MappingProxyType(dict(self.expected_parent_identity)),
        )

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "path": self.path,
            "json_pointer": list(self.json_pointer),
        }
        if self.expected_identity is not None:
            result["expected_publication"] = {
                "sha256": self.expected_sha256,
                "bytes": self.expected_size,
                "identity": dict(self.expected_identity),
                "parent_identity": dict(self.expected_parent_identity or {}),
                "mount_id": self.expected_mount_id,
            }
        return result


def _open_parent_chain(path: str) -> tuple[int, str, tuple[tuple[int, int], ...]]:
    parent = os.path.dirname(path)
    name = os.path.basename(path)
    if not name:
        fail("INVALID_PATH", "external document basename is empty")
    fd = os.open("/", DIR_FLAGS)
    expected = "/"
    ancestors: list[tuple[int, int]] = []
    try:
        _assert_exact_fd_path(expected, fd, "external ancestor")
        ancestors.append(_inode_key(os.fstat(fd)))
        for component in parent.split("/")[1:]:
            next_fd = os.open(component, DIR_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = next_fd
            expected = f"/{component}" if expected == "/" else f"{expected}/{component}"
            _assert_exact_fd_path(expected, fd, "external ancestor")
            ancestors.append(_inode_key(os.fstat(fd)))
        if expected != parent or os.path.realpath(parent) != parent:
            fail("NONCANONICAL_EXTERNAL_PATH", "external parent path differs")
        return fd, name, tuple(ancestors)
    except Exception as exc:
        os.close(fd)
        if isinstance(exc, StrictRunError):
            raise
        raise StrictRunError(
            "UNSAFE_EXTERNAL_PATH",
            "external document ancestor is missing, linked, or unsafe",
        ) from exc


class ExternalDocument:
    """One canonical external JSON document held by its original descriptors."""

    __slots__ = (
        "path",
        "parent_path",
        "sha256",
        "_raw_bytes",
        "_identity",
        "_parent_identity",
        "_file_fd",
        "_parent_fd",
        "_closed",
    )

    def __init__(
        self,
        *,
        path: str,
        raw_bytes: bytes,
        digest: str,
        identity: dict[str, object],
        parent_identity: dict[str, object],
        file_fd: int,
        parent_fd: int,
    ) -> None:
        self.path = path
        self.parent_path = os.path.dirname(path)
        self.sha256 = digest
        self._raw_bytes = bytes(raw_bytes)
        self._identity = MappingProxyType(dict(identity))
        self._parent_identity = MappingProxyType(dict(parent_identity))
        self._file_fd = file_fd
        self._parent_fd = parent_fd
        self._closed = False

    @property
    def raw_bytes(self) -> bytes:
        return self._raw_bytes

    @property
    def identity(self) -> Mapping[str, object]:
        return self._identity

    @property
    def parent_identity(self) -> Mapping[str, object]:
        return self._parent_identity

    @property
    def value(self) -> object:
        self.assert_current()
        return _decode_canonical_json(self._raw_bytes, "cached external document")

    @property
    def json(self) -> object:
        return self.value

    @property
    def document(self) -> object:
        return self.value

    def evidence(self) -> dict[str, object]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "bytes": len(self._raw_bytes),
            "identity": dict(self._identity),
            "parent_identity": dict(self._parent_identity),
        }

    def assert_current(self) -> None:
        if self._closed:
            fail("EXTERNAL_DOCUMENT_CLOSED", "external document is closed")
        try:
            file_stat = os.fstat(self._file_fd)
            parent_stat = os.fstat(self._parent_fd)
            current = os.stat(
                os.path.basename(self.path),
                dir_fd=self._parent_fd,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise StrictRunError(
                "EXTERNAL_DOCUMENT_MUTATED",
                "external document descriptor or namespace changed",
            ) from exc
        if (
            _full_identity(file_stat) != dict(self._identity)
            or _full_identity(current) != dict(self._identity)
            or _full_identity(parent_stat) != dict(self._parent_identity)
            or not stat.S_ISREG(file_stat.st_mode)
            or file_stat.st_nlink != 1
        ):
            fail("EXTERNAL_DOCUMENT_MUTATED", "external document identity changed")
        _assert_exact_fd_path(self.path, self._file_fd, "external document")
        _assert_exact_fd_path(self.parent_path, self._parent_fd, "external parent")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        os.close(self._file_fd)
        os.close(self._parent_fd)

    def __enter__(self) -> "ExternalDocument":
        self.assert_current()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @classmethod
    def load_pre_reservation(
        cls,
        path: str,
        *,
        strict_parent: str,
        sealed_run_id: str,
        run_root: str,
        max_document_bytes: int = DEFAULT_MAX_DOCUMENT_BYTES,
    ) -> "ExternalDocument":
        return load_pre_reservation_document(
            path,
            strict_parent=strict_parent,
            sealed_run_id=sealed_run_id,
            run_root=run_root,
            max_document_bytes=max_document_bytes,
        )


def _load_external_document(
    path: str,
    *,
    forbidden_inodes: frozenset[tuple[int, int]],
    forbidden_roots: tuple[str, ...],
    max_document_bytes: int,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
    expected_identity: Mapping[str, object] | None = None,
    expected_parent_identity: Mapping[str, object] | None = None,
    expected_mount_id: int | None = None,
) -> ExternalDocument:
    path = validate_absolute_path_text(path, "external document")
    max_document_bytes = _validate_document_limit(max_document_bytes)
    if any(_inside(path, root) for root in forbidden_roots):
        fail("TRANSCRIPT_IN_RUN_TREE", "external document is inside a strict tree")
    parent_fd = -1
    file_fd = -1
    try:
        parent_fd, name, ancestors = _open_parent_chain(path)
        if any(identity in forbidden_inodes for identity in ancestors):
            fail("EXTERNAL_ANCESTOR_ALIAS", "external ancestor aliases strict storage")
        parent_before = os.fstat(parent_fd)
        file_fd = os.open(name, EXTERNAL_READ_FLAGS, dir_fd=parent_fd)
        before = os.fstat(file_fd)
        namespace_before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or _full_identity(namespace_before) != _full_identity(before)
        ):
            fail("EXTERNAL_NOT_REGULAR", "external document is not a stable regular file")
        if before.st_nlink != 1:
            fail("EXTERNAL_HARDLINK", "external document must have exactly one link")
        if _inode_key(before) in forbidden_inodes:
            fail("EXTERNAL_TREE_ALIAS", "external document aliases strict tree content")
        _assert_exact_fd_path(path, file_fd, "external document")
        if expected_identity is not None:
            _require_expected_mount_id(
                parent_fd, expected_mount_id, "external document parent"
            )
            _require_expected_mount_id(
                file_fd, expected_mount_id, "external document"
            )
            if (
                expected_parent_identity is None
                or expected_sha256 is None
                or expected_size is None
                or _full_identity(parent_before)
                != dict(expected_parent_identity)
                or _full_identity(before) != dict(expected_identity)
                or _full_identity(namespace_before) != dict(expected_identity)
                or before.st_size != expected_size
            ):
                fail(
                    "EXTERNAL_DOCUMENT_IDENTITY_MISMATCH",
                    "external document differs from publication evidence",
                )

        chunks: list[bytes] = []
        digest = hashlib.sha256()
        total = 0
        while True:
            block = os.read(file_fd, 1024 * 1024)
            if not block:
                break
            total += len(block)
            if total > max_document_bytes:
                fail("EXTERNAL_DOCUMENT_TOO_LARGE", "external document exceeds its limit")
            chunks.append(block)
            digest.update(block)
        raw = b"".join(chunks)
        after = os.fstat(file_fd)
        parent_after = os.fstat(parent_fd)
        namespace_after = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            _full_identity(after) != _full_identity(before)
            or _full_identity(namespace_after) != _full_identity(before)
            or _full_identity(parent_after) != _full_identity(parent_before)
            or total != after.st_size
            or (
                expected_sha256 is not None
                and (
                    digest.hexdigest() != expected_sha256
                    or total != expected_size
                )
            )
        ):
            fail("EXTERNAL_DOCUMENT_MUTATED", "external document changed while loading")
        _decode_canonical_json(raw, "external document")
        document = ExternalDocument(
            path=path,
            raw_bytes=raw,
            digest=digest.hexdigest(),
            identity=_full_identity(after),
            parent_identity=_full_identity(parent_after),
            file_fd=file_fd,
            parent_fd=parent_fd,
        )
        file_fd = parent_fd = -1
        try:
            document.assert_current()
        except Exception:
            document.close()
            raise
        return document
    except OSError as exc:
        raise StrictRunError(
            "UNSAFE_EXTERNAL_DOCUMENT",
            "external document could not be opened without following links",
        ) from exc
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        if parent_fd >= 0:
            os.close(parent_fd)


class _StrictTreeSnapshot:
    __slots__ = (
        "strict_parent",
        "run_root",
        "inode_keys",
        "_fds",
        "_identities",
        "_parent_fd",
        "_root_fd",
        "_closed",
    )

    def __init__(
        self,
        *,
        strict_parent: str,
        run_root: str,
        inode_keys: frozenset[tuple[int, int]],
        fds: tuple[int, ...],
        identities: tuple[dict[str, object], ...],
        parent_fd: int,
        root_fd: int,
    ) -> None:
        self.strict_parent = strict_parent
        self.run_root = run_root
        self.inode_keys = inode_keys
        self._fds = fds
        self._identities = identities
        self._parent_fd = parent_fd
        self._root_fd = root_fd
        self._closed = False

    def assert_current(self) -> None:
        if self._closed:
            fail("STRICT_TREE_SNAPSHOT_CLOSED", "strict tree snapshot is closed")
        assert_path_matches_fd(self.strict_parent, self._parent_fd)
        assert_path_matches_fd(self.run_root, self._root_fd)
        for fd, expected in zip(self._fds, self._identities):
            if _full_identity(os.fstat(fd)) != expected:
                fail("STRICT_TREE_MUTATED", "strict tree directory changed after snapshot")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for fd in reversed(self._fds):
            os.close(fd)


def _snapshot_strict_tree(
    strict_parent: str,
    sealed_run_id: str,
    run_root: str,
    run_binding: RunDescriptorBinding | None = None,
) -> _StrictTreeSnapshot:
    handle = (
        open_run_handle(strict_parent, sealed_run_id, run_root)
        if run_binding is None
        else open_bound_run_handle(
            strict_parent, sealed_run_id, run_root, run_binding
        )
    )
    fds: list[int] = [handle.parent_fd, handle.root_fd]
    identities: list[dict[str, object]] = [
        _full_identity(os.fstat(handle.parent_fd)),
        _full_identity(os.fstat(handle.root_fd)),
    ]
    inode_keys: set[tuple[int, int]] = {
        _inode_key(os.fstat(handle.parent_fd)),
        _inode_key(os.fstat(handle.root_fd)),
    }

    def scan(directory_fd: int) -> None:
        before = _full_identity(os.fstat(directory_fd))
        with os.scandir(directory_fd) as entries:
            names = sorted(entry.name for entry in entries)
        for name in names:
            current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            inode_keys.add(_inode_key(current))
            if stat.S_ISDIR(current.st_mode):
                child_fd = os.open(name, DIR_FLAGS, dir_fd=directory_fd)
                child = os.fstat(child_fd)
                if _full_identity(child) != _full_identity(current):
                    os.close(child_fd)
                    fail("STRICT_TREE_MUTATED", "strict directory changed during scan")
                fds.append(child_fd)
                identities.append(_full_identity(child))
                scan(child_fd)
            elif not stat.S_ISREG(current.st_mode):
                fail("UNSAFE_STRICT_TREE", "strict tree contains a linked/special entry")
        if _full_identity(os.fstat(directory_fd)) != before:
            fail("STRICT_TREE_MUTATED", "strict directory changed during scan")

    try:
        scan(handle.root_fd)
        snapshot = _StrictTreeSnapshot(
            strict_parent=handle.strict_parent,
            run_root=handle.run_root,
            inode_keys=frozenset(inode_keys),
            fds=tuple(fds),
            identities=tuple(identities),
            parent_fd=handle.parent_fd,
            root_fd=handle.root_fd,
        )
        snapshot.assert_current()
        return snapshot
    except Exception:
        for fd in reversed(fds):
            os.close(fd)
        raise


def _resolve_pointer(
    document: object, pointer: tuple[str | int, ...]
) -> tuple[object, tuple[tuple[str, str | int], ...]]:
    current = document
    resolved: list[tuple[str, str | int]] = []
    for token in pointer:
        if isinstance(current, dict):
            if not isinstance(token, str) or token not in current:
                fail("JSON_POINTER_MISSING", "JSON pointer object key is absent")
            current = current[token]
            resolved.append(("key", token))
        elif isinstance(current, list):
            if type(token) is int:
                index = token
            elif isinstance(token, str) and _ARRAY_INDEX.fullmatch(token):
                index = int(token)
            else:
                fail("JSON_POINTER_INVALID", "JSON pointer list index is not canonical")
            if index >= len(current):
                fail("JSON_POINTER_MISSING", "JSON pointer list index is out of range")
            current = current[index]
            resolved.append(("index", index))
        else:
            fail("JSON_POINTER_MISSING", "JSON pointer traverses a scalar")
    if not isinstance(current, dict):
        fail("TRANSCRIPT_POINTER_NOT_OBJECT", "transcript pointer must select an object")
    return current, tuple(resolved)


class ExternalTranscriptRegistry(Mapping[str, object]):
    """Immutable digest-to-canonical-transcript index backed by held documents."""

    def __init__(
        self,
        records: Iterable[ExternalTranscriptRecord],
        *,
        strict_parent: str,
        sealed_run_id: str,
        run_root: str,
        max_document_bytes: int = DEFAULT_MAX_DOCUMENT_BYTES,
        run_binding: RunDescriptorBinding | None = None,
    ) -> None:
        sealed_run_id = validate_sealed_run_id(sealed_run_id)
        strict_parent, run_root = validate_run_root_text(
            strict_parent, sealed_run_id, run_root
        )
        max_document_bytes = _validate_document_limit(max_document_bytes)
        rows = tuple(records)
        if not rows or any(not isinstance(row, ExternalTranscriptRecord) for row in rows):
            fail("EXTERNAL_RECORDS_INVALID", "registry records must be nonempty typed rows")
        if run_binding is not None and not isinstance(
            run_binding, RunDescriptorBinding
        ):
            fail("RUN_BINDING_INVALID", "external registry run binding is not typed")

        snapshot = _snapshot_strict_tree(
            strict_parent, sealed_run_id, run_root, run_binding
        )
        documents: dict[str, ExternalDocument] = {}
        canonical_bytes: dict[str, bytes] = {}
        digest_documents: dict[str, str] = {}
        pointer_index: dict[
            tuple[str, tuple[tuple[str, str | int], ...]], str
        ] = {}
        try:
            for path in dict.fromkeys(row.path for row in rows):
                path_rows = tuple(row for row in rows if row.path == path)
                first = path_rows[0]
                expected = (
                    first.expected_sha256,
                    first.expected_size,
                    None
                    if first.expected_identity is None
                    else dict(first.expected_identity),
                    None
                    if first.expected_parent_identity is None
                    else dict(first.expected_parent_identity),
                    first.expected_mount_id,
                )
                if any(
                    (
                        row.expected_sha256,
                        row.expected_size,
                        None
                        if row.expected_identity is None
                        else dict(row.expected_identity),
                        None
                        if row.expected_parent_identity is None
                        else dict(row.expected_parent_identity),
                        row.expected_mount_id,
                    )
                    != expected
                    for row in path_rows[1:]
                ):
                    fail(
                        "EXTERNAL_EXPECTATION_CONFLICT",
                        "one external path has conflicting publication evidence",
                    )
                document = _load_external_document(
                    path,
                    forbidden_inodes=snapshot.inode_keys,
                    forbidden_roots=(strict_parent, run_root),
                    max_document_bytes=max_document_bytes,
                    expected_sha256=first.expected_sha256,
                    expected_size=first.expected_size,
                    expected_identity=first.expected_identity,
                    expected_parent_identity=first.expected_parent_identity,
                    expected_mount_id=first.expected_mount_id,
                )
                duplicate_path = digest_documents.get(document.sha256)
                if duplicate_path is not None:
                    document.close()
                    fail(
                        "DUPLICATE_EXTERNAL_DOCUMENT_DIGEST",
                        "distinct external documents have identical bytes",
                    )
                documents[path] = document
                digest_documents[document.sha256] = path

            decoded_documents = {
                path: document.value for path, document in documents.items()
            }
            for record in rows:
                selected, resolved = _resolve_pointer(
                    decoded_documents[record.path], tuple(record.json_pointer)
                )
                pointer_key = (record.path, resolved)
                if pointer_key in pointer_index:
                    fail(
                        "DUPLICATE_TRANSCRIPT_POINTER",
                        "external transcript pointer is duplicated",
                    )
                payload = canonical_json(selected)
                digest = sha256_bytes(payload)
                if digest in canonical_bytes:
                    fail(
                        "DUPLICATE_TRANSCRIPT_DIGEST",
                        "external transcript digest is duplicated",
                    )
                pointer_index[pointer_key] = digest
                canonical_bytes[digest] = payload

            snapshot.assert_current()
            for document in documents.values():
                document.assert_current()
        except Exception:
            for document in documents.values():
                document.close()
            snapshot.close()
            raise

        self._snapshot = snapshot
        self._documents = MappingProxyType(documents)
        self._canonical_bytes = MappingProxyType(canonical_bytes)
        self._pointer_index = MappingProxyType(pointer_index)
        self._digests = tuple(canonical_bytes)
        self._closed = False

    @classmethod
    def load(
        cls,
        records: Iterable[ExternalTranscriptRecord],
        *,
        strict_parent: str,
        sealed_run_id: str,
        run_root: str,
        max_document_bytes: int = DEFAULT_MAX_DOCUMENT_BYTES,
        run_binding: RunDescriptorBinding | None = None,
    ) -> "ExternalTranscriptRegistry":
        return cls(
            records,
            strict_parent=strict_parent,
            sealed_run_id=sealed_run_id,
            run_root=run_root,
            max_document_bytes=max_document_bytes,
            run_binding=run_binding,
        )

    @classmethod
    def from_records(
        cls,
        records: Iterable[ExternalTranscriptRecord],
        *,
        strict_parent: str,
        sealed_run_id: str,
        run_root: str,
        max_document_bytes: int = DEFAULT_MAX_DOCUMENT_BYTES,
        run_binding: RunDescriptorBinding | None = None,
    ) -> "ExternalTranscriptRegistry":
        return cls.load(
            records,
            strict_parent=strict_parent,
            sealed_run_id=sealed_run_id,
            run_root=run_root,
            max_document_bytes=max_document_bytes,
            run_binding=run_binding,
        )

    @property
    def documents(self) -> Mapping[str, ExternalDocument]:
        return self._documents

    @property
    def canonical_bytes(self) -> Mapping[str, bytes]:
        return self._canonical_bytes

    @property
    def pointer_index(
        self,
    ) -> Mapping[tuple[str, tuple[tuple[str, str | int], ...]], str]:
        return self._pointer_index

    @property
    def digests(self) -> tuple[str, ...]:
        return self._digests

    def verify_current(self) -> None:
        if self._closed:
            fail("EXTERNAL_REGISTRY_CLOSED", "external transcript registry is closed")
        self._snapshot.assert_current()
        for document in self._documents.values():
            document.assert_current()

    def get_exact(self, digest: str) -> object:
        if self._closed:
            fail("EXTERNAL_REGISTRY_CLOSED", "external transcript registry is closed")
        digest = require_sha256(digest, "external transcript digest")
        self.verify_current()
        payload = self._canonical_bytes.get(digest)
        if payload is None:
            fail("TRANSCRIPT_NOT_REGISTERED", "external transcript digest is absent")
        if sha256_bytes(payload) != digest:
            fail("EXTERNAL_CACHE_CORRUPT", "cached transcript bytes do not match key")
        return _decode_canonical_json(payload, "cached external transcript")

    def __getitem__(self, digest: str) -> object:
        if digest not in self._canonical_bytes:
            raise KeyError(digest)
        return self.get_exact(digest)

    def __iter__(self) -> Iterator[str]:
        return iter(self._digests)

    def __len__(self) -> int:
        return len(self._digests)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for document in self._documents.values():
            document.close()
        self._snapshot.close()

    def __enter__(self) -> "ExternalTranscriptRegistry":
        self.verify_current()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _require_absent(parent_fd: int, name: str) -> None:
    try:
        os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise StrictRunError(
            "UNSAFE_EXTERNAL_OUTPUT",
            "external output namespace could not be inspected",
        ) from exc
    fail("EXTERNAL_OUTPUT_EXISTS", "external output path already exists")


@dataclass(frozen=True)
class ExternalOutputEvidence:
    path: str
    sha256: str
    size: int
    identity: dict[str, object]
    parent_identity: dict[str, object]


class ExternalOutputReservation:
    """Hold output and strict-tree identities through no-replace publication."""

    __slots__ = (
        "path",
        "parent_path",
        "strict_parent",
        "sealed_run_id",
        "run_root",
        "_name",
        "_parent_fd",
        "_strict_parent_fd",
        "_parent_identity",
        "_strict_parent_identity",
        "_run_root_fd",
        "_run_root_identity",
        "_run_binding",
        "_run_must_exist",
        "_ancestor_inodes",
        "_max_output_bytes",
        "_published",
        "_closed",
    )

    def __init__(
        self,
        *,
        path: str,
        strict_parent: str,
        sealed_run_id: str,
        run_root: str,
        name: str,
        parent_fd: int,
        strict_parent_fd: int,
        parent_identity: dict[str, object],
        strict_parent_identity: dict[str, object],
        run_root_fd: int,
        run_root_identity: dict[str, object] | None,
        run_binding: RunDescriptorBinding | None,
        run_must_exist: bool,
        ancestor_inodes: tuple[tuple[int, int], ...],
        max_output_bytes: int,
    ) -> None:
        self.path = path
        self.parent_path = os.path.dirname(path)
        self.strict_parent = strict_parent
        self.sealed_run_id = sealed_run_id
        self.run_root = run_root
        self._name = name
        self._parent_fd = parent_fd
        self._strict_parent_fd = strict_parent_fd
        self._parent_identity = MappingProxyType(dict(parent_identity))
        self._strict_parent_identity = MappingProxyType(
            dict(strict_parent_identity)
        )
        self._run_root_fd = run_root_fd
        self._run_root_identity = (
            None
            if run_root_identity is None
            else MappingProxyType(dict(run_root_identity))
        )
        self._run_binding = run_binding
        self._run_must_exist = run_must_exist
        self._ancestor_inodes = ancestor_inodes
        self._max_output_bytes = max_output_bytes
        self._published = False
        self._closed = False

    @classmethod
    def create(
        cls,
        path: str,
        *,
        strict_parent: str,
        sealed_run_id: str,
        run_root: str,
        max_output_bytes: int = DEFAULT_MAX_DOCUMENT_BYTES,
        expected_mount_id: int | None = None,
    ) -> "ExternalOutputReservation":
        sealed_run_id = validate_sealed_run_id(sealed_run_id)
        strict_parent, run_root = validate_run_root_text(
            strict_parent, sealed_run_id, run_root
        )
        max_output_bytes = _validate_document_limit(max_output_bytes)
        path = validate_absolute_path_text(path, "external output")
        if _inside(path, strict_parent) or _inside(path, run_root):
            fail("TRANSCRIPT_IN_RUN_TREE", "external output is in strict storage")

        parent_fd = -1
        strict_parent_fd = -1
        try:
            parent_fd, name, ancestors = _open_parent_chain(path)
            _require_expected_mount_id(
                parent_fd, expected_mount_id, "external parent"
            )
            parent_identity = _full_identity(os.fstat(parent_fd))
            _require_absent(parent_fd, name)
            strict_parent_fd = open_absolute_directory(strict_parent)
            _require_expected_mount_id(
                strict_parent_fd, expected_mount_id, "strict parent"
            )
            strict_parent_stat = os.fstat(strict_parent_fd)
            if _inode_key(strict_parent_stat) in ancestors:
                fail(
                    "EXTERNAL_ANCESTOR_ALIAS",
                    "external output ancestor aliases strict storage",
                )
            try:
                os.stat(
                    sealed_run_id,
                    dir_fd=strict_parent_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                fail("RUN_ALREADY_EXISTS", "planned strict child already exists")
            reservation = cls(
                path=path,
                strict_parent=strict_parent,
                sealed_run_id=sealed_run_id,
                run_root=run_root,
                name=name,
                parent_fd=parent_fd,
                strict_parent_fd=strict_parent_fd,
                parent_identity=parent_identity,
                strict_parent_identity=stable_identity(strict_parent_stat),
                run_root_fd=-1,
                run_root_identity=None,
                run_binding=None,
                run_must_exist=False,
                ancestor_inodes=ancestors,
                max_output_bytes=max_output_bytes,
            )
            parent_fd = strict_parent_fd = -1
            try:
                reservation.assert_preflight_current()
            except Exception:
                reservation.close()
                raise
            return reservation
        except Exception:
            if parent_fd >= 0:
                os.close(parent_fd)
            if strict_parent_fd >= 0:
                os.close(strict_parent_fd)
            raise

    @classmethod
    def create_for_existing_run(
        cls,
        path: str,
        *,
        strict_parent: str,
        sealed_run_id: str,
        run_root: str,
        max_output_bytes: int = DEFAULT_MAX_DOCUMENT_BYTES,
        expected_mount_id: int | None = None,
        expected_run_binding: RunDescriptorBinding | None = None,
    ) -> "ExternalOutputReservation":
        """Preflight an output while holding the current strict child identity."""

        sealed_run_id = validate_sealed_run_id(sealed_run_id)
        strict_parent, run_root = validate_run_root_text(
            strict_parent, sealed_run_id, run_root
        )
        max_output_bytes = _validate_document_limit(max_output_bytes)
        path = validate_absolute_path_text(path, "external output")
        if _inside(path, strict_parent) or _inside(path, run_root):
            fail("TRANSCRIPT_IN_RUN_TREE", "external output is in strict storage")
        if expected_run_binding is not None and not isinstance(
            expected_run_binding, RunDescriptorBinding
        ):
            fail("RUN_BINDING_INVALID", "external output run binding is not typed")

        parent_fd = -1
        strict_parent_fd = -1
        run_root_fd = -1
        snapshot: _StrictTreeSnapshot | None = None
        try:
            parent_fd, name, ancestors = _open_parent_chain(path)
            _require_expected_mount_id(
                parent_fd, expected_mount_id, "external parent"
            )
            parent_identity = stable_identity(os.fstat(parent_fd))
            _require_absent(parent_fd, name)

            handle = (
                open_run_handle(strict_parent, sealed_run_id, run_root)
                if expected_run_binding is None
                else open_bound_run_handle(
                    strict_parent,
                    sealed_run_id,
                    run_root,
                    expected_run_binding,
                )
            )
            strict_parent_fd = handle.parent_fd
            run_root_fd = handle.root_fd
            _require_expected_mount_id(
                strict_parent_fd, expected_mount_id, "strict parent"
            )
            _require_expected_mount_id(
                run_root_fd, expected_mount_id, "strict run root"
            )
            strict_parent_identity = stable_identity(os.fstat(strict_parent_fd))
            run_root_identity = stable_identity(os.fstat(run_root_fd))

            snapshot = _snapshot_strict_tree(
                strict_parent,
                sealed_run_id,
                run_root,
                expected_run_binding,
            )
            if (
                stable_identity(os.fstat(snapshot._parent_fd))
                != strict_parent_identity
                or stable_identity(os.fstat(snapshot._root_fd))
                != run_root_identity
            ):
                fail(
                    "STRICT_ROOT_MUTATED",
                    "strict parent or run root changed during output preflight",
                )
            if snapshot.inode_keys.intersection(ancestors):
                fail(
                    "EXTERNAL_OUTPUT_TREE_ALIAS",
                    "external output ancestor aliases the current strict tree",
                )
            snapshot.assert_current()
            snapshot.close()
            snapshot = None

            reservation = cls(
                path=path,
                strict_parent=strict_parent,
                sealed_run_id=sealed_run_id,
                run_root=run_root,
                name=name,
                parent_fd=parent_fd,
                strict_parent_fd=strict_parent_fd,
                parent_identity=parent_identity,
                strict_parent_identity=strict_parent_identity,
                run_root_fd=run_root_fd,
                run_root_identity=run_root_identity,
                run_binding=expected_run_binding,
                run_must_exist=True,
                ancestor_inodes=ancestors,
                max_output_bytes=max_output_bytes,
            )
            parent_fd = strict_parent_fd = run_root_fd = -1
            try:
                reservation.assert_preflight_current()
            except Exception:
                reservation.close()
                raise
            return reservation
        except Exception:
            if snapshot is not None:
                snapshot.close()
            if parent_fd >= 0:
                os.close(parent_fd)
            if run_root_fd >= 0:
                os.close(run_root_fd)
            if strict_parent_fd >= 0:
                os.close(strict_parent_fd)
            raise

    @classmethod
    def preflight(
        cls,
        path: str,
        *,
        strict_parent: str,
        sealed_run_id: str,
        run_root: str,
        max_output_bytes: int = DEFAULT_MAX_DOCUMENT_BYTES,
        expected_mount_id: int | None = None,
    ) -> "ExternalOutputReservation":
        return cls.create(
            path,
            strict_parent=strict_parent,
            sealed_run_id=sealed_run_id,
            run_root=run_root,
            max_output_bytes=max_output_bytes,
            expected_mount_id=expected_mount_id,
        )

    def _assert_open(self) -> None:
        if self._closed:
            fail("EXTERNAL_OUTPUT_CLOSED", "external output reservation is closed")
        if self._published:
            fail(
                "EXTERNAL_OUTPUT_ALREADY_PUBLISHED",
                "external output reservation is one-shot",
            )

    def _current_parent_identity(self) -> dict[str, object]:
        current = os.fstat(self._parent_fd)
        if self._run_must_exist:
            return stable_identity(current)
        return _full_identity(current)

    def _assert_bound_namespaces(self, *, require_planned_run_absent: bool) -> None:
        assert_path_matches_fd(self.parent_path, self._parent_fd)
        assert_path_matches_fd(self.strict_parent, self._strict_parent_fd)
        _require_absent(self._parent_fd, self._name)
        if self._current_parent_identity() != dict(self._parent_identity):
            fail("EXTERNAL_OUTPUT_PARENT_MUTATED", "external output parent changed")
        if stable_identity(os.fstat(self._strict_parent_fd)) != dict(
            self._strict_parent_identity
        ):
            fail("STRICT_PARENT_MUTATED", "strict parent identity changed")
        if self._run_must_exist:
            if self._run_root_identity is None or self._run_root_fd < 0:
                fail("STRICT_ROOT_MUTATED", "held strict run identity is absent")
            assert_path_matches_fd(self.run_root, self._run_root_fd)
            if stable_identity(os.fstat(self._run_root_fd)) != dict(
                self._run_root_identity
            ):
                fail("STRICT_ROOT_MUTATED", "strict run root identity changed")
            if self._run_binding is not None:
                self._run_binding.assert_fds(
                    self._strict_parent_fd,
                    self._run_root_fd,
                    strict_parent=self.strict_parent,
                    run_root=self.run_root,
                )
            return
        if not require_planned_run_absent:
            return
        try:
            os.stat(
                self.sealed_run_id,
                dir_fd=self._strict_parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return
        fail("RUN_ALREADY_EXISTS", "planned strict child appeared during preflight")

    def assert_preflight_current(self) -> None:
        self._assert_open()
        self._assert_bound_namespaces(require_planned_run_absent=True)

    def _snapshot_after_reservation(
        self,
        expected_run_binding: RunDescriptorBinding | None = None,
    ) -> _StrictTreeSnapshot:
        self._assert_open()
        if expected_run_binding is not None and not isinstance(
            expected_run_binding, RunDescriptorBinding
        ):
            fail("RUN_BINDING_INVALID", "external publication run binding is not typed")
        if (
            self._run_binding is not None
            and expected_run_binding is not None
            and self._run_binding != expected_run_binding
        ):
            fail(
                "RUN_BINDING_INVALID",
                "external publication run binding conflicts with preflight",
            )
        self._assert_bound_namespaces(require_planned_run_absent=False)
        effective_binding = (
            self._run_binding
            if expected_run_binding is None
            else expected_run_binding
        )
        snapshot = _snapshot_strict_tree(
            self.strict_parent,
            self.sealed_run_id,
            self.run_root,
            effective_binding,
        )
        if _inode_key(os.fstat(snapshot._parent_fd)) != _inode_key(
            os.fstat(self._strict_parent_fd)
        ):
            snapshot.close()
            fail("STRICT_PARENT_MUTATED", "strict parent descriptor differs")
        if self._run_must_exist and (
            self._run_root_identity is None
            or stable_identity(os.fstat(snapshot._root_fd))
            != dict(self._run_root_identity)
            or _inode_key(os.fstat(snapshot._root_fd))
            != _inode_key(os.fstat(self._run_root_fd))
        ):
            snapshot.close()
            fail("STRICT_ROOT_MUTATED", "strict run root descriptor differs")
        if snapshot.inode_keys.intersection(self._ancestor_inodes):
            snapshot.close()
            fail(
                "EXTERNAL_OUTPUT_TREE_ALIAS",
                "external output ancestor aliases the new strict tree",
            )
        return snapshot

    def publish_bytes(
        self,
        payload: bytes,
        *,
        expected_run_binding: RunDescriptorBinding | None = None,
    ) -> ExternalOutputEvidence:
        self._assert_open()
        if not isinstance(payload, bytes):
            fail("EXTERNAL_OUTPUT_INVALID", "external output payload must be bytes")
        if len(payload) > self._max_output_bytes:
            fail("EXTERNAL_OUTPUT_TOO_LARGE", "external output exceeds its limit")
        snapshot = self._snapshot_after_reservation(expected_run_binding)
        fd = -1
        read_fd = -1
        try:
            assert_path_matches_fd(self.parent_path, self._parent_fd)
            _require_absent(self._parent_fd, self._name)
            if self._current_parent_identity() != dict(self._parent_identity):
                fail(
                    "EXTERNAL_OUTPUT_PARENT_MUTATED",
                    "external output parent changed before open",
                )
            try:
                fd = os.open(
                    self._name,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | os.O_NOFOLLOW
                    | os.O_CLOEXEC,
                    0o444,
                    dir_fd=self._parent_fd,
                )
            except FileExistsError as exc:
                raise StrictRunError(
                    "EXTERNAL_OUTPUT_EXISTS",
                    "external output path won the no-replace race",
                ) from exc
            os.fchmod(fd, 0o444)
            view = memoryview(payload)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    fail("SHORT_WRITE", "external output write made no progress")
                view = view[written:]
            os.fsync(fd)
            written_stat = os.fstat(fd)
            if (
                not stat.S_ISREG(written_stat.st_mode)
                or stat.S_IMODE(written_stat.st_mode) != 0o444
                or written_stat.st_nlink != 1
                or written_stat.st_size != len(payload)
            ):
                fail("EXTERNAL_OUTPUT_INVALID", "external output identity differs")
            os.close(fd)
            fd = -1
            os.fsync(self._parent_fd)

            read_fd = os.open(
                self._name, EXTERNAL_READ_FLAGS, dir_fd=self._parent_fd
            )
            _assert_exact_fd_path(self.path, read_fd, "external output")
            before = os.fstat(read_fd)
            if _full_identity(before) != _full_identity(written_stat):
                fail(
                    "EXTERNAL_OUTPUT_SUBSTITUTED",
                    "external output identity changed before readback",
                )
            digest = hashlib.sha256()
            chunks: list[bytes] = []
            total = 0
            while True:
                block = os.read(read_fd, 1024 * 1024)
                if not block:
                    break
                total += len(block)
                digest.update(block)
                chunks.append(block)
            after = os.fstat(read_fd)
            namespace = os.stat(
                self._name,
                dir_fd=self._parent_fd,
                follow_symlinks=False,
            )
            if (
                _full_identity(before) != _full_identity(after)
                or _full_identity(namespace) != _full_identity(after)
                or not stat.S_ISREG(after.st_mode)
                or stat.S_IMODE(after.st_mode) != 0o444
                or after.st_nlink != 1
                or after.st_size != len(payload)
                or total != len(payload)
                or b"".join(chunks) != payload
                or digest.hexdigest() != sha256_bytes(payload)
                or _inode_key(after) in snapshot.inode_keys
            ):
                fail(
                    "EXTERNAL_OUTPUT_SUBSTITUTED",
                    "external output readback differs",
                )
            assert_path_matches_fd(self.parent_path, self._parent_fd)
            if stable_identity(os.fstat(self._parent_fd)) != stable_identity(
                os.stat(self.parent_path, follow_symlinks=False)
            ):
                fail("EXTERNAL_OUTPUT_PARENT_MUTATED", "output parent was replaced")
            snapshot.assert_current()
            self._published = True
            return ExternalOutputEvidence(
                self.path,
                digest.hexdigest(),
                total,
                _full_identity(after),
                _full_identity(os.fstat(self._parent_fd)),
            )
        except OSError as exc:
            raise StrictRunError(
                "EXTERNAL_OUTPUT_WRITE_FAILED",
                "external output publication failed",
            ) from exc
        finally:
            if fd >= 0:
                os.close(fd)
            if read_fd >= 0:
                os.close(read_fd)
            snapshot.close()

    def publish_json(
        self,
        value: object,
        *,
        expected_run_binding: RunDescriptorBinding | None = None,
    ) -> ExternalOutputEvidence:
        return self.publish_bytes(
            canonical_json(value),
            expected_run_binding=expected_run_binding,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        os.close(self._parent_fd)
        if self._run_root_fd >= 0:
            os.close(self._run_root_fd)
        os.close(self._strict_parent_fd)

    def __enter__(self) -> "ExternalOutputReservation":
        self.assert_preflight_current()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def load_pre_reservation_document(
    path: str,
    *,
    strict_parent: str,
    sealed_run_id: str,
    run_root: str,
    max_document_bytes: int = DEFAULT_MAX_DOCUMENT_BYTES,
) -> ExternalDocument:
    """Load one external input while the exact planned strict child is absent."""

    sealed_run_id = validate_sealed_run_id(sealed_run_id)
    strict_parent, run_root = validate_run_root_text(
        strict_parent, sealed_run_id, run_root
    )
    max_document_bytes = _validate_document_limit(max_document_bytes)
    path = validate_absolute_path_text(path, "external document")
    if _inside(path, run_root) or _inside(path, strict_parent):
        fail("TRANSCRIPT_IN_RUN_TREE", "pre-reservation input is in strict storage")
    parent_fd = open_absolute_directory(strict_parent)
    try:
        parent_identity = _full_identity(os.fstat(parent_fd))
        try:
            os.stat(sealed_run_id, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            fail("RUN_ALREADY_EXISTS", "planned strict child already exists")
        document = _load_external_document(
            path,
            forbidden_inodes=frozenset({_inode_key(os.fstat(parent_fd))}),
            forbidden_roots=(strict_parent, run_root),
            max_document_bytes=max_document_bytes,
        )
        try:
            assert_path_matches_fd(strict_parent, parent_fd)
            if _full_identity(os.fstat(parent_fd)) != parent_identity:
                fail("STRICT_PARENT_MUTATED", "strict parent changed during preflight")
            try:
                os.stat(sealed_run_id, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                return document
            fail("RUN_ALREADY_EXISTS", "planned strict child appeared during preflight")
        except Exception:
            document.close()
            raise
    finally:
        os.close(parent_fd)
