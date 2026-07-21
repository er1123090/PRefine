"""Descriptor-relative, race-checked manifest primitives.

This module is loaded only from bytes authenticated by run_protected.py.
"""
import base64
import errno
import hashlib
import json
import os
import stat


def canon(obj):
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def stat_tuple(st):
    # atime is observation-induced and is deliberately excluded.
    return (st.st_mode, st.st_dev, st.st_ino, st.st_nlink, st.st_uid, st.st_gid,
            st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def identity(st):
    return {
        "mode": st.st_mode, "dev": st.st_dev, "inode": st.st_ino,
        "nlink": st.st_nlink, "uid": st.st_uid, "gid": st.st_gid,
        "size": st.st_size, "mtime_ns": st.st_mtime_ns,
        "ctime_ns": st.st_ctime_ns,
    }


def _b64(data):
    return base64.b64encode(data).decode("ascii")


def safe_components(path):
    if not isinstance(path, str) or not os.path.isabs(path):
        raise RuntimeError("unsafe absolute path")
    parts = os.fsencode(path).split(b"/")[1:]
    if not parts or any(p in (b"", b".", b"..") or b"/" in p or b"\0" in p for p in parts):
        raise RuntimeError("unsafe absolute path")
    return parts


def open_regular_at(dir_fd, name, flags=os.O_RDONLY, opener=os.open):
    """Open a regular-file candidate; retry O_NOATIME only for EPERM."""
    base = flags | os.O_CLOEXEC | os.O_NOFOLLOW
    noatime = getattr(os, "O_NOATIME", 0)
    try:
        return opener(name, base | noatime, dir_fd=dir_fd)
    except OSError as exc:
        if noatime and exc.errno == errno.EPERM:
            return opener(name, base, dir_fd=dir_fd)
        raise


def open_from_root(path, final_flags, expect, opener=os.open):
    """Component-wise O_NOFOLLOW walk from / with final type validation."""
    parts = safe_components(path)
    fd = opener("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for part in parts[:-1]:
            nxt = opener(part, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                         dir_fd=fd)
            os.close(fd)
            fd = nxt
        if expect == "regular":
            out = open_regular_at(fd, parts[-1], final_flags, opener)
        else:
            out = opener(parts[-1], final_flags | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=fd)
        st = os.fstat(out)
        if expect == "dir" and not stat.S_ISDIR(st.st_mode):
            os.close(out)
            raise RuntimeError("expected directory")
        if expect == "regular" and not stat.S_ISREG(st.st_mode):
            os.close(out)
            raise RuntimeError("expected regular file")
        return out
    finally:
        os.close(fd)


def stable_read_fd(fd, expected=None, hook=None, on_chunk=None):
    pre = os.fstat(fd)
    if not stat.S_ISREG(pre.st_mode):
        raise RuntimeError("expected regular file")
    if expected is not None and stat_tuple(pre) != stat_tuple(expected):
        raise RuntimeError("path substitution")
    if hook:
        hook("after_fstat", fd)
    os.lseek(fd, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    chunks = []
    total = 0
    while True:
        chunk = os.read(fd, 1 << 20)
        if not chunk:
            break
        chunks.append(chunk)
        digest.update(chunk)
        total += len(chunk)
        if on_chunk:
            on_chunk(len(chunk))
        if hook:
            hook("read", fd)
    post = os.fstat(fd)
    if stat_tuple(pre) != stat_tuple(post) or total != pre.st_size:
        raise RuntimeError("unstable file")
    if hook:
        hook("after_read", fd)
    return b"".join(chunks), digest.hexdigest(), total, pre


def _plain(raw):
    """Return a reversible JSON display spelling for arbitrary POSIX bytes."""
    text = raw.decode("utf-8", "surrogateescape")
    if text.encode("utf-8", "surrogateescape") != raw:
        raise RuntimeError("path does not round-trip")
    return text


def _entry_type(mode):
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISREG(mode):
        return "regular"
    if stat.S_ISLNK(mode):
        return "symlink"
    return "other"


def _record_base(record_type, root_id, root_path, raw_path, st):
    return {
        "schema": "g0-source-record/v5", "record_type": record_type,
        "root_id": root_id, "root_path": root_path,
        "record_id": sha(root_id.encode("utf-8") + b"\0" + raw_path),
        "path": _plain(raw_path), "path_b64": _b64(raw_path),
        "entry_type": _entry_type(st.st_mode), **identity(st),
    }


def scan_root_fd(root_id, root_path, root_fd, event, hooks=None):
    """Scan from an already pinned root FD; every access crosses event()."""
    hooks = hooks or {}
    root_pre = os.fstat(root_fd)
    root_raw = os.fsencode(root_path)
    records = [_record_base("root", root_id, root_path, root_raw, root_pre)]

    def emit(operation, stage, raw_path, st, result="ok", err=0, byte_count=0,
             content_sha256=None):
        event(root_id, raw_path, operation, stage, st, result, err, byte_count,
              content_sha256)

    def before(operation, raw_path, st=None):
        emit(operation, "before", raw_path, st, "pending", 0, 0)

    def after(operation, raw_path, st=None, result="ok", err=0, byte_count=0,
              content_sha256=None):
        emit(operation, "after", raw_path, st, result, err, byte_count, content_sha256)

    def walk(fd, rel):
        dir_before = os.fstat(fd)
        event_path = root_raw if rel == b"" else rel
        before("enumerate", event_path, dir_before)
        try:
            names = os.listdir(fd)
        except OSError as exc:
            after("enumerate", event_path, dir_before, "error", exc.errno, 0)
            raise
        after("enumerate", event_path, dir_before, "ok", 0, 0)
        for name in sorted(names, key=os.fsencode):
            raw_name = os.fsencode(name)
            raw_rel = raw_name if not rel else rel + b"/" + raw_name
            before("lstat", raw_rel)
            try:
                lst = os.stat(name, dir_fd=fd, follow_symlinks=False)
            except OSError as exc:
                after("lstat", raw_rel, None, "error", exc.errno, 0)
                raise
            after("lstat", raw_rel, lst)
            typ = _entry_type(lst.st_mode)
            rec = _record_base("entry", root_id, root_path, raw_rel, lst)
            if hooks.get("before_open"):
                hooks["before_open"](root_id, raw_rel, fd, name, lst)
            if typ == "directory":
                before("open_directory", raw_rel, lst)
                try:
                    child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                                    dir_fd=fd)
                except OSError as exc:
                    after("open_directory", raw_rel, None, "error", exc.errno, 0)
                    raise
                try:
                    actual = os.fstat(child)
                    after("open_directory", raw_rel, actual)
                    if stat_tuple(actual) != stat_tuple(lst):
                        raise RuntimeError("path substitution")
                    records.append(rec)
                    walk(child, raw_rel)
                finally:
                    os.close(child)
            elif typ == "regular":
                before("open_regular", raw_rel, lst)
                try:
                    child = open_regular_at(fd, name)
                except OSError as exc:
                    after("open_regular", raw_rel, None, "error", exc.errno, 0)
                    raise
                try:
                    actual = os.fstat(child)
                    after("open_regular", raw_rel, actual)
                    if stat_tuple(actual) != stat_tuple(lst):
                        raise RuntimeError("path substitution")
                    before("read_file", raw_rel, actual)
                    data, digest, count, stable = stable_read_fd(
                        child, lst, hooks.get("read_hook"),
                        None,
                    )
                    del data
                    rec["sha256"] = digest
                    rec["byte_count"] = count
                    after("read_file", raw_rel, stable, "ok", 0, count, digest)
                    before("fstat_post", raw_rel)
                    after("fstat_post", raw_rel, os.fstat(child))
                finally:
                    os.close(child)
                records.append(rec)
            elif typ == "symlink":
                before("readlink", raw_rel, lst)
                try:
                    target = os.readlink(name, dir_fd=fd)
                except OSError as exc:
                    after("readlink", raw_rel, lst, "error", exc.errno, 0)
                    raise
                target_raw = os.fsencode(target)
                rec["target"] = _plain(target_raw)
                rec["target_b64"] = _b64(target_raw)
                after("readlink", raw_rel, lst, "ok", 0, len(target_raw), sha(target_raw))
                records.append(rec)
            else:
                records.append(rec)
        dir_after = os.fstat(fd)
        before("directory_fstat_post", event_path)
        after("directory_fstat_post", event_path, dir_after)
        if stat_tuple(dir_before) != stat_tuple(dir_after):
            raise RuntimeError("unstable directory")

    walk(root_fd, b"")

    # Close the after-read race window: re-resolve and re-read every accepted
    # descendant only after the complete traversal has finished.
    for rec in records[1:]:
        raw_rel = base64.b64decode(rec["path_b64"], validate=True)
        parts = raw_rel.split(b"/")
        dfd = os.dup(root_fd)
        try:
            for component in parts[:-1]:
                nxt = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC |
                              os.O_NOFOLLOW, dir_fd=dfd)
                os.close(dfd)
                dfd = nxt
            name = parts[-1]
            before("final_revalidate", raw_rel)
            lst = os.stat(name, dir_fd=dfd, follow_symlinks=False)
            expected = (rec["mode"], rec["dev"], rec["inode"], rec["nlink"], rec["uid"],
                        rec["gid"], rec["size"], rec["mtime_ns"], rec["ctime_ns"])
            if stat_tuple(lst) != expected:
                raise RuntimeError("final path revalidation")
            if rec["entry_type"] == "regular":
                child = open_regular_at(dfd, name)
                try:
                    _, digest, count, stable = stable_read_fd(child, lst)
                finally:
                    os.close(child)
                if digest != rec["sha256"] or count != rec["byte_count"]:
                    raise RuntimeError("final file revalidation")
                after("final_revalidate", raw_rel, stable, "ok", 0, count, digest)
            elif rec["entry_type"] == "directory":
                child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC |
                                os.O_NOFOLLOW, dir_fd=dfd)
                try:
                    actual = os.fstat(child)
                finally:
                    os.close(child)
                if stat_tuple(actual) != expected:
                    raise RuntimeError("final directory revalidation")
                after("final_revalidate", raw_rel, actual)
            elif rec["entry_type"] == "symlink":
                target = os.fsencode(os.readlink(name, dir_fd=dfd))
                if _b64(target) != rec["target_b64"]:
                    raise RuntimeError("final symlink revalidation")
                after("final_revalidate", raw_rel, lst, "ok", 0, len(target), sha(target))
            else:
                after("final_revalidate", raw_rel, lst)
        finally:
            os.close(dfd)
    root_post = os.fstat(root_fd)
    if stat_tuple(root_pre) != stat_tuple(root_post):
        raise RuntimeError("unstable root")
    return records


def scan_root(root_id, root_path, event, hooks=None):
    fd = open_from_root(root_path, os.O_RDONLY | os.O_DIRECTORY, "dir")
    try:
        return scan_root_fd(root_id, root_path, fd, event, hooks), identity(os.fstat(fd))
    finally:
        os.close(fd)


def manifest_bytes(records):
    def raw_full_path(record):
        raw = base64.b64decode(record["path_b64"], validate=True)
        if record["record_type"] == "root":
            return raw
        return os.fsencode(record["root_path"]) + b"/" + raw

    keys = [raw_full_path(record) for record in records]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate manifest")
    ordered = sorted(records, key=raw_full_path)
    return b"".join(canon(rec) for rec in ordered)
