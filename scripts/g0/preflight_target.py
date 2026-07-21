#!/usr/bin/env python3
"""Foreign-preserving experiments7 bootstrap using only no-follow namespace metadata."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import stat
import sys
import uuid
from datetime import datetime, timezone

SCHEMA = "experiments7-owner/v1"
README_BYTES = b"# experiments7\n\nSee [docs/README.md](docs/README.md).\n"
MANAGED_PREFIXES = (
    "src", "configs", "data", "scripts", "tests", "docs", "variants",
    "lineage", "runs", "paper_outputs", "manifests",
)
OWNER_NAME = ".experiments7-owner.json"
README_NAME = "README.md"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def kind(mode: int) -> str:
    if stat.S_ISDIR(mode): return "directory"
    if stat.S_ISREG(mode): return "regular"
    if stat.S_ISLNK(mode): return "symlink"
    return "other"


def identity(st: os.stat_result) -> dict[str, object]:
    return {
        "type": kind(st.st_mode), "dev": st.st_dev, "inode": st.st_ino,
        "mode": stat.S_IMODE(st.st_mode), "uid": st.st_uid, "gid": st.st_gid,
        "size": st.st_size, "nlink": st.st_nlink,
        "mtime_ns": st.st_mtime_ns, "ctime_ns": st.st_ctime_ns,
    }


def entry_identity(fd: int, name: str) -> dict[str, object]:
    st = os.stat(name, dir_fd=fd, follow_symlinks=False)
    result = identity(st)
    result["name_b64"] = base64.b64encode(os.fsencode(name)).decode("ascii")
    return result


def list_metadata(fd: int) -> list[dict[str, object]]:
    return [entry_identity(fd, n) for n in sorted(os.listdir(fd), key=os.fsencode)]


def check_real_dir(path: str) -> os.stat_result:
    st = os.lstat(path)
    if not stat.S_ISDIR(st.st_mode) or os.path.realpath(path) != path:
        raise RuntimeError(f"not an exact real directory: {path}")
    return st


def fsync_dir(fd: int) -> None:
    os.fsync(fd)


def write_exclusive(fd: int, name: str, data: bytes, mode: int = 0o644) -> os.stat_result:
    out = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, mode, dir_fd=fd)
    try:
        view = memoryview(data)
        while view:
            count = os.write(out, view)
            if count <= 0: raise OSError("short write")
            view = view[count:]
        os.fsync(out)
        st = os.fstat(out)
    finally:
        os.close(out)
    return st


def exact_resume(root_fd: int, target_st: os.stat_result, paper_record: dict[str, object]) -> dict[str, object]:
    owner_fd = os.open(OWNER_NAME, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=root_fd)
    try:
        raw = b""
        while True:
            chunk = os.read(owner_fd, 1 << 20)
            if not chunk: break
            raw += chunk
        owner_st = os.fstat(owner_fd)
    finally:
        os.close(owner_fd)
    record = json.loads(raw)
    if record.get("schema") != SCHEMA or record.get("state") != "complete":
        raise RuntimeError("mismatched or incomplete owner record")
    binding = record.get("target_binding", {})
    if binding.get("dev") != target_st.st_dev or binding.get("inode") != target_st.st_ino:
        raise RuntimeError("target identity differs from owner record")
    if record.get("paper", {}).get("identity") != paper_record["identity"]:
        raise RuntimeError("paper namespace identity differs from owner record")
    for name in MANAGED_PREFIXES:
        st = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        expected = record.get("managed_prefixes", {}).get(name)
        if not stat.S_ISDIR(st.st_mode) or expected != identity(st):
            raise RuntimeError(f"managed prefix resume mismatch: {name}")
    readme_fd = os.open(README_NAME, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=root_fd)
    try:
        raw_readme = os.read(readme_fd, len(README_BYTES) + 1)
        readme_st = os.fstat(readme_fd)
    finally:
        os.close(readme_fd)
    if raw_readme != README_BYTES or record.get("root_readme", {}).get("identity") != identity(readme_st):
        raise RuntimeError("root README resume mismatch")
    return {"status": "resume_validated", "owner": record, "owner_identity": identity(owner_st)}


def bootstrap(target: str, paper: str, sealed_run_id: str) -> dict[str, object]:
    target = os.path.abspath(target)
    paper = os.path.abspath(paper)
    if target != "/data/minseo/experiments7":
        raise RuntimeError("target must be exact canonical experiments7 path")
    expected_paper_parent = os.path.join(target, "_paper")
    if os.path.dirname(paper) != expected_paper_parent:
        raise RuntimeError("paper must be the exact supplied path directly beneath _paper")
    for ancestor in ("/data", "/data/minseo", target):
        check_real_dir(ancestor)
    target_st = check_real_dir(target)
    paper_dir_st = check_real_dir(expected_paper_parent)
    paper_st = os.lstat(paper)
    if not stat.S_ISREG(paper_st.st_mode) or os.path.realpath(paper) != paper:
        raise RuntimeError("paper is not an exact regular file")

    root_fd = os.open(target, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    paper_dir_fd = os.open("_paper", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=root_fd)
    created_dirs: list[tuple[str, tuple[int, int]]] = []
    created_files: list[tuple[str, tuple[int, int]]] = []
    tmp_name = f".{OWNER_NAME}.tmp.{sealed_run_id}"
    try:
        if os.fstat(root_fd).st_ino != target_st.st_ino or os.fstat(paper_dir_fd).st_ino != paper_dir_st.st_ino:
            raise RuntimeError("namespace substitution during bootstrap")
        root_entries = list_metadata(root_fd)
        paper_entries = list_metadata(paper_dir_fd)
        paper_by_name = {base64.b64decode(e["name_b64"]): e for e in paper_entries}
        paper_name_b = os.fsencode(os.path.basename(paper))
        if paper_name_b not in paper_by_name or paper_by_name[paper_name_b] != dict(identity(paper_st), name_b64=base64.b64encode(paper_name_b).decode("ascii")):
            raise RuntimeError("paper identity changed during namespace scan")
        paper_record = {
            "path": paper, "ownership": "preserved_foreign_readonly",
            "identity": identity(paper_st),
        }
        names = {os.fsdecode(base64.b64decode(e["name_b64"])) for e in root_entries}
        if OWNER_NAME in names:
            return exact_resume(root_fd, target_st, paper_record)
        collisions = names.intersection(set(MANAGED_PREFIXES) | {README_NAME})
        if collisions:
            raise RuntimeError(f"managed-prefix collision in ownerless target: {sorted(collisions)!r}")
        if any(n.startswith(f".{OWNER_NAME}.tmp.") for n in names):
            raise RuntimeError("foreign or interrupted owner temp collision")

        for name in MANAGED_PREFIXES:
            os.mkdir(name, 0o755, dir_fd=root_fd)
            st = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
            created_dirs.append((name, (st.st_dev, st.st_ino)))
        readme_st = write_exclusive(root_fd, README_NAME, README_BYTES)
        created_files.append((README_NAME, (readme_st.st_dev, readme_st.st_ino)))

        managed = {name: identity(os.stat(name, dir_fd=root_fd, follow_symlinks=False)) for name in MANAGED_PREFIXES}
        owner = {
            "schema": SCHEMA, "state": "complete", "project_uuid": str(uuid.uuid4()),
            "sealed_run_id": sealed_run_id, "created_at": now(),
            "target_path": target, "target_binding": identity(os.fstat(root_fd)),
            "owned_root_files": [OWNER_NAME, README_NAME],
            "managed_prefixes": managed,
            "root_readme": {"sha256": hashlib.sha256(README_BYTES).hexdigest(), "size": len(README_BYTES), "identity": identity(readme_st)},
            "foreign_root_entries": [e for e in root_entries if base64.b64decode(e["name_b64"]) == b"_paper" or os.fsdecode(base64.b64decode(e["name_b64"])) not in MANAGED_PREFIXES],
            "paper_directory": {"path": expected_paper_parent, "ownership": "preserved_foreign_readonly", "identity": identity(os.fstat(paper_dir_fd)), "entries": paper_entries},
            "paper": paper_record,
            "ownership_exclusions": [expected_paper_parent, paper],
        }
        owner_bytes = canonical(owner)
        tmp_st = write_exclusive(root_fd, tmp_name, owner_bytes, 0o644)
        try:
            os.link(tmp_name, OWNER_NAME, src_dir_fd=root_fd, dst_dir_fd=root_fd, follow_symlinks=False)
        except Exception:
            os.unlink(tmp_name, dir_fd=root_fd)
            raise
        os.unlink(tmp_name, dir_fd=root_fd)
        fsync_dir(root_fd)
        owner_st = os.stat(OWNER_NAME, dir_fd=root_fd, follow_symlinks=False)
        if owner_st.st_ino != tmp_st.st_ino:
            raise RuntimeError("owner no-replace publication identity mismatch")
        return {"status": "bootstrapped", "owner": owner, "owner_identity": identity(owner_st)}
    except Exception:
        # Roll back only objects whose identity was created by this invocation.
        try:
            os.unlink(tmp_name, dir_fd=root_fd)
        except OSError:
            pass
        for name, expected in reversed(created_files):
            try:
                st = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
                if (st.st_dev, st.st_ino) == expected: os.unlink(name, dir_fd=root_fd)
            except OSError:
                pass
        for name, expected in reversed(created_dirs):
            try:
                st = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
                if (st.st_dev, st.st_ino) == expected: os.rmdir(name, dir_fd=root_fd)
            except OSError:
                pass
        raise
    finally:
        os.close(paper_dir_fd)
        os.close(root_fd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--foreign-readonly-prefix", default="_paper")
    ap.add_argument("--paper", required=True)
    ap.add_argument("--sealed-run-id", required=True)
    args = ap.parse_args()
    if args.foreign_readonly_prefix != "_paper":
        raise SystemExit("only exact _paper is supported")
    try:
        result = bootstrap(args.target, args.paper, args.sealed_run_id)
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(2)
    sys.stdout.buffer.write(canonical(result))


if __name__ == "__main__":
    main()
