#!/usr/bin/env python3
"""G0v2 isolated conductor, executor, publisher, and hostile supervisor.

Only the Python standard library is imported at module load time. Project code is
loaded only after the complete six-member frozen bundle is authenticated and
exact-set checked by both the external launcher and this conductor.
"""
import argparse
import base64
import errno
import hashlib
import importlib.abc
import importlib.util
import io
import json
import os
import platform
import re
import resource
import secrets
import select
import stat
import subprocess
import sys
import sysconfig
import time
import types

sys.dont_write_bytecode = True

INPUT_NAMES = {
    "boundary.json", "owner-binding.json", "manifest-config.json",
    "argv-pre.json", "argv-post.json", "runtime-lock.json",
    "frozen-scripts.json", "bundle-lock.json",
}
SCRIPT_NAMES = {"run_protected.py", "provider.py", "manifest.py", "compare.py"}
FROZEN_NAMES = SCRIPT_NAMES | {"files.json", "bundle-lock.json"}
EVENT_KEYS = {
    "schema", "sequence", "phase", "sealed_run_id", "task_id", "role",
    "bundle_sha256", "envelope_id", "root_id", "record_id", "path", "path_b64",
    "operation", "stage", "result", "errno", "bytes_read", "record_identity",
    "content_sha256", "provider_ready",
}
IDENTITY_KEYS = {"mode", "dev", "inode", "nlink", "uid", "gid", "size",
                 "mtime_ns", "ctime_ns"}
READ_BYTE_OPERATIONS = {"read_file", "read_declaration", "readlink", "final_revalidate"}
WRITE_OPERATIONS = {"write_denial_control", "pdf_write_open_denial"}
ALLOWED_OPERATIONS = {
    "enumerate", "lstat", "open_directory", "open_regular", "read_file",
    "fstat_post", "readlink", "directory_fstat_post", "final_revalidate",
    "root_fd_validate", "root_path_revalidate", "read_declaration",
    "write_denial_control", "lstat_managed", "lstat_foreign", "lstat_post",
    "pdf_buffer_parse", "pdf_write_open_denial",
}
SOURCE_OPERATIONS = {"enumerate", "lstat", "open_directory", "open_regular",
                     "read_file", "fstat_post", "readlink", "directory_fstat_post",
                     "final_revalidate", "root_fd_validate", "root_path_revalidate"}
ROOT_OPERATION_MAP = {
    "paper": {"lstat", "open_regular", "read_file", "pdf_buffer_parse", "lstat_post",
              "pdf_write_open_denial"},
    "owner": {"lstat", "open_regular", "read_file", "lstat_managed", "lstat_foreign"},
    "denial-fixture": {"read_declaration", "write_denial_control"},
}
FINAL_INPUT_ORDER = [
    "boundary.json", "owner-binding.json", "manifest-config.json", "argv-pre.json",
    "argv-post.json", "runtime-lock.json", "frozen-scripts.json", "bundle-lock.json",
]
PRE_OUTPUT_ORDER = [
    "provider-evidence.json", "protected-access-pre.jsonl", "source-pre.jsonl",
    "paper-pre.json", "cp0-validation.json", "g0-task-evidence.json", "cp0-seal.json",
]
EXECUTOR_PRE_ORDER = [
    "provider-evidence.json", "protected-access-pre.jsonl", "source-pre.jsonl",
    "paper-pre.json",
]
EXECUTOR_POST_ORDER = ["protected-access-post.jsonl", "source-post.jsonl", "paper-post.json"]
PRE_ORDER = FINAL_INPUT_ORDER + PRE_OUTPUT_ORDER
POST_ORDER = [
    "protected-access-post.jsonl", "source-post.jsonl", "paper-post.json",
    "comparison.json", "protected-access-closure.json", "terminal-seal.json",
]
ENVIRONMENT = {
    "PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C", "TZ": "UTC",
    "PYTHONHASHSEED": "0",
}
TERMINAL_EVIDENCE_NAMES = {
    "copies.jsonl", "copy-seal.json", "mutating-test-seal.json",
    "final-admission-seal.json",
}
DENIAL_CONTROL_EVENT_ORDER = (
    "create", "open_truncate", "truncate", "unlink", "rename",
    "renameat2_noreplace", "mkdir", "rmdir", "chmod", "chown", "utime",
    "setxattr", "removexattr", "symlink", "hardlink",
)
RECONCILIATION_ANCHOR_KEYS = {
    "schema", "sealed_run_id", "bundle_id", "bundle_sha256",
    "boundary_sha256", "owner_binding_sha256", "provider_evidence_sha256",
    "source_manifest_sha256", "paper_record_sha256", "source_roots", "paper",
    "record_set_sha256", "operation_sequence_sha256", "operation_pair_count",
}
ROOT_ANCHOR_KEYS = {
    "root_id", "record_id", "path", "path_b64", *IDENTITY_KEYS,
}
PAPER_ANCHOR_KEYS = {
    "record_id", "path", "path_b64", "parent_path", "parent_dev", "parent_inode",
    *IDENTITY_KEYS, "sha256", "byte_count", "pages",
}


def canon(obj):
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def safe_name(name):
    if not isinstance(name, str) or not name or name in (".", "..") or "/" in name or "\0" in name:
        raise RuntimeError("unsafe component name")


def _write_all(fd, data):
    offset = 0
    while offset < len(data):
        count = os.write(fd, data[offset:])
        if count <= 0:
            raise OSError(errno.EIO, "short write")
        offset += count


def _open_dir(path):
    if not os.path.isabs(path):
        raise RuntimeError("directory path must be absolute")
    parts = os.fsencode(path).split(b"/")[1:]
    if not parts or any(part in (b"", b".", b"..") for part in parts):
        raise RuntimeError("unsafe directory path")
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for part in parts:
            nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                          dir_fd=fd)
            os.close(fd)
            fd = nxt
        return fd
    except Exception:
        os.close(fd)
        raise


def _open_regular_at(dfd, name, flags=os.O_RDONLY, opener=os.open):
    safe_name(os.fsdecode(name))
    base = flags | os.O_CLOEXEC | os.O_NOFOLLOW
    noatime = getattr(os, "O_NOATIME", 0)
    try:
        return opener(name, base | noatime, dir_fd=dfd)
    except OSError as exc:
        if noatime and exc.errno == errno.EPERM:
            return opener(name, base, dir_fd=dfd)
        raise


def _stable_read(fd, expected=None, hook=None):
    pre = os.fstat(fd)
    fields = lambda st: (st.st_mode, st.st_dev, st.st_ino, st.st_nlink, st.st_uid,
                         st.st_gid, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
    if not stat.S_ISREG(pre.st_mode):
        raise RuntimeError("expected regular file")
    if expected is not None and fields(pre) != fields(expected):
        raise RuntimeError("path substitution")
    os.lseek(fd, 0, os.SEEK_SET)
    chunks = []
    digest = hashlib.sha256()
    total = 0
    while True:
        chunk = os.read(fd, 1 << 20)
        if not chunk:
            break
        chunks.append(chunk)
        digest.update(chunk)
        total += len(chunk)
        if hook:
            hook(len(chunk), pre)
    post = os.fstat(fd)
    if fields(pre) != fields(post) or total != pre.st_size:
        raise RuntimeError("unstable file")
    return b"".join(chunks), digest.hexdigest(), total, pre


def _read_at(dfd, name):
    safe_name(name)
    lst = os.stat(name, dir_fd=dfd, follow_symlinks=False)
    fd = _open_regular_at(dfd, name)
    try:
        return _stable_read(fd, lst)
    finally:
        os.close(fd)


def _read_rel_at(root_fd, relative):
    parts = relative.split("/")
    if not parts or any(not part or part in (".", "..") for part in parts):
        raise RuntimeError("unsafe relative component path")
    dfd = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                          dir_fd=dfd)
            os.close(dfd)
            dfd = nxt
        return _read_at(dfd, parts[-1])
    finally:
        os.close(dfd)


def _load_canon_at(dfd, name):
    data, digest, size, st = _read_at(dfd, name)
    try:
        obj = json.loads(data)
    except Exception as exc:
        raise RuntimeError("invalid JSON " + name) from exc
    if data != canon(obj):
        raise RuntimeError("noncanonical JSON " + name)
    return obj, data, digest, size, st


def _identity(st):
    return {
        "mode": st.st_mode, "dev": st.st_dev, "inode": st.st_ino,
        "nlink": st.st_nlink, "uid": st.st_uid, "gid": st.st_gid,
        "size": st.st_size, "mtime_ns": st.st_mtime_ns,
        "ctime_ns": st.st_ctime_ns,
    }


def _expect_keys(obj, keys, label):
    if not isinstance(obj, dict) or set(obj) != set(keys):
        raise RuntimeError(label + " schema keys")


def verify_startup():
    flags = sys.flags
    if flags.isolated != 1 or flags.no_site != 1 or flags.dont_write_bytecode != 1:
        raise RuntimeError("required Python flags are -I -S -B")
    if dict(os.environ) != ENVIRONMENT:
        raise RuntimeError("startup environment is not exact and empty-derived")
    for entry in sys.path:
        lowered = entry.lower()
        if not os.path.isabs(entry) or "site-packages" in lowered or "dist-packages" in lowered or entry == os.getcwd():
            raise RuntimeError("unsafe isolated sys.path")
    if any(name in sys.modules for name in ("provider", "manifest", "compare")):
        raise RuntimeError("local module loaded before verification")


def _runtime_actual():
    modules = "argparse base64 bz2 ctypes ctypes.util _ctypes errno fnmatch hashlib importlib.abc importlib.util io json locale lzma os platform re resource secrets select shutil stat subprocess time types sysconfig zlib".split()
    for name in modules:
        importlib.import_module(name)
    ctypes_module = sys.modules["ctypes"]
    ctypes_module.CDLL(None, use_errno=True)

    def record(path):
        path = os.path.realpath(path)
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            _, digest, size, st = _stable_read(fd)
        finally:
            os.close(fd)
        return {"path": path, "sha256": digest, "size": size, "dev": st.st_dev,
                "inode": st.st_ino, "mode": st.st_mode}
    stdlib = os.path.realpath(sysconfig.get_paths()["stdlib"])
    paths = set()
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if not path:
            continue
        path = os.path.realpath(path)
        if path.endswith((".pyc", ".pyo")) and os.path.exists(path[:-1]):
            path = path[:-1]
        try:
            inside_stdlib = os.path.commonpath((path, stdlib)) == stdlib
        except ValueError:
            inside_stdlib = False
        if inside_stdlib and "site-packages" not in path and "dist-packages" not in path:
            paths.add(path)
    mapped_paths = set()
    with open("/proc/self/maps", "rb", buffering=0) as mappings:
        for line in mappings:
            fields = line.rstrip(b"\n").split(None, 5)
            if len(fields) == 6 and fields[5].startswith(b"/") and \
                    not fields[5].endswith(b" (deleted)"):
                mapped_paths.add(os.path.realpath(os.fsdecode(fields[5])))
    stdlib_st = os.stat(stdlib, follow_symlinks=False)
    records = [record(path) for path in sorted(paths)]
    mapped_records = [record(path) for path in sorted(mapped_paths)]
    provider_names = ["ctypes", "ctypes.util", "_ctypes"]
    provider_paths = []
    for name in provider_names:
        path = os.path.realpath(sys.modules[name].__file__)
        if path.endswith((".pyc", ".pyo")) and os.path.exists(path[:-1]):
            path = path[:-1]
        provider_paths.append(path)
    provider_records = [record(path) for path in sorted(set(provider_paths))]
    combined = {item["path"]: item for item in records + mapped_records}
    dynamic_records = [combined[path] for path in sorted(combined)
                       if ".so" in os.path.basename(path)]
    exe = os.path.realpath(sys.executable)
    fd = os.open(exe, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        _, digest, size, st = _stable_read(fd)
    finally:
        os.close(fd)
    current_umask = os.umask(0)
    os.umask(current_umask)
    return {
        "schema": "g0-runtime-lock/v5", "python_executable": exe,
        "python_executable_sha256": digest, "python_executable_size": size,
        "python_executable_dev": st.st_dev, "python_executable_inode": st.st_ino,
        "python_version": sys.version, "implementation": sys.implementation.name,
        "cache_tag": sys.implementation.cache_tag, "abi_flags": getattr(sys, "abiflags", ""),
        "platform": platform.platform(), "machine": platform.machine(),
        "byteorder": sys.byteorder, "umask": current_umask,
        "isolated_sys_path": list(sys.path),
        "required_flags": ["-I", "-S", "-B"], "environment": ENVIRONMENT,
        "stdlib_directory": {"path": stdlib, "dev": stdlib_st.st_dev,
                             "inode": stdlib_st.st_ino, "mode": stdlib_st.st_mode},
        "stdlib_records": records,
        "provider_module_names": provider_names,
        "provider_dependency_records": provider_records,
        "mapped_library_records": mapped_records,
        "dynamic_library_records": dynamic_records,
    }


def _validate_runtime_lock_record(runtime):
    keys = {"schema", "python_executable", "python_executable_sha256",
            "python_executable_size", "python_executable_dev", "python_executable_inode",
            "python_version", "implementation", "cache_tag", "abi_flags", "platform",
            "machine", "byteorder", "umask", "isolated_sys_path", "required_flags",
            "environment", "stdlib_directory", "stdlib_records", "provider_module_names",
            "provider_dependency_records", "mapped_library_records",
            "dynamic_library_records"}
    _expect_keys(runtime, keys, "runtime lock")
    if runtime["schema"] != "g0-runtime-lock/v5" or \
            runtime["provider_module_names"] != ["ctypes", "ctypes.util", "_ctypes"]:
        raise RuntimeError("runtime provider dependency schema")
    record_keys = {"path", "sha256", "size", "dev", "inode", "mode"}
    lists = ("stdlib_records", "provider_dependency_records", "mapped_library_records",
             "dynamic_library_records")
    for field in lists:
        records = runtime[field]
        if not isinstance(records, list) or not records:
            raise RuntimeError("empty runtime dependency closure " + field)
        paths = []
        for record in records:
            _expect_keys(record, record_keys, "runtime dependency")
            if not isinstance(record["path"], str) or not os.path.isabs(record["path"]) or \
                    not re.fullmatch(r"[0-9a-f]{64}", record.get("sha256", "")) or \
                    any(not _valid_int(record[key]) for key in ("size", "dev", "inode", "mode")):
                raise RuntimeError("invalid runtime dependency record")
            paths.append(record["path"])
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise RuntimeError("runtime dependency ordering/uniqueness " + field)
    provider_names = {os.path.basename(record["path"])
                      for record in runtime["provider_dependency_records"]}
    if not any(name == "util.py" for name in provider_names) or \
            not any(name == "__init__.py" for name in provider_names) or \
            not any("_ctypes" in name and ".so" in name for name in provider_names):
        raise RuntimeError("ctypes provider closure incomplete")
    mapped = {record["path"]: record for record in runtime["mapped_library_records"]}
    dynamic = {record["path"]: record for record in runtime["dynamic_library_records"]}
    expected_dynamic = {path: record for path, record in {
        **{item["path"]: item for item in runtime["stdlib_records"]}, **mapped}.items()
                        if ".so" in os.path.basename(path)}
    if dynamic != expected_dynamic:
        raise RuntimeError("dynamic runtime dependency closure mismatch")
    for marker in ("libc.so", "libffi"):
        present = [path for path in mapped if marker in os.path.basename(path)]
        if present and any(path not in dynamic for path in present):
            raise RuntimeError("mapped provider library missing from closure " + marker)


def _expand_argv(record, key, bundle_sha256, accepted_cp0_sha256,
                 terminal_readiness_sha256):
    argv = record.get(key)
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise RuntimeError("argv record type")
    replacements = {"$BUNDLE_SHA256": bundle_sha256,
                    "$ACCEPTED_CP0_SHA256": accepted_cp0_sha256,
                    "$TERMINAL_READINESS_SHA256": terminal_readiness_sha256}
    return [replacements.get(item, item) for item in argv]


def verify_bundle(args):
    """Exact-set verify every immutable input and frozen component."""
    frozen_fd = _open_dir(args.frozen)
    inputs_fd = _open_dir(args.inputs)
    try:
        frozen_st = os.fstat(frozen_fd)
        if stat.S_IMODE(frozen_st.st_mode) != 0o555 or \
                stat.S_IMODE(os.fstat(inputs_fd).st_mode) != 0o555:
            raise RuntimeError("immutable directory mode mismatch")
        if set(os.listdir(frozen_fd)) != FROZEN_NAMES:
            raise RuntimeError("frozen exact-set mismatch")
        if set(os.listdir(inputs_fd)) != INPUT_NAMES:
            raise RuntimeError("input exact-set mismatch")
        lock_f, lock_bytes, lock_digest, _, _ = _load_canon_at(frozen_fd, "bundle-lock.json")
        lock_i, lock_bytes_i, lock_digest_i, _, _ = _load_canon_at(inputs_fd, "bundle-lock.json")
        if stat.S_IMODE(os.stat("bundle-lock.json", dir_fd=frozen_fd, follow_symlinks=False).st_mode) != 0o444 or \
                stat.S_IMODE(os.stat("bundle-lock.json", dir_fd=inputs_fd, follow_symlinks=False).st_mode) != 0o444:
            raise RuntimeError("bundle lock mode mismatch")
        if lock_bytes != lock_bytes_i or lock_digest != lock_digest_i:
            raise RuntimeError("bundle lock copies differ")
        if lock_digest != args.bundle_sha256:
            raise RuntimeError("bundle SHA-256 mismatch")
        _expect_keys(lock_f, {"schema", "sealed_run_id", "bundle_id", "input_files",
                              "input_sha256", "files_json_sha256", "launcher_sha256",
                              "run_protected_sha256", "python_executable",
                              "python_executable_sha256", "frozen_directory"}, "bundle lock")
        if lock_f["schema"] != "g0-bundle-lock/v5" or lock_f["sealed_run_id"] != args.sealed_run_id:
            raise RuntimeError("bundle run binding mismatch")
        _expect_keys(lock_f["frozen_directory"], {"path", "dev", "inode", "mode"},
                     "frozen directory")
        if lock_f["frozen_directory"] != {
                "path": args.frozen, "dev": frozen_st.st_dev, "inode": frozen_st.st_ino,
                "mode": frozen_st.st_mode}:
            raise RuntimeError("frozen root identity drift")
        if lock_f["launcher_sha256"] != globals().get("LAUNCHER_SOURCE_SHA256") or \
                lock_f["run_protected_sha256"] != args.run_protected_sha256:
            raise RuntimeError("launcher/run_protected trust anchor mismatch")
        expected_inputs = sorted(INPUT_NAMES - {"bundle-lock.json"})
        if lock_f["input_files"] != expected_inputs or set(lock_f["input_sha256"]) != set(expected_inputs):
            raise RuntimeError("bundle input closure mismatch")
        loaded = {}
        for name in expected_inputs:
            obj, data, digest, _, component_st = _load_canon_at(inputs_fd, name)
            if digest != lock_f["input_sha256"].get(name):
                raise RuntimeError("input component tamper " + name)
            if stat.S_IMODE(component_st.st_mode) != 0o444:
                raise RuntimeError("input component mode mismatch " + name)
            loaded[name] = (obj, data)
        files, files_bytes, files_digest, _, _ = _load_canon_at(frozen_fd, "files.json")
        if stat.S_IMODE(os.stat("files.json", dir_fd=frozen_fd, follow_symlinks=False).st_mode) != 0o444:
            raise RuntimeError("files index mode mismatch")
        if files_digest != lock_f["files_json_sha256"]:
            raise RuntimeError("frozen files index tamper")
        _expect_keys(files, {"schema", "scripts", "exact_top_level"}, "files index")
        if files["schema"] != "g0-files/v5" or set(files["exact_top_level"]) != FROZEN_NAMES:
            raise RuntimeError("frozen index schema")
        if set(files["scripts"]) != SCRIPT_NAMES:
            raise RuntimeError("script exact-set mismatch")
        for name, meta in files["scripts"].items():
            _expect_keys(meta, {"sha256", "size", "mode"}, "script metadata")
            data, digest, size, st = _read_at(frozen_fd, name)
            del data
            if digest != meta["sha256"] or size != meta["size"] or stat.S_IMODE(st.st_mode) != meta["mode"]:
                raise RuntimeError("frozen script tamper " + name)
        if files["scripts"]["run_protected.py"]["sha256"] != args.run_protected_sha256 or \
                lock_f["run_protected_sha256"] != args.run_protected_sha256:
            raise RuntimeError("run_protected trust-anchor mismatch")
        scripts_lock = loaded["frozen-scripts.json"][0]
        _expect_keys(scripts_lock, {"schema", "sealed_run_id", "bundle_id", "files_json_sha256",
                                    "exact_file_count", "loader_policy"},
                     "frozen scripts lock")
        if scripts_lock["schema"] != "g0-frozen-scripts-lock/v5":
            raise RuntimeError("frozen scripts schema")
        if scripts_lock["files_json_sha256"] != files_digest:
            raise RuntimeError("frozen scripts index mismatch")
        policy = scripts_lock["loader_policy"]
        if scripts_lock["exact_file_count"] != len(SCRIPT_NAMES):
            raise RuntimeError("frozen exact file count mismatch")
        if policy != {"ordinary_imports": False, "pyc_allowed": False,
                      "host_project_fallback_allowed": False}:
            raise RuntimeError("verified loader policy mismatch")
        runtime = loaded["runtime-lock.json"][0]
        runtime_actual = _runtime_actual()
        if runtime != runtime_actual:
            differing = sorted(key for key in set(runtime) | set(runtime_actual)
                               if runtime.get(key) != runtime_actual.get(key))
            expected_paths = {item["path"] for item in runtime.get("stdlib_records", [])}
            actual_paths = {item["path"] for item in runtime_actual.get("stdlib_records", [])}
            raise RuntimeError("runtime lock mismatch: " + ",".join(differing) +
                               " actual-only=" + repr(sorted(actual_paths - expected_paths)) +
                               " expected-only=" + repr(sorted(expected_paths - actual_paths)))
        _validate_runtime_lock_record(runtime)
        if lock_f["python_executable"] != runtime["python_executable"] or \
                lock_f["python_executable_sha256"] != runtime["python_executable_sha256"]:
            raise RuntimeError("bundle Python runtime anchor mismatch")
        config = loaded["manifest-config.json"][0]
        _expect_keys(config, {"schema", "sealed_run_id", "bundle_id", "envelope_id", "task_id"},
                     "manifest config")
        if config != {"schema": "g0-manifest-config/v5", "sealed_run_id": args.sealed_run_id,
                      "bundle_id": lock_f["bundle_id"], "envelope_id": config.get("envelope_id"),
                      "task_id": "G008"}:
            raise RuntimeError("manifest config binding mismatch")
        for name in ("boundary.json", "owner-binding.json", "frozen-scripts.json"):
            obj = loaded[name][0]
            if obj.get("sealed_run_id") != args.sealed_run_id or obj.get("bundle_id") != lock_f["bundle_id"]:
                raise RuntimeError(name + " run/bundle binding mismatch")
        for name, mode in (("argv-pre.json", "pre"), ("argv-post.json", "terminal-post")):
            rec = loaded[name][0]
            _expect_keys(rec, {"schema", "sealed_run_id", "bundle_id", "mode", "orig_argv",
                               "process_argv"}, "argv lock")
            if rec["schema"] != "g0-orig-argv-lock/v5" or rec["sealed_run_id"] != args.sealed_run_id or \
                    rec["bundle_id"] != lock_f["bundle_id"] or rec["mode"] != mode:
                raise RuntimeError("argv lock binding mismatch")
            expected = list(globals().get("LAUNCHER_ORIG_ARGV", sys.argv))
            expected[expected.index("--mode") + 1] = mode
            if _expand_argv(rec, "orig_argv", lock_digest,
                            args.accepted_cp0_seal_sha256,
                            args.terminal_readiness_sha256) != expected:
                raise RuntimeError("current argv mismatch")
            recorded_process = _expand_argv(rec, "process_argv", lock_digest,
                                            args.accepted_cp0_seal_sha256,
                                            args.terminal_readiness_sha256)
            if len(recorded_process) < 6 or recorded_process[:5] != [
                    runtime["python_executable"], "-I", "-S", "-B", "-c"] or \
                    sha(recorded_process[5].encode()) != lock_f["launcher_sha256"] or \
                    recorded_process[6:] != expected[1:]:
                raise RuntimeError("full process argv mismatch")
        current = loaded["argv-pre.json" if args.mode == "pre" else "argv-post.json"][0]
        if _expand_argv(current, "orig_argv", lock_digest,
                        args.accepted_cp0_seal_sha256,
                        args.terminal_readiness_sha256) != list(
                            globals().get("LAUNCHER_ORIG_ARGV", sys.argv)):
            raise RuntimeError("selected argv mismatch")
        with open("/proc/self/cmdline", "rb", buffering=0) as command_line:
            process_actual = [os.fsdecode(item) for item in command_line.read().split(b"\0") if item]
        if _expand_argv(current, "process_argv", lock_digest,
                        args.accepted_cp0_seal_sha256,
                        args.terminal_readiness_sha256) != process_actual:
            raise RuntimeError("selected full process argv mismatch")
        return {
            "bundle": lock_f, "bundle_sha256": lock_digest, "inputs": loaded,
            "files": files, "frozen_fd": frozen_fd, "inputs_fd": inputs_fd,
            "bundle_lock_bytes": lock_bytes,
        }
    except Exception:
        os.close(frozen_fd)
        os.close(inputs_fd)
        raise


def verified_load_project(bundle_state):
    modules = {}
    frozen_fd = bundle_state["frozen_fd"]
    for filename in ("provider.py", "manifest.py", "compare.py"):
        data, digest, size, _ = _read_at(frozen_fd, filename)
        meta = bundle_state["files"]["scripts"][filename]
        if digest != meta["sha256"] or size != meta["size"]:
            raise RuntimeError("verified module mismatch")
        name = filename[:-3]
        module = types.ModuleType(name)
        module.__file__ = "<verified:" + filename + ">"
        module.__package__ = ""
        sys.modules[name] = module
        exec(compile(data, module.__file__, "exec", dont_inherit=True, optimize=0), module.__dict__)
        modules[name] = module
    return modules


def _check_identity(st, expected, label):
    for field in ("dev", "inode", "mode", "size"):
        actual = {"dev": st.st_dev, "inode": st.st_ino, "mode": st.st_mode,
                  "size": st.st_size}[field]
        if expected.get(field) != actual:
            raise RuntimeError(label + " identity drift")


def _check_managed_prefix_identity(st, expected, label):
    if not isinstance(expected, dict):
        raise RuntimeError(label + " identity schema")
    mode = expected.get("mode")
    if type(mode) is not int or mode < 0 or mode > 0o177777:
        raise RuntimeError(label + " mode encoding")
    type_bits = stat.S_IFMT(mode)
    if type_bits not in (0, stat.S_IFDIR):
        raise RuntimeError(label + " mode encoding")
    if not stat.S_ISDIR(st.st_mode):
        raise RuntimeError(label + " type drift")
    for field, actual in (("dev", st.st_dev), ("inode", st.st_ino),
                          ("size", st.st_size)):
        value = expected.get(field)
        if type(value) is not int or value < 0 or value != actual:
            raise RuntimeError(label + " identity drift")
    if type_bits == 0:
        if stat.S_IMODE(st.st_mode) != mode:
            raise RuntimeError(label + " mode drift")
    elif st.st_mode != mode:
        raise RuntimeError(label + " mode drift")


class AccessLog:
    def __init__(self, args, bundle_sha256, config, event_fd=None):
        self.args, self.bundle_sha256, self.config = args, bundle_sha256, config
        self.events = []
        self.event_fd = event_fd
        self.provider_ready = False
        self.provider_identity = None

    def activate(self, landlock, seccomp):
        if self.provider_ready:
            raise RuntimeError("provider readiness transition repeated")
        _expect_keys(landlock, {"provider", "result", "abi", "handled_access_fs",
                                "allow_rule_count", "descriptor_bound"}, "landlock identity")
        _expect_keys(seccomp, {"provider", "result", "audit_arch", "default",
                               "denied_syscalls", "errno"}, "seccomp identity")
        if landlock.get("provider") != "landlock" or landlock.get("result") != "enforced" or \
                landlock.get("abi", 0) < 3 or landlock.get("descriptor_bound") is not True or \
                seccomp.get("provider") != "seccomp-bpf" or \
                seccomp.get("result") != "enforced":
            raise RuntimeError("provider readiness transition invalid")
        self.provider_identity = json.loads(canon({"landlock": landlock, "seccomp": seccomp}))
        self.provider_ready = True

    def emit(self, root, raw_path, operation, stage="after", st=None, result="ok", err=0,
             byte_count=0, content_sha256=None):
        if not self.provider_ready:
            raise RuntimeError("protected event before provider readiness")
        if not isinstance(raw_path, bytes):
            raw_path = os.fsencode(raw_path)
        plain = raw_path.decode("utf-8", "surrogateescape")
        if plain.encode("utf-8", "surrogateescape") != raw_path:
            raise RuntimeError("protected path does not round-trip")
        seq = len(self.events) + 1
        record_id = sha(root.encode() + b"\0" + raw_path)
        self.events.append({
            "schema": "g0-protected-access-event/v5", "sequence": seq,
            "phase": self.args.mode, "sealed_run_id": self.args.sealed_run_id,
            "task_id": self.config["task_id"], "role": "executor",
            "bundle_sha256": self.bundle_sha256,
            "envelope_id": self.config["envelope_id"], "root_id": root,
            "record_id": record_id, "path": plain,
            "path_b64": base64.b64encode(raw_path).decode("ascii"),
            "operation": operation, "stage": stage, "result": result, "errno": err,
            "bytes_read": byte_count, "record_identity": _identity(st) if st else None,
            "content_sha256": content_sha256,
            "provider_ready": self.provider_ready,
        })
        if self.event_fd is not None:
            _write_all(self.event_fd, canon(self.events[-1]))

    def bytes(self):
        return b"".join(canon(event) for event in self.events)

    def validate(self):
        if [event["sequence"] for event in self.events] != list(range(1, len(self.events) + 1)):
            raise RuntimeError("access event loss, duplication, or reorder")
        if any(not event["provider_ready"] for event in self.events):
            raise RuntimeError("access before provider readiness")
        return self.bytes()


def _reconcile_terminal_provider(pre_provider_bytes, terminal_identity, args, state):
    """Compare the in-memory terminal provider result to immutable CP0 evidence."""
    try:
        evidence = json.loads(pre_provider_bytes)
    except Exception as exc:
        raise RuntimeError("invalid CP0 provider evidence") from exc
    if pre_provider_bytes != canon(evidence):
        raise RuntimeError("noncanonical CP0 provider evidence")
    required = {"schema", "sealed_run_id", "bundle_sha256", "role", "kernel_release",
                "architecture", "python_executable", "python_executable_sha256",
                "provider_module_sha256", "landlock", "seccomp", "destructive_controls",
                "write_allow", "claims", "owner_validation"}
    _expect_keys(evidence, required, "CP0 provider evidence")
    runtime = state["inputs"]["runtime-lock.json"][0]
    expected_bindings = {
        "schema": "g0-provider-evidence/v5", "sealed_run_id": args.sealed_run_id,
        "bundle_sha256": state["bundle_sha256"], "role": "executor",
        "kernel_release": platform.release(), "architecture": platform.machine(),
        "python_executable": runtime["python_executable"],
        "python_executable_sha256": runtime["python_executable_sha256"],
        "provider_module_sha256": state["files"]["scripts"]["provider.py"]["sha256"],
    }
    if any(evidence.get(key) != value for key, value in expected_bindings.items()):
        raise RuntimeError("terminal provider binding drift")
    expected_identity = {"landlock": evidence["landlock"], "seccomp": evidence["seccomp"]}
    if terminal_identity != expected_identity:
        raise RuntimeError("terminal provider identity drift")
    if evidence.get("claims") != {"global_read_denial": False, "syscall_tracing": False}:
        raise RuntimeError("terminal provider claim drift")
    return expected_identity


def _read_logged(fd, root, path, log, expected, hook=None):
    actual = os.fstat(fd)
    _check_identity(actual, expected, root)
    def on_chunk(count, st):
        if hook:
            hook("read", fd)
    log.emit(root, os.fsencode(path), "read_file", "before", actual, "pending")
    data, digest, count, stable = _stable_read(fd, actual, on_chunk)
    log.emit(root, os.fsencode(path), "read_file", "after", stable, byte_count=count,
             content_sha256=digest)
    return data, digest, count, stable


def _logged_lstat(dfd, name, root, path, log, operation="lstat"):
    raw = os.fsencode(path)
    log.emit(root, raw, operation, "before", None, "pending")
    try:
        st = os.stat(name, dir_fd=dfd, follow_symlinks=False)
    except OSError as exc:
        log.emit(root, raw, operation, "after", None, "error", exc.errno, 0)
        raise
    log.emit(root, raw, operation, "after", st)
    return st


def _logged_open(manifest, dfd, name, root, path, log):
    raw = os.fsencode(path)
    log.emit(root, raw, "open_regular", "before", None, "pending")
    try:
        fd = manifest.open_regular_at(dfd, name)
    except OSError as exc:
        log.emit(root, raw, "open_regular", "after", None, "error", exc.errno, 0)
        raise
    log.emit(root, raw, "open_regular", "after", os.fstat(fd))
    return fd


def _validate_owner(owner, run_id, manifest, log, paper_record):
    expected_keys = {"schema", "sealed_run_id", "bundle_id", "project_uuid", "owner_file",
                     "target", "managed_prefixes", "readme", "foreign"}
    if set(owner) != expected_keys or owner.get("schema") != "g0-owner-binding-lock/v5" or \
            owner.get("sealed_run_id") != run_id:
        raise RuntimeError("mandatory owner binding missing")
    if not isinstance(owner["managed_prefixes"], dict) or not owner["managed_prefixes"] or \
            not isinstance(owner["foreign"], list) or not owner["foreign"]:
        raise RuntimeError("owner managed/foreign closure must be nonempty")
    if len({item.get("path") for item in owner["foreign"]}) != len(owner["foreign"]):
        raise RuntimeError("duplicate owner foreign binding")
    target = owner["target"]
    target_fd = _open_dir(target["path"])
    try:
        _check_identity(os.fstat(target_fd), target, "owner target")
        spec = owner["owner_file"]
        if spec["parent_path"] != target["path"]:
            raise RuntimeError("owner parent mismatch")
        lst = _logged_lstat(target_fd, spec["name"], "owner", spec["name"], log)
        _check_identity(lst, spec, "owner file")
        fd = _logged_open(manifest, target_fd, spec["name"], "owner", spec["name"], log)
        try:
            data, digest, count, st = _read_logged(fd, "owner", spec["name"], log, spec)
        finally:
            os.close(fd)
        if digest != spec["sha256"] or count != spec["size"]:
            raise RuntimeError("owner hash drift")
        record = json.loads(data)
        if data != canon(record) or record.get("project_uuid") != owner["project_uuid"]:
            raise RuntimeError("owner UUID drift")
        record_managed = record.get("managed_prefixes")
        if not isinstance(record_managed, dict) or \
                canon(record_managed) != canon(owner["managed_prefixes"]):
            raise RuntimeError("managed prefix drift")
        readme = owner["readme"]
        rlst = _logged_lstat(target_fd, readme["name"], "owner", readme["name"], log)
        _check_identity(rlst, readme, "README")
        rfd = _logged_open(manifest, target_fd, readme["name"], "owner", readme["name"], log)
        try:
            _, rdigest, rcount, _ = _read_logged(rfd, "owner", readme["name"], log, readme)
        finally:
            os.close(rfd)
        if rdigest != readme["sha256"] or rcount != readme["size"]:
            raise RuntimeError("README hash drift")
        for prefix, expected in sorted(owner["managed_prefixes"].items(),
                                       key=lambda item: os.fsencode(item[0])):
            safe_name(prefix)
            mst = _logged_lstat(target_fd, prefix, "owner", prefix, log, "lstat_managed")
            _check_managed_prefix_identity(mst, expected, "managed prefix")
        foreign_out = []
        for foreign in sorted(owner["foreign"], key=lambda item: os.fsencode(item["path"])):
            parent_fd = _open_dir(foreign["parent_path"])
            try:
                pst = os.fstat(parent_fd)
                if pst.st_dev != foreign["parent_dev"] or pst.st_ino != foreign["parent_inode"]:
                    raise RuntimeError("foreign parent drift")
                fst = _logged_lstat(parent_fd, foreign["name"], "owner", foreign["path"], log,
                                    "lstat_foreign")
                _check_identity(fst, foreign, "foreign")
                if stat.S_ISREG(fst.st_mode) and "sha256" in foreign:
                    shared = ("path", "parent_path", "parent_dev", "parent_inode",
                              "dev", "inode", "mode", "size", "sha256", "pages")
                    if any(foreign.get(field) != paper_record.get(field) for field in shared):
                        raise RuntimeError("foreign paper binding drift")
                foreign_out.append({"path": foreign["path"], "dev": fst.st_dev, "inode": fst.st_ino})
            finally:
                os.close(parent_fd)
        target_check = _open_dir(target["path"])
        try:
            _check_identity(os.fstat(target_check), target, "owner target post-validation")
        finally:
            os.close(target_check)
        return {
            "schema": "g0-owner-binding-evidence/v5", "sealed_run_id": run_id,
            "project_uuid": owner["project_uuid"], "owner_sha256": digest,
            "owner_dev": st.st_dev, "owner_inode": st.st_ino,
            "historical_sealed_run_id": record.get("sealed_run_id"),
            "target": target, "managed_prefixes": owner["managed_prefixes"],
            "readme_sha256": rdigest, "foreign": foreign_out,
        }
    finally:
        os.close(target_fd)


def _synthetic_controls(boundary, scratch_fd, log, provider):
    fixture = boundary["disposable_denial_fixture"]
    if fixture.get("disposable") is not True:
        raise RuntimeError("denial fixture is not disposable")
    fixture_fd = _open_dir(fixture["path"])
    results = {}
    try:
        marker, digest, marker_size, marker_st = _read_at(fixture_fd, ".g0-synthetic-fixture.json")
        log.emit("denial-fixture", b".g0-synthetic-fixture.json", "read_declaration",
                 "before", marker_st, "pending")
        log.emit("denial-fixture", b".g0-synthetic-fixture.json", "read_declaration",
                 "after", marker_st, byte_count=marker_size, content_sha256=digest)
        if digest != fixture["declaration_sha256"] or json.loads(marker).get("disposable") is not True:
            raise RuntimeError("denial fixture declaration mismatch")
        def denied(name, operation):
            try:
                operation()
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EPERM):
                    raise RuntimeError("unexpected denial errno for " + name) from exc
                results[name] = exc.errno
                log.emit("denial-fixture", os.fsencode(name), "write_denial_control", "before",
                         None, "pending")
                log.emit("denial-fixture", os.fsencode(name), "write_denial_control", "after",
                         None, "denied", exc.errno, 0)
                return
            raise RuntimeError("write denial failed: " + name)
        denied("create", lambda: os.close(os.open("new", os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                                                   0o600, dir_fd=fixture_fd)))
        denied("open_truncate", lambda: os.close(os.open("victim", os.O_WRONLY | os.O_TRUNC,
                                                          dir_fd=fixture_fd)))
        fixture_victim = "/proc/self/fd/%d/victim" % fixture_fd
        denied("truncate", lambda: os.truncate(fixture_victim, 0))
        denied("unlink", lambda: os.unlink("victim", dir_fd=fixture_fd))
        denied("rename", lambda: os.rename("spare", "renamed", src_dir_fd=fixture_fd,
                                            dst_dir_fd=fixture_fd))
        denied("renameat2_noreplace", lambda: provider.renameat2_noreplace(
            fixture_fd, "spare", fixture_fd, "renamed2"))
        denied("mkdir", lambda: os.mkdir("newdir", dir_fd=fixture_fd))
        denied("rmdir", lambda: os.rmdir("emptydir", dir_fd=fixture_fd))
        denied("chmod", lambda: os.chmod("victim", 0o600, dir_fd=fixture_fd,
                                         follow_symlinks=False))
        denied("chown", lambda: os.chown("victim", os.getuid(), os.getgid(), dir_fd=fixture_fd,
                                         follow_symlinks=False))
        denied("utime", lambda: os.utime("victim", ns=(1, 1), dir_fd=fixture_fd,
                                         follow_symlinks=False))
        if hasattr(os, "setxattr"):
            victim_path = fixture["path"] + "/victim"
            denied("setxattr", lambda: os.setxattr(victim_path, "user.g0", b"x"))
            denied("removexattr", lambda: os.removexattr(victim_path, "user.seed"))
        denied("symlink", lambda: os.symlink("victim", "link", dir_fd=fixture_fd))
        denied("hardlink", lambda: os.link("victim", "hard", src_dir_fd=fixture_fd,
                                            dst_dir_fd=fixture_fd, follow_symlinks=False))
    finally:
        os.close(fixture_fd)
    # Positive controls are confined to scratch and occur before protected enumeration.
    pfd = os.open("positive", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
                  0o600, dir_fd=scratch_fd)
    _write_all(pfd, b"ok")
    os.fsync(pfd)
    os.close(pfd)
    os.rename("positive", "positive-renamed", src_dir_fd=scratch_fd, dst_dir_fd=scratch_fd)
    os.unlink("positive-renamed", dir_fd=scratch_fd)
    results["scratch_create_rename_unlink"] = "ok"
    return results


class _PdfName(bytes):
    pass


class _PdfRef(tuple):
    def __new__(cls, number, generation):
        return tuple.__new__(cls, (number, generation))

    @property
    def number(self):
        return self[0]

    @property
    def generation(self):
        return self[1]


_PDF_WHITE = b"\x00\x09\x0a\x0c\x0d\x20"
_PDF_DELIMITERS = b"()<>[]{}/%"
_PDF_MAX_FILE = 512 * 1024 * 1024
_PDF_MAX_OBJECTS = 1_000_000
_PDF_MAX_DEPTH = 256
_PDF_MAX_STREAM = 512 * 1024 * 1024
_PDF_FLATE_CHUNK = 64 * 1024
_PDF_NULL = object()


class _PdfLexer:
    """Small bounded ISO-32000 lexical scanner for structural objects."""

    def __init__(self, data, start=0, end=None):
        self.data = data
        self.pos = start
        self.end = len(data) if end is None else end
        self.buffer = []
        self.token_count = 0

    def skip(self):
        while self.pos < self.end:
            byte = self.data[self.pos]
            if byte in _PDF_WHITE:
                self.pos += 1
                continue
            if byte == 0x25:  # comment through CR or LF
                self.pos += 1
                while self.pos < self.end and self.data[self.pos] not in (0x0a, 0x0d):
                    self.pos += 1
                continue
            break
        return self.pos

    def _literal_string(self):
        self.pos += 1
        depth = 1
        out = bytearray()
        while self.pos < self.end and depth:
            byte = self.data[self.pos]
            self.pos += 1
            if byte == 0x5c:
                if self.pos >= self.end:
                    raise RuntimeError("unterminated PDF string escape")
                escaped = self.data[self.pos]
                self.pos += 1
                mapping = {ord("n"): 0x0a, ord("r"): 0x0d, ord("t"): 0x09,
                           ord("b"): 0x08, ord("f"): 0x0c,
                           ord("("): ord("("), ord(")"): ord(")"), ord("\\"): ord("\\")}
                if escaped in mapping:
                    out.append(mapping[escaped])
                elif escaped in (0x0a, 0x0d):
                    if escaped == 0x0d and self.pos < self.end and self.data[self.pos] == 0x0a:
                        self.pos += 1
                elif 0x30 <= escaped <= 0x37:
                    digits = bytearray([escaped])
                    for _ in range(2):
                        if self.pos < self.end and 0x30 <= self.data[self.pos] <= 0x37:
                            digits.append(self.data[self.pos])
                            self.pos += 1
                        else:
                            break
                    out.append(int(digits, 8) & 0xff)
                else:
                    out.append(escaped)
            elif byte == 0x28:
                depth += 1
                if depth > _PDF_MAX_DEPTH:
                    raise RuntimeError("PDF string nesting limit")
                out.append(byte)
            elif byte == 0x29:
                depth -= 1
                if depth:
                    out.append(byte)
            else:
                out.append(byte)
        if depth:
            raise RuntimeError("unterminated PDF literal string")
        return bytes(out)

    def _hex_string(self):
        self.pos += 1
        digits = bytearray()
        while self.pos < self.end:
            byte = self.data[self.pos]
            self.pos += 1
            if byte == 0x3e:
                break
            if byte in _PDF_WHITE:
                continue
            if byte not in b"0123456789abcdefABCDEF":
                raise RuntimeError("invalid PDF hex string")
            digits.append(byte)
        else:
            raise RuntimeError("unterminated PDF hex string")
        if len(digits) % 2:
            digits.append(ord("0"))
        return bytes.fromhex(digits.decode("ascii"))

    def _next(self):
        self.skip()
        if self.pos >= self.end:
            return None
        self.token_count += 1
        if self.token_count > _PDF_MAX_OBJECTS * 64:
            raise RuntimeError("PDF token limit")
        start = self.pos
        byte = self.data[self.pos]
        if self.data.startswith(b"<<", self.pos):
            self.pos += 2
            return b"<<"
        if self.data.startswith(b">>", self.pos):
            self.pos += 2
            return b">>"
        if byte in b"[]":
            self.pos += 1
            return bytes([byte])
        if byte == 0x28:
            return self._literal_string()
        if byte == 0x3c:
            return self._hex_string()
        if byte == 0x2f:
            self.pos += 1
            raw = bytearray()
            while self.pos < self.end:
                current = self.data[self.pos]
                if current in _PDF_WHITE or current in _PDF_DELIMITERS:
                    break
                self.pos += 1
                if current == 0x23:
                    if self.pos + 2 > self.end:
                        raise RuntimeError("invalid PDF name escape")
                    pair = self.data[self.pos:self.pos + 2]
                    if any(value not in b"0123456789abcdefABCDEF" for value in pair):
                        raise RuntimeError("invalid PDF name escape")
                    raw.append(int(pair, 16))
                    self.pos += 2
                else:
                    raw.append(current)
            return _PdfName(raw)
        while self.pos < self.end and self.data[self.pos] not in _PDF_WHITE + _PDF_DELIMITERS:
            self.pos += 1
        if self.pos == start:
            raise RuntimeError("unexpected PDF delimiter")
        raw = self.data[start:self.pos]
        if re.fullmatch(rb"[+-]?\d+", raw):
            return int(raw)
        if re.fullmatch(rb"[+-]?(?:\d+\.\d*|\.\d+)", raw):
            return float(raw)
        if raw == b"true":
            return True
        if raw == b"false":
            return False
        if raw == b"null":
            return _PDF_NULL
        return raw

    def peek(self, distance=0):
        while len(self.buffer) <= distance:
            self.buffer.append(self._next())
        return self.buffer[distance]

    def take(self):
        if self.buffer:
            return self.buffer.pop(0)
        return self._next()

    def value(self, depth=0):
        if depth > _PDF_MAX_DEPTH:
            raise RuntimeError("PDF object nesting limit")
        token = self.take()
        if token == b"<<":
            result = {}
            while self.peek() != b">>":
                key = self.take()
                if not isinstance(key, _PdfName) or key in result:
                    raise RuntimeError("invalid or duplicate PDF dictionary key")
                result[key] = self.value(depth + 1)
            self.take()
            return result
        if token == b"[":
            result = []
            while self.peek() != b"]":
                if self.peek() is None:
                    raise RuntimeError("unterminated PDF array")
                result.append(self.value(depth + 1))
                if len(result) > _PDF_MAX_OBJECTS:
                    raise RuntimeError("PDF array limit")
            self.take()
            return result
        if isinstance(token, int) and not isinstance(token, bool) and \
                isinstance(self.peek(), int) and not isinstance(self.peek(), bool) and \
                self.peek(1) == b"R":
            generation = self.take()
            self.take()
            if token < 0 or generation < 0:
                raise RuntimeError("negative PDF indirect reference")
            return _PdfRef(token, generation)
        if token in (b">>", b"]", None):
            raise RuntimeError("unexpected PDF object terminator")
        if token is _PDF_NULL:
            return None
        return token


def _pdf_get(dictionary, name, default=None):
    if not isinstance(dictionary, dict):
        raise RuntimeError("expected PDF dictionary")
    return dictionary.get(_PdfName(name.encode("ascii")), default)


def _pdf_int(value, label, minimum=0):
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise RuntimeError("invalid PDF " + label)
    return value


def _parse_pdf_indirect(data, offset, length_resolver=None):
    if not isinstance(offset, int) or offset <= 0 or offset >= len(data):
        raise RuntimeError("invalid PDF object offset")
    lexer = _PdfLexer(data, offset)
    if lexer.skip() != offset:
        raise RuntimeError("PDF xref offset points to whitespace")
    number, generation, marker = lexer.take(), lexer.take(), lexer.take()
    if not isinstance(number, int) or isinstance(number, bool) or number <= 0 or \
            not isinstance(generation, int) or isinstance(generation, bool) or generation < 0 or \
            marker != b"obj":
        raise RuntimeError("PDF xref offset/object header mismatch")
    value = lexer.value()
    stream = None
    if isinstance(value, dict) and lexer.peek() == b"stream":
        lexer.take()
        position = lexer.pos
        if data.startswith(b"\r\n", position):
            position += 2
        elif position < len(data) and data[position] in (0x0a, 0x0d):
            position += 1
        else:
            raise RuntimeError("PDF stream keyword lacks EOL")
        length = _pdf_get(value, "Length")
        if isinstance(length, _PdfRef):
            if length_resolver is None:
                raise RuntimeError("unresolvable PDF stream length")
            length = length_resolver(length)
        length = _pdf_int(length, "stream length")
        if length > _PDF_MAX_STREAM or position + length > len(data):
            raise RuntimeError("PDF stream length out of bounds")
        stream = data[position:position + length]
        lexer.pos = position + length
        lexer.buffer.clear()
        if lexer.take() != b"endstream":
            raise RuntimeError("PDF stream length/endstream mismatch")
    if lexer.take() != b"endobj":
        raise RuntimeError("unterminated PDF indirect object")
    return number, generation, value, stream


def _png_predictor(data, parameters):
    predictor = _pdf_get(parameters, "Predictor", 1)
    predictor = _pdf_int(predictor, "predictor", 1)
    if predictor == 1:
        return data
    colors = _pdf_int(_pdf_get(parameters, "Colors", 1), "predictor Colors", 1)
    bits = _pdf_int(_pdf_get(parameters, "BitsPerComponent", 8),
                    "predictor BitsPerComponent", 1)
    columns = _pdf_int(_pdf_get(parameters, "Columns", 1), "predictor Columns", 1)
    if bits != 8:
        raise RuntimeError("unsupported PDF predictor bit width")
    row_size = colors * columns
    if row_size <= 0 or row_size > _PDF_MAX_STREAM:
        raise RuntimeError("PDF predictor row out of bounds")
    if predictor == 2:
        if len(data) % row_size:
            raise RuntimeError("invalid TIFF predictor length")
        out = bytearray(data)
        for row in range(0, len(out), row_size):
            for index in range(colors, row_size):
                out[row + index] = (out[row + index] + out[row + index - colors]) & 0xff
        return bytes(out)
    if predictor < 10 or predictor > 15:
        raise RuntimeError("unsupported PDF predictor")
    tagged = len(data) % (row_size + 1) == 0
    if not tagged and len(data) % row_size:
        raise RuntimeError("invalid PNG predictor length")
    previous = bytearray(row_size)
    output = bytearray()
    position = 0
    while position < len(data):
        if tagged:
            filter_type = data[position]
            position += 1
        else:
            filter_type = predictor - 10
        row = bytearray(data[position:position + row_size])
        position += row_size
        if filter_type > 4:
            raise RuntimeError("invalid PNG predictor filter")
        for index in range(row_size):
            left = row[index - colors] if index >= colors else 0
            up = previous[index]
            upper_left = previous[index - colors] if index >= colors else 0
            if filter_type == 1:
                prediction = left
            elif filter_type == 2:
                prediction = up
            elif filter_type == 3:
                prediction = (left + up) // 2
            elif filter_type == 4:
                p = left + up - upper_left
                distances = (abs(p - left), abs(p - up), abs(p - upper_left))
                prediction = (left, up, upper_left)[distances.index(min(distances))]
            else:
                prediction = 0
            row[index] = (row[index] + prediction) & 0xff
        output.extend(row)
        previous = row
        if len(output) > _PDF_MAX_STREAM:
            raise RuntimeError("decoded PDF stream limit")
    return bytes(output)


def _flate_decode_bounded(data, limit=None):
    """Decode exactly one zlib stream without allocating beyond the policy cap."""
    import zlib
    if not isinstance(data, bytes):
        raise RuntimeError("invalid PDF Flate input")
    if limit is None:
        limit = _PDF_MAX_STREAM
    if not _valid_int(limit, 1):
        raise RuntimeError("invalid PDF Flate output limit")
    if not _valid_int(_PDF_FLATE_CHUNK, 1):
        raise RuntimeError("invalid PDF Flate chunk limit")
    decoder = zlib.decompressobj()
    output = []
    total = 0
    position = 0
    try:
        while position < len(data):
            if decoder.eof:
                raise RuntimeError("trailing or concatenated PDF Flate stream")
            pending = data[position:position + _PDF_FLATE_CHUNK]
            position += len(pending)
            while pending:
                remaining = limit - total
                before = pending
                # Python's zlib API needs one bounded sentinel byte to
                # distinguish trailer-only input after an exact-limit decode
                # from a real decoded byte beyond the limit.  A call can
                # allocate at most one byte beyond `remaining` and never more
                # than `_PDF_FLATE_CHUNK`; that sentinel is rejected before it
                # is appended, so retained output never exceeds `limit`.
                decoded = decoder.decompress(
                    pending, min(_PDF_FLATE_CHUNK, remaining + 1))
                if len(decoded) > remaining:
                    raise RuntimeError("decoded PDF stream limit")
                if decoded:
                    output.append(decoded)
                    total += len(decoded)
                if decoder.unused_data:
                    raise RuntimeError("trailing or concatenated PDF Flate stream")
                pending = decoder.unconsumed_tail
                if pending == before and not decoded:
                    raise RuntimeError("invalid PDF Flate progress state")
                if decoder.eof:
                    if pending or position != len(data):
                        raise RuntimeError("trailing or concatenated PDF Flate stream")
                    break
        if not decoder.eof:
            raise RuntimeError("incomplete PDF Flate stream")
        if decoder.unused_data or decoder.unconsumed_tail:
            raise RuntimeError("invalid PDF Flate end state")
        # Do not call decompressobj.flush(): its return allocation is not capped.
        # eof plus empty unused/unconsumed state is the exact bounded end-state
        # oracle after every output byte has already crossed max_length.
    except zlib.error as exc:
        raise RuntimeError("invalid PDF Flate stream") from exc
    return b"".join(output)


def _decode_pdf_stream(dictionary, stream):
    filters = _pdf_get(dictionary, "Filter")
    parameters = _pdf_get(dictionary, "DecodeParms")
    if filters is None:
        if parameters is not None:
            raise RuntimeError("PDF DecodeParms without Filter")
        return stream
    filters = filters if isinstance(filters, list) else [filters]
    if not filters or not all(isinstance(item, _PdfName) for item in filters):
        raise RuntimeError("invalid PDF Filter")
    if parameters is None:
        parameters = [None] * len(filters)
    elif isinstance(parameters, dict):
        if len(filters) != 1:
            raise RuntimeError("ambiguous PDF DecodeParms")
        parameters = [parameters]
    elif isinstance(parameters, list):
        if len(parameters) != len(filters):
            raise RuntimeError("PDF Filter/DecodeParms length mismatch")
    else:
        raise RuntimeError("invalid PDF DecodeParms")
    data = stream
    for filter_name, params in zip(filters, parameters):
        if params is not None and not isinstance(params, dict):
            raise RuntimeError("invalid PDF filter parameters")
        name = bytes(filter_name)
        if name in (b"FlateDecode", b"Fl"):
            data = _flate_decode_bounded(data)
            data = _png_predictor(data, params or {})
        elif name in (b"ASCIIHexDecode", b"AHx"):
            clean = bytes(value for value in data if value not in _PDF_WHITE)
            if not clean.endswith(b">"):
                raise RuntimeError("unterminated PDF ASCIIHex stream")
            clean = clean[:-1]
            if any(value not in b"0123456789abcdefABCDEF" for value in clean):
                raise RuntimeError("invalid PDF ASCIIHex stream")
            if len(clean) % 2:
                clean += b"0"
            data = bytes.fromhex(clean.decode("ascii"))
        elif name in (b"ASCII85Decode", b"A85"):
            clean = bytes(value for value in data if value not in _PDF_WHITE)
            if not clean.endswith(b"~>"):
                raise RuntimeError("unterminated PDF ASCII85 stream")
            try:
                data = base64.a85decode(clean[:-2], adobe=False)
            except ValueError as exc:
                raise RuntimeError("invalid PDF ASCII85 stream") from exc
        elif name in (b"RunLengthDecode", b"RL"):
            out = bytearray()
            position = 0
            terminated = False
            while position < len(data):
                length = data[position]
                position += 1
                if length == 128:
                    terminated = True
                    break
                if length <= 127:
                    count = length + 1
                    if position + count > len(data):
                        raise RuntimeError("invalid PDF RunLength stream")
                    out.extend(data[position:position + count])
                    position += count
                else:
                    if position >= len(data):
                        raise RuntimeError("invalid PDF RunLength stream")
                    out.extend(data[position:position + 1] * (257 - length))
                    position += 1
                if len(out) > _PDF_MAX_STREAM:
                    raise RuntimeError("decoded PDF stream limit")
            if not terminated:
                raise RuntimeError("unterminated PDF RunLength stream")
            data = bytes(out)
        else:
            raise RuntimeError("unsupported PDF stream filter " + name.decode("latin1"))
        if len(data) > _PDF_MAX_STREAM:
            raise RuntimeError("decoded PDF stream limit")
    return data


class _PdfResolver:
    def __init__(self, data):
        self.data = data
        self.xref = {}
        self.cache = {}
        self.resolving = set()
        self.object_stream_cache = {}
        self.root = None
        self.size = None
        self._load_xref_chain(self._startxref())
        if not isinstance(self.root, _PdfRef):
            raise RuntimeError("PDF trailer lacks Catalog Root")
        self._validate_in_use_offsets()

    def _startxref(self):
        if len(self.data) > _PDF_MAX_FILE or not self.data.startswith(b"%PDF-"):
            raise RuntimeError("invalid PDF framing")
        stripped = self.data.rstrip(_PDF_WHITE)
        if not stripped.endswith(b"%%EOF"):
            raise RuntimeError("invalid PDF EOF")
        eof = len(stripped) - len(b"%%EOF")
        marker = self.data.rfind(b"startxref", 0, eof)
        if marker < 0:
            raise RuntimeError("invalid PDF startxref")
        lexer = _PdfLexer(self.data, marker, eof)
        if lexer.take() != b"startxref":
            raise RuntimeError("invalid PDF startxref marker")
        offset = _pdf_int(lexer.take(), "startxref", 1)
        if lexer.take() is not None or offset >= len(self.data):
            raise RuntimeError("invalid PDF startxref value")
        return offset

    def _classic_xref(self, offset):
        lexer = _PdfLexer(self.data, offset)
        if lexer.skip() != offset or lexer.take() != b"xref":
            raise RuntimeError("invalid classic PDF xref anchor")
        entries = {}
        covered = set()
        while lexer.peek() != b"trailer":
            first = _pdf_int(lexer.take(), "xref subsection start")
            count = _pdf_int(lexer.take(), "xref subsection count")
            if count > _PDF_MAX_OBJECTS or first + count > _PDF_MAX_OBJECTS:
                raise RuntimeError("PDF xref subsection limit")
            for number in range(first, first + count):
                if number in covered:
                    raise RuntimeError("duplicate or overlapping PDF xref entry")
                covered.add(number)
                location = _pdf_int(lexer.take(), "xref entry offset")
                generation = _pdf_int(lexer.take(), "xref generation")
                status = lexer.take()
                if status == b"n":
                    entries[number] = (1, location, generation)
                elif status == b"f":
                    entries[number] = (0, location, generation)
                else:
                    raise RuntimeError("invalid PDF xref entry")
        lexer.take()
        trailer = lexer.value()
        if not isinstance(trailer, dict):
            raise RuntimeError("invalid PDF trailer")
        size = _pdf_int(_pdf_get(trailer, "Size"), "trailer Size", 1)
        if size > _PDF_MAX_OBJECTS or any(number >= size for number in entries):
            raise RuntimeError("classic PDF xref /Size mismatch")
        return entries, trailer

    def _xref_stream(self, offset):
        number, generation, dictionary, stream = _parse_pdf_indirect(self.data, offset)
        if not isinstance(dictionary, dict) or stream is None or \
                _pdf_get(dictionary, "Type") != _PdfName(b"XRef"):
            raise RuntimeError("invalid PDF xref stream")
        widths = _pdf_get(dictionary, "W")
        if not isinstance(widths, list) or len(widths) != 3:
            raise RuntimeError("invalid PDF xref stream W")
        widths = [_pdf_int(value, "xref stream width") for value in widths]
        if sum(widths) <= 0 or any(value > 8 for value in widths):
            raise RuntimeError("invalid PDF xref stream widths")
        size = _pdf_int(_pdf_get(dictionary, "Size"), "xref stream Size", 1)
        if size > _PDF_MAX_OBJECTS:
            raise RuntimeError("PDF xref stream Size limit")
        index = _pdf_get(dictionary, "Index", [0, size])
        if not isinstance(index, list) or len(index) % 2 or not index:
            raise RuntimeError("invalid PDF xref stream Index")
        ranges = []
        range_keys = set()
        covered = set()
        total = 0
        for position in range(0, len(index), 2):
            first = _pdf_int(index[position], "xref Index start")
            count = _pdf_int(index[position + 1], "xref Index count")
            if first > size or first + count > size:
                raise RuntimeError("PDF xref stream Index outside /Size")
            if (first, count) in range_keys:
                raise RuntimeError("duplicate PDF xref stream Index range")
            range_keys.add((first, count))
            numbers = set(range(first, first + count))
            if covered & numbers:
                raise RuntimeError("overlapping PDF xref stream Index ranges")
            covered.update(numbers)
            ranges.append((first, count))
            total += count
        if total > size:
            raise RuntimeError("PDF xref stream entry count exceeds /Size")
        decoded = _decode_pdf_stream(dictionary, stream)
        entry_width = sum(widths)
        if len(decoded) != total * entry_width:
            raise RuntimeError("PDF xref stream length mismatch")
        entries = {}
        cursor = 0
        for first, count in ranges:
            for object_number in range(first, first + count):
                if object_number in entries:
                    raise RuntimeError("duplicate PDF xref stream entry")
                fields = []
                for width in widths:
                    value = int.from_bytes(decoded[cursor:cursor + width], "big") if width else 0
                    cursor += width
                    fields.append(value)
                entry_type = fields[0] if widths[0] else 1
                if entry_type == 0:
                    entries[object_number] = (0, fields[1], fields[2])
                elif entry_type == 1:
                    entries[object_number] = (1, fields[1], fields[2])
                elif entry_type == 2:
                    entries[object_number] = (2, fields[1], fields[2])
                else:
                    raise RuntimeError("unsupported PDF xref entry type")
        if number >= size or entries.get(number) != (1, offset, generation):
            raise RuntimeError("PDF xref stream does not index itself exactly")
        return entries, dictionary

    def _section(self, offset):
        if self.data.startswith(b"xref", offset):
            entries, trailer = self._classic_xref(offset)
            hybrid = _pdf_get(trailer, "XRefStm")
            if hybrid is not None:
                hybrid_entries, hybrid_trailer = self._xref_stream(
                    _pdf_int(hybrid, "hybrid xref offset", 1))
                if _pdf_get(trailer, "Size") != _pdf_get(hybrid_trailer, "Size") or \
                        set(hybrid_entries) & set(entries):
                    raise RuntimeError("ambiguous hybrid PDF xref entries")
                hybrid_entries.update(entries)
                entries = hybrid_entries
                if _pdf_get(trailer, "Root") is None and _pdf_get(hybrid_trailer, "Root") is not None:
                    trailer[_PdfName(b"Root")] = _pdf_get(hybrid_trailer, "Root")
            return entries, trailer
        return self._xref_stream(offset)

    def _load_xref_chain(self, offset):
        seen_offsets = set()
        while offset is not None:
            if offset in seen_offsets or len(seen_offsets) > _PDF_MAX_OBJECTS:
                raise RuntimeError("PDF xref /Prev cycle")
            seen_offsets.add(offset)
            entries, trailer = self._section(offset)
            section_size = _pdf_int(_pdf_get(trailer, "Size"), "trailer Size", 1)
            if section_size > _PDF_MAX_OBJECTS:
                raise RuntimeError("PDF trailer Size limit")
            if self.size is None:
                self.size = section_size
            elif section_size > self.size:
                raise RuntimeError("PDF prior revision /Size exceeds latest revision")
            if any(number >= self.size for number in entries):
                raise RuntimeError("PDF xref entry outside latest /Size")
            if _pdf_get(trailer, "Encrypt") is not None:
                raise RuntimeError("encrypted PDF is unsupported")
            if self.root is None and _pdf_get(trailer, "Root") is not None:
                self.root = _pdf_get(trailer, "Root")
            for number, entry in entries.items():
                if number not in self.xref:
                    self.xref[number] = entry
            previous = _pdf_get(trailer, "Prev")
            offset = None if previous is None else _pdf_int(previous, "Prev offset", 1)
        if len(self.xref) > _PDF_MAX_OBJECTS:
            raise RuntimeError("PDF xref object limit")
        if self.size is None or set(self.xref) != set(range(self.size)):
            raise RuntimeError("PDF xref entries do not exactly cover /Size")
        if self.xref.get(0) != (0, 0, 65535):
            raise RuntimeError("PDF object zero xref entry is invalid")
        for entry_type, first, second in self.xref.values():
            if entry_type == 0 and (first >= self.size or second > 65535):
                raise RuntimeError("invalid free PDF xref entry")
            if entry_type == 1 and (first <= 0 or first >= len(self.data) or second > 65535):
                raise RuntimeError("invalid in-use PDF xref entry")
            if entry_type == 2 and (first <= 0 or first >= self.size or
                                    second >= _PDF_MAX_OBJECTS):
                raise RuntimeError("invalid compressed PDF xref entry")

    def _validate_in_use_offsets(self):
        for number, entry in self.xref.items():
            if entry[0] != 1:
                continue
            offset, generation = entry[1], entry[2]
            lexer = _PdfLexer(self.data, offset)
            if lexer.skip() != offset or lexer.take() != number or \
                    lexer.take() != generation or lexer.take() != b"obj":
                raise RuntimeError("PDF xref offset/object mismatch")

    def _length(self, reference):
        value, stream = self.resolve(reference)
        if stream is not None:
            raise RuntimeError("PDF stream length resolves to stream")
        return _pdf_int(value, "indirect stream length")

    def _entry_reference(self, number):
        entry = self.xref.get(number)
        if entry is None or entry[0] == 0:
            raise RuntimeError("missing PDF object")
        return _PdfRef(number, entry[2] if entry[0] == 1 else 0)

    def _object_stream(self, number):
        if number in self.object_stream_cache:
            return self.object_stream_cache[number]
        reference = self._entry_reference(number)
        entry = self.xref[number]
        if entry[0] != 1:
            raise RuntimeError("nested compressed PDF object stream")
        _, _, dictionary, stream = _parse_pdf_indirect(
            self.data, entry[1], self._length)
        if not isinstance(dictionary, dict) or stream is None or \
                _pdf_get(dictionary, "Type") != _PdfName(b"ObjStm"):
            raise RuntimeError("invalid PDF object stream")
        count = _pdf_int(_pdf_get(dictionary, "N"), "object stream N", 1)
        first = _pdf_int(_pdf_get(dictionary, "First"), "object stream First")
        if count > _PDF_MAX_OBJECTS:
            raise RuntimeError("PDF object stream count limit")
        decoded = _decode_pdf_stream(dictionary, stream)
        if first > len(decoded):
            raise RuntimeError("PDF object stream header out of bounds")
        header = _PdfLexer(decoded, 0, first)
        pairs = []
        for _ in range(count):
            object_number = _pdf_int(header.take(), "object stream number", 1)
            relative = _pdf_int(header.take(), "object stream offset")
            pairs.append((object_number, relative))
        if header.take() is not None or len({item[0] for item in pairs}) != len(pairs):
            raise RuntimeError("invalid PDF object stream header")
        if any(pairs[index][1] > pairs[index + 1][1] for index in range(len(pairs) - 1)):
            raise RuntimeError("unsorted PDF object stream offsets")
        objects = []
        for index, (object_number, relative) in enumerate(pairs):
            start = first + relative
            end = first + pairs[index + 1][1] if index + 1 < len(pairs) else len(decoded)
            if start > end or end > len(decoded):
                raise RuntimeError("PDF object stream offset out of bounds")
            lexer = _PdfLexer(decoded, start, end)
            value = lexer.value()
            if lexer.take() is not None:
                raise RuntimeError("PDF object stream trailing tokens")
            objects.append((object_number, value))
        self.object_stream_cache[number] = objects
        return objects

    def resolve(self, reference):
        if not isinstance(reference, _PdfRef):
            raise RuntimeError("expected PDF indirect reference")
        key = (reference.number, reference.generation)
        if key in self.cache:
            return self.cache[key]
        if key in self.resolving:
            raise RuntimeError("cyclic PDF object resolution")
        self.resolving.add(key)
        entry = self.xref.get(reference.number)
        try:
            if entry is None or entry[0] == 0:
                raise RuntimeError("PDF reference resolves to free/missing object")
            if entry[0] == 1:
                if entry[2] != reference.generation:
                    raise RuntimeError("PDF reference generation mismatch")
                number, generation, value, stream = _parse_pdf_indirect(
                    self.data, entry[1], self._length)
                if (number, generation) != key:
                    raise RuntimeError("PDF object identity mismatch")
            else:
                if reference.generation != 0:
                    raise RuntimeError("compressed PDF object generation mismatch")
                objects = self._object_stream(entry[1])
                if entry[2] >= len(objects) or objects[entry[2]][0] != reference.number:
                    raise RuntimeError("PDF compressed object index mismatch")
                value, stream = objects[entry[2]][1], None
            self.cache[key] = (value, stream)
            return value, stream
        finally:
            self.resolving.remove(key)


def _verified_pdf_page_count(data):
    """Resolve and validate the latest catalog and its reachable page tree."""
    if not isinstance(data, bytes):
        raise RuntimeError("PDF input is not bytes")
    resolver = _PdfResolver(data)
    catalog, catalog_stream = resolver.resolve(resolver.root)
    if catalog_stream is not None or not isinstance(catalog, dict) or \
            _pdf_get(catalog, "Type") != _PdfName(b"Catalog"):
        raise RuntimeError("invalid PDF Catalog")
    pages_reference = _pdf_get(catalog, "Pages")
    if not isinstance(pages_reference, _PdfRef):
        raise RuntimeError("PDF Catalog lacks Pages reference")
    active = set()
    reached = set()

    def walk(reference, parent, depth):
        if depth > _PDF_MAX_DEPTH:
            raise RuntimeError("PDF page-tree depth limit")
        key = tuple(reference)
        if key in active:
            raise RuntimeError("PDF page-tree cycle")
        if key in reached:
            raise RuntimeError("duplicate PDF page-tree child")
        active.add(key)
        reached.add(key)
        value, stream = resolver.resolve(reference)
        if stream is not None or not isinstance(value, dict):
            raise RuntimeError("PDF page-tree node is not a dictionary")
        node_type = _pdf_get(value, "Type")
        declared_parent = _pdf_get(value, "Parent")
        if parent is None:
            if declared_parent is not None:
                raise RuntimeError("PDF root Pages has Parent")
        elif declared_parent != parent:
            raise RuntimeError("PDF page-tree Parent mismatch")
        if node_type == _PdfName(b"Page"):
            if _pdf_get(value, "Kids") is not None:
                raise RuntimeError("PDF Page unexpectedly has Kids")
            count = 1
        elif node_type == _PdfName(b"Pages"):
            kids = _pdf_get(value, "Kids")
            declared_count = _pdf_int(_pdf_get(value, "Count"), "page-tree Count")
            if not isinstance(kids, list) or not kids or \
                    not all(isinstance(child, _PdfRef) for child in kids):
                raise RuntimeError("invalid PDF Pages Kids")
            if len({tuple(child) for child in kids}) != len(kids):
                raise RuntimeError("duplicate PDF Pages Kids")
            count = sum(walk(child, reference, depth + 1) for child in kids)
            if declared_count != count:
                raise RuntimeError("PDF page-tree Count mismatch")
        else:
            raise RuntimeError("invalid PDF page-tree node Type")
        active.remove(key)
        return count

    pages = walk(pages_reference, None, 0)
    if pages <= 0:
        raise RuntimeError("PDF has no reachable Page leaves")
    return pages


def _paper_record(boundary, manifest, log, hooks=None):
    paper_dir = boundary["paper_directory"]
    pdir = _open_dir(paper_dir["path"])
    try:
        pbefore = os.fstat(pdir)
        _check_identity(pbefore, paper_dir, "paper directory")
        spec = boundary["paper"]
        if os.path.dirname(spec["path"]) != paper_dir["path"]:
            raise RuntimeError("paper parent binding mismatch")
        lst = _logged_lstat(pdir, spec["name"], "paper", spec["path"], log)
        _check_identity(lst, spec, "paper")
        raw_paper_path = os.fsencode(spec["path"])
        log.emit("paper", raw_paper_path, "pdf_write_open_denial", "before", lst,
                 "pending")
        try:
            write_fd = os.open(spec["name"], os.O_WRONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                               dir_fd=pdir)
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EPERM):
                log.emit("paper", raw_paper_path, "pdf_write_open_denial", "after", lst,
                         "error", exc.errno, 0)
                raise RuntimeError("unexpected exact-PDF write-open errno") from exc
            log.emit("paper", raw_paper_path, "pdf_write_open_denial", "after", lst,
                     "denied", exc.errno, 0)
        else:
            os.close(write_fd)
            log.emit("paper", raw_paper_path, "pdf_write_open_denial", "after", lst,
                     "ok", 0, 0)
            raise RuntimeError("exact-PDF write-capable open was not denied")
        # The first action after the denial is a stable no-follow full read that
        # revalidates the exact identity, bytes, size, and locked digest.
        fd = _logged_open(manifest, pdir, spec["name"], "paper", spec["path"], log)
        try:
            data, digest, count, st = _read_logged(fd, "paper", spec["path"], log, spec,
                                                   hooks.get("paper") if hooks else None)
        finally:
            os.close(fd)
        if digest != spec["sha256"] or count != spec["size"]:
            raise RuntimeError("paper hash or size drift")
        pages = _verified_pdf_page_count(data)
        log.emit("paper", os.fsencode(spec["path"]), "pdf_buffer_parse", "before", st,
                 "pending")
        log.emit("paper", os.fsencode(spec["path"]), "pdf_buffer_parse", "after", st,
                 byte_count=0)
        if pages != spec["pages"]:
            raise RuntimeError("paper page drift")
        pafter = os.fstat(pdir)
        if manifest.stat_tuple(pbefore) != manifest.stat_tuple(pafter):
            raise RuntimeError("paper directory changed")
        named_post = _logged_lstat(pdir, spec["name"], "paper", spec["path"], log,
                                   "lstat_post")
        if manifest.stat_tuple(lst) != manifest.stat_tuple(named_post):
            raise RuntimeError("paper name substitution")
        parent_check = _open_dir(paper_dir["path"])
        try:
            _check_identity(os.fstat(parent_check), paper_dir, "paper directory post-validation")
        finally:
            os.close(parent_check)
        return {
            "schema": "g0-paper-record/v5", "sealed_run_id": boundary["sealed_run_id"],
            "bundle_id": boundary["bundle_id"],
            "record_id": sha(b"paper\0" + raw_paper_path),
            "path": raw_paper_path.decode("utf-8", "surrogateescape"),
            "path_b64": base64.b64encode(raw_paper_path).decode("ascii"),
            "root_id": "paper", "entry_type": "regular",
            "parent_path": paper_dir["path"], "parent_dev": pbefore.st_dev,
            "parent_inode": pbefore.st_ino, "dev": st.st_dev, "inode": st.st_ino,
            "mode": st.st_mode, "nlink": st.st_nlink, "uid": st.st_uid,
            "gid": st.st_gid, "size": count, "mtime_ns": st.st_mtime_ns,
            "ctime_ns": st.st_ctime_ns, "sha256": digest, "byte_count": count,
            "pages": pages,
        }
    finally:
        os.close(pdir)


def _valid_int(value, minimum=0):
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _validate_record_identity(value):
    if value is None:
        return
    _expect_keys(value, IDENTITY_KEYS, "event record identity")
    if any(not _valid_int(value[key]) for key in IDENTITY_KEYS):
        raise RuntimeError("event record identity value")
    if value["mode"] == 0 or value["nlink"] == 0:
        raise RuntimeError("event record identity impossible")


def _mode_entry_type(mode):
    if stat.S_ISREG(mode):
        return "regular"
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISLNK(mode):
        return "symlink"
    return "other"


def _decode_event_path(event):
    try:
        raw = base64.b64decode(event["path_b64"], validate=True)
    except Exception as exc:
        raise RuntimeError("protected access raw path encoding") from exc
    plain = raw.decode("utf-8", "surrogateescape")
    if base64.b64encode(raw).decode("ascii") != event["path_b64"] or \
            plain != event["path"] or plain.encode("utf-8", "surrogateescape") != raw:
        raise RuntimeError("protected access path correspondence")
    return raw


def _validate_operation_identity(event):
    identity = event["record_identity"]
    if identity is None:
        return
    entry_type = _mode_entry_type(identity["mode"])
    directory_operations = {"enumerate", "open_directory", "directory_fstat_post",
                            "root_fd_validate", "root_path_revalidate", "lstat_managed"}
    regular_operations = {"open_regular", "read_file", "fstat_post", "read_declaration",
                          "pdf_buffer_parse", "lstat_post", "pdf_write_open_denial"}
    if event["operation"] in directory_operations and entry_type != "directory":
        raise RuntimeError("protected access operation requires directory")
    if event["operation"] in regular_operations and entry_type != "regular":
        raise RuntimeError("protected access operation requires regular file")
    if event["operation"] == "readlink" and entry_type != "symlink":
        raise RuntimeError("protected access readlink requires symlink")


def _validate_event_list(events, phase, sealed_run_id, bundle_sha256, task_id, envelope_id,
                         allowed_root_ids=None):
    if not isinstance(events, list) or not events:
        raise RuntimeError("empty protected access event stream")
    if [event.get("sequence") for event in events] != list(range(1, len(events) + 1)):
        raise RuntimeError("conductor event sequence mismatch")
    for event in events:
        _expect_keys(event, EVENT_KEYS, "protected access event")
        if event["schema"] != "g0-protected-access-event/v5" or \
                event["phase"] != phase or event["sealed_run_id"] != sealed_run_id or \
                event["task_id"] != task_id or event["role"] != "executor" or \
                event["bundle_sha256"] != bundle_sha256 or \
                event["envelope_id"] != envelope_id or event["provider_ready"] is not True:
            raise RuntimeError("conductor event binding mismatch")
        if not _valid_int(event["sequence"], 1) or not isinstance(event["root_id"], str) or \
                not event["root_id"] or not isinstance(event["path"], str) or \
                not isinstance(event["path_b64"], str) or \
                not isinstance(event["record_id"], str) or len(event["record_id"]) != 64 or \
                not isinstance(event["operation"], str) or event["operation"] not in ALLOWED_OPERATIONS or \
                event["stage"] not in ("before", "after") or not isinstance(event["result"], str) or \
                not _valid_int(event["errno"]) or not _valid_int(event["bytes_read"]) or \
                (event["content_sha256"] is not None and not re.fullmatch(
                    r"[0-9a-f]{64}", event["content_sha256"])):
            raise RuntimeError("protected access event value")
        if allowed_root_ids is not None and event["root_id"] not in allowed_root_ids:
            raise RuntimeError("protected access root-id domain")
        expected_operations = ROOT_OPERATION_MAP.get(event["root_id"], SOURCE_OPERATIONS)
        if event["operation"] not in expected_operations:
            raise RuntimeError("protected access operation/root pairing")
        raw = _decode_event_path(event)
        expected_record_id = sha(event["root_id"].encode("utf-8") + b"\0" + raw)
        if event["record_id"] != expected_record_id:
            raise RuntimeError("protected access record-id derivation")
        _validate_record_identity(event["record_identity"])
        _validate_operation_identity(event)
        if event["stage"] == "before":
            if event["result"] != "pending" or event["errno"] != 0 or \
                    event["bytes_read"] != 0 or event["content_sha256"] is not None:
                raise RuntimeError("protected access before semantics")
        elif event["result"] == "ok":
            if event["errno"] != 0 or event["operation"] in WRITE_OPERATIONS:
                raise RuntimeError("protected access success semantics")
            if event["record_identity"] is None:
                raise RuntimeError("protected access success identity")
            if event["operation"] not in READ_BYTE_OPERATIONS and event["bytes_read"] != 0:
                raise RuntimeError("protected access bytes semantics")
            if event["operation"] in {"read_file", "read_declaration", "readlink"} and \
                    event["bytes_read"] != event["record_identity"]["size"]:
                raise RuntimeError("protected access full-read byte semantics")
            if event["operation"] == "final_revalidate":
                expected_bytes = event["record_identity"]["size"] if _mode_entry_type(
                    event["record_identity"]["mode"]) in ("regular", "symlink") else 0
                if event["bytes_read"] != expected_bytes:
                    raise RuntimeError("protected access revalidation byte semantics")
            hashed = event["operation"] in {"read_file", "read_declaration", "readlink"} or \
                (event["operation"] == "final_revalidate" and _mode_entry_type(
                    event["record_identity"]["mode"]) in ("regular", "symlink"))
            if hashed != (event["content_sha256"] is not None):
                raise RuntimeError("protected access content-hash semantics")
        elif event["result"] == "denied":
            if event["operation"] not in WRITE_OPERATIONS or \
                    event["errno"] not in (errno.EACCES, errno.EPERM) or \
                    event["bytes_read"] != 0 or event["content_sha256"] is not None:
                raise RuntimeError("protected access denial semantics")
            if event["operation"] == "pdf_write_open_denial":
                if event["record_identity"] is None or _mode_entry_type(
                        event["record_identity"]["mode"]) != "regular":
                    raise RuntimeError("exact-PDF denial identity semantics")
            elif event["record_identity"] is not None:
                raise RuntimeError("disposable denial identity semantics")
        elif event["result"] == "error":
            if event["errno"] <= 0 or event["bytes_read"] != 0 or \
                    event["record_identity"] is not None or event["content_sha256"] is not None:
                raise RuntimeError("protected access error semantics")
        else:
            raise RuntimeError("protected access result semantics")
    if len(events) % 2:
        raise RuntimeError("unpaired protected access event")
    paired_fields = ("phase", "sealed_run_id", "task_id", "role", "bundle_sha256",
                     "envelope_id", "root_id", "record_id", "path", "path_b64", "operation")
    for before, after in zip(events[0::2], events[1::2]):
        if before["stage"] != "before" or after["stage"] != "after" or \
                any(before[field] != after[field] for field in paired_fields):
            raise RuntimeError("protected access before/after mismatch")
        if before["record_identity"] is not None and after["record_identity"] is not None and \
                before["record_identity"] != after["record_identity"]:
            raise RuntimeError("protected access paired record identity mismatch")
    return events


def _decode_record_path(record, label):
    try:
        raw = base64.b64decode(record["path_b64"], validate=True)
    except Exception as exc:
        raise RuntimeError(label + " path encoding") from exc
    display = raw.decode("utf-8", "surrogateescape")
    if record["path"] != display or display.encode("utf-8", "surrogateescape") != raw or \
            base64.b64encode(raw).decode("ascii") != record["path_b64"]:
        raise RuntimeError(label + " path correspondence")
    return raw


def _parse_source_records(data, sealed_run_id, bundle_sha256, envelope_id):
    records = []
    for line in data.splitlines(keepends=True):
        try:
            record = json.loads(line)
        except Exception as exc:
            raise RuntimeError("invalid source manifest JSONL") from exc
        if line != canon(record):
            raise RuntimeError("noncanonical source manifest JSONL")
        base_keys = {"schema", "record_type", "root_id", "root_path", "record_id",
                     "path", "path_b64", "entry_type", *IDENTITY_KEYS,
                     "sealed_run_id", "bundle_sha256", "envelope_id"}
        extra = set()
        if record.get("entry_type") == "regular":
            extra = {"sha256", "byte_count"}
        elif record.get("entry_type") == "symlink":
            extra = {"target", "target_b64"}
        _expect_keys(record, base_keys | extra, "source record")
        if record["schema"] != "g0-source-record/v5" or \
                record["record_type"] not in ("root", "entry") or \
                not isinstance(record["root_id"], str) or not record["root_id"] or \
                not isinstance(record["root_path"], str) or \
                record["sealed_run_id"] != sealed_run_id or \
                record["bundle_sha256"] != bundle_sha256 or \
                record["envelope_id"] != envelope_id:
            raise RuntimeError("source record binding")
        _validate_record_identity({key: record[key] for key in IDENTITY_KEYS})
        if _mode_entry_type(record["mode"]) != record["entry_type"]:
            raise RuntimeError("source record type/identity mismatch")
        raw = _decode_record_path(record, "source record")
        if record["record_type"] == "root":
            if record["entry_type"] != "directory" or raw != os.fsencode(record["root_path"]):
                raise RuntimeError("source root record path/type mismatch")
        else:
            components = raw.split(b"/")
            if not raw or raw.startswith(b"/") or any(
                    component in (b"", b".", b"..") or b"\0" in component
                    for component in components):
                raise RuntimeError("source entry relative path")
        if record["record_id"] != sha(record["root_id"].encode("utf-8") + b"\0" + raw):
            raise RuntimeError("source record-id derivation")
        if record["entry_type"] == "regular":
            if record["byte_count"] != record["size"] or not isinstance(
                    record["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"]):
                raise RuntimeError("source regular byte/hash semantics")
        elif record["entry_type"] == "symlink":
            try:
                target = base64.b64decode(record["target_b64"], validate=True)
            except Exception as exc:
                raise RuntimeError("source symlink target encoding") from exc
            target_display = target.decode("utf-8", "surrogateescape")
            if record["target"] != target_display or target_display.encode(
                    "utf-8", "surrogateescape") != target or len(target) != record["size"]:
                raise RuntimeError("source symlink target semantics")
        records.append(record)
    if not records:
        raise RuntimeError("empty source manifest")
    identities = [(record["root_id"], record["record_id"]) for record in records]
    if len(identities) != len(set(identities)):
        raise RuntimeError("duplicate source record identity")
    roots = [record for record in records if record["record_type"] == "root"]
    if len(roots) != len({record["root_id"] for record in roots}):
        raise RuntimeError("source root record multiplicity")
    root_map = {record["root_id"]: record for record in roots}
    if set(root_map) != {record["root_id"] for record in records}:
        raise RuntimeError("source entry lacks its root record")
    for record in records:
        root = root_map[record["root_id"]]
        if not os.path.isabs(record["root_path"]) or \
                record["root_path"] != root["root_path"]:
            raise RuntimeError("source record/root path mismatch")
    def full_raw(record):
        raw = _decode_record_path(record, "source record")
        return raw if record["record_type"] == "root" else \
            os.fsencode(record["root_path"]) + b"/" + raw
    physical = [full_raw(record) for record in records]
    if len(physical) != len(set(physical)):
        raise RuntimeError("duplicate source physical path")
    if physical != sorted(physical):
        raise RuntimeError("source manifest raw-path order mismatch")
    return records


def _parse_paper_record(data, sealed_run_id, bundle_sha256, expected_bundle_id=None):
    try:
        record = json.loads(data)
    except Exception as exc:
        raise RuntimeError("invalid paper record") from exc
    if data != canon(record):
        raise RuntimeError("noncanonical paper record")
    keys = {"schema", "sealed_run_id", "bundle_id", "bundle_sha256", "record_id",
            "root_id", "entry_type", "path", "path_b64", "parent_path",
            "parent_dev", "parent_inode", "dev", "inode", "mode", "nlink", "uid",
            "gid", "size", "mtime_ns", "ctime_ns", "sha256", "byte_count", "pages"}
    _expect_keys(record, keys, "paper record")
    if record["schema"] != "g0-paper-record/v5" or \
            record["sealed_run_id"] != sealed_run_id or \
            record["bundle_sha256"] != bundle_sha256 or record["root_id"] != "paper" or \
            record["entry_type"] != "regular" or not isinstance(record["parent_path"], str) or \
            not os.path.isabs(record["parent_path"]) or \
            not isinstance(record["bundle_id"], str) or not record["bundle_id"] or \
            (expected_bundle_id is not None and record["bundle_id"] != expected_bundle_id):
        raise RuntimeError("paper record binding")
    identity = {key: record[key] for key in IDENTITY_KEYS}
    _validate_record_identity(identity)
    if _mode_entry_type(record["mode"]) != "regular" or \
            record["byte_count"] != record["size"] or not _valid_int(record["pages"], 1) or \
            not isinstance(record["sha256"], str) or not re.fullmatch(
                r"[0-9a-f]{64}", record["sha256"]):
        raise RuntimeError("paper record byte/type semantics")
    raw = _decode_record_path(record, "paper record")
    parent_raw = raw.rsplit(b"/", 1)[0]
    if not raw.startswith(b"/") or not parent_raw or \
            parent_raw != os.fsencode(record["parent_path"]) or \
            not _valid_int(record["parent_dev"]) or not _valid_int(record["parent_inode"]) or \
            record["record_id"] != sha(b"paper\0" + raw):
        raise RuntimeError("paper record-id/path semantics")
    return record


def _manifest_identity(record):
    return {key: record[key] for key in IDENTITY_KEYS}


def _operation_plan_item(root_id, raw_path, operation, result="ok"):
    return {
        "root_id": root_id,
        "record_id": sha(root_id.encode("utf-8") + b"\0" + raw_path),
        "path_b64": base64.b64encode(raw_path).decode("ascii"),
        "operation": operation,
        "result": result,
    }


def _source_operation_plan(source_records, root_order):
    by_root = {}
    for record in source_records:
        by_root.setdefault(record["root_id"], []).append(record)
    if set(by_root) != set(root_order):
        raise RuntimeError("source operation-plan root set mismatch")
    plan = []
    for root_id in root_order:
        records = by_root[root_id]
        roots = [record for record in records if record["record_type"] == "root"]
        if len(roots) != 1:
            raise RuntimeError("source operation-plan root multiplicity")
        root = roots[0]
        root_raw = _decode_record_path(root, "source root")
        entries = {}
        children = {}
        for record in records:
            if record["record_type"] == "root":
                continue
            raw = _decode_record_path(record, "source entry")
            if raw in entries:
                raise RuntimeError("duplicate source operation-plan path")
            entries[raw] = record
            parent = raw.rsplit(b"/", 1)[0] if b"/" in raw else b""
            children.setdefault(parent, []).append(raw)
        for parent, names in children.items():
            names.sort()
            if parent and (parent not in entries or entries[parent]["entry_type"] != "directory"):
                raise RuntimeError("source operation-plan parent is not a directory")
        preorder = []

        def add(raw, operation, result="ok"):
            plan.append(_operation_plan_item(root_id, raw, operation, result))

        def walk(directory_raw):
            event_raw = root_raw if directory_raw == b"" else directory_raw
            add(event_raw, "enumerate")
            for raw in children.get(directory_raw, []):
                record = entries[raw]
                preorder.append(raw)
                add(raw, "lstat")
                if record["entry_type"] == "directory":
                    add(raw, "open_directory")
                    walk(raw)
                elif record["entry_type"] == "regular":
                    add(raw, "open_regular")
                    add(raw, "read_file")
                    add(raw, "fstat_post")
                elif record["entry_type"] == "symlink":
                    add(raw, "readlink")
            add(event_raw, "directory_fstat_post")

        add(root_raw, "root_fd_validate")
        walk(b"")
        if len(preorder) != len(entries) or set(preorder) != set(entries):
            raise RuntimeError("source operation-plan traversal is incomplete")
        for raw in preorder:
            add(raw, "final_revalidate")
        add(root_raw, "root_path_revalidate")
    return plan


def _canonical_operation_plan(boundary, owner, provider_evidence, source_records,
                              paper_record):
    controls = provider_evidence.get("destructive_controls")
    if not isinstance(controls, dict):
        raise RuntimeError("provider destructive-control evidence missing")
    optional = {"setxattr", "removexattr"}
    mandatory = set(DENIAL_CONTROL_EVENT_ORDER) - optional
    present = set(controls)
    if present - {"scratch_create_rename_unlink"} - set(DENIAL_CONTROL_EVENT_ORDER) or \
            not mandatory <= present or "scratch_create_rename_unlink" not in present or \
            bool(optional & present) != (optional <= present):
        raise RuntimeError("provider destructive-control exact set mismatch")
    plan = [_operation_plan_item("denial-fixture", b".g0-synthetic-fixture.json",
                                 "read_declaration")]
    for name in DENIAL_CONTROL_EVENT_ORDER:
        if name in controls:
            plan.append(_operation_plan_item("denial-fixture", name.encode("ascii"),
                                             "write_denial_control", "denied"))
    paper_raw = _decode_record_path(paper_record, "paper record")
    for operation, result in (
            ("lstat", "ok"), ("pdf_write_open_denial", "denied"),
            ("open_regular", "ok"), ("read_file", "ok"),
            ("pdf_buffer_parse", "ok"), ("lstat_post", "ok")):
        plan.append(_operation_plan_item("paper", paper_raw, operation, result))
    owner_specs = ((owner["owner_file"], ("lstat", "open_regular", "read_file")),
                   (owner["readme"], ("lstat", "open_regular", "read_file")))
    for spec, operations in owner_specs:
        raw = os.fsencode(spec["name"])
        for operation in operations:
            plan.append(_operation_plan_item("owner", raw, operation))
    for prefix in sorted(owner["managed_prefixes"], key=os.fsencode):
        plan.append(_operation_plan_item("owner", os.fsencode(prefix), "lstat_managed"))
    for foreign in sorted(owner["foreign"], key=lambda item: os.fsencode(item["path"])):
        plan.append(_operation_plan_item("owner", os.fsencode(foreign["path"]),
                                         "lstat_foreign"))
    root_order = [root["root_id"] for root in boundary["source_roots"]]
    plan.extend(_source_operation_plan(source_records, root_order))
    return plan


def _actual_operation_plan(events):
    return [{key: event[key] for key in
             ("root_id", "record_id", "path_b64", "operation", "result")}
            for event in events if event["stage"] == "after"]


def _root_anchor(record):
    return {key: record[key] for key in ROOT_ANCHOR_KEYS}


def _paper_anchor(record):
    return {key: record[key] for key in PAPER_ANCHOR_KEYS}


def build_reconciliation_anchors(boundary_bytes, owner_bytes, provider_bytes, source,
                                 paper_bytes, bundle_sha256, bundle_id, sealed_run_id,
                                 envelope_id):
    """Issue trusted closure anchors from authenticated immutable producer state."""
    parsed = []
    for label, data in (("boundary", boundary_bytes), ("owner", owner_bytes),
                        ("provider", provider_bytes)):
        try:
            obj = json.loads(data)
        except Exception as exc:
            raise RuntimeError("invalid reconciliation " + label) from exc
        if data != canon(obj):
            raise RuntimeError("noncanonical reconciliation " + label)
        parsed.append(obj)
    boundary, owner, provider_evidence = parsed
    if boundary.get("sealed_run_id") != sealed_run_id or \
            boundary.get("bundle_id") != bundle_id or \
            owner.get("sealed_run_id") != sealed_run_id or owner.get("bundle_id") != bundle_id or \
            provider_evidence.get("sealed_run_id") != sealed_run_id or \
            provider_evidence.get("bundle_sha256") != bundle_sha256:
        raise RuntimeError("reconciliation anchor authority binding mismatch")
    source_records = _parse_source_records(source, sealed_run_id, bundle_sha256, envelope_id)
    paper_record = _parse_paper_record(paper_bytes, sealed_run_id, bundle_sha256, bundle_id)
    root_map = {record["root_id"]: record for record in source_records
                if record["record_type"] == "root"}
    root_order = [spec.get("root_id") for spec in boundary.get("source_roots", [])]
    if not root_order or len(root_order) != len(set(root_order)) or \
            root_order != sorted(root_order, key=os.fsencode) or set(root_order) != set(root_map):
        raise RuntimeError("reconciliation source-root order/set mismatch")
    roots = []
    for spec in boundary["source_roots"]:
        record = root_map[spec["root_id"]]
        if record["root_path"] != spec.get("path") or any(
                record[field] != spec.get(field) for field in ("dev", "inode", "mode", "size")):
            raise RuntimeError("reconciliation source-root authority mismatch")
        roots.append(_root_anchor(record))
    paper_spec = boundary.get("paper", {})
    paper_directory = boundary.get("paper_directory", {})
    for field in ("path", "dev", "inode", "mode", "size", "sha256", "pages"):
        if paper_record.get(field) != paper_spec.get(field):
            raise RuntimeError("reconciliation paper authority mismatch")
    if paper_record["parent_path"] != paper_directory.get("path") or \
            paper_record["parent_dev"] != paper_directory.get("dev") or \
            paper_record["parent_inode"] != paper_directory.get("inode"):
        raise RuntimeError("reconciliation paper-parent authority mismatch")
    foreign = owner.get("foreign")
    if not isinstance(foreign, list) or len(foreign) != 1:
        raise RuntimeError("reconciliation owner foreign exact set mismatch")
    for field in ("path", "parent_path", "parent_dev", "parent_inode", "dev", "inode",
                  "mode", "size", "sha256", "pages"):
        if foreign[0].get(field) != paper_record.get(field):
            raise RuntimeError("reconciliation owner/paper authority mismatch")
    operation_plan = _canonical_operation_plan(boundary, owner, provider_evidence,
                                               source_records, paper_record)
    return {
        "schema": "g0-reconciliation-anchors/v5", "sealed_run_id": sealed_run_id,
        "bundle_id": bundle_id, "bundle_sha256": bundle_sha256,
        "boundary_sha256": sha(boundary_bytes),
        "owner_binding_sha256": sha(owner_bytes),
        "provider_evidence_sha256": sha(provider_bytes),
        "source_manifest_sha256": sha(source),
        "paper_record_sha256": sha(paper_bytes),
        "source_roots": roots, "paper": _paper_anchor(paper_record),
        "record_set_sha256": sha(canon({"source": source_records,
                                          "paper": paper_record})),
        "operation_sequence_sha256": sha(canon(operation_plan)),
        "operation_pair_count": len(operation_plan),
    }


def _validate_reconciliation_anchors(anchors, expected_sha256, source_records,
                                     paper_record, source, paper_bytes, bundle_sha256,
                                     bundle_id, sealed_run_id, events, *,
                                     expected_boundary_sha256,
                                     expected_owner_binding_sha256,
                                     expected_provider_evidence_sha256):
    _expect_keys(anchors, RECONCILIATION_ANCHOR_KEYS, "reconciliation anchors")
    if not isinstance(expected_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", expected_sha256) or sha(canon(anchors)) != expected_sha256:
        raise RuntimeError("reconciliation anchor digest mismatch")
    if anchors["schema"] != "g0-reconciliation-anchors/v5" or \
            anchors["sealed_run_id"] != sealed_run_id or \
            anchors["bundle_id"] != bundle_id or \
            anchors["bundle_sha256"] != bundle_sha256 or \
            anchors["source_manifest_sha256"] != sha(source) or \
            anchors["paper_record_sha256"] != sha(paper_bytes):
        raise RuntimeError("reconciliation anchor content binding mismatch")
    for field in ("boundary_sha256", "owner_binding_sha256", "provider_evidence_sha256",
                  "record_set_sha256", "operation_sequence_sha256"):
        if not isinstance(anchors[field], str) or not re.fullmatch(r"[0-9a-f]{64}",
                                                                   anchors[field]):
            raise RuntimeError("reconciliation anchor hash field")
    authority = {
        "boundary_sha256": expected_boundary_sha256,
        "owner_binding_sha256": expected_owner_binding_sha256,
        "provider_evidence_sha256": expected_provider_evidence_sha256,
    }
    for field, expected in authority.items():
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise RuntimeError("invalid expected reconciliation authority hash " + field)
        if anchors[field] != expected:
            raise RuntimeError("reconciliation authority hash mismatch " + field)
    roots = [record for record in source_records if record["record_type"] == "root"]
    roots.sort(key=lambda record: os.fsencode(record["root_id"]))
    expected_roots = [_root_anchor(record) for record in roots]
    if anchors["source_roots"] != expected_roots or any(
            not isinstance(item, dict) or set(item) != ROOT_ANCHOR_KEYS
            for item in anchors["source_roots"]):
        raise RuntimeError("reconciliation source-root anchors mismatch")
    if anchors["paper"] != _paper_anchor(paper_record) or \
            set(anchors["paper"]) != PAPER_ANCHOR_KEYS:
        raise RuntimeError("reconciliation paper anchor mismatch")
    if anchors["record_set_sha256"] != sha(canon({"source": source_records,
                                                   "paper": paper_record})):
        raise RuntimeError("reconciliation record-set anchor mismatch")
    actual_plan = _actual_operation_plan(events)
    if not _valid_int(anchors["operation_pair_count"], 1) or \
            anchors["operation_pair_count"] != len(actual_plan) or \
            anchors["operation_sequence_sha256"] != sha(canon(actual_plan)):
        raise RuntimeError("protected global operation sequence mismatch")


def _require_event_operations(events, record, required):
    actual = {}
    for event in events:
        if event["stage"] == "after":
            key = (event["operation"], event["result"])
            actual[key] = actual.get(key, 0) + 1
    expected = {(operation, result): 1 for operation, result in required}
    if actual != expected:
        raise RuntimeError("protected event/record operation closure mismatch " +
                           record["record_id"])


def _reconcile_events_to_records(events, source_records, paper_record):
    source_map = {(record["root_id"], record["record_id"]): record
                  for record in source_records}
    source_events = {}
    for event in events:
        key = (event["root_id"], event["record_id"])
        if event["root_id"] in {record["root_id"] for record in source_records}:
            record = source_map.get(key)
            if record is None or _decode_event_path(event) != _decode_record_path(
                    record, "source record"):
                raise RuntimeError("protected event/source record mapping")
            if event["record_identity"] is not None and \
                    event["record_identity"] != _manifest_identity(record):
                raise RuntimeError("protected event/source identity mapping")
            source_events.setdefault(key, []).append(event)
    for key, record in source_map.items():
        related = source_events.get(key, [])
        entry_type = record["entry_type"]
        if record["record_type"] == "root":
            required = [("root_fd_validate", "ok"), ("enumerate", "ok"),
                        ("directory_fstat_post", "ok"), ("root_path_revalidate", "ok")]
        elif entry_type == "regular":
            required = [("lstat", "ok"), ("open_regular", "ok"), ("read_file", "ok"),
                        ("fstat_post", "ok"), ("final_revalidate", "ok")]
        elif entry_type == "directory":
            required = [("lstat", "ok"), ("open_directory", "ok"), ("enumerate", "ok"),
                        ("directory_fstat_post", "ok"), ("final_revalidate", "ok")]
        elif entry_type == "symlink":
            required = [("lstat", "ok"), ("readlink", "ok"),
                        ("final_revalidate", "ok")]
        else:
            required = [("lstat", "ok"), ("final_revalidate", "ok")]
        _require_event_operations(related, record, required)
        for event in related:
            if event["stage"] != "after" or event["result"] != "ok":
                continue
            if event["operation"] in ("read_file", "final_revalidate") and \
                    entry_type == "regular" and (event["bytes_read"] != record["byte_count"] or
                                                  event["content_sha256"] != record["sha256"]):
                raise RuntimeError("protected event/source byte/hash mismatch")
            if event["operation"] in ("readlink", "final_revalidate") and \
                    entry_type == "symlink":
                target = base64.b64decode(record["target_b64"], validate=True)
                if event["bytes_read"] != record["size"] or \
                        event["content_sha256"] != sha(target):
                    raise RuntimeError("protected event/symlink byte/hash mismatch")
    paper_events = [event for event in events if event["root_id"] == "paper"]
    paper_raw = _decode_record_path(paper_record, "paper record")
    for event in paper_events:
        if event["record_id"] != paper_record["record_id"] or \
                _decode_event_path(event) != paper_raw or \
                (event["record_identity"] is not None and
                 event["record_identity"] != _manifest_identity(paper_record)):
            raise RuntimeError("protected event/paper record mapping")
        if event["stage"] == "after" and event["operation"] == "read_file" and \
                (event["bytes_read"] != paper_record["byte_count"] or
                 event["content_sha256"] != paper_record["sha256"]):
            raise RuntimeError("protected event/paper byte/hash mismatch")
    _require_event_operations(paper_events, paper_record, [
        ("lstat", "ok"), ("pdf_write_open_denial", "denied"),
        ("open_regular", "ok"), ("read_file", "ok"), ("pdf_buffer_parse", "ok"),
        ("lstat_post", "ok"),
    ])
    denial_after = next(index for index, event in enumerate(events)
                        if event["root_id"] == "paper" and
                        event["operation"] == "pdf_write_open_denial" and
                        event["stage"] == "after")
    immediate_read = [
        ("open_regular", "before"), ("open_regular", "after"),
        ("read_file", "before"), ("read_file", "after"),
    ]
    following = events[denial_after + 1:denial_after + 5]
    if len(following) != len(immediate_read) or \
            any(event["root_id"] != "paper" or
                (event["operation"], event["stage"]) != expected
                for event, expected in zip(following, immediate_read)):
        raise RuntimeError("exact-PDF denial is not immediately followed by stable read")


def _event_metrics(events):
    return {
        "content_read_bytes": sum(event["bytes_read"] for event in events
                                  if event["stage"] == "after" and
                                  event["operation"] in READ_BYTE_OPERATIONS and
                                  event["result"] == "ok"),
        "unwrapped_event_count": sum(event["role"] != "executor" or
                                     event["provider_ready"] is not True for event in events),
        "protected_write_count": sum(event["stage"] == "after" and
                                     event["operation"] in WRITE_OPERATIONS and
                                     event["result"] == "ok" for event in events),
    }


def closure_for(events, source, paper_bytes, bundle_sha256, phase, sealed_run_id,
                trusted_anchors_sha256):
    if not isinstance(trusted_anchors_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", trusted_anchors_sha256):
        raise RuntimeError("missing trusted reconciliation anchor digest")
    event_bytes = b"".join(canon(event) for event in events)
    sequences = [event.get("sequence") for event in events]
    metrics = _event_metrics(events)
    return {
        "schema": "g0-access-stream-closure/v5", "phase": phase,
        "sealed_run_id": sealed_run_id,
        "bundle_sha256": bundle_sha256,
        "trusted_anchors_sha256": trusted_anchors_sha256,
        "access_event_count": len(events),
        "access_sha256": sha(event_bytes), "source_sha256": sha(source),
        "paper_record_sha256": sha(paper_bytes),
        "content_read_bytes": metrics["content_read_bytes"],
        "first_sequence": sequences[0] if sequences else 0,
        "last_sequence": sequences[-1] if sequences else 0,
        "unwrapped_event_count": metrics["unwrapped_event_count"],
        "protected_write_count": metrics["protected_write_count"],
    }


def verify_closure(closure, events, source, paper_bytes, bundle_sha256, phase, sealed_run_id,
                   task_id, envelope_id, allowed_root_ids, expected_bundle_id,
                   trusted_anchors, trusted_anchors_sha256,
                   expected_boundary_sha256, expected_owner_binding_sha256,
                   expected_provider_evidence_sha256):
    if allowed_root_ids is None or expected_bundle_id is None or \
            trusted_anchors is None or trusted_anchors_sha256 is None:
        raise RuntimeError("mandatory reconciliation authority is missing")
    _validate_event_list(events, phase, sealed_run_id, bundle_sha256, task_id, envelope_id,
                         allowed_root_ids)
    source_records = _parse_source_records(source, sealed_run_id, bundle_sha256, envelope_id)
    paper_record = _parse_paper_record(paper_bytes, sealed_run_id, bundle_sha256,
                                       expected_bundle_id)
    _validate_reconciliation_anchors(
        trusted_anchors, trusted_anchors_sha256, source_records, paper_record, source,
        paper_bytes, bundle_sha256, expected_bundle_id, sealed_run_id, events,
        expected_boundary_sha256=expected_boundary_sha256,
        expected_owner_binding_sha256=expected_owner_binding_sha256,
        expected_provider_evidence_sha256=expected_provider_evidence_sha256)
    anchored_roots = {item["root_id"] for item in trusted_anchors["source_roots"]}
    if set(allowed_root_ids) != anchored_roots | {"paper", "owner", "denial-fixture"}:
        raise RuntimeError("protected access root authority mismatch")
    _reconcile_events_to_records(events, source_records, paper_record)
    expected = closure_for(events, source, paper_bytes, bundle_sha256, phase, sealed_run_id,
                           trusted_anchors_sha256)
    if closure != expected:
        raise RuntimeError("access closure mismatch")
    if expected["unwrapped_event_count"] != 0 or expected["protected_write_count"] != 0:
        raise RuntimeError("unwrapped or writing protected access")
    return b"".join(canon(event) for event in events)


def _stage_write(stage_fd, name, data):
    safe_name(name)
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                 0o400, dir_fd=stage_fd)
    try:
        _write_all(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)


def executor(args, state, modules, hooks=None, event_fd=None):
    manifest = modules["manifest"]
    provider = modules["provider"]
    config = state["inputs"]["manifest-config.json"][0]
    boundary = state["inputs"]["boundary.json"][0]
    owner = state["inputs"]["owner-binding.json"][0]
    boundary_keys = {"schema", "sealed_run_id", "bundle_id", "envelope_id", "source_roots",
                     "paper_directory", "paper", "disposable_denial_fixture"}
    if set(boundary) != boundary_keys or boundary.get("schema") != "g0-boundary-lock/v5" or \
            boundary.get("sealed_run_id") != args.sealed_run_id or \
            boundary.get("bundle_id") != state["bundle"]["bundle_id"]:
        raise RuntimeError("boundary schema")
    roots_spec = boundary.get("source_roots")
    required_ids = ["experiments4", "experiments5", "experiments6"]
    if not isinstance(roots_spec, list) or len(roots_spec) != 3 or \
            [root.get("root_id") for root in roots_spec] != required_ids:
        raise RuntimeError("boundary requires exact three source roots")
    root_paths = [root.get("path") for root in roots_spec]
    root_identities = [(root.get("dev"), root.get("inode")) for root in roots_spec]
    if len(set(root_paths)) != 3 or len(set(root_identities)) != 3:
        raise RuntimeError("duplicate protected source root")
    if any(not isinstance(path, str) or not os.path.isabs(path) for path in root_paths) or \
            any(not root.get("size") for root in roots_spec):
        raise RuntimeError("source root identity/path closure")
    paper_directory = boundary.get("paper_directory")
    paper = boundary.get("paper")
    if not isinstance(paper_directory, dict) or not isinstance(paper, dict) or \
            not os.path.isabs(paper_directory.get("path", "")) or \
            not os.path.isabs(paper.get("path", "")) or paper.get("size", 0) <= 0 or \
            paper.get("pages", 0) <= 0 or paper.get("name") != os.path.basename(paper["path"]):
        raise RuntimeError("nonempty paper boundary closure")
    foreign_paths = {item.get("path") for item in owner.get("foreign", [])}
    if foreign_paths != {paper["path"]}:
        raise RuntimeError("owner foreign set does not exactly bind the paper")
    scratch_fd = _open_dir(args.scratch)
    try:
        existing = set(os.listdir(scratch_fd))
        allowed = set() if args.mode == "pre" else {
            "pre", os.path.basename(args.accepted_cp0_seal_file),
            os.path.basename(args.terminal_readiness_file), "terminal-evidence",
        }
        if existing != allowed:
            raise RuntimeError("same-run retry rejected")
        os.mkdir(args.mode, 0o700, dir_fd=scratch_fd)
        stage_fd = os.open(args.mode, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                           dir_fd=scratch_fd)
        try:
            # Pin roots without enumerating descendants, then install the write boundary.
            roots = []
            for root in roots_spec:
                fd = _open_dir(root["path"])
                _check_identity(os.fstat(fd), root, "source root")
                roots.append((root, fd))
            paper_parent_fd = _open_dir(boundary["paper_directory"]["path"])
            _check_identity(os.fstat(paper_parent_fd), boundary["paper_directory"], "paper directory")
            os.close(paper_parent_fd)
            landlock = provider.enforce_landlock([scratch_fd])
            seccomp = provider.enforce_metadata_seccomp()
            log = AccessLog(args, state["bundle_sha256"], config, event_fd)
            log.activate(landlock, seccomp)
            if args.mode == "terminal-post":
                _reconcile_terminal_provider(state["pre"]["provider-evidence.json"],
                                             log.provider_identity, args, state)
            controls = _synthetic_controls(boundary, scratch_fd, log, provider)
            # The exact-PDF no-truncate write-open denial and its immediate
            # stable byte/identity revalidation precede source descendant enumeration.
            paper_record = _paper_record(boundary, manifest, log, hooks)
            owner_evidence = _validate_owner(owner, args.sealed_run_id, manifest, log,
                                             paper_record)
            records = []
            try:
                for root, fd in roots:
                    log.emit(root["root_id"], os.fsencode(root["path"]), "root_fd_validate",
                             "before", None, "pending")
                    log.emit(root["root_id"], os.fsencode(root["path"]), "root_fd_validate",
                             "after", os.fstat(fd))
                    records.extend(manifest.scan_root_fd(root["root_id"], root["path"], fd,
                                                         log.emit, hooks))
                    check = _open_dir(root["path"])
                    try:
                        _check_identity(os.fstat(check), root, "source root post-scan")
                        log.emit(root["root_id"], os.fsencode(root["path"]),
                                 "root_path_revalidate", "before", None, "pending")
                        log.emit(root["root_id"], os.fsencode(root["path"]),
                                 "root_path_revalidate", "after", os.fstat(check))
                    finally:
                        os.close(check)
            finally:
                for _, fd in roots:
                    os.close(fd)
            for record in records:
                record["sealed_run_id"] = args.sealed_run_id
                record["bundle_sha256"] = state["bundle_sha256"]
                record["envelope_id"] = config["envelope_id"]
            for root_id in required_ids:
                root_records = [record for record in records if record["root_id"] == root_id]
                if len(root_records) < 2 or not any(record["record_type"] == "entry" and
                                                    record["entry_type"] == "regular"
                                                    for record in root_records):
                    raise RuntimeError("nonempty source manifest required for " + root_id)
            source = manifest.manifest_bytes(records)
            paper = paper_record
            paper["bundle_sha256"] = state["bundle_sha256"]
            paper_bytes = canon(paper)
            access_bytes = log.validate()
            suffix = "pre" if args.mode == "pre" else "post"
            payloads = {
                "protected-access-" + suffix + ".jsonl": access_bytes,
                "source-" + suffix + ".jsonl": source,
                "paper-" + suffix + ".json": paper_bytes,
            }
            if args.mode == "pre":
                provider_evidence = {
                    "schema": "g0-provider-evidence/v5", "sealed_run_id": args.sealed_run_id,
                    "bundle_sha256": state["bundle_sha256"], "role": "executor",
                    "kernel_release": platform.release(), "architecture": platform.machine(),
                    "python_executable": state["inputs"]["runtime-lock.json"][0][
                        "python_executable"],
                    "python_executable_sha256": state["inputs"]["runtime-lock.json"][0][
                        "python_executable_sha256"],
                    "provider_module_sha256": state["files"]["scripts"]["provider.py"][
                        "sha256"],
                    "landlock": landlock, "seccomp": seccomp,
                    "destructive_controls": controls, "write_allow": "scan-scratch-only",
                    "claims": {"global_read_denial": False, "syscall_tracing": False},
                    "owner_validation": owner_evidence,
                }
                payloads["provider-evidence.json"] = canon(provider_evidence)
            expected = EXECUTOR_PRE_ORDER if args.mode == "pre" else EXECUTOR_POST_ORDER
            if set(payloads) != set(expected):
                raise RuntimeError("stage artifact exact-set mismatch")
            for name in expected:
                _stage_write(stage_fd, name, payloads[name])
            os.fsync(stage_fd)
            return {"order": expected, "hashes": {name: sha(payloads[name]) for name in expected},
                    "event_count": len(log.events), "event_sha256": sha(access_bytes),
                    "provider_identity": log.provider_identity}
        finally:
            os.close(stage_fd)
    finally:
        os.close(scratch_fd)


def _cleanup_temp(dfd, name, dev, ino):
    try:
        st = os.stat(name, dir_fd=dfd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if (st.st_dev, st.st_ino) == (dev, ino):
        os.unlink(name, dir_fd=dfd)


def publish_at(dfd, name, data, mode=0o444, inject=None):
    """Fsynced, hash-verified, mode-correct, per-file no-replace publication."""
    safe_name(name)
    temp = ".g0tmp-" + secrets.token_hex(16)
    fd = os.open(temp, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                 0o600, dir_fd=dfd)
    initial = os.fstat(fd)
    linked = False
    try:
        _write_all(fd, data)
        if inject:
            inject(name, "written")
        os.fsync(fd)
        os.fchmod(fd, mode)
        os.fsync(fd)
        os.lseek(fd, 0, os.SEEK_SET)
        got, digest, size, st = _stable_read(fd)
        if got != data or digest != sha(data) or size != len(data) or stat.S_IMODE(st.st_mode) != mode:
            raise RuntimeError("publication verification mismatch")
        if inject:
            inject(name, "verified")
        os.link(temp, name, src_dir_fd=dfd, dst_dir_fd=dfd, follow_symlinks=False)
        linked = True
        os.fsync(dfd)
        if inject:
            inject(name, "linked")
    finally:
        os.close(fd)
        _cleanup_temp(dfd, temp, initial.st_dev, initial.st_ino)
    return {"name": name, "sha256": sha(data), "size": len(data), "mode": mode,
            "linked": linked}


def _read_stage(stage_fd, name, expected_hash):
    data, digest, _, st = _read_at(stage_fd, name)
    if digest != expected_hash or stat.S_IMODE(st.st_mode) != 0o400:
        raise RuntimeError("stage verification mismatch " + name)
    return data


def publish_set(publication_fd, stage_fd, order, hashes, existing, existing_hashes=None,
                inject_after_links=None):
    if set(os.listdir(publication_fd)) != set(existing):
        raise RuntimeError("publication boundary exact-set mismatch")
    if set(os.listdir(stage_fd)) != set(order) or set(hashes) != set(order):
        raise RuntimeError("stage exact-set mismatch")
    if existing_hashes is not None:
        if set(existing_hashes) != set(existing):
            raise RuntimeError("existing hash plan mismatch")
        for name, expected_hash in existing_hashes.items():
            _, digest, _, st = _read_at(publication_fd, name)
            if digest != expected_hash or stat.S_IMODE(st.st_mode) != 0o444:
                raise RuntimeError("preexisting immutable input drift " + name)
    linked = 0
    for name in order:
        data = _read_stage(stage_fd, name, hashes[name])
        publish_at(publication_fd, name, data, 0o444)
        linked += 1
        if inject_after_links is not None and linked == inject_after_links:
            raise RuntimeError("injected publication boundary failure")
    if set(os.listdir(publication_fd)) != set(existing) | set(order):
        raise RuntimeError("published exact-set mismatch")
    for name in set(existing) | set(order):
        st = os.stat(name, dir_fd=publication_fd, follow_symlinks=False)
        if not stat.S_ISREG(st.st_mode) or stat.S_IMODE(st.st_mode) != 0o444:
            raise RuntimeError("published mode/type mismatch")
    if existing_hashes is not None:
        for name, expected_hash in existing_hashes.items():
            _, digest, _, _ = _read_at(publication_fd, name)
            if digest != expected_hash:
                raise RuntimeError("preexisting immutable input drift " + name)
    os.fsync(publication_fd)


def _close_fds_except(allowed):
    allowed = set(allowed) | {0, 1, 2}
    try:
        candidates = [int(name) for name in os.listdir("/proc/self/fd") if name.isdigit()]
    except OSError:
        limit = min(resource.getrlimit(resource.RLIMIT_NOFILE)[0], 1 << 20)
        candidates = range(3, limit)
    for fd in candidates:
        if fd not in allowed:
            try:
                os.close(fd)
            except OSError:
                pass


def _child_call(function, *args, capture_events=False):
    read_fd, write_fd = os.pipe2(os.O_CLOEXEC)
    event_read = event_write = None
    if capture_events:
        event_read, event_write = os.pipe2(os.O_CLOEXEC)
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        if event_read is not None:
            os.close(event_read)
        _close_fds_except({write_fd} | ({event_write} if event_write is not None else set()))
        try:
            result = function(*args, event_fd=event_write) if capture_events else function(*args)
            message = canon({"ok": True, "result": result})
            code = 0
        except BaseException as exc:
            message = canon({"ok": False, "error": type(exc).__name__ + ": " + str(exc)})
            code = 1
        try:
            _write_all(write_fd, message)
        finally:
            if event_write is not None:
                os.close(event_write)
            os.close(write_fd)
        os._exit(code)
    os.close(write_fd)
    if event_write is not None:
        os.close(event_write)
    streams = {read_fd: []}
    if event_read is not None:
        streams[event_read] = []
    open_fds = set(streams)
    while open_fds:
        ready, _, _ = select.select(sorted(open_fds), [], [])
        for fd in ready:
            chunk = os.read(fd, 65536)
            if chunk:
                streams[fd].append(chunk)
            else:
                os.close(fd)
                open_fds.remove(fd)
    _, status = os.waitpid(pid, 0)
    try:
        response = json.loads(b"".join(streams[read_fd]))
    except Exception as exc:
        raise RuntimeError("child produced invalid status") from exc
    if status != 0 or not response.get("ok"):
        raise RuntimeError(response.get("error", "child failed"))
    if capture_events:
        return response["result"], b"".join(streams[event_read])
    return response["result"]


def _publisher_child(args, result, existing_hashes, provider):
    publication_fd = _open_dir(args.publication)
    scratch_fd = _open_dir(args.scratch)
    try:
        stage_fd = os.open(args.mode, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                           dir_fd=scratch_fd)
        try:
            provider.enforce_publisher_landlock([scratch_fd, publication_fd], [publication_fd])
            existing = FINAL_INPUT_ORDER if args.mode == "pre" else PRE_ORDER
            publish_set(publication_fd, stage_fd, result["order"], result["hashes"], existing,
                        existing_hashes)
        finally:
            os.close(stage_fd)
    finally:
        os.close(scratch_fd)
        os.close(publication_fd)
    return {"published": result["order"], "seal": result["order"][-1]}


def _load_pre(publication, sealed_run_id, bundle_sha256, accepted_cp0_sha256):
    fd = _open_dir(publication)
    try:
        if set(os.listdir(fd)) != set(PRE_ORDER):
            raise RuntimeError("pre publication exact-set mismatch")
        out = {}
        for name in PRE_ORDER:
            data, _, _, st = _read_at(fd, name)
            if stat.S_IMODE(st.st_mode) != 0o444:
                raise RuntimeError("pre publication mode mismatch")
            out[name] = data
        if sha(out["cp0-seal.json"]) != accepted_cp0_sha256:
            raise RuntimeError("CP0 seal is not the externally accepted seal")
        seal = json.loads(out["cp0-seal.json"])
        _expect_keys(seal, {"schema", "result", "mode", "sealed_run_id", "bundle_sha256",
                            "artifacts", "exact_files"}, "CP0 seal")
        if seal.get("schema") != "g0-cp0-seal/v5" or seal.get("result") != "PASS" or \
                seal.get("mode") != "pre" or seal.get("sealed_run_id") != sealed_run_id or \
                seal.get("bundle_sha256") != bundle_sha256 or seal.get("exact_files") != PRE_ORDER:
            raise RuntimeError("invalid CP0 seal")
        if set(seal["artifacts"]) != set(PRE_ORDER) - {"cp0-seal.json"}:
            raise RuntimeError("CP0 sealed artifact exact-set mismatch")
        for name, digest in seal["artifacts"].items():
            if sha(out[name]) != digest:
                raise RuntimeError("CP0 artifact tamper")
        out["bundle_sha256"] = seal["bundle_sha256"]
        out["accepted_cp0_seal_sha256"] = accepted_cp0_sha256
        return out
    finally:
        os.close(fd)


def _accepted_cp0(args):
    if args.mode != "terminal-post":
        return None
    if os.path.dirname(args.accepted_cp0_seal_file) != args.scratch or \
            os.path.basename(args.accepted_cp0_seal_file) != "accepted-cp0-seal.json":
        raise RuntimeError("accepted CP0 capability path is not scratch-bound")
    fd = _open_dir(os.path.dirname(args.accepted_cp0_seal_file))
    try:
        obj, _, _, _, record_st = _load_canon_at(
            fd, os.path.basename(args.accepted_cp0_seal_file))
    finally:
        os.close(fd)
    _expect_keys(obj, {"schema", "sealed_run_id", "cp0_seal_sha256"}, "accepted CP0")
    digest = obj["cp0_seal_sha256"]
    if obj["schema"] != "g0-accepted-cp0-seal/v5" or \
            obj["sealed_run_id"] != args.sealed_run_id or not isinstance(digest, str) or \
            len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise RuntimeError("invalid accepted CP0 record")
    if stat.S_IMODE(record_st.st_mode) != 0o444:
        raise RuntimeError("accepted CP0 record mode mismatch")
    if digest != args.accepted_cp0_seal_sha256:
        raise RuntimeError("accepted CP0 record is not bound to the external seal hash")
    return digest


def _validate_terminal_evidence_record(obj, schema, task_id, agent_type, lane_id,
                                       args, state, accepted):
    common = {"schema", "result", "sealed_run_id", "bundle_sha256",
              "accepted_cp0_seal_sha256", "task_id", "agent_type", "lane_id",
              "terminal_state"}
    extras = {
        "g0-copy-seal/v5": {"copy_count", "copies_sha256"},
        "g0-mutating-test-seal/v5": {"test_count"},
        "g0-final-admission-seal/v5": {
            "total_result_count", "verified_count", "unresolved_count"},
    }[schema]
    _expect_keys(obj, common | extras, schema)
    if obj.get("schema") != schema or obj.get("result") != "PASS" or \
            obj.get("sealed_run_id") != args.sealed_run_id or \
            obj.get("bundle_sha256") != state["bundle_sha256"] or \
            obj.get("accepted_cp0_seal_sha256") != accepted or \
            obj.get("task_id") != task_id or obj.get("agent_type") != agent_type or \
            obj.get("lane_id") != lane_id or obj.get("terminal_state") != "complete":
        raise RuntimeError("terminal prerequisite binding mismatch " + schema)
    if schema == "g0-copy-seal/v5" and not _valid_int(obj["copy_count"], 1):
        raise RuntimeError("terminal copy prerequisite is vacuous")
    if schema == "g0-mutating-test-seal/v5" and not _valid_int(obj["test_count"], 1):
        raise RuntimeError("terminal mutating-test prerequisite is vacuous")
    if schema == "g0-final-admission-seal/v5":
        total = obj["total_result_count"]
        if not _valid_int(total, 1) or obj["verified_count"] != total or \
                obj["unresolved_count"] != 0:
            raise RuntimeError("terminal final-admission prerequisite incomplete")


def _terminal_readiness(args, state, accepted):
    if args.mode != "terminal-post":
        return None
    if os.path.dirname(args.terminal_readiness_file) != args.scratch or \
            os.path.basename(args.terminal_readiness_file) != "terminal-readiness.json":
        raise RuntimeError("terminal readiness capability path is not scratch-bound")
    scratch_fd = _open_dir(args.scratch)
    try:
        record, record_bytes, digest, _, readiness_st = _load_canon_at(
            scratch_fd, "terminal-readiness.json")
        if digest != args.terminal_readiness_sha256 or \
                stat.S_IMODE(readiness_st.st_mode) != 0o444:
            raise RuntimeError("terminal readiness hash/mode mismatch")
        keys = {"schema", "readiness_state", "sealed_run_id", "bundle_sha256",
                "accepted_cp0_seal_sha256", "task_id", "agent_type", "lane_id",
                "required_terminal_states", "evidence"}
        _expect_keys(record, keys, "terminal readiness")
        if record != {
                "schema": "g0-terminal-readiness/v5", "readiness_state": "READY",
                "sealed_run_id": args.sealed_run_id,
                "bundle_sha256": state["bundle_sha256"],
                "accepted_cp0_seal_sha256": accepted,
                "task_id": "g0_post", "agent_type": "executor",
                "lane_id": "g0-protection-owner",
                "required_terminal_states": {
                    "copy": "complete", "mutating_tests": "complete",
                    "final_admission": "complete"},
                "evidence": record.get("evidence")}:
            raise RuntimeError("terminal readiness run/task/lane binding mismatch")
        evidence = record["evidence"]
        if not isinstance(evidence, dict) or set(evidence) != TERMINAL_EVIDENCE_NAMES:
            raise RuntimeError("terminal readiness evidence exact-set mismatch")
        evidence_dir_path = os.path.join(args.scratch, "terminal-evidence")
        evidence_fd = _open_dir(evidence_dir_path)
        try:
            if stat.S_IMODE(os.fstat(evidence_fd).st_mode) != 0o555 or \
                    set(os.listdir(evidence_fd)) != TERMINAL_EVIDENCE_NAMES:
                raise RuntimeError("terminal evidence directory closure mismatch")
            payloads = {}
            for name in sorted(TERMINAL_EVIDENCE_NAMES):
                spec = evidence[name]
                _expect_keys(spec, {"path", "sha256", "dev", "inode", "mode", "size"},
                             "terminal evidence reference")
                if spec["path"] != os.path.join(evidence_dir_path, name):
                    raise RuntimeError("terminal evidence path binding mismatch")
                data, actual_digest, size, evidence_st = _read_at(evidence_fd, name)
                actual = {"path": spec["path"], "sha256": actual_digest,
                          "dev": evidence_st.st_dev, "inode": evidence_st.st_ino,
                          "mode": evidence_st.st_mode, "size": size}
                if spec != actual or stat.S_IMODE(evidence_st.st_mode) != 0o444:
                    raise RuntimeError("terminal evidence identity/hash drift")
                payloads[name] = data
        finally:
            os.close(evidence_fd)
    finally:
        os.close(scratch_fd)
    copies = []
    for line in payloads["copies.jsonl"].splitlines(keepends=True):
        try:
            item = json.loads(line)
        except Exception as exc:
            raise RuntimeError("invalid terminal copy ledger") from exc
        if line != canon(item):
            raise RuntimeError("noncanonical terminal copy ledger")
        _expect_keys(item, {"schema", "sealed_run_id", "bundle_sha256",
                            "accepted_cp0_seal_sha256", "source_record_id",
                            "destination_record_id", "byte_count", "sha256", "result"},
                     "terminal copy record")
        if item["schema"] != "g0-copy-record/v5" or item["result"] != "COPIED" or \
                item["sealed_run_id"] != args.sealed_run_id or \
                item["bundle_sha256"] != state["bundle_sha256"] or \
                item["accepted_cp0_seal_sha256"] != accepted or \
                not _valid_int(item["byte_count"], 1) or not re.fullmatch(
                    r"[0-9a-f]{64}", item.get("sha256", "")):
            raise RuntimeError("terminal copy ledger binding mismatch")
        copies.append(item)
    if not copies or len({item["destination_record_id"] for item in copies}) != len(copies):
        raise RuntimeError("terminal copy ledger is empty or duplicated")
    copy_seal = json.loads(payloads["copy-seal.json"])
    if payloads["copy-seal.json"] != canon(copy_seal):
        raise RuntimeError("noncanonical terminal copy seal")
    _validate_terminal_evidence_record(copy_seal, "g0-copy-seal/v5", "g0_copy",
                                       "executor", "copy-owner", args, state, accepted)
    if copy_seal["copy_count"] != len(copies) or \
            copy_seal["copies_sha256"] != sha(payloads["copies.jsonl"]):
        raise RuntimeError("terminal copy seal/ledger mismatch")
    mutating = json.loads(payloads["mutating-test-seal.json"])
    if payloads["mutating-test-seal.json"] != canon(mutating):
        raise RuntimeError("noncanonical mutating-test seal")
    _validate_terminal_evidence_record(mutating, "g0-mutating-test-seal/v5",
                                       "g0_mutating_tests", "test-engineer",
                                       "test-engineer", args, state, accepted)
    admission = json.loads(payloads["final-admission-seal.json"])
    if payloads["final-admission-seal.json"] != canon(admission):
        raise RuntimeError("noncanonical final-admission seal")
    _validate_terminal_evidence_record(admission, "g0-final-admission-seal/v5",
                                       "admission_final", "test-engineer",
                                       "admission-validator", args, state, accepted)
    return {"record": record, "bytes": record_bytes, "sha256": digest,
            "evidence_sha256": {name: sha(data) for name, data in payloads.items()}}


def _parse_event_stream(data, args, bundle_sha256, config, allowed_root_ids=None):
    events = []
    for line in data.splitlines(keepends=True):
        try:
            event = json.loads(line)
        except Exception as exc:
            raise RuntimeError("invalid conductor event stream") from exc
        if line != canon(event):
            raise RuntimeError("noncanonical conductor event stream")
        events.append(event)
    return _validate_event_list(events, args.mode, args.sealed_run_id, bundle_sha256,
                                config["task_id"], config["envelope_id"], allowed_root_ids)


def _input_bytes(state):
    out = {name: state["inputs"][name][1] for name in FINAL_INPUT_ORDER
           if name != "bundle-lock.json"}
    out["bundle-lock.json"] = state["bundle_lock_bytes"]
    return out


def _validate_flat_inputs(publication, state):
    dfd = _open_dir(publication)
    try:
        existing = set(os.listdir(dfd))
        if existing != set(FINAL_INPUT_ORDER):
            if existing & set(PRE_OUTPUT_ORDER):
                raise RuntimeError("same-run retry rejected")
            raise RuntimeError("flat sealed-run input exact-set mismatch")
        for name, expected in _input_bytes(state).items():
            data, _, _, st = _read_at(dfd, name)
            if data != expected or stat.S_IMODE(st.st_mode) != 0o444:
                raise RuntimeError("flat sealed-run immutable input mismatch " + name)
    finally:
        os.close(dfd)


def _finalize_stage(args, state, result, observed_event_bytes):
    scratch_fd = _open_dir(args.scratch)
    try:
        stage_fd = os.open(args.mode, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                           dir_fd=scratch_fd)
        try:
            config = state["inputs"]["manifest-config.json"][0]
            allowed_root_ids = {"paper", "owner", "denial-fixture"} | {
                root["root_id"] for root in state["inputs"]["boundary.json"][0]["source_roots"]}
            observed = _parse_event_stream(observed_event_bytes, args, state["bundle_sha256"],
                                           config, allowed_root_ids)
            suffix = "pre" if args.mode == "pre" else "post"
            access_name = "protected-access-" + suffix + ".jsonl"
            access_bytes = _read_stage(stage_fd, access_name, result["hashes"][access_name])
            if access_bytes != observed_event_bytes or result["event_count"] != len(observed) or \
                    result["event_sha256"] != sha(observed_event_bytes):
                raise RuntimeError("conductor/executor access reconciliation mismatch")
            source_name, paper_name = "source-" + suffix + ".jsonl", "paper-" + suffix + ".json"
            source = _read_stage(stage_fd, source_name, result["hashes"][source_name])
            paper = _read_stage(stage_fd, paper_name, result["hashes"][paper_name])
            if args.mode == "pre":
                provider_stage = _read_stage(stage_fd, "provider-evidence.json",
                                             result["hashes"]["provider-evidence.json"])
                trusted_anchors = build_reconciliation_anchors(
                    state["inputs"]["boundary.json"][1],
                    state["inputs"]["owner-binding.json"][1], provider_stage, source, paper,
                    state["bundle_sha256"], state["bundle"]["bundle_id"],
                    args.sealed_run_id, config["envelope_id"])
                trusted_anchors_sha256 = sha(canon(trusted_anchors))
                pre_validation = None
            else:
                pre_validation = json.loads(state["pre"]["cp0-validation.json"])
                if state["pre"]["cp0-validation.json"] != canon(pre_validation):
                    raise RuntimeError("noncanonical accepted CP0 validation")
                validation_keys = {"schema", "result", "sealed_run_id", "bundle_sha256",
                                   "required_root_ids", "source_record_count",
                                   "conductor_reconciliation", "trusted_anchors",
                                   "trusted_anchors_sha256", "access"}
                _expect_keys(pre_validation, validation_keys, "accepted CP0 validation")
                trusted_anchors = pre_validation["trusted_anchors"]
                trusted_anchors_sha256 = pre_validation["trusted_anchors_sha256"]
                reissued_anchors = build_reconciliation_anchors(
                    state["inputs"]["boundary.json"][1],
                    state["inputs"]["owner-binding.json"][1],
                    state["pre"]["provider-evidence.json"], source, paper,
                    state["bundle_sha256"], state["bundle"]["bundle_id"],
                    args.sealed_run_id, config["envelope_id"])
                if trusted_anchors != reissued_anchors or \
                        trusted_anchors_sha256 != sha(canon(reissued_anchors)):
                    raise RuntimeError("terminal reconciliation anchor authority mismatch")
            phase_closure = closure_for(
                observed, source, paper, state["bundle_sha256"], args.mode,
                args.sealed_run_id, trusted_anchors_sha256)
            verify_closure(phase_closure, observed, source, paper, state["bundle_sha256"],
                           args.mode, args.sealed_run_id, config["task_id"],
                           config["envelope_id"], allowed_root_ids,
                           state["bundle"]["bundle_id"], trusted_anchors,
                           trusted_anchors_sha256,
                           expected_boundary_sha256=sha(
                               state["inputs"]["boundary.json"][1]),
                           expected_owner_binding_sha256=sha(
                               state["inputs"]["owner-binding.json"][1]),
                           expected_provider_evidence_sha256=sha(
                               provider_stage if args.mode == "pre" else
                               state["pre"]["provider-evidence.json"]))
            if args.mode == "pre":
                provider_record = json.loads(provider_stage)
                if {"landlock": provider_record.get("landlock"),
                    "seccomp": provider_record.get("seccomp")} != result["provider_identity"]:
                    raise RuntimeError("pre provider identity serialization mismatch")
                validation = {
                    "schema": "g0-cp0-validation/v5", "result": "PASS",
                    "sealed_run_id": args.sealed_run_id,
                    "bundle_sha256": state["bundle_sha256"],
                    "required_root_ids": ["experiments4", "experiments5", "experiments6"],
                    "source_record_count": len(source.splitlines()),
                    "conductor_reconciliation": "exact_event_stream_match",
                    "trusted_anchors": trusted_anchors,
                    "trusted_anchors_sha256": trusted_anchors_sha256,
                    "access": phase_closure,
                }
                additions = {
                    "cp0-validation.json": canon(validation),
                    "g0-task-evidence.json": canon({
                    "schema": "g0-task-evidence/v5", "result": "PASS",
                    "sealed_run_id": args.sealed_run_id,
                    "bundle_sha256": state["bundle_sha256"],
                    "task_id": "g0_pre", "agent_type": "executor",
                    "lane_id": "g0-protection-owner", "terminal_state": "complete",
                    "source_sha256": sha(source), "paper_record_sha256": sha(paper),
                        "trusted_anchors_sha256": trusted_anchors_sha256,
                        "protected_write_count": phase_closure["protected_write_count"],
                        "unwrapped_event_count": phase_closure["unwrapped_event_count"],
                    }),
                }
                base = _input_bytes(state)
                for name in EXECUTOR_PRE_ORDER:
                    base[name] = _read_stage(stage_fd, name, result["hashes"][name])
                base.update(additions)
                artifacts = {name: sha(data) for name, data in base.items()}
                if set(artifacts) != set(PRE_ORDER) - {"cp0-seal.json"}:
                    raise RuntimeError("CP0 complete artifact set mismatch")
                additions["cp0-seal.json"] = canon({
                    "schema": "g0-cp0-seal/v5", "result": "PASS", "mode": "pre",
                    "sealed_run_id": args.sealed_run_id,
                    "bundle_sha256": state["bundle_sha256"], "artifacts": artifacts,
                    "exact_files": PRE_ORDER,
                })
                for name in ("cp0-validation.json", "g0-task-evidence.json", "cp0-seal.json"):
                    _stage_write(stage_fd, name, additions[name])
                order = PRE_OUTPUT_ORDER
            else:
                pre = state["pre"]
                _reconcile_terminal_provider(pre["provider-evidence.json"],
                                             result["provider_identity"], args, state)
                if pre["source-pre.jsonl"] != source or pre["paper-pre.json"] != paper or \
                        pre["bundle_sha256"] != state["bundle_sha256"]:
                    raise RuntimeError("terminal exact comparison mismatch")
                comparison = {
                    "schema": "g0-comparison/v5", "result": "PASS",
                    "sealed_run_id": args.sealed_run_id,
                    "bundle_sha256": state["bundle_sha256"],
                    "accepted_cp0_seal_sha256": pre["accepted_cp0_seal_sha256"],
                    "terminal_readiness_sha256": state["terminal_readiness"]["sha256"],
                    "trusted_anchors_sha256": trusted_anchors_sha256,
                    "source_bytes_equal": True, "paper_bytes_equal": True,
                    "bundle_equal": True, "provider_identity_equal": True,
                    "terminal_provider_identity_sha256": sha(canon(result["provider_identity"])),
                    "cp0_provider_identity_sha256": sha(canon({
                        "landlock": json.loads(pre["provider-evidence.json"])["landlock"],
                        "seccomp": json.loads(pre["provider-evidence.json"])["seccomp"],
                    })),
                }
                closure = {
                    "schema": "g0-protected-access-closure/v5",
                    "sealed_run_id": args.sealed_run_id,
                    "bundle_sha256": state["bundle_sha256"],
                    "accepted_cp0_seal_sha256": pre["accepted_cp0_seal_sha256"],
                    "terminal_readiness_sha256": state["terminal_readiness"]["sha256"],
                    "trusted_anchors_sha256": trusted_anchors_sha256,
                    "pre": pre_validation["access"], "post": phase_closure,
                    "pre_access_sha256": sha(pre["protected-access-pre.jsonl"]),
                    "post_access_sha256": sha(observed_event_bytes),
                    "source_bytes_equal": True, "paper_bytes_equal": True,
                    "unwrapped_event_count": (pre_validation["access"]["unwrapped_event_count"] +
                                              phase_closure["unwrapped_event_count"]),
                    "protected_write_count": (pre_validation["access"]["protected_write_count"] +
                                               phase_closure["protected_write_count"]),
                    "conductor_reconciliation": "before_after_exact_streams",
                }
                additions = {"comparison.json": canon(comparison),
                             "protected-access-closure.json": canon(closure)}
                base = {name: pre[name] for name in PRE_ORDER}
                for name in EXECUTOR_POST_ORDER:
                    base[name] = _read_stage(stage_fd, name, result["hashes"][name])
                base.update(additions)
                artifacts = {name: sha(data) for name, data in base.items()}
                if set(artifacts) != set(PRE_ORDER + POST_ORDER) - {"terminal-seal.json"}:
                    raise RuntimeError("terminal complete artifact set mismatch")
                additions["terminal-seal.json"] = canon({
                    "schema": "g0-terminal-seal/v5", "result": "PASS",
                    "mode": "terminal-post", "sealed_run_id": args.sealed_run_id,
                    "bundle_sha256": state["bundle_sha256"],
                    "accepted_cp0_seal_sha256": pre["accepted_cp0_seal_sha256"],
                    "terminal_readiness_sha256": state["terminal_readiness"]["sha256"],
                    "trusted_anchors_sha256": trusted_anchors_sha256,
                    "task_id": "g0_post", "agent_type": "executor",
                    "lane_id": "g0-protection-owner", "terminal_state": "complete",
                    "artifacts": artifacts, "exact_files": PRE_ORDER + POST_ORDER,
                })
                for name in ("comparison.json", "protected-access-closure.json",
                             "terminal-seal.json"):
                    _stage_write(stage_fd, name, additions[name])
                order = POST_ORDER
            os.fsync(stage_fd)
            final_hashes = {}
            for name in order:
                if name in additions:
                    final_hashes[name] = sha(additions[name])
                else:
                    final_hashes[name] = result["hashes"][name]
            return {"order": order, "hashes": final_hashes}
        finally:
            os.close(stage_fd)
    finally:
        os.close(scratch_fd)


def run(args, hooks=None):
    verify_startup()
    state = verify_bundle(args)
    try:
        if args.mode == "pre":
            _validate_flat_inputs(args.publication, state)
        accepted = _accepted_cp0(args)
        if args.mode == "terminal-post":
            state["pre"] = _load_pre(args.publication, args.sealed_run_id,
                                     state["bundle_sha256"], accepted)
            state["terminal_readiness"] = _terminal_readiness(args, state, accepted)
        # Local frozen project modules remain unloaded until bundle/runtime,
        # accepted-CP0, and (for terminal) external readiness all authenticate.
        modules = verified_load_project(state)
        result, observed = _child_call(executor, args, state, modules, hooks,
                                       capture_events=True)
        result = _finalize_stage(args, state, result, observed)
        # The publisher is forked only after executor success and inherits neither
        # immutable-input nor frozen-bundle descriptors (and the conductor never
        # opens source-root or paper descriptors).
        os.close(state["frozen_fd"])
        os.close(state["inputs_fd"])
        state["frozen_fd"] = -1
        state["inputs_fd"] = -1
        if args.mode == "pre":
            existing_hashes = {name: sha(data) for name, data in _input_bytes(state).items()}
        else:
            existing_hashes = {name: sha(state["pre"][name]) for name in PRE_ORDER}
        published = _child_call(_publisher_child, args, result, existing_hashes,
                                modules["provider"])
        return {"executor": result, "publisher": published,
                "bundle_sha256": state["bundle_sha256"]}
    finally:
        for key in ("frozen_fd", "inputs_fd"):
            if state[key] >= 0:
                os.close(state[key])


def _fixture_manifest(root):
    records = []
    for base, dirs, files in os.walk(root, topdown=True, followlinks=False):
        dirs.sort()
        files.sort()
        for name in files:
            path = os.path.join(base, name)
            rel = os.path.relpath(path, root)
            with open(path, "rb") as handle:
                data = handle.read()
            st = os.lstat(path)
            records.append({"path": rel, "sha256": sha(data), "size": len(data),
                            "dev": st.st_dev, "inode": st.st_ino, "mode": st.st_mode})
    return records


def _hostile_zero_write_preexec():
    provider = sys.modules.get("provider")
    if provider is None:
        import provider as provider_module
        provider = provider_module
    provider.enforce_landlock([])


def _authenticate_hostile_cp0(publication, accepted_cp0_file):
    """Authenticate an immutable CP0 publication and external acceptance capability."""
    if not isinstance(publication, str) or not os.path.isabs(publication) or \
            not isinstance(accepted_cp0_file, str) or not os.path.isabs(accepted_cp0_file):
        raise RuntimeError("hostile CP0 capabilities require absolute paths")
    publication_fd = _open_dir(publication)
    try:
        if set(os.listdir(publication_fd)) != set(PRE_ORDER):
            raise RuntimeError("hostile CP0 publication exact-set mismatch")
        payloads = {}
        for name in PRE_ORDER:
            data, _, _, record_st = _read_at(publication_fd, name)
            if stat.S_IMODE(record_st.st_mode) != 0o444:
                raise RuntimeError("hostile CP0 immutable mode mismatch")
            payloads[name] = data
    finally:
        os.close(publication_fd)
    bundle_sha256 = sha(payloads["bundle-lock.json"])
    try:
        bundle = json.loads(payloads["bundle-lock.json"])
    except Exception as exc:
        raise RuntimeError("invalid hostile bundle lock") from exc
    if payloads["bundle-lock.json"] != canon(bundle):
        raise RuntimeError("noncanonical hostile bundle lock")
    _expect_keys(bundle, {"schema", "sealed_run_id", "bundle_id", "input_files",
                          "input_sha256", "files_json_sha256", "launcher_sha256",
                          "run_protected_sha256", "python_executable",
                          "python_executable_sha256", "frozen_directory"},
                 "hostile bundle lock")
    sealed_run_id = bundle.get("sealed_run_id")
    if bundle.get("schema") != "g0-bundle-lock/v5" or \
            not isinstance(sealed_run_id, str) or not sealed_run_id:
        raise RuntimeError("hostile bundle run binding")
    expected_inputs = sorted(set(FINAL_INPUT_ORDER) - {"bundle-lock.json"})
    if bundle["input_files"] != expected_inputs or set(bundle["input_sha256"]) != set(
            expected_inputs):
        raise RuntimeError("hostile bundle input closure")
    input_objects = {}
    for name in expected_inputs:
        if sha(payloads[name]) != bundle["input_sha256"][name]:
            raise RuntimeError("hostile bundle input hash drift " + name)
        try:
            obj = json.loads(payloads[name])
        except Exception as exc:
            raise RuntimeError("invalid hostile CP0 input " + name) from exc
        if payloads[name] != canon(obj):
            raise RuntimeError("noncanonical hostile CP0 input " + name)
        input_objects[name] = obj
    for name in ("boundary.json", "owner-binding.json", "manifest-config.json",
                 "argv-pre.json", "argv-post.json", "frozen-scripts.json"):
        obj = input_objects[name]
        if obj.get("sealed_run_id") != sealed_run_id or obj.get("bundle_id") != bundle["bundle_id"]:
            raise RuntimeError("hostile CP0 mutual run/bundle mismatch " + name)
    runtime = input_objects["runtime-lock.json"]
    _validate_runtime_lock_record(runtime)
    if bundle["python_executable"] != runtime["python_executable"] or \
            bundle["python_executable_sha256"] != runtime["python_executable_sha256"] or \
            input_objects["frozen-scripts.json"].get("files_json_sha256") != \
            bundle["files_json_sha256"]:
        raise RuntimeError("hostile bundle runtime/frozen closure mismatch")
    try:
        seal = json.loads(payloads["cp0-seal.json"])
    except Exception as exc:
        raise RuntimeError("invalid hostile CP0 seal") from exc
    if payloads["cp0-seal.json"] != canon(seal):
        raise RuntimeError("noncanonical hostile CP0 seal")
    _expect_keys(seal, {"schema", "result", "mode", "sealed_run_id", "bundle_sha256",
                        "artifacts", "exact_files"}, "hostile CP0 seal")
    if seal.get("schema") != "g0-cp0-seal/v5" or seal.get("result") != "PASS" or \
            seal.get("mode") != "pre" or seal.get("sealed_run_id") != sealed_run_id or \
            seal.get("bundle_sha256") != bundle_sha256 or seal.get("exact_files") != PRE_ORDER or \
            set(seal.get("artifacts", {})) != set(PRE_ORDER) - {"cp0-seal.json"}:
        raise RuntimeError("hostile CP0 seal binding/schema mismatch")
    for name, digest in seal["artifacts"].items():
        if sha(payloads[name]) != digest:
            raise RuntimeError("hostile CP0 sealed artifact drift " + name)
    accepted_fd = _open_dir(os.path.dirname(accepted_cp0_file))
    try:
        accepted, accepted_bytes, _, _, accepted_st = _load_canon_at(
            accepted_fd, os.path.basename(accepted_cp0_file))
    finally:
        os.close(accepted_fd)
    _expect_keys(accepted, {"schema", "sealed_run_id", "cp0_seal_sha256"},
                 "hostile accepted CP0")
    cp0_sha256 = sha(payloads["cp0-seal.json"])
    if accepted_bytes != canon(accepted) or stat.S_IMODE(accepted_st.st_mode) != 0o444 or \
            accepted != {"schema": "g0-accepted-cp0-seal/v5",
                         "sealed_run_id": sealed_run_id,
                         "cp0_seal_sha256": cp0_sha256}:
        raise RuntimeError("hostile accepted CP0 mismatch")
    provider = json.loads(payloads["provider-evidence.json"])
    if payloads["provider-evidence.json"] != canon(provider) or \
            provider.get("schema") != "g0-provider-evidence/v5" or \
            provider.get("sealed_run_id") != sealed_run_id or \
            provider.get("bundle_sha256") != bundle_sha256:
        raise RuntimeError("hostile provider evidence binding")
    config = input_objects["manifest-config.json"]
    events = []
    for line in payloads["protected-access-pre.jsonl"].splitlines(keepends=True):
        try:
            event = json.loads(line)
        except Exception as exc:
            raise RuntimeError("invalid hostile CP0 event stream") from exc
        if line != canon(event):
            raise RuntimeError("noncanonical hostile CP0 event stream")
        events.append(event)
    allowed = {"paper", "owner", "denial-fixture"} | {
        item["root_id"] for item in input_objects["boundary.json"]["source_roots"]}
    validation = json.loads(payloads["cp0-validation.json"])
    if payloads["cp0-validation.json"] != canon(validation):
        raise RuntimeError("noncanonical hostile CP0 validation")
    _expect_keys(validation, {"schema", "result", "sealed_run_id", "bundle_sha256",
                              "required_root_ids", "source_record_count",
                              "conductor_reconciliation", "trusted_anchors",
                              "trusted_anchors_sha256", "access"},
                 "hostile CP0 validation")
    if validation.get("schema") != "g0-cp0-validation/v5" or \
            validation.get("result") != "PASS" or \
            validation.get("sealed_run_id") != sealed_run_id or \
            validation.get("bundle_sha256") != bundle_sha256:
        raise RuntimeError("hostile CP0 validation binding")
    issued_anchors = build_reconciliation_anchors(
        payloads["boundary.json"], payloads["owner-binding.json"],
        payloads["provider-evidence.json"], payloads["source-pre.jsonl"],
        payloads["paper-pre.json"], bundle_sha256, bundle["bundle_id"], sealed_run_id,
        config["envelope_id"])
    if validation["trusted_anchors"] != issued_anchors or \
            validation["trusted_anchors_sha256"] != sha(canon(issued_anchors)):
        raise RuntimeError("hostile CP0 reconciliation anchor mismatch")
    verify_closure(validation["access"], events, payloads["source-pre.jsonl"],
                   payloads["paper-pre.json"], bundle_sha256, "pre", sealed_run_id,
                   config["task_id"], config["envelope_id"], allowed,
                   bundle["bundle_id"], validation["trusted_anchors"],
                   validation["trusted_anchors_sha256"],
                   expected_boundary_sha256=sha(payloads["boundary.json"]),
                   expected_owner_binding_sha256=sha(payloads["owner-binding.json"]),
                   expected_provider_evidence_sha256=sha(
                       payloads["provider-evidence.json"]))
    task = json.loads(payloads["g0-task-evidence.json"])
    task_keys = {"schema", "result", "sealed_run_id", "bundle_sha256", "task_id",
                 "agent_type", "lane_id", "terminal_state", "source_sha256",
                 "paper_record_sha256", "trusted_anchors_sha256",
                 "protected_write_count", "unwrapped_event_count"}
    _expect_keys(task, task_keys, "hostile G0 task evidence")
    if task != {"schema": "g0-task-evidence/v5", "result": "PASS",
                "sealed_run_id": sealed_run_id, "bundle_sha256": bundle_sha256,
                "task_id": "g0_pre", "agent_type": "executor",
                "lane_id": "g0-protection-owner", "terminal_state": "complete",
                "source_sha256": sha(payloads["source-pre.jsonl"]),
                "paper_record_sha256": sha(payloads["paper-pre.json"]),
                "trusted_anchors_sha256": validation["trusted_anchors_sha256"],
                "protected_write_count": validation["access"]["protected_write_count"],
                "unwrapped_event_count": validation["access"]["unwrapped_event_count"]}:
        raise RuntimeError("hostile G0 task evidence binding")
    return {"sealed_run_id": sealed_run_id, "bundle_sha256": bundle_sha256,
            "accepted_cp0_seal_sha256": cp0_sha256}


def supervise_hostile_rg(fixture_root, rg_path, rg_sha256, output_dir,
                         publication=None, accepted_cp0_file=None, locked_argv=None):
    """Independent disposable-fixture demonstration; never eligible for sealing."""
    if not all(isinstance(path, str) and os.path.isabs(path)
               for path in (fixture_root, rg_path, output_dir)):
        raise RuntimeError("hostile supervisor requires absolute paths")
    authenticated = _authenticate_hostile_cp0(publication, accepted_cp0_file)
    sealed_run_id = authenticated["sealed_run_id"]
    bundle_sha256 = authenticated["bundle_sha256"]
    accepted_cp0_seal_sha256 = authenticated["accepted_cp0_seal_sha256"]
    expected_argv = [rg_path, "--files", "--hidden", "--no-ignore", fixture_root]
    if locked_argv != expected_argv:
        raise RuntimeError("hostile argv binding mismatch")
    rg_fd = os.open(rg_path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        _, actual_rg_sha, _, rg_st = _stable_read(rg_fd)
    finally:
        os.close(rg_fd)
    if not stat.S_ISREG(rg_st.st_mode) or actual_rg_sha != rg_sha256:
        raise RuntimeError("rg executable lock mismatch")
    before = _fixture_manifest(fixture_root)
    argv = list(locked_argv)
    started = time.monotonic_ns()
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C"},
                            close_fds=True, preexec_fn=_hostile_zero_write_preexec)
    child_pid = proc.pid
    stdout, stderr = proc.communicate()
    ended = time.monotonic_ns()
    after = _fixture_manifest(fixture_root)
    if before != after:
        raise RuntimeError("hostile fixture changed")
    observed = sorted(line.decode("utf-8", "strict") for line in stdout.splitlines())
    expected = sorted(os.path.join(fixture_root, item["path"]) for item in before)
    if proc.returncode != 0 or observed != expected:
        raise RuntimeError("hostile stdout/fixture reconciliation failed")
    downstream_decision = "rejected_invalidated_run"
    record = {
        "schema": "g0-supervisor-event/v5", "kind": "violation",
        "phase": "hostile-fixture", "sealed_run_id": sealed_run_id,
        "bundle_sha256": bundle_sha256,
        "accepted_cp0_seal_sha256": accepted_cp0_seal_sha256,
        "eligible_for_sealing": False,
        "executable": rg_path, "executable_sha256": rg_sha256, "command": argv,
        "argv": argv, "argv_sha256": sha(canon(argv)), "zero_write_landlock": True,
        "child_pid": child_pid, "started_monotonic_ns": started,
        "ended_monotonic_ns": ended, "status": proc.returncode,
        "stdout_sha256": sha(stdout), "stderr_sha256": sha(stderr),
        "stdout_b64": base64.b64encode(stdout).decode("ascii"),
        "stderr_b64": base64.b64encode(stderr).decode("ascii"),
        "pre_manifest": before, "post_manifest": after,
        "pre_manifest_sha256": sha(canon(before)), "post_manifest_sha256": sha(canon(after)),
        "observed_path_count": len(observed), "wrapper_event_count": 0,
        "reconciliation": "stdout_exact_fixture_match",
        "violation": "unwrapped_protected_enumeration", "result": "invalidated",
        "claims": {"global_read_denial": False, "syscall_tracing": False},
    }
    invalidation = {
        "schema": "g0-hostile-invalidation/v5", "kind": "invalidation",
        "result": "invalidated", "phase": "hostile-fixture",
        "sealed_run_id": sealed_run_id, "bundle_sha256": bundle_sha256,
        "accepted_cp0_seal_sha256": accepted_cp0_seal_sha256,
        "reason": "unwrapped_protected_enumeration", "operation": "descendant_enumeration",
        "content_reads": 0, "writes": 0,
        "violated_gates": ["G0-ORD-01", "I-G0-01"],
        "eligible_for_sealing": False, "decision": downstream_decision,
        "downstream_acceptance": "rejected", "supervision_sha256": sha(canon(record)),
    }
    outfd = _open_dir(output_dir)
    try:
        if os.listdir(outfd):
            raise RuntimeError("hostile output directory not empty")
        for name, data in (("hostile-supervision.json", canon(record)),
                           ("hostile-invalidation.json", canon(invalidation))):
            publish_at(outfd, name, data, 0o444)
        if any("seal" in name for name in os.listdir(outfd)):
            raise RuntimeError("hostile output must not contain a seal")
    finally:
        os.close(outfd)
    return record, invalidation


def parse(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--sealed-run-id", required=True)
    parser.add_argument("--mode", required=True, choices=("pre", "terminal-post"))
    parser.add_argument("--frozen", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--scratch", required=True)
    parser.add_argument("--publication", required=True)
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--launcher-sha256", required=True)
    parser.add_argument("--run-protected-sha256", required=True)
    parser.add_argument("--accepted-cp0-seal-file", required=True)
    parser.add_argument("--accepted-cp0-seal-sha256", required=True)
    parser.add_argument("--terminal-readiness-file", required=True)
    parser.add_argument("--terminal-readiness-sha256", required=True)
    args = parser.parse_args(argv)
    digest = args.accepted_cp0_seal_sha256
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        parser.error("--accepted-cp0-seal-sha256 must be 64 lowercase hex characters")
    if args.mode == "pre" and digest != "0" * 64:
        parser.error("pre mode requires the null accepted-CP0 hash anchor")
    if args.mode == "terminal-post" and digest == "0" * 64:
        parser.error("terminal-post requires a non-null accepted-CP0 hash anchor")
    readiness_digest = args.terminal_readiness_sha256
    if len(readiness_digest) != 64 or any(
            character not in "0123456789abcdef" for character in readiness_digest):
        parser.error("--terminal-readiness-sha256 must be 64 lowercase hex characters")
    if args.mode == "pre" and readiness_digest != "0" * 64:
        parser.error("pre mode requires the null terminal-readiness hash anchor")
    if args.mode == "terminal-post" and readiness_digest == "0" * 64:
        parser.error("terminal-post requires a non-null terminal-readiness hash anchor")
    return args


def main():
    result = run(parse())
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
