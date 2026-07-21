#!/usr/bin/env python3
"""Create the immutable manifest/config/argv/runtime bundle before protected reads."""
from __future__ import annotations

import argparse
import hashlib
import json
import locale
import os
import platform
import stat
import sys
from datetime import datetime, timezone

import manifest_tool
import provider

HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN_DIR = os.path.join(HERE, "frozen")
TARGET = "/data/minseo/experiments7"
PAPER = "/data/minseo/experiments7/_paper/8. Latent_Preference_Modeling_for_Cross_Session_Personalized_Tool_Calling.pdf"
SOURCES = (
    ("experiments4", "/data/minseo/experiments4"),
    ("experiments5", "/data/minseo/experiments5"),
    ("experiments6", "/data/minseo/experiments6"),
)
ENVIRONMENT_KEYS = (
    "HOME", "TMPDIR", "XDG_CACHE_HOME", "PYTHONPYCACHEPREFIX",
    "PYTHONDONTWRITEBYTECODE", "LC_ALL", "LANG", "TZ", "PATH",
    "SOURCE_DATE_EPOCH",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def hash_file(path: str) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    with open(path, "rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
            total += len(block)
    return digest.hexdigest(), total


def binding(path: str, expected_type: str) -> dict[str, object]:
    if os.path.realpath(path) != path:
        raise RuntimeError(f"non-canonical binding: {path}")
    st = os.lstat(path)
    actual = "directory" if stat.S_ISDIR(st.st_mode) else "regular" if stat.S_ISREG(st.st_mode) else "other"
    if actual != expected_type:
        raise RuntimeError(f"unexpected binding type: {path}: {actual}")
    return {"dev": st.st_dev, "inode": st.st_ino, "type": actual, "mode": stat.S_IMODE(st.st_mode)}


def load_owner() -> dict[str, object]:
    path = os.path.join(TARGET, ".experiments7-owner.json")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        with os.fdopen(fd, "rb") as handle:
            owner = json.load(handle)
    except Exception:
        try: os.close(fd)
        except OSError: pass
        raise
    if owner.get("schema") != "experiments7-owner/v1" or owner.get("state") != "complete":
        raise RuntimeError("invalid owner record")
    return owner


def current_umask() -> int:
    value = os.umask(0)
    os.umask(value)
    return value


def component(path: str) -> dict[str, object]:
    digest, size = hash_file(path)
    return {"path": os.path.abspath(path), "sha256": digest, "size": size}


def publish(path: str, value: object) -> dict[str, object]:
    return manifest_tool.atomic_publish(path, [canonical(value)])


def freeze(sealed_run_id: str) -> dict[str, object]:
    owner = load_owner()
    if owner.get("sealed_run_id") != sealed_run_id:
        raise RuntimeError("sealed_run_id differs from owner bootstrap")
    os.mkdir(FROZEN_DIR, 0o755)
    owner_path = os.path.join(TARGET, ".experiments7-owner.json")
    readme_path = os.path.join(TARGET, "README.md")
    readme_hash, readme_size = hash_file(readme_path)
    if readme_hash != owner["root_readme"]["sha256"] or readme_size != owner["root_readme"]["size"]:
        raise RuntimeError("root README bytes differ from owner record")
    managed_prefixes = []
    for name in owner["managed_prefixes"]:
        path = os.path.join(TARGET, name)
        managed_prefixes.append({"name": name, "path": path, "binding": binding(path, "directory")})
    config = {
        "schema": "experiments7-manifest-config/v1",
        "sealed_run_id": sealed_run_id,
        "created_at": now(),
        "protected_sources": [
            {"root_id": root_id, "path": path, "binding": binding(path, "directory")}
            for root_id, path in SOURCES
        ],
        "target": {"path": TARGET, "binding": binding(TARGET, "directory")},
        "owner_record": {"path": owner_path, "binding": binding(owner_path, "regular")},
        "root_readme": {"path": readme_path, "binding": binding(readme_path, "regular"), "sha256": readme_hash, "size": readme_size},
        "managed_prefixes": managed_prefixes,
        "paper": {
            "path": PAPER,
            "parent": os.path.dirname(PAPER),
            "binding": binding(PAPER, "regular"),
            "parent_binding": binding(os.path.dirname(PAPER), "directory"),
            "ownership": "preserved_foreign_readonly",
        },
        "metadata_scope": ["type", "mode", "uid", "gid", "nlink", "size", "mtime_ns", "ctime_ns", "dev", "inode"],
        "excluded_metadata": ["atime_ns", "extended_attributes", "acl", "filesystem_generation"],
        "ordering": "root_id_utf8_then_raw_relative_path_bytes",
        "path_encoding": "base64_of_os_path_bytes",
        "serialization": "utf8_canonical_jsonl_sort_keys_compact_lf",
        "hash_algorithm": "sha256",
        "symlink_policy": "lstat_and_readlink_bytes_no_traversal",
        "writable_during_g0": [os.path.join(TARGET, "manifests"), "/tmp"],
        "outputs": {
            "build-pre": {
                "source": os.path.join(TARGET, "manifests", "source-pre.jsonl"),
                "paper": os.path.join(TARGET, "manifests", "paper-pre.json"),
            },
            "build-post": {
                "source": os.path.join(TARGET, "manifests", "source-post.jsonl"),
                "paper": os.path.join(TARGET, "manifests", "paper-post.json"),
            },
        },
    }
    wrapper = os.path.join(HERE, "run_protected.py")
    pre_args = [
        "--verify-envelope", "--task-id", "/root/executor_g0_cp0",
        "--audit-output", os.path.join(TARGET, "manifests", "envelope-audit-pre.json"),
        "--", "build-pre",
    ]
    post_args = [
        "--verify-envelope", "--task-id", "g0_post",
        "--audit-output", os.path.join(TARGET, "manifests", "envelope-audit-post.json"),
        "--require-cp0-bundle", "--", "build-post",
    ]
    argv_lock = {
        "schema": "experiments7-manifest-argv/v1",
        "commands": {
            "build-pre": {"task_id": "/root/executor_g0_cp0", "executable": os.path.realpath(sys.executable), "script": wrapper, "wrapper_args": pre_args, "canonical_argv_sha256": hashlib.sha256(canonical([os.path.realpath(sys.executable), wrapper, *pre_args])).hexdigest()},
            "build-post": {"task_id": "g0_post", "executable": os.path.realpath(sys.executable), "script": wrapper, "wrapper_args": post_args, "canonical_argv_sha256": hashlib.sha256(canonical([os.path.realpath(sys.executable), wrapper, *post_args])).hexdigest()},
        },
    }
    python_path = os.path.realpath(sys.executable)
    python_hash, python_size = hash_file(python_path)
    runtime = {
        "schema": "experiments7-manifest-runtime/v1",
        "python": {
            "realpath": python_path,
            "sha256": python_hash,
            "size": python_size,
            "version": platform.python_version(),
            "build": list(platform.python_build()),
            "implementation": platform.python_implementation(),
        },
        "platform": platform.platform(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "locale": {"preferred_encoding": locale.getpreferredencoding(False), "filesystem_encoding": sys.getfilesystemencoding()},
        "umask": current_umask(),
        "environment": {key: os.environ.get(key) for key in ENVIRONMENT_KEYS},
        "provider": provider.provider_identity(),
    }
    config_path = os.path.join(FROZEN_DIR, "manifest-config.json")
    argv_path = os.path.join(FROZEN_DIR, "manifest-argv.json")
    runtime_path = os.path.join(FROZEN_DIR, "runtime-lock.json")
    publish(config_path, config)
    publish(argv_path, argv_lock)
    publish(runtime_path, runtime)
    components = {
        "preflight_target": component(os.path.join(HERE, "preflight_target.py")),
        "provider": component(os.path.join(HERE, "provider.py")),
        "provider_verifier": component(os.path.join(HERE, "verify_provider.py")),
        "g0_test_suite": component(os.path.join(HERE, "test_g0.py")),
        "manifest_executable": component(os.path.join(HERE, "manifest_tool.py")),
        "comparison_executable": component(os.path.join(HERE, "compare_manifests.py")),
        "wrapper": component(wrapper),
        "bundle_freezer": component(os.path.join(HERE, "freeze_manifest_bundle.py")),
        "config": component(config_path),
        "argv": component(argv_path),
        "runtime": component(runtime_path),
    }
    lock = {
        "schema": "experiments7-manifest-bundle-lock/v1",
        "sealed_run_id": sealed_run_id,
        "created_at": now(),
        "components": components,
        "config_path": config_path,
        "argv_path": argv_path,
        "runtime_path": runtime_path,
        "component_order": sorted(components),
        "aggregate_components_sha256": hashlib.sha256(canonical({name: components[name]["sha256"] for name in sorted(components)})).hexdigest(),
    }
    lock_path = os.path.join(FROZEN_DIR, "manifest-bundle-lock.json")
    lock_publish = publish(lock_path, lock)
    # Convenience mirrors are evidence only; the wrapper trusts only frozen/.
    manifests = os.path.join(TARGET, "manifests")
    mirrors = {}
    for filename, value in (
        ("manifest-config.json", config),
        ("manifest-argv.json", argv_lock),
        ("runtime-lock.json", runtime),
        ("manifest-bundle-lock.json", lock),
    ):
        mirrors[filename] = publish(os.path.join(manifests, filename), value)
    return {"status": "PASS", "frozen_dir": FROZEN_DIR, "bundle_lock": lock_publish, "components": components, "mirrors": mirrors}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sealed-run-id", required=True)
    args = parser.parse_args()
    try:
        result = freeze(args.sealed_run_id)
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "reason": repr(exc)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(2)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
