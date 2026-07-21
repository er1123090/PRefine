#!/usr/bin/env python3
"""Fail-closed 30-profile production-dispatch gate for V6 CP2."""

from __future__ import annotations

import argparse
import os
from pathlib import Path, PurePosixPath
import stat
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facade.external import _require_effective_subject_immutable  # noqa: E402
from facade.registry import (  # noqa: E402
    FacadeError,
    HEX64,
    SAFE_ID,
    load_bundle,
    load_json,
    require,
)
from facade.runner import run_external  # noqa: E402
from facade.selection import canonical_bytes, sha256_file  # noqa: E402


INDEX_SCHEMA = "experiments7-v6-profile-dispatch-index/v1"


def _safe_index(value: str) -> Path:
    require(isinstance(value, str) and value and "\x00" not in value,
            "INVALID_PROFILE_DISPATCH_INDEX")
    relative = PurePosixPath(value)
    require(
        not relative.is_absolute()
        and all(part not in {"", ".", ".."} for part in relative.parts),
        "INVALID_PROFILE_DISPATCH_INDEX",
    )
    declared = ROOT / relative.as_posix()
    resolved = declared.resolve(strict=True)
    require(declared == resolved and not declared.is_symlink(),
            "UNSAFE_PROFILE_DISPATCH_INDEX")
    state = declared.lstat()
    require(stat.S_ISREG(state.st_mode) and state.st_mode & 0o222 == 0,
            "PROFILE_DISPATCH_INDEX_NOT_IMMUTABLE")
    cursor = ROOT
    _require_effective_subject_immutable(
        cursor, code="PROFILE_DISPATCH_INDEX_ANCESTOR_NOT_EFFECTIVELY_IMMUTABLE"
    )
    for part in relative.parts[:-1]:
        cursor = cursor / part
        _require_effective_subject_immutable(
            cursor, code="PROFILE_DISPATCH_INDEX_ANCESTOR_NOT_EFFECTIVELY_IMMUTABLE"
        )
    _require_effective_subject_immutable(
        declared, code="PROFILE_DISPATCH_INDEX_NOT_EFFECTIVELY_IMMUTABLE"
    )
    return declared


def _validate_index(path: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    bundle = load_bundle(ROOT)
    value = load_json(path)
    require(set(value) == {
        "schema",
        "publication_id",
        "publication_manifest_sha256",
        "runtime_id",
        "runtime_manifest_sha256",
        "profile_count",
        "profiles",
    }, "PROFILE_DISPATCH_INDEX_FIELDS")
    require(value.get("schema") == INDEX_SCHEMA, "PROFILE_DISPATCH_INDEX_SCHEMA")
    publication_id = value.get("publication_id")
    runtime_id = value.get("runtime_id")
    require(SAFE_ID.fullmatch(str(publication_id)) is not None,
            "INVALID_PROFILE_DISPATCH_PUBLICATION")
    require(SAFE_ID.fullmatch(str(runtime_id)) is not None,
            "INVALID_PROFILE_DISPATCH_RUNTIME")
    require(
        HEX64.fullmatch(str(value.get("publication_manifest_sha256"))) is not None,
        "INVALID_PROFILE_DISPATCH_PUBLICATION_HASH",
    )
    require(
        HEX64.fullmatch(str(value.get("runtime_manifest_sha256"))) is not None,
        "INVALID_PROFILE_DISPATCH_RUNTIME_HASH",
    )
    rows = value.get("profiles")
    require(isinstance(rows, list) and value.get("profile_count") == len(rows),
            "PROFILE_DISPATCH_COUNT")
    require(len(rows) == 30, "PROFILE_DISPATCH_COUNT", len(rows or []))
    require(all(isinstance(row, dict) and set(row) == {
        "profile_id",
        "run_config_id",
        "run_config_sha256",
        "output_dir",
        "dispatch_path",
        "golden_mode",
    } for row in rows), "PROFILE_DISPATCH_ROW_FIELDS")
    require(
        [row["profile_id"] for row in rows]
        == sorted(bundle.profiles_by_id),
        "PROFILE_DISPATCH_PROFILE_SET",
    )
    output_dirs: set[str] = set()
    normalized: list[dict[str, str]] = []
    for row in rows:
        require(row["dispatch_path"] == "production_external_contract",
                "PROFILE_DISPATCH_SHORTCUT_FORBIDDEN", row["profile_id"])
        require(row["golden_mode"] == "exact_output_hashes",
                "PROFILE_DISPATCH_GOLDEN_REQUIRED", row["profile_id"])
        require(SAFE_ID.fullmatch(str(row["run_config_id"])) is not None,
                "INVALID_RUN_CONFIG_ID", row["profile_id"])
        require(
            HEX64.fullmatch(str(row["run_config_sha256"])) is not None,
            "INVALID_RUN_CONFIG_HASH",
            row["profile_id"],
        )
        output = PurePosixPath(str(row["output_dir"]))
        require(
            not output.is_absolute()
            and len(output.parts) >= 3
            and output.parts[:2] == ("runs", "v6-hermetic")
            and all(part not in {"", ".", ".."} for part in output.parts),
            "INVALID_PROFILE_DISPATCH_OUTPUT",
            row["profile_id"],
        )
        rendered = output.as_posix()
        require(rendered not in output_dirs, "DUPLICATE_PROFILE_DISPATCH_OUTPUT", rendered)
        output_dirs.add(rendered)
        normalized.append({key: str(row[key]) for key in row})
    return {
        "publication_id": str(publication_id),
        "publication_manifest_sha256": str(value["publication_manifest_sha256"]),
        "runtime_id": str(runtime_id),
        "runtime_manifest_sha256": str(value["runtime_manifest_sha256"]),
        "index_sha256": sha256_file(path)[0],
        "profile_count": len(rows),
    }, normalized


def validate_all(index_path: Path) -> dict[str, Any]:
    header, rows = _validate_index(index_path)
    bundle = load_bundle(ROOT)
    receipts: list[dict[str, Any]] = []
    for row in rows:
        run_config_path = ROOT / "configs/external-runs" / f"{row['run_config_id']}.json"
        require(
            run_config_path.is_file() and not run_config_path.is_symlink(),
            "MISSING_EXTERNAL_RUN_CONFIG",
            row["run_config_id"],
        )
        require(
            sha256_file(run_config_path)[0] == row["run_config_sha256"],
            "PROFILE_DISPATCH_RUN_CONFIG_HASH_MISMATCH",
            row["profile_id"],
        )
        result = run_external(
            bundle,
            row["profile_id"],
            row["output_dir"],
            allow_external=True,
            publication_id=header["publication_id"],
            runtime_id=header["runtime_id"],
            run_config_id=row["run_config_id"],
        )
        external = result["external_result"]
        require(external["state"] == "PASS", "PROFILE_DISPATCH_NOT_PASS", row["profile_id"])
        require(external["bounded_runnable_claim"].startswith("local hermetic"),
                "PROFILE_DISPATCH_CLAIM_OVERBROAD", row["profile_id"])
        require(
            external["contract"]["run_config_sha256"] == row["run_config_sha256"]
            and external["contract"]["runtime"]["manifest_sha256"]
            == header["runtime_manifest_sha256"]
            and external["contract"]["publication"]["manifest_sha256"]
            == header["publication_manifest_sha256"],
            "PROFILE_DISPATCH_CLOSURE_HASH_MISMATCH",
            row["profile_id"],
        )
        receipts.append({
            "profile_id": row["profile_id"],
            "selection_sha256": result["selection"]["selection_sha256"],
            "run_config_sha256": external["contract"]["run_config_sha256"],
            "runtime_manifest_sha256": external["contract"]["runtime"]["manifest_sha256"],
            "publication_manifest_sha256": external["contract"]["publication"]["manifest_sha256"],
            "step_count": len(external["steps"]),
            "dispatch_path": row["dispatch_path"],
            "golden_mode": row["golden_mode"],
            "state": "PASS",
        })
    require(len(receipts) == 30, "PROFILE_DISPATCH_COUNT", len(receipts))
    return {
        "schema": "experiments7-v6-profile-dispatch-validation/v1",
        "state": "PASS",
        "claim": "30/30 local hermetic production-dispatch replay; no live inference claim",
        "profile_count": 30,
        "passed_profile_count": 30,
        "index_sha256": header["index_sha256"],
        "profiles": receipts,
        "protected_reads": 0,
        "protected_writes": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index")
    parser.add_argument("--json", action="store_true", required=True)
    args = parser.parse_args(argv)
    try:
        if not args.index:
            raise FacadeError("MISSING_PROFILE_DISPATCH_INDEX")
        report = validate_all(_safe_index(args.index))
        returncode = 0
    except (FacadeError, FileNotFoundError) as exc:
        reason = exc.code if isinstance(exc, FacadeError) else "MISSING_PROFILE_DISPATCH_INDEX"
        report = {
            "schema": "experiments7-v6-profile-dispatch-validation/v1",
            "state": "BLOCKED",
            "reason_codes": [reason],
            "profile_count": 30,
            "passed_profile_count": 0,
            "claim": "no 30/30 runnable claim published",
            "protected_reads": 0,
            "protected_writes": 0,
        }
        returncode = 2
    sys.stdout.buffer.write(canonical_bytes(report))
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
