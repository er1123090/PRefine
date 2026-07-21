#!/usr/bin/env python3
"""Capture one readonly CP1 paper extraction into external no-replace evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from _common import preflight_existing_external_output, print_summary
from strict_run import canonical_json
from strict_run.canonical import fail, require_sha256


EXTRACTION_SCHEMA = "experiments7-cp1-protected-paper-extraction/v6"
EXECUTION_SCHEMA = "experiments7-cp1-paper-extraction-execution/v6"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load_canonical_object(raw: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is not JSON") from exc
    if not isinstance(value, dict) or canonical_json(value) != raw:
        raise RuntimeError(f"{label} is not canonical JSON")
    return value


def _exact_extraction(
    value: dict[str, object],
    *,
    sealed_run_id: str,
    run_root: str,
    expected_paper_sha256: str,
) -> None:
    required = {
        "schema",
        "sealed_run_id",
        "run_root",
        "paper",
        "provider",
        "paper_write_probe",
        "parser",
        "page_count",
        "pages",
    }
    if set(value) != required or value["schema"] != EXTRACTION_SCHEMA:
        raise RuntimeError("protected extraction schema differs")
    if value["sealed_run_id"] != sealed_run_id or value["run_root"] != run_root:
        raise RuntimeError("protected extraction run binding differs")
    paper = value["paper"]
    if not isinstance(paper, dict) or paper.get("sha256") != expected_paper_sha256:
        raise RuntimeError("protected extraction paper binding differs")
    require_sha256(paper.get("sha256"), "extracted paper sha256")
    if type(paper.get("bytes")) is not int or paper["bytes"] <= 0:
        raise RuntimeError("protected extraction paper bytes are invalid")
    probe = value["paper_write_probe"]
    if not isinstance(probe, dict) or probe.get("denied") is not True:
        raise RuntimeError("protected extraction write denial is absent")
    provider = value["provider"]
    if not isinstance(provider, dict) or provider.get("writable_directories") != []:
        raise RuntimeError("protected extraction provider is not readonly")
    parser = value["parser"]
    if (
        not isinstance(parser, dict)
        or parser.get("module") != "pypdf"
        or not isinstance(parser.get("path"), str)
    ):
        raise RuntimeError("protected extraction parser identity is invalid")
    require_sha256(parser.get("sha256"), "parser sha256")
    page_count = value["page_count"]
    pages = value["pages"]
    if type(page_count) is not int or page_count <= 0 or not isinstance(pages, list):
        raise RuntimeError("protected extraction pages are invalid")
    if len(pages) != page_count:
        raise RuntimeError("protected extraction page count differs")
    for index, page in enumerate(pages, start=1):
        if (
            not isinstance(page, dict)
            or set(page) != {"page", "layout_text", "layout_text_sha256"}
            or page.get("page") != index
            or not isinstance(page.get("layout_text"), str)
        ):
            raise RuntimeError("protected extraction page record differs")
        layout = page["layout_text"].encode("utf-8")
        if page.get("layout_text_sha256") != hashlib.sha256(layout).hexdigest():
            raise RuntimeError("protected extraction page text digest differs")


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--strict-parent", required=True)
    parser.add_argument("--sealed-run-id", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--external-extraction", required=True)
    parser.add_argument("--external-execution", required=True)
    parser.add_argument("--extractor", required=True)
    parser.add_argument("--paper-path", required=True)
    parser.add_argument("--expected-paper-sha256", required=True)
    parser.add_argument("--provider-path", required=True)
    parser.add_argument("--provider-sha256", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    args = parser.parse_args()
    if args.external_extraction == args.external_execution:
        fail("EXTERNAL_OUTPUT_PATH_COLLISION", "CP1 extraction outputs must be distinct")
    if type(args.timeout_seconds) is not int or args.timeout_seconds <= 0:
        fail("CP1_EXTRACTION_TIMEOUT_INVALID", "timeout must be a positive integer")
    expected_paper_sha256 = require_sha256(
        args.expected_paper_sha256, "expected paper sha256"
    )
    provider_sha256 = require_sha256(args.provider_sha256, "provider sha256")
    extractor = Path(args.extractor)
    provider = Path(args.provider_path)
    if not extractor.is_file() or not provider.is_file():
        fail("CP1_EXTRACTION_INPUT_INVALID", "extractor and provider must be regular files")
    argv = [
        sys.executable,
        "-B",
        str(extractor),
        "--sealed-run-id",
        args.sealed_run_id,
        "--run-root",
        args.run_root,
        "--paper-path",
        args.paper_path,
        "--expected-paper-sha256",
        expected_paper_sha256,
        "--provider-path",
        str(provider),
        "--provider-sha256",
        provider_sha256,
    ]
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", "/tmp"),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
    }
    with (
        preflight_existing_external_output(
            args.external_extraction,
            strict_parent=args.strict_parent,
            sealed_run_id=args.sealed_run_id,
            run_root=args.run_root,
        ) as extraction_output,
        preflight_existing_external_output(
            args.external_execution,
            strict_parent=args.strict_parent,
            sealed_run_id=args.sealed_run_id,
            run_root=args.run_root,
        ) as execution_output,
    ):
        completed = subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            check=False,
            timeout=args.timeout_seconds,
        )
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", "replace").strip()
            raise RuntimeError(f"CP1 protected extraction failed: {detail}")
        extraction = _load_canonical_object(completed.stdout, "CP1 protected extraction")
        _exact_extraction(
            extraction,
            sealed_run_id=args.sealed_run_id,
            run_root=args.run_root,
            expected_paper_sha256=expected_paper_sha256,
        )
        extraction_output.publish_json(extraction)
        execution = {
            "schema": EXECUTION_SCHEMA,
            "sealed_run_id": args.sealed_run_id,
            "run_root": args.run_root,
            "argv": argv,
            "environment": {
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONHASHSEED": "0",
                "child_invocation": "python -B",
            },
            "extractor": {"path": str(extractor), "sha256": _sha256_file(extractor)},
            "provider": {"path": str(provider), "sha256": provider_sha256},
            "expected_paper_sha256": expected_paper_sha256,
            "child_stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
            "child_stdout_bytes": len(completed.stdout),
            "extraction_sha256": hashlib.sha256(canonical_json(extraction)).hexdigest(),
            "page_count": extraction["page_count"],
        }
        execution_output.publish_json(execution)
    print_summary(
        {
            "sealed_run_id": args.sealed_run_id,
            "run_root": args.run_root,
            "page_count": extraction["page_count"],
            "extraction_sha256": execution["extraction_sha256"],
            "external_extraction": args.external_extraction,
            "external_execution": args.external_execution,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
