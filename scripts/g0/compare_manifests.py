#!/usr/bin/env python3
"""Strict same-bundle comparison for canonical source and paper manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: str) -> tuple[list[dict[str, object]], bytes]:
    with open(path, "rb") as handle:
        raw = handle.read()
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    for number, line in enumerate(raw.splitlines(), 1):
        if not line:
            raise RuntimeError(f"empty JSONL line: {path}:{number}")
        record = json.loads(line)
        record_id = record.get("record_id")
        if not isinstance(record_id, str) or record_id in seen:
            raise RuntimeError(f"missing/duplicate record_id: {path}:{number}")
        seen.add(record_id)
        records.append(record)
    return records, raw


def read_json(path: str) -> tuple[dict[str, object], bytes]:
    with open(path, "rb") as handle:
        raw = handle.read()
    return json.loads(raw), raw


def compare(
    source_pre: str,
    source_post: str,
    paper_pre: str,
    paper_post: str,
    expected_bundle_lock: str,
) -> dict[str, object]:
    pre_records, pre_raw = read_jsonl(source_pre)
    post_records, post_raw = read_jsonl(source_post)
    pre_paper, pre_paper_raw = read_json(paper_pre)
    post_paper, post_paper_raw = read_json(paper_post)
    failures: list[str] = []
    if pre_raw != post_raw:
        failures.append("source_manifest_bytes_differ")
    if pre_records != post_records:
        failures.append("source_records_differ")
    if pre_paper != post_paper:
        failures.append("paper_baseline_differs")
    for label, paper in (("pre", pre_paper), ("post", post_paper)):
        if paper.get("bundle_lock_sha256") != expected_bundle_lock:
            failures.append(f"{label}_paper_bundle_lock_mismatch")
    if pre_paper_raw != post_paper_raw:
        failures.append("paper_manifest_bytes_differ")
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "source_record_count": len(pre_records),
        "source_pre_sha256": hashlib.sha256(pre_raw).hexdigest(),
        "source_post_sha256": hashlib.sha256(post_raw).hexdigest(),
        "paper_pre_manifest_sha256": hashlib.sha256(pre_paper_raw).hexdigest(),
        "paper_post_manifest_sha256": hashlib.sha256(post_paper_raw).hexdigest(),
        "paper_sha256": pre_paper.get("sha256"),
        "bundle_lock_sha256": expected_bundle_lock,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-pre", required=True)
    parser.add_argument("--source-post", required=True)
    parser.add_argument("--paper-pre", required=True)
    parser.add_argument("--paper-post", required=True)
    parser.add_argument("--expected-bundle-lock", required=True)
    args = parser.parse_args()
    try:
        result = compare(
            os.path.abspath(args.source_pre),
            os.path.abspath(args.source_post),
            os.path.abspath(args.paper_pre),
            os.path.abspath(args.paper_post),
            args.expected_bundle_lock,
        )
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(2)
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
