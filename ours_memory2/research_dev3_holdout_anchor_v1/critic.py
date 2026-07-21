"""Post-evaluation integrity critic; it reports no per-case information."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ecpr.io import load_json, sha256_file, write_json_once
from seal import validate


ROOT = Path(__file__).resolve().parent


def audit(root: Path) -> dict[str, Any]:
    root = root.resolve()
    implementation = validate(root)
    attempt_path = root / "artifacts/final_attempt.json"
    receipt_path = root / "artifacts/pre_gold_receipt.json"
    summary_path = root / "reports/summary.json"
    evidence_path = root / "reports/evaluation_evidence.json"
    result_path = root / "reports/final_result.json"
    for path in (attempt_path, receipt_path, summary_path, evidence_path, result_path):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing final artifact: {path.relative_to(root)}")
    attempt = load_json(attempt_path)
    receipt = load_json(receipt_path)
    summary = load_json(summary_path)
    evidence = load_json(evidence_path)
    result = load_json(result_path)
    if attempt.get("implementation_manifest_sha256") != implementation["implementation_manifest_sha256"]:
        raise ValueError("attempt is not bound to current immutable implementation")
    if receipt.get("equal_action_budget") is not True:
        raise ValueError("pre-gold receipt lacks equal action budget")
    for arm in ("baseline", "candidate"):
        expected = receipt.get("outputs", {}).get(arm, {}).get("sha256")
        actual = sha256_file(root / "artifacts" / f"predictions.{arm}.jsonl")
        if expected != actual:
            raise ValueError(f"{arm} predictions changed after pre-gold receipt")
    if evidence.get("registered_summary_sha256") != sha256_file(summary_path):
        raise ValueError("evaluation evidence does not bind summary")
    if evidence.get("agreement") is not True or evidence.get("gold_opened_once") is not True:
        raise ValueError("dual-evaluator or single-open evidence failed")
    if result.get("execution_status") != "COMPLETE" or result.get("integrity_status") != "PASS":
        raise ValueError("final runner did not complete with integrity PASS")
    expected_axis = "PASS" if summary.get("pass") is True else "FAIL"
    if result.get("performance_status") != expected_axis:
        raise ValueError("final performance axis does not match sealed summary")
    return {
        "schema_version": 1,
        "kind": "fresh_holdout_post_evaluation_critic_v1",
        "audit_status": "PASS",
        "implementation_manifest_sha256": implementation["implementation_manifest_sha256"],
        "summary_sha256": sha256_file(summary_path),
        "evaluation_evidence_sha256": sha256_file(evidence_path),
        "performance_status": expected_axis,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="audit fresh-holdout final artifacts")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = root / "reports/critic.json"
    if output.exists() or output.is_symlink():
        raise FileExistsError("critic receipt already exists")
    result = audit(root)
    write_json_once(output, result)
    print(json.dumps({"audit_status": result["audit_status"], "performance_status": result["performance_status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
