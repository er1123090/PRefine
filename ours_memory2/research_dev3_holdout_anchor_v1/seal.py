"""Immutable pre-GPU implementation seal for the fresh holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ecpr.contracts import FORBIDDEN_KEYS, TASK_FIELDS
from ecpr.firewall import validate_sanitized_history
from ecpr.io import iter_jsonl, load_json, sha256_file, verify_sha256, write_json_once


ROOT = Path(__file__).resolve().parent
MANIFEST = Path("manifests/implementation_manifest.json")


def _static_paths(root: Path) -> list[Path]:
    explicit = [
        root / "README.md",
        root / "NONCHEATING_PROTOCOL.md",
        root / "AUTORESEARCH_HOLDOUT_RUBRIC.md",
        root / "preregistration.json",
        root / "fresh_prepare.py",
        root / "holdout_runner.py",
        root / "holdout_evaluator.py",
        root / "run_gpu.py",
        root / "critic.py",
        root / "seal.py",
        root / "tests/test_anchor_holdout.py",
        root / "artifacts/history.sanitized.jsonl",
        root / "artifacts/tasks.jsonl",
        root / "evaluator_vault/gold.jsonl",
        root / "evaluator_vault/sealed_manifest.json",
    ]
    explicit.extend(sorted((root / "configs").glob("*.json")))
    explicit.extend(sorted((root / "ecpr").glob("*.py")))
    if any(not path.is_file() or path.is_symlink() for path in explicit):
        missing = [str(path.relative_to(root)) for path in explicit if not path.is_file() or path.is_symlink()]
        raise ValueError("missing or unsafe static path: " + ", ".join(missing))
    return sorted(set(explicit))


def _validate_inputs(root: Path, preregistration: dict[str, Any]) -> None:
    if preregistration.get("status") != "locked_before_target_metrics":
        raise ValueError("preregistration is not locked")
    inputs = preregistration.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("preregistration inputs are missing")
    for name, spec in inputs.items():
        if not isinstance(spec, dict) or not isinstance(spec.get("path"), str) or not isinstance(spec.get("sha256"), str):
            raise ValueError(f"invalid preregistered input: {name}")
        verify_sha256(spec["path"], spec["sha256"])
    local = {
        "single_query": "query_singleturn.json",
        "single_schema": "schema_single.json",
        "multi_query": "query_multiturn-domain.json",
        "multi_schema": "schema_multi.json",
        "preference_slots": "preference_slots.json",
        "preference_groups": "preference_groups.json",
    }
    for name, filename in local.items():
        if sha256_file(root / "configs" / filename) != inputs[name]["sha256"]:
            raise ValueError(f"local frozen config hash mismatch: {name}")


def _validate_evaluator_seal(root: Path, preregistration: dict[str, Any]) -> dict[str, Any]:
    sealed = load_json(root / "evaluator_vault/sealed_manifest.json")
    history = root / "artifacts/history.sanitized.jsonl"
    tasks = root / "artifacts/tasks.jsonl"
    gold = root / "evaluator_vault/gold.jsonl"
    if sealed.get("preregistration_sha256") != sha256_file(root / "preregistration.json"):
        raise ValueError("evaluator seal preregistration mismatch")
    if sealed.get("history_sha256") != sha256_file(history):
        raise ValueError("evaluator seal sanitized-history mismatch")
    if sealed.get("task_sha256") != sha256_file(tasks):
        raise ValueError("evaluator seal public-task mismatch")
    if sealed.get("gold_sha256") != sha256_file(gold):
        raise ValueError("evaluator seal gold mismatch")
    history_count = validate_sanitized_history(history)
    task_rows = list(iter_jsonl(tasks))
    if len(task_rows) != sealed.get("case_count") or not task_rows:
        raise ValueError("evaluator seal case-count mismatch")
    if history_count != sealed.get("fresh_example_count"):
        raise ValueError("evaluator seal fresh-history-count mismatch")
    expected_fresh = int(preregistration["fresh_holdout"]["expected_example_count"])
    if history_count != expected_fresh:
        raise ValueError("fresh history differs from preregistration")
    keys: set[str] = set()
    for task in task_rows:
        if set(task) != TASK_FIELDS:
            raise ValueError("public task field contract violation")
        if {str(key).casefold() for key in task} & FORBIDDEN_KEYS:
            raise ValueError("public task leaked evaluator-only field")
        case_key = str(task.get("case_key", ""))
        if not case_key or case_key in keys:
            raise ValueError("public task key universe is invalid")
        keys.add(case_key)
    return {
        "history_count": history_count,
        "case_count": len(task_rows),
        "history_sha256": sha256_file(history),
        "task_sha256": sha256_file(tasks),
        "gold_sha256": sha256_file(gold),
        "evaluator_seal_sha256": sha256_file(root / "evaluator_vault/sealed_manifest.json"),
    }


def _files(root: Path) -> dict[str, dict[str, Any]]:
    return {
        str(path.relative_to(root)): {"sha256": sha256_file(path), "size": path.stat().st_size}
        for path in _static_paths(root)
    }


def _payload(root: Path) -> dict[str, Any]:
    preregistration = load_json(root / "preregistration.json")
    _validate_inputs(root, preregistration)
    evaluator = _validate_evaluator_seal(root, preregistration)
    return {
        "schema_version": 1,
        "kind": "fresh_holdout_implementation_seal_v1",
        "candidate": preregistration["candidate"],
        "baseline": preregistration["baseline"],
        "preregistration_sha256": sha256_file(root / "preregistration.json"),
        "candidate_parameters": preregistration["candidate_parameters"],
        "inference_budget": preregistration["inference_budget"],
        "model": preregistration["model"],
        "evaluator": evaluator,
        "files": _files(root),
    }


def seal(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = root / MANIFEST
    if manifest_path.exists() or manifest_path.is_symlink():
        raise FileExistsError("implementation seal already exists")
    forbidden_outputs = (
        root / "artifacts/final_attempt.json",
        root / "artifacts/memory.prefine_anchor.jsonl",
        root / "artifacts/predictions.baseline.jsonl",
        root / "artifacts/predictions.candidate.jsonl",
        root / "artifacts/pre_gold_receipt.json",
        root / "reports/summary.json",
        root / "reports/evaluation_evidence.json",
        root / "reports/final_result.json",
        root / "reports/critic.json",
    )
    present = [str(path.relative_to(root)) for path in forbidden_outputs if path.exists() or path.is_symlink()]
    if present:
        raise ValueError("cannot seal after final-run artifacts exist: " + ", ".join(present))
    payload = _payload(root)
    write_json_once(manifest_path, payload)
    return {
        "implementation_manifest_sha256": sha256_file(manifest_path),
        "case_count": payload["evaluator"]["case_count"],
        "fresh_example_count": payload["evaluator"]["history_count"],
    }


def validate(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = root / MANIFEST
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("implementation seal is missing")
    expected = load_json(manifest_path)
    actual = _payload(root)
    if actual != expected:
        raise ValueError("immutable implementation seal mismatch")
    return {
        "implementation_manifest_sha256": sha256_file(manifest_path),
        "case_count": actual["evaluator"]["case_count"],
        "fresh_example_count": actual["evaluator"]["history_count"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="seal or validate fresh holdout implementation")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--seal", action="store_true")
    mode.add_argument("--validate", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    result = seal(args.root) if args.seal else validate(args.root)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
