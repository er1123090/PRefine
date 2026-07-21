"""Create a deterministic, target-blind recovery2 inventory and amendment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True

from recovery2_common import (  # noqa: E402
    BOUND_SCRIPTS,
    GOLD_RELATIVE_PATH,
    ROOT,
    SOURCE_INVENTORY,
    SOURCE_ROOT,
    build_source_inventory,
    canonical_json,
    load_json,
    sha256_bytes,
    sha256_file,
    write_json_once,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inventory-output", type=Path, default=SOURCE_INVENTORY)
    return parser


def _require_parent_failure() -> tuple[dict, dict]:
    attempt = load_json(SOURCE_ROOT / "artifacts/final_test.attempt.json")
    result = load_json(SOURCE_ROOT / "artifacts/final_test.result.json")
    payload = result.get("payload", {})
    if (
        result.get("stage") != "result"
        or payload.get("execution_status") != "FAILED"
        or payload.get("integrity_status") != "FAIL"
        or payload.get("performance_status") != "NOT_RUN"
        or payload.get("summary") is not None
        or payload.get("evaluation_evidence") is not None
        or "stage=runtime" not in str(payload.get("detail"))
        or "error_class=RuntimeError" not in str(payload.get("detail"))
    ):
        raise ValueError("source run is not the sealed pre-gold runtime failure")
    if attempt.get("attempt_id") != result.get("attempt_id"):
        raise ValueError("source attempt/result identity mismatch")
    forbidden = (
        "artifacts/final_test.pre_gold.json",
        "artifacts/final_test.gold_open.json",
        "artifacts/final_test.critic.json",
        "reports/summary.json",
        "reports/final_evaluation.evidence.json",
    )
    existing = [path for path in forbidden if (SOURCE_ROOT / path).exists()]
    if existing:
        raise ValueError(f"source run unexpectedly accessed evaluation state: {existing}")
    return attempt, result


def main() -> int:
    args = _parser().parse_args()
    attempt, result = _require_parent_failure()
    inventory = build_source_inventory(SOURCE_ROOT)
    if args.inventory_output.exists():
        if load_json(args.inventory_output) != inventory:
            raise ValueError("existing target-blind source inventory differs from current source_run")
    else:
        write_json_once(args.inventory_output, inventory)

    sealed = load_json(SOURCE_ROOT / "evaluator_vault/sealed_manifest.json")
    binding = attempt.get("binding", {})
    if binding.get("declared_gold_sha256") != sealed.get("gold_sha256"):
        raise ValueError("attempt and evaluator vault declare different gold hashes")
    source_files = {
        "attempt": "artifacts/final_test.attempt.json",
        "runtime": "artifacts/final_test.runtime.json",
        "failed_result": "artifacts/final_test.result.json",
        "tasks": "artifacts/tasks.jsonl",
        "history": "artifacts/history.sanitized.jsonl",
        "prefine_memory": "artifacts/memory.prefine.jsonl",
        "prefine_manifest": "artifacts/memory.prefine.jsonl.manifest.json",
        "candidate_memory": "artifacts/memory.ecpr.jsonl",
        "candidate_memory_manifest": "artifacts/memory.ecpr.jsonl.manifest.json",
        "baseline_predictions": "artifacts/predictions.baseline.jsonl",
        "baseline_manifest": "artifacts/predictions.baseline.jsonl.manifest.json",
        "candidate_predictions": "artifacts/predictions.candidate.jsonl",
        "candidate_manifest": "artifacts/predictions.candidate.jsonl.manifest.json",
        "provider_journal": "artifacts/provider_calls.final.jsonl",
        "implementation_manifest": "manifests/implementation_manifest.json",
        "preregistration": "preregistration.json",
        "sealed_evaluator_manifest": "evaluator_vault/sealed_manifest.json",
    }
    source_hashes = {name: sha256_file(SOURCE_ROOT / path) for name, path in source_files.items()}
    amendment = {
        "schema_version": 1,
        "kind": "post_runtime_nvml_recovery_amendment_v1",
        "parent_attempt_id": attempt["attempt_id"],
        "parent_attempt_record_sha256": attempt.get("attempt_record_sha256"),
        "parent_failed_result_record_sha256": result.get("record_sha256"),
        "parent_performance_status": "NOT_RUN",
        "parent_failure_boundary": "after_both_prediction_manifests_before_post_provider_topology_and_pre_gold",
        "parent_failure_cause": "post_provider_nvidia_smi_NVML_Unknown_Error_propagated_as_RuntimeError",
        "gold_rows_read_before_recovery": False,
        "gold_relative_path": GOLD_RELATIVE_PATH,
        "declared_gold_sha256": sealed["gold_sha256"],
        "source_inventory_sha256": sha256_file(args.inventory_output),
        "source_inventory_rows_sha256": inventory["rows_sha256"],
        "source_hashes": source_hashes,
        "bound_scripts": {name: sha256_file(ROOT / name) for name in BOUND_SCRIPTS},
        "evaluator_code": {
            "registered": sha256_file(SOURCE_ROOT / "ecpr/evaluate.py"),
            "independent": sha256_file(SOURCE_ROOT / "ecpr/independent_audit.py"),
            "statistics": sha256_file(SOURCE_ROOT / "ecpr/statistics.py"),
            "parsing": sha256_file(SOURCE_ROOT / "ecpr/parsing.py"),
        },
        "recovery_contract": {
            "provider_calls_after_parent": 0,
            "prediction_regeneration": False,
            "prediction_bytes_reused_exactly": True,
            "method_or_prompt_changes": False,
            "target_adaptation": False,
            "allowed_gold_content_opens": 1,
            "gold_open_timing": "only_after_source_inventory_attempt_failure_and_full_provider_DAG_validation",
            "metric_axis": "unchanged_preregistered_registered_plus_independent_pair",
            "performance_gates": "unchanged_parent_preregistration",
            "post_provider_gpu_topology_claim": "unavailable_not_fabricated",
            "parent_gpu_use_evidence": "runtime_receipt_four_unique_UUIDs_TP4_and_completed_provider_DAG",
        },
        "amendment_content_sha256": None,
    }
    digest_view = dict(amendment)
    digest_view["amendment_content_sha256"] = None
    amendment["amendment_content_sha256"] = sha256_bytes(
        canonical_json(digest_view).encode("utf-8")
    )
    write_json_once(args.output, amendment)
    print(canonical_json({"sealed": True, "target_content_reads": 0, "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
