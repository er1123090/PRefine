"""One-shot evaluator recovery after a target-blind post-provider NVML failure."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

from recovery2_common import (  # noqa: E402
    DEFAULT_AMENDMENT,
    RECOVERY_OUTPUTS,
    ROOT,
    SOURCE_ROOT,
    assert_recovery_outputs_absent,
    canonical_json,
    load_json,
    sha256_bytes,
    sha256_file,
    stage_record,
    validate_amendment,
    write_json_once,
)

sys.path.insert(0, str(SOURCE_ROOT))

from ecpr.evaluate import compute_paired_summary  # noqa: E402
from ecpr.final_protocol import (  # noqa: E402
    open_regular_bytes_once,
    parse_jsonl_bytes,
    validate_final_attempt_v2,
)
from ecpr.independent_audit import (  # noqa: E402
    INDEPENDENCE_BOUNDARY,
    assert_registered_agreement,
    audit_paired_rows,
)
from ecpr.integrity import validate_preference_slots  # noqa: E402
from ecpr.io import load_json as source_load_json  # noqa: E402
from ecpr.ledger import validate_run_dag  # noqa: E402


SOURCE_PATHS = {
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--final", action="store_true")
    parser.add_argument("--amendment", type=Path, default=DEFAULT_AMENDMENT)
    return parser


def _validate_parent_failure(amendment: dict[str, Any]) -> tuple[dict, dict, dict]:
    expected_hashes = amendment.get("source_hashes")
    if not isinstance(expected_hashes, dict) or set(expected_hashes) != set(SOURCE_PATHS):
        raise ValueError("recovery amendment source hash inventory is incomplete")
    for name, relative in SOURCE_PATHS.items():
        if sha256_file(SOURCE_ROOT / relative) != expected_hashes[name]:
            raise ValueError(f"sealed source input changed: {name}")

    attempt, implementation, _ = validate_final_attempt_v2(SOURCE_ROOT)
    runtime = load_json(SOURCE_ROOT / SOURCE_PATHS["runtime"])
    result = load_json(SOURCE_ROOT / SOURCE_PATHS["failed_result"])
    payload = result.get("payload", {})
    if (
        attempt.get("attempt_id") != amendment.get("parent_attempt_id")
        or result.get("attempt_id") != attempt.get("attempt_id")
        or result.get("record_sha256") != amendment.get("parent_failed_result_record_sha256")
        or result.get("previous_stage") != "runtime"
        or result.get("previous_record_sha256") != runtime.get("record_sha256")
        or payload.get("performance_status") != "NOT_RUN"
        or payload.get("summary") is not None
        or payload.get("evaluation_evidence") is not None
        or "stage=runtime" not in str(payload.get("detail"))
        or "error_class=RuntimeError" not in str(payload.get("detail"))
    ):
        raise ValueError("parent failure receipt no longer proves pre-gold runtime failure")
    forbidden = (
        "artifacts/final_test.pre_gold.json",
        "artifacts/final_test.gold_open.json",
        "artifacts/final_test.critic.json",
        "reports/summary.json",
        "reports/final_evaluation.evidence.json",
    )
    if any((SOURCE_ROOT / relative).exists() for relative in forbidden):
        raise ValueError("parent source run contains target-evaluation artifacts")

    runtime_payload = runtime.get("payload", {})
    gpus = runtime_payload.get("gpus")
    contract = implementation.get("expected_contract", {})
    if (
        not isinstance(gpus, list)
        or len(gpus) != 4
        or len({row.get("uuid") for row in gpus if isinstance(row, dict)}) != 4
        or contract.get("model", {}).get("tensor_parallel_size") != 4
        or runtime_payload.get("environment", {}).get("values", {}).get("CUDA_VISIBLE_DEVICES")
        != "0,1,2,3"
    ):
        raise ValueError("parent runtime does not prove the registered four-GPU TP4 launch")
    return attempt, implementation, runtime


def validate_parent_and_dag(amendment: dict[str, Any]) -> tuple[dict, dict, dict, dict]:
    attempt, implementation, runtime = _validate_parent_failure(amendment)
    baseline = SOURCE_ROOT / SOURCE_PATHS["baseline_predictions"]
    candidate = SOURCE_ROOT / SOURCE_PATHS["candidate_predictions"]
    dag = validate_run_dag(SOURCE_ROOT, attempt["attempt_id"], baseline, candidate)
    if (
        dag.get("equal_action_budget") is not True
        or dag.get("baseline_action_call_count") != 2194
        or dag.get("candidate_action_call_count") != 2194
        or dag.get("arms") != ["baseline", "candidate"]
    ):
        raise ValueError("sealed source provider DAG is incomplete or budget-unequal")
    return attempt, implementation, runtime, dag


def _prediction_rows(amendment: dict[str, Any]) -> tuple[dict[str, list[dict]], dict[str, str]]:
    rows: dict[str, list[dict]] = {}
    hashes: dict[str, str] = {}
    for arm, key in (("baseline", "baseline_predictions"), ("candidate", "candidate_predictions")):
        path = SOURCE_ROOT / SOURCE_PATHS[key]
        payload, _identity = open_regular_bytes_once(path)
        digest = sha256_bytes(payload)
        if digest != amendment["source_hashes"][key]:
            raise ValueError(f"{arm} prediction bytes changed after source DAG validation")
        rows[arm] = parse_jsonl_bytes(payload, f"sealed recovery {arm} predictions")
        hashes[arm] = digest
    if len(rows["baseline"]) != 2194 or len(rows["candidate"]) != 2194:
        raise ValueError("recovery predictions do not preserve exact paired coverage")
    return rows, hashes


def _pre_gold_payload(
    amendment: dict[str, Any],
    amendment_sha256: str,
    attempt: dict,
    runtime: dict,
    dag: dict,
    prediction_hashes: dict[str, str],
) -> dict[str, Any]:
    runtime_payload = runtime["payload"]
    return {
        "parent_attempt_id": attempt["attempt_id"],
        "parent_failure_result_sha256": amendment["source_hashes"]["failed_result"],
        "parent_performance_status": "NOT_RUN",
        "source_inventory_sha256": amendment["source_inventory_sha256"],
        "amendment_sha256": amendment_sha256,
        "provider_dag": dag,
        "provider_dag_sha256": sha256_bytes(canonical_json(dag).encode("utf-8")),
        "official_outputs": {
            "baseline": {
                "path": SOURCE_PATHS["baseline_predictions"],
                "sha256": prediction_hashes["baseline"],
                "row_count": 2194,
            },
            "candidate": {
                "path": SOURCE_PATHS["candidate_predictions"],
                "sha256": prediction_hashes["candidate"],
                "row_count": 2194,
            },
        },
        "parent_gpu_runtime": {
            "gpus": runtime_payload["gpus"],
            "cuda_visible_devices": runtime_payload["environment"]["values"]["CUDA_VISIBLE_DEVICES"],
            "tensor_parallel_size": 4,
            "topology_at_readiness": runtime_payload["vllm"]["topology_at_readiness"],
        },
        "post_provider_topology_claim": "unavailable_not_fabricated_due_NVML_Unknown_Error",
        "provider_calls_after_parent": 0,
        "prediction_regeneration": False,
        "gold_content_opens_so_far": 0,
        "target_adaptation": False,
    }


def run_preflight(amendment_path: Path) -> int:
    amendment, amendment_sha256 = validate_amendment(amendment_path)
    attempt, _implementation, _runtime, dag = validate_parent_and_dag(amendment)
    print(
        canonical_json(
            {
                "preflight": "PASS",
                "parent_attempt_id": attempt["attempt_id"],
                "amendment_sha256": amendment_sha256,
                "task_count": dag["task_count"],
                "baseline_calls": dag["baseline_action_call_count"],
                "candidate_calls": dag["candidate_action_call_count"],
                "gold_content_opens": 0,
            }
        )
    )
    return 0


def run_final(amendment_path: Path) -> int:
    assert_recovery_outputs_absent()
    amendment, amendment_sha256 = validate_amendment(amendment_path)
    last_stage: dict[str, Any] | None = None
    try:
        attempt, _implementation, runtime, dag = validate_parent_and_dag(amendment)
        predictions, prediction_hashes = _prediction_rows(amendment)
        preregistration = source_load_json(SOURCE_ROOT / "preregistration.json")
        sealed = source_load_json(SOURCE_ROOT / "evaluator_vault/sealed_manifest.json")
        if sealed.get("gold_sha256") != amendment.get("declared_gold_sha256"):
            raise ValueError("declared gold hash differs from sealed evaluator manifest")
        preference_slots = validate_preference_slots(SOURCE_ROOT)

        last_stage = stage_record(
            stage="attempt",
            sequence=0,
            payload={
                "parent_attempt_id": attempt["attempt_id"],
                "parent_failed_result_record_sha256": amendment[
                    "parent_failed_result_record_sha256"
                ],
                "source_inventory_sha256": amendment["source_inventory_sha256"],
                "method_or_prompt_changes": False,
                "gold_content_opens_so_far": 0,
            },
            amendment_sha256=amendment_sha256,
            previous=None,
        )
        write_json_once(RECOVERY_OUTPUTS["attempt"], last_stage)

        pre_gold = stage_record(
            stage="pre_gold",
            sequence=1,
            payload=_pre_gold_payload(
                amendment,
                amendment_sha256,
                attempt,
                runtime,
                dag,
                prediction_hashes,
            ),
            amendment_sha256=amendment_sha256,
            previous=last_stage,
        )
        write_json_once(RECOVERY_OUTPUTS["pre_gold"], pre_gold)
        last_stage = pre_gold

        gold_open = stage_record(
            stage="gold_open",
            sequence=2,
            payload={
                "declared_gold_sha256": amendment["declared_gold_sha256"],
                "pre_gold_record_sha256": pre_gold["record_sha256"],
                "authorization": "single_O_NOFOLLOW_open_after_recovery_pre_gold",
                "allowed_content_opens": 1,
                "content_opened_before_receipt": False,
            },
            amendment_sha256=amendment_sha256,
            previous=pre_gold,
        )
        write_json_once(RECOVERY_OUTPUTS["gold_open"], gold_open)
        last_stage = gold_open

        gold_path = SOURCE_ROOT / amendment["gold_relative_path"]
        gold_bytes, gold_stat = open_regular_bytes_once(gold_path)
        gold_sha256 = sha256_bytes(gold_bytes)
        if gold_sha256 != amendment["declared_gold_sha256"]:
            raise ValueError("single-open gold bytes differ from preregistered hash")
        gold_rows = parse_jsonl_bytes(gold_bytes, "sealed recovery gold")

        summary = compute_paired_summary(
            gold_rows=gold_rows,
            baseline_rows=predictions["baseline"],
            candidate_rows=predictions["candidate"],
            preference_slots=preference_slots,
            preregistration=preregistration,
            provenance=dag,
            input_hashes={
                "gold": gold_sha256,
                "baseline_predictions": prediction_hashes["baseline"],
                "candidate_predictions": prediction_hashes["candidate"],
            },
        )
        independent = audit_paired_rows(
            gold_rows=gold_rows,
            baseline_rows=predictions["baseline"],
            candidate_rows=predictions["candidate"],
            preference_slots=preference_slots,
            preregistration=preregistration,
            provenance=dag,
        )
        assert_registered_agreement(summary, independent)
        write_json_once(RECOVERY_OUTPUTS["summary"], summary)
        evidence = {
            "schema_version": 1,
            "kind": "post_runtime_nvml_recovery_dual_evaluator_evidence_v1",
            "parent_attempt_id": attempt["attempt_id"],
            "amendment_sha256": amendment_sha256,
            "gold_fd_identity": {
                "device": int(gold_stat.st_dev),
                "inode": int(gold_stat.st_ino),
                "size": int(gold_stat.st_size),
                "sha256": gold_sha256,
                "opened_once": True,
                "nofollow": True,
            },
            "registered_summary_sha256": sha256_file(RECOVERY_OUTPUTS["summary"]),
            "independence_boundary": INDEPENDENCE_BOUNDARY,
            "independent_audit": independent,
            "agreement": True,
            "axes": {
                "integrity": "PASS",
                "execution": "COMPLETE",
                "performance": "PASS" if independent["pass"] else "FAIL",
            },
            "provider_calls_after_parent": 0,
            "prediction_regeneration": False,
            "target_adaptation": False,
        }
        write_json_once(RECOVERY_OUTPUTS["evidence"], evidence)

        result = stage_record(
            stage="result",
            sequence=3,
            payload={
                "execution_status": "COMPLETE",
                "integrity_status": "PASS",
                "performance_status": "PASS" if summary["pass"] else "FAIL",
                "summary": {
                    "path": str(RECOVERY_OUTPUTS["summary"].relative_to(ROOT)),
                    "sha256": sha256_file(RECOVERY_OUTPUTS["summary"]),
                    "reported_pass": bool(summary["pass"]),
                },
                "evaluation_evidence": {
                    "path": str(RECOVERY_OUTPUTS["evidence"].relative_to(ROOT)),
                    "sha256": sha256_file(RECOVERY_OUTPUTS["evidence"]),
                },
                "detail": "sealed_parent_outputs_single_gold_open_same_registered_and_independent_metrics",
            },
            amendment_sha256=amendment_sha256,
            previous=gold_open,
        )
        write_json_once(RECOVERY_OUTPUTS["result"], result)
        print(
            canonical_json(
                {
                    "case_count": summary["case_count"],
                    "pass": summary["pass"],
                    "gates": summary["gates"],
                    "gold_content_opens": 1,
                }
            )
        )
        return 0
    except Exception as exc:
        if not RECOVERY_OUTPUTS["failure"].exists():
            write_json_once(
                RECOVERY_OUTPUTS["failure"],
                {
                    "schema_version": 1,
                    "kind": "post_runtime_nvml_recovery_failure",
                    "last_stage": None if last_stage is None else last_stage.get("stage"),
                    "error_class": type(exc).__name__,
                    "retry_allowed": False,
                },
            )
        raise


def main() -> int:
    args = _parser().parse_args()
    if args.preflight_only:
        return run_preflight(args.amendment)
    return run_final(args.amendment)


if __name__ == "__main__":
    raise SystemExit(main())

