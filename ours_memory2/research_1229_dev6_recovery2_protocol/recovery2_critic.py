"""Strict read-only audit plus one no-replace receipt for recovery2."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True

from recovery2_common import (  # noqa: E402
    DEFAULT_AMENDMENT,
    RECOVERY_OUTPUTS,
    ROOT,
    SOURCE_ROOT,
    canonical_json,
    load_json,
    sha256_bytes,
    sha256_file,
    stage_record,
    validate_amendment,
    validate_stage_record,
    write_json_once,
)
from recovery2_runner import validate_parent_and_dag  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--strict", action="store_true")
    parser.add_argument("--amendment", type=Path, default=DEFAULT_AMENDMENT)
    return parser


def _validate_chain(amendment_sha256: str) -> tuple[dict, dict, dict, dict]:
    attempt = load_json(RECOVERY_OUTPUTS["attempt"])
    pre_gold = load_json(RECOVERY_OUTPUTS["pre_gold"])
    gold_open = load_json(RECOVERY_OUTPUTS["gold_open"])
    result = load_json(RECOVERY_OUTPUTS["result"])
    validate_stage_record(
        attempt,
        stage="attempt",
        sequence=0,
        amendment_sha256=amendment_sha256,
        previous=None,
    )
    validate_stage_record(
        pre_gold,
        stage="pre_gold",
        sequence=1,
        amendment_sha256=amendment_sha256,
        previous=attempt,
    )
    validate_stage_record(
        gold_open,
        stage="gold_open",
        sequence=2,
        amendment_sha256=amendment_sha256,
        previous=pre_gold,
    )
    validate_stage_record(
        result,
        stage="result",
        sequence=3,
        amendment_sha256=amendment_sha256,
        previous=gold_open,
    )
    return attempt, pre_gold, gold_open, result


def _validate_outputs(amendment: dict, amendment_sha256: str, dag: dict) -> tuple[dict, dict, dict]:
    if RECOVERY_OUTPUTS["failure"].exists():
        raise ValueError("recovery failure receipt exists")
    _attempt, pre_gold, gold_open, result = _validate_chain(amendment_sha256)
    summary = load_json(RECOVERY_OUTPUTS["summary"])
    evidence = load_json(RECOVERY_OUTPUTS["evidence"])
    payload = result.get("payload", {})
    if (
        pre_gold.get("payload", {}).get("provider_dag") != dag
        or pre_gold.get("payload", {}).get("gold_content_opens_so_far") != 0
        or pre_gold.get("payload", {}).get("provider_calls_after_parent") != 0
        or pre_gold.get("payload", {}).get("prediction_regeneration") is not False
        or gold_open.get("payload", {}).get("allowed_content_opens") != 1
        or gold_open.get("payload", {}).get("content_opened_before_receipt") is not False
    ):
        raise ValueError("recovery pre-gold/gold-open receipts violate the amendment")
    if (
        evidence.get("kind") != "post_runtime_nvml_recovery_dual_evaluator_evidence_v1"
        or evidence.get("amendment_sha256") != amendment_sha256
        or evidence.get("agreement") is not True
        or evidence.get("registered_summary_sha256") != sha256_file(RECOVERY_OUTPUTS["summary"])
        or evidence.get("gold_fd_identity", {}).get("sha256") != amendment["declared_gold_sha256"]
        or evidence.get("gold_fd_identity", {}).get("opened_once") is not True
        or evidence.get("gold_fd_identity", {}).get("nofollow") is not True
        or evidence.get("provider_calls_after_parent") != 0
        or evidence.get("prediction_regeneration") is not False
        or evidence.get("target_adaptation") is not False
    ):
        raise ValueError("dual evaluator evidence contract mismatch")
    independent = evidence.get("independent_audit", {})
    summary_pass = summary.get("pass")
    if not isinstance(summary_pass, bool) or independent.get("pass") is not summary_pass:
        raise ValueError("registered and independent pass axes differ")
    gates = summary.get("gates")
    if not isinstance(gates, dict) or summary_pass is not all(value is True for value in gates.values()):
        raise ValueError("summary pass value differs from preregistered gates")
    expected_performance = "PASS" if summary_pass else "FAIL"
    if (
        payload.get("execution_status") != "COMPLETE"
        or payload.get("integrity_status") != "PASS"
        or payload.get("performance_status") != expected_performance
        or payload.get("summary", {}).get("sha256") != sha256_file(RECOVERY_OUTPUTS["summary"])
        or payload.get("evaluation_evidence", {}).get("sha256")
        != sha256_file(RECOVERY_OUTPUTS["evidence"])
        or evidence.get("axes", {}).get("performance") != expected_performance
    ):
        raise ValueError("result axes or output bindings are inconsistent")
    source_forbidden = (
        "artifacts/final_test.pre_gold.json",
        "artifacts/final_test.gold_open.json",
        "artifacts/final_test.critic.json",
        "reports/summary.json",
        "reports/final_evaluation.evidence.json",
    )
    if any((SOURCE_ROOT / relative).exists() for relative in source_forbidden):
        raise ValueError("recovery mutated the immutable source run")
    return summary, evidence, result


def run_preflight(amendment_path: Path) -> int:
    amendment, amendment_sha256 = validate_amendment(amendment_path)
    attempt, _implementation, _runtime, dag = validate_parent_and_dag(amendment)
    print(
        canonical_json(
            {
                "critic_preflight": "PASS",
                "parent_attempt_id": attempt["attempt_id"],
                "amendment_sha256": amendment_sha256,
                "provider_dag_task_count": dag["task_count"],
                "gold_content_opens": 0,
            }
        )
    )
    return 0


def run_strict(amendment_path: Path) -> int:
    if RECOVERY_OUTPUTS["critic"].exists():
        raise FileExistsError("strict critic receipt already exists")
    amendment, amendment_sha256 = validate_amendment(amendment_path)
    attempt, _implementation, runtime, dag = validate_parent_and_dag(amendment)
    summary, evidence, result = _validate_outputs(amendment, amendment_sha256, dag)
    critic = stage_record(
        stage="critic",
        sequence=4,
        payload={
            "audit_status": "PASS",
            "performance_status": "PASS" if summary["pass"] else "FAIL",
            "parent_attempt_id": attempt["attempt_id"],
            "source_inventory_sha256": amendment["source_inventory_sha256"],
            "provider_dag_sha256": sha256_bytes(canonical_json(dag).encode("utf-8")),
            "summary_sha256": sha256_file(RECOVERY_OUTPUTS["summary"]),
            "evidence_sha256": sha256_file(RECOVERY_OUTPUTS["evidence"]),
            "result_record_sha256": result["record_sha256"],
            "registered_independent_agreement": evidence["agreement"],
            "single_gold_open_attested": evidence["gold_fd_identity"]["opened_once"],
            "source_run_unmodified": True,
            "four_gpu_parent_runtime": len(runtime["payload"]["gpus"]) == 4,
            "post_provider_topology_claim": "unavailable_not_fabricated",
        },
        amendment_sha256=amendment_sha256,
        previous=result,
    )
    write_json_once(RECOVERY_OUTPUTS["critic"], critic)
    print(
        canonical_json(
            {
                "audit_status": "PASS",
                "performance_status": critic["payload"]["performance_status"],
                "gold_reopened_by_critic": False,
            }
        )
    )
    return 0


def main() -> int:
    args = _parser().parse_args()
    if args.preflight_only:
        return run_preflight(args.amendment)
    return run_strict(args.amendment)


if __name__ == "__main__":
    raise SystemExit(main())
