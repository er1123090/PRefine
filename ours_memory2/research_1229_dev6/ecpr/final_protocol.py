"""One-shot final-run capabilities, receipts, and fail-closed state validation.

This module deliberately contains no model or evaluator logic. It only binds
the immutable implementation seal to an in-memory runner capability and to a
small append-only chain of execution receipts.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import stat
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable

from .integrity import (
    ATTEMPT_LOCK,
    build_attempt_binding,
    commit_json_once,
    validate_final_attempt,
)
from .io import canonical_json, load_json, sha256_bytes, sha256_file


RUNNER_NONCE_BYTES = 32
ZERO_SHA256 = "0" * 64
STAGE_ORDER = ("runtime", "pre_gold", "gold_open", "result", "critic")
STAGE_PATHS = {
    "runtime": Path("artifacts/final_test.runtime.json"),
    "pre_gold": Path("artifacts/final_test.pre_gold.json"),
    "gold_open": Path("artifacts/final_test.gold_open.json"),
    "result": Path("artifacts/final_test.result.json"),
    "critic": Path("artifacts/final_test.critic.json"),
}
EVALUATION_EVIDENCE = Path("reports/final_evaluation.evidence.json")
PRE_GOLD_OUTPUT_PATHS = {
    "journal": "artifacts/provider_calls.final.jsonl",
    "baseline_latent": "artifacts/memory.prefine.jsonl",
    "baseline_latent_manifest": "artifacts/memory.prefine.jsonl.manifest.json",
    "memory": "artifacts/memory.ecpr.jsonl",
    "memory_manifest": "artifacts/memory.ecpr.jsonl.manifest.json",
    "baseline": "artifacts/predictions.baseline.jsonl",
    "baseline_manifest": "artifacts/predictions.baseline.jsonl.manifest.json",
    "candidate": "artifacts/predictions.candidate.jsonl",
    "candidate_manifest": "artifacts/predictions.candidate.jsonl.manifest.json",
}
RUN_DAG_FIELDS = {
    "schema_version",
    "attempt_id",
    "provider_journal_sha256",
    "memory_manifest_sha256",
    "baseline_prediction_manifest_sha256",
    "candidate_prediction_manifest_sha256",
    "equal_action_budget",
    "task_count",
    "arms",
    "memory_call_count",
    "baseline_action_call_count",
    "candidate_action_call_count",
    "total_call_count",
    "expected_total_call_count",
    "journal_record_count",
    "expected_journal_record_count",
    "final_journal_sequence",
    "final_journal_record_sha256",
    "phases",
    "no_extra_provider_calls_or_phases",
    "authoritative_action_contract",
    "authoritative_action_contract_sha256",
}


def _require_digest(value: Any, label: str, *, allow_zero: bool = False) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        or (not allow_zero and value == ZERO_SHA256)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _canonical_uuid(value: Any, label: str) -> str:
    try:
        canonical = str(uuid.UUID(str(value)))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid {label}") from exc
    if canonical != value:
        raise ValueError(f"{label} is not canonical")
    return canonical


def stage_record_sha256(record: dict[str, Any]) -> str:
    payload = dict(record)
    payload.pop("record_sha256", None)
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _validate_nonce_bytes(value: bytes | bytearray, label: str) -> bytes:
    if not isinstance(value, (bytes, bytearray)) or len(value) != RUNNER_NONCE_BYTES:
        raise ValueError(f"{label} must contain exactly {RUNNER_NONCE_BYTES} bytes")
    return bytes(value)


def read_secret_fd(fd: int, label: str) -> bytes:
    """Consume exactly one 32-byte secret from an inherited anonymous pipe."""
    if not isinstance(fd, int) or isinstance(fd, bool) or fd < 3:
        raise ValueError(f"{label} descriptor must be an inherited non-standard FD")
    payload = bytearray()
    try:
        while len(payload) <= RUNNER_NONCE_BYTES:
            chunk = os.read(fd, RUNNER_NONCE_BYTES + 1 - len(payload))
            if not chunk:
                break
            payload.extend(chunk)
    except OSError as exc:
        raise ValueError(f"unable to read {label} descriptor") from exc
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
    if len(payload) != RUNNER_NONCE_BYTES:
        for index in range(len(payload)):
            payload[index] = 0
        raise ValueError(f"{label} pipe must contain exactly {RUNNER_NONCE_BYTES} bytes")
    result = bytes(payload)
    for index in range(len(payload)):
        payload[index] = 0
    return result


def acquire_final_attempt_v2(
    root: Path,
    binding: dict[str, Any],
    runner_nonce: bytes | bytearray,
    *,
    attempt_id_factory: Callable[[], Any] = uuid.uuid4,
    clock: Callable[[], int] = time.time_ns,
) -> dict[str, Any]:
    """Commit the sole v2 attempt while retaining the nonce only in memory."""
    root = root.resolve()
    nonce = _validate_nonce_bytes(runner_nonce, "runner nonce")
    expected_binding = build_attempt_binding(root)
    if binding != expected_binding:
        raise ValueError("requested final attempt binding is stale or incomplete")
    attempt_id = str(uuid.UUID(str(attempt_id_factory())))
    record = {
        "schema_version": 2,
        "kind": "sealed_final_attempt",
        "attempt_id": attempt_id,
        "acquired_unix_ns": int(clock()),
        "runner_nonce_sha256": sha256_bytes(nonce),
        "binding": expected_binding,
    }
    commit_json_once(root / ATTEMPT_LOCK, record)
    return record


def validate_final_attempt_v2(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    root = root.resolve()
    attempt_bytes, _attempt_stat = open_regular_bytes_once(root / ATTEMPT_LOCK)
    try:
        same_fd_attempt = json.loads(attempt_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid v2 final-attempt JSON") from exc
    attempt, implementation, digest = validate_final_attempt(root)
    if not isinstance(same_fd_attempt, dict) or same_fd_attempt != attempt:
        raise ValueError("final attempt changed across same-FD validation")
    if attempt.get("schema_version") != 2 or set(attempt) != {
        "schema_version",
        "kind",
        "attempt_id",
        "acquired_unix_ns",
        "runner_nonce_sha256",
        "binding",
    }:
        raise ValueError("Phase3B-B requires the v2 final-attempt contract")
    _require_digest(attempt.get("runner_nonce_sha256"), "runner nonce digest")
    return attempt, implementation, digest


def _attempt_record_sha256_once(
    root: Path, expected_attempt: dict[str, Any]
) -> str:
    payload, _identity = open_regular_bytes_once(root / ATTEMPT_LOCK)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid final-attempt JSON") from exc
    if value != expected_attempt:
        raise ValueError("final attempt changed before receipt validation")
    return sha256_bytes(payload)


def verify_runner_nonce(attempt: dict[str, Any], runner_nonce: bytes | bytearray) -> None:
    nonce = _validate_nonce_bytes(runner_nonce, "runner nonce")
    expected = _require_digest(
        attempt.get("runner_nonce_sha256"), "runner nonce digest"
    )
    if not hmac.compare_digest(sha256_bytes(nonce), expected):
        raise ValueError("wrong runner nonce")


def authorize_final_cli(
    root: Path,
    attempt_id: str,
    runner_nonce_fd: int,
    *,
    require_runtime: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    """Authorize one child command; the caller must discard the returned nonce."""
    attempt, implementation, _ = validate_final_attempt_v2(root)
    if attempt.get("attempt_id") != attempt_id:
        raise ValueError("CLI attempt ID does not match the sealed final attempt")
    nonce = read_secret_fd(runner_nonce_fd, "runner nonce")
    verify_runner_nonce(attempt, nonce)
    if require_runtime:
        stages = validate_stage_chain(root)
        if "runtime" not in stages or "result" in stages:
            raise ValueError("final CLI requires an active sealed runtime")
    return attempt, implementation, nonce


def stage_path(root: Path, stage: str) -> Path:
    if stage not in STAGE_PATHS:
        raise ValueError(f"unknown final-protocol stage: {stage}")
    return root.resolve() / STAGE_PATHS[stage]


def _result_contract(payload: Any) -> None:
    if not isinstance(payload, dict) or set(payload) != {
        "execution_status",
        "integrity_status",
        "performance_status",
        "summary",
        "evaluation_evidence",
        "detail",
    }:
        raise ValueError("result payload field contract violation")
    execution = payload.get("execution_status")
    performance = payload.get("performance_status")
    integrity = payload.get("integrity_status")
    if execution not in {"FAILED", "COMPLETE"}:
        raise ValueError("result execution status must be terminal")
    if integrity not in {"PASS", "FAIL"}:
        raise ValueError("result integrity status mismatch")
    if not isinstance(payload.get("detail"), str) or len(payload["detail"]) > 2000:
        raise ValueError("result detail contract violation")
    summary = payload.get("summary")
    evidence = payload.get("evaluation_evidence")
    if execution == "COMPLETE":
        if performance not in {"PASS", "FAIL"} or integrity != "PASS":
            raise ValueError("completed result has inconsistent axes")
        if not isinstance(summary, dict) or set(summary) != {
            "path",
            "sha256",
            "reported_pass",
        }:
            raise ValueError("completed result lacks a sealed summary identity")
        _require_digest(summary.get("sha256"), "summary digest")
        if summary.get("reported_pass") is not (performance == "PASS"):
            raise ValueError("result performance disagrees with summary")
        if not isinstance(evidence, dict) or set(evidence) != {"path", "sha256"}:
            raise ValueError("completed result lacks sealed dual-evaluator evidence")
        _require_digest(evidence.get("sha256"), "evaluation evidence digest")
    elif performance != "NOT_RUN" or summary is not None or evidence is not None:
        raise ValueError("failed execution cannot report performance")


def _critic_contract(payload: Any) -> None:
    if not isinstance(payload, dict) or set(payload) != {
        "kind",
        "audit_status",
        "integrity_status",
        "execution_status",
        "performance_status",
        "result_record_sha256",
        "implementation_manifest_sha256",
        "critic_source_sha256",
        "summary_sha256",
        "evaluation_evidence_sha256",
        "audit_axes_sha256",
        "detail",
    }:
        raise ValueError("critic receipt payload field contract violation")
    for field in (
        "result_record_sha256",
        "implementation_manifest_sha256",
        "critic_source_sha256",
        "summary_sha256",
        "evaluation_evidence_sha256",
        "audit_axes_sha256",
    ):
        _require_digest(payload.get(field), f"critic {field}")
    if (
        payload.get("kind") != "strict_final_critic_receipt"
        or payload.get("audit_status") not in {"PASS", "FAIL"}
        or payload.get("execution_status") != "COMPLETE"
        or not isinstance(payload.get("detail"), str)
        or len(payload["detail"]) > 2000
    ):
        raise ValueError("critic receipt identity or status contract violation")
    if payload["audit_status"] == "PASS":
        if (
            payload.get("integrity_status") != "PASS"
            or payload.get("performance_status") not in {"PASS", "FAIL"}
            or payload.get("detail") != "strict_critic_audit_passed"
        ):
            raise ValueError("PASS critic receipt has inconsistent axes")
    elif (
        payload.get("integrity_status") != "FAIL"
        or payload.get("performance_status") != "NOT_RUN"
        or payload.get("detail") != "strict_critic_audit_failed_no_retry"
    ):
        raise ValueError("FAIL critic receipt has inconsistent axes")


def _process_identity_contract(value: Any, label: str) -> None:
    if not isinstance(value, dict) or set(value) != {
        "pid",
        "start_ticks",
        "executable",
        "executable_sha256",
        "argv_sha256",
    }:
        raise ValueError(f"{label} process identity field contract violation")
    if (
        not isinstance(value.get("pid"), int)
        or value["pid"] <= 0
        or not isinstance(value.get("start_ticks"), int)
        or value["start_ticks"] <= 0
        or not isinstance(value.get("executable"), str)
        or not value["executable"]
    ):
        raise ValueError(f"{label} process identity value contract violation")
    _require_digest(value.get("argv_sha256"), f"{label} argv digest")
    _require_digest(value.get("executable_sha256"), f"{label} executable digest")


def _topology_contract(value: Any, label: str) -> None:
    if not isinstance(value, dict) or set(value) != {
        "process_group_id",
        "descendant_processes",
        "compute_app_mappings",
        "mapped_gpu_uuids",
    }:
        raise ValueError(f"{label} topology field contract violation")
    if not isinstance(value.get("process_group_id"), int) or value["process_group_id"] <= 0:
        raise ValueError(f"{label} process group is invalid")
    processes = value.get("descendant_processes")
    mappings = value.get("compute_app_mappings")
    uuids = value.get("mapped_gpu_uuids")
    if not isinstance(processes, list) or not processes:
        raise ValueError(f"{label} descendant list is empty")
    for identity in processes:
        _process_identity_contract(identity, f"{label} descendant")
    process_ids = {identity["pid"] for identity in processes}
    if len(process_ids) != len(processes):
        raise ValueError(f"{label} descendant process identities are duplicated")
    if (
        not isinstance(mappings, list)
        or not mappings
        or any(
            not isinstance(mapping, dict)
            or set(mapping) != {"pid", "gpu_uuid"}
            or mapping.get("pid") not in process_ids
            or not isinstance(mapping.get("gpu_uuid"), str)
            or not mapping["gpu_uuid"]
            for mapping in mappings
        )
    ):
        raise ValueError(f"{label} compute-app mapping is invalid")
    if len({(mapping["pid"], mapping["gpu_uuid"]) for mapping in mappings}) != len(
        mappings
    ):
        raise ValueError(f"{label} compute-app mappings are duplicated")
    mapped = sorted({mapping["gpu_uuid"] for mapping in mappings})
    if not isinstance(uuids, list) or uuids != mapped or len(mapped) != 4:
        raise ValueError(f"{label} must map exactly four unique GPU UUIDs")


def _runtime_contract(payload: Any) -> None:
    if not isinstance(payload, dict) or set(payload) != {
        "python",
        "vllm",
        "model",
        "gpus",
        "server",
        "environment",
        "runner",
        "planned_commands",
        "gvr_call_graph",
    }:
        raise ValueError("runtime payload field contract violation")
    python = payload.get("python")
    if not isinstance(python, dict) or set(python) != {
        "executable",
        "sha256",
        "version",
        "argv",
        "argv_sha256",
    }:
        raise ValueError("runtime Python identity field contract violation")
    vllm = payload.get("vllm")
    if not isinstance(vllm, dict) or set(vllm) != {
        "executable",
        "sha256",
        "version",
        "argv",
        "argv_sha256",
        "process",
        "topology_at_readiness",
    }:
        raise ValueError("runtime vLLM identity field contract violation")
    for label, value in (("Python", python), ("vLLM", vllm)):
        if (
            not isinstance(value.get("executable"), str)
            or not value["executable"]
            or not isinstance(value.get("version"), str)
            or not value["version"]
            or not isinstance(value.get("argv"), list)
            or not value["argv"]
            or any(not isinstance(item, str) or not item for item in value["argv"])
        ):
            raise ValueError(f"runtime {label} identity value contract violation")
        _require_digest(value.get("sha256"), f"runtime {label} executable digest")
        if value.get("argv_sha256") != hashlib.sha256(
            canonical_json(value["argv"]).encode("utf-8")
        ).hexdigest():
            raise ValueError(f"runtime {label} argv digest mismatch")
    _process_identity_contract(vllm.get("process"), "vLLM")
    _topology_contract(vllm.get("topology_at_readiness"), "runtime readiness")
    _process_identity_contract(payload.get("runner"), "runner")
    model = payload.get("model")
    if not isinstance(model, dict) or set(model) != {"snapshot_path", "tree_sha256"}:
        raise ValueError("runtime model identity field contract violation")
    if not isinstance(model.get("snapshot_path"), str) or not model["snapshot_path"]:
        raise ValueError("runtime model path is missing")
    _require_digest(model.get("tree_sha256"), "runtime model tree digest")
    gpus = payload.get("gpus")
    if (
        not isinstance(gpus, list)
        or not gpus
        or any(
            not isinstance(item, dict)
            or set(item) != {"index", "uuid", "pci_bus_id"}
            or not isinstance(item.get("index"), int)
            or not isinstance(item.get("uuid"), str)
            or not item["uuid"]
            or not isinstance(item.get("pci_bus_id"), str)
            or not item["pci_bus_id"]
            for item in gpus
        )
        or len(gpus) != 4
        or len({item["index"] for item in gpus}) != len(gpus)
        or len({item["uuid"] for item in gpus}) != len(gpus)
        or len({item["pci_bus_id"] for item in gpus}) != len(gpus)
    ):
        raise ValueError("runtime GPU identity contract violation")
    server = payload.get("server")
    if not isinstance(server, dict) or set(server) != {
        "base_url",
        "models_response",
        "models_response_raw_sha256",
    }:
        raise ValueError("runtime server receipt field contract violation")
    if not isinstance(server.get("base_url"), str) or not isinstance(
        server.get("models_response"), dict
    ):
        raise ValueError("runtime server receipt value contract violation")
    _require_digest(server.get("models_response_raw_sha256"), "models response digest")
    environment = payload.get("environment")
    if not isinstance(environment, dict) or set(environment) != {
        "policy",
        "values",
        "sha256",
    }:
        raise ValueError("runtime environment receipt field contract violation")
    if environment.get("policy") != "fixed_allowlist_no_credentials" or not isinstance(
        environment.get("values"), dict
    ):
        raise ValueError("runtime environment policy mismatch")
    if environment.get("sha256") != hashlib.sha256(
        canonical_json(environment["values"]).encode("utf-8")
    ).hexdigest():
        raise ValueError("runtime environment receipt digest mismatch")
    planned = payload.get("planned_commands")
    if (
        not isinstance(planned, list)
        or [item.get("name") for item in planned]
        != ["memory", "baseline", "candidate", "evaluate", "critic"]
        or any(
            not isinstance(item, dict)
            or set(item) != {"name", "argv_sha256", "timeout_seconds"}
            or not isinstance(item.get("timeout_seconds"), int)
            or item["timeout_seconds"] <= 0
            for item in planned
        )
    ):
        raise ValueError("runtime planned-command contract violation")
    for item in planned:
        _require_digest(item.get("argv_sha256"), "planned command digest")
    gvr = payload.get("gvr_call_graph")
    if not isinstance(gvr, dict) or set(gvr) != {
        "state_machine_source_sha256",
        "max_refinements_per_session",
        "sequence",
        "generation",
        "verification",
        "transitions",
        "held_out_action_feedback",
    }:
        raise ValueError("runtime GVR call-graph field contract violation")
    _require_digest(gvr.get("state_machine_source_sha256"), "GVR source digest")
    if (
        gvr.get("max_refinements_per_session") != 10
        or gvr.get("sequence") != ["draft", "verify", "refine_if_invalid"]
        or gvr.get("held_out_action_feedback")
        != "forbidden_no_outgoing_provider_or_memory_edges"
    ):
        raise ValueError("runtime GVR transition contract mismatch")
    for name in ("generation", "verification"):
        value = gvr.get(name)
        if not isinstance(value, dict):
            raise ValueError(f"runtime GVR {name} contract is missing")
        for key, item in value.items():
            if key.endswith("sha256"):
                _require_digest(item, f"GVR {name}/{key}")
    if not isinstance(gvr.get("transitions"), dict):
        raise ValueError("runtime GVR transitions are missing")


def _pre_gold_contract(payload: Any) -> None:
    if not isinstance(payload, dict) or set(payload) != {
        "run_dag",
        "run_dag_sha256",
        "official_outputs",
        "server_stopped",
        "runtime_revalidated_before_gold",
        "implementation_manifest_sha256",
        "runtime_record_sha256",
        "provider_journal_seal",
        "post_first_provider_topology",
        "post_first_provider_topology_sha256",
        "post_provider_topology",
        "post_provider_topology_sha256",
    }:
        raise ValueError("pre_gold payload field contract violation")
    dag = payload.get("run_dag")
    if not isinstance(dag, dict) or set(dag) != RUN_DAG_FIELDS:
        raise ValueError("pre_gold run DAG field contract violation")
    if payload.get("run_dag_sha256") != hashlib.sha256(
        canonical_json(dag).encode("utf-8")
    ).hexdigest():
        raise ValueError("pre_gold DAG digest mismatch")
    for name in (
        "provider_journal_sha256",
        "memory_manifest_sha256",
        "baseline_prediction_manifest_sha256",
        "candidate_prediction_manifest_sha256",
        "final_journal_record_sha256",
        "authoritative_action_contract_sha256",
    ):
        _require_digest(dag.get(name), f"run DAG {name}")
    counts = {
        name: dag.get(name)
        for name in (
            "task_count",
            "memory_call_count",
            "baseline_action_call_count",
            "candidate_action_call_count",
            "total_call_count",
            "expected_total_call_count",
            "journal_record_count",
            "expected_journal_record_count",
            "final_journal_sequence",
        )
    }
    if any(not isinstance(value, int) or value <= 0 for value in counts.values()):
        raise ValueError("run DAG cardinalities must be positive integers")
    task_count = counts["task_count"]
    total_calls = counts["total_call_count"]
    if (
        dag.get("schema_version") != 1
        or _canonical_uuid(dag.get("attempt_id"), "run DAG attempt ID")
        != dag.get("attempt_id")
        or dag.get("equal_action_budget") is not True
        or dag.get("arms") != ["baseline", "candidate"]
        or counts["baseline_action_call_count"] != task_count
        or counts["candidate_action_call_count"] != task_count
        or total_calls != counts["memory_call_count"] + 2 * task_count
        or counts["expected_total_call_count"] != total_calls
        or counts["journal_record_count"] != 2 * total_calls
        or counts["expected_journal_record_count"] != 2 * total_calls
        or counts["final_journal_sequence"] != counts["journal_record_count"]
        or dag.get("phases")
        != ["memory_generation", "action_baseline", "action_candidate"]
        or dag.get("no_extra_provider_calls_or_phases") is not True
    ):
        raise ValueError("run DAG exact call cardinality or phase contract mismatch")
    authoritative = dag.get("authoritative_action_contract")
    if not isinstance(authoritative, dict) or set(authoritative) != {
        "model",
        "action_budget",
        "provider_timeout_seconds",
        "call_count_per_arm",
    }:
        raise ValueError("run DAG authoritative action contract is missing")
    model = authoritative.get("model")
    budget = authoritative.get("action_budget")
    if (
        not isinstance(model, dict)
        or not isinstance(model.get("id"), str)
        or not model["id"]
        or not isinstance(budget, dict)
        or type(budget.get("seed")) is not int
        or not isinstance(budget.get("max_tokens"), int)
        or budget["max_tokens"] <= 0
        or budget.get("calls_per_case") != 1
        or budget.get("retries") != 0
        or budget.get("n") != 1
        or authoritative.get("call_count_per_arm") != task_count
        or isinstance(authoritative.get("provider_timeout_seconds"), bool)
        or not isinstance(authoritative.get("provider_timeout_seconds"), (int, float))
        or authoritative["provider_timeout_seconds"] <= 0
        or dag.get("authoritative_action_contract_sha256")
        != hashlib.sha256(canonical_json(authoritative).encode("utf-8")).hexdigest()
    ):
        raise ValueError("run DAG model/seed/budget binding mismatch")
    outputs = payload.get("official_outputs")
    if not isinstance(outputs, dict) or set(outputs) != set(PRE_GOLD_OUTPUT_PATHS):
        raise ValueError("pre_gold exact official outputs are missing")
    for name, value in outputs.items():
        if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
            raise ValueError("pre_gold output identity field contract violation")
        if value.get("path") != PRE_GOLD_OUTPUT_PATHS[name]:
            raise ValueError("pre_gold output path contract violation")
        _require_digest(value.get("sha256"), "pre_gold output digest")
    stopped = payload.get("server_stopped")
    if not isinstance(stopped, dict) or set(stopped) != {
        "process",
        "process_group_id",
        "recorded_processes",
        "return_code",
        "all_recorded_processes_dead",
        "process_group_empty",
    }:
        raise ValueError("pre_gold server-stop receipt field contract violation")
    _process_identity_contract(stopped.get("process"), "stopped vLLM")
    if (
        not isinstance(stopped.get("return_code"), int)
        or not isinstance(stopped.get("process_group_id"), int)
        or stopped["process_group_id"] <= 0
        or stopped.get("all_recorded_processes_dead") is not True
        or stopped.get("process_group_empty") is not True
        or not isinstance(stopped.get("recorded_processes"), list)
        or not stopped["recorded_processes"]
    ):
        raise ValueError("pre_gold server-stop receipt value contract violation")
    for identity in stopped["recorded_processes"]:
        _process_identity_contract(identity, "recorded vLLM descendant")
    recorded_keys = {
        (identity["pid"], identity["start_ticks"])
        for identity in stopped["recorded_processes"]
    }
    if (
        len(recorded_keys) != len(stopped["recorded_processes"])
        or (
            stopped["process"]["pid"],
            stopped["process"]["start_ticks"],
        )
        not in recorded_keys
    ):
        raise ValueError("server-stop recorded-process coverage mismatch")
    if payload.get("runtime_revalidated_before_gold") is not True:
        raise ValueError("pre_gold lacks runtime revalidation")
    _require_digest(
        payload.get("implementation_manifest_sha256"),
        "pre_gold implementation-manifest digest",
    )
    _require_digest(payload.get("runtime_record_sha256"), "pre_gold runtime digest")
    journal = payload.get("provider_journal_seal")
    if not isinstance(journal, dict) or set(journal) != {
        "path",
        "sha256",
        "device",
        "inode",
        "size",
        "record_count",
        "final_sequence",
        "final_record_sha256",
        "mode",
    }:
        raise ValueError("pre_gold provider-journal seal field contract violation")
    _require_digest(journal.get("sha256"), "sealed provider-journal digest")
    _require_digest(
        journal.get("final_record_sha256"), "sealed provider-journal terminal digest"
    )
    if (
        journal.get("path") != PRE_GOLD_OUTPUT_PATHS["journal"]
        or journal.get("mode") != "0400"
        or not isinstance(journal.get("device"), int)
        or journal["device"] < 0
        or not isinstance(journal.get("inode"), int)
        or journal["inode"] <= 0
        or not isinstance(journal.get("size"), int)
        or journal["size"] <= 0
        or not isinstance(journal.get("record_count"), int)
        or journal["record_count"] <= 0
        or journal.get("final_sequence") != journal["record_count"]
    ):
        raise ValueError("pre_gold provider-journal seal value contract violation")
    for name in ("post_first_provider_topology", "post_provider_topology"):
        topology = payload.get(name)
        _topology_contract(topology, name)
        if payload.get(f"{name}_sha256") != hashlib.sha256(
            canonical_json(topology).encode("utf-8")
        ).hexdigest():
            raise ValueError(f"pre_gold {name} digest mismatch")


def _gold_open_contract(payload: Any) -> None:
    if not isinstance(payload, dict) or set(payload) != {
        "declared_gold_sha256",
        "pre_gold_record_sha256",
        "capability_transport",
        "capability_secret_persisted",
        "capability_digest_persisted",
    }:
        raise ValueError("gold_open payload field contract violation")
    _require_digest(payload.get("declared_gold_sha256"), "declared gold digest")
    _require_digest(payload.get("pre_gold_record_sha256"), "pre_gold receipt digest")
    if (
        payload.get("capability_transport") != "anonymous_pipe_fd"
        or payload.get("capability_secret_persisted") is not False
        or payload.get("capability_digest_persisted") is not False
    ):
        raise ValueError("gold_open capability persistence contract violation")


def _stage_payload_contract(stage: str, payload: Any) -> None:
    if stage == "runtime":
        _runtime_contract(payload)
    elif stage == "pre_gold":
        _pre_gold_contract(payload)
    elif stage == "gold_open":
        _gold_open_contract(payload)
    elif stage == "result":
        _result_contract(payload)
    elif stage == "critic":
        _critic_contract(payload)


def _validate_stage_record(
    record: Any,
    *,
    stage: str,
    attempt: dict[str, Any],
    attempt_record_sha256: str,
    sequence: int,
    previous_stage: str,
    previous_sha256: str,
) -> dict[str, Any]:
    if not isinstance(record, dict) or set(record) != {
        "schema_version",
        "kind",
        "stage",
        "sequence",
        "attempt_id",
        "attempt_record_sha256",
        "runner_nonce_sha256",
        "previous_stage",
        "previous_record_sha256",
        "committed_unix_ns",
        "payload",
        "record_sha256",
    }:
        raise ValueError(f"{stage} receipt field contract violation")
    if (
        record.get("schema_version") != 1
        or record.get("kind") != "final_protocol_stage"
        or record.get("stage") != stage
        or record.get("sequence") != sequence
        or record.get("attempt_id") != attempt["attempt_id"]
        or record.get("attempt_record_sha256") != attempt_record_sha256
        or record.get("runner_nonce_sha256") != attempt["runner_nonce_sha256"]
        or record.get("previous_stage") != previous_stage
        or record.get("previous_record_sha256") != previous_sha256
        or not isinstance(record.get("committed_unix_ns"), int)
        or record.get("record_sha256") != stage_record_sha256(record)
    ):
        raise ValueError(f"{stage} receipt hash-chain contract violation")
    if not isinstance(record.get("payload"), dict):
        raise ValueError(f"{stage} receipt payload must be an object")
    _stage_payload_contract(stage, record["payload"])
    return record


def validate_stage_chain(root: Path) -> dict[str, dict[str, Any]]:
    root = root.resolve()
    attempt, implementation, _ = validate_final_attempt_v2(root)
    attempt_record_sha256 = _attempt_record_sha256_once(root, attempt)
    previous_stage = "attempt"
    previous_sha256 = attempt_record_sha256
    sequence = 1
    records: dict[str, dict[str, Any]] = {}
    gap_seen = False
    for stage in STAGE_ORDER:
        path = stage_path(root, stage)
        exists = path.exists() or path.is_symlink()
        if not exists:
            if stage != "result":
                gap_seen = True
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{stage} receipt must be a regular non-symlink file")
        if stage != "result" and gap_seen:
            raise ValueError("final-protocol stage ordering has a gap")
        record = _validate_stage_record(
            load_regular_json_once(path),
            stage=stage,
            attempt=attempt,
            attempt_record_sha256=attempt_record_sha256,
            sequence=sequence,
            previous_stage=previous_stage,
            previous_sha256=previous_sha256,
        )
        records[stage] = record
        previous_stage = stage
        previous_sha256 = record["record_sha256"]
        sequence += 1
    runtime = records.get("runtime")
    if runtime is not None:
        runtime_payload = runtime["payload"]
        runtime_process = runtime_payload["vllm"]["process"]
        runtime_topology = runtime_payload["vllm"]["topology_at_readiness"]
        expected_gpu_uuids = sorted(gpu["uuid"] for gpu in runtime_payload["gpus"])
        if runtime_topology["mapped_gpu_uuids"] != expected_gpu_uuids:
            raise ValueError("runtime topology GPU UUIDs differ from the PCI GPU receipt")
        if runtime_process not in runtime_topology["descendant_processes"]:
            raise ValueError("runtime vLLM process is absent from its descendant receipt")
        if runtime_topology["process_group_id"] != runtime_process["pid"]:
            raise ValueError("runtime vLLM process is not the sealed process-group leader")
        pre_gold = records.get("pre_gold")
        if pre_gold is not None:
            pre_payload = pre_gold["payload"]
            stopped = pre_payload["server_stopped"]
            dag = pre_payload["run_dag"]
            journal = pre_payload["provider_journal_seal"]
            outputs = pre_payload["official_outputs"]
            if (
                stopped["process"] != runtime_process
                or stopped["process_group_id"]
                != runtime_topology["process_group_id"]
                or dag["attempt_id"] != attempt["attempt_id"]
                or journal["sha256"] != dag["provider_journal_sha256"]
                or outputs["journal"]["sha256"] != journal["sha256"]
                or journal["record_count"] != dag["journal_record_count"]
                or journal["final_sequence"] != dag["final_journal_sequence"]
                or journal["final_record_sha256"]
                != dag["final_journal_record_sha256"]
                or outputs["memory_manifest"]["sha256"]
                != dag["memory_manifest_sha256"]
                or outputs["baseline_manifest"]["sha256"]
                != dag["baseline_prediction_manifest_sha256"]
                or outputs["candidate_manifest"]["sha256"]
                != dag["candidate_prediction_manifest_sha256"]
            ):
                raise ValueError(
                    "pre_gold process, attempt, output, journal, or DAG binding mismatch"
                )
            recorded = {
                (identity["pid"], identity["start_ticks"])
                for identity in stopped["recorded_processes"]
            }
            for name, topology in (
                ("runtime readiness", runtime_topology),
                (
                    "post_first_provider_topology",
                    pre_payload["post_first_provider_topology"],
                ),
                (
                    "post_provider_topology",
                    pre_payload["post_provider_topology"],
                ),
            ):
                if (
                    topology["process_group_id"]
                    != runtime_topology["process_group_id"]
                    or topology["mapped_gpu_uuids"] != expected_gpu_uuids
                ):
                    raise ValueError(f"{name} differs from the runtime GPU/process group")
                descendants = {
                    (identity["pid"], identity["start_ticks"])
                    for identity in topology["descendant_processes"]
                }
                if not descendants.issubset(recorded):
                    raise ValueError(
                        f"server-stop receipt omits {name} descendants"
                    )
    result = records.get("result")
    critic = records.get("critic")
    if critic is not None:
        if (
            result is None
            or result["payload"]["execution_status"] != "COMPLETE"
            or set(records) != set(STAGE_ORDER)
        ):
            raise ValueError("critic receipt requires the complete result chain")
        critic_payload = critic["payload"]
        result_payload = result["payload"]
        critic_path = root / "critic.py"
        manifest_critic = implementation.get("files", {}).get("critic.py")
        if (
            critic_path.is_symlink()
            or not critic_path.is_file()
            or not isinstance(manifest_critic, dict)
            or set(manifest_critic) != {"sha256", "size"}
            or critic_payload["critic_source_sha256"] != sha256_file(critic_path)
            or critic_payload["critic_source_sha256"]
            != manifest_critic.get("sha256")
        ):
            raise ValueError("critic receipt source binding mismatch")
        if (
            critic_payload["result_record_sha256"] != result["record_sha256"]
            or critic_payload["implementation_manifest_sha256"]
            != attempt["binding"]["implementation_manifest_sha256"]
            or critic_payload["summary_sha256"]
            != result_payload["summary"]["sha256"]
            or critic_payload["evaluation_evidence_sha256"]
            != result_payload["evaluation_evidence"]["sha256"]
        ):
            raise ValueError("critic receipt result or implementation binding mismatch")
        if critic_payload["audit_status"] == "PASS" and (
            critic_payload["integrity_status"] != result_payload["integrity_status"]
            or critic_payload["execution_status"] != result_payload["execution_status"]
            or critic_payload["performance_status"]
            != result_payload["performance_status"]
        ):
            raise ValueError("PASS critic receipt differs from result axes")
    return records


def commit_stage_once(
    root: Path,
    attempt_id: str,
    stage: str,
    payload: dict[str, Any],
    runner_nonce: bytes | bytearray,
    *,
    clock: Callable[[], int] = time.time_ns,
) -> dict[str, Any]:
    root = root.resolve()
    _canonical_uuid(attempt_id, "attempt ID")
    attempt, _implementation, _ = validate_final_attempt_v2(root)
    if attempt["attempt_id"] != attempt_id:
        raise ValueError("stage attempt ID mismatch")
    verify_runner_nonce(attempt, runner_nonce)
    records = validate_stage_chain(root)
    attempt_record_sha256 = _attempt_record_sha256_once(root, attempt)
    if stage in records or stage_path(root, stage).exists():
        raise FileExistsError(f"{stage} receipt is already committed")
    if "result" in records and stage != "critic":
        raise ValueError("final protocol is already terminal")
    if stage == "runtime":
        if records:
            raise ValueError("runtime must be the first final-protocol stage")
    elif stage == "pre_gold":
        if list(records) != ["runtime"]:
            raise ValueError("pre_gold requires exactly one runtime predecessor")
    elif stage == "gold_open":
        if list(records) != ["runtime", "pre_gold"]:
            raise ValueError("gold_open requires runtime and pre_gold")
    elif stage == "result":
        _result_contract(payload)
        if payload["execution_status"] == "COMPLETE" and list(records) != [
            "runtime",
            "pre_gold",
            "gold_open",
        ]:
            raise ValueError("COMPLETE result requires gold_open")
    elif stage == "critic":
        if list(records) != ["runtime", "pre_gold", "gold_open", "result"]:
            raise ValueError("critic receipt requires exactly one complete result predecessor")
        if records["result"]["payload"]["execution_status"] != "COMPLETE":
            raise ValueError("critic receipt cannot follow a failed result")
        _critic_contract(payload)
    else:
        raise ValueError(f"unknown final-protocol stage: {stage}")
    _stage_payload_contract(stage, payload)
    previous_stage = next(reversed(records), "attempt")
    previous_sha256 = (
        records[previous_stage]["record_sha256"]
        if records
        else attempt_record_sha256
    )
    record = {
        "schema_version": 1,
        "kind": "final_protocol_stage",
        "stage": stage,
        "sequence": len(records) + 1,
        "attempt_id": attempt_id,
        "attempt_record_sha256": attempt_record_sha256,
        "runner_nonce_sha256": attempt["runner_nonce_sha256"],
        "previous_stage": previous_stage,
        "previous_record_sha256": previous_sha256,
        "committed_unix_ns": int(clock()),
        "payload": payload,
    }
    record["record_sha256"] = stage_record_sha256(record)
    commit_json_once(stage_path(root, stage), record)
    validate_stage_chain(root)
    return record


def classify_protocol_state(root: Path) -> dict[str, Any]:
    root = root.resolve()
    if not (root / ATTEMPT_LOCK).exists():
        from .ledger import final_paths

        orphaned = [
            str(path.relative_to(root))
            for path in [
                *(stage_path(root, stage) for stage in STAGE_ORDER),
                root / EVALUATION_EVIDENCE,
                *final_paths(root).values(),
            ]
            if path.exists() or path.is_symlink()
        ]
        if orphaned:
            return {
                "integrity": "FAIL",
                "execution": "NOT_RUN",
                "performance": "NOT_RUN",
                "attempt_id": None,
                "reason": f"orphan final artifacts without attempt: {sorted(orphaned)}",
            }
        return {
            "integrity": "PASS",
            "execution": "NOT_RUN",
            "performance": "NOT_RUN",
            "attempt_id": None,
        }
    attempt, _implementation, _ = validate_final_attempt_v2(root)
    records = validate_stage_chain(root)
    result = records.get("result")
    if result is None:
        return {
            "integrity": "PASS",
            "execution": "CONSUMED_INCOMPLETE",
            "performance": "NOT_RUN",
            "attempt_id": attempt["attempt_id"],
            "last_stage": next(reversed(records), "attempt"),
        }
    payload = result["payload"]
    if payload["execution_status"] == "COMPLETE":
        critic = records.get("critic")
        if critic is None:
            return {
                "integrity": "PASS",
                "execution": "CONSUMED_INCOMPLETE",
                "performance": "NOT_RUN",
                "attempt_id": attempt["attempt_id"],
                "last_stage": "result",
                "reason": "strict critic receipt is missing",
            }
        critic_payload = critic["payload"]
        return {
            "integrity": critic_payload["integrity_status"],
            "execution": critic_payload["execution_status"],
            "performance": critic_payload["performance_status"],
            "attempt_id": attempt["attempt_id"],
            "last_stage": "critic",
            "critic_audit": critic_payload["audit_status"],
        }
    return {
        "integrity": payload["integrity_status"],
        "execution": payload["execution_status"],
        "performance": payload["performance_status"],
        "attempt_id": attempt["attempt_id"],
        "last_stage": "result",
    }


def hash_tree(root: Path) -> str:
    """Hash a model snapshot by relative path, link target, bytes, and size."""
    root = root.resolve()
    if not root.is_dir():
        raise ValueError("model snapshot is not a directory")
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: str(item.relative_to(root))):
        relative = str(path.relative_to(root))
        if path.is_symlink():
            target = os.readlink(path)
            if not path.is_file():
                raise ValueError(f"model snapshot symlink is not a file: {relative}")
            entries.append(
                {
                    "path": relative,
                    "kind": "symlink_file",
                    "target": target,
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        elif path.is_file():
            entries.append(
                {
                    "path": relative,
                    "kind": "file",
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        elif not path.is_dir():
            raise ValueError(f"unsupported model snapshot entry: {relative}")
    if not entries:
        raise ValueError("model snapshot tree is empty")
    return hashlib.sha256(canonical_json(entries).encode("utf-8")).hexdigest()


def process_start_identity(pid: int) -> dict[str, Any]:
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        raise ValueError("process PID must be positive")
    proc = Path("/proc") / str(pid)
    stat_text = (proc / "stat").read_text(encoding="utf-8")
    close = stat_text.rfind(") ")
    if close < 0:
        raise ValueError("malformed /proc process stat")
    tail = stat_text[close + 2 :].split()
    if len(tail) < 20:
        raise ValueError("truncated /proc process stat")
    executable = (proc / "exe").resolve()
    return {
        "pid": pid,
        "start_ticks": int(tail[19]),
        "executable": str(executable),
        "executable_sha256": sha256_file(executable),
        "argv_sha256": sha256_bytes((proc / "cmdline").read_bytes()),
    }


def process_identity_is_live(identity: dict[str, Any]) -> bool:
    try:
        return process_start_identity(int(identity.get("pid", -1))) == identity
    except (OSError, ValueError):
        return False


def consume_gold_open_capability(fd: int, runtime_payload: dict[str, Any]) -> None:
    """Consume an unrecorded bearer secret after verifying the sealed parent."""
    runner = runtime_payload.get("runner")
    if not isinstance(runner, dict) or process_start_identity(os.getppid()) != runner:
        raise ValueError("gold-open capability did not originate from the sealed runner")
    secret = bytearray(read_secret_fd(fd, "gold-open capability"))
    for index in range(len(secret)):
        secret[index] = 0


def open_regular_bytes_once(path: Path) -> tuple[bytes, os.stat_result]:
    """Open once with O_NOFOLLOW and return bytes read from that same descriptor."""
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError("unable to open sealed regular file as a non-symlink") from exc
    try:
        identity = os.fstat(descriptor)
        if not stat.S_ISREG(identity.st_mode):
            raise ValueError("sealed input must be a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks), identity
    finally:
        os.close(descriptor)


def parse_json_object_bytes(payload: bytes, label: str) -> dict[str, Any]:
    """Parse one JSON object from already identity-bound bytes."""
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not a valid UTF-8 JSON object") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def load_regular_json_once(path: Path) -> dict[str, Any]:
    payload, _identity = open_regular_bytes_once(path)
    return parse_json_object_bytes(payload, f"JSON receipt {path}")


def parse_jsonl_bytes(payload: bytes, label: str) -> list[dict[str, Any]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not UTF-8") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"{label}:{line_number}: blank rows are forbidden")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label}:{line_number}: invalid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{label}:{line_number}: expected an object")
        rows.append(value)
    return rows


def command_sha256(argv: Iterable[str]) -> str:
    values = list(argv)
    if not values or any(not isinstance(value, str) or not value for value in values):
        raise ValueError("planned command argv must contain nonempty strings")
    return hashlib.sha256(canonical_json(values).encode("utf-8")).hexdigest()


def sanitized_child_environment(root: Path, gpu_indices: Iterable[int]) -> dict[str, str]:
    """Construct a credential-free, deterministic local child environment."""
    inherited = ("PATH", "HOME", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "TMPDIR")
    environment = {
        key: os.environ[key]
        for key in inherited
        if key in os.environ and os.environ[key]
    }
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": ",".join(str(index) for index in gpu_indices),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONNOUSERSITE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONPATH": str(root.resolve()),
        }
    )
    forbidden = ("AUTH", "SECRET", "PASSWORD", "API_KEY", "CREDENTIAL")
    if any(any(part in key.upper() for part in forbidden) for key in environment):
        raise AssertionError("sanitized child environment contains a credential-like key")
    return environment


def environment_receipt(environment: dict[str, str]) -> dict[str, Any]:
    ordered = {key: environment[key] for key in sorted(environment)}
    return {
        "policy": "fixed_allowlist_no_credentials",
        "values": ordered,
        "sha256": hashlib.sha256(canonical_json(ordered).encode("utf-8")).hexdigest(),
    }


def strict_success(axes: dict[str, Any]) -> bool:
    return (
        axes.get("integrity") == "PASS"
        and axes.get("execution") == "COMPLETE"
        and axes.get("performance") == "PASS"
    )


def assert_no_stage_artifacts(root: Path) -> None:
    root = root.resolve()
    existing = [
        str(stage_path(root, stage).relative_to(root))
        for stage in STAGE_ORDER
        if stage_path(root, stage).exists() or stage_path(root, stage).is_symlink()
    ]
    evidence_path = root / EVALUATION_EVIDENCE
    if evidence_path.exists() or evidence_path.is_symlink():
        existing.append(str(EVALUATION_EVIDENCE))
    if existing:
        raise FileExistsError(f"final-protocol receipt already exists: {existing}")
