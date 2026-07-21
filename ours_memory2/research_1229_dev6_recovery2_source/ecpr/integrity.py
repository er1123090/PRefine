"""Immutable influence-manifest and final-attempt validation."""

from __future__ import annotations

import hashlib
import os
import time
import unicodedata
import uuid
from pathlib import Path
from typing import Any, Callable

from .contracts import TASK_FIELDS
from .firewall import validate_sanitized_history
from .io import canonical_json, iter_jsonl, load_json, sha256_file
from .latent_firewall import (
    ACTIVE_CANDIDATE_REVISION,
    ACTIVE_VLT_CANDIDATE_MEMORY_SCOPE_CONTRACT,
    ACTIVE_VLT_CONTRACT,
    ACTIVE_VLT_SUPERSESSION_CONTRACT,
    CANDIDATE_REVISION,
    VLT1_CANDIDATE_MEMORY_SCOPE_CONTRACT,
    VLT1_CANDIDATE_REVISION,
    VLT1_CONTRACT,
    VLT1_SUPERSESSION_CONTRACT,
    VLT_CANDIDATE_MEMORY_SCOPE_CONTRACT,
    VLT_CONTRACT,
    VLT_SUPERSESSION_CONTRACT,
    VLT2_CANDIDATE_MEMORY_SCOPE_CONTRACT,
    VLT2_CANDIDATE_REVISION,
    VLT2_CONTRACT,
    VLT2_SUPERSESSION_CONTRACT,
    VLT3_CANDIDATE_REVISION,
    VLT3_LATENT_TRAIT_ONTOLOGY_CANONICAL_SHA256,
    validate_latent_trait_ontology,
    validate_latent_trait_ontology_v1,
    validate_latent_trait_ontology_v2,
    validate_latent_trait_ontology_v3,
    validate_vlt_schema_contract,
)
from .parsing import schema_domain_slots
from .prepare import build_public_tasks
from .query_gate import (
    CANDIDATE_MEMORY_SCOPE_CONTRACT,
    ROUTING_GATE_CONTRACT,
    gate_public_query,
    validate_ontology,
)


IMPLEMENTATION_MANIFEST = Path("manifests/implementation_manifest.json")
ATTEMPT_LOCK = Path("artifacts/final_test.attempt.json")
AMENDMENT = Path("PROTOCOL_AMENDMENT.json")
ROUTING_INTEGRITY_AMENDMENT = Path("ROUTING_INTEGRITY_AMENDMENT.json")
LATENT_SCOPE_AMENDMENT = Path("LATENT_SCOPE_AMENDMENT.json")
SAFE_TRANSFER_AMENDMENT = Path("SAFE_TRANSFER_AMENDMENT.json")
VLT_AUDIT_AMENDMENT = Path("VLT_AUDIT_AMENDMENT.json")
R3_RUBRIC = Path("AUTORESEARCH_R3_RUBRIC.md")
R3_METHOD_SCOPE_CLARIFICATION = Path("R3_METHOD_SCOPE_CLARIFICATION.json")
R3_ACTION_PROMPT_AMENDMENT = Path("R3_ACTION_PROMPT_AMENDMENT.json")
VLT3_AUDIT_AMENDMENT = Path("VLT3_AUDIT_AMENDMENT.json")
TARGET_FREE_VLT3_DIAGNOSTICS = Path("target_free_vlt3_diagnostics.py")
TARGET_FREE_VLT3_REPORT = Path("reports/TARGET_FREE_VLT3_DIAGNOSTICS.json")
RUNTIME_RECOVERY_AMENDMENT = Path("R4_RUNTIME_RECOVERY_AMENDMENT.json")
PID_NAMESPACE_TOPOLOGY_DIAGNOSTIC = Path(
    "reports/PID_NAMESPACE_TOPOLOGY_DIAGNOSTIC.json"
)
PARENT_IMPLEMENTATION_MANIFEST = Path(
    "manifests/parent_implementation_manifest.json"
)
PUBLIC_DOMAIN_ONTOLOGY = Path("configs/public_domain_ontology.json")
LATENT_TRAIT_ONTOLOGY_V1 = Path("configs/latent_trait_ontology.json")
LATENT_TRAIT_ONTOLOGY_V2 = Path("configs/latent_trait_ontology.vlt2.json")
LATENT_TRAIT_ONTOLOGY_V3 = Path("configs/latent_trait_ontology.vlt3.json")
LATENT_TRAIT_ONTOLOGY = LATENT_TRAIT_ONTOLOGY_V3
PQR_ROUTING_COVERAGE = Path("reports/PQR_ROUTING_COVERAGE.json")
CANDIDATE_REGISTRATION = Path("CANDIDATE_REGISTRATION.json")
HISTORICAL_EXPECTED_RUNTIME_CONTRACT = Path(
    "manifests/expected_runtime_contract.json"
)
EXPECTED_RUNTIME_CONTRACT_V1 = Path(
    "manifests/expected_runtime_contract.vlt1.json"
)
EXPECTED_RUNTIME_CONTRACT_V2 = Path(
    "manifests/expected_runtime_contract.vlt2.json"
)
EXPECTED_RUNTIME_CONTRACT_V3 = Path(
    "manifests/expected_runtime_contract.vlt3.json"
)
EXPECTED_RUNTIME_CONTRACT = EXPECTED_RUNTIME_CONTRACT_V3

HISTORICAL_EXPECTED_RUNTIME_CONTRACT_SHA256 = (
    "b5a0a2327401913063b3fb76b95ff7113bb733bbafd9e4462cba2a704c7b840f"
)
VLT1_EXPECTED_RUNTIME_CONTRACT_SHA256 = (
    "effbdec6656f927c2b4c3ba009cb97556debdd22fa1724757ed25f9181c13c2d"
)
VLT1_SAFE_TRANSFER_AMENDMENT_SHA256 = (
    "2c5f29e58818d3a0d4d7fec6c94decfc049bf1d0ad825a456d5c0995468ea9d7"
)
VLT1_LATENT_TRAIT_ONTOLOGY_FILE_SHA256 = (
    "1d9ffdf29a2af37367bbd63033c927473789e6eddb71926999bb88e56bb198a1"
)
ROUTING_INTEGRITY_AMENDMENT_SHA256 = (
    "cf3282c515a5d65c159fa15b1b3aea5709c205984e9aa2c1d4b04d92f6fcde0a"
)
LATENT_SCOPE_AMENDMENT_SHA256 = (
    "40ff3655dc6ca0be53eb40e7fc44b7f9aa9f593d237eb2e6de7c2a09cca9aec2"
)
VLT2_LATENT_TRAIT_ONTOLOGY_FILE_SHA256 = (
    "be2070439b5928d6f2b1468ea0b8b63c0a42f48b578b4755a6ef949880fe005b"
)
R3_RUBRIC_SHA256 = (
    "c317887ddd2b9ce34266b5aff8eb2571a4e7bcbd66ed5be096f217d4f14a2dba"
)
R3_METHOD_SCOPE_CLARIFICATION_SHA256 = (
    "a5e582293faa22c7e5b1496362a4e0b9c61f03ce66b2aaaa1e2db05a73be09cc"
)
VLT3_LATENT_TRAIT_ONTOLOGY_FILE_SHA256 = (
    "652febe52842c3ac9d4b5fb4bb1f4e81a8a0c3072610e9717513eca121efc0d3"
)
RESEARCH_ACTION_PROMPTS_SHA256 = (
    "2e76418a980d251e3eef467ab67793b977a631f8b92cba9481b333b4f0673f57"
)
LOCAL_PAPER_SHA256 = (
    "58de92a7746f671917a4faef5301efa02f05b440888f27845c34d40db5c3d6d8"
)
R3_ACTION_PROMPT_AMENDMENT_SHA256 = (
    "a73d15836e20a8e56f87c2c93e1eacf8408cb12b9b6160fe303612efbcd59e98"
)
VLT3_AUDIT_AMENDMENT_SHA256 = (
    "236dc25599924efc94c29297e2444760790070155f8629c6d6915d9f30f68852"
)
VLT3_EXPECTED_RUNTIME_CONTRACT_SHA256 = (
    "5de17cca8ae3649a3f4fc8e2da46a6615906e6896b5436652b4eeb6d69450855"
)
TARGET_FREE_VLT3_REPORT_SHA256 = (
    "d32f7f8e85bf043e721004531190f8f506878fd1bed28111c3138f4694b2c7bc"
)
ORIGINAL_SOURCE_PROMPT_SHA256 = (
    "2a7e69a3ab13b40727793de7b935b4643c14069983f5bc83900e64707d3ce411"
)
ORIGINAL_PREREGISTRATION_SHA256 = (
    "2ec42394f9157fa0fe690195dc3010afc81c2bb117d0e94a9dfeef0535b5dae0"
)
PARENT_IMPLEMENTATION_MANIFEST_SHA256 = (
    "04a479a708d362f812a5c4467c9669915d6c2f74de227c3f36abb44c3b5f47b2"
)
PARENT_IMPLEMENTATION_INVENTORY_SHA256 = (
    "da46c33fbd3320039b001fdbb8d86a4700019471e21aa0f97d268e9f0e14d8de"
)
PID_NAMESPACE_TOPOLOGY_DIAGNOSTIC_SHA256 = (
    "9f4a396e0cc802e1d32e59a90ed7bdd71d2eb6671cc37b333af2df8a9a0fb777"
)
RUNTIME_RECOVERY_CHANGED_PARENT_PATHS = frozenset(
    {"run_gpu.py", "ecpr/final_protocol.py", "ecpr/integrity.py"}
)

VLT1_IMPLEMENTATION_HASHES = {
    "ecpr/contracts.py": "c29b2f0259fbe13133a226b3bf7f8235c81b4ce6c42523ef490a7d988f018365",
    "ecpr/inference.py": "e9cec664a6909d6920d6ee99a2515c1df906d653b950852fb9e8a8baca3cf13a",
    "ecpr/latent_firewall.py": "5322bb572559b4ce1d4388369cb6951d915b5f548d1913000c40e21c4a34ba77",
    "ecpr/ledger.py": "d720001d211d9c67b250264d2f3e47d717a4644a16bd3e6485546173606f0a55",
    "ecpr/memory.py": "f68aa9aba1102b0e96e235e34bd1e654e54259cf9aadbe2ec285cd94e2758bcc",
    "ecpr/prompts.py": "b85bbfa1839b0ef35ff4c31b8b30df76375d633f7160c08172ceaec30edc0e8f",
    "ecpr/router.py": "0d384b55acb7daabddcc220f0e0abcf30d1096f8690ab968866dae027894301f",
}
VLT_IMPLEMENTATION_PATHS = (
    "critic.py",
    "ecpr/action_case.py",
    "ecpr/cli.py",
    "ecpr/contracts.py",
    "ecpr/inference.py",
    "ecpr/io.py",
    "ecpr/latent_firewall.py",
    "ecpr/ledger.py",
    "ecpr/memory.py",
    "ecpr/parsing.py",
    "ecpr/prefine.py",
    "ecpr/prompts.py",
    "ecpr/provider.py",
    "ecpr/query_gate.py",
    "ecpr/replay.py",
    "ecpr/request_contract.py",
    "ecpr/router.py",
)
VLT2_ROUTING_GATE_CONTRACT = {
    **ROUTING_GATE_CONTRACT,
    "latent_policy": (
        "ABSTAIN always omits; SELECT may serialize authorized closed VLT "
        "even with zero target-routed typed evidence"
    ),
}
VLT2_AUDIT_GUARANTEES = {
    "provider_request": "one immutable CallSpec supplies wire bytes and audit fields",
    "memory_replay": "live and replay share one PReFine callback state machine",
    "authority": "only official journal+slice+attempt+manifest validation mints authority",
    "action_replay": "live and ledger share one action-case builder",
    "prompt_lock": "both arms share one prompt skeleton and only memory block differs",
    "audit_payload": "hash-only decision and semantic-replay artifacts contain no raw latent",
}
R3_ACTION_PROMPT_CONTRACT = {
    "mode_schema_pairs": {
        "singleturn": "single",
        "multiturn": "multi",
    },
    "mode_schema_mismatch": "fail_closed_before_provider_call",
    "template_authority": "source_local_detailed_SINGLE_and_MULTI_prefine_templates",
    "same_mode_arm_isolation": (
        "baseline_and_candidate_prompt_bytes_are_identical_except_exactly_one_"
        "serialized_memory_block"
    ),
    "schema_serialization": "ensure_ascii_false_sort_keys_true_indent_2",
    "precedence": (
        "current_user_dialogue_and_schema_override_memory;memory_fills_only_"
        "otherwise_missing_preference_slots"
    ),
    "transient_and_replay_policy": (
        "never_replay_full_historical_call_or_copy_transient_date_time_location_"
        "address_name_identifier_or_itinerary_values"
    ),
    "output": "exactly_one_FunctionName_slot_value_call_without_wrapper_or_explanation",
    "post_freeze_prompt_variants_allowed": 0,
}
VLT3_ROUTING_GATE_CONTRACT = VLT2_ROUTING_GATE_CONTRACT
VLT3_AUDIT_GUARANTEES = {
    "append_only_lineage": (
        "frozen_v1_v2_files_remain_byte_identical_and_no_missing_v2_"
        "amendment_or_runtime_is_retroactively_created"
    ),
    "source_authority": (
        "paper_r3_scope_standalone_prompt_and_source_local_prompt_hashes_are_bound"
    ),
    "target_free_diagnostics": (
        "aggregate_sanitized_history_only_no_query_gold_prediction_metric_or_model_call"
    ),
    "typed_wire_support": "exact_JSON_scalar_source_literal_type_and_value",
    "prompt_projection": "value_type_and_source_literal_never_enter_action_prompt",
    "provider_request": "one_immutable_CallSpec_supplies_wire_bytes_and_audit_fields",
    "one_shot_final": (
        "nonce_capabilities_commit_once_stage_chain_same_fd_inputs_and_strict_"
        "critic_receipt_required"
    ),
    "dual_metric_boundary": (
        "same_process_shared_parser_but_separate_metric_statistics_and_gate_implementations"
    ),
}
VLT3_ONE_SHOT_PROTOCOL = {
    "stage_order": ["runtime", "pre_gold", "gold_open", "result", "critic"],
    "strict_success": "hash_bound_PASS_critic_receipt_required",
    "failure_integrity_status": "FAIL",
    "same_fd_inputs": [
        "predictions",
        "gold",
        "summary",
        "evaluation_evidence",
        "provider_journal",
    ],
    "sealed_command_timeout_seconds": {
        "memory": 172800,
        "baseline": 43200,
        "candidate": 43200,
        "evaluate": 1800,
        "critic": 1800,
    },
}

EXPECTED_ECPR_PYTHON = frozenset(
    {
        "__init__.py",
        "__main__.py",
        "action_case.py",
        "cli.py",
        "contracts.py",
        "evaluate.py",
        "firewall.py",
        "final_protocol.py",
        "inference.py",
        "independent_audit.py",
        "integrity.py",
        "ledger.py",
        "io.py",
        "latent_firewall.py",
        "memory.py",
        "parsing.py",
        "prefine.py",
        "prepare.py",
        "query_gate.py",
        "prompts.py",
        "provider.py",
        "replay.py",
        "request_contract.py",
        "router.py",
        "statistics.py",
    }
)
EXPECTED_ROOT_PYTHON = frozenset(
    {"critic.py", "run_gpu.py", "target_free_vlt3_diagnostics.py"}
)
EXPECTED_CONFIG_FILES = frozenset(
    {
        "action_contract.json",
        "latent_trait_ontology.json",
        "latent_trait_ontology.vlt2.json",
        "latent_trait_ontology.vlt3.json",
        "preference_groups.json",
        "preference_slots.json",
        "public_domain_ontology.json",
        "query_multiturn-domain.json",
        "query_singleturn.json",
        "schema_multi.json",
        "schema_single.json",
    }
)
VLT3_IMPLEMENTATION_PATHS = tuple(
    sorted(
        {
            *(f"ecpr/{name}" for name in EXPECTED_ECPR_PYTHON if name != "integrity.py"),
            *EXPECTED_ROOT_PYTHON,
        }
    )
)
EXPECTED_EXTERNAL_INPUTS = frozenset(
    {
        "history_data",
        "single_query",
        "single_schema",
        "multi_query",
        "multi_schema",
        "preference_slots",
        "preference_groups",
    }
)
EXPECTED_PIPELINE_STAGES = [
    "serve_model",
    "build_sealed_memory",
    "infer_baseline",
    "infer_candidate",
    "evaluate_sealed",
    "critic_strict",
]
TERMINAL_OUTCOMES = frozenset(
    {
        "pipeline_complete",
        "pipeline_failed",
        "pipeline_exception",
        "post_lock_validation_failed",
    }
)


def _digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _vlt_implementation_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in VLT_IMPLEMENTATION_PATHS:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"VLT implementation file is missing: {relative}")
        hashes[relative] = sha256_file(path)
    return hashes


def _vlt3_implementation_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in VLT3_IMPLEMENTATION_PATHS:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"VLT3 implementation file is missing: {relative}")
        hashes[relative] = sha256_file(path)
    return hashes


def _validate_runtime_recovery_chain(
    root: Path,
    audit: dict[str, Any],
    current_implementation_hashes: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    parent_path = root / PARENT_IMPLEMENTATION_MANIFEST
    diagnostic_path = root / PID_NAMESPACE_TOPOLOGY_DIAGNOSTIC
    recovery_path = root / RUNTIME_RECOVERY_AMENDMENT
    for label, path in (
        ("parent implementation manifest", parent_path),
        ("PID namespace diagnostic", diagnostic_path),
        ("runtime recovery amendment", recovery_path),
    ):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{label} must be a regular file")
    if sha256_file(parent_path) != PARENT_IMPLEMENTATION_MANIFEST_SHA256:
        raise ValueError("parent implementation manifest hash changed")
    if sha256_file(diagnostic_path) != PID_NAMESPACE_TOPOLOGY_DIAGNOSTIC_SHA256:
        raise ValueError("PID namespace diagnostic hash changed")

    parent = load_json(parent_path)
    parent_files = parent.get("files")
    if (
        parent.get("schema_version") != 3
        or parent.get("kind") != "immutable_influence_manifest"
        or parent.get("candidate_revision") != VLT3_CANDIDATE_REVISION
        or parent.get("inventory_sha256")
        != PARENT_IMPLEMENTATION_INVENTORY_SHA256
        or not isinstance(parent_files, dict)
    ):
        raise ValueError("parent implementation manifest contract mismatch")
    for relative, receipt in parent_files.items():
        if not isinstance(receipt, dict) or set(receipt) != {"sha256", "size"}:
            raise ValueError("parent implementation file receipt is malformed")
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"parent influential file is missing: {relative}")
        if relative not in RUNTIME_RECOVERY_CHANGED_PARENT_PATHS and (
            sha256_file(path) != receipt["sha256"]
            or path.stat().st_size != receipt["size"]
        ):
            raise ValueError(f"non-runtime parent influence changed: {relative}")

    parent_vlt3_hashes = {
        relative: parent_files[relative]["sha256"]
        for relative in VLT3_IMPLEMENTATION_PATHS
    }
    if audit.get("implementation_sha256") != parent_vlt3_hashes:
        raise ValueError("VLT3 audit no longer binds the parent implementation")
    allowed_vlt3_runtime_changes = {"run_gpu.py", "ecpr/final_protocol.py"}
    for relative, digest in current_implementation_hashes.items():
        if relative not in allowed_vlt3_runtime_changes and (
            digest != parent_vlt3_hashes[relative]
        ):
            raise ValueError(f"candidate method file changed in recovery: {relative}")

    diagnostic = load_json(diagnostic_path)
    expected_failed_attempt = {
        "attempt_id": "34615be6-82ab-4f17-aa7f-5522a24a8a96",
        "implementation_manifest_sha256": PARENT_IMPLEMENTATION_MANIFEST_SHA256,
        "attempt_receipt_sha256": (
            "e169e00a359b2d5d7a39deed9553888e70b1111da0fa74eddf201e33c5cfc72a"
        ),
        "result_receipt_sha256": (
            "f33488d29d784a4b969ab1d26b7aeba6e06c258cb71797e35a7c68822d6c97d5"
        ),
        "vllm_log_sha256": (
            "9ccb7a10ec4b58d0cf3ef9ef2a356ce7eb91a47e18a37824790d40c050528b68"
        ),
        "v1_models_get_count": 1,
        "completion_post_count": 0,
        "provider_journal_created": False,
        "gold_open_count": 0,
        "target_metric_count": 0,
    }
    expected_recovery_contract = {
        "ownership_contract": (
            "exclusive_empty_baseline_then_exact_four_uuid_contexts"
        ),
        "pid_namespace_contract": (
            "host_compute_pids_are_opaque_and_never_compared_to_container_pids"
        ),
        "pre_attempt_requirement": "all nvidia compute-app rows empty",
        "post_attempt_prelaunch_requirement": (
            "all nvidia compute-app rows still empty"
        ),
        "readiness_requirement": (
            "exactly four unique positive-memory rows matching the sealed UUID "
            "and PCI pairs"
        ),
        "continuity_requirement": (
            "host PID UUID PCI identities remain fixed after first provider "
            "stage and after all provider stages"
        ),
        "container_lifecycle_requirement": (
            "procfs descendant and process-group identities remain independently "
            "responsible for exact shutdown"
        ),
    }
    if (
        set(diagnostic)
        != {
            "schema_version",
            "kind",
            "status",
            "failed_attempt",
            "diagnostic_evidence",
            "finding",
            "recovery_contract",
            "forbidden_sources_not_used",
        }
        or diagnostic.get("schema_version") != 1
        or diagnostic.get("kind")
        != "target_free_pid_namespace_gpu_topology_diagnostic"
        or diagnostic.get("status")
        != "complete_no_inference_no_gold_open_no_target_metric"
        or diagnostic.get("failed_attempt") != expected_failed_attempt
        or diagnostic.get("recovery_contract") != expected_recovery_contract
        or diagnostic.get("diagnostic_evidence", {}).get(
            "model_completion_requests"
        )
        != 0
        or diagnostic.get("diagnostic_evidence", {}).get("gold_opens") != 0
        or diagnostic.get("diagnostic_evidence", {}).get("target_metrics") != 0
        or diagnostic.get("finding", {}).get(
            "host_and_container_pid_sets_are_disjoint"
        )
        is not True
    ):
        raise ValueError("PID namespace diagnostic contract mismatch")

    recovery = load_json(recovery_path)
    if set(recovery) != {
        "schema_version",
        "kind",
        "status",
        "candidate_id",
        "candidate_revision",
        "parent_implementation_manifest_sha256",
        "parent_inventory_sha256",
        "failed_attempt",
        "diagnostic_report_sha256",
        "changed_files",
        "unchanged_influence_contract",
        "runtime_recovery_contract",
        "prior_target_model_calls",
        "prior_gold_opens",
        "prior_target_metrics_computed",
    }:
        raise ValueError("runtime recovery amendment field contract violation")
    expected_unchanged = {
        "parent_manifest_files_checked": True,
        "allowed_changed_parent_files": sorted(
            RUNTIME_RECOVERY_CHANGED_PARENT_PATHS
        ),
        "method_prompt_query_task_history_and_evaluator_contract_bytes_unchanged": True,
        "candidate_id_and_revision_unchanged": True,
        "gold_file_not_opened": True,
    }
    expected_scopes = {
        "run_gpu.py": "exclusive_gpu_lease_and_pid_namespace_runtime_telemetry_only",
        "ecpr/final_protocol.py": "exclusive_gpu_lease_receipt_validation_only",
        "ecpr/integrity.py": "append_only_recovery_lineage_validation_only",
    }
    expected_changed_files = {
        relative: {
            "before_sha256": parent_files[relative]["sha256"],
            "after_sha256": sha256_file(root / relative),
            "scope": expected_scopes[relative],
        }
        for relative in sorted(RUNTIME_RECOVERY_CHANGED_PARENT_PATHS)
    }
    if (
        recovery.get("schema_version") != 1
        or recovery.get("kind")
        != "append_only_pid_namespace_runtime_recovery_amendment"
        or recovery.get("status")
        != "immutable_after_failed_pre_inference_attempt_before_any_target_call_gold_open_or_metric"
        or recovery.get("candidate_id") != "ecpr_v1"
        or recovery.get("candidate_revision") != VLT3_CANDIDATE_REVISION
        or recovery.get("parent_implementation_manifest_sha256")
        != PARENT_IMPLEMENTATION_MANIFEST_SHA256
        or recovery.get("parent_inventory_sha256")
        != PARENT_IMPLEMENTATION_INVENTORY_SHA256
        or recovery.get("failed_attempt") != expected_failed_attempt
        or recovery.get("diagnostic_report_sha256")
        != PID_NAMESPACE_TOPOLOGY_DIAGNOSTIC_SHA256
        or recovery.get("changed_files") != expected_changed_files
        or recovery.get("unchanged_influence_contract") != expected_unchanged
        or recovery.get("runtime_recovery_contract")
        != expected_recovery_contract
        or recovery.get("prior_target_model_calls") != 0
        or recovery.get("prior_gold_opens") != 0
        or recovery.get("prior_target_metrics_computed") != 0
    ):
        raise ValueError("runtime recovery amendment contract mismatch")
    return parent, recovery


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _normalized_identifier(value: Any, label: str) -> tuple[str, str]:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    canonical = unicodedata.normalize("NFKC", value).strip()
    if not canonical or canonical != value:
        raise ValueError(f"{label} must be nonempty and canonically normalized")
    return canonical, canonical.casefold()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def commit_json_once(path: Path, value: dict[str, Any]) -> None:
    """Atomically publish canonical JSON once and durably fsync the file and directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"refusing to replace symlink: {path}")
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(canonical_json(value) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
            _fsync_directory(path.parent)
        except FileNotFoundError:
            pass


def validate_preference_slots(root: Path) -> dict[str, list[str]]:
    root = root.resolve()
    slots = load_json(root / "configs/preference_slots.json")
    if not isinstance(slots, dict) or not slots:
        raise ValueError("preference_slots must be a nonempty object")

    schema_domain_names: dict[str, str] = {}
    schema_slot_names: dict[str, dict[str, str]] = {}
    for name in ("schema_single.json", "schema_multi.json"):
        schema = load_json(root / "configs" / name)
        for raw_domain, raw_slots in schema_domain_slots(schema).items():
            domain, normalized_domain = _normalized_identifier(
                raw_domain, f"{name} domain"
            )
            prior_domain = schema_domain_names.setdefault(normalized_domain, domain)
            if prior_domain != domain:
                raise ValueError(
                    f"normalized schema-domain collision: {prior_domain!r}/{domain!r}"
                )
            domain_slots = schema_slot_names.setdefault(normalized_domain, {})
            for raw_slot in raw_slots:
                slot, normalized_slot = _normalized_identifier(
                    raw_slot, f"{name}/{domain} slot"
                )
                prior_slot = domain_slots.setdefault(normalized_slot, slot)
                if prior_slot != slot:
                    raise ValueError(
                        f"normalized schema-slot collision: {domain}/{prior_slot!r}/{slot!r}"
                    )

    normalized: dict[str, list[str]] = {}
    seen_domains: dict[str, str] = {}
    for raw_domain, values in slots.items():
        domain, normalized_domain = _normalized_identifier(
            raw_domain, "preference domain"
        )
        prior_domain = seen_domains.setdefault(normalized_domain, domain)
        if prior_domain != domain:
            raise ValueError(
                f"normalized preference-domain collision: {prior_domain!r}/{domain!r}"
            )
        if not isinstance(values, list) or not values:
            raise ValueError("each preference domain and slot list must be nonempty and typed")
        if normalized_domain not in schema_domain_names:
            raise ValueError(
                f"preference domain is not contained in frozen schemas: {domain}"
            )
        if schema_domain_names[normalized_domain] != domain:
            raise ValueError(f"preference domain spelling is not schema-canonical: {domain}")

        seen_slots: dict[str, str] = {}
        validated_slots: list[str] = []
        for raw_slot in values:
            slot, normalized_slot = _normalized_identifier(
                raw_slot, f"preference slot for {domain}"
            )
            prior_slot = seen_slots.setdefault(normalized_slot, slot)
            if prior_slot != slot or normalized_slot in {
                value.casefold() for value in validated_slots
            }:
                raise ValueError(f"duplicate normalized preference slot for {domain}")
            schema_slots = schema_slot_names[normalized_domain]
            if normalized_slot not in schema_slots:
                raise ValueError(
                    f"preference slot is not contained in frozen schemas: {domain}/{slot}"
                )
            if schema_slots[normalized_slot] != slot:
                raise ValueError(
                    f"preference slot spelling is not schema-canonical: {domain}/{slot}"
                )
            validated_slots.append(slot)
        normalized[domain] = validated_slots
    return {domain: normalized[domain] for domain in sorted(normalized)}


def _validate_python_inventory(root: Path) -> None:
    actual_ecpr = {
        path.name
        for path in (root / "ecpr").glob("*.py")
        if path.is_file() or path.is_symlink()
    }
    if actual_ecpr != EXPECTED_ECPR_PYTHON:
        raise ValueError(
            "unexpected or missing ecpr Python source: "
            f"missing={sorted(EXPECTED_ECPR_PYTHON - actual_ecpr)} "
            f"unexpected={sorted(actual_ecpr - EXPECTED_ECPR_PYTHON)}"
        )
    actual_root = {
        path.name
        for path in root.glob("*.py")
        if path.is_file() or path.is_symlink()
    }
    if actual_root != EXPECTED_ROOT_PYTHON:
        raise ValueError(
            "unexpected or missing root Python source: "
            f"missing={sorted(EXPECTED_ROOT_PYTHON - actual_root)} "
            f"unexpected={sorted(actual_root - EXPECTED_ROOT_PYTHON)}"
        )
    expected_relative = {
        *(Path("ecpr") / name for name in EXPECTED_ECPR_PYTHON),
        *(Path(name) for name in EXPECTED_ROOT_PYTHON),
    }
    unexpected_nested: list[str] = []
    for path in root.rglob("*.py"):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts or relative.parts[0] == "tests":
            continue
        if relative not in expected_relative:
            unexpected_nested.append(str(relative))
    if unexpected_nested:
        raise ValueError(
            f"unexpected scoped Python source: {sorted(unexpected_nested)}"
        )


def _validate_config_inventory(root: Path) -> None:
    actual = {
        path.name
        for path in (root / "configs").glob("*.json")
        if path.is_file() or path.is_symlink()
    }
    if actual != EXPECTED_CONFIG_FILES:
        raise ValueError(
            "unexpected or missing frozen config: "
            f"missing={sorted(EXPECTED_CONFIG_FILES - actual)} "
            f"unexpected={sorted(actual - EXPECTED_CONFIG_FILES)}"
        )


def influential_paths(root: Path) -> list[Path]:
    root = root.resolve()
    _validate_python_inventory(root)
    _validate_config_inventory(root)
    relatives = [
        *(Path("ecpr") / name for name in sorted(EXPECTED_ECPR_PYTHON)),
        *(Path(name) for name in sorted(EXPECTED_ROOT_PYTHON)),
        Path("PREREGISTRATION.md"),
        Path("preregistration.json"),
        AMENDMENT,
        CANDIDATE_REGISTRATION,
        *(Path("configs") / name for name in sorted(EXPECTED_CONFIG_FILES)),
        ROUTING_INTEGRITY_AMENDMENT,
        LATENT_SCOPE_AMENDMENT,
        HISTORICAL_EXPECTED_RUNTIME_CONTRACT,
        SAFE_TRANSFER_AMENDMENT,
        EXPECTED_RUNTIME_CONTRACT_V1,
        Path("artifacts/history.sanitized.jsonl"),
        Path("artifacts/tasks.jsonl"),
        Path("evaluator_vault/sealed_manifest.json"),
        Path("manifests/environment.json"),
        Path("manifests/singleturn.bundle.json"),
        Path("manifests/multiturn.bundle.json"),
    ]
    if (root / VLT3_AUDIT_AMENDMENT).is_file():
        relatives.extend(
            [
                R3_RUBRIC,
                R3_METHOD_SCOPE_CLARIFICATION,
                R3_ACTION_PROMPT_AMENDMENT,
                VLT3_AUDIT_AMENDMENT,
                TARGET_FREE_VLT3_REPORT,
                EXPECTED_RUNTIME_CONTRACT_V3,
            ]
        )
        if (root / RUNTIME_RECOVERY_AMENDMENT).is_file():
            relatives.extend(
                [
                    RUNTIME_RECOVERY_AMENDMENT,
                    PID_NAMESPACE_TOPOLOGY_DIAGNOSTIC,
                    PARENT_IMPLEMENTATION_MANIFEST,
                ]
            )
    else:
        relatives.extend(
            [
                VLT_AUDIT_AMENDMENT,
                EXPECTED_RUNTIME_CONTRACT_V2,
            ]
        )
    paths: list[Path] = []
    for relative in relatives:
        if relative in {IMPLEMENTATION_MANIFEST, ATTEMPT_LOCK}:
            raise ValueError("self-referential mutable file in influence inventory")
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"influential regular file missing: {relative}")
        paths.append(path)
    if len(paths) != len(set(relatives)):
        raise ValueError("duplicate influential path")
    return paths


def _validate_registration_chain_v2(
    root: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    preregistration_path = root / "preregistration.json"
    preregistration = load_json(preregistration_path)
    preregistration_sha256 = sha256_file(preregistration_path)
    if preregistration_sha256 != ORIGINAL_PREREGISTRATION_SHA256:
        raise ValueError("original preregistration hash changed")
    if preregistration.get("status") != "locked_before_target_metrics":
        raise ValueError("original preregistration is not locked")
    if preregistration.get("candidate") != "ecpr_v1":
        raise ValueError("original preregistration candidate mismatch")

    amendment_path = root / AMENDMENT
    amendment = load_json(amendment_path)
    if amendment.get("status") != "immutable_before_any_target_metric_or_model_call":
        raise ValueError("protocol amendment is not immutable")
    if amendment.get("original_preregistration_sha256") != preregistration_sha256:
        raise ValueError("protocol amendment does not bind original preregistration")
    if amendment.get("confirmatory_candidate") != "ecpr_v1":
        raise ValueError("protocol amendment candidate mismatch")
    if amendment.get("confirmatory_candidate_count") != 1:
        raise ValueError("protocol amendment must register exactly one candidate")
    if amendment.get("development_candidates_evaluated") != 0:
        raise ValueError("protocol amendment reports prior candidate evaluation")
    if amendment.get("development_stopping_rule_invoked") is not False:
        raise ValueError("protocol amendment reports development stopping")
    if amendment.get("final_attempts_allowed") != 1:
        raise ValueError("protocol amendment final-attempt count mismatch")

    registration_path = root / CANDIDATE_REGISTRATION
    registration = load_json(registration_path)
    expected_registration_fields = {
        "schema_version",
        "kind",
        "status",
        "candidate_id",
        "candidate_count",
        "confirmatory_ablation_flags",
        "original_preregistration_sha256",
        "protocol_amendment_sha256",
    }
    if set(registration) != expected_registration_fields:
        raise ValueError("candidate registration field contract violation")
    if registration.get("kind") != "confirmatory_candidate_registration":
        raise ValueError("candidate registration kind mismatch")
    if registration.get("status") != "immutable_before_any_target_model_or_metric_call":
        raise ValueError("candidate registration is not immutable")
    if registration.get("candidate_id") != "ecpr_v1":
        raise ValueError("candidate registration ID mismatch")
    if registration.get("candidate_count") != 1:
        raise ValueError("candidate registration must contain exactly one candidate")
    if registration.get("confirmatory_ablation_flags") != []:
        raise ValueError("confirmatory candidate must have no ablations")
    if registration.get("original_preregistration_sha256") != preregistration_sha256:
        raise ValueError("candidate registration preregistration binding mismatch")
    if registration.get("protocol_amendment_sha256") != sha256_file(amendment_path):
        raise ValueError("candidate registration amendment binding mismatch")
    routing_path = root / ROUTING_INTEGRITY_AMENDMENT
    routing = load_json(routing_path)
    expected_routing_fields = {
        "schema_version",
        "kind",
        "status",
        "candidate_id",
        "original_preregistration_sha256",
        "protocol_amendment_sha256",
        "candidate_registration_sha256",
        "public_domain_ontology_sha256",
        "routing_gate",
        "prior_target_model_calls",
        "prior_target_metrics_computed",
    }
    if not isinstance(routing, dict) or set(routing) != expected_routing_fields:
        raise ValueError("routing integrity amendment field contract violation")
    if routing.get("schema_version") != 1:
        raise ValueError("routing integrity amendment schema mismatch")
    if routing.get("kind") != "public_query_relevance_gate_integrity_amendment":
        raise ValueError("routing integrity amendment kind mismatch")
    if routing.get("status") != "immutable_before_any_target_model_or_metric_call":
        raise ValueError("routing integrity amendment is not immutable")
    if routing.get("candidate_id") != "ecpr_v1":
        raise ValueError("routing integrity amendment candidate mismatch")
    if routing.get("original_preregistration_sha256") != preregistration_sha256:
        raise ValueError("routing amendment preregistration binding mismatch")
    if routing.get("protocol_amendment_sha256") != sha256_file(amendment_path):
        raise ValueError("routing amendment protocol binding mismatch")
    if routing.get("candidate_registration_sha256") != sha256_file(registration_path):
        raise ValueError("routing amendment registration binding mismatch")
    ontology_path = root / PUBLIC_DOMAIN_ONTOLOGY
    if routing.get("public_domain_ontology_sha256") != sha256_file(ontology_path):
        raise ValueError("routing amendment ontology binding mismatch")
    if routing.get("routing_gate") != ROUTING_GATE_CONTRACT:
        raise ValueError("routing amendment algorithm contract mismatch")
    if routing.get("prior_target_model_calls") != 0:
        raise ValueError("routing amendment reports prior target model calls")
    if routing.get("prior_target_metrics_computed") != 0:
        raise ValueError("routing amendment reports prior target metrics")
    latent_scope_path = root / LATENT_SCOPE_AMENDMENT
    latent_scope = load_json(latent_scope_path)
    expected_latent_scope_fields = {
        "schema_version",
        "kind",
        "status",
        "candidate_id",
        "parent_routing_integrity_amendment_sha256",
        "candidate_memory_scope",
        "trigger",
        "prior_target_model_calls",
        "prior_target_metrics_computed",
    }
    if (
        not isinstance(latent_scope, dict)
        or set(latent_scope) != expected_latent_scope_fields
    ):
        raise ValueError("latent scope amendment field contract violation")
    if latent_scope.get("schema_version") != 1:
        raise ValueError("latent scope amendment schema mismatch")
    if latent_scope.get("kind") != "candidate_typed_memory_scope_integrity_amendment":
        raise ValueError("latent scope amendment kind mismatch")
    if latent_scope.get("status") != "immutable_before_any_target_model_or_metric_call":
        raise ValueError("latent scope amendment is not immutable")
    if latent_scope.get("candidate_id") != "ecpr_v1":
        raise ValueError("latent scope amendment candidate mismatch")
    if latent_scope.get("parent_routing_integrity_amendment_sha256") != sha256_file(routing_path):
        raise ValueError("latent scope amendment routing-parent binding mismatch")
    if latent_scope.get("candidate_memory_scope") != CANDIDATE_MEMORY_SCOPE_CONTRACT:
        raise ValueError("latent scope amendment candidate-memory contract mismatch")
    if latent_scope.get("trigger") != (
        "static_cross_domain_latent_leakage_audit_before_any_target_model_or_metric_call"
    ):
        raise ValueError("latent scope amendment audit trigger mismatch")
    if latent_scope.get("prior_target_model_calls") != 0:
        raise ValueError("latent scope amendment reports prior target model calls")
    if latent_scope.get("prior_target_metrics_computed") != 0:
        raise ValueError("latent scope amendment reports prior target metrics")
    ontology = load_json(ontology_path)
    schema_union = load_json(root / "configs/schema_single.json") + load_json(
        root / "configs/schema_multi.json"
    )
    validate_ontology(ontology, schema_union)

    if sha256_file(routing_path) != ROUTING_INTEGRITY_AMENDMENT_SHA256:
        raise ValueError("frozen routing integrity amendment hash changed")
    if sha256_file(latent_scope_path) != LATENT_SCOPE_AMENDMENT_SHA256:
        raise ValueError("frozen latent-scope amendment hash changed")

    historical_runtime_path = root / HISTORICAL_EXPECTED_RUNTIME_CONTRACT
    if (
        sha256_file(historical_runtime_path)
        != HISTORICAL_EXPECTED_RUNTIME_CONTRACT_SHA256
    ):
        raise ValueError("historical expected runtime contract hash changed")
    v1_runtime_path = root / EXPECTED_RUNTIME_CONTRACT_V1
    if sha256_file(v1_runtime_path) != VLT1_EXPECTED_RUNTIME_CONTRACT_SHA256:
        raise ValueError("frozen v1 expected runtime contract hash changed")

    v1_latent_trait_path = root / LATENT_TRAIT_ONTOLOGY_V1
    if (
        sha256_file(v1_latent_trait_path)
        != VLT1_LATENT_TRAIT_ONTOLOGY_FILE_SHA256
    ):
        raise ValueError("frozen v1 latent trait ontology hash changed")
    v1_latent_trait_ontology = load_json(v1_latent_trait_path)
    validate_latent_trait_ontology_v1(v1_latent_trait_ontology)
    v1_provenance = v1_latent_trait_ontology.get("provenance")
    if not isinstance(v1_provenance, dict):
        raise ValueError("frozen v1 VLT ontology provenance is missing")

    safe_path = root / SAFE_TRANSFER_AMENDMENT
    if sha256_file(safe_path) != VLT1_SAFE_TRANSFER_AMENDMENT_SHA256:
        raise ValueError("frozen SAFE_TRANSFER v1 amendment hash changed")
    safe = load_json(safe_path)
    expected_safe_fields = {
        "schema_version",
        "kind",
        "status",
        "candidate_id",
        "candidate_revision",
        "parent_latent_scope_amendment_sha256",
        "parent_expected_runtime_contract_sha256",
        "latent_trait_ontology_sha256",
        "provenance_contract_sha256",
        "safe_transfer_contract",
        "candidate_memory_scope",
        "supersession",
        "implementation_sha256",
        "justification",
        "prior_target_model_calls",
        "prior_target_metrics_computed",
    }
    if not isinstance(safe, dict) or set(safe) != expected_safe_fields:
        raise ValueError("SAFE_TRANSFER v1 amendment field contract violation")
    if (
        safe.get("schema_version") != 1
        or safe.get("kind")
        != "verified_latent_transfer_safe_restoration_amendment"
        or safe.get("status")
        != "immutable_before_any_target_model_or_metric_call"
        or safe.get("candidate_id") != "ecpr_v1"
        or safe.get("candidate_revision") != VLT1_CANDIDATE_REVISION
    ):
        raise ValueError("SAFE_TRANSFER v1 identity contract mismatch")
    if (
        safe.get("parent_latent_scope_amendment_sha256")
        != LATENT_SCOPE_AMENDMENT_SHA256
        or safe.get("parent_expected_runtime_contract_sha256")
        != HISTORICAL_EXPECTED_RUNTIME_CONTRACT_SHA256
        or safe.get("latent_trait_ontology_sha256")
        != VLT1_LATENT_TRAIT_ONTOLOGY_FILE_SHA256
        or safe.get("provenance_contract_sha256")
        != _digest_json(v1_provenance)
    ):
        raise ValueError("SAFE_TRANSFER v1 parent/provenance binding mismatch")
    if (
        safe.get("safe_transfer_contract") != VLT1_CONTRACT
        or safe.get("candidate_memory_scope")
        != VLT1_CANDIDATE_MEMORY_SCOPE_CONTRACT
        or safe.get("supersession") != VLT1_SUPERSESSION_CONTRACT
        or safe.get("implementation_sha256") != VLT1_IMPLEMENTATION_HASHES
    ):
        raise ValueError("SAFE_TRANSFER v1 frozen contract mismatch")
    if safe.get("justification") != (
        "safe_restoration_of_originally_preregistered_latent_channel_before_any_"
        "target_model_call_or_target_metric"
    ):
        raise ValueError("SAFE_TRANSFER v1 justification mismatch")
    if (
        safe.get("prior_target_model_calls") != 0
        or safe.get("prior_target_metrics_computed") != 0
    ):
        raise ValueError("SAFE_TRANSFER v1 reports prior target execution")

    v2_latent_trait_path = root / LATENT_TRAIT_ONTOLOGY_V2
    if (
        sha256_file(v2_latent_trait_path)
        != VLT2_LATENT_TRAIT_ONTOLOGY_FILE_SHA256
    ):
        raise ValueError("audited v2 latent trait ontology hash changed")
    v2_latent_trait_ontology = load_json(v2_latent_trait_path)
    validate_latent_trait_ontology_v2(v2_latent_trait_ontology)
    v2_provenance = v2_latent_trait_ontology.get("provenance")
    if not isinstance(v2_provenance, dict):
        raise ValueError("audited v2 VLT ontology provenance is missing")
    source_prompt = v2_provenance.get("original_source_prompt")
    if source_prompt != {
        "repository_relative_path": "ours_memory2/prompts.py",
        "sha256": ORIGINAL_SOURCE_PROMPT_SHA256,
        "runtime_dependency": False,
        "retained_concepts": ["cost sensitivity"],
        "removed_claims": ["service_level_implies_high_cost"],
    }:
        raise ValueError("audited v2 source-prompt provenance mismatch")
    if v2_provenance.get("forbidden_sources") != [
        "current_public_target_queries",
        "current_or_legacy_latent_outputs",
        "preference_groups",
        "gold_or_answers",
        "target_metadata",
        "metrics",
    ]:
        raise ValueError("audited v2 forbidden-source provenance mismatch")
    local_provenance_paths = {
        "preference_slots": root / "configs/preference_slots.json",
        "single": root / "configs/schema_single.json",
        "multi": root / "configs/schema_multi.json",
    }
    for key, path in local_provenance_paths.items():
        declaration = (
            v2_provenance.get("preference_slots")
            if key == "preference_slots"
            else v2_provenance.get("schemas", {}).get(key)
        )
        if (
            not isinstance(declaration, dict)
            or declaration.get("path") != str(path.relative_to(root))
            or declaration.get("sha256") != sha256_file(path)
        ):
            raise ValueError(f"audited v2 ontology provenance mismatch: {key}")
    validate_vlt_schema_contract(
        v2_latent_trait_ontology,
        {
            "single": load_json(root / "configs/schema_single.json"),
            "multi": load_json(root / "configs/schema_multi.json"),
        },
        load_json(root / "configs/preference_slots.json"),
    )

    audit_path = root / VLT_AUDIT_AMENDMENT
    audit = load_json(audit_path)
    expected_audit_fields = {
        "schema_version",
        "kind",
        "status",
        "candidate_id",
        "candidate_revision",
        "parent_safe_transfer_amendment_sha256",
        "parent_expected_runtime_contract_vlt1_sha256",
        "parent_latent_trait_ontology_vlt1_sha256",
        "latent_trait_ontology_vlt2_sha256",
        "provenance_contract_sha256",
        "safe_transfer_contract",
        "candidate_memory_scope",
        "supersession",
        "implementation_sha256",
        "audit_guarantees",
        "prior_target_model_calls",
        "prior_target_metrics_computed",
    }
    if not isinstance(audit, dict) or set(audit) != expected_audit_fields:
        raise ValueError("VLT_AUDIT v2 amendment field contract violation")
    if (
        audit.get("schema_version") != 2
        or audit.get("kind") != "audited_verified_latent_transfer_v2_amendment"
        or audit.get("status")
        != "immutable_before_any_target_model_or_metric_call"
        or audit.get("candidate_id") != "ecpr_v1"
        or audit.get("candidate_revision") != VLT2_CANDIDATE_REVISION
    ):
        raise ValueError("VLT_AUDIT v2 identity contract mismatch")
    if (
        audit.get("parent_safe_transfer_amendment_sha256")
        != VLT1_SAFE_TRANSFER_AMENDMENT_SHA256
        or audit.get("parent_expected_runtime_contract_vlt1_sha256")
        != VLT1_EXPECTED_RUNTIME_CONTRACT_SHA256
        or audit.get("parent_latent_trait_ontology_vlt1_sha256")
        != VLT1_LATENT_TRAIT_ONTOLOGY_FILE_SHA256
        or audit.get("latent_trait_ontology_vlt2_sha256")
        != VLT2_LATENT_TRAIT_ONTOLOGY_FILE_SHA256
        or audit.get("provenance_contract_sha256")
        != _digest_json(v2_provenance)
    ):
        raise ValueError("VLT_AUDIT v2 parent/provenance binding mismatch")
    if (
        audit.get("safe_transfer_contract") != VLT2_CONTRACT
        or audit.get("candidate_memory_scope")
        != VLT2_CANDIDATE_MEMORY_SCOPE_CONTRACT
        or audit.get("supersession") != VLT2_SUPERSESSION_CONTRACT
        or audit.get("implementation_sha256") != _vlt_implementation_hashes(root)
        or audit.get("audit_guarantees") != VLT2_AUDIT_GUARANTEES
    ):
        raise ValueError("VLT_AUDIT v2 implementation contract mismatch")
    if (
        audit.get("prior_target_model_calls") != 0
        or audit.get("prior_target_metrics_computed") != 0
    ):
        raise ValueError("VLT_AUDIT v2 reports prior target execution")
    return (
        preregistration,
        amendment,
        registration,
        routing,
        latent_scope,
        safe,
        audit,
    )


def _validate_common_v1_chain(
    root: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    preregistration_path = root / "preregistration.json"
    preregistration = load_json(preregistration_path)
    preregistration_sha256 = sha256_file(preregistration_path)
    if (
        preregistration_sha256 != ORIGINAL_PREREGISTRATION_SHA256
        or preregistration.get("status") != "locked_before_target_metrics"
        or preregistration.get("candidate") != "ecpr_v1"
    ):
        raise ValueError("original preregistration identity changed")

    amendment_path = root / AMENDMENT
    amendment = load_json(amendment_path)
    if (
        amendment.get("status")
        != "immutable_before_any_target_metric_or_model_call"
        or amendment.get("original_preregistration_sha256")
        != preregistration_sha256
        or amendment.get("confirmatory_candidate") != "ecpr_v1"
        or amendment.get("confirmatory_candidate_count") != 1
        or amendment.get("development_candidates_evaluated") != 0
        or amendment.get("development_stopping_rule_invoked") is not False
        or amendment.get("final_attempts_allowed") != 1
    ):
        raise ValueError("protocol amendment contract mismatch")

    registration_path = root / CANDIDATE_REGISTRATION
    registration = load_json(registration_path)
    if (
        set(registration)
        != {
            "schema_version",
            "kind",
            "status",
            "candidate_id",
            "candidate_count",
            "confirmatory_ablation_flags",
            "original_preregistration_sha256",
            "protocol_amendment_sha256",
        }
        or registration.get("kind") != "confirmatory_candidate_registration"
        or registration.get("status")
        != "immutable_before_any_target_model_or_metric_call"
        or registration.get("candidate_id") != "ecpr_v1"
        or registration.get("candidate_count") != 1
        or registration.get("confirmatory_ablation_flags") != []
        or registration.get("original_preregistration_sha256")
        != preregistration_sha256
        or registration.get("protocol_amendment_sha256")
        != sha256_file(amendment_path)
    ):
        raise ValueError("candidate registration contract mismatch")

    routing_path = root / ROUTING_INTEGRITY_AMENDMENT
    routing = load_json(routing_path)
    if (
        set(routing)
        != {
            "schema_version",
            "kind",
            "status",
            "candidate_id",
            "original_preregistration_sha256",
            "protocol_amendment_sha256",
            "candidate_registration_sha256",
            "public_domain_ontology_sha256",
            "routing_gate",
            "prior_target_model_calls",
            "prior_target_metrics_computed",
        }
        or routing.get("schema_version") != 1
        or routing.get("kind")
        != "public_query_relevance_gate_integrity_amendment"
        or routing.get("status")
        != "immutable_before_any_target_model_or_metric_call"
        or routing.get("candidate_id") != "ecpr_v1"
        or routing.get("original_preregistration_sha256")
        != preregistration_sha256
        or routing.get("protocol_amendment_sha256")
        != sha256_file(amendment_path)
        or routing.get("candidate_registration_sha256")
        != sha256_file(registration_path)
        or routing.get("public_domain_ontology_sha256")
        != sha256_file(root / PUBLIC_DOMAIN_ONTOLOGY)
        or routing.get("routing_gate") != ROUTING_GATE_CONTRACT
        or routing.get("prior_target_model_calls") != 0
        or routing.get("prior_target_metrics_computed") != 0
        or sha256_file(routing_path) != ROUTING_INTEGRITY_AMENDMENT_SHA256
    ):
        raise ValueError("routing integrity amendment contract mismatch")

    latent_scope_path = root / LATENT_SCOPE_AMENDMENT
    latent_scope = load_json(latent_scope_path)
    if (
        set(latent_scope)
        != {
            "schema_version",
            "kind",
            "status",
            "candidate_id",
            "parent_routing_integrity_amendment_sha256",
            "candidate_memory_scope",
            "trigger",
            "prior_target_model_calls",
            "prior_target_metrics_computed",
        }
        or latent_scope.get("schema_version") != 1
        or latent_scope.get("kind")
        != "candidate_typed_memory_scope_integrity_amendment"
        or latent_scope.get("status")
        != "immutable_before_any_target_model_or_metric_call"
        or latent_scope.get("candidate_id") != "ecpr_v1"
        or latent_scope.get("parent_routing_integrity_amendment_sha256")
        != ROUTING_INTEGRITY_AMENDMENT_SHA256
        or latent_scope.get("candidate_memory_scope")
        != CANDIDATE_MEMORY_SCOPE_CONTRACT
        or latent_scope.get("trigger")
        != "static_cross_domain_latent_leakage_audit_before_any_target_model_or_metric_call"
        or latent_scope.get("prior_target_model_calls") != 0
        or latent_scope.get("prior_target_metrics_computed") != 0
        or sha256_file(latent_scope_path) != LATENT_SCOPE_AMENDMENT_SHA256
    ):
        raise ValueError("latent-scope amendment contract mismatch")

    public_ontology = load_json(root / PUBLIC_DOMAIN_ONTOLOGY)
    schema_union = load_json(root / "configs/schema_single.json") + load_json(
        root / "configs/schema_multi.json"
    )
    validate_ontology(public_ontology, schema_union)

    historical_runtime_path = root / HISTORICAL_EXPECTED_RUNTIME_CONTRACT
    v1_runtime_path = root / EXPECTED_RUNTIME_CONTRACT_V1
    v1_ontology_path = root / LATENT_TRAIT_ONTOLOGY_V1
    safe_path = root / SAFE_TRANSFER_AMENDMENT
    if (
        sha256_file(historical_runtime_path)
        != HISTORICAL_EXPECTED_RUNTIME_CONTRACT_SHA256
        or sha256_file(v1_runtime_path) != VLT1_EXPECTED_RUNTIME_CONTRACT_SHA256
        or sha256_file(v1_ontology_path)
        != VLT1_LATENT_TRAIT_ONTOLOGY_FILE_SHA256
        or sha256_file(safe_path) != VLT1_SAFE_TRANSFER_AMENDMENT_SHA256
    ):
        raise ValueError("frozen v1 lineage bytes changed")
    v1_ontology = validate_latent_trait_ontology_v1(
        load_json(v1_ontology_path)
    )
    v1_provenance = v1_ontology.get("provenance")
    if not isinstance(v1_provenance, dict):
        raise ValueError("frozen v1 ontology provenance is missing")
    safe = load_json(safe_path)
    if (
        set(safe)
        != {
            "schema_version",
            "kind",
            "status",
            "candidate_id",
            "candidate_revision",
            "parent_latent_scope_amendment_sha256",
            "parent_expected_runtime_contract_sha256",
            "latent_trait_ontology_sha256",
            "provenance_contract_sha256",
            "safe_transfer_contract",
            "candidate_memory_scope",
            "supersession",
            "implementation_sha256",
            "justification",
            "prior_target_model_calls",
            "prior_target_metrics_computed",
        }
        or safe.get("schema_version") != 1
        or safe.get("kind")
        != "verified_latent_transfer_safe_restoration_amendment"
        or safe.get("status")
        != "immutable_before_any_target_model_or_metric_call"
        or safe.get("candidate_id") != "ecpr_v1"
        or safe.get("candidate_revision") != VLT1_CANDIDATE_REVISION
        or safe.get("parent_latent_scope_amendment_sha256")
        != LATENT_SCOPE_AMENDMENT_SHA256
        or safe.get("parent_expected_runtime_contract_sha256")
        != HISTORICAL_EXPECTED_RUNTIME_CONTRACT_SHA256
        or safe.get("latent_trait_ontology_sha256")
        != VLT1_LATENT_TRAIT_ONTOLOGY_FILE_SHA256
        or safe.get("provenance_contract_sha256")
        != _digest_json(v1_provenance)
        or safe.get("safe_transfer_contract") != VLT1_CONTRACT
        or safe.get("candidate_memory_scope")
        != VLT1_CANDIDATE_MEMORY_SCOPE_CONTRACT
        or safe.get("supersession") != VLT1_SUPERSESSION_CONTRACT
        or safe.get("implementation_sha256") != VLT1_IMPLEMENTATION_HASHES
        or safe.get("prior_target_model_calls") != 0
        or safe.get("prior_target_metrics_computed") != 0
    ):
        raise ValueError("SAFE_TRANSFER v1 frozen contract mismatch")
    return (
        preregistration,
        amendment,
        registration,
        routing,
        latent_scope,
        safe,
    )


def _validate_registration_chain_v3(
    root: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    common = _validate_common_v1_chain(root)
    preregistration, _amendment, _registration, _routing, _latent, safe = common

    v2_path = root / LATENT_TRAIT_ONTOLOGY_V2
    if sha256_file(v2_path) != VLT2_LATENT_TRAIT_ONTOLOGY_FILE_SHA256:
        raise ValueError("frozen VLT2 ontology bytes changed")
    v2_ontology = validate_latent_trait_ontology_v2(load_json(v2_path))
    validate_vlt_schema_contract(
        v2_ontology,
        {
            "single": load_json(root / "configs/schema_single.json"),
            "multi": load_json(root / "configs/schema_multi.json"),
        },
        load_json(root / "configs/preference_slots.json"),
    )

    rubric_path = root / R3_RUBRIC
    scope_path = root / R3_METHOD_SCOPE_CLARIFICATION
    if sha256_file(rubric_path) != R3_RUBRIC_SHA256:
        raise ValueError("frozen R3 rubric hash changed")
    if sha256_file(scope_path) != R3_METHOD_SCOPE_CLARIFICATION_SHA256:
        raise ValueError("R3 method-scope clarification hash changed")
    scope = load_json(scope_path)
    if (
        scope.get("schema_version") != 1
        or scope.get("kind") != "autoresearch_r3_method_scope_clarification"
        or scope.get("status")
        != "immutable_before_any_target_model_call_gold_open_or_target_metric"
        or scope.get("candidate_id") != "ecpr_v1"
        or scope.get("candidate_revision") != VLT3_CANDIDATE_REVISION
        or scope.get("parent_r3_rubric_sha256") != R3_RUBRIC_SHA256
        or scope.get("parent_preregistration_sha256")
        != ORIGINAL_PREREGISTRATION_SHA256
        or scope.get("parent_safe_transfer_amendment_sha256")
        != VLT1_SAFE_TRANSFER_AMENDMENT_SHA256
        or scope.get("parent_vlt2_ontology_sha256")
        != VLT2_LATENT_TRAIT_ONTOLOGY_FILE_SHA256
        or scope.get("prior_target_model_calls") != 0
        or scope.get("prior_gold_opens") != 0
        or scope.get("prior_target_metrics_computed") != 0
    ):
        raise ValueError("R3 method-scope clarification contract mismatch")

    prompt_path = root / R3_ACTION_PROMPT_AMENDMENT
    if sha256_file(prompt_path) != R3_ACTION_PROMPT_AMENDMENT_SHA256:
        raise ValueError("R3 action-prompt amendment hash changed")
    prompt = load_json(prompt_path)
    if (
        set(prompt)
        != {
            "schema_version",
            "kind",
            "status",
            "candidate_id",
            "candidate_revision",
            "parent_preregistration_sha256",
            "parent_r3_rubric_sha256",
            "parent_r3_method_scope_clarification_sha256",
            "local_paper_sha256",
            "standalone_prompts_sha256",
            "research_action_prompts",
            "action_prompt_contract",
            "prior_target_model_calls",
            "prior_gold_opens",
            "prior_target_metrics_computed",
        }
        or prompt.get("schema_version") != 1
        or prompt.get("kind") != "r3_source_faithful_action_prompt_amendment"
        or prompt.get("status")
        != "immutable_before_any_target_model_call_gold_open_or_target_metric"
        or prompt.get("candidate_id") != "ecpr_v1"
        or prompt.get("candidate_revision") != VLT3_CANDIDATE_REVISION
        or prompt.get("parent_preregistration_sha256")
        != ORIGINAL_PREREGISTRATION_SHA256
        or prompt.get("parent_r3_rubric_sha256") != R3_RUBRIC_SHA256
        or prompt.get("parent_r3_method_scope_clarification_sha256")
        != R3_METHOD_SCOPE_CLARIFICATION_SHA256
        or prompt.get("local_paper_sha256") != LOCAL_PAPER_SHA256
        or prompt.get("standalone_prompts_sha256")
        != ORIGINAL_SOURCE_PROMPT_SHA256
        or prompt.get("research_action_prompts")
        != {
            "path": "ecpr/prompts.py",
            "sha256": RESEARCH_ACTION_PROMPTS_SHA256,
        }
        or prompt.get("action_prompt_contract") != R3_ACTION_PROMPT_CONTRACT
        or prompt.get("prior_target_model_calls") != 0
        or prompt.get("prior_gold_opens") != 0
        or prompt.get("prior_target_metrics_computed") != 0
        or sha256_file(root / "ecpr/prompts.py")
        != RESEARCH_ACTION_PROMPTS_SHA256
    ):
        raise ValueError("R3 action-prompt amendment contract mismatch")

    ontology_path = root / LATENT_TRAIT_ONTOLOGY_V3
    if sha256_file(ontology_path) != VLT3_LATENT_TRAIT_ONTOLOGY_FILE_SHA256:
        raise ValueError("VLT3 ontology file hash changed")
    ontology = validate_latent_trait_ontology_v3(load_json(ontology_path))
    if _digest_json(ontology) != VLT3_LATENT_TRAIT_ONTOLOGY_CANONICAL_SHA256:
        raise ValueError("VLT3 ontology canonical hash mismatch")
    validate_vlt_schema_contract(
        ontology,
        {
            "single": load_json(root / "configs/schema_single.json"),
            "multi": load_json(root / "configs/schema_multi.json"),
        },
        load_json(root / "configs/preference_slots.json"),
    )

    report_path = root / TARGET_FREE_VLT3_REPORT
    if sha256_file(report_path) != TARGET_FREE_VLT3_REPORT_SHA256:
        raise ValueError("target-free VLT3 report hash changed")
    report = load_json(report_path)
    report_inputs = report.get("inputs")
    if (
        report.get("schema_version") != 1
        or report.get("kind") != "target_free_vlt3_structural_diagnostics"
        or report.get("status")
        != "aggregate_only_no_query_gold_latent_model_output_or_metric"
        or report.get("candidate_id") != "ecpr_v1"
        or report.get("candidate_revision") != VLT3_CANDIDATE_REVISION
        or not isinstance(report_inputs, dict)
        or report_inputs.get("sanitized_history", {}).get("sha256")
        != sha256_file(root / "artifacts/history.sanitized.jsonl")
        or report_inputs.get("preference_slots", {}).get("sha256")
        != sha256_file(root / "configs/preference_slots.json")
        or report_inputs.get("latent_trait_ontology", {}).get("sha256")
        != VLT3_LATENT_TRAIT_ONTOLOGY_FILE_SHA256
        or report_inputs.get("preregistration", {}).get("sha256")
        != ORIGINAL_PREREGISTRATION_SHA256
        or "evaluator_vault/gold.jsonl"
        not in report.get("forbidden_sources_not_opened", [])
        or "configs/query_singleturn.json"
        not in report.get("forbidden_sources_not_opened", [])
    ):
        raise ValueError("target-free VLT3 report contract mismatch")

    audit_path = root / VLT3_AUDIT_AMENDMENT
    if sha256_file(audit_path) != VLT3_AUDIT_AMENDMENT_SHA256:
        raise ValueError("VLT3 audit amendment hash changed")
    audit = load_json(audit_path)
    v3_provenance = ontology.get("provenance")
    current_implementation_hashes = _vlt3_implementation_hashes(root)
    if (
        (root / RUNTIME_RECOVERY_AMENDMENT).exists()
        or (root / RUNTIME_RECOVERY_AMENDMENT).is_symlink()
    ):
        _validate_runtime_recovery_chain(
            root, audit, current_implementation_hashes
        )
        expected_audited_implementation = audit.get("implementation_sha256")
    else:
        expected_audited_implementation = current_implementation_hashes
    if (
        set(audit)
        != {
            "schema_version",
            "kind",
            "status",
            "candidate_id",
            "candidate_revision",
            "parent_safe_transfer_amendment_sha256",
            "parent_expected_runtime_contract_vlt1_sha256",
            "parent_latent_trait_ontology_vlt2_sha256",
            "parent_r3_rubric_sha256",
            "parent_r3_method_scope_clarification_sha256",
            "parent_r3_action_prompt_amendment_sha256",
            "latent_trait_ontology_vlt3_sha256",
            "target_free_diagnostics_script_sha256",
            "target_free_diagnostics_report_sha256",
            "provenance_contract_sha256",
            "safe_transfer_contract",
            "candidate_memory_scope",
            "supersession",
            "implementation_sha256",
            "audit_guarantees",
            "prior_target_model_calls",
            "prior_gold_opens",
            "prior_target_metrics_computed",
        }
        or audit.get("schema_version") != 3
        or audit.get("kind")
        != "append_only_table5_verified_latent_transfer_v3_amendment"
        or audit.get("status")
        != "immutable_before_any_target_model_call_gold_open_or_target_metric"
        or audit.get("candidate_id") != "ecpr_v1"
        or audit.get("candidate_revision") != ACTIVE_CANDIDATE_REVISION
        or audit.get("parent_safe_transfer_amendment_sha256")
        != VLT1_SAFE_TRANSFER_AMENDMENT_SHA256
        or audit.get("parent_expected_runtime_contract_vlt1_sha256")
        != VLT1_EXPECTED_RUNTIME_CONTRACT_SHA256
        or audit.get("parent_latent_trait_ontology_vlt2_sha256")
        != VLT2_LATENT_TRAIT_ONTOLOGY_FILE_SHA256
        or audit.get("parent_r3_rubric_sha256") != R3_RUBRIC_SHA256
        or audit.get("parent_r3_method_scope_clarification_sha256")
        != R3_METHOD_SCOPE_CLARIFICATION_SHA256
        or audit.get("parent_r3_action_prompt_amendment_sha256")
        != R3_ACTION_PROMPT_AMENDMENT_SHA256
        or audit.get("latent_trait_ontology_vlt3_sha256")
        != VLT3_LATENT_TRAIT_ONTOLOGY_FILE_SHA256
        or audit.get("target_free_diagnostics_script_sha256")
        != sha256_file(root / TARGET_FREE_VLT3_DIAGNOSTICS)
        or audit.get("target_free_diagnostics_report_sha256")
        != TARGET_FREE_VLT3_REPORT_SHA256
        or not isinstance(v3_provenance, dict)
        or audit.get("provenance_contract_sha256")
        != _digest_json(v3_provenance)
        or audit.get("safe_transfer_contract") != ACTIVE_VLT_CONTRACT
        or audit.get("candidate_memory_scope")
        != ACTIVE_VLT_CANDIDATE_MEMORY_SCOPE_CONTRACT
        or audit.get("supersession") != ACTIVE_VLT_SUPERSESSION_CONTRACT
        or audit.get("implementation_sha256")
        != expected_audited_implementation
        or audit.get("audit_guarantees") != VLT3_AUDIT_GUARANTEES
        or audit.get("prior_target_model_calls") != 0
        or audit.get("prior_gold_opens") != 0
        or audit.get("prior_target_metrics_computed") != 0
    ):
        raise ValueError("VLT3 audit amendment contract mismatch")
    if safe.get("candidate_id") != audit.get("candidate_id"):
        raise ValueError("VLT3 audit candidate differs from frozen V1 parent")
    return (*common, scope, prompt, audit)


def validate_registration_chain(root: Path) -> tuple[dict[str, Any], ...]:
    root = root.resolve()
    v3_markers = (
        VLT3_AUDIT_AMENDMENT,
        EXPECTED_RUNTIME_CONTRACT_V3,
        R3_ACTION_PROMPT_AMENDMENT,
    )
    if any((root / path).exists() or (root / path).is_symlink() for path in v3_markers):
        return _validate_registration_chain_v3(root)
    return _validate_registration_chain_v2(root)


def validate_frozen_external_inputs(
    root: Path,
    preregistration: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Validate declarations without opening raw external experiment sources."""
    inputs = preregistration.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != EXPECTED_EXTERNAL_INPUTS:
        raise ValueError("frozen external input inventory mismatch")
    internal_copies = {
        "single_query": root / "configs/query_singleturn.json",
        "single_schema": root / "configs/schema_single.json",
        "multi_query": root / "configs/query_multiturn-domain.json",
        "multi_schema": root / "configs/schema_multi.json",
        "preference_slots": root / "configs/preference_slots.json",
    }
    result: dict[str, dict[str, Any]] = {}
    for name in sorted(inputs):
        specification = inputs[name]
        if not isinstance(specification, dict):
            raise ValueError(f"invalid frozen input specification: {name}")
        raw_path = specification.get("path")
        declared_sha256 = _require_sha256(
            specification.get("sha256"), f"{name} preregistered digest"
        )
        if not isinstance(raw_path, str) or not raw_path or not Path(raw_path).is_absolute():
            raise ValueError(f"invalid frozen input declaration: {name}")
        declaration: dict[str, Any] = {
            "declared_path": raw_path,
            "declared_sha256": declared_sha256,
            "runtime_dependency": False,
            "external_file_opened": False,
        }
        internal_path = internal_copies.get(name)
        if internal_path is not None:
            if internal_path.is_symlink() or not internal_path.is_file():
                raise ValueError(f"sealed internal input copy is missing: {name}")
            internal_sha256 = sha256_file(internal_path)
            if internal_sha256 != declared_sha256:
                raise ValueError(f"sealed internal input copy hash mismatch: {name}")
            declaration["sealed_internal_copy"] = {
                "path": str(internal_path.relative_to(root)),
                "sha256": internal_sha256,
            }
        result[name] = declaration
    return result


def build_routing_coverage(root: Path) -> dict[str, Any]:
    """Recompute target-free routing coverage; this is not an accuracy metric."""
    root = root.resolve()
    task_path = root / "artifacts/tasks.jsonl"
    ontology_path = root / PUBLIC_DOMAIN_ONTOLOGY
    schema_paths = {
        "single": root / "configs/schema_single.json",
        "multi": root / "configs/schema_multi.json",
    }
    ontology = load_json(ontology_path)
    schemas = {
        key: load_json(path)
        for key, path in schema_paths.items()
    }
    counts = {
        "single": {"select": 0, "abstain": 0, "total": 0},
        "multi": {"select": 0, "abstain": 0, "total": 0},
    }
    for task in iter_jsonl(task_path):
        if not isinstance(task, dict) or set(task) != TASK_FIELDS:
            raise ValueError("routing coverage task field contract violation")
        schema_key = task.get("schema_key")
        if schema_key not in schemas:
            raise ValueError("routing coverage schema key mismatch")
        decision = gate_public_query(
            task["query"],
            task["mode"],
            schemas[schema_key],
            ontology,
        )
        label = decision.status.casefold()
        if label not in {"select", "abstain"}:
            raise ValueError("routing coverage decision label mismatch")
        counts[schema_key][label] += 1
        counts[schema_key]["total"] += 1
    counts["total"] = {
        label: counts["single"][label] + counts["multi"][label]
        for label in ("select", "abstain", "total")
    }
    return {
        "schema_version": 1,
        "kind": "routing_coverage_not_accuracy",
        "status": "frozen_post_lock_determinism_snapshot",
        "authority": (
            "post_lock_determinism_snapshot_only_cannot_change_ontology_or_thresholds"
        ),
        "task_sha256": sha256_file(task_path),
        "schema_sha256": {
            key: sha256_file(path)
            for key, path in schema_paths.items()
        },
        "ontology_sha256": sha256_file(ontology_path),
        "routing_integrity_amendment_sha256": sha256_file(
            root / ROUTING_INTEGRITY_AMENDMENT
        ),
        "latent_scope_amendment_sha256": sha256_file(
            root / LATENT_SCOPE_AMENDMENT
        ),
        "routing_gate_contract_sha256": _digest_json(ROUTING_GATE_CONTRACT),
        "counts": counts,
    }


def validate_routing_coverage(root: Path) -> dict[str, Any]:
    root = root.resolve()
    path = root / PQR_ROUTING_COVERAGE
    if path.is_symlink() or not path.is_file():
        raise ValueError("frozen routing coverage evidence is missing")
    coverage = load_json(path)
    expected = build_routing_coverage(root)
    if coverage != expected:
        raise ValueError("frozen routing coverage recomputation mismatch")
    return {
        "path": str(PQR_ROUTING_COVERAGE),
        "sha256": sha256_file(path),
        "size": path.stat().st_size,
        "kind": "routing_coverage_not_accuracy",
        "authority": "non_authoritative_determinism_evidence",
    }


def _validate_expected_runtime_contract_v2(
    root: Path, preregistration: dict[str, Any]
) -> dict[str, Any]:
    historical = load_json(root / HISTORICAL_EXPECTED_RUNTIME_CONTRACT)
    contract = historical
    expected_fields = {
        "schema_version",
        "kind",
        "runner_mode",
        "candidate_id",
        "final_attempts_allowed",
        "model",
        "runtime",
        "action_budget",
        "candidate_policy",
        "routing_gate",
        "candidate_memory_scope",
        "latent_scope_amendment_sha256",
        "pipeline",
    }
    if not isinstance(contract, dict) or set(contract) != expected_fields:
        raise ValueError("expected runtime contract field mismatch")
    if contract.get("kind") != "expected_final_runtime_model_pipeline_contract":
        raise ValueError("expected runtime contract kind mismatch")
    if contract.get("runner_mode") != "explicit_--final_or_--preflight-only":
        raise ValueError("expected runner-mode contract mismatch")
    if contract.get("candidate_id") != "ecpr_v1":
        raise ValueError("expected runtime candidate mismatch")
    if contract.get("final_attempts_allowed") != 1:
        raise ValueError("expected runtime attempt budget mismatch")

    model = contract.get("model")
    preregistered_model = preregistration.get("model")
    if not isinstance(model, dict) or not isinstance(preregistered_model, dict):
        raise ValueError("missing model contract")
    if model.get("id") != preregistered_model.get("id"):
        raise ValueError("expected model ID mismatch")
    if model.get("snapshot") != preregistered_model.get("snapshot"):
        raise ValueError("expected model snapshot mismatch")
    if model.get("tensor_parallel_size") != preregistered_model.get(
        "tensor_parallel_size"
    ):
        raise ValueError("expected tensor-parallel contract mismatch")
    if model.get("served_model_name") != model.get("id"):
        raise ValueError("served model name must equal the preregistered model ID")
    snapshot_path = model.get("snapshot_path")
    if (
        not isinstance(snapshot_path, str)
        or Path(snapshot_path).name != model.get("snapshot")
    ):
        raise ValueError("model snapshot path does not bind the registered revision")

    expected_action_budget = dict(preregistration.get("inference_budget", {}))
    expected_action_budget["n"] = 1
    if contract.get("action_budget") != expected_action_budget:
        raise ValueError("expected action budget differs from preregistration")
    if contract.get("candidate_policy") != preregistration.get(
        "candidate_parameters"
    ):
        raise ValueError("expected candidate policy differs from preregistration")
    if contract.get("routing_gate") != ROUTING_GATE_CONTRACT:
        raise ValueError("expected public query routing gate contract mismatch")
    routing_amendment = load_json(root / ROUTING_INTEGRITY_AMENDMENT)
    if contract.get("routing_gate") != routing_amendment.get("routing_gate"):
        raise ValueError("runtime routing gate differs from immutable amendment")
    latent_scope_path = root / LATENT_SCOPE_AMENDMENT
    latent_scope_amendment = load_json(latent_scope_path)
    if contract.get("latent_scope_amendment_sha256") != sha256_file(
        latent_scope_path
    ):
        raise ValueError("runtime latent-scope amendment parent binding mismatch")
    if contract.get("candidate_memory_scope") != CANDIDATE_MEMORY_SCOPE_CONTRACT:
        raise ValueError("runtime candidate-memory scope mismatch")
    if contract.get("candidate_memory_scope") != latent_scope_amendment.get(
        "candidate_memory_scope"
    ):
        raise ValueError("runtime candidate-memory scope differs from amendment")

    pipeline = contract.get("pipeline")
    if not isinstance(pipeline, dict):
        raise ValueError("missing expected pipeline contract")
    if pipeline.get("stages") != EXPECTED_PIPELINE_STAGES:
        raise ValueError("expected pipeline stage sequence mismatch")
    if pipeline.get("forbidden_stages") != ["prepare"]:
        raise ValueError("expected forbidden-stage contract mismatch")
    if any(
        "prepare" in str(stage).casefold() for stage in pipeline.get("stages", [])
    ):
        raise ValueError("target preparation is forbidden in the final pipeline")
    if pipeline.get("base_url") != "http://127.0.0.1:8129":
        raise ValueError("final provider must use the frozen loopback endpoint")
    expected_outputs = {
        "journal": "artifacts/provider_calls.final.jsonl",
        "baseline_latent": "artifacts/memory.prefine.jsonl",
        "baseline_latent_manifest": "artifacts/memory.prefine.jsonl.manifest.json",
        "memory": "artifacts/memory.ecpr.jsonl",
        "memory_manifest": "artifacts/memory.ecpr.jsonl.manifest.json",
        "baseline": "artifacts/predictions.baseline.jsonl",
        "baseline_manifest": "artifacts/predictions.baseline.jsonl.manifest.json",
        "candidate": "artifacts/predictions.candidate.jsonl",
        "candidate_manifest": "artifacts/predictions.candidate.jsonl.manifest.json",
        "summary": "reports/summary.json",
    }
    if pipeline.get("outputs") != expected_outputs:
        raise ValueError("expected final output paths mismatch")

    runtime = contract.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("missing expected runtime settings")
    if runtime.get("host") != "127.0.0.1" or runtime.get("port") != 8129:
        raise ValueError("runtime endpoint must be fixed loopback port 8129")
    if runtime.get("gpu_indices") != [0, 1, 2, 3]:
        raise ValueError("runtime GPU inventory mismatch")
    if runtime.get("gpu_preflight_command") != [
        "timeout",
        "5",
        "nvidia-smi",
        "--query-gpu=index",
        "--format=csv,noheader",
    ]:
        raise ValueError("runtime GPU preflight command mismatch")
    if runtime.get("gpu_preflight_subprocess_timeout_seconds") != 8:
        raise ValueError("runtime GPU preflight timeout mismatch")
    if runtime.get("server_start_timeout_seconds") != 600:
        raise ValueError("runtime server-start timeout mismatch")

    environment = load_json(root / "manifests/environment.json")
    if environment.get("model_snapshot") != model.get("snapshot_path"):
        raise ValueError("environment/model snapshot path mismatch")
    if environment.get("vllm_executable") != runtime.get("vllm_executable"):
        raise ValueError("environment/vLLM path mismatch")
    gpu_contract = environment.get("gpu_contract", {})
    if gpu_contract.get("indices") != runtime.get("gpu_indices"):
        raise ValueError("environment/runtime GPU index mismatch")
    if gpu_contract.get("tensor_parallel_size") != model.get(
        "tensor_parallel_size"
    ):
        raise ValueError("environment/runtime tensor-parallel mismatch")
    v1 = load_json(root / EXPECTED_RUNTIME_CONTRACT_V1)
    expected_v1_fields = {
        *expected_fields,
        "candidate_revision",
        "parent_expected_runtime_contract_sha256",
        "safe_transfer_amendment_sha256",
        "latent_trait_ontology_sha256",
        "safe_transfer_contract",
    }
    if not isinstance(v1, dict) or set(v1) != expected_v1_fields:
        raise ValueError("frozen VLT v1 runtime contract field mismatch")
    if sha256_file(root / EXPECTED_RUNTIME_CONTRACT_V1) != (
        VLT1_EXPECTED_RUNTIME_CONTRACT_SHA256
    ):
        raise ValueError("frozen VLT v1 runtime hash changed")
    v1_shared_fields = {
        "schema_version",
        "kind",
        "runner_mode",
        "candidate_id",
        "final_attempts_allowed",
        "model",
        "runtime",
        "action_budget",
        "candidate_policy",
        "routing_gate",
        "latent_scope_amendment_sha256",
        "pipeline",
    }
    if any(v1.get(field) != historical.get(field) for field in v1_shared_fields):
        raise ValueError("frozen VLT v1 runtime changed a historical field")
    if (
        v1.get("candidate_revision") != VLT1_CANDIDATE_REVISION
        or v1.get("parent_expected_runtime_contract_sha256")
        != HISTORICAL_EXPECTED_RUNTIME_CONTRACT_SHA256
        or v1.get("safe_transfer_amendment_sha256")
        != VLT1_SAFE_TRANSFER_AMENDMENT_SHA256
        or v1.get("latent_trait_ontology_sha256")
        != VLT1_LATENT_TRAIT_ONTOLOGY_FILE_SHA256
        or v1.get("safe_transfer_contract") != VLT1_CONTRACT
        or v1.get("candidate_memory_scope")
        != VLT1_CANDIDATE_MEMORY_SCOPE_CONTRACT
    ):
        raise ValueError("frozen VLT v1 runtime lineage mismatch")
    safe = load_json(root / SAFE_TRANSFER_AMENDMENT)
    if (
        v1["safe_transfer_contract"] != safe["safe_transfer_contract"]
        or v1["candidate_memory_scope"] != safe["candidate_memory_scope"]
    ):
        raise ValueError("frozen VLT v1 runtime differs from its amendment")

    active = load_json(root / EXPECTED_RUNTIME_CONTRACT_V2)
    expected_v2_fields = {
        *expected_v1_fields,
        "vlt_audit_amendment_sha256",
        "latent_trait_ontology_path",
        "provider_timeout_seconds",
    }
    if not isinstance(active, dict) or set(active) != expected_v2_fields:
        raise ValueError("active audited VLT v2 runtime contract field mismatch")
    v2_unchanged_fields = {
        "schema_version",
        "kind",
        "runner_mode",
        "candidate_id",
        "final_attempts_allowed",
        "model",
        "runtime",
        "action_budget",
        "candidate_policy",
        "latent_scope_amendment_sha256",
        "pipeline",
        "safe_transfer_amendment_sha256",
    }
    if any(active.get(field) != v1.get(field) for field in v2_unchanged_fields):
        raise ValueError("active audited VLT v2 changed an unrelated v1 field")
    expected_v2_routing = dict(v1["routing_gate"])
    expected_v2_routing["latent_policy"] = VLT2_ROUTING_GATE_CONTRACT[
        "latent_policy"
    ]
    if (
        active.get("routing_gate") != expected_v2_routing
        or active.get("routing_gate") != VLT2_ROUTING_GATE_CONTRACT
    ):
        raise ValueError("active audited VLT v2 routing diff is not exact")
    audit_path = root / VLT_AUDIT_AMENDMENT
    audit = load_json(audit_path)
    if (
        active.get("candidate_revision") != VLT2_CANDIDATE_REVISION
        or active.get("parent_expected_runtime_contract_sha256")
        != VLT1_EXPECTED_RUNTIME_CONTRACT_SHA256
        or active.get("vlt_audit_amendment_sha256")
        != sha256_file(audit_path)
        or active.get("latent_trait_ontology_sha256")
        != VLT2_LATENT_TRAIT_ONTOLOGY_FILE_SHA256
        or active.get("latent_trait_ontology_path")
        != str(LATENT_TRAIT_ONTOLOGY_V2)
        or active.get("provider_timeout_seconds") != 120.0
        or active.get("safe_transfer_contract") != VLT2_CONTRACT
        or active.get("candidate_memory_scope")
        != VLT2_CANDIDATE_MEMORY_SCOPE_CONTRACT
    ):
        raise ValueError("active audited VLT v2 lineage mismatch")
    if (
        active["safe_transfer_contract"] != audit["safe_transfer_contract"]
        or active["candidate_memory_scope"] != audit["candidate_memory_scope"]
    ):
        raise ValueError("active audited VLT v2 differs from VLT_AUDIT amendment")
    return active


def _validate_expected_runtime_contract_v3(
    root: Path, preregistration: dict[str, Any]
) -> dict[str, Any]:
    """Validate the real append-only VLT3 runtime without inventing V2 artifacts."""
    historical_path = root / HISTORICAL_EXPECTED_RUNTIME_CONTRACT
    v1_path = root / EXPECTED_RUNTIME_CONTRACT_V1
    active_path = root / EXPECTED_RUNTIME_CONTRACT_V3
    if sha256_file(historical_path) != HISTORICAL_EXPECTED_RUNTIME_CONTRACT_SHA256:
        raise ValueError("historical expected runtime contract hash changed")
    if sha256_file(v1_path) != VLT1_EXPECTED_RUNTIME_CONTRACT_SHA256:
        raise ValueError("frozen VLT1 expected runtime contract hash changed")
    if sha256_file(active_path) != VLT3_EXPECTED_RUNTIME_CONTRACT_SHA256:
        raise ValueError("VLT3 expected runtime contract hash changed")

    historical = load_json(historical_path)
    v1 = load_json(v1_path)
    active = load_json(active_path)
    historical_fields = {
        "schema_version",
        "kind",
        "runner_mode",
        "candidate_id",
        "final_attempts_allowed",
        "model",
        "runtime",
        "action_budget",
        "candidate_policy",
        "routing_gate",
        "candidate_memory_scope",
        "latent_scope_amendment_sha256",
        "pipeline",
    }
    v1_fields = {
        *historical_fields,
        "candidate_revision",
        "parent_expected_runtime_contract_sha256",
        "safe_transfer_amendment_sha256",
        "latent_trait_ontology_sha256",
        "safe_transfer_contract",
    }
    v3_fields = {
        *v1_fields,
        "vlt3_audit_amendment_sha256",
        "r3_method_scope_clarification_sha256",
        "r3_action_prompt_amendment_sha256",
        "latent_trait_ontology_path",
        "target_free_vlt3_report_sha256",
        "provider_timeout_seconds",
        "one_shot_protocol",
    }
    if not isinstance(historical, dict) or set(historical) != historical_fields:
        raise ValueError("historical expected runtime field contract mismatch")
    if not isinstance(v1, dict) or set(v1) != v1_fields:
        raise ValueError("frozen VLT1 expected runtime field contract mismatch")
    if not isinstance(active, dict) or set(active) != v3_fields:
        raise ValueError("active VLT3 expected runtime field contract mismatch")

    v1_shared = historical_fields - {"candidate_memory_scope"}
    if any(v1.get(field) != historical.get(field) for field in v1_shared):
        raise ValueError("frozen VLT1 runtime changed a historical field")
    if (
        v1.get("candidate_revision") != VLT1_CANDIDATE_REVISION
        or v1.get("parent_expected_runtime_contract_sha256")
        != HISTORICAL_EXPECTED_RUNTIME_CONTRACT_SHA256
        or v1.get("safe_transfer_amendment_sha256")
        != VLT1_SAFE_TRANSFER_AMENDMENT_SHA256
        or v1.get("latent_trait_ontology_sha256")
        != VLT1_LATENT_TRAIT_ONTOLOGY_FILE_SHA256
        or v1.get("safe_transfer_contract") != VLT1_CONTRACT
        or v1.get("candidate_memory_scope")
        != VLT1_CANDIDATE_MEMORY_SCOPE_CONTRACT
    ):
        raise ValueError("frozen VLT1 runtime lineage mismatch")

    unchanged_v1_fields = {
        "schema_version",
        "kind",
        "runner_mode",
        "candidate_id",
        "final_attempts_allowed",
        "model",
        "runtime",
        "action_budget",
        "candidate_policy",
        "latent_scope_amendment_sha256",
        "pipeline",
        "safe_transfer_amendment_sha256",
    }
    if any(active.get(field) != v1.get(field) for field in unchanged_v1_fields):
        raise ValueError("VLT3 runtime changed a frozen unrelated VLT1 field")
    preregistered_model = preregistration.get("model")
    model = active.get("model")
    if not isinstance(model, dict) or not isinstance(preregistered_model, dict):
        raise ValueError("VLT3 runtime model contract is missing")
    if (
        model.get("id") != preregistered_model.get("id")
        or model.get("snapshot") != preregistered_model.get("snapshot")
        or model.get("tensor_parallel_size")
        != preregistered_model.get("tensor_parallel_size")
        or model.get("served_model_name") != model.get("id")
    ):
        raise ValueError("VLT3 runtime model differs from preregistration")
    expected_action_budget = dict(preregistration.get("inference_budget", {}))
    expected_action_budget["n"] = 1
    if active.get("action_budget") != expected_action_budget:
        raise ValueError("VLT3 action budget differs from preregistration")
    if active.get("candidate_policy") != preregistration.get("candidate_parameters"):
        raise ValueError("VLT3 candidate policy differs from preregistration")

    audit_path = root / VLT3_AUDIT_AMENDMENT
    audit = load_json(audit_path)
    if (
        active.get("candidate_revision") != ACTIVE_CANDIDATE_REVISION
        or active.get("parent_expected_runtime_contract_sha256")
        != VLT1_EXPECTED_RUNTIME_CONTRACT_SHA256
        or active.get("safe_transfer_amendment_sha256")
        != VLT1_SAFE_TRANSFER_AMENDMENT_SHA256
        or active.get("vlt3_audit_amendment_sha256")
        != VLT3_AUDIT_AMENDMENT_SHA256
        or active.get("vlt3_audit_amendment_sha256") != sha256_file(audit_path)
        or active.get("r3_method_scope_clarification_sha256")
        != R3_METHOD_SCOPE_CLARIFICATION_SHA256
        or active.get("r3_action_prompt_amendment_sha256")
        != R3_ACTION_PROMPT_AMENDMENT_SHA256
        or active.get("latent_trait_ontology_sha256")
        != VLT3_LATENT_TRAIT_ONTOLOGY_FILE_SHA256
        or active.get("latent_trait_ontology_path")
        != str(LATENT_TRAIT_ONTOLOGY_V3)
        or active.get("target_free_vlt3_report_sha256")
        != TARGET_FREE_VLT3_REPORT_SHA256
        or active.get("provider_timeout_seconds") != 120.0
        or active.get("routing_gate") != VLT3_ROUTING_GATE_CONTRACT
        or active.get("candidate_memory_scope")
        != ACTIVE_VLT_CANDIDATE_MEMORY_SCOPE_CONTRACT
        or active.get("safe_transfer_contract") != ACTIVE_VLT_CONTRACT
        or active.get("one_shot_protocol") != VLT3_ONE_SHOT_PROTOCOL
    ):
        raise ValueError("active VLT3 runtime lineage or execution contract mismatch")
    if (
        active["candidate_revision"] != audit.get("candidate_revision")
        or active["safe_transfer_contract"] != audit.get("safe_transfer_contract")
        or active["candidate_memory_scope"] != audit.get("candidate_memory_scope")
        or active["latent_trait_ontology_sha256"]
        != audit.get("latent_trait_ontology_vlt3_sha256")
    ):
        raise ValueError("active VLT3 runtime differs from its audit amendment")
    return active


def validate_expected_runtime_contract(
    root: Path, preregistration: dict[str, Any]
) -> dict[str, Any]:
    root = root.resolve()
    v3_markers = (
        EXPECTED_RUNTIME_CONTRACT_V3,
        VLT3_AUDIT_AMENDMENT,
        R3_ACTION_PROMPT_AMENDMENT,
    )
    if any((root / path).exists() or (root / path).is_symlink() for path in v3_markers):
        return _validate_expected_runtime_contract_v3(root, preregistration)
    return _validate_expected_runtime_contract_v2(root, preregistration)


def validate_sealed_artifacts(
    root: Path, preregistration_sha256: str
) -> dict[str, Any]:
    sealed_path = root / "evaluator_vault/sealed_manifest.json"
    sealed = load_json(sealed_path)
    if sealed.get("preregistration_sha256") != preregistration_sha256:
        raise ValueError("sealed evaluator manifest preregistration mismatch")
    task_sha256 = sha256_file(root / "artifacts/tasks.jsonl")
    history_sha256 = sha256_file(root / "artifacts/history.sanitized.jsonl")
    if sealed.get("task_sha256") != task_sha256:
        raise ValueError("sealed task hash mismatch")
    if sealed.get("history_sha256") != history_sha256:
        raise ValueError("sealed history hash mismatch")
    gold_sha256 = _require_sha256(
        sealed.get("gold_sha256"), "sealed evaluator gold declaration"
    )
    if sealed.get("unassigned_target_variant_count", 0) != 0:
        raise ValueError("sealed evaluator manifest has unassigned target variants")

    tasks = list(iter_jsonl(root / "artifacts/tasks.jsonl"))
    if any(set(task) != TASK_FIELDS for task in tasks):
        raise ValueError("sealed task field contract violation")
    schema_hashes = {
        "single": sha256_file(root / "configs/schema_single.json"),
        "multi": sha256_file(root / "configs/schema_multi.json"),
    }
    if tasks != build_public_tasks(tasks, schema_hashes):
        raise ValueError("sealed public task identity/order contract mismatch")
    history_count = validate_sanitized_history(
        root / "artifacts/history.sanitized.jsonl"
    )
    if sealed.get("case_count") != len(tasks):
        raise ValueError("sealed task count mismatch")
    if sealed.get("history_count") != history_count:
        raise ValueError("sealed history count mismatch")
    return {
        "sealed_manifest_sha256": sha256_file(sealed_path),
        "tasks_sha256": task_sha256,
        "history_sha256": history_sha256,
        "declared_gold_sha256": gold_sha256,
        "case_count": len(tasks),
        "history_count": history_count,
        "gold_rows_read": False,
    }


def build_implementation_manifest(root: Path) -> dict[str, Any]:
    root = root.resolve()
    chain = validate_registration_chain(root)
    if len(chain) not in {7, 9}:
        raise ValueError("unsupported registration-chain shape")
    is_v3 = len(chain) == 9
    is_recovery = is_v3 and (root / RUNTIME_RECOVERY_AMENDMENT).is_file()
    prereg, amendment, registration = chain[:3]
    prereg_hash = sha256_file(root / "preregistration.json")
    external_inputs = validate_frozen_external_inputs(root, prereg)
    preference_slots = validate_preference_slots(root)
    expected_contract = validate_expected_runtime_contract(root, prereg)
    routing_coverage_evidence = validate_routing_coverage(root)
    sealed_artifacts = validate_sealed_artifacts(root, prereg_hash)
    files = {
        str(path.relative_to(root)): {
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
        }
        for path in influential_paths(root)
    }

    amendment_sha256 = files[str(AMENDMENT)]["sha256"]
    registration_sha256 = files[str(CANDIDATE_REGISTRATION)]["sha256"]
    routing_amendment_sha256 = files[str(ROUTING_INTEGRITY_AMENDMENT)]["sha256"]
    latent_scope_amendment_sha256 = files[str(LATENT_SCOPE_AMENDMENT)]["sha256"]
    historical_runtime_sha256 = files[
        str(HISTORICAL_EXPECTED_RUNTIME_CONTRACT)
    ]["sha256"]
    safe_transfer_amendment_sha256 = files[str(SAFE_TRANSFER_AMENDMENT)]["sha256"]
    v1_runtime_sha256 = files[str(EXPECTED_RUNTIME_CONTRACT_V1)]["sha256"]
    active_audit_path = VLT3_AUDIT_AMENDMENT if is_v3 else VLT_AUDIT_AMENDMENT
    active_ontology_path = LATENT_TRAIT_ONTOLOGY_V3 if is_v3 else LATENT_TRAIT_ONTOLOGY_V2
    active_runtime_path = (
        EXPECTED_RUNTIME_CONTRACT_V3 if is_v3 else EXPECTED_RUNTIME_CONTRACT_V2
    )
    vlt_audit_amendment_sha256 = files[str(active_audit_path)]["sha256"]
    public_ontology_sha256 = files[str(PUBLIC_DOMAIN_ONTOLOGY)]["sha256"]
    v1_latent_trait_ontology_sha256 = files[
        str(LATENT_TRAIT_ONTOLOGY_V1)
    ]["sha256"]
    v2_latent_trait_ontology_sha256 = files[
        str(LATENT_TRAIT_ONTOLOGY_V2)
    ]["sha256"]
    latent_trait_ontology_sha256 = files[str(active_ontology_path)]["sha256"]
    expected_contract_sha256 = files[str(active_runtime_path)]["sha256"]

    lineage = {
        "original_preregistration": {
            "sha256": prereg_hash,
            "parent_sha256": None,
        },
        "protocol_amendment": {
            "sha256": amendment_sha256,
            "parent_sha256": prereg_hash,
        },
        "candidate_registration": {
            "sha256": registration_sha256,
            "parent_sha256": amendment_sha256,
        },
        "routing_integrity_amendment": {
            "sha256": routing_amendment_sha256,
            "parent_sha256": registration_sha256,
        },
        "latent_scope_amendment": {
            "sha256": latent_scope_amendment_sha256,
            "parent_sha256": routing_amendment_sha256,
        },
        "historical_expected_runtime_contract": {
            "sha256": historical_runtime_sha256,
            "parent_sha256": latent_scope_amendment_sha256,
        },
        "latent_trait_ontology_v1": {
            "sha256": v1_latent_trait_ontology_sha256,
            "parent_sha256": historical_runtime_sha256,
        },
        "safe_transfer_amendment_v1": {
            "sha256": safe_transfer_amendment_sha256,
            "parent_sha256s": [
                latent_scope_amendment_sha256,
                historical_runtime_sha256,
                v1_latent_trait_ontology_sha256,
            ],
        },
        "expected_runtime_contract_vlt1": {
            "sha256": v1_runtime_sha256,
            "parent_sha256": safe_transfer_amendment_sha256,
        },
        "latent_trait_ontology_v2": {
            "sha256": v2_latent_trait_ontology_sha256,
            "parent_sha256": v1_latent_trait_ontology_sha256,
        },
    }
    v3_fields: dict[str, Any] = {}
    if is_v3:
        r3_rubric_sha256 = files[str(R3_RUBRIC)]["sha256"]
        scope_sha256 = files[str(R3_METHOD_SCOPE_CLARIFICATION)]["sha256"]
        prompt_sha256 = files[str(R3_ACTION_PROMPT_AMENDMENT)]["sha256"]
        target_free_report_sha256 = files[str(TARGET_FREE_VLT3_REPORT)]["sha256"]
        target_free_script_sha256 = files[str(TARGET_FREE_VLT3_DIAGNOSTICS)]["sha256"]
        lineage.update(
            {
                "r3_rubric": {
                    "sha256": r3_rubric_sha256,
                    "parent_sha256": prereg_hash,
                },
                "r3_method_scope_clarification": {
                    "sha256": scope_sha256,
                    "parent_sha256s": [
                        r3_rubric_sha256,
                        prereg_hash,
                        safe_transfer_amendment_sha256,
                        v2_latent_trait_ontology_sha256,
                    ],
                },
                "r3_action_prompt_amendment": {
                    "sha256": prompt_sha256,
                    "parent_sha256s": [r3_rubric_sha256, scope_sha256],
                },
                "target_free_vlt3_diagnostics": {
                    "sha256": target_free_report_sha256,
                    "parent_sha256s": [
                        prereg_hash,
                        latent_trait_ontology_sha256,
                        files["artifacts/history.sanitized.jsonl"]["sha256"],
                    ],
                },
                "vlt3_audit_amendment": {
                    "sha256": vlt_audit_amendment_sha256,
                    "parent_sha256s": [
                        safe_transfer_amendment_sha256,
                        v1_runtime_sha256,
                        v2_latent_trait_ontology_sha256,
                        r3_rubric_sha256,
                        scope_sha256,
                        prompt_sha256,
                        target_free_report_sha256,
                    ],
                },
                "expected_runtime_contract_vlt3": {
                    "sha256": expected_contract_sha256,
                    "parent_sha256s": [
                        v1_runtime_sha256,
                        vlt_audit_amendment_sha256,
                        scope_sha256,
                        prompt_sha256,
                        target_free_report_sha256,
                    ],
                },
            }
        )
        v3_fields = {
            "v2_latent_trait_ontology_sha256": v2_latent_trait_ontology_sha256,
            "r3_rubric_sha256": r3_rubric_sha256,
            "r3_method_scope_clarification_sha256": scope_sha256,
            "r3_action_prompt_amendment_sha256": prompt_sha256,
            "vlt3_audit_amendment_sha256": vlt_audit_amendment_sha256,
            "target_free_vlt3_diagnostics_script_sha256": target_free_script_sha256,
            "target_free_vlt3_report_sha256": target_free_report_sha256,
        }
        if is_recovery:
            recovery_amendment_sha256 = files[
                str(RUNTIME_RECOVERY_AMENDMENT)
            ]["sha256"]
            pid_diagnostic_sha256 = files[
                str(PID_NAMESPACE_TOPOLOGY_DIAGNOSTIC)
            ]["sha256"]
            parent_implementation_sha256 = files[
                str(PARENT_IMPLEMENTATION_MANIFEST)
            ]["sha256"]
            lineage.update(
                {
                    "parent_sealed_implementation": {
                        "sha256": parent_implementation_sha256,
                        "parent_sha256": vlt_audit_amendment_sha256,
                    },
                    "pid_namespace_topology_diagnostic": {
                        "sha256": pid_diagnostic_sha256,
                        "parent_sha256": parent_implementation_sha256,
                    },
                    "runtime_recovery_amendment": {
                        "sha256": recovery_amendment_sha256,
                        "parent_sha256s": [
                            parent_implementation_sha256,
                            pid_diagnostic_sha256,
                        ],
                    },
                }
            )
            v3_fields.update(
                {
                    "parent_implementation_manifest_sha256": (
                        parent_implementation_sha256
                    ),
                    "pid_namespace_topology_diagnostic_sha256": (
                        pid_diagnostic_sha256
                    ),
                    "runtime_recovery_amendment_sha256": (
                        recovery_amendment_sha256
                    ),
                    "runtime_recovery_only": True,
                }
            )
    else:
        lineage.update(
            {
                "vlt_audit_amendment_v2": {
                    "sha256": vlt_audit_amendment_sha256,
                    "parent_sha256s": [
                        safe_transfer_amendment_sha256,
                        v1_runtime_sha256,
                        v1_latent_trait_ontology_sha256,
                    ],
                },
                "expected_runtime_contract_vlt2": {
                    "sha256": expected_contract_sha256,
                    "parent_sha256s": [
                        v1_runtime_sha256,
                        vlt_audit_amendment_sha256,
                    ],
                },
            }
        )
    manifest = {
        "schema_version": 4 if is_recovery else (3 if is_v3 else 2),
        "kind": "immutable_influence_manifest",
        "excludes": [
            str(IMPLEMENTATION_MANIFEST),
            str(ATTEMPT_LOCK),
            "reports/final_attempts/*.terminal.json",
            "reports/GPU_BLOCKER.json",
            "reports/summary.json",
            "artifacts/predictions.*",
            "artifacts/memory.*",
            "artifacts/provider_calls.*",
            "logs/*",
            "**/__pycache__/*",
        ],
        "routing_integrity_amendment_sha256": routing_amendment_sha256,
        "latent_scope_amendment_sha256": latent_scope_amendment_sha256,
        "historical_expected_runtime_contract_sha256": historical_runtime_sha256,
        "safe_transfer_amendment_sha256": safe_transfer_amendment_sha256,
        "v1_expected_runtime_contract_sha256": v1_runtime_sha256,
        "v1_latent_trait_ontology_sha256": v1_latent_trait_ontology_sha256,
        "vlt_audit_amendment_sha256": vlt_audit_amendment_sha256,
        "latent_trait_ontology_sha256": latent_trait_ontology_sha256,
        "candidate_revision": (
            ACTIVE_CANDIDATE_REVISION if is_v3 else VLT2_CANDIDATE_REVISION
        ),
        "public_domain_ontology_sha256": public_ontology_sha256,
        "files": files,
        "inventory_sha256": _digest_json(files),
        "routing_coverage_evidence": routing_coverage_evidence,
        "lineage": lineage,
        "preregistration_sha256": prereg_hash,
        "protocol_amendment_sha256": amendment_sha256,
        "candidate_registration_sha256": registration_sha256,
        "expected_runtime_contract_sha256": expected_contract_sha256,
        "preference_slots_sha256": files["configs/preference_slots.json"][
            "sha256"
        ],
        "preference_slot_domain_count": len(preference_slots),
        "frozen_external_inputs": external_inputs,
        "frozen_external_inputs_sha256": _digest_json(external_inputs),
        "sealed_artifacts": sealed_artifacts,
        "expected_contract": expected_contract,
        "model_contract_sha256": _digest_json(expected_contract["model"]),
        "runtime_contract_sha256": _digest_json(expected_contract["runtime"]),
        "pipeline_contract_sha256": _digest_json(expected_contract["pipeline"]),
        "routing_gate_contract_sha256": _digest_json(
            expected_contract["routing_gate"]
        ),
        "candidate_memory_scope_contract_sha256": _digest_json(
            expected_contract["candidate_memory_scope"]
        ),
        "safe_transfer_contract_sha256": _digest_json(
            expected_contract["safe_transfer_contract"]
        ),
        "action_budget_sha256": _digest_json(
            expected_contract["action_budget"]
        ),
        "candidate_policy_sha256": _digest_json(
            expected_contract["candidate_policy"]
        ),
        "candidate": registration["candidate_id"],
        "confirmatory_ablation_flags": registration[
            "confirmatory_ablation_flags"
        ],
    }
    manifest.update(v3_fields)
    return manifest


def seal_implementation(root: Path) -> dict[str, Any]:
    manifest = build_implementation_manifest(root)
    commit_json_once(root.resolve() / IMPLEMENTATION_MANIFEST, manifest)
    return manifest


def validate_implementation_manifest(root: Path) -> tuple[dict[str, Any], str]:
    root = root.resolve()
    path = root / IMPLEMENTATION_MANIFEST
    if path.is_symlink() or not path.is_file():
        raise ValueError("immutable implementation manifest is missing")
    manifest = load_json(path)
    expected = build_implementation_manifest(root)
    if manifest != expected:
        raise ValueError("immutable implementation manifest content/hash chain mismatch")
    return manifest, sha256_file(path)


def build_attempt_binding(root: Path) -> dict[str, Any]:
    manifest, manifest_digest = validate_implementation_manifest(root)
    binding = {
        "implementation_manifest_sha256": manifest_digest,
        "inventory_sha256": manifest["inventory_sha256"],
        "preregistration_sha256": manifest["preregistration_sha256"],
        "protocol_amendment_sha256": manifest[
            "protocol_amendment_sha256"
        ],
        "candidate_registration_sha256": manifest[
            "candidate_registration_sha256"
        ],
        "routing_integrity_amendment_sha256": manifest[
            "routing_integrity_amendment_sha256"
        ],
        "latent_scope_amendment_sha256": manifest[
            "latent_scope_amendment_sha256"
        ],
        "historical_expected_runtime_contract_sha256": manifest[
            "historical_expected_runtime_contract_sha256"
        ],
        "safe_transfer_amendment_sha256": manifest[
            "safe_transfer_amendment_sha256"
        ],
        "v1_expected_runtime_contract_sha256": manifest[
            "v1_expected_runtime_contract_sha256"
        ],
        "vlt_audit_amendment_sha256": manifest[
            "vlt_audit_amendment_sha256"
        ],
        "latent_trait_ontology_sha256": manifest[
            "latent_trait_ontology_sha256"
        ],
        "candidate_revision": manifest["candidate_revision"],
        "candidate_memory_scope_contract_sha256": manifest[
            "candidate_memory_scope_contract_sha256"
        ],
        "safe_transfer_contract_sha256": manifest[
            "safe_transfer_contract_sha256"
        ],
        "routing_coverage_evidence": manifest[
            "routing_coverage_evidence"
        ],
        "public_domain_ontology_sha256": manifest[
            "public_domain_ontology_sha256"
        ],
        "routing_gate_contract_sha256": manifest[
            "routing_gate_contract_sha256"
        ],
        "expected_runtime_contract_sha256": manifest[
            "expected_runtime_contract_sha256"
        ],
        "model_contract_sha256": manifest["model_contract_sha256"],
        "runtime_contract_sha256": manifest["runtime_contract_sha256"],
        "pipeline_contract_sha256": manifest["pipeline_contract_sha256"],
        "action_budget_sha256": manifest["action_budget_sha256"],
        "tasks_sha256": manifest["sealed_artifacts"]["tasks_sha256"],
        "history_sha256": manifest["sealed_artifacts"]["history_sha256"],
        "sealed_evaluator_manifest_sha256": manifest["sealed_artifacts"][
            "sealed_manifest_sha256"
        ],
        "declared_gold_sha256": manifest["sealed_artifacts"][
            "declared_gold_sha256"
        ],
    }
    for key in (
        "v2_latent_trait_ontology_sha256",
        "r3_rubric_sha256",
        "r3_method_scope_clarification_sha256",
        "r3_action_prompt_amendment_sha256",
        "vlt3_audit_amendment_sha256",
        "target_free_vlt3_diagnostics_script_sha256",
        "target_free_vlt3_report_sha256",
    ):
        if key in manifest:
            binding[key] = manifest[key]
    return binding


def acquire_final_attempt(
    root: Path,
    binding: dict[str, Any] | None = None,
    *,
    attempt_id_factory: Callable[[], Any] = uuid.uuid4,
    clock: Callable[[], int] = time.time_ns,
) -> dict[str, Any]:
    root = root.resolve()
    expected_binding = build_attempt_binding(root)
    if binding is not None and binding != expected_binding:
        raise ValueError("requested final attempt binding is stale or incomplete")
    attempt_id = str(uuid.UUID(str(attempt_id_factory())))
    record = {
        "schema_version": 1,
        "kind": "sealed_final_attempt",
        "attempt_id": attempt_id,
        "acquired_unix_ns": int(clock()),
        "binding": expected_binding,
    }
    commit_json_once(root / ATTEMPT_LOCK, record)
    return record


def validate_final_attempt(root: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    root = root.resolve()
    manifest, manifest_digest = validate_implementation_manifest(root)
    attempt_path = root / ATTEMPT_LOCK
    if not attempt_path.is_file():
        raise ValueError("sealed final operation requires a committed final attempt")
    attempt = load_json(attempt_path)
    expected_fields = {
        "schema_version",
        "kind",
        "attempt_id",
        "acquired_unix_ns",
        "binding",
    }
    if attempt.get("schema_version") == 2:
        expected_fields.add("runner_nonce_sha256")
    if set(attempt) != expected_fields:
        raise ValueError("final attempt field contract violation")
    if attempt.get("schema_version") not in {1, 2}:
        raise ValueError("unsupported final attempt schema version")
    if attempt.get("schema_version") == 2:
        _require_sha256(attempt.get("runner_nonce_sha256"), "runner nonce digest")
    if attempt.get("kind") != "sealed_final_attempt":
        raise ValueError("invalid final attempt record")
    try:
        attempt_id = str(uuid.UUID(str(attempt.get("attempt_id"))))
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError("invalid final attempt ID") from exc
    if attempt_id != attempt.get("attempt_id"):
        raise ValueError("final attempt ID is not canonical")
    if not isinstance(attempt.get("acquired_unix_ns"), int):
        raise ValueError("invalid final attempt acquisition timestamp")
    expected_binding = build_attempt_binding(root)
    if expected_binding["implementation_manifest_sha256"] != manifest_digest:
        raise ValueError("current manifest digest changed during validation")
    if attempt.get("binding") != expected_binding:
        raise ValueError("attempt does not bind the current immutable run seal")
    return attempt, manifest, manifest_digest


def attempt_terminal_path(root: Path, attempt_id: str) -> Path:
    canonical_attempt_id = str(uuid.UUID(str(attempt_id)))
    if canonical_attempt_id != attempt_id:
        raise ValueError("attempt ID must be canonical")
    return (
        root.resolve()
        / "reports"
        / "final_attempts"
        / f"{canonical_attempt_id}.terminal.json"
    )


def commit_attempt_terminal(
    root: Path,
    attempt_id: str,
    outcome: str,
    detail: str = "",
    *,
    clock: Callable[[], int] = time.time_ns,
) -> dict[str, Any]:
    if outcome not in TERMINAL_OUTCOMES:
        raise ValueError(f"invalid final-attempt terminal outcome: {outcome}")
    root = root.resolve()
    attempt_path = root / ATTEMPT_LOCK
    if attempt_path.is_symlink() or not attempt_path.is_file():
        raise ValueError("cannot write terminal record without immutable attempt")
    attempt = load_json(attempt_path)
    if attempt.get("kind") != "sealed_final_attempt":
        raise ValueError("invalid final attempt record")
    if attempt.get("attempt_id") != attempt_id:
        raise ValueError("terminal record attempt ID mismatch")
    record = {
        "schema_version": 1,
        "kind": "sealed_final_attempt_terminal",
        "attempt_id": attempt_id,
        "attempt_record_sha256": sha256_file(attempt_path),
        "implementation_manifest_sha256": attempt.get("binding", {}).get(
            "implementation_manifest_sha256"
        ),
        "outcome": outcome,
        "detail": str(detail)[-2000:],
        "completed_unix_ns": int(clock()),
    }
    commit_json_once(attempt_terminal_path(root, attempt_id), record)
    return record


def validate_attempt_terminal(root: Path, attempt_id: str) -> dict[str, Any]:
    root = root.resolve()
    path = attempt_terminal_path(root, attempt_id)
    if path.is_symlink() or not path.is_file():
        raise ValueError("attempt-specific terminal record is missing")
    terminal = load_json(path)
    expected_fields = {
        "schema_version",
        "kind",
        "attempt_id",
        "attempt_record_sha256",
        "implementation_manifest_sha256",
        "outcome",
        "detail",
        "completed_unix_ns",
    }
    if set(terminal) != expected_fields:
        raise ValueError("attempt terminal field contract violation")
    attempt_path = root / ATTEMPT_LOCK
    attempt = load_json(attempt_path)
    if terminal.get("kind") != "sealed_final_attempt_terminal":
        raise ValueError("attempt terminal kind mismatch")
    if terminal.get("attempt_id") != attempt_id:
        raise ValueError("attempt terminal ID mismatch")
    if terminal.get("attempt_record_sha256") != sha256_file(attempt_path):
        raise ValueError("attempt terminal record binding mismatch")
    if terminal.get("implementation_manifest_sha256") != attempt.get(
        "binding", {}
    ).get("implementation_manifest_sha256"):
        raise ValueError("attempt terminal seal binding mismatch")
    if terminal.get("outcome") not in TERMINAL_OUTCOMES:
        raise ValueError("attempt terminal outcome mismatch")
    return terminal
