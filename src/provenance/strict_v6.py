"""Strict V6 provenance/admission/copy mechanisms.

The module is data-only: it never enumerates protected experiment roots and
never copies bytes. G0 supplies sealed typed records; these functions validate
them. That also makes every mechanism testable with disposable fixtures.
"""

from __future__ import annotations

import base64
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Sequence


class V6ContractError(RuntimeError):
    """A strict V6 predicate could not be proved."""


LEGAL_STATES = frozenset({"UNRESOLVED", "ADMITTED_FOR_COPY", "VERIFIED"})
PHASE_POSITIVE_STATE = {"precopy": "ADMITTED_FOR_COPY", "final": "VERIFIED"}
PROTECTED_SOURCE_ROOTS = frozenset({"experiments4", "experiments5", "experiments6"})
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

AUDIT_UNRESOLVED_RELATIVE = Path(
    "paper_outputs/admission/audits/unresolved-v1/results.jsonl"
)
AUDIT_DEFECTS_RELATIVE = Path(
    "paper_outputs/admission/audits/legacy-v1-defects/defects.jsonl"
)
AUDIT_UNRESOLVED_SHA256 = (
    "d408eef89d648ce2f03377b17f94f707aaeb68ad56174343c9a2c74414d58d09"
)
AUDIT_DEFECTS_SHA256 = (
    "eafde9d16051be316cbdb8240d01e417f5177aeec0c03a4930c93106c38117cf"
)
AUDIT_UNRESOLVED_ID_SET_SHA256 = (
    "bc962b8458256fe867fad56c3a2cecb7004d878647b5f3a6ffa2e7396acb60bb"
)
DRIFT_ONLY_RAW_ID = (
    "raw:e115af70a3788565a20a1c4a3741bfd81a2dfb2c820a12faa48d781f41c4812d"
)
DRIFT_ONLY_RAW_SHA256 = (
    "59f9d18975a60b513626c3058e76182eb1a5a08d970f099ad170c93287a30bb9"
)

TABLE10_INCOMPLETE_PARENT_RESULT_IDS = frozenset(
    {
        "result:t10-context-free-codegemma-7b-induction-f1:5d3cb6281ec6",
        "result:t10-context-free-codegemma-7b-induction-precision:62ba9554665a",
        "result:t10-context-free-codegemma-7b-induction-recall:51b29892f1a3",
        "result:t10-context-free-codegemma-7b-transfer-f1:75fdda66515e",
        "result:t10-context-free-codegemma-7b-transfer-precision:8828d764a09f",
        "result:t10-context-free-codegemma-7b-transfer-recall:54dce82fafa0",
        "result:t10-context-free-gemini-3-flash-induction-f1:a7f78f9e5d48",
        "result:t10-context-free-gemini-3-flash-induction-precision:1b0fed88c0d4",
        "result:t10-context-free-gemini-3-flash-induction-recall:ed5b8db1ef5e",
        "result:t10-context-free-gemini-3-flash-recall-f1:55542d45af9b",
        "result:t10-context-free-gemini-3-flash-recall-precision:c2d43aa529f9",
        "result:t10-context-free-gemini-3-flash-recall-recall:a512d7e34bc0",
        "result:t10-context-free-gemini-3-flash-transfer-f1:a09f4d163f46",
        "result:t10-context-free-gemini-3-flash-transfer-precision:e9b5666779da",
        "result:t10-context-free-gemini-3-flash-transfer-recall:622640f7b151",
        "result:t10-context-free-gpt-5-mini-transfer-f1:f4972bbc2d38",
        "result:t10-context-free-gpt-5-mini-transfer-precision:ab0a305d7da7",
        "result:t10-context-free-gpt-5-mini-transfer-recall:d54650689d2b",
        "result:t10-context-free-r1-distill-llama-8b-induction-f1:41c1425b6658",
        "result:t10-context-free-r1-distill-llama-8b-induction-precision:60f7c0581654",
        "result:t10-context-free-r1-distill-llama-8b-induction-recall:605499cb5f3e",
        "result:t10-context-free-r1-distill-qwen-7b-induction-f1:aec056f6d50b",
        "result:t10-context-free-r1-distill-qwen-7b-induction-precision:b89f66083ef4",
        "result:t10-context-free-r1-distill-qwen-7b-induction-recall:e04e6ff0bf11",
        "result:t10-context-free-r1-distill-qwen-7b-recall-f1:634ce58e1861",
        "result:t10-context-free-r1-distill-qwen-7b-recall-precision:c3b35ca008e5",
        "result:t10-context-free-r1-distill-qwen-7b-recall-recall:22678f672c93",
        "result:t10-context-free-r1-distill-qwen-7b-transfer-f1:9b514098266f",
        "result:t10-context-free-r1-distill-qwen-7b-transfer-precision:246754d06f20",
        "result:t10-context-free-r1-distill-qwen-7b-transfer-recall:60c9cd580ea0",
        "result:t10-context-guided-gemini-3-flash-transfer-ea-f1:30f18c7af625",
        "result:t10-context-guided-gemini-3-flash-transfer-oa-f1:7300d4372838",
        "result:t10-context-guided-gemini-3-flash-transfer-p-em:94be90953166",
        "result:t10-context-guided-gemma-3-12b-transfer-p-em:1d09c15a1fb7",
    }
)
FIGURE5_ZERO_RAW_RESULT_IDS = frozenset(
    {
        "result:f5-derived-baseline-reduction-range:1c65cf14e6cb",
        "result:f5-derived-prefine-over-base-ratio:3cd5130f988f",
        "result:f5-derived-session-token-range:9e117e92da60",
    }
)

CLOSURE_FIELDS = (
    "producer_node_ids",
    "config_node_ids",
    "script_node_ids",
    "runtime_node_ids",
    "input_node_ids",
    "recomputation_node_ids",
    "rounding_proof_node_ids",
)
PROVENANCE_NODE_SCHEMA = "experiments7-provenance-node/v6"
PROVENANCE_EDGE_SCHEMA = "experiments7-provenance-edge/v6"
PROVENANCE_RECOMPUTATION_SCHEMA = "experiments7-provenance-recomputation/v6"
PROVENANCE_ROUNDING_SCHEMA = "experiments7-provenance-rounding-proof/v6"
PROVENANCE_NODE_TYPES = frozenset(
    {"producer", "config", "script", "runtime", "input", "raw_artifact"}
)
ARTIFACT_PROVENANCE_NODE_TYPES = PROVENANCE_NODE_TYPES - {"raw_artifact"}
RAW_PROVENANCE_NODE_KEYS = frozenset(
    {
        "schema",
        "node_id",
        "node_type",
        "raw_kind",
        "copy_required",
        "source_manifest_record_id",
        "source_pre_record_sha256",
        "root_id",
        "relative_path",
        "sha256",
        "size",
        "descriptor_identity",
        "declared_drift",
        "unrelated",
        "support_only",
        "derived_only",
    }
)
RAW_EXCLUSION_FIELDS = (
    "declared_drift",
    "unrelated",
    "support_only",
    "derived_only",
)
PROVENANCE_NODE_TYPE_BY_FIELD = {
    "producer_node_ids": "producer",
    "config_node_ids": "config",
    "script_node_ids": "script",
    "runtime_node_ids": "runtime",
    "input_node_ids": "input",
}
PROVENANCE_RELATION_BY_FIELD = {
    "producer_node_ids": "producer_supports_result",
    "config_node_ids": "config_supports_result",
    "script_node_ids": "script_supports_result",
    "runtime_node_ids": "runtime_supports_result",
    "input_node_ids": "input_supports_result",
    "raw_artifact_ids": "raw_artifact_supports_result",
    "recomputation_node_ids": "recomputation_supports_result",
    "rounding_proof_node_ids": "rounding_proof_supports_result",
    "closed_parent_result_ids": "parent_result_supports_result",
}
COPY_LEDGER_ROW_KEYS = frozenset(
    {
        "schema",
        "copy_plan_entry_sha256",
        "raw_artifact_id",
        "source_manifest_record_id",
        "source_root_id",
        "source_relative_path",
        "source_pre_record_sha256",
        "source_sha256",
        "size",
        "destination_relative_path",
        "copied_artifact_id",
        "copied_to_edge_id",
        "source_before",
        "source_after",
        "source_sha256_before",
        "source_sha256_after",
        "destination_identity",
        "publication_method",
        "hardlink_used",
        "reflink_used",
        "clone_used",
        "cache_used",
        "destination_sha256",
        "destination_size",
    }
)
ADMISSION_BOOLEAN_FIELDS = (
    "copy_required",
    "source_pre_bound",
    "declared_drift",
    "ambiguous",
    "support_only",
    "audit_positive_evidence",
)
ADMISSION_PRECOPY_KEYS = frozenset(
    {
        "schema",
        "result_id",
        "paper_section_id",
        "phase",
        "state",
        "result_kind",
        "copy_required",
        "raw_artifact_ids",
        "candidate_raw_artifact_ids",
        "required_parent_result_ids",
        "closed_parent_result_ids",
        "required_contributor_node_ids",
        "sealed_contributor_node_ids",
        "positive_evidence_origin",
        "source_pre_bound",
        "declared_drift",
        "ambiguous",
        "support_only",
        "audit_positive_evidence",
        "reason_codes",
        "copied_artifact_ids",
        "copied_to_edge_ids",
        *CLOSURE_FIELDS,
    }
)
DESCRIPTOR_IDENTITY_FIELDS = (
    "file_type",
    "st_dev",
    "st_ino",
    "mode",
    "size",
    "mtime_ns",
)
SOURCE_PRE_PATH_KEYS = frozenset(
    {
        "schema",
        "record_id",
        "root_id",
        "relative_path",
        "type",
        "sha256",
        "size",
        "descriptor_identity",
    }
)
SOURCE_PRE_B64_PATH_KEYS = (SOURCE_PRE_PATH_KEYS - {"relative_path"}) | {
    "relative_path_b64"
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def digest_records(records: Sequence[Mapping[str, Any]]) -> str:
    payload = "".join(canonical_json(record) + "\n" for record in records)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _required_string(record: Mapping[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise V6ContractError(f"{field} must be a nonempty string")
    return value


def _strings(record: Mapping[str, Any], field: str, *, nonempty: bool = False) -> list[str]:
    value = record.get(field, [])
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise V6ContractError(f"{field} must be a list of nonempty strings")
    if len(value) != len(set(value)):
        raise V6ContractError(f"{field} contains duplicates")
    if nonempty and not value:
        raise V6ContractError(f"{field} must not be empty")
    return list(value)


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise V6ContractError(f"{field} must be a lowercase SHA-256")
    return value


def _relative_path(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise V6ContractError(f"{field} is not a canonical relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise V6ContractError(f"{field} escapes its root")
    if path.as_posix() != value:
        raise V6ContractError(f"{field} is not canonical")
    return value


def _manifest_path(record: Mapping[str, Any]) -> str:
    if "relative_path" in record:
        return _relative_path(record["relative_path"], "relative_path")
    encoded = record.get("relative_path_b64")
    if not isinstance(encoded, str):
        raise V6ContractError("record lacks a relative path")
    try:
        decoded = base64.b64decode(encoded, validate=True).decode("utf-8")
    except (ValueError, UnicodeDecodeError) as exc:
        raise V6ContractError("relative_path_b64 is invalid") from exc
    return _relative_path(decoded, "relative_path_b64")


def _identity(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(DESCRIPTOR_IDENTITY_FIELDS):
        raise V6ContractError(f"{field} has incomplete descriptor identity")
    if value.get("file_type") != "regular":
        raise V6ContractError(f"{field} is not a regular file")
    for name in DESCRIPTOR_IDENTITY_FIELDS[1:]:
        if (
            not isinstance(value.get(name), int)
            or isinstance(value[name], bool)
            or value[name] < 0
        ):
            raise V6ContractError(f"{field}.{name} is not a nonnegative integer")
    return dict(value)


def validate_source_pre_records(
    source_pre_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate the exact typed CP0 source-pre record schema."""

    if not isinstance(source_pre_records, Sequence) or isinstance(
        source_pre_records, (str, bytes, bytearray)
    ):
        raise V6ContractError("source-pre records must be a sequence")
    if not source_pre_records:
        raise V6ContractError("source-pre records must not be empty")
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_paths: set[tuple[str, str]] = set()
    for source in source_pre_records:
        if not isinstance(source, Mapping):
            raise V6ContractError("source-pre record must be an object")
        if set(source) not in {SOURCE_PRE_PATH_KEYS, SOURCE_PRE_B64_PATH_KEYS}:
            raise V6ContractError("source-pre record schema keys differ")
        if source.get("schema") != "experiments7-source-pre/v6":
            raise V6ContractError("source-pre schema mismatch")
        record_id = _required_string(source, "record_id")
        if record_id in seen_ids:
            raise V6ContractError(f"duplicate source-pre record: {record_id}")
        seen_ids.add(record_id)
        root_id = _required_string(source, "root_id")
        if root_id not in PROTECTED_SOURCE_ROOTS:
            raise V6ContractError(f"legacy or non-protected source root: {root_id}")
        if source.get("type") != "regular":
            raise V6ContractError(f"source-pre record is not regular: {record_id}")
        relative_path = _manifest_path(source)
        path_key = (root_id, relative_path)
        if path_key in seen_paths:
            raise V6ContractError(f"duplicate source-pre path: {root_id}/{relative_path}")
        seen_paths.add(path_key)
        _sha256(source.get("sha256"), f"source-pre {record_id} sha256")
        size = source.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise V6ContractError(f"source-pre size is invalid: {record_id}")
        identity = _identity(
            source.get("descriptor_identity"), f"source-pre {record_id} identity"
        )
        if size != identity["size"]:
            raise V6ContractError(f"source-pre size/identity mismatch: {record_id}")
        normalized.append(json.loads(canonical_json(source)))
    return normalized


def _positive_failures(record: Mapping[str, Any], phase: str) -> list[str]:
    """Return every reason the record cannot hold the phase-positive state."""

    failures: list[str] = []
    result_kind = record.get("result_kind")
    if result_kind not in {"inference", "derived_inference", "non_inference"}:
        failures.append("unsupported_result_kind")
    if record.get("positive_evidence_origin") != "run_local_strict":
        failures.append("positive_evidence_not_run_local_strict")
    for field, reason in (
        ("declared_drift", "declared_drift"),
        ("ambiguous", "ambiguous_evidence"),
        ("support_only", "support_only_evidence"),
        ("audit_positive_evidence", "audit_used_as_positive_evidence"),
    ):
        value = record.get(field)
        if type(value) is not bool:
            failures.append(f"{field}_not_boolean")
        elif value:
            failures.append(reason)

    raw_ids = _strings(record, "raw_artifact_ids")
    candidate_ids = _strings(record, "candidate_raw_artifact_ids")
    if set(raw_ids) & set(candidate_ids):
        failures.append("candidate_promoted_to_raw")
    if DRIFT_ONLY_RAW_ID in raw_ids:
        failures.append("pinned_drift_only_raw")

    required_parents = _strings(record, "required_parent_result_ids")
    closed_parents = _strings(record, "closed_parent_result_ids")
    if set(required_parents) != set(closed_parents):
        failures.append("partial_parent_union")
    if result_kind == "derived_inference" and not required_parents:
        failures.append("derived_inference_without_parents")

    closure: dict[str, list[str]] = {}
    for field in CLOSURE_FIELDS:
        closure[field] = _strings(record, field)
        if not closure[field]:
            failures.append(f"missing_{field}")
    expected_contributors = set(raw_ids) | set(closed_parents)
    for values in closure.values():
        expected_contributors.update(values)
    if set(_strings(record, "required_contributor_node_ids")) != expected_contributors:
        failures.append("required_contributor_set_incomplete")
    if set(_strings(record, "sealed_contributor_node_ids")) != expected_contributors:
        failures.append("sealed_contributor_set_incomplete")

    copy_required = record.get("copy_required")
    source_pre_bound = record.get("source_pre_bound")
    if type(copy_required) is not bool:
        failures.append("copy_required_not_boolean")
    if type(source_pre_bound) is not bool:
        failures.append("source_pre_bound_not_boolean")
    if result_kind in {"inference", "derived_inference"}:
        if not copy_required or not raw_ids:
            failures.append("inference_without_copy_required_raw")
        if source_pre_bound is not True:
            failures.append("raw_not_source_pre_bound")
    elif result_kind == "non_inference" and (copy_required or raw_ids):
        failures.append("non_inference_has_raw_or_copy")
    elif result_kind == "non_inference" and source_pre_bound is not False:
        failures.append("non_inference_source_pre_binding")

    copied_ids = _strings(record, "copied_artifact_ids")
    copied_edges = _strings(record, "copied_to_edge_ids")
    if phase == "precopy" and (copied_ids or copied_edges):
        failures.append("precopy_contains_copy_evidence")
    if phase == "final":
        if raw_ids and (len(copied_ids) != len(raw_ids) or len(copied_edges) != len(raw_ids)):
            failures.append("final_copy_reference_cardinality")
        if not raw_ids and (copied_ids or copied_edges):
            failures.append("zero_raw_result_has_copy_evidence")
    return sorted(set(failures))


def derive_admission_state(record: Mapping[str, Any], phase: str) -> str:
    """Derive the state from typed evidence; never trust a claimed state."""

    if phase not in PHASE_POSITIVE_STATE:
        raise V6ContractError(f"unsupported admission phase: {phase!r}")
    return "UNRESOLVED" if _positive_failures(record, phase) else PHASE_POSITIVE_STATE[phase]


def validate_admission_record(record: Mapping[str, Any], phase: str) -> dict[str, Any]:
    result_id = _required_string(record, "result_id")
    expected_keys = set(ADMISSION_PRECOPY_KEYS)
    if phase == "final":
        expected_keys.add("precopy_record_sha256")
    extra_keys = set(record) - expected_keys
    if extra_keys:
        raise V6ContractError(
            f"{result_id}: admission keys differ: unexpected {sorted(extra_keys)}"
        )
    if record.get("schema") != "experiments7-admission-result/v6":
        raise V6ContractError(f"{result_id}: admission schema mismatch")
    _required_string(record, "paper_section_id")
    if record.get("phase") != phase:
        raise V6ContractError(f"{result_id}: phase mismatch")
    claimed = record.get("state")
    if claimed not in LEGAL_STATES:
        raise V6ContractError(f"{result_id}: illegal state {claimed!r}")
    if phase == "precopy" and claimed == "VERIFIED":
        raise V6ContractError(f"{result_id}: VERIFIED is final-only")
    if phase == "final" and claimed == "ADMITTED_FOR_COPY":
        raise V6ContractError(f"{result_id}: ADMITTED_FOR_COPY is precopy-only")
    for field in ADMISSION_BOOLEAN_FIELDS:
        if type(record.get(field)) is not bool:
            raise V6ContractError(f"{result_id}: {field} must be an explicit boolean")

    reason_codes = _strings(record, "reason_codes")
    if reason_codes != sorted(reason_codes):
        raise V6ContractError(f"{result_id}: reason_codes must be sorted and unique")

    derived = derive_admission_state(record, phase)
    raw_ids = _strings(record, "raw_artifact_ids")
    copied_ids = _strings(record, "copied_artifact_ids")
    copied_edges = _strings(record, "copied_to_edge_ids")
    if claimed == "UNRESOLVED":
        if raw_ids or record.get("copy_required") is not False:
            raise V6ContractError(f"{result_id}: unresolved raw belongs in candidates")
        if copied_ids or copied_edges:
            raise V6ContractError(f"{result_id}: unresolved result has copy edges")
        if not reason_codes:
            raise V6ContractError(f"{result_id}: reason_codes must not be empty")
    elif claimed != derived:
        raise V6ContractError(
            f"{result_id}: positive state forgery: {','.join(_positive_failures(record, phase))}"
        )
    elif reason_codes:
        raise V6ContractError(f"{result_id}: positive result reason_codes must be empty")
    missing_keys = expected_keys - set(record)
    if missing_keys:
        raise V6ContractError(
            f"{result_id}: admission keys differ: missing {sorted(missing_keys)}"
        )
    if phase == "final":
        _sha256(record.get("precopy_record_sha256"), f"{result_id} precopy record")
    return json.loads(canonical_json(record))


CP1_STRUCTURE_COUNT = 51
CP1_RESULT_COUNT = 1908
CP1_ALIAS_COUNT = 86
CP1_INVENTORY_SEAL_SCHEMA = "experiments7-cp1-inventory-seal/v6"


def _normalize_result_inventory(
    inventory: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for row in inventory:
        if not isinstance(row, Mapping):
            raise V6ContractError("inventory result row must be an object")
        result_id = _required_string(row, "result_id")
        _required_string(row, "paper_section_id")
        if result_id in seen:
            raise V6ContractError(f"duplicate inventory result: {result_id}")
        seen.add(result_id)
        try:
            normalized.append(json.loads(canonical_json(row)))
        except (TypeError, ValueError) as exc:
            raise V6ContractError(
                f"inventory result row is not canonical JSON: {result_id}"
            ) from exc
    return normalized


def result_inventory_digest(inventory: Sequence[Mapping[str, Any]]) -> str:
    """Bind the ordered result rows used by CP3 admission buckets."""

    return digest_records(_normalize_result_inventory(inventory))


def inventory_digest(
    structure: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    aliases: Sequence[Mapping[str, Any]],
) -> str:
    """Bind the exact CP1 51/1,908/86 inventory and its references."""

    if len(structure) != CP1_STRUCTURE_COUNT:
        raise V6ContractError("inventory structure must contain exactly 51 records")
    if len(results) != CP1_RESULT_COUNT:
        raise V6ContractError("inventory results must contain exactly 1908 records")
    if len(aliases) != CP1_ALIAS_COUNT:
        raise V6ContractError("inventory aliases must contain exactly 86 records")

    section_ids: set[str] = set()
    normalized_structure: list[dict[str, Any]] = []
    for row in structure:
        if not isinstance(row, Mapping):
            raise V6ContractError("inventory structure row must be an object")
        section_id = _required_string(row, "paper_section_id")
        if section_id in section_ids:
            raise V6ContractError(f"duplicate inventory structure: {section_id}")
        section_ids.add(section_id)
        try:
            normalized_structure.append(json.loads(canonical_json(row)))
        except (TypeError, ValueError) as exc:
            raise V6ContractError(
                f"inventory structure row is not canonical JSON: {section_id}"
            ) from exc

    normalized_results = _normalize_result_inventory(results)
    result_ids = {str(row["result_id"]) for row in normalized_results}
    for row in normalized_results:
        if row["paper_section_id"] not in section_ids:
            raise V6ContractError(
                f"inventory result references missing structure: {row['result_id']}"
            )

    alias_ids: set[str] = set()
    normalized_aliases: list[dict[str, Any]] = []
    for row in aliases:
        if not isinstance(row, Mapping):
            raise V6ContractError("inventory alias row must be an object")
        alias_id = _required_string(row, "alias_id")
        target_result_id = _required_string(row, "target_result_id")
        if alias_id in alias_ids:
            raise V6ContractError(f"duplicate inventory alias: {alias_id}")
        if target_result_id not in result_ids:
            raise V6ContractError(
                f"inventory alias references missing result: {alias_id}"
            )
        alias_ids.add(alias_id)
        try:
            normalized_aliases.append(json.loads(canonical_json(row)))
        except (TypeError, ValueError) as exc:
            raise V6ContractError(
                f"inventory alias row is not canonical JSON: {alias_id}"
            ) from exc

    return digest_json(
        {
            "schema": CP1_INVENTORY_SEAL_SCHEMA,
            "structure_count": len(normalized_structure),
            "structure_sha256": digest_records(normalized_structure),
            "result_count": len(normalized_results),
            "results_sha256": digest_records(normalized_results),
            "alias_count": len(normalized_aliases),
            "aliases_sha256": digest_records(normalized_aliases),
        }
    )


_DECIMAL_TEXT_RE = re.compile(r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$")


def _decimal_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or _DECIMAL_TEXT_RE.fullmatch(value) is None:
        raise V6ContractError(f"{field} must be a canonical decimal string")
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise V6ContractError(f"{field} must be a finite decimal") from exc
    if not parsed.is_finite() or _canonical_decimal(parsed) != value:
        raise V6ContractError(f"{field} is not canonical")
    return value


def _canonical_decimal(value: Decimal) -> str:
    if not value.is_finite():
        raise V6ContractError("decimal result is not finite")
    if value == 0:
        return "0"
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _json_pointer(value: Any, field: str) -> list[str | int] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise V6ContractError(f"{field} must be null or a JSON-pointer segment list")
    result: list[str | int] = []
    for segment in value:
        if type(segment) is int:
            if segment < 0:
                raise V6ContractError(f"{field} contains a negative index")
        elif isinstance(segment, str):
            if not segment or "\x00" in segment:
                raise V6ContractError(f"{field} contains an invalid key")
        else:
            raise V6ContractError(f"{field} contains an invalid segment")
        result.append(segment)
    return result


def _resolve_json_pointer(value: Any, pointer: Sequence[str | int], field: str) -> Any:
    current = value
    for segment in pointer:
        if isinstance(current, Mapping) and isinstance(segment, str) and segment in current:
            current = current[segment]
        elif isinstance(current, list) and type(segment) is int and segment < len(current):
            current = current[segment]
        else:
            raise V6ContractError(f"{field} does not resolve in its sealed artifact")
    return current


def _json_decimal(value: Any, field: str) -> str:
    if type(value) is int:
        return str(value)
    return _decimal_text(value, field)


def make_raw_provenance_node(
    source_pre_record: Mapping[str, Any],
    *,
    raw_kind: str = "raw_inference",
    copy_required: bool = True,
    declared_drift: bool = False,
    unrelated: bool = False,
    support_only: bool = False,
    derived_only: bool = False,
) -> dict[str, Any]:
    """Build one content-addressed raw node from an exact CP0 source-pre row."""

    records = validate_source_pre_records([source_pre_record])
    source = records[0]
    if raw_kind != "raw_inference":
        raise V6ContractError("raw provenance node must be raw_inference")
    if copy_required is not True:
        raise V6ContractError("raw provenance node must be copy-required")
    exclusions = {
        "declared_drift": declared_drift,
        "unrelated": unrelated,
        "support_only": support_only,
        "derived_only": derived_only,
    }
    if any(value is not False for value in exclusions.values()):
        raise V6ContractError("raw provenance node has an exclusion flag")
    material = {
        "node_type": "raw_artifact",
        "raw_kind": raw_kind,
        "copy_required": copy_required,
        "source_manifest_record_id": source["record_id"],
        "source_pre_record_sha256": digest_json(source),
        "root_id": source["root_id"],
        "relative_path": _manifest_path(source),
        "sha256": source["sha256"],
        "size": source["size"],
        "descriptor_identity": dict(source["descriptor_identity"]),
        **exclusions,
    }
    return {
        "schema": PROVENANCE_NODE_SCHEMA,
        "node_id": f"raw:{digest_json(material)}",
        **material,
    }


def _validate_raw_provenance_node(
    raw_node: Mapping[str, Any],
    source_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if set(raw_node) != RAW_PROVENANCE_NODE_KEYS:
        raise V6ContractError("raw provenance node schema keys differ")
    source_record_id = _required_string(raw_node, "source_manifest_record_id")
    source = source_by_id.get(source_record_id)
    if source is None:
        raise V6ContractError(
            f"raw provenance source-pre record is absent: {source_record_id}"
        )
    rebuilt = make_raw_provenance_node(
        source,
        raw_kind=raw_node.get("raw_kind"),
        copy_required=raw_node.get("copy_required"),
        declared_drift=raw_node.get("declared_drift"),
        unrelated=raw_node.get("unrelated"),
        support_only=raw_node.get("support_only"),
        derived_only=raw_node.get("derived_only"),
    )
    if dict(raw_node) != rebuilt:
        raise V6ContractError("raw provenance node is not content-addressed exactly")
    return rebuilt


def make_provenance_node(
    node_type: str,
    artifact_relative_path: str,
    artifact_evidence_sha256: str,
    *,
    json_pointer: Sequence[str | int] | None = None,
    numeric_value: str | None = None,
) -> dict[str, Any]:
    """Build one content-addressed contributor node bound to sealed file evidence."""

    if (
        not isinstance(node_type, str)
        or node_type not in ARTIFACT_PROVENANCE_NODE_TYPES
    ):
        raise V6ContractError("provenance node type is invalid")
    relative = _relative_path(artifact_relative_path, "artifact_relative_path")
    evidence_sha256 = _sha256(
        artifact_evidence_sha256, "artifact_evidence_sha256"
    )
    if json_pointer is None:
        pointer = None
    elif isinstance(json_pointer, Sequence) and not isinstance(
        json_pointer, (str, bytes, bytearray)
    ):
        pointer = _json_pointer(list(json_pointer), "json_pointer")
    else:
        raise V6ContractError("json_pointer must be a segment sequence")
    if node_type == "input":
        if pointer is None:
            raise V6ContractError("input provenance node requires a JSON pointer")
        numeric = _decimal_text(numeric_value, "numeric_value")
    else:
        if pointer is not None or numeric_value is not None:
            raise V6ContractError(
                "only input provenance nodes may carry numeric extraction fields"
            )
        numeric = None
    material = {
        "node_type": node_type,
        "artifact_relative_path": relative,
        "artifact_evidence_sha256": evidence_sha256,
        "json_pointer": pointer,
        "numeric_value": numeric,
    }
    return {
        "schema": PROVENANCE_NODE_SCHEMA,
        "node_id": f"node:{digest_json(material)}",
        **material,
    }


def make_provenance_edge(
    source_id: str, target_result_id: str, relation: str
) -> dict[str, Any]:
    """Build one content-addressed exact contributor-to-result edge."""

    if not isinstance(source_id, str) or not source_id:
        raise V6ContractError("provenance edge source_id is invalid")
    if not isinstance(target_result_id, str) or not target_result_id:
        raise V6ContractError("provenance edge target_result_id is invalid")
    if (
        not isinstance(relation, str)
        or relation not in set(PROVENANCE_RELATION_BY_FIELD.values())
    ):
        raise V6ContractError("provenance edge relation is invalid")
    material = {
        "source_id": source_id,
        "target_result_id": target_result_id,
        "relation": relation,
    }
    return {
        "schema": PROVENANCE_EDGE_SCHEMA,
        "edge_id": f"edge:{digest_json(material)}",
        **material,
    }


def make_provenance_recomputation(
    terms: Sequence[Mapping[str, Any]], denominator: str
) -> dict[str, Any]:
    """Build a deterministic decimal weighted-sum/divide recomputation."""

    if not isinstance(terms, Sequence) or isinstance(
        terms, (str, bytes, bytearray)
    ):
        raise V6ContractError("recomputation terms must be a sequence")
    normalized_terms: list[dict[str, str]] = []
    for term in terms:
        if not isinstance(term, Mapping) or set(term) != {
            "input_node_id",
            "coefficient",
        }:
            raise V6ContractError("recomputation term schema differs")
        input_node_id = _required_string(term, "input_node_id")
        coefficient = _decimal_text(term.get("coefficient"), "coefficient")
        normalized_terms.append(
            {"input_node_id": input_node_id, "coefficient": coefficient}
        )
    if not normalized_terms:
        raise V6ContractError("recomputation terms must not be empty")
    if normalized_terms != sorted(
        normalized_terms, key=lambda row: row["input_node_id"]
    ) or len({row["input_node_id"] for row in normalized_terms}) != len(
        normalized_terms
    ):
        raise V6ContractError("recomputation terms must be sorted and unique")
    denominator_text = _decimal_text(denominator, "denominator")
    if Decimal(denominator_text) == 0:
        raise V6ContractError("recomputation denominator is zero")
    material = {
        "operation": "decimal_weighted_sum_divide",
        "terms": normalized_terms,
        "denominator": denominator_text,
    }
    return {
        "schema": PROVENANCE_RECOMPUTATION_SCHEMA,
        "recomputation_id": f"recomputation:{digest_json(material)}",
        **material,
    }


def make_provenance_rounding_proof(
    recomputation_id: str,
    unrounded_numerator: int,
    unrounded_denominator: int,
    decimal_places: int,
) -> dict[str, Any]:
    """Build one exact rational ROUND_HALF_UP paper-display proof."""

    if not isinstance(recomputation_id, str) or not recomputation_id:
        raise V6ContractError("rounding recomputation_id is invalid")
    if type(unrounded_numerator) is not int:
        raise V6ContractError("unrounded_numerator must be an exact integer")
    if type(unrounded_denominator) is not int or unrounded_denominator <= 0:
        raise V6ContractError("unrounded_denominator must be a positive integer")
    if math.gcd(abs(unrounded_numerator), unrounded_denominator) != 1:
        raise V6ContractError("unrounded rational must be reduced")
    if type(decimal_places) is not int or not 0 <= decimal_places <= 18:
        raise V6ContractError("decimal_places must be an exact integer in [0, 18]")
    scale = 10**decimal_places
    quotient, remainder = divmod(abs(unrounded_numerator) * scale, unrounded_denominator)
    if remainder * 2 >= unrounded_denominator:
        quotient += 1
    scaled = -quotient if unrounded_numerator < 0 else quotient
    sign = "-" if unrounded_numerator < 0 else ""
    digits = str(abs(scaled))
    if decimal_places:
        digits = digits.zfill(decimal_places + 1)
        rendered = f"{sign}{digits[:-decimal_places]}.{digits[-decimal_places:]}"
    else:
        rendered = f"{sign}{digits}"
    material = {
        "recomputation_id": recomputation_id,
        "unrounded_numerator": unrounded_numerator,
        "unrounded_denominator": unrounded_denominator,
        "decimal_places": decimal_places,
        "rounding_mode": "ROUND_HALF_UP",
        "rounded_decimal": rendered,
        "display_value": rendered,
    }
    return {
        "schema": PROVENANCE_ROUNDING_SCHEMA,
        "rounding_proof_id": f"rounding:{digest_json(material)}",
        **material,
    }


def validate_cp3_provenance(
    inventory: Sequence[Mapping[str, Any]],
    admission_records: Sequence[Mapping[str, Any]],
    node_rows: Sequence[Mapping[str, Any]],
    edge_rows: Sequence[Mapping[str, Any]],
    recomputation_rows: Sequence[Mapping[str, Any]],
    rounding_rows: Sequence[Mapping[str, Any]],
    source_pre_records: Sequence[Mapping[str, Any]],
    artifact_bindings: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate exact typed graph closure and deterministic paper numerics."""

    result_inventory_digest(inventory)
    inventory_by_id: dict[str, Mapping[str, Any]] = {
        _required_string(row, "result_id"): row for row in inventory
    }
    records_by_id: dict[str, dict[str, Any]] = {}
    for raw_record in admission_records:
        record = validate_admission_record(raw_record, "precopy")
        result_id = record["result_id"]
        if result_id in records_by_id:
            raise V6ContractError(f"duplicate admission record: {result_id}")
        records_by_id[result_id] = record
    if set(records_by_id) != set(inventory_by_id):
        raise V6ContractError("provenance admission set is not the exact inventory")
    source_by_id = _index_source_pre_records(source_pre_records)

    nodes: dict[str, dict[str, Any]] = {}
    for raw_node in node_rows:
        if not isinstance(raw_node, Mapping):
            raise V6ContractError("provenance node must be an object")
        if raw_node.get("node_type") == "raw_artifact":
            rebuilt = _validate_raw_provenance_node(raw_node, source_by_id)
        else:
            if set(raw_node) != {
                "schema",
                "node_id",
                "node_type",
                "artifact_relative_path",
                "artifact_evidence_sha256",
                "json_pointer",
                "numeric_value",
            }:
                raise V6ContractError("provenance node schema keys differ")
            rebuilt = make_provenance_node(
                raw_node.get("node_type"),
                raw_node.get("artifact_relative_path"),
                raw_node.get("artifact_evidence_sha256"),
                json_pointer=raw_node.get("json_pointer"),
                numeric_value=raw_node.get("numeric_value"),
            )
        if dict(raw_node) != rebuilt:
            raise V6ContractError("provenance node is not content-addressed exactly")
        node_id = rebuilt["node_id"]
        if node_id in nodes:
            raise V6ContractError(f"duplicate provenance node: {node_id}")
        if rebuilt["node_type"] != "raw_artifact":
            binding = artifact_bindings.get(rebuilt["artifact_relative_path"])
            if (
                not isinstance(binding, Mapping)
                or set(binding) != {"evidence_sha256", "json_value"}
                or binding.get("evidence_sha256")
                != rebuilt["artifact_evidence_sha256"]
            ):
                raise V6ContractError(f"unsealed provenance artifact: {node_id}")
            if rebuilt["node_type"] == "input":
                pointed = _resolve_json_pointer(
                    binding.get("json_value"),
                    rebuilt["json_pointer"],
                    f"input node {node_id}",
                )
                if _json_decimal(pointed, f"input node {node_id}") != rebuilt["numeric_value"]:
                    raise V6ContractError(f"input numeric extraction differs: {node_id}")
        nodes[node_id] = rebuilt

    recomputations: dict[str, dict[str, Any]] = {}
    recomputed_values: dict[str, Fraction] = {}
    for raw_recomputation in recomputation_rows:
        if not isinstance(raw_recomputation, Mapping) or set(raw_recomputation) != {
            "schema",
            "recomputation_id",
            "operation",
            "terms",
            "denominator",
        }:
            raise V6ContractError("recomputation schema keys differ")
        rebuilt = make_provenance_recomputation(
            raw_recomputation.get("terms", []), raw_recomputation.get("denominator")
        )
        if dict(raw_recomputation) != rebuilt:
            raise V6ContractError("recomputation is not content-addressed exactly")
        recomputation_id = rebuilt["recomputation_id"]
        if recomputation_id in recomputations:
            raise V6ContractError(f"duplicate recomputation: {recomputation_id}")
        total = Fraction(0)
        for term in rebuilt["terms"]:
            node = nodes.get(term["input_node_id"])
            if node is None or node["node_type"] != "input":
                raise V6ContractError(
                    f"recomputation input node is absent: {term['input_node_id']}"
                )
            total += Fraction(Decimal(node["numeric_value"])) * Fraction(
                Decimal(term["coefficient"])
            )
        actual = total / Fraction(Decimal(rebuilt["denominator"]))
        recomputed_values[recomputation_id] = actual
        recomputations[recomputation_id] = rebuilt

    rounding_proofs: dict[str, dict[str, Any]] = {}
    for raw_rounding in rounding_rows:
        if not isinstance(raw_rounding, Mapping) or set(raw_rounding) != {
            "schema",
            "rounding_proof_id",
            "recomputation_id",
            "unrounded_numerator",
            "unrounded_denominator",
            "decimal_places",
            "rounding_mode",
            "rounded_decimal",
            "display_value",
        }:
            raise V6ContractError("rounding-proof schema keys differ")
        rebuilt = make_provenance_rounding_proof(
            raw_rounding.get("recomputation_id"),
            raw_rounding.get("unrounded_numerator"),
            raw_rounding.get("unrounded_denominator"),
            raw_rounding.get("decimal_places"),
        )
        if dict(raw_rounding) != rebuilt:
            raise V6ContractError("rounding proof is not content-addressed exactly")
        proof_id = rebuilt["rounding_proof_id"]
        if proof_id in rounding_proofs:
            raise V6ContractError(f"duplicate rounding proof: {proof_id}")
        actual = recomputed_values.get(rebuilt["recomputation_id"])
        if (
            actual is None
            or actual.numerator != rebuilt["unrounded_numerator"]
            or actual.denominator != rebuilt["unrounded_denominator"]
        ):
            raise V6ContractError(f"rounding proof does not bind recomputation: {proof_id}")
        rounding_proofs[proof_id] = rebuilt

    edges: dict[str, dict[str, Any]] = {}
    for raw_edge in edge_rows:
        if not isinstance(raw_edge, Mapping) or set(raw_edge) != {
            "schema",
            "edge_id",
            "source_id",
            "target_result_id",
            "relation",
        }:
            raise V6ContractError("provenance edge schema keys differ")
        rebuilt = make_provenance_edge(
            raw_edge.get("source_id"),
            raw_edge.get("target_result_id"),
            raw_edge.get("relation"),
        )
        if dict(raw_edge) != rebuilt:
            raise V6ContractError("provenance edge is not content-addressed exactly")
        edge_id = rebuilt["edge_id"]
        if edge_id in edges:
            raise V6ContractError(f"duplicate provenance edge: {edge_id}")
        edges[edge_id] = rebuilt

    expected_node_ids: set[str] = set()
    expected_recomputation_ids: set[str] = set()
    expected_rounding_ids: set[str] = set()
    expected_edges: dict[str, dict[str, Any]] = {}
    for result_id, record in records_by_id.items():
        input_ids = set(record["input_node_ids"])
        used_input_ids: set[str] = set()
        if any(
            parent_id not in inventory_by_id
            for parent_id in record["closed_parent_result_ids"]
        ):
            raise V6ContractError(f"{result_id}: closed parent result is absent")
        for field, node_type in PROVENANCE_NODE_TYPE_BY_FIELD.items():
            for node_id in record[field]:
                node = nodes.get(node_id)
                if node is None or node["node_type"] != node_type:
                    raise V6ContractError(
                        f"{result_id}: typed contributor is absent or wrong: {node_id}"
                    )
                expected_node_ids.add(node_id)
        for raw_id in record["raw_artifact_ids"]:
            node = nodes.get(raw_id)
            if node is None or node["node_type"] != "raw_artifact":
                raise V6ContractError(f"{result_id}: typed raw node is absent: {raw_id}")
            expected_node_ids.add(raw_id)
        recomputation_ids = set(record["recomputation_node_ids"])
        rounding_ids = set(record["rounding_proof_node_ids"])
        if not recomputation_ids or not rounding_ids:
            raise V6ContractError(f"{result_id}: numeric closure is empty")
        for recomputation_id in recomputation_ids:
            recomputation = recomputations.get(recomputation_id)
            if recomputation is None:
                raise V6ContractError(
                    f"{result_id}: typed recomputation is absent: {recomputation_id}"
                )
            term_ids = {term["input_node_id"] for term in recomputation["terms"]}
            if not term_ids <= input_ids:
                raise V6ContractError(
                    f"{result_id}: recomputation uses an undeclared input"
                )
            used_input_ids.update(term_ids)
            expected_recomputation_ids.add(recomputation_id)
        if used_input_ids != input_ids:
            raise V6ContractError(f"{result_id}: recomputation input union differs")
        inventory_row = inventory_by_id[result_id]
        paper_numeric = _json_decimal(
            inventory_row.get("numeric_value"), f"{result_id} paper numeric_value"
        )
        paper_display = inventory_row.get("display_value")
        if not isinstance(paper_display, str):
            raise V6ContractError(f"{result_id}: paper display_value must be a string")
        for proof_id in rounding_ids:
            proof = rounding_proofs.get(proof_id)
            if proof is None or proof["recomputation_id"] not in recomputation_ids:
                raise V6ContractError(
                    f"{result_id}: typed rounding proof is absent or foreign: {proof_id}"
                )
            if (
                Decimal(proof["rounded_decimal"]) != Decimal(paper_numeric)
                or proof["display_value"] != paper_display
            ):
                raise V6ContractError(
                    f"{result_id}: recomputation does not match exact paper rounding"
                )
            expected_rounding_ids.add(proof_id)
        for field, relation in PROVENANCE_RELATION_BY_FIELD.items():
            for source_id in record[field]:
                edge = make_provenance_edge(source_id, result_id, relation)
                expected_edges[edge["edge_id"]] = edge

    if set(nodes) != expected_node_ids:
        raise V6ContractError("provenance node set is not the exact admission closure")
    if set(recomputations) != expected_recomputation_ids:
        raise V6ContractError("recomputation set is not the exact admission closure")
    if set(rounding_proofs) != expected_rounding_ids:
        raise V6ContractError("rounding-proof set is not the exact admission closure")
    if edges != expected_edges:
        raise V6ContractError("provenance edge set is not the exact admission closure")
    return {
        "schema": "experiments7-cp3-provenance-reconciliation/v6",
        "node_count": len(nodes),
        "edge_count": len(edges),
        "recomputation_count": len(recomputations),
        "rounding_proof_count": len(rounding_proofs),
        "nodes_sha256": digest_records([nodes[key] for key in sorted(nodes)]),
        "edges_sha256": digest_records([edges[key] for key in sorted(edges)]),
        "recomputations_sha256": digest_records(
            [recomputations[key] for key in sorted(recomputations)]
        ),
        "rounding_proofs_sha256": digest_records(
            [rounding_proofs[key] for key in sorted(rounding_proofs)]
        ),
    }


def make_bucket_subseal(
    inventory_sha256: str,
    paper_section_id: str,
    admission_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Create deterministic bucket material; no publication happens here."""

    _sha256(inventory_sha256, "inventory_sha256")
    if not paper_section_id:
        raise V6ContractError("paper_section_id must not be empty")
    rows = sorted(
        (validate_admission_record(row, "precopy") for row in admission_records),
        key=lambda row: row["result_id"],
    )
    if not rows:
        raise V6ContractError(f"empty admission bucket: {paper_section_id}")
    if any(row["paper_section_id"] != paper_section_id for row in rows):
        raise V6ContractError(f"cross-bucket record in {paper_section_id}")
    result_ids = [row["result_id"] for row in rows]
    if len(result_ids) != len(set(result_ids)):
        raise V6ContractError(f"duplicate result in bucket {paper_section_id}")
    raw_union = sorted(
        {
            raw_id
            for row in rows
            if row["copy_required"]
            for raw_id in row["raw_artifact_ids"]
        }
    )
    return {
        "schema": "experiments7-admission-bucket-subseal/v6",
        "phase": "precopy",
        "paper_section_id": paper_section_id,
        "inventory_sha256": inventory_sha256,
        "result_ids": result_ids,
        "result_records": rows,
        "result_records_sha256": digest_records(rows),
        "copy_required_raw_artifact_ids": raw_union,
    }


def reconcile_cp3_buckets(
    inventory: Sequence[Mapping[str, Any]],
    bucket_subseals: Sequence[Mapping[str, Any]],
    *,
    expected_result_count: int = 1908,
) -> dict[str, Any]:
    """Prove a disjoint exact 1,908-result union before CP3."""

    if type(expected_result_count) is not int or expected_result_count <= 0:
        raise V6ContractError("expected_result_count must be a positive integer")
    if len(inventory) != expected_result_count:
        raise V6ContractError("CP3 inventory cardinality mismatch")
    inv_hash = result_inventory_digest(inventory)
    inventory_by_id = {row["result_id"]: row["paper_section_id"] for row in inventory}
    expected_sections = set(inventory_by_id.values())
    seen_sections: set[str] = set()
    seen_results: set[str] = set()
    all_raw: set[str] = set()
    bucket_hashes: dict[str, str] = {}
    for bucket in bucket_subseals:
        section = _required_string(bucket, "paper_section_id")
        if section in seen_sections:
            raise V6ContractError(f"duplicate bucket: {section}")
        seen_sections.add(section)
        if bucket.get("schema") != "experiments7-admission-bucket-subseal/v6":
            raise V6ContractError(f"{section}: bucket schema mismatch")
        if bucket.get("phase") != "precopy" or bucket.get("inventory_sha256") != inv_hash:
            raise V6ContractError(f"{section}: stale or foreign bucket")
        rows = bucket.get("result_records")
        if not isinstance(rows, list):
            raise V6ContractError(f"{section}: result_records must be a list")
        validated = [validate_admission_record(row, "precopy") for row in rows]
        ids = [row["result_id"] for row in validated]
        if len(ids) != len(set(ids)):
            raise V6ContractError(f"{section}: duplicate result in bucket")
        if ids != bucket.get("result_ids") or ids != sorted(ids):
            raise V6ContractError(f"{section}: result-ID seal mismatch")
        if digest_records(validated) != bucket.get("result_records_sha256"):
            raise V6ContractError(f"{section}: mutated admission records")
        if any(row["paper_section_id"] != section for row in validated):
            raise V6ContractError(f"{section}: cross-bucket result")
        if any(inventory_by_id.get(row["result_id"]) != section for row in validated):
            raise V6ContractError(f"{section}: inventory result section mismatch")
        if any(row["state"] != "ADMITTED_FOR_COPY" for row in validated):
            raise V6ContractError(f"{section}: unresolved result blocks CP3")
        overlap = seen_results & set(ids)
        if overlap:
            raise V6ContractError(f"duplicate result across buckets: {sorted(overlap)[0]}")
        seen_results.update(ids)
        raw_union = sorted(
            {
                raw_id
                for row in validated
                if row["copy_required"]
                for raw_id in row["raw_artifact_ids"]
            }
        )
        if raw_union != bucket.get("copy_required_raw_artifact_ids"):
            raise V6ContractError(f"{section}: copy-required raw union mismatch")
        all_raw.update(raw_union)
        bucket_hashes[section] = digest_json(bucket)
    if seen_sections != expected_sections:
        raise V6ContractError("CP3 bucket-section union mismatch")
    if seen_results != set(inventory_by_id):
        raise V6ContractError("CP3 result union is not the exact inventory")
    summary = {
        "schema": "experiments7-cp3-admission-reconciliation/v6",
        "phase": "precopy",
        "checkpoint_state": "READY_FOR_CP3",
        "inventory_sha256": inv_hash,
        "total_result_count": len(seen_results),
        "precopy_admitted_count": len(seen_results),
        "precopy_unresolved_count": 0,
        "paper_section_bucket_count": len(seen_sections),
        "bucket_subseal_sha256s": dict(sorted(bucket_hashes.items())),
        "copy_required_raw_artifact_ids": sorted(all_raw),
    }
    summary["copy_plan_input_sha256"] = digest_json(
        summary["copy_required_raw_artifact_ids"]
    )
    return summary


def _copy_ids(raw_id: str, destination: str) -> tuple[str, str]:
    copied_digest = hashlib.sha256(
        (raw_id + "\0" + destination).encode("utf-8")
    ).hexdigest()
    copied_id = f"copied:{copied_digest}"
    edge_digest = hashlib.sha256(
        (raw_id + "\0" + copied_id + "\0copied_to").encode("utf-8")
    ).hexdigest()
    return copied_id, f"edge:{edge_digest}"


def _validate_cp3_summary(
    cp3_summary: Mapping[str, Any], expected_result_count: int
) -> list[str]:
    """Validate the complete CP3 envelope before any copy-side operation."""

    if (
        not isinstance(expected_result_count, int)
        or isinstance(expected_result_count, bool)
        or expected_result_count <= 0
    ):
        raise V6ContractError("expected_result_count must be a positive integer")
    if cp3_summary.get("schema") != "experiments7-cp3-admission-reconciliation/v6":
        raise V6ContractError("CP3 summary schema mismatch")
    if cp3_summary.get("phase") != "precopy":
        raise V6ContractError("CP3 summary phase mismatch")
    if cp3_summary.get("checkpoint_state") != "READY_FOR_CP3":
        raise V6ContractError("CP3 checkpoint is not ready")
    if (
        type(cp3_summary.get("total_result_count")) is not int
        or cp3_summary.get("total_result_count") != expected_result_count
        or type(cp3_summary.get("precopy_admitted_count")) is not int
        or cp3_summary.get("precopy_admitted_count") != expected_result_count
        or type(cp3_summary.get("precopy_unresolved_count")) is not int
        or cp3_summary.get("precopy_unresolved_count") != 0
    ):
        raise V6ContractError("copy planning requires accepted zero-unresolved CP3")
    _sha256(cp3_summary.get("inventory_sha256"), "CP3 inventory_sha256")

    bucket_count = cp3_summary.get("paper_section_bucket_count")
    bucket_hashes = cp3_summary.get("bucket_subseal_sha256s")
    if (
        not isinstance(bucket_count, int)
        or isinstance(bucket_count, bool)
        or bucket_count <= 0
        or not isinstance(bucket_hashes, Mapping)
        or len(bucket_hashes) != bucket_count
    ):
        raise V6ContractError("CP3 bucket sub-seal cardinality mismatch")
    for section, bucket_hash in bucket_hashes.items():
        if not isinstance(section, str) or not section:
            raise V6ContractError("CP3 bucket section is invalid")
        _sha256(bucket_hash, f"CP3 bucket sub-seal {section}")

    expected_raw = _strings(cp3_summary, "copy_required_raw_artifact_ids")
    if expected_raw != sorted(expected_raw):
        raise V6ContractError("CP3 raw artifact union is not canonical")
    _sha256(cp3_summary.get("copy_plan_input_sha256"), "CP3 copy-plan input")
    if digest_json(expected_raw) != cp3_summary.get("copy_plan_input_sha256"):
        raise V6ContractError("CP3 copy-plan input was mutated")
    return expected_raw


def _validate_copy_plan_entry(entry: Mapping[str, Any]) -> str:
    """Recompute every security-sensitive copy identity from plan primitives."""

    raw_id = _required_string(entry, "raw_artifact_id")
    if entry.get("schema") != "experiments7-copy-plan-entry/v6":
        raise V6ContractError(f"copy-plan entry schema mismatch: {raw_id}")
    _required_string(entry, "source_manifest_record_id")
    source_root = _required_string(entry, "source_root_id")
    if source_root not in PROTECTED_SOURCE_ROOTS:
        raise V6ContractError(f"copy-plan source root is not protected: {raw_id}")
    source_path = _relative_path(
        entry.get("source_relative_path"), f"copy-plan source path {raw_id}"
    )
    _sha256(
        entry.get("source_pre_record_sha256"),
        f"copy-plan source-pre record {raw_id}",
    )
    _sha256(entry.get("source_sha256"), f"copy-plan source hash {raw_id}")
    size = entry.get("size")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise V6ContractError(f"copy-plan size is invalid: {raw_id}")
    identity = _identity(
        entry.get("expected_source_descriptor_identity"), f"plan identity {raw_id}"
    )
    if identity["size"] != size:
        raise V6ContractError(f"copy-plan source identity size mismatch: {raw_id}")
    destination = _relative_path(
        entry.get("destination_relative_path"), f"copy-plan destination {raw_id}"
    )
    expected_destination = f"raw/verified/{source_root}/{source_path}"
    if destination != expected_destination:
        raise V6ContractError(f"copy-plan destination identity mismatch: {raw_id}")
    expected_copied_id, expected_edge_id = _copy_ids(raw_id, destination)
    if (
        entry.get("copied_artifact_id") != expected_copied_id
        or entry.get("copied_to_edge_id") != expected_edge_id
    ):
        raise V6ContractError(f"copy-plan copied identity mismatch: {raw_id}")
    return raw_id


def _index_source_pre_records(
    source_pre_records: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    """Validate and index the exact source-pre records supplied by G0."""

    return {
        str(source["record_id"]): source
        for source in validate_source_pre_records(source_pre_records)
    }


def plan_source_pre_copies(
    cp3_summary: Mapping[str, Any],
    raw_nodes: Sequence[Mapping[str, Any]],
    source_pre_records: Sequence[Mapping[str, Any]],
    *,
    expected_result_count: int = 1908,
) -> dict[str, Any]:
    """Create a metadata-only copy plan from this run's exact source-pre."""

    expected_raw = _validate_cp3_summary(cp3_summary, expected_result_count)

    source_by_id = _index_source_pre_records(source_pre_records)

    raw_by_id: dict[str, Mapping[str, Any]] = {}
    for node in raw_nodes:
        if not isinstance(node, Mapping):
            raise V6ContractError("raw provenance node must be an object")
        validated_node = _validate_raw_provenance_node(node, source_by_id)
        node_id = str(validated_node["node_id"])
        if node_id in raw_by_id:
            raise V6ContractError(f"duplicate raw node: {node_id}")
        raw_by_id[node_id] = validated_node
    if set(raw_by_id) != set(expected_raw):
        raise V6ContractError("raw-node set is not the exact CP3 copy union")

    used_source_records: set[str] = set()
    entries: list[dict[str, Any]] = []
    for raw_id in sorted(raw_by_id):
        node = raw_by_id[raw_id]
        if raw_id == DRIFT_ONLY_RAW_ID:
            raise V6ContractError(f"excluded raw node in copy plan: {raw_id}")
        source_record_id = _required_string(node, "source_manifest_record_id")
        source = source_by_id.get(source_record_id)
        if source is None:
            raise V6ContractError(f"raw absent from this source-pre: {raw_id}")
        if source_record_id in used_source_records:
            raise V6ContractError(f"raw nodes alias source-pre record: {source_record_id}")
        used_source_records.add(source_record_id)
        relative_path = _manifest_path(source)
        if node.get("root_id") != source["root_id"] or _manifest_path(node) != relative_path:
            raise V6ContractError(f"raw/source path identity mismatch: {raw_id}")
        if node.get("sha256") != source["sha256"] or node.get("size") != source["size"]:
            raise V6ContractError(f"raw/source content identity mismatch: {raw_id}")
        destination = _relative_path(
            f"raw/verified/{source['root_id']}/{relative_path}", "destination"
        )
        copied_id, edge_id = _copy_ids(raw_id, destination)
        entries.append(
            {
                "schema": "experiments7-copy-plan-entry/v6",
                "raw_artifact_id": raw_id,
                "source_manifest_record_id": source_record_id,
                "source_root_id": source["root_id"],
                "source_relative_path": relative_path,
                "source_pre_record_sha256": digest_json(source),
                "source_sha256": source["sha256"],
                "size": source["size"],
                "expected_source_descriptor_identity": dict(source["descriptor_identity"]),
                "destination_relative_path": destination,
                "copied_artifact_id": copied_id,
                "copied_to_edge_id": edge_id,
            }
        )
    destinations = [entry["destination_relative_path"] for entry in entries]
    copied_ids = [entry["copied_artifact_id"] for entry in entries]
    copied_edges = [entry["copied_to_edge_id"] for entry in entries]
    if (
        len(destinations) != len(set(destinations))
        or len(copied_ids) != len(set(copied_ids))
        or len(copied_edges) != len(set(copied_edges))
    ):
        raise V6ContractError("copy plan contains aliased destinations or copy identities")
    plan = {
        "schema": "experiments7-copy-plan/v6",
        "cp3_summary_sha256": digest_json(cp3_summary),
        "copy_plan_input_sha256": cp3_summary["copy_plan_input_sha256"],
        "raw_artifact_ids": [entry["raw_artifact_id"] for entry in entries],
        "entries": entries,
    }
    plan["entries_sha256"] = digest_records(entries)
    return plan


def validate_copy_ledger(
    cp3_summary: Mapping[str, Any],
    copy_plan: Mapping[str, Any],
    ledger_rows: Sequence[Mapping[str, Any]],
    copied_payloads: Mapping[str, bytes],
    source_pre_records: Sequence[Mapping[str, Any]],
    *,
    expected_result_count: int = 1908,
) -> dict[str, Any]:
    """Prove exact source identity, independent copies, and current bytes."""

    expected_raw = _validate_cp3_summary(cp3_summary, expected_result_count)
    source_by_id = _index_source_pre_records(source_pre_records)
    if copy_plan.get("schema") != "experiments7-copy-plan/v6":
        raise V6ContractError("copy plan schema mismatch")
    _sha256(copy_plan.get("cp3_summary_sha256"), "copy plan CP3 summary")
    if copy_plan.get("cp3_summary_sha256") != digest_json(cp3_summary):
        raise V6ContractError("copy plan does not bind the current CP3 summary")
    _sha256(copy_plan.get("copy_plan_input_sha256"), "copy plan raw-union input")
    if copy_plan.get("copy_plan_input_sha256") != cp3_summary["copy_plan_input_sha256"]:
        raise V6ContractError("copy plan does not bind the CP3 raw-union digest")
    entries = copy_plan.get("entries")
    if not isinstance(entries, list):
        raise V6ContractError("copy plan entries must be a list")
    _sha256(copy_plan.get("entries_sha256"), "copy plan entries")
    if digest_records(entries) != copy_plan.get("entries_sha256"):
        raise V6ContractError("copy plan entries were mutated")
    plan_by_raw: dict[str, Mapping[str, Any]] = {}
    used_source_records: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise V6ContractError("copy-plan entry must be an object")
        raw_id = _validate_copy_plan_entry(entry)
        if raw_id in plan_by_raw:
            raise V6ContractError(f"duplicate copy-plan raw: {raw_id}")
        source_record_id = _required_string(entry, "source_manifest_record_id")
        source = source_by_id.get(source_record_id)
        if source is None:
            raise V6ContractError(f"copy plan source is absent from source-pre: {raw_id}")
        if source_record_id in used_source_records:
            raise V6ContractError(f"copy-plan raws alias source-pre record: {source_record_id}")
        used_source_records.add(source_record_id)
        source_path = _manifest_path(source)
        if (
            entry.get("source_manifest_record_id") != source["record_id"]
            or entry.get("source_pre_record_sha256") != digest_json(source)
            or entry.get("source_root_id") != source["root_id"]
            or entry.get("source_relative_path") != source_path
            or entry.get("source_sha256") != source["sha256"]
            or entry.get("size") != source["size"]
            or entry.get("expected_source_descriptor_identity")
            != source["descriptor_identity"]
        ):
            raise V6ContractError(f"copy plan/source-pre exact-record mismatch: {raw_id}")
        plan_by_raw[raw_id] = entry
    plan_raw_ids = _strings(copy_plan, "raw_artifact_ids")
    ordered_entry_raw_ids = [entry["raw_artifact_id"] for entry in entries]
    if plan_raw_ids != ordered_entry_raw_ids:
        raise V6ContractError("copy plan raw list does not bind its ordered entries")
    if set(plan_by_raw) != set(expected_raw):
        raise V6ContractError("copy plan no longer equals the CP3 raw union")

    ledger_by_raw: dict[str, Mapping[str, Any]] = {}
    destinations: set[str] = set()
    copied_ids: set[str] = set()
    for row in ledger_rows:
        if not isinstance(row, Mapping):
            raise V6ContractError("copy ledger row must be an object")
        if set(row) != COPY_LEDGER_ROW_KEYS:
            raise V6ContractError("copy ledger row keys differ")
        raw_id = _required_string(row, "raw_artifact_id")
        destination = _required_string(row, "destination_relative_path")
        copied_id = _required_string(row, "copied_artifact_id")
        if raw_id in ledger_by_raw:
            raise V6ContractError(f"duplicate copy ledger row: {raw_id}")
        if destination in destinations or copied_id in copied_ids:
            raise V6ContractError("duplicate copy destination or copied artifact")
        ledger_by_raw[raw_id] = row
        destinations.add(destination)
        copied_ids.add(copied_id)
    if set(ledger_by_raw) != set(plan_by_raw):
        raise V6ContractError("copy ledger has missing, extra, or orphan raw nodes")

    raw_to_copy: dict[str, str] = {}
    raw_to_edge: dict[str, str] = {}
    for raw_id, entry in plan_by_raw.items():
        row = ledger_by_raw[raw_id]
        if row.get("schema") != "experiments7-copy-ledger-row/v6":
            raise V6ContractError(f"copy ledger schema mismatch: {raw_id}")
        if row.get("copy_plan_entry_sha256") != digest_json(entry):
            raise V6ContractError(f"copy row is not bound to its plan: {raw_id}")
        bound_fields = (
            "source_manifest_record_id",
            "source_root_id",
            "source_relative_path",
            "source_pre_record_sha256",
            "source_sha256",
            "size",
            "destination_relative_path",
            "copied_artifact_id",
            "copied_to_edge_id",
        )
        if any(row.get(key) != entry.get(key) for key in bound_fields):
            raise V6ContractError(f"copy row/plan mismatch: {raw_id}")
        expected_identity = _identity(
            entry.get("expected_source_descriptor_identity"), f"plan identity {raw_id}"
        )
        before = _identity(row.get("source_before"), f"source_before {raw_id}")
        after = _identity(row.get("source_after"), f"source_after {raw_id}")
        if before != expected_identity or after != expected_identity:
            raise V6ContractError(f"source-pre metadata drift: {raw_id}")
        if (
            row.get("source_sha256_before") != entry["source_sha256"]
            or row.get("source_sha256_after") != entry["source_sha256"]
        ):
            raise V6ContractError(f"source byte drift: {raw_id}")
        destination_identity = _identity(
            row.get("destination_identity"), f"destination_identity {raw_id}"
        )
        if destination_identity["size"] != entry["size"]:
            raise V6ContractError(f"destination size identity mismatch: {raw_id}")
        if (destination_identity["st_dev"], destination_identity["st_ino"]) == (
            expected_identity["st_dev"],
            expected_identity["st_ino"],
        ):
            raise V6ContractError(f"hardlink identity detected: {raw_id}")
        if row.get("publication_method") != "byte_stream_no_replace":
            raise V6ContractError(f"non-stream/no-replace copy method: {raw_id}")
        copy_flags = ("hardlink_used", "reflink_used", "clone_used", "cache_used")
        if any(row.get(flag) is not False for flag in copy_flags):
            raise V6ContractError(f"link/clone/cache copy detected: {raw_id}")
        if (
            row.get("destination_sha256") != entry["source_sha256"]
            or row.get("destination_size") != entry["size"]
        ):
            raise V6ContractError(f"destination declaration mismatch: {raw_id}")
        copied_id = entry["copied_artifact_id"]
        payload = copied_payloads.get(copied_id)
        if not isinstance(payload, bytes):
            raise V6ContractError(f"copied bytes unavailable: {raw_id}")
        if (
            len(payload) != entry["size"]
            or hashlib.sha256(payload).hexdigest() != entry["source_sha256"]
        ):
            raise V6ContractError(f"copied bytes mutated or mismatched: {raw_id}")
        raw_to_copy[raw_id] = copied_id
        raw_to_edge[raw_id] = entry["copied_to_edge_id"]
    if set(copied_payloads) != copied_ids:
        raise V6ContractError("copied payload set has extra or missing artifacts")
    return {
        "schema": "experiments7-copy-reconciliation/v6",
        "copy_state": "COPIED_BYTES_VERIFIED",
        "copy_plan_sha256": digest_json(copy_plan),
        "copy_ledger_sha256": digest_records(list(ledger_rows)),
        "raw_artifact_count": len(plan_by_raw),
        "copied_artifact_count": len(copied_ids),
        "raw_to_copied_artifact_id": dict(sorted(raw_to_copy.items())),
        "raw_to_copied_to_edge_id": dict(sorted(raw_to_edge.items())),
    }


def _binding_view(record: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "result_id",
        "paper_section_id",
        "result_kind",
        "copy_required",
        "raw_artifact_ids",
        "candidate_raw_artifact_ids",
        "required_parent_result_ids",
        "closed_parent_result_ids",
        "required_contributor_node_ids",
        "sealed_contributor_node_ids",
        "positive_evidence_origin",
        "source_pre_bound",
        "declared_drift",
        "ambiguous",
        "support_only",
        "audit_positive_evidence",
        *CLOSURE_FIELDS,
    )
    return {field: record.get(field) for field in fields}


def reconcile_final_admission(
    inventory: Sequence[Mapping[str, Any]],
    bucket_subseals: Sequence[Mapping[str, Any]],
    cp3_summary: Mapping[str, Any],
    final_records: Sequence[Mapping[str, Any]],
    copy_plan: Mapping[str, Any],
    ledger_rows: Sequence[Mapping[str, Any]],
    copied_payloads: Mapping[str, bytes],
    source_pre_records: Sequence[Mapping[str, Any]],
    *,
    expected_result_count: int = 1908,
) -> dict[str, Any]:
    """Prove VERIFIED from unchanged precopy bindings and current copied bytes."""

    recomputed_cp3 = reconcile_cp3_buckets(
        inventory, bucket_subseals, expected_result_count=expected_result_count
    )
    if canonical_json(recomputed_cp3) != canonical_json(cp3_summary):
        raise V6ContractError("CP3 summary or bucket sub-seals were mutated")
    copy_summary = validate_copy_ledger(
        cp3_summary,
        copy_plan,
        ledger_rows,
        copied_payloads,
        source_pre_records,
        expected_result_count=expected_result_count,
    )
    precopy_by_id: dict[str, Mapping[str, Any]] = {}
    for bucket in bucket_subseals:
        for row in bucket["result_records"]:
            result_id = row["result_id"]
            if result_id in precopy_by_id:
                raise V6ContractError(f"duplicate precopy result: {result_id}")
            precopy_by_id[result_id] = row
    final_by_id: dict[str, Mapping[str, Any]] = {}
    for row in final_records:
        validated = validate_admission_record(row, "final")
        result_id = validated["result_id"]
        if result_id in final_by_id:
            raise V6ContractError(f"duplicate final result: {result_id}")
        final_by_id[result_id] = validated
    inventory_ids = {row["result_id"] for row in inventory}
    if set(precopy_by_id) != inventory_ids or set(final_by_id) != inventory_ids:
        raise V6ContractError("final admission is not the exact inventory")

    referenced_raw: set[str] = set()
    for result_id in sorted(inventory_ids):
        precopy = precopy_by_id[result_id]
        final = final_by_id[result_id]
        if final.get("precopy_record_sha256") != digest_json(precopy):
            raise V6ContractError(f"final result lacks exact precopy binding: {result_id}")
        if _binding_view(final) != _binding_view(precopy):
            raise V6ContractError(f"final/precopy provenance changed: {result_id}")
        raw_ids = _strings(final, "raw_artifact_ids")
        try:
            expected_copied = sorted(
                copy_summary["raw_to_copied_artifact_id"][raw_id] for raw_id in raw_ids
            )
            expected_edges = sorted(
                copy_summary["raw_to_copied_to_edge_id"][raw_id] for raw_id in raw_ids
            )
        except KeyError as exc:
            raise V6ContractError(f"final result references uncopied raw: {result_id}") from exc
        if sorted(_strings(final, "copied_artifact_ids")) != expected_copied:
            raise V6ContractError(f"final copied-artifact closure mismatch: {result_id}")
        if sorted(_strings(final, "copied_to_edge_ids")) != expected_edges:
            raise V6ContractError(f"final copied_to closure mismatch: {result_id}")
        referenced_raw.update(raw_ids)
    if referenced_raw != set(copy_summary["raw_to_copied_artifact_id"]):
        raise V6ContractError("orphan copied raw artifact in final admission")
    ordered_final = [final_by_id[result_id] for result_id in sorted(final_by_id)]
    return {
        "schema": "experiments7-final-admission-reconciliation/v6",
        "phase": "final",
        "final_state": "VERIFIED",
        "total_result_count": len(final_by_id),
        "final_verified_count": len(final_by_id),
        "final_unresolved_count": 0,
        "cp3_summary_sha256": digest_json(cp3_summary),
        "copy_reconciliation_sha256": digest_json(copy_summary),
        "final_results_sha256": digest_records(ordered_final),
    }


__all__ = [
    "ADMISSION_BOOLEAN_FIELDS",
    "AUDIT_DEFECTS_RELATIVE",
    "AUDIT_DEFECTS_SHA256",
    "AUDIT_UNRESOLVED_ID_SET_SHA256",
    "AUDIT_UNRESOLVED_RELATIVE",
    "AUDIT_UNRESOLVED_SHA256",
    "CLOSURE_FIELDS",
    "DRIFT_ONLY_RAW_ID",
    "DRIFT_ONLY_RAW_SHA256",
    "FIGURE5_ZERO_RAW_RESULT_IDS",
    "LEGAL_STATES",
    "PROVENANCE_EDGE_SCHEMA",
    "PROVENANCE_NODE_SCHEMA",
    "PROVENANCE_RECOMPUTATION_SCHEMA",
    "PROVENANCE_ROUNDING_SCHEMA",
    "TABLE10_INCOMPLETE_PARENT_RESULT_IDS",
    "V6ContractError",
    "canonical_json",
    "derive_admission_state",
    "digest_json",
    "digest_records",
    "inventory_digest",
    "result_inventory_digest",
    "make_bucket_subseal",
    "make_provenance_edge",
    "make_provenance_node",
    "make_raw_provenance_node",
    "make_provenance_recomputation",
    "make_provenance_rounding_proof",
    "plan_source_pre_copies",
    "reconcile_cp3_buckets",
    "reconcile_final_admission",
    "validate_admission_record",
    "validate_cp3_provenance",
    "validate_copy_ledger",
    "validate_source_pre_records",
]
