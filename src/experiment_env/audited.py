from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator

from .snapshot import (
    PublicationError,
    ValidationError,
    _copy_record,
    _freeze_directories,
    _rename_noreplace,
    _write_bytes,
    validate_snapshot,
)
from .util import (
    canonical_json_bytes,
    clean_relative_path,
    credential_presence,
    is_relative_to,
    load_json,
    redact_argv,
    resolve_config_path,
    sha256_bytes,
    sha256_file,
    write_json_exclusive,
)


AUDIT_SCHEMA = "experiments7-audit-overlay/v1"
OVERLAY_PUBLICATION_SCHEMA = "experiments7-audited-overlay-publication/v2"
OVERLAY_SEAL_SCHEMA = "experiments7-audited-overlay-seal/v2"
OVERLAY_MANIFEST_SCHEMA = "experiments7-audited-overlay-file/v2"
REGISTRY_SCHEMA = "experiments7-audited-variant-registry/v2"
PROFILES_SCHEMA = "experiments7-audited-variant-profiles/v2"
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
FORBIDDEN_LABEL_RE = re.compile(r"(?:^|_)(?:eval6|infer6)(?:_|$)")
EXPECTED_SHA256_KEYS = frozenset({"audit", "cp0", "profiles", "registry", "source_pre"})
CONTROL_NAMES = frozenset({"audit.json", "manifest.jsonl", "profiles.json", "registry.json", "seal.json"})
PROFILE_ID = "exp45_paper_audited_v2"
EXPECTED_REQUIRED_LABELS = {
    "eval4_single_legacy",
    "eval4_multiturn_legacy",
    "eval5_canonical",
    "infer5_memory_text_api",
    "infer5_memory_text_vllm",
    "infer5_v3_api",
    "infer5_v3_vllm",
    "memory5_main_true_blind",
    "memory5_v1_legacy",
    "table15_iteration_cap_eval4",
    "table16_fixed400_eval4",
    "table16_fixed400_eval5",
}
EXPECTED_AUDIT_SHA256 = "782df2d84077f93b69aed4c79b334643f7ef5cb33470fe5f38fc412434f90b6b"


class AuditedEnvironmentError(RuntimeError):
    pass


def _json_bytes(value: Any) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _write_or_verify(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        if path.is_symlink() or path.read_bytes() != data:
            raise AuditedEnvironmentError(f"refusing to replace non-identical generated file: {path}")


def _walk_audit_records(
    value: Any,
    *,
    label: str | None = None,
    relation: str | None = None,
) -> Iterator[tuple[str, str, str | None, str | None]]:
    if isinstance(value, dict):
        current_label = value.get("label", label)
        path = value.get("path")
        digest = value.get("sha256")
        if isinstance(path, str) and path.startswith("/data/minseo/experiments") and isinstance(digest, str):
            yield path, digest, current_label, relation
        for key, child in value.items():
            yield from _walk_audit_records(child, label=current_label, relation=key)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_audit_records(child, label=label, relation=relation)


def selected_audit_files(audit: dict[str, Any]) -> list[dict[str, Any]]:
    if audit.get("schema") != AUDIT_SCHEMA:
        raise AuditedEnvironmentError("unsupported audit overlay schema")
    by_path: dict[str, dict[str, Any]] = {}
    for absolute, digest, label, relation in _walk_audit_records(audit):
        path = Path(absolute)
        if len(path.parts) < 5 or path.parts[:3] != ("/", "data", "minseo"):
            raise AuditedEnvironmentError(f"unexpected selected path: {absolute}")
        root_id = path.parts[3]
        if root_id not in {"experiments4", "experiments5"}:
            raise AuditedEnvironmentError(f"overlay source is outside experiments4/5: {absolute}")
        relative = PurePosixPath(*path.parts[4:]).as_posix()
        clean_relative_path(relative)
        item = by_path.setdefault(
            absolute,
            {
                "absolute_path": absolute,
                "audit_labels": set(),
                "audit_relations": set(),
                "relative_path": relative,
                "root_id": root_id,
                "sha256": digest,
            },
        )
        if item["sha256"] != digest:
            raise AuditedEnvironmentError(f"conflicting audit hashes for {absolute}")
        if label:
            item["audit_labels"].add(label)
        if relation:
            item["audit_relations"].add(relation)
    result: list[dict[str, Any]] = []
    for item in by_path.values():
        item["audit_labels"] = sorted(item["audit_labels"])
        item["audit_relations"] = sorted(item["audit_relations"])
        result.append(item)
    result.sort(key=lambda item: (item["root_id"], item["relative_path"]))
    expected = audit.get("counts", {}).get("selected_unique_overlay_file_count")
    if expected is not None and len(result) != expected:
        raise AuditedEnvironmentError(f"audit selected-file count mismatch: {len(result)} != {expected}")
    return result


def _source_pre_index(source_pre_path: Path, selected: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    needed = {(item["root_id"], item["relative_path"]): item for item in selected}
    found: dict[str, dict[str, Any]] = {}
    with source_pre_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AuditedEnvironmentError(f"invalid source-pre line {line_number}") from exc
            root_id = record.get("root_id")
            if root_id not in {"experiments4", "experiments5"}:
                continue
            try:
                relative = base64.b64decode(record["relative_path_b64"], validate=True).decode("utf-8")
            except (KeyError, ValueError, UnicodeDecodeError) as exc:
                raise AuditedEnvironmentError(f"invalid source-pre path at line {line_number}") from exc
            key = (root_id, relative)
            audit_item = needed.get(key)
            if audit_item is None:
                continue
            absolute = audit_item["absolute_path"]
            if absolute in found:
                raise AuditedEnvironmentError(f"duplicate source-pre record: {absolute}")
            if record.get("type") != "regular" or record.get("sha256") != audit_item["sha256"]:
                raise AuditedEnvironmentError(f"source-pre/audit mismatch: {absolute}")
            enriched = dict(record)
            enriched["relative_path"] = relative
            found[absolute] = enriched
    missing = sorted(set(item["absolute_path"] for item in needed.values()) - set(found))
    if missing:
        raise AuditedEnvironmentError(f"selected files missing from source-pre: {missing[:3]}")
    return found


def _snapshot_plan_overlap(repo_root: Path, selected: Iterable[dict[str, Any]]) -> int:
    plan = load_json(repo_root / "configs" / "snapshot-plan.json")
    planned = {
        (
            record.get("origin_root"),
            record.get("origin_relative_path"),
            record.get("origin_sha256"),
        )
        for record in plan.get("records", [])
    }
    return sum((item["root_id"], item["relative_path"], item["sha256"]) in planned for item in selected)


def _unique(values: Iterable[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[str] = set()
    for value in values:
        identity = json.dumps(value, sort_keys=True, ensure_ascii=False)
        if identity not in seen:
            seen.add(identity)
            result.append(value)
    return result


def _overlay_path(root_id: str, relative: str) -> str:
    return f"origins/{root_id}/{relative}"


def _labels_to_selected(selected: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for item in selected:
        for label in item["audit_labels"]:
            result.setdefault(label, []).append(item)
    return result


def _turn_entry(base: dict[str, Any], target: str, current_ids: list[str]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    result["variant_id"] = target
    result["selector"] = f"{result.get('stage', 'evaluation')}={target}"
    turn = "multiturn" if "multiturn" in target else "singleturn"
    if result.get("stage") == "evaluation":
        suffix = "multiturn_legacy" if turn == "multiturn" else "single_legacy"
        result["expands_to"] = {
            "aggregation": f"aggregation4_{suffix}",
            "denominator": f"denominator4_{suffix}",
            "error_policy": f"error4_{suffix}",
            "metric": f"metric4_{suffix}",
            "normalization": (
                "normalization4_multiturn_dateparser" if turn == "multiturn" else "normalization4_single_raw"
            ),
            "parser": "parser4_inline",
            "reporting": f"report4_{suffix}",
        }
    evidence = [
        item
        for item in result.get("origin_evidence", [])
        if item.get("origin_root") == "experiments4" and turn in item.get("relative_path", "").lower()
    ]
    if not evidence:
        evidence = [item for item in result.get("origin_evidence", []) if item.get("origin_root") == "experiments4"]
    result["origin_evidence"] = _unique(evidence)
    result["destinations"] = [
        _overlay_path(item["origin_root"], item["relative_path"]) for item in result["origin_evidence"]
    ]
    result["lineage_ids"] = [
        f"audited:{item.get('source_manifest_record_id', item.get('sha256'))}" for item in result["origin_evidence"]
    ]
    result["affected_outputs"] = _unique(list(result.get("affected_outputs", [])) + [f"{turn} final metrics"])
    result["distinction"] = f"turn-specific experiments4 {turn} behavior; experiments6 is a release/path alias"
    result["equivalence_state"] = "exp4_exp6_release_alias"
    result["replaces"] = current_ids
    result["release_aliases"] = [value for value in current_ids if "6" in value]
    return result


def _merge_entries(entries: dict[str, dict[str, Any]], target: str, current_ids: list[str], reason: str) -> None:
    parts: list[dict[str, Any]] = []
    if target in entries and target not in current_ids:
        parts.append(entries.pop(target))
    for variant_id in current_ids:
        value = entries.pop(variant_id, None)
        if value is not None:
            parts.append(value)
    if not parts:
        raise AuditedEnvironmentError(f"merge has no registry inputs: {target}")
    result = copy.deepcopy(parts[0])
    result["variant_id"] = target
    result["selector"] = f"{result.get('stage', 'variant')}={target}"
    for key in ("affected_outputs", "origin_evidence", "lineage_ids", "destinations"):
        result[key] = _unique(value for part in parts for value in part.get(key, []))
    result["distinction"] = reason
    result["equivalence_state"] = "audited_semantic_merge"
    result["merged_from"] = _unique(value for part in parts for value in part.get("merged_from", [part["variant_id"]]))
    result["release_aliases"] = sorted(value for value in result["merged_from"] if "6" in value)
    entries[target] = result


def _stage_for_add(item: dict[str, Any]) -> str:
    label = item["label"]
    category = item["category"]
    if category in {"figure5", "table15"}:
        return "reporting"
    if category == "table16":
        return "evaluation"
    if any(token in label for token in ("_index", "_ingest", "_build", "memory5_")):
        return "preprocessing"
    if label in {"self_refine_core45", "remem_support"}:
        return "support"
    return "inference"


def _evidence_from_selected(
    paths: Iterable[dict[str, Any]], source_index: dict[str, dict[str, Any]], *, role: str
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in paths:
        source = source_index[item["absolute_path"]]
        record_id = source["record_id"]
        result.append(
            {
                "audit_role": role,
                "evidence_id": f"source-pre:{record_id}",
                "origin_root": item["root_id"],
                "relative_path": item["relative_path"],
                "sha256": item["sha256"],
                "source_manifest_record_id": record_id,
            }
        )
    return result


def validate_registry_document(registry: dict[str, Any], *, expected_count: int | None = None) -> list[str]:
    if registry.get("schema") != REGISTRY_SCHEMA or not isinstance(registry.get("variants"), list):
        raise AuditedEnvironmentError("invalid audited registry schema")
    variants = registry["variants"]
    labels = [value.get("variant_id") for value in variants]
    if any(not isinstance(label, str) for label in labels) or len(labels) != len(set(labels)):
        raise AuditedEnvironmentError("audited registry labels are missing or duplicated")
    if registry.get("variant_count") != len(labels) or (expected_count is not None and len(labels) != expected_count):
        raise AuditedEnvironmentError("audited registry count mismatch")
    missing_required = sorted(EXPECTED_REQUIRED_LABELS - set(labels))
    forbidden = sorted(label for label in labels if FORBIDDEN_LABEL_RE.search(label))
    if missing_required or forbidden or "preprocess5_v1" in labels:
        raise AuditedEnvironmentError(
            f"audited label policy failed: missing={missing_required}, forbidden={forbidden}"
        )
    label_set = set(labels)
    dangling = sorted(
        (value["variant_id"], stage, target)
        for value in variants
        for stage, target in value.get("expands_to", {}).items()
        if target not in label_set
    )
    if dangling:
        raise AuditedEnvironmentError(f"dangling expands_to values: {dangling}")
    for value in variants:
        entrypoints = value.get("entrypoints", [])
        if not isinstance(entrypoints, list):
            raise AuditedEnvironmentError(f"invalid entrypoints for {value['variant_id']}")
        for entrypoint in entrypoints:
            try:
                relative = clean_relative_path(entrypoint)
            except (TypeError, ValueError) as exc:
                raise AuditedEnvironmentError(f"unsafe entrypoint for {value['variant_id']}") from exc
            if relative.suffix not in {".py", ".sh"}:
                raise AuditedEnvironmentError(f"non-executable entrypoint for {value['variant_id']}: {entrypoint}")
    return labels


def build_registry_documents(
    repo_root: Path,
    audit: dict[str, Any],
    source_pre_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected = selected_audit_files(audit)
    source_index = _source_pre_index(source_pre_path, selected)
    by_label = _labels_to_selected(selected)
    legacy = load_json(repo_root / "variants" / "registry.json")
    entries = {value["variant_id"]: copy.deepcopy(value) for value in legacy["variants"]}

    for rule in audit["replace_with_turn_specific_labels"]:
        old = [entries.pop(value) for value in rule["current_ids"]]
        base = old[0]
        for target in rule["target_labels"]:
            if target in entries:
                raise AuditedEnvironmentError(f"turn-specific target already exists: {target}")
            value = _turn_entry(base, target, rule["current_ids"])
            selected_for_target = by_label.get(target, [])
            if selected_for_target:
                value["origin_evidence"] = _evidence_from_selected(
                    selected_for_target, source_index, role="turn_specific_overlay"
                )
                value["destinations"] = [
                    _overlay_path(item["root_id"], item["relative_path"]) for item in selected_for_target
                ]
                value["entrypoints"] = value["destinations"]
            entries[target] = value

    for rule in audit["merge"]:
        _merge_entries(entries, rule["target_label"], rule["current_ids"], rule["reason"])

    for rule in audit["remove_from_executable_registry"]:
        for variant_id in rule["current_ids"]:
            if entries.pop(variant_id, None) is None:
                raise AuditedEnvironmentError(f"remove target is absent: {variant_id}")

    for rule in audit["rename"]:
        value = entries.pop(rule["current_id"])
        target = rule["target_label"]
        value["variant_id"] = target
        value["selector"] = f"{value.get('stage', 'preprocessing')}={target}"
        value["renamed_from"] = rule["current_id"]
        value["distinction"] = rule["reason"]
        entries[target] = value

    for item in audit["add"]:
        label = item["label"]
        if label in entries:
            raise AuditedEnvironmentError(f"audit add collides with existing label: {label}")
        entrypoints = item.get("runnable_entrypoints", [])
        companions = item.get("companions", [])
        derived = item.get("derived_artifacts", [])
        evidence = []
        for role, values in (("runnable_entrypoint", entrypoints), ("companion", companions), ("derived_artifact", derived)):
            selected_values = []
            for value in values:
                match = next((candidate for candidate in selected if candidate["absolute_path"] == value["path"]), None)
                if match is None or match["sha256"] != value["sha256"]:
                    raise AuditedEnvironmentError(f"add record is not in selected overlay: {value['path']}")
                selected_values.append(match)
            evidence.extend(_evidence_from_selected(selected_values, source_index, role=role))
        stage = _stage_for_add(item)
        effective_entrypoints = companions if label in {"self_refine5_api", "self_refine5_vllm"} else entrypoints
        entries[label] = {
            "affected_outputs": [item["semantic_difference"]],
            "category": item["category"],
            "companions": [
                _overlay_path(Path(value["path"]).parts[3], PurePosixPath(*Path(value["path"]).parts[4:]).as_posix())
                for value in companions
            ],
            "dependencies": item.get("existing_dependencies", []),
            "destinations": [_overlay_path(value["origin_root"], value["relative_path"]) for value in evidence],
            "distinction": item["semantic_difference"],
            "entrypoints": [
                _overlay_path(Path(value["path"]).parts[3], PurePosixPath(*Path(value["path"]).parts[4:]).as_posix())
                for value in effective_entrypoints
            ],
            "equivalence_state": "retained_distinct",
            "kind": "audited_variant",
            "neutral_family": f"family.{stage}",
            "origin_evidence": evidence,
            "paper_role": item["paper_role"],
            "selectable": True,
            "selector": f"{stage}={label}",
            "semantic_parent": f"family.{stage}",
            "stage": stage,
            "variant_id": label,
        }

    for label, paths in by_label.items():
        value = entries.get(label)
        if value is None:
            continue
        value["overlay_paths"] = [
            _overlay_path(item["root_id"], item["relative_path"]) for item in paths
        ]
        if label == "eval5_canonical":
            by_relative = {item["relative_path"]: _overlay_path(item["root_id"], item["relative_path"]) for item in paths}
            value["entrypoints"] = [
                by_relative["evaluation/eval_singleturn.py"],
                by_relative["evaluation/eval_multiturn.py"],
            ]
            value["supporting_paths"] = [by_relative["src/evaluation/metrics.py"]]

    variants = sorted(entries.values(), key=lambda value: value["variant_id"])
    labels = [value["variant_id"] for value in variants]
    expected_count = audit["counts"]["projected_registry_count_after_merges_splits_and_adds"]
    if len(variants) != expected_count or len(labels) != len(set(labels)):
        raise AuditedEnvironmentError(f"audited registry count mismatch: {len(variants)} != {expected_count}")
    missing_required = sorted(EXPECTED_REQUIRED_LABELS - set(labels))
    forbidden = sorted(label for label in labels if FORBIDDEN_LABEL_RE.search(label))
    if missing_required or forbidden or "preprocess5_v1" in labels:
        raise AuditedEnvironmentError(
            f"audited label policy failed: missing={missing_required}, forbidden={forbidden}"
        )
    registry = {
        "audit_sha256": EXPECTED_AUDIT_SHA256,
        "families": sorted({value["stage"] for value in variants}),
        "label_policy": audit["label_policy"],
        "legacy_registry_sha256": sha256_file(repo_root / "variants" / "registry.json")[0],
        "release_alias_policy": {
            "experiments6": "experiments4 release/path alias; no independent output-affecting eval6/infer6 label",
        },
        "schema": REGISTRY_SCHEMA,
        "selected_overlay_file_count": len(selected),
        "source_pre_sha256": sha256_file(source_pre_path)[0],
        "variant_count": len(variants),
        "variants": variants,
    }
    validate_registry_document(registry, expected_count=expected_count)
    registry_sha = sha256_bytes(_json_bytes(registry))
    profiles = {
        "audit_sha256": registry["audit_sha256"],
        "base": {
            "physical_snapshot": "environments/exp6-base-v1-4f6a15ffa3db8dc3",
            "release_alias_of": "experiments4",
            "semantic_label": "exp4_release_alias",
        },
        "implicit_default": False,
        "overlay": "environments/exp45-audited-overlay-v2-782df2d84077f93b",
        "profiles": [
            {
                "description": "Audited experiments4/5 semantic variants over the immutable experiments6 release snapshot alias.",
                "execution_policy": "dry-run any sealed entrypoint; execute only the contained environment smoke recipe",
                "profile_id": PROFILE_ID,
                "variant_labels": labels,
            }
        ],
        "registry_sha256": registry_sha,
        "schema": PROFILES_SCHEMA,
    }
    return registry, profiles


def generate_audited_configs(
    repo_root: Path,
    audit_input: Path,
    source_pre_path: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve(strict=True)
    audit_input = audit_input.resolve(strict=True)
    audit_bytes = audit_input.read_bytes()
    audit_sha = sha256_bytes(audit_bytes)
    audit = json.loads(audit_bytes)
    expected_audit = EXPECTED_AUDIT_SHA256
    if audit_sha != expected_audit:
        raise AuditedEnvironmentError(f"unexpected audit SHA256: {audit_sha}")
    selected = selected_audit_files(audit)
    expected_overlap = audit["counts"]["selected_overlay_files_already_in_snapshot_plan"]
    overlap = _snapshot_plan_overlap(repo_root, selected)
    if overlap != expected_overlap:
        raise AuditedEnvironmentError(f"snapshot-plan overlap mismatch: {overlap} != {expected_overlap}")
    registry, profiles = build_registry_documents(repo_root, audit, source_pre_path)
    config_root = repo_root / "configs" / "environment"
    audit_output = config_root / f"audit-overlay-{audit_sha}.json"
    registry_output = config_root / "variant-registry-audited-v2.json"
    profiles_output = config_root / "variant-profiles-audited-v2.json"
    _write_or_verify(audit_output, audit_bytes)
    _write_or_verify(registry_output, _json_bytes(registry))
    _write_or_verify(profiles_output, _json_bytes(profiles))
    return {
        "audit_path": str(audit_output),
        "audit_sha256": audit_sha,
        "profiles_path": str(profiles_output),
        "profiles_sha256": sha256_file(profiles_output)[0],
        "registry_path": str(registry_output),
        "registry_sha256": sha256_file(registry_output)[0],
        "selected_file_count": len(selected),
        "snapshot_plan_overlap": overlap,
        "variant_count": registry["variant_count"],
    }


def _load_spec(repo_root: Path, spec_path: Path) -> tuple[bytes, dict[str, Any]]:
    spec_bytes = spec_path.resolve(strict=True).read_bytes()
    spec = json.loads(spec_bytes)
    if spec.get("schema") != OVERLAY_PUBLICATION_SCHEMA:
        raise AuditedEnvironmentError("unsupported audited overlay publication schema")
    actual_keys = set(spec.get("expected_sha256", {}))
    if actual_keys != EXPECTED_SHA256_KEYS:
        raise AuditedEnvironmentError(
            f"expected_sha256 keys must be exactly {sorted(EXPECTED_SHA256_KEYS)}; got {sorted(actual_keys)}"
        )
    return spec_bytes, spec


def publish_audited_overlay(repo_root: Path, spec_path: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve(strict=True)
    spec_bytes, spec = _load_spec(repo_root, spec_path)
    canonical_spec = audited_spec_path(repo_root).resolve(strict=True)
    if spec_path.resolve(strict=True) != canonical_spec:
        raise AuditedEnvironmentError("publication requires the checked-in canonical v2 spec")
    audit_path = resolve_config_path(repo_root, spec["audit_path"]).resolve(strict=True)
    source_pre_path = resolve_config_path(repo_root, spec["source_pre_path"]).resolve(strict=True)
    cp0_path = resolve_config_path(repo_root, spec["cp0_path"]).resolve(strict=True)
    registry_path = resolve_config_path(repo_root, spec["registry_path"]).resolve(strict=True)
    profiles_path = resolve_config_path(repo_root, spec["profiles_path"]).resolve(strict=True)
    base_snapshot = resolve_config_path(repo_root, spec["base_snapshot"]).resolve(strict=True)
    destination = resolve_config_path(repo_root, spec["destination"])
    environments_root = (repo_root / "environments").resolve(strict=False)
    if destination.parent.resolve(strict=False) != environments_root:
        raise AuditedEnvironmentError("audited overlays must be direct children of experiments7/environments")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)

    bound_paths = {
        "audit": audit_path,
        "source_pre": source_pre_path,
        "cp0": cp0_path,
        "registry": registry_path,
        "profiles": profiles_path,
    }
    bound_hashes = {name: sha256_file(path)[0] for name, path in bound_paths.items()}
    if bound_hashes != spec["expected_sha256"]:
        mismatched = sorted(name for name in EXPECTED_SHA256_KEYS if bound_hashes[name] != spec["expected_sha256"][name])
        raise AuditedEnvironmentError(f"publication spec hash mismatch: {mismatched}")
    base_result = validate_snapshot(
        repo_root, base_snapshot, expected_seal_sha256=spec["expected_base_seal_sha256"]
    )
    cp0 = load_json(cp0_path)
    if cp0.get("checkpoint") != 0 or cp0.get("semantic_bindings", {}).get("source_pre_record_count") != spec["expected_source_pre_record_count"]:
        raise AuditedEnvironmentError("CP0/source-pre binding mismatch")
    audit_bytes = audit_path.read_bytes()
    audit = json.loads(audit_bytes)
    selected = selected_audit_files(audit)
    expected_counts = spec["expected_counts"]
    root_counts: dict[str, int] = {}
    for item in selected:
        root_counts[item["root_id"]] = root_counts.get(item["root_id"], 0) + 1
    if len(selected) != expected_counts["selected"] or root_counts != expected_counts["by_root"]:
        raise AuditedEnvironmentError("selected overlay counts do not match publication spec")
    overlap = _snapshot_plan_overlap(repo_root, selected)
    if overlap != expected_counts["already_planned"] or len(selected) - overlap != expected_counts["newly_selected"]:
        raise AuditedEnvironmentError("snapshot-plan overlap counts do not match publication spec")
    source_index = _source_pre_index(source_pre_path, selected)
    registry = load_json(registry_path)
    profiles = load_json(profiles_path)
    validate_registry_document(registry, expected_count=expected_counts["variants"])
    if (
        profiles.get("schema") != PROFILES_SCHEMA
        or profiles.get("registry_sha256") != bound_hashes["registry"]
        or profiles.get("audit_sha256") != bound_hashes["audit"]
        or profiles.get("overlay") != spec["destination"]
    ):
        raise AuditedEnvironmentError("invalid audited profiles")

    environments_root.mkdir(parents=True, exist_ok=True)
    stage = environments_root / f".{destination.name}.staging.{os.getpid()}.{uuid.uuid4().hex}"
    stage.mkdir(mode=0o700)
    source_roots = {root_id: Path("/data/minseo") / root_id for root_id in root_counts}
    before = {root_id: path.stat() for root_id, path in source_roots.items()}
    try:
        records: list[dict[str, Any]] = []
        for item in selected:
            root_id = item["root_id"]
            origin_stage = stage / "origins" / root_id
            record = _copy_record(source_roots[root_id], origin_stage, source_index[item["absolute_path"]])
            record["schema"] = OVERLAY_MANIFEST_SCHEMA
            record["path"] = _overlay_path(root_id, item["relative_path"])
            record["audit_labels"] = item["audit_labels"]
            record["audit_relations"] = item["audit_relations"]
            records.append(record)
        records.sort(key=lambda value: value["path"])
        manifest_bytes = b"".join(_json_bytes(record) for record in records)
        control = stage / ".experiment-env-overlay"
        _write_bytes(control / "audit.json", audit_bytes)
        _write_bytes(control / "manifest.jsonl", manifest_bytes)
        _write_bytes(control / "profiles.json", profiles_path.read_bytes())
        _write_bytes(control / "registry.json", registry_path.read_bytes())
        seal = {
            "already_planned_file_count": overlap,
            "audit_path": spec["audit_path"],
            "audit_sha256": bound_hashes["audit"],
            "base_seal_sha256": base_result["seal_sha256"],
            "base_semantic_label": "exp4_release_alias",
            "base_snapshot": spec["base_snapshot"],
            "cp0_path": spec["cp0_path"],
            "cp0_sha256": bound_hashes["cp0"],
            "manifest_record_count": len(records),
            "manifest_sha256": sha256_bytes(manifest_bytes),
            "newly_selected_file_count": len(records) - overlap,
            "profiles_sha256": bound_hashes["profiles"],
            "regular_bytes": sum(record["bytes"] for record in records),
            "registry_sha256": bound_hashes["registry"],
            "root_file_counts": root_counts,
            "schema": OVERLAY_SEAL_SCHEMA,
            "snapshot_id": destination.name,
            "source_pre_path": spec["source_pre_path"],
            "source_pre_record_count": spec["expected_source_pre_record_count"],
            "source_pre_sha256": bound_hashes["source_pre"],
            "spec_path": str(canonical_spec.relative_to(repo_root)),
            "spec_sha256": sha256_bytes(spec_bytes),
            "variant_count": registry["variant_count"],
        }
        seal_bytes = _json_bytes(seal)
        _write_bytes(control / "seal.json", seal_bytes)
        for root_id, path in source_roots.items():
            after = path.stat()
            prior = before[root_id]
            if (prior.st_dev, prior.st_ino, prior.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_mtime_ns):
                raise AuditedEnvironmentError(f"source root changed during publication: {root_id}")
        _freeze_directories(stage)
        _rename_noreplace(stage, destination)
    except Exception:
        if stage.exists():
            for current, _, files in os.walk(stage, topdown=False, followlinks=False):
                os.chmod(current, 0o700)
                for name in files:
                    path = Path(current) / name
                    if not path.is_symlink():
                        os.chmod(path, 0o600)
            shutil.rmtree(stage)
        raise
    return {"destination": str(destination), "seal": seal, "seal_sha256": sha256_bytes(seal_bytes)}


def validate_audited_overlay(
    repo_root: Path, overlay_path: Path, *, expected_seal_sha256: str | None = None
) -> dict[str, Any]:
    repo_root = repo_root.resolve(strict=True)
    overlay = overlay_path.resolve(strict=True)
    environments_root = (repo_root / "environments").resolve(strict=True)
    if overlay.parent != environments_root or overlay_path.is_symlink():
        raise ValidationError("overlay is not a direct non-symlink child of experiments7/environments")
    control = overlay / ".experiment-env-overlay"
    control_info = control.lstat()
    if control.is_symlink() or not stat.S_ISDIR(control_info.st_mode) or control_info.st_mode & 0o222:
        raise ValidationError("overlay control directory is missing, symlinked, or writable")
    with os.scandir(control) as entries:
        members = {entry.name: entry for entry in entries}
    if set(members) != CONTROL_NAMES:
        raise ValidationError(f"overlay control membership mismatch: {sorted(members)}")
    paths = {name: control / name for name in CONTROL_NAMES}
    for name, entry in members.items():
        path = paths[name]
        if entry.is_symlink() or not entry.is_file(follow_symlinks=False) or path.lstat().st_mode & 0o222:
            raise ValidationError(f"invalid or writable overlay control file: {path}")

    seal_bytes = paths["seal.json"].read_bytes()
    seal_sha = sha256_bytes(seal_bytes)
    if expected_seal_sha256 and seal_sha != expected_seal_sha256:
        raise ValidationError("overlay seal hash does not match expected hash")
    seal = json.loads(seal_bytes)
    if seal.get("schema") != OVERLAY_SEAL_SCHEMA:
        raise ValidationError("unsupported audited overlay seal schema")
    canonical_spec_path = audited_spec_path(repo_root).resolve(strict=True)
    spec_bytes, spec = _load_spec(repo_root, canonical_spec_path)
    if seal.get("spec_path") != str(canonical_spec_path.relative_to(repo_root)):
        raise ValidationError("overlay seal does not name the canonical spec")
    if seal.get("spec_sha256") != sha256_bytes(spec_bytes):
        raise ValidationError("overlay seal/canonical spec hash mismatch")
    if seal.get("snapshot_id") != Path(spec["destination"]).name:
        raise ValidationError("overlay seal snapshot id differs from canonical spec")
    if seal.get("base_seal_sha256") != spec["expected_base_seal_sha256"]:
        raise ValidationError("overlay base seal differs from canonical spec")

    bound_paths = {
        "audit": resolve_config_path(repo_root, spec["audit_path"]),
        "cp0": resolve_config_path(repo_root, spec["cp0_path"]),
        "profiles": resolve_config_path(repo_root, spec["profiles_path"]),
        "registry": resolve_config_path(repo_root, spec["registry_path"]),
        "source_pre": resolve_config_path(repo_root, spec["source_pre_path"]),
    }
    bound_hashes = {name: sha256_file(path)[0] for name, path in bound_paths.items()}
    if bound_hashes != spec["expected_sha256"]:
        raise ValidationError("canonical spec inputs no longer match pinned hashes")
    if (
        seal.get("audit_path") != spec["audit_path"]
        or seal.get("cp0_path") != spec["cp0_path"]
        or seal.get("source_pre_path") != spec["source_pre_path"]
        or seal.get("base_snapshot") != spec["base_snapshot"]
        or seal.get("audit_sha256") != bound_hashes["audit"]
        or seal.get("cp0_sha256") != bound_hashes["cp0"]
        or seal.get("source_pre_sha256") != bound_hashes["source_pre"]
        or seal.get("registry_sha256") != bound_hashes["registry"]
        or seal.get("profiles_sha256") != bound_hashes["profiles"]
    ):
        raise ValidationError("overlay seal differs from canonical spec bindings")
    checks = {
        "audit.json": "audit_sha256",
        "manifest.jsonl": "manifest_sha256",
        "profiles.json": "profiles_sha256",
        "registry.json": "registry_sha256",
    }
    for name, field in checks.items():
        if sha256_file(paths[name])[0] != seal.get(field):
            raise ValidationError(f"overlay {name} hash mismatch")
    validate_snapshot(
        repo_root,
        resolve_config_path(repo_root, seal["base_snapshot"]),
        expected_seal_sha256=spec["expected_base_seal_sha256"],
    )

    audit = json.loads(paths["audit.json"].read_bytes())
    selected = selected_audit_files(audit)
    expected = {_overlay_path(item["root_id"], item["relative_path"]): item for item in selected}
    seen: set[str] = set()
    total_bytes = 0
    root_counts: dict[str, int] = {}
    for line_number, line in enumerate(paths["manifest.jsonl"].read_bytes().splitlines(), start=1):
        try:
            record = json.loads(line)
            relative = clean_relative_path(record["path"]).as_posix()
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            raise ValidationError(f"invalid overlay manifest record at line {line_number}") from exc
        if relative in seen or relative.startswith(".experiment-env-overlay/"):
            raise ValidationError(f"duplicate or reserved overlay path: {relative}")
        seen.add(relative)
        audit_item = expected.get(relative)
        if audit_item is None or record.get("sha256") != audit_item["sha256"]:
            raise ValidationError(f"overlay record is not audit-bound: {relative}")
        path = overlay.joinpath(*PurePosixPath(relative).parts)
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or path.is_symlink() or info.st_mode & 0o222:
            raise ValidationError(f"overlay payload is not frozen regular file: {relative}")
        actual_hash, size = sha256_file(path)
        if actual_hash != record["sha256"] or size != record["bytes"]:
            raise ValidationError(f"overlay payload content mismatch: {relative}")
        if sorted(record.get("audit_labels", [])) != audit_item["audit_labels"]:
            raise ValidationError(f"overlay audit labels mismatch: {relative}")
        total_bytes += size
        root_counts[audit_item["root_id"]] = root_counts.get(audit_item["root_id"], 0) + 1
    actual: set[str] = set()
    for current, names_in_dir, files in os.walk(overlay, topdown=True, followlinks=False):
        base = Path(current)
        if base.stat().st_mode & 0o222:
            raise ValidationError(f"overlay directory is writable: {base}")
        relative_base = base.relative_to(overlay)
        if relative_base == Path(".experiment-env-overlay"):
            names_in_dir[:] = []
            continue
        for name in list(names_in_dir):
            path = base / name
            if path.is_symlink():
                raise ValidationError(f"overlay payload symlink is forbidden: {path}")
        for name in files:
            actual.add((relative_base / name).as_posix())
    if seen != set(expected) or actual != seen:
        raise ValidationError("overlay payload differs from audit-sealed manifest")
    if len(seen) != seal["manifest_record_count"] or total_bytes != seal["regular_bytes"]:
        raise ValidationError("overlay totals differ from seal")
    if root_counts != seal["root_file_counts"]:
        raise ValidationError("overlay root totals differ from seal")

    registry = json.loads(paths["registry.json"].read_bytes())
    profiles = json.loads(paths["profiles.json"].read_bytes())
    try:
        labels = validate_registry_document(registry, expected_count=seal["variant_count"])
    except AuditedEnvironmentError as exc:
        raise ValidationError(str(exc)) from exc
    if (
        profiles.get("schema") != PROFILES_SCHEMA
        or profiles.get("registry_sha256") != seal["registry_sha256"]
        or profiles.get("audit_sha256") != seal["audit_sha256"]
        or profiles.get("overlay") != spec["destination"]
    ):
        raise ValidationError("overlay registry/profile relationship mismatch")
    for value in registry["variants"]:
        for entrypoint in value.get("entrypoints", []):
            if entrypoint not in seen:
                raise ValidationError(f"registry entrypoint is not sealed payload: {entrypoint}")
            path = overlay.joinpath(*clean_relative_path(entrypoint).parts)
            if path.is_symlink() or not path.is_file():
                raise ValidationError(f"registry entrypoint is not a regular file: {entrypoint}")
    return {
        "already_planned_file_count": seal["already_planned_file_count"],
        "manifest_record_count": len(seen),
        "newly_selected_file_count": seal["newly_selected_file_count"],
        "regular_bytes": total_bytes,
        "root_file_counts": root_counts,
        "seal_sha256": seal_sha,
        "snapshot": str(overlay),
        "status": "ok",
        "variant_count": len(labels),
    }


def audited_spec_path(repo_root: Path) -> Path:
    return repo_root / "configs" / "environment" / "exp45-audited-overlay-v2.snapshot.json"


def audited_doctor(repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve(strict=True)
    _, spec = _load_spec(repo_root, audited_spec_path(repo_root))
    overlay = resolve_config_path(repo_root, spec["destination"])
    overlay_result = validate_audited_overlay(repo_root, overlay)
    base_result = validate_snapshot(
        repo_root,
        resolve_config_path(repo_root, spec["base_snapshot"]),
        expected_seal_sha256=spec["expected_base_seal_sha256"],
    )
    return {
        "base": base_result,
        "base_semantics": "experiments4_release_alias",
        "overlay": overlay_result,
        "profile_id": PROFILE_ID,
        "status": "ok",
    }


def _sealed_registry(repo_root: Path) -> tuple[Path, dict[str, Any]]:
    _, spec = _load_spec(repo_root, audited_spec_path(repo_root))
    overlay = resolve_config_path(repo_root, spec["destination"]).resolve(strict=True)
    validate_audited_overlay(repo_root, overlay)
    registry = load_json(overlay / ".experiment-env-overlay" / "registry.json")
    return overlay, registry


def audited_variants(repo_root: Path) -> list[dict[str, Any]]:
    _, registry = _sealed_registry(repo_root)
    return [
        {
            "category": value.get("category", "legacy_stage_variant"),
            "entrypoint_count": len(value.get("entrypoints", [])),
            "stage": value["stage"],
            "variant_id": value["variant_id"],
        }
        for value in registry["variants"]
    ]


def audited_variant(repo_root: Path, variant_id: str) -> dict[str, Any]:
    _, registry = _sealed_registry(repo_root)
    for value in registry["variants"]:
        if value["variant_id"] == variant_id:
            return value
    raise AuditedEnvironmentError(f"unknown audited variant: {variant_id}")


def build_audited_plan(
    repo_root: Path,
    variant_id: str,
    entrypoint_index: int,
    run_id: str,
    passthrough: list[str] | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve(strict=True)
    if not RUN_ID_RE.fullmatch(run_id):
        raise AuditedEnvironmentError(f"invalid run id: {run_id!r}")
    overlay, registry = _sealed_registry(repo_root)
    try:
        variant = next(value for value in registry["variants"] if value["variant_id"] == variant_id)
    except StopIteration as exc:
        raise AuditedEnvironmentError(f"unknown audited variant: {variant_id}") from exc
    if entrypoint_index < 0:
        raise AuditedEnvironmentError("entrypoint index must be non-negative")
    entrypoints = variant.get("entrypoints", [])
    try:
        relative = clean_relative_path(entrypoints[entrypoint_index])
    except (IndexError, ValueError) as exc:
        raise AuditedEnvironmentError(
            f"variant {variant_id} has no entrypoint at index {entrypoint_index}"
        ) from exc
    entrypoint = overlay.joinpath(*relative.parts).resolve(strict=True)
    if not is_relative_to(entrypoint, overlay):
        raise AuditedEnvironmentError("sealed entrypoint escapes overlay")
    if entrypoint.suffix == ".py":
        argv = [sys.executable, str(entrypoint)]
    elif entrypoint.suffix == ".sh":
        argv = ["/bin/bash", str(entrypoint)]
    else:
        raise AuditedEnvironmentError(f"selected entrypoint is not executable source: {relative}")
    argv.extend(passthrough or [])
    return {
        "argv": redact_argv(argv),
        "execution_supported": False,
        "execution_support_reason": "legacy code may require external services and hard-coded output roots; use this plan for audited reproduction setup only",
        "profile_id": PROFILE_ID,
        "run_dir": str(repo_root / "runs" / run_id),
        "stage": variant["stage"],
        "variant_id": variant_id,
    }


def run_audited_smoke(repo_root: Path, run_id: str) -> int:
    repo_root = repo_root.resolve(strict=True)
    if not RUN_ID_RE.fullmatch(run_id):
        raise AuditedEnvironmentError(f"invalid run id: {run_id!r}")
    doctor_result = audited_doctor(repo_root)
    runs_root = (repo_root / "runs").resolve(strict=True)
    run_dir = runs_root / run_id
    if run_dir.exists() or run_dir.is_symlink():
        raise FileExistsError(run_dir)
    run_dir.mkdir(mode=0o700)
    if run_dir.resolve(strict=True).parent != runs_root:
        raise AuditedEnvironmentError("smoke run directory containment failed")
    output = run_dir / "smoke.txt"
    command = [
        sys.executable,
        "-B",
        "-c",
        "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('ok\\n', encoding='utf-8')",
        str(output),
    ]
    environment = dict(os.environ)
    environment.update(
        {
            "EXPERIMENTS7_ROOT": str(repo_root),
            "EXPERIMENT_RUN_DIR": str(run_dir),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "TMPDIR": str(run_dir),
        }
    )
    write_json_exclusive(
        run_dir / "run-manifest.json",
        {
            "argv": redact_argv(command),
            "credential_environment": credential_presence(environment),
            "doctor": doctor_result,
            "profile_id": PROFILE_ID,
            "run_dir": str(run_dir),
            "schema": "experiments7-audited-smoke-manifest/v2",
            "working_directory": str(repo_root),
        },
    )
    completed = subprocess.run(command, cwd=repo_root, env=environment, check=False)
    write_json_exclusive(
        run_dir / "result.json",
        {
            "exit_code": completed.returncode,
            "profile_id": PROFILE_ID,
            "schema": "experiments7-audited-smoke-result/v2",
        },
    )
    return completed.returncode
