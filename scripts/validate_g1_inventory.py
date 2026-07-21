#!/usr/bin/env python3
"""Fail-closed G1 registry, lineage, and paper-inventory validator.

This validator is deliberately read-only.  It reads only generated
experiments7 artifacts plus the installed pypdf module named in task evidence.
It never opens experiments4/5/6, experiments7/_paper, or a PDF.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from typing import Any, Iterable


REPORT_SCHEMA = "experiments7-g1-inventory-validation/v1"
FORBIDDEN_KEYS = frozenset({"status", "verified", "derived_state"})
HEX64 = re.compile(r"^[0-9a-f]{64}$")
NUMBER = re.compile(r"^[+-]?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$")
ROOT_README_BYTES = b"# experiments7\n\nSee [docs/README.md](docs/README.md).\n"
ROOT_README_SHA256 = "3f32154f9bf480fcc077e94d26378a6997811bed4fc955160567a615dd4b32fc"
PAPER_RELATIVE_PATH = "_paper/8. Latent_Preference_Modeling_for_Cross_Session_Personalized_Tool_Calling.pdf"
PAPER_SHA256 = "58de92a7746f671917a4faef5301efa02f05b440888f27845c34d40db5c3d6d8"
PAPER_BYTES = 1_851_838
PAPER_PAGES = 25
PAPER_TASK_ID = "/root/provenance_g1_paper_inventory_v2"
G1_TASK_ID = "/root/executor_g1_registry_writer"
EXPECTED_FAMILY_COUNT = 16
EXPECTED_VARIANT_COUNT = 85
EXPECTED_STRUCTURE_COUNT = 51
EXPECTED_RESULT_COUNT = 1_908
EXPECTED_ALIAS_COUNT = 86
EXPECTED_FIGURE5_SHA256 = "df0638a549c1430fb7089d0f8eb0fb13b02b0ef9079dd10446e16b015b38953a"
EXPECTED_FIGURE5_BYTES = 229_060
EXPECTED_PYPDF_PATH = Path("/home/minseo/.local/lib/python3.10/site-packages/pypdf/__init__.py")
EXPECTED_PYPDF_VERSION = "6.12.2"
EXPECTED_PYPDF_SHA256 = "612ff566b4378c13c7b1180cb2da89ac0b6cde5500a63fe530e30ae6a88b1b9c"

REQUIRED_VARIANTS = {
    "infer4_compat": ("stage_variant", "inference", "family.inference", "retained_distinct"),
    "infer5_native": ("stage_variant", "inference", "family.inference", "retained_distinct"),
    "infer6_release": ("stage_variant", "inference", "family.inference", "retained_distinct"),
    "eval4_legacy": ("stage_variant", "evaluation", "family.evaluation", "retained_distinct"),
    "eval5_canonical": ("stage_variant", "evaluation", "family.evaluation", "retained_distinct"),
    "eval6_release": ("stage_variant", "evaluation", "family.evaluation", "retained_distinct"),
    "eval4_mt_parse_0103a": ("stage_variant", "parser", "family.parser", "unresolved_equivalence"),
    "eval4_mt_parse_0103b": ("stage_variant", "parser", "family.parser", "unresolved_equivalence"),
}

EXPECTED_MATRICES = {
    "table3": {"locations": 434, "numeric": 434, "structural_blanks": 6},
    "table10": {"locations": 144, "numeric": 144},
    "table11": {"locations": 504, "numeric": 504},
    "table12": {"locations": 336, "numeric": 336},
    "table13": {"locations": 252, "numeric": 252},
    "table14": {"locations": 108, "canonical_results": 54, "aliases": 54},
    "table15": {"locations": 18, "numeric": 18},
    "table16": {"locations": 60, "numeric": 48, "not_applicable": 12},
    "figure4": {"marks": 32, "references": 2},
    "figure5": {"panel_a_marks": 5, "panel_b_marks": 40},
}

STRUCTURE_KEYS = {
    "schema", "structure_id", "kind", "page", "line_start", "line_end",
    "caption", "text_sha256",
}
RESULT_KEYS = {
    "schema", "result_id", "result_kind", "page", "section", "locator",
    "method", "model", "dataset", "setting", "preference_type", "metric",
    "displayed_value", "numeric_value", "value_type", "displayed_marker",
    "source_lexeme", "rounding_rule", "evidence_id",
}
ALIAS_KEYS = {
    "schema", "alias_id", "alias_of", "page", "section", "locator",
    "displayed_lexeme", "reason",
}
LINEAGE_KEYS = {
    "lineage_id", "origin_root", "origin_relative_path", "origin_sha256",
    "source_manifest_record_id", "planned_destination", "action", "stage",
    "family", "variant_id", "delta", "evidence_id",
}


class Blocked(RuntimeError):
    def __init__(self, code: str, detail: Any = None):
        super().__init__(code)
        self.code = code
        self.detail = detail


def require(condition: bool, code: str, detail: Any = None) -> None:
    if not condition:
        raise Blocked(code, detail)


def no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Blocked("DUPLICATE_JSON_KEY", key)
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    require(path.is_file() and not path.is_symlink(), "MISSING_OR_UNSAFE_JSON", str(path))
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return json.load(handle, object_pairs_hook=no_duplicate_object)
    except Blocked:
        raise
    except Exception as exc:
        raise Blocked("INVALID_JSON", {"path": str(path), "error": str(exc)}) from exc


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file() and not path.is_symlink(), "MISSING_OR_UNSAFE_JSONL", str(path))
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            raw_lines = list(handle)
            while raw_lines and raw_lines[-1].strip() == "":
                raw_lines.pop()
            for line_number, line in enumerate(raw_lines, start=1):
                require(line.strip() != "", "BLANK_JSONL_RECORD", {"path": str(path), "line": line_number})
                value = json.loads(line, object_pairs_hook=no_duplicate_object)
                require(isinstance(value, dict), "JSONL_RECORD_NOT_OBJECT", {"path": str(path), "line": line_number})
                rows.append(value)
    except Blocked:
        raise
    except Exception as exc:
        raise Blocked("INVALID_JSONL", {"path": str(path), "error": str(exc)}) from exc
    require(rows, "EMPTY_JSONL", str(path))
    return rows


def sha256_file(path: Path) -> tuple[str, int]:
    st = path.lstat()
    require(stat.S_ISREG(st.st_mode) and not path.is_symlink(), "UNSAFE_HASH_TARGET", str(path))
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def exact_keys(value: dict[str, Any], expected: set[str], code: str, optional: set[str] | None = None) -> None:
    allowed = expected | (optional or set())
    require(set(value) <= allowed and expected <= set(value), code, {
        "missing": sorted(expected - set(value)),
        "unknown": sorted(set(value) - allowed),
    })


def forbidden_paths(value: Any, prefix: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            here = f"{prefix}.{key}"
            if key in FORBIDDEN_KEYS:
                found.append(here)
            found.extend(forbidden_paths(child, here))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(forbidden_paths(child, f"{prefix}[{index}]"))
    return found


def unique(values: Iterable[Any], code: str) -> None:
    seen: set[Any] = set()
    duplicates: list[Any] = []
    for value in values:
        if value in seen:
            duplicates.append(value)
        seen.add(value)
    require(not duplicates, code, sorted({str(item) for item in duplicates}))


def safe_relative_path(value: Any) -> bool:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        return False
    if any(char in value for char in "*?["):
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and all(part not in ("", ".", "..") for part in path.parts)


def claim_matches_path(claim: Any, root: Path, relative: str) -> bool:
    if not isinstance(claim, str):
        return False
    return claim == relative or claim == str(root / relative)


def require_sha(value: Any, code: str) -> None:
    require(isinstance(value, str) and HEX64.fullmatch(value) is not None, code, value)


def counter_dict(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def locator_number(locator: dict[str, Any], key: str) -> int | None:
    value = locator.get(key)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        match = re.search(r"(?<![0-9])([0-9]{1,2})(?![0-9])", value)
        if match:
            return int(match.group(1))
    return None


def location_key(record: dict[str, Any]) -> tuple[int, str, str]:
    return (record["page"], record["section"], canonical_json(record["locator"]))


def validate_acyclic(adjacency: dict[str, list[str]], code: str) -> None:
    state: dict[str, int] = {}
    stack: list[str] = []

    def visit(node: str) -> None:
        mark = state.get(node, 0)
        if mark == 2:
            return
        if mark == 1:
            start = stack.index(node)
            raise Blocked(code, stack[start:] + [node])
        state[node] = 1
        stack.append(node)
        for target in adjacency.get(node, []):
            visit(target)
        stack.pop()
        state[node] = 2

    for node in sorted(adjacency):
        visit(node)


def validate_registry_and_lineage(root: Path, report: dict[str, Any]) -> None:
    registry_path = root / "variants/registry.json"
    evidence_path = root / "variants/g1-task-evidence.json"
    code_path = root / "lineage/code.jsonl"
    config_path = root / "lineage/config.jsonl"
    source_path = root / "manifests/source-pre.jsonl"
    cp0_path = root / "manifests/cp0-seal.json"
    readme_path = root / "README.md"

    registry = load_json(registry_path)
    evidence = load_json(evidence_path)
    code = load_jsonl(code_path)
    config = load_jsonl(config_path)
    lineage = code + config
    source = load_jsonl(source_path)
    cp0 = load_json(cp0_path)

    forbidden = (
        forbidden_paths(registry, "$.registry")
        + forbidden_paths(lineage, "$.lineage")
        + forbidden_paths(evidence, "$.g1_task_evidence")
    )
    require(not forbidden, "FORBIDDEN_STATE_FIELD", forbidden)

    exact_keys(
        registry,
        {"schema", "registry_schema_version", "source_manifest", "selection_policy", "families", "variants"},
        "REGISTRY_TOP_LEVEL_SCHEMA",
    )
    require(isinstance(registry["schema"], str) and registry["schema"], "REGISTRY_SCHEMA_ID")
    require(isinstance(registry["registry_schema_version"], int) and registry["registry_schema_version"] > 0,
            "REGISTRY_SCHEMA_VERSION")
    require(isinstance(registry["families"], list) and isinstance(registry["variants"], list),
            "REGISTRY_COLLECTION_TYPES")

    families = registry["families"]
    variants = registry["variants"]
    require(len(families) == EXPECTED_FAMILY_COUNT, "FAMILY_COUNT_MISMATCH", len(families))
    require(len(variants) == EXPECTED_VARIANT_COUNT, "VARIANT_COUNT_MISMATCH", len(variants))
    unique((item.get("family_id") for item in families), "DUPLICATE_FAMILY_ID")
    unique((item.get("variant_id") for item in variants), "DUPLICATE_VARIANT_ID")
    unique((item.get("selector") for item in variants), "DUPLICATE_SELECTOR")

    family_ids: set[str] = set()
    for item in families:
        exact_keys(item, {
            "family_id", "kind", "selector", "selectable", "stage", "semantic_parent",
            "neutral_family", "affected_outputs", "description",
        }, "FAMILY_SCHEMA")
        expected_id = f"family.{item['stage']}"
        require(item["family_id"] == expected_id, "NON_NEUTRAL_FAMILY_ID", item["family_id"])
        require(
            item["kind"] == "neutral_family"
            and item["selector"] is None
            and item["selectable"] is False
            and item["semantic_parent"] is None
            and item["neutral_family"] == item["family_id"],
            "INVALID_NEUTRAL_FAMILY",
            item["family_id"],
        )
        require(
            isinstance(item["affected_outputs"], list)
            and all(isinstance(output, str) and output.strip() for output in item["affected_outputs"]),
            "FAMILY_OUTPUTS_INVALID",
            item["family_id"],
        )
        require(isinstance(item["description"], str) and item["description"].strip(),
                "FAMILY_DESCRIPTION_INVALID", item["family_id"])
        family_ids.add(item["family_id"])

    variants_by_id: dict[str, dict[str, Any]] = {}
    registry_lineage_ids: list[str] = []
    registry_destinations: list[str] = []
    registry_origins: list[tuple[str, str, str, str, str]] = []
    for item in variants:
        exact_keys(item, {
            "variant_id", "kind", "selector", "selectable", "stage", "semantic_parent",
            "neutral_family", "affected_outputs", "equivalence_state", "distinction",
            "origin_evidence", "lineage_ids", "destinations",
        }, "VARIANT_SCHEMA", {"expands_to"})
        variant_id = item["variant_id"]
        expected_parent = f"family.{item['stage']}"
        require(
            item["kind"] == "stage_variant"
            and item["selectable"] is True
            and item["semantic_parent"] == expected_parent
            and item["neutral_family"] == expected_parent
            and expected_parent in family_ids,
            "INVALID_VARIANT_PARENT",
            variant_id,
        )
        require(item["selector"] == f"{item['stage']}={variant_id}", "INVALID_VARIANT_SELECTOR", variant_id)
        require(
            isinstance(item["affected_outputs"], list)
            and item["affected_outputs"]
            and all(isinstance(output, str) and output.strip() for output in item["affected_outputs"]),
            "VARIANT_OUTPUTS_INVALID",
            variant_id,
        )
        require(isinstance(item["distinction"], str) and item["distinction"].strip(),
                "VARIANT_DISTINCTION_INVALID", variant_id)
        require(isinstance(item["equivalence_state"], str) and item["equivalence_state"].strip(),
                "VARIANT_EQUIVALENCE_STATE_INVALID", variant_id)
        origins = item["origin_evidence"]
        lineage_ids = item["lineage_ids"]
        destinations = item["destinations"]
        require(
            isinstance(origins, list) and isinstance(lineage_ids, list) and isinstance(destinations, list)
            and len(origins) == len(lineage_ids) == len(destinations) > 0,
            "VARIANT_LINEAGE_CARDINALITY",
            variant_id,
        )
        for origin in origins:
            exact_keys(origin, {
                "origin_root", "relative_path", "sha256", "source_manifest_record_id", "evidence_id",
            }, "ORIGIN_EVIDENCE_SCHEMA")
            require(origin["origin_root"] in {"experiments4", "experiments5", "experiments6"},
                    "INVALID_ORIGIN_ROOT", origin["origin_root"])
            require(safe_relative_path(origin["relative_path"]), "INVALID_ORIGIN_PATH", origin["relative_path"])
            require_sha(origin["sha256"], "INVALID_ORIGIN_SHA256")
            require_sha(origin["source_manifest_record_id"], "INVALID_SOURCE_RECORD_ID")
            require(origin["evidence_id"] == f"source-pre:{origin['source_manifest_record_id']}",
                    "INVALID_ORIGIN_EVIDENCE_ID", origin["evidence_id"])
            registry_origins.append((
                origin["origin_root"], origin["relative_path"], origin["sha256"],
                origin["source_manifest_record_id"], origin["evidence_id"],
            ))
        for destination in destinations:
            require(safe_relative_path(destination), "INVALID_DESTINATION_PATH", destination)
        registry_lineage_ids.extend(lineage_ids)
        registry_destinations.extend(destinations)
        variants_by_id[variant_id] = item

    unique(registry_lineage_ids, "DUPLICATE_REGISTRY_LINEAGE_REFERENCE")
    unique(registry_destinations, "DUPLICATE_REGISTRY_DESTINATION")

    for variant_id, expected in REQUIRED_VARIANTS.items():
        item = variants_by_id.get(variant_id)
        require(item is not None, "MISSING_REQUIRED_VARIANT", variant_id)
        actual = (item["kind"], item["stage"], item["semantic_parent"], item["equivalence_state"])
        require(actual == expected, "REQUIRED_VARIANT_MISMATCH", {
            "variant_id": variant_id, "expected": expected, "actual": actual,
        })

    for path in (root / "variants/profiles.json", root / "variants/compatibility.json"):
        require(not os.path.lexists(path), "PREMATURE_G2_ARTIFACT", str(path))

    lineage_by_id: dict[str, dict[str, Any]] = {}
    origin_variant_keys: list[tuple[str, str, str]] = []
    for row in lineage:
        exact_keys(row, LINEAGE_KEYS, "LINEAGE_SCHEMA")
        require_sha(row["origin_sha256"], "INVALID_LINEAGE_ORIGIN_SHA256")
        require_sha(row["source_manifest_record_id"], "INVALID_LINEAGE_SOURCE_RECORD_ID")
        require(row["origin_root"] in {"experiments4", "experiments5", "experiments6"},
                "INVALID_LINEAGE_ORIGIN_ROOT", row["origin_root"])
        require(safe_relative_path(row["origin_relative_path"]), "INVALID_LINEAGE_ORIGIN_PATH",
                row["origin_relative_path"])
        require(safe_relative_path(row["planned_destination"]), "INVALID_LINEAGE_DESTINATION",
                row["planned_destination"])
        require(row["evidence_id"] == f"source-pre:{row['source_manifest_record_id']}",
                "INVALID_LINEAGE_EVIDENCE_ID", row["lineage_id"])
        require(isinstance(row["delta"], str) and row["delta"].startswith("pending:")
                and row["delta"].removeprefix("pending:").strip(),
                "INVALID_LINEAGE_DELTA", row["lineage_id"])
        lineage_by_id[row["lineage_id"]] = row
        origin_variant_keys.append((row["origin_root"], row["origin_relative_path"], row["variant_id"]))

    unique((row["lineage_id"] for row in lineage), "DUPLICATE_LINEAGE_ID")
    unique((row["planned_destination"] for row in lineage), "DUPLICATE_PLANNED_DESTINATION")
    unique(origin_variant_keys, "DUPLICATE_ORIGIN_PATH_VARIANT")
    require(set(registry_lineage_ids) == set(lineage_by_id), "REGISTRY_LINEAGE_SET_MISMATCH")
    require(set(registry_destinations) == {row["planned_destination"] for row in lineage},
            "REGISTRY_DESTINATION_SET_MISMATCH")

    for variant_id, item in variants_by_id.items():
        expected_origins = {
            (
                origin["origin_root"], origin["relative_path"], origin["sha256"],
                origin["source_manifest_record_id"], origin["evidence_id"],
            )
            for origin in item["origin_evidence"]
        }
        expected_destinations = set(item["destinations"])
        for lineage_id in item["lineage_ids"]:
            row = lineage_by_id[lineage_id]
            require(
                row["variant_id"] == variant_id
                and row["stage"] == item["stage"]
                and row["family"] == item["neutral_family"]
                and row["planned_destination"] in expected_destinations
                and (
                    row["origin_root"], row["origin_relative_path"], row["origin_sha256"],
                    row["source_manifest_record_id"], row["evidence_id"],
                ) in expected_origins,
                "REGISTRY_LINEAGE_BINDING_MISMATCH",
                lineage_id,
            )

    source_by_id: dict[str, dict[str, Any]] = {}
    root_counts: Counter[str] = Counter()
    for row in source:
        record_id = row.get("record_id")
        require(isinstance(record_id, str) and HEX64.fullmatch(record_id), "INVALID_SOURCE_PRE_RECORD_ID")
        require(record_id not in source_by_id, "DUPLICATE_SOURCE_PRE_RECORD_ID", record_id)
        source_by_id[record_id] = row
        root_counts[row.get("root_id")] += 1

    for row in lineage:
        source_row = source_by_id.get(row["source_manifest_record_id"])
        require(source_row is not None, "LINEAGE_SOURCE_PRE_DANGLING", row["lineage_id"])
        try:
            relative = base64.b64decode(source_row["relative_path_b64"], validate=True).decode("utf-8")
        except Exception as exc:
            raise Blocked("SOURCE_PRE_PATH_DECODE_FAILED", row["lineage_id"]) from exc
        require(
            source_row.get("type") == "regular"
            and source_row.get("root_id") == row["origin_root"]
            and relative == row["origin_relative_path"]
            and source_row.get("sha256") == row["origin_sha256"],
            "LINEAGE_SOURCE_PRE_BINDING_MISMATCH",
            row["lineage_id"],
        )

    source_hash, source_bytes = sha256_file(source_path)
    cp0_hash, cp0_bytes = sha256_file(cp0_path)
    registry_hash, registry_bytes = sha256_file(registry_path)
    code_hash, code_bytes = sha256_file(code_path)
    config_hash, config_bytes = sha256_file(config_path)
    readme_hash, readme_bytes = sha256_file(readme_path)
    require(readme_path.read_bytes() == ROOT_README_BYTES, "ROOT_README_BYTES_MISMATCH")
    require(readme_hash == ROOT_README_SHA256, "ROOT_README_HASH_MISMATCH", readme_hash)
    require(cp0.get("checkpoint") == "CP0" and cp0.get("decision") == "SEALED",
            "CP0_NOT_SEALED")
    require(cp0["artifact_hashes"]["root_readme"]["sha256"] == readme_hash,
            "CP0_README_HASH_MISMATCH")
    require(cp0["source_pre"]["sha256"] == source_hash, "CP0_SOURCE_PRE_HASH_MISMATCH")
    require(cp0["source_pre"]["record_count"] == len(source), "CP0_SOURCE_PRE_COUNT_MISMATCH")
    require(cp0["source_pre"]["root_counts"] == dict(root_counts), "CP0_SOURCE_PRE_ROOT_COUNTS_MISMATCH")
    require(registry["source_manifest"] == {
        "path": "manifests/source-pre.jsonl", "sha256": source_hash,
    }, "REGISTRY_SOURCE_MANIFEST_MISMATCH")

    exact_keys(evidence, {
        "schema", "canonical_task_id", "lane", "agent_type", "terminal_state",
        "writer_scope", "reason_codes", "artifacts", "baseline_hashes",
        "validation_assertions", "registry_to_lineage_closure", "protected_writes",
    }, "G1_TASK_EVIDENCE_SCHEMA")
    require(
        evidence["canonical_task_id"] == G1_TASK_ID
        and evidence["lane"] == "Executor"
        and evidence["agent_type"] == "executor"
        and evidence["terminal_state"] == "COMPLETE"
        and evidence["writer_scope"] == ["experiments7/variants/**", "experiments7/lineage/**"],
        "G1_TASK_IDENTITY_OR_SCOPE_MISMATCH",
    )
    require(evidence["protected_writes"] == 0, "G1_PROTECTED_WRITE_NONZERO")
    require(evidence["registry_to_lineage_closure"] == {
        "referenced": len(lineage), "resolved_exactly_once": len(lineage),
    }, "G1_CLOSURE_EVIDENCE_MISMATCH")
    require(evidence["baseline_hashes"] == {
        "cp0_seal_sha256": cp0_hash,
        "source_pre_sha256": source_hash,
        "root_readme_sha256": readme_hash,
    }, "G1_BASELINE_HASH_MISMATCH")

    actual_artifacts = {
        "variants/registry.json": (registry_hash, registry_bytes, None),
        "lineage/code.jsonl": (code_hash, code_bytes, len(code)),
        "lineage/config.jsonl": (config_hash, config_bytes, len(config)),
    }
    claims = evidence["artifacts"]
    require(isinstance(claims, list) and len(claims) == 3, "G1_ARTIFACT_CLAIMS")
    unique((item.get("path") for item in claims), "DUPLICATE_G1_ARTIFACT_CLAIM")
    require({item.get("path") for item in claims} == set(actual_artifacts), "G1_ARTIFACT_SET_MISMATCH")
    for item in claims:
        path = item["path"]
        digest, size, count = actual_artifacts[path]
        require(item.get("sha256") == digest and item.get("bytes") == size,
                "G1_ARTIFACT_HASH_OR_SIZE_MISMATCH", path)
        if count is not None:
            require(item.get("record_count") == count, "G1_ARTIFACT_COUNT_MISMATCH", path)
    registry_claim = next(item for item in claims if item["path"] == "variants/registry.json")
    require(registry_claim.get("record_counts") == {
        "families": len(families),
        "variants": len(variants),
        "required_variants": len(REQUIRED_VARIANTS),
        "additional_variants": len(variants) - len(REQUIRED_VARIANTS),
    }, "G1_REGISTRY_COUNTS_MISMATCH")

    report["counts"].update({
        "families": len(families),
        "variants": len(variants),
        "required_variants": len(REQUIRED_VARIANTS),
        "additional_variants": len(variants) - len(REQUIRED_VARIANTS),
        "lineage_code": len(code),
        "lineage_config": len(config),
        "lineage_total": len(lineage),
        "source_pre_records": len(source),
        "source_pre_root_counts": dict(sorted(root_counts.items())),
    })
    report["hashes"].update({
        "root_readme_sha256": readme_hash,
        "cp0_seal_sha256": cp0_hash,
        "source_pre_sha256": source_hash,
        "registry_sha256": registry_hash,
        "lineage_code_sha256": code_hash,
        "lineage_config_sha256": config_hash,
    })


def validate_structure(rows: list[dict[str, Any]], layout_hash: str) -> dict[str, int]:
    unique((row.get("structure_id") for row in rows), "DUPLICATE_STRUCTURE_ID")
    locations: list[tuple[Any, ...]] = []
    counts: Counter[str] = Counter()
    for row in rows:
        exact_keys(row, STRUCTURE_KEYS, "STRUCTURE_RECORD_SCHEMA")
        require(row["schema"] == "paper-structure-v1", "STRUCTURE_SCHEMA_ID")
        require(row["kind"] in {"page", "table", "figure"}, "STRUCTURE_KIND", row["kind"])
        require(isinstance(row["page"], int) and 1 <= row["page"] <= PAPER_PAGES,
                "STRUCTURE_PAGE", row["structure_id"])
        require(
            isinstance(row["line_start"], int) and isinstance(row["line_end"], int)
            and 1 <= row["line_start"] <= row["line_end"],
            "STRUCTURE_LINE_RANGE",
            row["structure_id"],
        )
        require(row["caption"] is None or isinstance(row["caption"], str), "STRUCTURE_CAPTION")
        require(row["text_sha256"] == layout_hash, "STRUCTURE_TEXT_HASH_MISMATCH", row["structure_id"])
        counts[row["kind"]] += 1
        locations.append((row["kind"], row["page"], row["line_start"], row["line_end"]))
    unique(locations, "DUPLICATE_STRUCTURE_LOCATION")
    require(counts == Counter({"page": 25, "table": 16, "figure": 10}),
            "STRUCTURE_COUNTS_MISMATCH", dict(counts))
    pages = sorted(row["page"] for row in rows if row["kind"] == "page")
    require(pages == list(range(1, 26)), "STRUCTURE_PAGE_COVERAGE", pages)
    return {"pages": 25, "tables": 16, "figures": 10, "total": 51}


def validate_results(rows: list[dict[str, Any]]) -> None:
    unique((row.get("result_id") for row in rows), "DUPLICATE_RESULT_ID")
    for row in rows:
        exact_keys(row, RESULT_KEYS, "RESULT_RECORD_SCHEMA", {"derived_from"})
        require(row["schema"] == "paper-result-v1", "RESULT_SCHEMA_ID")
        require(row["result_kind"] in {
            "table_cell", "figure_mark", "narrative_value", "derived_value",
        }, "RESULT_KIND", row["result_id"])
        require(isinstance(row["page"], int) and 1 <= row["page"] <= PAPER_PAGES,
                "RESULT_PAGE", row["result_id"])
        require(isinstance(row["section"], str) and row["section"], "RESULT_SECTION", row["result_id"])
        require(isinstance(row["locator"], dict) and row["locator"], "RESULT_LOCATOR", row["result_id"])
        for key in ("method", "model", "dataset", "setting", "preference_type", "metric",
                    "displayed_value", "numeric_value", "displayed_marker", "source_lexeme"):
            require(row[key] is None or isinstance(row[key], str), "RESULT_FIELD_TYPE", {
                "result_id": row["result_id"], "field": key,
            })
        require(row["value_type"] in {
            "numeric", "not_applicable", "unlabeled_graphical_mark", "range", "derived",
        }, "RESULT_VALUE_TYPE", row["result_id"])
        rounding = row["rounding_rule"]
        exact_keys(rounding, {"display_precision", "mode", "evidence_requirement"},
                   "ROUNDING_RULE_SCHEMA")
        require(rounding["mode"] == "not_inferred_from_layout", "ROUNDING_MODE", row["result_id"])
        require(rounding["evidence_requirement"] == "producer_binding",
                "ROUNDING_EVIDENCE_REQUIREMENT", row["result_id"])
        require(
            rounding["display_precision"] is None
            or (isinstance(rounding["display_precision"], int) and rounding["display_precision"] >= 0),
            "ROUNDING_PRECISION",
            row["result_id"],
        )
        require(isinstance(row["evidence_id"], str) and row["evidence_id"], "RESULT_EVIDENCE_ID")
        if row["value_type"] in {"numeric", "derived"}:
            require(
                isinstance(row["numeric_value"], str)
                and NUMBER.fullmatch(row["numeric_value"]) is not None,
                "NUMERIC_VALUE_ENCODING",
                row["result_id"],
            )
            try:
                Decimal(row["numeric_value"])
            except InvalidOperation as exc:
                raise Blocked("INVALID_DECIMAL", row["result_id"]) from exc
        elif row["value_type"] == "not_applicable":
            require(
                row["numeric_value"] is None and row["displayed_marker"] == "N/A",
                "NOT_APPLICABLE_ENCODING",
                row["result_id"],
            )
        elif row["value_type"] == "unlabeled_graphical_mark":
            require(
                row["numeric_value"] is None
                and row["displayed_value"] is None
                and row["displayed_marker"] == "UNLABELED_GRAPHICAL_MARK",
                "UNLABELED_MARK_ENCODING",
                row["result_id"],
            )
        if "derived_from" in row:
            require(
                row["result_kind"] == "derived_value"
                and isinstance(row["derived_from"], list)
                and row["derived_from"]
                and all(isinstance(item, str) and item for item in row["derived_from"]),
                "DERIVED_FROM_SCHEMA",
                row["result_id"],
            )


def validate_aliases(
    aliases: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> None:
    result_ids = {row["result_id"] for row in results}
    alias_ids = {row.get("alias_id") for row in aliases}
    unique(alias_ids, "DUPLICATE_ALIAS_ID")
    require(result_ids.isdisjoint(alias_ids), "RESULT_ALIAS_ID_COLLISION")
    adjacency: dict[str, list[str]] = {}
    for row in aliases:
        exact_keys(row, ALIAS_KEYS, "ALIAS_RECORD_SCHEMA")
        require(row["schema"] == "paper-alias-v1", "ALIAS_SCHEMA_ID")
        require(isinstance(row["page"], int) and 1 <= row["page"] <= PAPER_PAGES,
                "ALIAS_PAGE", row["alias_id"])
        require(isinstance(row["section"], str) and row["section"], "ALIAS_SECTION", row["alias_id"])
        require(isinstance(row["locator"], dict) and row["locator"], "ALIAS_LOCATOR", row["alias_id"])
        require(isinstance(row["displayed_lexeme"], str) and row["displayed_lexeme"],
                "ALIAS_DISPLAYED_LEXEME", row["alias_id"])
        require(isinstance(row["reason"], str) and row["reason"], "ALIAS_REASON", row["alias_id"])
        require(row["alias_of"] in result_ids, "ALIAS_ENDPOINT_NOT_DIRECT_RESULT", {
            "alias_id": row["alias_id"], "alias_of": row["alias_of"],
        })
        adjacency[row["alias_id"]] = [row["alias_of"]]
    validate_acyclic(adjacency, "ALIAS_CYCLE")


def validate_locations_and_derivations(
    results: list[dict[str, Any]],
    aliases: list[dict[str, Any]],
) -> None:
    combined = [location_key(row) for row in results] + [location_key(row) for row in aliases]
    unique(combined, "DUPLICATE_PAPER_LOCATION")
    result_ids = {row["result_id"] for row in results}
    adjacency: dict[str, list[str]] = {}
    for row in results:
        sources = row.get("derived_from", [])
        for source in sources:
            require(source in result_ids, "DERIVED_FROM_DANGLING", {
                "result_id": row["result_id"], "source": source,
            })
        adjacency[row["result_id"]] = list(sources)
    validate_acyclic(adjacency, "DERIVATION_CYCLE")


def validate_expected_matrices(
    results: list[dict[str, Any]],
    aliases: list[dict[str, Any]],
) -> None:
    result_tables: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    alias_tables: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    figures: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    result_by_id = {row["result_id"]: row for row in results}

    for row in results:
        table = locator_number(row["locator"], "table")
        figure = locator_number(row["locator"], "figure")
        if table is not None:
            result_tables[table].append(row)
        if figure is not None:
            figures[figure].append(row)
    for row in aliases:
        table = locator_number(row["locator"], "table")
        if table is not None:
            alias_tables[table].append(row)

    for table, expected in ((3, 434), (10, 144), (11, 504), (12, 336), (13, 252), (15, 18)):
        rows = result_tables[table]
        require(len(rows) == expected, "TABLE_RESULT_COUNT_MISMATCH", {
            "table": table, "expected": expected, "actual": len(rows),
        })
        require(sum(row["value_type"] == "numeric" for row in rows) == expected,
                "TABLE_NUMERIC_COUNT_MISMATCH", table)

    table14_results = result_tables[14]
    table14_aliases = alias_tables[14]
    require(
        len(table14_results) == 54
        and len(table14_aliases) == 54
        and len(table14_results) + len(table14_aliases) == 108,
        "TABLE14_MATRIX_MISMATCH",
        {"results": len(table14_results), "aliases": len(table14_aliases)},
    )
    for alias in table14_aliases:
        endpoint = result_by_id[alias["alias_of"]]
        require(locator_number(endpoint["locator"], "table") == 3,
                "TABLE14_ALIAS_NOT_TO_TABLE3", alias["alias_id"])

    table16 = result_tables[16]
    require(len(table16) == 60, "TABLE16_LOCATION_COUNT", len(table16))
    require(sum(row["value_type"] == "numeric" for row in table16) == 48,
            "TABLE16_NUMERIC_COUNT")
    require(sum(row["value_type"] == "not_applicable" for row in table16) == 12,
            "TABLE16_NA_COUNT")

    figure4_marks = [
        row for row in figures[4] if row["result_kind"] == "figure_mark"
    ]
    references = [
        row for row in figure4_marks
        if "ground_truth" in row["result_id"].lower()
        or (isinstance(row["method"], str) and row["method"].lower().replace(" ", "_") == "ground_truth")
    ]
    require(len(references) == 2 and len(figure4_marks) - len(references) == 32,
            "FIGURE4_MATRIX_MISMATCH", {
                "marks": len(figure4_marks) - len(references), "references": len(references),
            })

    figure5_marks = [
        row for row in figures[5] if row["result_kind"] == "figure_mark"
    ]
    panel_a = [row for row in figure5_marks if ".f05.a." in row["result_id"]]
    panel_b = [row for row in figure5_marks if ".f05.b." in row["result_id"]]
    require(len(panel_a) == 5 and len(panel_b) == 40 and len(figure5_marks) == 45,
            "FIGURE5_MATRIX_MISMATCH", {
                "panel_a": len(panel_a), "panel_b": len(panel_b), "all_marks": len(figure5_marks),
            })


def validate_claimed_file(
    root: Path,
    claim: dict[str, Any],
    relative: str,
    expected_records: int | None = None,
) -> tuple[str, int]:
    exact_keys(claim, {"path", "sha256", "bytes", "records"}, "ARTIFACT_HASH_CLAIM_SCHEMA")
    require(claim_matches_path(claim["path"], root, relative), "ARTIFACT_CLAIM_PATH", claim["path"])
    digest, size = sha256_file(root / relative)
    require(claim["sha256"] == digest and claim["bytes"] == size,
            "ARTIFACT_CLAIM_HASH_OR_SIZE", relative)
    if expected_records is not None:
        require(claim["records"] == expected_records, "ARTIFACT_CLAIM_RECORDS", relative)
    return digest, size


def validate_paper_inventory(root: Path, report: dict[str, Any]) -> None:
    inventory = root / "paper_outputs/inventory"
    structure_path = inventory / "structure.jsonl"
    results_path = inventory / "results.jsonl"
    aliases_path = inventory / "aliases.jsonl"
    evidence_path = inventory / "task-evidence.json"
    layout_path = inventory / ".paper-layout.txt"
    figure5_path = inventory / "diagnostics/figure5-source.png"
    cp0_path = root / "manifests/cp0-seal.json"
    provider_path = root / "scripts/g0/provider.py"

    structure = load_jsonl(structure_path)
    results = load_jsonl(results_path)
    aliases = load_jsonl(aliases_path)
    evidence = load_json(evidence_path)
    forbidden = (
        forbidden_paths(structure, "$.structure")
        + forbidden_paths(results, "$.results")
        + forbidden_paths(aliases, "$.aliases")
        + forbidden_paths(evidence, "$.paper_task_evidence")
    )
    require(not forbidden, "FORBIDDEN_PAPER_STATE_FIELD", forbidden)

    layout_hash, layout_bytes = sha256_file(layout_path)
    figure5_hash, figure5_bytes = sha256_file(figure5_path)
    cp0_hash, _ = sha256_file(cp0_path)
    provider_hash, _ = sha256_file(provider_path)

    structure_counts = validate_structure(structure, layout_hash)
    validate_results(results)
    validate_aliases(aliases, results)
    validate_locations_and_derivations(results, aliases)
    validate_expected_matrices(results, aliases)

    exact_keys(evidence, {
        "schema", "task_id", "terminal_state", "source_bindings", "artifact_hashes",
        "record_counts", "expected_matrices", "reason_codes", "protected_writes",
    }, "PAPER_TASK_EVIDENCE_SCHEMA")
    require(evidence["schema"] == "paper-inventory-evidence-v1", "PAPER_TASK_EVIDENCE_SCHEMA_ID")
    require(evidence["task_id"] == PAPER_TASK_ID, "PAPER_TASK_ID", evidence["task_id"])
    require(str(evidence["terminal_state"]).upper() == "COMPLETE", "PAPER_TASK_TERMINAL_STATE")
    require(evidence["reason_codes"] == [], "PAPER_TASK_REASON_CODES", evidence["reason_codes"])
    require(evidence["protected_writes"] == {
        "experiments4": 0, "experiments5": 0, "experiments6": 0, "paper": 0,
    }, "PAPER_PROTECTED_WRITES")

    bindings = evidence["source_bindings"]
    exact_keys(bindings, {
        "paper", "layout", "figure5_source", "cp0_seal", "provider", "pypdf",
    }, "SOURCE_BINDINGS_SCHEMA")

    paper = bindings["paper"]
    exact_keys(paper, {"path", "sha256", "bytes", "pages"}, "PAPER_BINDING_SCHEMA")
    require(claim_matches_path(paper["path"], root, PAPER_RELATIVE_PATH), "PAPER_BINDING_PATH")
    require(
        paper["sha256"] == PAPER_SHA256
        and paper["bytes"] == PAPER_BYTES
        and paper["pages"] == PAPER_PAGES,
        "PAPER_BINDING_IDENTITY",
    )
    # Intentionally do not stat, hash, or open the protected PDF.

    layout = bindings["layout"]
    exact_keys(layout, {"path", "sha256", "bytes"}, "LAYOUT_BINDING_SCHEMA")
    require(claim_matches_path(layout["path"], root, "paper_outputs/inventory/.paper-layout.txt"),
            "LAYOUT_BINDING_PATH")
    require(layout["sha256"] == layout_hash and layout["bytes"] == layout_bytes,
            "LAYOUT_BINDING_IDENTITY")

    figure5 = bindings["figure5_source"]
    exact_keys(figure5, {"path", "sha256", "bytes"}, "FIGURE5_BINDING_SCHEMA")
    require(claim_matches_path(
        figure5["path"], root, "paper_outputs/inventory/diagnostics/figure5-source.png"
    ), "FIGURE5_BINDING_PATH")
    require(figure5["sha256"] == figure5_hash and figure5["bytes"] == figure5_bytes,
            "FIGURE5_BINDING_IDENTITY")
    require(figure5_hash == EXPECTED_FIGURE5_SHA256 and figure5_bytes == EXPECTED_FIGURE5_BYTES,
            "FIGURE5_PIN_MISMATCH")

    cp0 = bindings["cp0_seal"]
    exact_keys(cp0, {"path", "sha256"}, "CP0_BINDING_SCHEMA")
    require(claim_matches_path(cp0["path"], root, "manifests/cp0-seal.json"), "CP0_BINDING_PATH")
    require(cp0["sha256"] == cp0_hash, "CP0_BINDING_IDENTITY")

    provider = bindings["provider"]
    exact_keys(provider, {"path", "sha256"}, "PROVIDER_BINDING_SCHEMA")
    require(claim_matches_path(provider["path"], root, "scripts/g0/provider.py"), "PROVIDER_BINDING_PATH")
    require(provider["sha256"] == provider_hash, "PROVIDER_BINDING_IDENTITY")

    pypdf = bindings["pypdf"]
    exact_keys(pypdf, {"path", "version", "sha256"}, "PYPDF_BINDING_SCHEMA")
    require(pypdf["path"] == str(EXPECTED_PYPDF_PATH), "PYPDF_BINDING_PATH")
    require(pypdf["version"] == EXPECTED_PYPDF_VERSION, "PYPDF_BINDING_VERSION")
    require(pypdf["sha256"] == EXPECTED_PYPDF_SHA256, "PYPDF_BINDING_SHA256")
    pypdf_hash, _ = sha256_file(EXPECTED_PYPDF_PATH)
    require(pypdf_hash == EXPECTED_PYPDF_SHA256, "PYPDF_BINDING_IDENTITY")

    claims = evidence["artifact_hashes"]
    exact_keys(claims, {"structure", "results", "aliases"}, "PAPER_ARTIFACT_HASHES_SCHEMA")
    structure_hash, structure_bytes = validate_claimed_file(
        root, claims["structure"], "paper_outputs/inventory/structure.jsonl", len(structure)
    )
    results_hash, results_bytes = validate_claimed_file(
        root, claims["results"], "paper_outputs/inventory/results.jsonl", len(results)
    )
    aliases_hash, aliases_bytes = validate_claimed_file(
        root, claims["aliases"], "paper_outputs/inventory/aliases.jsonl", len(aliases)
    )

    record_counts = evidence["record_counts"]
    exact_keys(record_counts, {"structure", "results", "aliases"}, "RECORD_COUNTS_SCHEMA")
    require(record_counts["structure"] == structure_counts, "STRUCTURE_EVIDENCE_COUNTS")
    result_counts = record_counts["results"]
    exact_keys(result_counts, {"total", "by_result_kind", "by_value_type", "by_source"},
               "RESULT_COUNTS_SCHEMA")
    require(result_counts["total"] == len(results), "RESULT_TOTAL_EVIDENCE")
    require(result_counts["by_result_kind"] == counter_dict(row["result_kind"] for row in results),
            "RESULT_KIND_EVIDENCE")
    require(result_counts["by_value_type"] == counter_dict(row["value_type"] for row in results),
            "RESULT_VALUE_TYPE_EVIDENCE")
    require(
        isinstance(result_counts["by_source"], dict)
        and all(isinstance(key, str) and isinstance(value, int) and value >= 0
                for key, value in result_counts["by_source"].items())
        and sum(result_counts["by_source"].values()) == len(results),
        "RESULT_SOURCE_EVIDENCE",
    )
    alias_counts = record_counts["aliases"]
    exact_keys(alias_counts, {"total", "by_reason"}, "ALIAS_COUNTS_SCHEMA")
    require(alias_counts["total"] == len(aliases), "ALIAS_TOTAL_EVIDENCE")
    require(alias_counts["by_reason"] == counter_dict(row["reason"] for row in aliases),
            "ALIAS_REASON_EVIDENCE")
    require(len(structure) == EXPECTED_STRUCTURE_COUNT, "STRUCTURE_COUNT_MISMATCH", len(structure))
    require(len(results) == EXPECTED_RESULT_COUNT, "RESULT_COUNT_MISMATCH", len(results))
    require(len(aliases) == EXPECTED_ALIAS_COUNT, "ALIAS_COUNT_MISMATCH", len(aliases))
    require(evidence["expected_matrices"] == EXPECTED_MATRICES,
            "EXPECTED_MATRICES_EVIDENCE_MISMATCH")

    report["counts"].update({
        "paper_pages": 25,
        "paper_tables": 16,
        "paper_figures": 10,
        "paper_results": len(results),
        "paper_aliases": len(aliases),
        "paper_structure_records": len(structure),
    })
    report["hashes"].update({
        "paper_layout_sha256": layout_hash,
        "figure5_source_sha256": figure5_hash,
        "paper_structure_sha256": structure_hash,
        "paper_results_sha256": results_hash,
        "paper_aliases_sha256": aliases_hash,
    })
    report["bytes"] = {
        "paper_structure": structure_bytes,
        "paper_results": results_bytes,
        "paper_aliases": aliases_bytes,
    }


def validate(root: Path) -> dict[str, Any]:
    require(root.is_absolute(), "ROOT_MUST_BE_ABSOLUTE", str(root))
    st = root.lstat()
    require(stat.S_ISDIR(st.st_mode) and not root.is_symlink(), "UNSAFE_EXPERIMENTS7_ROOT", str(root))
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "state": "PASS",
        "checks": {
            "registry_lineage": "PASS",
            "paper_inventory": "PASS",
            "alias_closure": "PASS",
            "forbidden_state_fields": "PASS",
            "artifact_hashes": "PASS",
            "root_readme": "PASS",
        },
        "counts": {},
        "hashes": {},
        "reason_codes": [],
    }
    validate_registry_and_lineage(root, report)
    validate_paper_inventory(root, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="absolute experiments7 root (default: script parent parent)",
    )
    args = parser.parse_args()
    try:
        report = validate(args.root)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 0
    except Blocked as exc:
        report = {
            "schema": REPORT_SCHEMA,
            "state": "BLOCKED",
            "checks": {},
            "counts": {},
            "hashes": {},
            "reason_codes": [exc.code],
        }
        if exc.detail is not None:
            report["detail"] = exc.detail
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 2
    except Exception as exc:
        report = {
            "schema": REPORT_SCHEMA,
            "state": "BLOCKED",
            "checks": {},
            "counts": {},
            "hashes": {},
            "reason_codes": ["UNEXPECTED_VALIDATOR_ERROR"],
            "detail": {"type": type(exc).__name__, "message": str(exc)},
        }
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
