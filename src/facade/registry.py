"""Fail-closed loaders and validators for G2 facade control artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Iterable

from .metadata import REFERENCE_ONLY, STAGES, _ensure_selector_expansion, _execution_kind
from .selection import sha256_file


FORBIDDEN_INPUT_KEYS = frozenset({"status", "verified", "derived_state"})
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
SECRET_VALUE = re.compile(r"(?:sk-[A-Za-z0-9]|AIza[A-Za-z0-9]|-----BEGIN [A-Z ]+PRIVATE KEY-----)")


class FacadeError(RuntimeError):
    def __init__(self, code: str, detail: Any = None):
        super().__init__(code)
        self.code = code
        self.detail = detail


def require(condition: bool, code: str, detail: Any = None) -> None:
    if not condition:
        raise FacadeError(code, detail)


def _object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, "DUPLICATE_JSON_KEY", key)
        result[key] = value
    return result


def _safe_regular(path: Path) -> None:
    try:
        st = path.lstat()
    except FileNotFoundError as exc:
        raise FacadeError("MISSING_CONTROL_ARTIFACT", str(path)) from exc
    require(stat.S_ISREG(st.st_mode) and not path.is_symlink(), "UNSAFE_CONTROL_ARTIFACT", str(path))


def load_json(path: Path) -> Any:
    _safe_regular(path)
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return json.load(handle, object_pairs_hook=_object_no_duplicates)
    except FacadeError:
        raise
    except Exception as exc:
        raise FacadeError("INVALID_JSON", {"path": str(path), "error": str(exc)}) from exc


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    _safe_regular(path)
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            raw_lines = list(handle)
            while raw_lines and raw_lines[-1].strip() == "":
                raw_lines.pop()
            for number, line in enumerate(raw_lines, start=1):
                require(line.strip() != "", "BLANK_JSONL_RECORD", {"path": str(path), "line": number})
                value = json.loads(line, object_pairs_hook=_object_no_duplicates)
                require(isinstance(value, dict), "JSONL_RECORD_NOT_OBJECT", number)
                rows.append(value)
    except FacadeError:
        raise
    except Exception as exc:
        raise FacadeError("INVALID_JSONL", {"path": str(path), "error": str(exc)}) from exc
    require(rows, "EMPTY_JSONL", str(path))
    return rows


def _unique(values: Iterable[Any], code: str) -> None:
    seen: set[Any] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(str(value))
        seen.add(value)
    require(not duplicates, code, sorted(duplicates))


def _safe_relative(value: Any) -> bool:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        return False
    if any(char in value for char in "*?["):
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and all(part not in {"", ".", ".."} for part in path.parts)


def _forbidden_paths(value: Any, prefix: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            here = f"{prefix}.{key}"
            if key in FORBIDDEN_INPUT_KEYS:
                found.append(here)
            found.extend(_forbidden_paths(child, here))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_forbidden_paths(child, f"{prefix}[{index}]"))
    return found


def _secret_values(value: Any, prefix: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            found.extend(_secret_values(child, f"{prefix}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_secret_values(child, f"{prefix}[{index}]"))
    elif isinstance(value, str) and SECRET_VALUE.search(value):
        found.append(prefix)
    return found


@dataclass(frozen=True)
class Bundle:
    root: Path
    registry: dict[str, Any]
    profiles: dict[str, Any]
    compatibility: dict[str, Any]
    adapters: dict[str, Any]
    code_lineage: tuple[dict[str, Any], ...]
    config_lineage: tuple[dict[str, Any], ...]
    hashes: dict[str, str]
    variants: dict[str, dict[str, Any]]
    families: dict[str, dict[str, Any]]
    adapters_by_variant: dict[str, dict[str, Any]]
    profiles_by_id: dict[str, dict[str, Any]]
    allowed_profiles: dict[str, dict[str, Any]]
    lineage_by_id: dict[str, dict[str, Any]]


def _validate_registry(registry: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    require(registry.get("schema") == "experiments7-open-stage-registry/v1", "REGISTRY_SCHEMA")
    require(registry.get("registry_schema_version") == 1, "REGISTRY_SCHEMA_VERSION")
    require(registry.get("selection_policy", {}).get("implicit_default") is False, "IMPLICIT_DEFAULT_ENABLED")
    families = registry.get("families")
    variants = registry.get("variants")
    require(isinstance(families, list) and len(families) == 16, "FAMILY_COUNT", len(families or []))
    require(isinstance(variants, list) and len(variants) == 85, "VARIANT_COUNT", len(variants or []))
    _unique((item.get("family_id") for item in families), "DUPLICATE_FAMILY_ID")
    _unique((item.get("variant_id") for item in variants), "DUPLICATE_VARIANT_ID")
    family_map = {item["family_id"]: item for item in families}
    variant_map = {item["variant_id"]: item for item in variants}
    require(set(family_map) == {f"family.{stage}" for stage in STAGES}, "FAMILY_SET")
    for stage in STAGES:
        item = family_map[f"family.{stage}"]
        require(
            item.get("kind") == "neutral_family"
            and item.get("stage") == stage
            and item.get("selector") is None
            and item.get("selectable") is False,
            "INVALID_NEUTRAL_FAMILY",
            item.get("family_id"),
        )
    for variant_id, item in variant_map.items():
        require(SAFE_ID.fullmatch(variant_id) is not None, "INVALID_VARIANT_ID", variant_id)
        stage = item.get("stage")
        require(stage in STAGES, "INVALID_VARIANT_STAGE", variant_id)
        require(
            item.get("kind") == "stage_variant"
            and item.get("selectable") is True
            and item.get("neutral_family") == f"family.{stage}"
            and item.get("semantic_parent") == f"family.{stage}"
            and item.get("selector") == f"{stage}={variant_id}",
            "INVALID_VARIANT_CONTRACT",
            variant_id,
        )
        require(isinstance(item.get("affected_outputs"), list) and item["affected_outputs"],
                "EMPTY_AFFECTED_OUTPUTS", variant_id)
        require(isinstance(item.get("lineage_ids"), list) and item["lineage_ids"],
                "EMPTY_VARIANT_LINEAGE", variant_id)
        require(isinstance(item.get("destinations"), list) and item["destinations"],
                "EMPTY_VARIANT_DESTINATIONS", variant_id)
        require(all(_safe_relative(path) for path in item["destinations"]),
                "INVALID_VARIANT_DESTINATION", variant_id)
    return family_map, variant_map


def _validate_lineage(
    code: list[dict[str, Any]], config: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    require(len(code) == 118 and len(config) == 11, "LINEAGE_COUNT", {"code": len(code), "config": len(config)})
    rows = code + config
    _unique((row.get("lineage_id") for row in rows), "DUPLICATE_LINEAGE_ID")
    _unique((row.get("planned_destination") for row in rows), "DUPLICATE_LINEAGE_DESTINATION")
    for row in rows:
        require(row.get("origin_root") in {"experiments4", "experiments5", "experiments6"},
                "INVALID_LINEAGE_ORIGIN", row.get("lineage_id"))
        require(_safe_relative(row.get("origin_relative_path")), "INVALID_LINEAGE_SOURCE_PATH", row.get("lineage_id"))
        require(_safe_relative(row.get("planned_destination")), "INVALID_LINEAGE_DESTINATION", row.get("lineage_id"))
        require(HEX64.fullmatch(str(row.get("origin_sha256"))) is not None,
                "INVALID_LINEAGE_SHA256", row.get("lineage_id"))
        require(HEX64.fullmatch(str(row.get("source_manifest_record_id"))) is not None,
                "INVALID_SOURCE_RECORD_ID", row.get("lineage_id"))
        require(row.get("evidence_id") == f"source-pre:{row['source_manifest_record_id']}",
                "INVALID_LINEAGE_EVIDENCE", row.get("lineage_id"))
    return {row["lineage_id"]: row for row in rows}


def load_bundle(root: Path) -> Bundle:
    root = root.resolve(strict=True)
    require(root.is_dir() and not root.is_symlink(), "UNSAFE_ROOT", str(root))
    paths = {
        "registry": root / "variants/registry.json",
        "profiles": root / "variants/profiles.json",
        "compatibility": root / "variants/compatibility.json",
        "adapters": root / "configs/adapters.json",
        "code_lineage": root / "lineage/code.jsonl",
        "config_lineage": root / "lineage/config.jsonl",
    }
    hashes = {name: sha256_file(path)[0] for name, path in paths.items()}
    registry = load_json(paths["registry"])
    profiles = load_json(paths["profiles"])
    compatibility = load_json(paths["compatibility"])
    adapters = load_json(paths["adapters"])
    code = load_jsonl(paths["code_lineage"])
    config = load_jsonl(paths["config_lineage"])
    forbidden = _forbidden_paths(registry) + _forbidden_paths(profiles) + _forbidden_paths(code) + _forbidden_paths(config)
    require(not forbidden, "FORBIDDEN_STATE_FIELD", forbidden)
    secrets = _secret_values(profiles) + _secret_values(compatibility) + _secret_values(adapters)
    require(not secrets, "SECRET_VALUE_IN_CONTROL_ARTIFACT", secrets)

    families, variants = _validate_registry(registry)
    lineage = _validate_lineage(code, config)
    require(profiles.get("schema") == "experiments7-g2-profiles/v1", "PROFILES_SCHEMA")
    require(profiles.get("implicit_default") is False and profiles.get("stages") == list(STAGES),
            "PROFILES_POLICY")
    require(profiles.get("registry_sha256") == hashes["registry"], "PROFILE_REGISTRY_HASH")
    require(adapters.get("schema") == "experiments7-g2-adapters/v1", "ADAPTER_SCHEMA")
    require(adapters.get("registry_sha256") == hashes["registry"], "ADAPTER_REGISTRY_HASH")
    require(adapters.get("code_lineage_sha256") == hashes["code_lineage"], "ADAPTER_CODE_HASH")
    require(adapters.get("config_lineage_sha256") == hashes["config_lineage"], "ADAPTER_CONFIG_HASH")
    adapter_rows = adapters.get("adapters")
    require(isinstance(adapter_rows, list) and len(adapter_rows) == 85, "ADAPTER_COUNT")
    _unique((item.get("variant_id") for item in adapter_rows), "DUPLICATE_ADAPTER_VARIANT")
    adapter_map = {item["variant_id"]: item for item in adapter_rows}
    require(set(adapter_map) == set(variants), "ADAPTER_VARIANT_SET")
    for variant_id, adapter in adapter_map.items():
        variant = variants[variant_id]
        require(adapter.get("adapter_id") == f"adapter.{variant_id}.v1", "ADAPTER_ID", variant_id)
        require(adapter.get("stage") == variant["stage"], "ADAPTER_STAGE", variant_id)
        require(adapter.get("neutral_family") == variant["neutral_family"], "ADAPTER_FAMILY", variant_id)
        require(adapter.get("destinations") == variant["destinations"], "ADAPTER_DESTINATIONS", variant_id)
        require(adapter.get("source_lineage_ids") == variant["lineage_ids"], "ADAPTER_LINEAGE_IDS", variant_id)
        require(adapter.get("affected_outputs") == variant["affected_outputs"], "ADAPTER_AFFECTED_OUTPUTS", variant_id)
        require(adapter.get("equivalence_state") == variant["equivalence_state"], "ADAPTER_EQUIVALENCE", variant_id)
        expected_kind = _execution_kind(variant)
        require(adapter.get("execution_kind") == expected_kind, "ADAPTER_EXECUTION_KIND", variant_id)
        for lineage_id in adapter["source_lineage_ids"]:
            require(lineage_id in lineage and lineage[lineage_id]["variant_id"] == variant_id,
                    "ADAPTER_LINEAGE_CLOSURE", {"variant_id": variant_id, "lineage_id": lineage_id})
        require(adapter.get("snapshot_state") in {"unpublished", "reference_unpublished"},
                "INVALID_SNAPSHOT_STATE", variant_id)
        require("published_origin_snapshot" in adapter.get("external_prerequisites", [])
                or adapter.get("execution_kind") not in {"subprocess", "prompt", "library"},
                "MISSING_SNAPSHOT_PREREQUISITE", variant_id)

    profile_rows = profiles.get("profiles")
    require(isinstance(profile_rows, list) and profile_rows, "EMPTY_PROFILES")
    _unique((item.get("profile_id") for item in profile_rows), "DUPLICATE_PROFILE_ID")
    profile_map = {item["profile_id"]: item for item in profile_rows}
    require(not ({"default", "canonical", "latest", "release"} & set(profile_map)), "FORBIDDEN_PROFILE_ALIAS")
    coverage: set[str] = set()
    for profile_id, profile in profile_map.items():
        require(SAFE_ID.fullmatch(profile_id) is not None, "INVALID_PROFILE_ID", profile_id)
        selected = profile.get("stages")
        na = profile.get("not_applicable")
        require(isinstance(selected, dict) and isinstance(na, list), "PROFILE_SHAPE", profile_id)
        na_stages = [item.get("stage") for item in na]
        require(len(na_stages) == len(set(na_stages)), "DUPLICATE_NOT_APPLICABLE_STAGE", profile_id)
        require(set(selected).isdisjoint(na_stages), "PROFILE_STAGE_OVERLAP", profile_id)
        require(set(selected) | set(na_stages) == set(STAGES), "PROFILE_STAGE_COVERAGE", profile_id)
        for stage, variant_id in selected.items():
            require(variant_id in variants and variants[variant_id]["stage"] == stage,
                    "PROFILE_VARIANT_STAGE", {"profile_id": profile_id, "stage": stage, "variant_id": variant_id})
        expanded = _ensure_selector_expansion(selected, variants)
        require(expanded == selected, "PROFILE_SELECTOR_NOT_EXPANDED", profile_id)
        if profile.get("kind") == "reference":
            require(set(selected.values()) <= REFERENCE_ONLY, "INVALID_REFERENCE_PROFILE", profile_id)
        coverage.update(selected.values())
    require(coverage == set(variants), "PROFILE_VARIANT_COVERAGE", {
        "missing": sorted(set(variants) - coverage), "unknown": sorted(coverage - set(variants)),
    })

    require(compatibility.get("schema") == "experiments7-g2-compatibility/v1", "COMPATIBILITY_SCHEMA")
    bindings = compatibility.get("hash_bindings", {})
    expected_bindings = {
        "registry_sha256": hashes["registry"],
        "profiles_sha256": hashes["profiles"],
        "adapters_sha256": hashes["adapters"],
        "code_lineage_sha256": hashes["code_lineage"],
        "config_lineage_sha256": hashes["config_lineage"],
    }
    require(bindings == expected_bindings, "STALE_CONTROL_HASH", {"expected": expected_bindings, "actual": bindings})
    allowed_rows = compatibility.get("allowed_profiles")
    require(isinstance(allowed_rows, list), "ALLOWED_PROFILES_TYPE")
    _unique((item.get("profile_id") for item in allowed_rows), "DUPLICATE_ALLOWED_PROFILE")
    allowed = {item["profile_id"]: item for item in allowed_rows}
    require(set(allowed) == set(profile_map), "ALLOWED_PROFILE_SET")
    for profile_id, profile in profile_map.items():
        require(allowed[profile_id].get("expanded_stages") == profile["stages"],
                "COMPATIBILITY_SELECTION_MISMATCH", profile_id)
        require(allowed[profile_id].get("not_applicable") == [item["stage"] for item in profile["not_applicable"]],
                "COMPATIBILITY_NA_MISMATCH", profile_id)

    return Bundle(
        root=root,
        registry=registry,
        profiles=profiles,
        compatibility=compatibility,
        adapters=adapters,
        code_lineage=tuple(code),
        config_lineage=tuple(config),
        hashes=hashes,
        variants=variants,
        families=families,
        adapters_by_variant=adapter_map,
        profiles_by_id=profile_map,
        allowed_profiles=allowed,
        lineage_by_id=lineage,
    )
