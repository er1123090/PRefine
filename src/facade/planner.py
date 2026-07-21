"""Deterministic profile expansion and selection evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .metadata import STAGES
from .registry import Bundle, FacadeError, require
from .selection import canonical_bytes, sha256_bytes


def inspect_variant(bundle: Bundle, variant_id: str) -> dict[str, Any]:
    require(variant_id not in bundle.families, "FAMILY_IS_NOT_SELECTABLE", variant_id)
    variant = bundle.variants.get(variant_id)
    require(variant is not None, "UNKNOWN_VARIANT", variant_id)
    return {
        "schema": "experiments7-g2-variant-inspection/v1",
        "variant": variant,
        "adapter": bundle.adapters_by_variant[variant_id],
        "covered_by_profiles": sorted(
            profile_id
            for profile_id, profile in bundle.profiles_by_id.items()
            if variant_id in profile["stages"].values()
        ),
        "control_hashes": dict(bundle.hashes),
    }


def profile_selection(
    bundle: Bundle,
    profile_id: str,
    *,
    execution: str,
    run_id: str | None,
    output_root: Path | None,
) -> dict[str, Any]:
    require(profile_id, "PROFILE_ID_REQUIRED")
    profile = bundle.profiles_by_id.get(profile_id)
    require(profile is not None, "UNKNOWN_PROFILE", profile_id)
    allowed = bundle.allowed_profiles.get(profile_id)
    require(allowed is not None, "PROFILE_NOT_ALLOWED", profile_id)
    selected = dict(profile["stages"])
    require(selected == allowed["expanded_stages"], "PROFILE_COMPATIBILITY_DRIFT", profile_id)
    evaluation = selected.get("evaluation")
    expansion = bundle.variants[evaluation].get("expands_to") if evaluation else None
    adapter_rows = []
    dataset_config = []
    prerequisites: set[str] = set()
    reference_only_selected = False
    for stage in STAGES:
        variant_id = selected.get(stage)
        if variant_id is None:
            continue
        adapter = bundle.adapters_by_variant[variant_id]
        if adapter["execution_kind"] == "reference_only":
            reference_only_selected = True
        prerequisites.update(adapter["external_prerequisites"])
        sources = [
            {
                "lineage_id": binding["lineage_id"],
                "source_manifest_record_id": binding["source_manifest_record_id"],
                "origin_root": binding["origin_root"],
                "origin_relative_path": binding["origin_relative_path"],
                "origin_sha256": binding["origin_sha256"],
                "planned_destination": binding["planned_destination"],
            }
            for binding in adapter["source_bindings"]
        ]
        row = {
            "stage": stage,
            "variant_id": variant_id,
            "adapter_id": adapter["adapter_id"],
            "label": adapter["label"],
            "execution_kind": adapter["execution_kind"],
            "origin_label": adapter["origin_label"],
            "entrypoint_destination": adapter["entrypoint_destination"],
            "argv_template": adapter["argv_template"],
            "source_bindings": sources,
            "final_snapshot_bindings": adapter["final_snapshot_bindings"],
            "snapshot_state": adapter["snapshot_state"],
            "external_prerequisites": adapter["external_prerequisites"],
            "environment_keys": adapter["environment_keys"],
            "expected_outputs": adapter["expected_outputs"],
        }
        adapter_rows.append(row)
        if adapter["execution_kind"] == "data":
            dataset_config.extend(sources)
    selection = {
        "schema": "experiments7-g2-selection/v1",
        "run_id": run_id,
        "profile_id": profile_id,
        "profile_label": profile["label"],
        "execution_mode": execution,
        "selected_stages": {stage: selected[stage] for stage in STAGES if stage in selected},
        "not_applicable": list(profile["not_applicable"]),
        "evaluation_selector": evaluation,
        "evaluation_expansion": expansion or {},
        "adapters": adapter_rows,
        "dataset_config_artifacts": dataset_config,
        "control_hashes": {
            "registry_sha256": bundle.hashes["registry"],
            "profiles_sha256": bundle.hashes["profiles"],
            "compatibility_sha256": bundle.hashes["compatibility"],
            "adapters_sha256": bundle.hashes["adapters"],
            "code_lineage_sha256": bundle.hashes["code_lineage"],
            "config_lineage_sha256": bundle.hashes["config_lineage"],
        },
        "external_prerequisites": sorted(prerequisites),
        "external_runnable": not prerequisites and not reference_only_selected,
        "output_root": str(output_root) if output_root is not None else None,
        "secret_values_emitted": False,
    }
    selection["selection_sha256"] = sha256_bytes(canonical_bytes(selection))
    return selection
