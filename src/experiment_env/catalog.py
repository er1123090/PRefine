from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .snapshot import validate_snapshot
from .util import clean_relative_path, load_json, sha256_bytes


class CatalogError(RuntimeError):
    pass


def catalog_path(repo_root: Path) -> Path:
    return repo_root / "configs" / "environment" / "catalog.json"


def load_catalog(repo_root: Path) -> dict[str, Any]:
    path = catalog_path(repo_root)
    value = load_json(path)
    if value.get("schema") != "experiments7-environment-catalog/v1" or not isinstance(value.get("profiles"), dict):
        raise CatalogError(f"invalid environment catalog: {path}")
    return value


def profile(repo_root: Path, profile_id: str) -> dict[str, Any]:
    catalog = load_catalog(repo_root)
    try:
        value = dict(catalog["profiles"][profile_id])
    except KeyError as exc:
        raise CatalogError(f"unknown profile: {profile_id}") from exc
    snapshot_relative = clean_relative_path(value["snapshot"])
    snapshot = repo_root.joinpath(*snapshot_relative.parts)
    value["profile_id"] = profile_id
    value["snapshot_path"] = snapshot
    return value


def recipes_for_profile(repo_root: Path, profile_value: dict[str, Any]) -> dict[str, Any]:
    recipes_path = profile_value["snapshot_path"] / ".experiment-env" / "recipes.json"
    raw = recipes_path.read_bytes()
    recipes = json.loads(raw)
    if recipes.get("schema") != "experiments7-environment-recipes/v1":
        raise CatalogError("invalid sealed recipe schema")
    if recipes.get("profile_id") != profile_value["profile_id"]:
        raise CatalogError("sealed recipes belong to a different profile")
    seal = load_json(profile_value["snapshot_path"] / ".experiment-env" / "seal.json")
    if sha256_bytes(raw) != seal.get("recipes_sha256"):
        raise CatalogError("sealed recipe hash mismatch")
    return recipes


def doctor(repo_root: Path, profile_id: str) -> dict[str, Any]:
    value = profile(repo_root, profile_id)
    result = validate_snapshot(
        repo_root,
        value["snapshot_path"],
        expected_seal_sha256=value.get("seal_sha256"),
    )
    recipes = recipes_for_profile(repo_root, value)
    result.update(
        {
            "profile_id": profile_id,
            "recipe_count": len(recipes.get("recipes", {})),
            "variant_label": value["variant_label"],
        }
    )
    return result


def all_profiles(repo_root: Path) -> list[dict[str, Any]]:
    catalog = load_catalog(repo_root)
    return [
        {
            "description": value.get("description", ""),
            "profile_id": profile_id,
            "snapshot": value["snapshot"],
            "variant_label": value["variant_label"],
        }
        for profile_id, value in sorted(catalog["profiles"].items())
    ]
