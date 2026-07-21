"""Evaluator-owned setup for an isolated 1229_dev6 paired-evaluation root.

This program copies sealed public artifacts and the private vault as opaque
bytes.  It never prints a query, reference, prediction, latent memory, or
per-case result.  It must run before any final model call and only against a
source evaluator whose manifest already declares every input hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ours_memory3.contracts import OverlayPolicy


_PUBLIC_FILES = {
    "history": Path("artifacts/history.sanitized.jsonl"),
    "tasks": Path("artifacts/tasks.jsonl"),
    "preference_slots": Path("configs/preference_slots.json"),
    "schema_single": Path("configs/schema_single.json"),
    "schema_multi": Path("configs/schema_multi.json"),
    "domain_ontology": Path("configs/public_domain_ontology.json"),
}
_GOLD = Path("evaluator_vault/gold.jsonl")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _copy_checked(source: Path, destination: Path, expected_sha256: str | None = None) -> str:
    if not source.is_file():
        raise FileNotFoundError(f"sealed evaluator input is missing: {source}")
    source_hash = _sha256(source)
    if expected_sha256 is not None and source_hash != expected_sha256:
        raise ValueError(f"sealed source hash mismatch: {source.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"destination already exists: {destination}")
    shutil.copyfile(source, destination)
    if _sha256(destination) != source_hash:
        raise ValueError(f"copy digest mismatch: {destination.name}")
    return source_hash


def prepare(root: Path, source_root: Path) -> dict[str, object]:
    root = root.resolve()
    source_root = source_root.resolve()
    if root.exists():
        raise FileExistsError("evaluation root must not already exist")
    seal_path = source_root / "evaluator_vault/sealed_manifest.json"
    source_seal = _load_json(seal_path)
    if not isinstance(source_seal, dict):
        raise ValueError("source sealed manifest must be an object")
    source_task_sha = source_seal.get("task_sha256")
    source_history_sha = source_seal.get("history_sha256")
    source_gold_sha = source_seal.get("gold_sha256")
    if not all(isinstance(value, str) and len(value) == 64 for value in (source_task_sha, source_history_sha, source_gold_sha)):
        raise ValueError("source sealed manifest lacks input digests")
    root.mkdir(parents=True)
    try:
        output_hashes: dict[str, str] = {}
        for name, relative in _PUBLIC_FILES.items():
            expected = source_history_sha if name == "history" else source_task_sha if name == "tasks" else None
            output_hashes[name] = _copy_checked(source_root / relative, root / "public" / relative.name, expected)
        gold_hash = _copy_checked(source_root / _GOLD, root / "vault/gold.jsonl", source_gold_sha)
        source_runtime = _load_json(
            source_root / "manifests/expected_runtime_contract.vlt3.json"
        )
        model = source_runtime.get("model") if isinstance(source_runtime, dict) else None
        runtime = source_runtime.get("runtime") if isinstance(source_runtime, dict) else None
        if not isinstance(model, dict) or not isinstance(runtime, dict):
            raise ValueError("source runtime contract lacks model configuration")
        required_model_fields = ("id", "snapshot_path", "served_model_name", "tensor_parallel_size", "gpu_indices")
        model = {**model, "gpu_indices": runtime.get("gpu_indices")}
        if any(field not in model for field in required_model_fields):
            raise ValueError("source runtime model configuration is incomplete")
        manifest = {
            "schema_version": 1,
            "kind": "ours_memory3_1229_dev6_sealed_paired_evaluator_v1",
            "source_evaluator_manifest_sha256": _sha256(seal_path),
            "public": output_hashes,
            "gold_sha256": gold_hash,
            "model": {field: model[field] for field in required_model_fields},
            "inference": {
                "temperature": 0.0,
                "base_seed": 2026072000,
                "max_tokens": 1024,
                "calls_per_case": 1,
                "retries": 0,
                "request_workers": 8,
            },
            "prefine": {"maximum_attempts": 10},
            "overlay_policy": asdict(OverlayPolicy()),
            "pass": {"minimum_each_mode_delta_percentage_points": 0.5},
            "source_case_count": _count_jsonl(root / "public/tasks.jsonl"),
        }
        (root / "sealed_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except BaseException:
        shutil.rmtree(root)
        raise
    return {
        "status": "PREPARED",
        "public_case_count": manifest["source_case_count"],
        "public_inputs_sha256": hashlib.sha256(json.dumps(output_hashes, sort_keys=True).encode("utf-8")).hexdigest(),
        "gold_copied_as_opaque_vault_bytes": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="prepare isolated evaluator-owned 1229_dev6 vault")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(prepare(args.root, args.source_root), sort_keys=True))
        return 0
    except BaseException as exc:
        print(json.dumps({"status": "FAILED", "error_class": type(exc).__name__}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
