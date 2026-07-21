#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


BASE_DIR = Path("/data/minseo/experiments6")
RERUN_ROOT = BASE_DIR / "ours_memory" / "inference" / "extended" / "0327_rerun"

COMMON_PATHS = {
    "input_path": BASE_DIR / "data" / "1229_dev_6.json",
    "pref_list_path": BASE_DIR / "pref_list_extended.json",
    "pref_group_path": BASE_DIR / "pref_group_extended.json",
}

TURN_CONFIGS = {
    "singleturn": {
        "ours_script": BASE_DIR / "ours_memory" / "Preference_Memory_step2_ACTION_singleturn_api.py",
        "query_arg": "--query_path",
        "query_path": BASE_DIR / "query_new_single.json",
        "tools_schema_path": BASE_DIR / "schema_easy_extended.json",
        "vanilla_root": BASE_DIR / "vanillaLLM" / "inference" / "extended" / "outputs" / "singleturn" / "api",
        "ours_output_root": BASE_DIR / "ours_memory" / "inference" / "extended" / "outputs" / "singleturn" / "api",
    },
    "multiturn": {
        "ours_script": BASE_DIR / "ours_memory" / "Preference_Memory_step2_ACTION_multiturn_api.py",
        "query_arg": "--multiturn_path",
        "query_path": BASE_DIR / "query_new_multi.json",
        "tools_schema_path": BASE_DIR / "schema_all_extended_complete.json",
        "vanilla_root": BASE_DIR / "vanillaLLM" / "inference" / "extended" / "outputs" / "multiturn" / "api",
        "ours_output_root": BASE_DIR / "ours_memory" / "inference" / "extended" / "outputs" / "multiturn" / "api",
    },
}

MAIN_JSON_EXCLUDE_TOKENS = ("_query_",)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def normalize_model_dir_name(value: str) -> str:
    return re.sub(r"([_-])(high|minimal|medium|low|default)$", "", value)


def iter_main_json_files(root: Path) -> List[Path]:
    if not root.exists():
        return []
    result = []
    for path in root.rglob("*.json"):
        if any(token in path.name for token in MAIN_JSON_EXCLUDE_TOKENS):
            continue
        result.append(path)
    return result


def choose_latest(paths: Sequence[Path]) -> Path:
    if not paths:
        raise ValueError("No candidate paths were provided.")
    return sorted(paths, key=lambda path: (path.stat().st_mtime, str(path)), reverse=True)[0]


def discover_canonical_json(turn_type: str, action_model: str) -> Path:
    config = TURN_CONFIGS[turn_type]
    candidates = []
    for path in iter_main_json_files(config["vanilla_root"]):
        model_dir = path.parent.parent.name
        if normalize_model_dir_name(model_dir) == action_model:
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            f"Could not find a vanilla canonical JSON for turn_type={turn_type} action_model={action_model}"
        )
    return choose_latest(candidates)


def discover_existing_json(
    turn_type: str,
    memory_model: str,
    context_type: str,
    pref_type: str,
    action_model: str,
) -> Optional[Path]:
    config = TURN_CONFIGS[turn_type]
    search_root = config["ours_output_root"] / memory_model / context_type / pref_type
    candidates = []
    for path in iter_main_json_files(search_root):
        model_dir = path.parent.parent.name
        if normalize_model_dir_name(model_dir) == action_model:
            candidates.append(path)

    if not candidates:
        return None
    return choose_latest(candidates)


def load_rows(path: Optional[Path]) -> List[Dict[str, Any]]:
    if path is None:
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list JSON at {path}, found {type(data).__name__}")
    return data


def extract_example_id_subs(rows: Sequence[Dict[str, Any]], source_path: Optional[Path]) -> List[str]:
    ids = []
    for index, row in enumerate(rows):
        example_id_sub = row.get("example_id_sub")
        if not example_id_sub:
            raise ValueError(
                f"Row {index} from {source_path or '<in-memory>'} is missing example_id_sub."
            )
        ids.append(str(example_id_sub))
    return ids


def unique_preserve_order(items: Sequence[str]) -> List[str]:
    seen = set()
    ordered = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def build_row_map(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    row_map: Dict[str, Dict[str, Any]] = {}
    duplicate_ids = set()
    for row in rows:
        example_id_sub = str(row.get("example_id_sub"))
        if example_id_sub in row_map:
            duplicate_ids.add(example_id_sub)
        row_map[example_id_sub] = row
    if duplicate_ids:
        preview = ", ".join(sorted(duplicate_ids)[:10])
        raise ValueError(f"Duplicate example_id_sub values found while building row map: {preview}")
    return row_map


def build_audit(canonical_ids: Sequence[str], existing_ids: Sequence[str]) -> Dict[str, Any]:
    canonical_unique = unique_preserve_order(canonical_ids)
    existing_unique = unique_preserve_order(existing_ids)

    canonical_set = set(canonical_unique)
    existing_set = set(existing_unique)

    missing_ids = [item for item in canonical_unique if item not in existing_set]
    extra_ids = [item for item in existing_unique if item not in canonical_set]
    shared_ids = [item for item in canonical_unique if item in existing_set]

    canonical_shared_order = [item for item in canonical_unique if item in existing_set]
    existing_shared_order = [item for item in existing_unique if item in canonical_set]
    order_matches_for_shared_ids = canonical_shared_order == existing_shared_order

    return {
        "canonical_count": len(canonical_ids),
        "canonical_unique_count": len(canonical_unique),
        "existing_count": len(existing_ids),
        "existing_unique_count": len(existing_unique),
        "shared_count": len(shared_ids),
        "missing_count": len(missing_ids),
        "extra_count": len(extra_ids),
        "set_match": canonical_set == existing_set,
        "order_match_for_shared_ids": order_matches_for_shared_ids,
        "missing_ids": missing_ids,
        "extra_ids": extra_ids,
        "canonical_first10": canonical_unique[:10],
        "existing_first10": existing_unique[:10],
    }


def merge_rows(
    canonical_ids: Sequence[str],
    existing_rows: Sequence[Dict[str, Any]],
    rerun_rows: Sequence[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[str]]:
    merged_map = build_row_map(existing_rows)
    rerun_map = build_row_map(rerun_rows)
    merged_map.update(rerun_map)

    merged_rows = []
    unresolved_ids = []
    for example_id_sub in canonical_ids:
        row = merged_map.get(example_id_sub)
        if row is None:
            unresolved_ids.append(example_id_sub)
            continue
        merged_rows.append(row)
    return merged_rows, unresolved_ids


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def run_inference(
    args: argparse.Namespace,
    config: Dict[str, Any],
    missing_ids_path: Path,
    rerun_output_path: Path,
    rerun_log_path: Path,
) -> None:
    memory_path = (
        BASE_DIR / "ours_memory" / "inference" / "1231_MEMORY3" / args.memory_model / "_memory1.jsonl"
    )
    if not memory_path.exists():
        raise FileNotFoundError(f"Memory file not found: {memory_path}")

    cmd = [
        sys.executable,
        str(config["ours_script"]),
        "--input_path",
        str(COMMON_PATHS["input_path"]),
        "--memory_path",
        str(memory_path),
        "--output_path",
        str(rerun_output_path),
        "--log_path",
        str(rerun_log_path),
        config["query_arg"],
        str(config["query_path"]),
        "--pref_list_path",
        str(COMMON_PATHS["pref_list_path"]),
        "--pref_group_path",
        str(COMMON_PATHS["pref_group_path"]),
        "--tools_schema_path",
        str(config["tools_schema_path"]),
        "--context_type",
        args.context_type,
        "--pref_type",
        args.pref_type,
        "--model_name",
        args.action_model,
        "--concurrency",
        str(args.concurrency),
        "--reasoning_effort",
        args.reasoning_effort,
        "--example_id_sub_filter_path",
        str(missing_ids_path),
    ]

    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turn-type", choices=sorted(TURN_CONFIGS.keys()), required=True)
    parser.add_argument("--memory-model", required=True)
    parser.add_argument("--action-model", required=True)
    parser.add_argument("--context-type", default="memory_api")
    parser.add_argument("--pref-type", default="hard")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--canonical-json", default=None)
    parser.add_argument("--existing-json", default=None)
    parser.add_argument("--rerun-root", default=str(RERUN_ROOT))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = TURN_CONFIGS[args.turn_type]

    rerun_root = Path(args.rerun_root)
    job_dir = (
        rerun_root
        / args.turn_type
        / safe_name(args.memory_model)
        / safe_name(args.context_type)
        / safe_name(args.pref_type)
        / safe_name(args.action_model)
    )
    manifest_dir = job_dir / "manifest"
    results_dir = job_dir / "results"
    logs_dir = job_dir / "logs"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    canonical_json = Path(args.canonical_json) if args.canonical_json else discover_canonical_json(
        args.turn_type, args.action_model
    )
    existing_json = Path(args.existing_json) if args.existing_json else discover_existing_json(
        turn_type=args.turn_type,
        memory_model=args.memory_model,
        context_type=args.context_type,
        pref_type=args.pref_type,
        action_model=args.action_model,
    )

    canonical_rows = load_rows(canonical_json)
    existing_rows = load_rows(existing_json)
    canonical_ids = extract_example_id_subs(canonical_rows, canonical_json)
    existing_ids = extract_example_id_subs(existing_rows, existing_json)

    audit_before = build_audit(canonical_ids, existing_ids)
    missing_ids = audit_before["missing_ids"]

    metadata = {
        "turn_type": args.turn_type,
        "memory_model": args.memory_model,
        "action_model": args.action_model,
        "context_type": args.context_type,
        "pref_type": args.pref_type,
        "reasoning_effort": args.reasoning_effort,
        "canonical_json": str(canonical_json),
        "existing_json": str(existing_json) if existing_json else None,
        "dry_run": args.dry_run,
    }
    write_json(manifest_dir / "job_meta.json", metadata)
    write_json(manifest_dir / "audit_before.json", audit_before)
    write_json(manifest_dir / "canonical_ids.json", canonical_ids)
    write_json(manifest_dir / "missing_ids.json", missing_ids)

    rerun_output_path = results_dir / "rerun_missing.json"
    rerun_log_path = logs_dir / "rerun_missing.log"
    rerun_rows: List[Dict[str, Any]] = []

    if missing_ids:
        print(
            f"[INFO] {args.turn_type} memory={args.memory_model} action={args.action_model} "
            f"missing={len(missing_ids)}"
        )
        if not args.dry_run:
            run_inference(
                args=args,
                config=config,
                missing_ids_path=manifest_dir / "missing_ids.json",
                rerun_output_path=rerun_output_path,
                rerun_log_path=rerun_log_path,
            )
            rerun_rows = load_rows(rerun_output_path)
        else:
            print("[INFO] Dry-run mode enabled; skipping inference.")
    else:
        print(
            f"[INFO] {args.turn_type} memory={args.memory_model} action={args.action_model} "
            "has no missing example_id_sub values."
        )

    merged_rows, unresolved_ids = merge_rows(canonical_ids, existing_rows, rerun_rows)
    write_json(results_dir / "merged_output.json", merged_rows)

    audit_after = build_audit(canonical_ids, extract_example_id_subs(merged_rows, results_dir / "merged_output.json"))
    audit_after["unresolved_ids_after_merge"] = unresolved_ids
    audit_after["rerun_output_json"] = str(rerun_output_path) if rerun_rows else None
    write_json(manifest_dir / "audit_after.json", audit_after)

    if unresolved_ids:
        write_json(manifest_dir / "unresolved_ids_after_merge.json", unresolved_ids)
        if args.dry_run:
            print(
                f"[INFO] Dry-run left {len(unresolved_ids)} unresolved example_id_sub values, "
                "which is expected because inference was skipped."
            )
            return 0
        print(
            f"[ERROR] Merge finished with {len(unresolved_ids)} unresolved example_id_sub values. "
            f"See {manifest_dir / 'unresolved_ids_after_merge.json'}"
        )
        return 1

    print(f"[DONE] merged_output={results_dir / 'merged_output.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
