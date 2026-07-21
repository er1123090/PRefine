#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

from session_memory_eval_common import (
    DEFAULT_DATASET_PATH,
    DEFAULT_MEMORY_ROOT,
    DEFAULT_PREP_ROOT,
    filter_api_calls_by_session,
    iter_jsonl,
    load_json,
    stringify_implicit_pref,
    write_csv,
    write_json,
    write_jsonl,
)


VALID_COHORTS = {"all", "false_anytime"}


def load_dataset_bundle(path: str | Path) -> Tuple[Any, List[Dict[str, Any]], str | None]:
    raw = load_json(path)
    if isinstance(raw, list):
        return raw, raw, None
    if isinstance(raw, dict):
        for key in ("dataset", "data", "examples", "items"):
            if key in raw and isinstance(raw[key], list):
                return raw, raw[key], key
    raise ValueError(f"Unsupported dataset structure in {path}")


def build_subset_payload(template: Any, items: List[Dict[str, Any]], list_key: str | None) -> Any:
    if list_key is None:
        return items
    payload = dict(template)
    payload[list_key] = items
    return payload


def load_false_anytime_ids(verifier_path: Path) -> set[str]:
    false_ids: set[str] = set()
    for row in iter_jsonl(verifier_path):
        if row.get("is_valid") is False:
            false_ids.add(str(row.get("example_id")))
    return false_ids


def load_session_snapshots(memory_path: Path) -> Tuple[Dict[str, Dict[int, Dict[str, Any]]], int]:
    snapshots_by_example: Dict[str, Dict[int, Dict[str, Any]]] = {}
    max_session = 0

    for record in iter_jsonl(memory_path):
        example_id = str(record.get("example_id"))
        api_calls = record.get("final_accumulated_api_calls", [])
        history = record.get("preference_evolution_history", [])
        session_map: Dict[int, Dict[str, Any]] = {}

        for session_record in history:
            session_index = int(session_record.get("session_index", 0) or 0)
            if session_index <= 0:
                continue

            final_pref = session_record.get("final_preference_at_session", {})
            session_map[session_index] = {
                "example_id": example_id,
                "session_index": session_index,
                "final_implicit_preference": stringify_implicit_pref(final_pref),
                "final_accumulated_api_calls": filter_api_calls_by_session(api_calls, session_index),
            }
            max_session = max(max_session, session_index)

        snapshots_by_example[example_id] = session_map

    return snapshots_by_example, max_session


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory_root", type=str, default=str(DEFAULT_MEMORY_ROOT))
    parser.add_argument("--dataset_path", type=str, default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--output_root", type=str, default=str(DEFAULT_PREP_ROOT))
    parser.add_argument("--cohorts", nargs="+", default=["all", "false_anytime"])
    parser.add_argument("--max_session", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    invalid = [cohort for cohort in args.cohorts if cohort not in VALID_COHORTS]
    if invalid:
        raise ValueError(f"Unsupported cohort(s): {invalid}")

    dataset_template, dataset_rows, list_key = load_dataset_bundle(args.dataset_path)
    output_root = Path(args.output_root)
    memory_root = Path(args.memory_root)

    manifest_rows: List[Dict[str, Any]] = []
    stats_rows: List[Dict[str, Any]] = []

    for model_dir in sorted(path for path in memory_root.iterdir() if path.is_dir()):
        verifier_path = model_dir / "_verifier_logs1.jsonl"
        memory_path = model_dir / "_memory1.jsonl"
        if not verifier_path.exists() or not memory_path.exists():
            continue

        false_anytime_ids = load_false_anytime_ids(verifier_path)
        snapshots_by_example, model_max_session = load_session_snapshots(memory_path)
        if args.max_session is not None:
            model_max_session = min(model_max_session, args.max_session)

        available_examples = sum(1 for sessions in snapshots_by_example.values() if sessions)
        print(
            f"[MODEL] {model_dir.name} "
            f"available_examples={available_examples} "
            f"false_anytime={len(false_anytime_ids)} "
            f"max_session={model_max_session}"
        )

        for cohort in args.cohorts:
            cohort_ids = set(snapshots_by_example.keys())
            if cohort == "false_anytime":
                cohort_ids &= false_anytime_ids

            for session_index in range(1, model_max_session + 1):
                subset_rows: List[Dict[str, Any]] = []
                memory_rows: List[Dict[str, Any]] = []

                for row in dataset_rows:
                    example_id = str(row.get("example_id"))
                    if example_id not in cohort_ids:
                        continue

                    snapshot = snapshots_by_example.get(example_id, {}).get(session_index)
                    if not snapshot:
                        continue

                    subset_rows.append(row)
                    memory_rows.append(snapshot)

                if not subset_rows:
                    continue

                target_dir = output_root / model_dir.name / cohort / f"session_{session_index}"
                dataset_out = target_dir / "dataset.json"
                memory_out = target_dir / "memory.jsonl"

                if not args.dry_run:
                    write_json(dataset_out, build_subset_payload(dataset_template, subset_rows, list_key))
                    write_jsonl(memory_out, memory_rows)

                manifest_row = {
                    "memory_model": model_dir.name,
                    "cohort": cohort,
                    "session_index": session_index,
                    "eligible_count": len(subset_rows),
                    "dataset_path": str(dataset_out),
                    "memory_path": str(memory_out),
                    "source_memory_path": str(memory_path),
                    "source_verifier_path": str(verifier_path),
                    "false_anytime_count": len(false_anytime_ids),
                    "available_examples": available_examples,
                }
                manifest_rows.append(manifest_row)

                stats_rows.append(
                    {
                        "memory_model": model_dir.name,
                        "cohort": cohort,
                        "session_index": session_index,
                        "eligible_count": len(subset_rows),
                        "false_anytime_count": len(false_anytime_ids),
                        "available_examples": available_examples,
                        "max_session": model_max_session,
                    }
                )
                print(
                    f"  - cohort={cohort} session_index={session_index} eligible_count={len(subset_rows)}"
                )

    if args.dry_run:
        print("[DRY RUN] No files written.")
        return

    write_jsonl(output_root / "prep_manifest.jsonl", manifest_rows)
    write_csv(
        output_root / "prep_manifest.csv",
        [
            "memory_model",
            "cohort",
            "session_index",
            "eligible_count",
            "dataset_path",
            "memory_path",
            "source_memory_path",
            "source_verifier_path",
            "false_anytime_count",
            "available_examples",
        ],
        manifest_rows,
    )
    write_csv(
        output_root / "prep_stats.csv",
        [
            "memory_model",
            "cohort",
            "session_index",
            "eligible_count",
            "false_anytime_count",
            "available_examples",
            "max_session",
        ],
        stats_rows,
    )
    print(f"[DONE] Wrote prep artifacts to {output_root}")


if __name__ == "__main__":
    main()
