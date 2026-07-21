#!/usr/bin/env python3
"""Build conflict-majority easy/medium/hard task files for experiment 5."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.conflict_majority import (  # noqa: E402
    DIFFICULTIES,
    TURN_TYPES,
    build_conflict_majority_tasks,
    build_task_bundle,
    load_json,
    write_task_bundle,
)
from src.data_utils import load_multiturn_data, load_query_map  # noqa: E402


def _load_examples(path: str) -> List[Dict[str, Any]]:
    raw = load_json(path)
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and isinstance(raw.get("dataset"), list):
        return raw["dataset"]
    raise ValueError(f"Unsupported input dataset format: {path}")


def _write_split_files(
    out_dir: Path,
    turn_type: str,
    tasks: List[Dict[str, Any]],
    summary: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Dict[str, str]:
    turn_dir = out_dir / turn_type
    written: Dict[str, str] = {}

    all_path = turn_dir / "all.json"
    write_task_bundle(str(all_path), build_task_bundle(tasks, summary, metadata))
    written["all"] = str(all_path)

    for difficulty in DIFFICULTIES:
        split_tasks = [task for task in tasks if task.get("difficulty") == difficulty]
        split_summary = dict(summary)
        split_summary["total_tasks"] = len(split_tasks)
        split_summary["tasks_by_difficulty"] = {
            key: (len(split_tasks) if key == difficulty else 0)
            for key in DIFFICULTIES
        }
        split_path = turn_dir / f"{difficulty}.json"
        write_task_bundle(str(split_path), build_task_bundle(split_tasks, split_summary, metadata))
        written[difficulty] = str(split_path)

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build conflict-majority task bundles for exp5 and exp4-wrapper inference."
    )
    parser.add_argument(
        "--input_path",
        default=str(ROOT / "data" / "e_dev_conflict_ordered_ratio.json"),
        help="Conflict dataset JSON file.",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Directory to write singleturn/multiturn task bundles.",
    )
    parser.add_argument(
        "--pref_group_path",
        default=str(ROOT / "config" / "pref_group.json"),
        help="Preference value-group rules.",
    )
    parser.add_argument(
        "--singleturn_query_path",
        default=str(ROOT / "config" / "query_singleturn.json"),
        help="Single-turn query map.",
    )
    parser.add_argument(
        "--multiturn_query_path",
        default=str(ROOT / "config" / "query_multiturn-domain.json"),
        help="Multi-turn query templates.",
    )
    parser.add_argument(
        "--turn_type",
        choices=["all", *TURN_TYPES],
        default="all",
        help="Which turn type to build.",
    )
    parser.add_argument(
        "--include_ties",
        action="store_true",
        help="Include all max-count value groups when the majority count ties.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)

    examples = _load_examples(args.input_path)
    pref_group_data = load_json(args.pref_group_path)

    requested_turn_types = list(TURN_TYPES) if args.turn_type == "all" else [args.turn_type]
    top_summary: Dict[str, Any] = {
        "input_path": args.input_path,
        "pref_group_path": args.pref_group_path,
        "include_ties": args.include_ties,
        "turn_types": {},
        "written_files": {},
    }

    for turn_type in requested_turn_types:
        if turn_type == "singleturn":
            query_catalog = load_query_map(args.singleturn_query_path)
            query_path = args.singleturn_query_path
        else:
            query_catalog = load_multiturn_data(args.multiturn_query_path)
            query_path = args.multiturn_query_path

        tasks, summary = build_conflict_majority_tasks(
            examples=examples,
            pref_group_data=pref_group_data,
            query_catalog=query_catalog,
            turn_type=turn_type,
            include_ties=args.include_ties,
        )
        metadata = {
            "input_path": args.input_path,
            "pref_group_path": args.pref_group_path,
            "query_path": query_path,
            "turn_type": turn_type,
            "include_ties": args.include_ties,
            "builder": os.path.relpath(__file__, ROOT),
        }
        top_summary["turn_types"][turn_type] = summary
        top_summary["written_files"][turn_type] = _write_split_files(
            out_dir=out_dir,
            turn_type=turn_type,
            tasks=tasks,
            summary=summary,
            metadata=metadata,
        )

    summary_path = out_dir / "summary.json"
    os.makedirs(summary_path.parent, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(top_summary, f, indent=2, ensure_ascii=False)

    print(f"Wrote conflict-majority tasks -> {out_dir}")
    print(json.dumps(top_summary["turn_types"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
