#!/usr/bin/env python3
"""Create a fixed PEToolBench sample manifest with per-history counts."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_petool_memory_gvr import normalize_rows  # noqa: E402


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset_test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--p", type=int, default=100)
    parser.add_argument("--r", type=int, default=100)
    parser.add_argument("--c", type=int, default=100)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows: List[Dict[str, Any]] = []
    counts = {"p": args.p, "r": args.r, "c": args.c}
    for history_type, count in counts.items():
        records = normalize_rows(args.dataset_dir, history_type, None)
        if count > len(records):
            raise SystemExit(f"requested {count} {history_type} rows, only {len(records)} available")
        selected_indices = sorted(rng.sample(range(len(records)), count))
        for rank, source_index in enumerate(selected_indices):
            record = records[source_index]
            rows.append(
                {
                    "history_type": history_type,
                    "source_index": int(record["source_index"]),
                    "example_id": record["example_id"],
                    "rank_within_history": rank,
                }
            )

    manifest = {
        "seed": args.seed,
        "history_counts": counts,
        "total": len(rows),
        "rows": rows,
    }
    write_json(args.output, manifest)
    print(json.dumps({"output": str(args.output), "total": len(rows), "history_counts": counts}, indent=2))


if __name__ == "__main__":
    main()
