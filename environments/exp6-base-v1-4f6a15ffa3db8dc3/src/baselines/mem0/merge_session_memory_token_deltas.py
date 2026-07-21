import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge shard CSV/summary files for Mem0 session token delta measurements."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/data/minseo/experiments6/mem0",
    )
    parser.add_argument(
        "--csv_glob",
        type=str,
        default="session_memory_token_deltas_1229_dev_6.shard*.csv",
    )
    parser.add_argument(
        "--summary_glob",
        type=str,
        default="session_memory_token_deltas_1229_dev_6.shard*.summary.json",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="/data/minseo/experiments6/mem0/session_memory_token_deltas_1229_dev_6.merged.csv",
    )
    parser.add_argument(
        "--output_summary",
        type=str,
        default="/data/minseo/experiments6/mem0/session_memory_token_deltas_1229_dev_6.merged.summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    csv_paths = sorted(input_dir.glob(args.csv_glob))
    summary_paths = sorted(input_dir.glob(args.summary_glob))

    if not csv_paths:
        raise FileNotFoundError(f"No shard CSV files matched: {args.csv_glob}")

    merged_rows: List[Dict[str, str]] = []
    fieldnames = None

    for path in csv_paths:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            for row in reader:
                merged_rows.append(row)

    if fieldnames is None:
        raise RuntimeError("No CSV fieldnames found.")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)

    deltas = [
        int(row["delta_memory_tokens"])
        for row in merged_rows
        if row.get("status") == "OK" and row.get("delta_memory_tokens") not in ("", None)
    ]
    error_rows = [row for row in merged_rows if row.get("status") == "ERROR"]

    shard_summaries = []
    for path in summary_paths:
        shard_summaries.append(json.loads(path.read_text(encoding="utf-8")))

    merged_summary = {
        "merged_csv": str(output_csv),
        "merged_from_csv_files": [str(path) for path in csv_paths],
        "merged_from_summary_files": [str(path) for path in summary_paths],
        "rows_total": len(merged_rows),
        "rows_ok": sum(1 for row in merged_rows if row.get("status") == "OK"),
        "rows_error": len(error_rows),
        "rows_other": sum(1 for row in merged_rows if row.get("status") not in {"OK", "ERROR"}),
        "delta_token_sum": sum(deltas) if deltas else 0,
        "delta_token_abs_sum": sum(abs(x) for x in deltas) if deltas else 0,
        "avg_delta_tokens": (sum(deltas) / len(deltas)) if deltas else None,
        "avg_abs_delta_tokens": (sum(abs(x) for x in deltas) / len(deltas)) if deltas else None,
        "min_delta_tokens": min(deltas) if deltas else None,
        "max_delta_tokens": max(deltas) if deltas else None,
        "nonzero_delta_sessions": sum(1 for x in deltas if x != 0),
        "negative_delta_sessions": sum(1 for x in deltas if x < 0),
        "shard_summaries": shard_summaries,
    }

    output_summary = Path(args.output_summary)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(merged_summary, indent=2), encoding="utf-8")

    print(json.dumps(merged_summary, indent=2))


if __name__ == "__main__":
    main()
