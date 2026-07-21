"""Normalize PEToolBench test files into JSONL records."""

from __future__ import annotations

import argparse

from common import load_petoolbench_rows, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare PEToolBench JSON into normalized JSONL.")
    parser.add_argument("--input", required=True, help="PEToolBench user_entries_test_*.json file.")
    parser.add_argument("--history_type", required=True, choices=["p", "r", "c"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = load_petoolbench_rows(args.input, args.history_type, args.limit)
    write_jsonl(args.output, rows)
    print(f"prepared={len(rows)} output={args.output}")


if __name__ == "__main__":
    main()

