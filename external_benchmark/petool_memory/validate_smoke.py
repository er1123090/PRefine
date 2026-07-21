"""Validate the petool_memory path on deterministic PEToolBench smoke cases."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
EXTERNAL_BENCHMARK_DIR = THIS_DIR.parent
PETOOLBENCH_DIR = EXTERNAL_BENCHMARK_DIR / "PEToolBench"
PREPARE_SCRIPT = EXTERNAL_BENCHMARK_DIR / "petoolbench_ours_memory" / "prepare_petoolbench.py"
EVALUATE_SCRIPT = EXTERNAL_BENCHMARK_DIR / "petoolbench_ours_memory" / "evaluate.py"


def run(cmd):
    print("+ " + " ".join(str(part) for part in cmd))
    subprocess.run([str(part) for part in cmd], check=True)


def line_count(path: Path) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate PEToolBench petool_memory smoke path.")
    parser.add_argument("--work-dir", required=True)
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    summaries = {}
    for history_type in ["p", "r", "c"]:
        input_path = PETOOLBENCH_DIR / "dataset_test" / f"user_entries_test_{history_type}.json"
        normalized_path = work_dir / f"normalized_{history_type}.jsonl"
        memory_path = work_dir / f"petool_memory_{history_type}.jsonl"
        predictions_path = work_dir / f"predictions_{history_type}.json"
        summary_path = work_dir / f"summary_{history_type}.json"

        run(
            [
                sys.executable,
                PREPARE_SCRIPT,
                "--input",
                input_path,
                "--history_type",
                history_type,
                "--limit",
                "1",
                "--output",
                normalized_path,
            ]
        )
        assert line_count(normalized_path) == 1, normalized_path

        run(
            [
                sys.executable,
                THIS_DIR / "build_memory.py",
                "--input",
                normalized_path,
                "--output",
                memory_path,
                "--mode",
                "deterministic",
            ]
        )
        assert line_count(memory_path) == 1, memory_path

        run(
            [
                sys.executable,
                THIS_DIR / "infer.py",
                "--input",
                normalized_path,
                "--memory",
                memory_path,
                "--output",
                predictions_path,
                "--mock",
            ]
        )

        run(
            [
                sys.executable,
                EVALUATE_SCRIPT,
                "--predictions",
                predictions_path,
                "--output_summary",
                summary_path,
            ]
        )
        summary = json.load(open(summary_path, "r", encoding="utf-8"))
        assert summary["n"] == 1, summary
        assert summary["tool_accuracy"] == 1.0, summary
        assert summary["parameter_accuracy"] == 1.0, summary
        summaries[history_type] = summary

    print(
        json.dumps(
            {
                "passed": True,
                "work_dir": str(work_dir),
                "history_types": sorted(summaries),
                "summaries": summaries,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

