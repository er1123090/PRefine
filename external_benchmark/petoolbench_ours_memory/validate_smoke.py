"""Run a deterministic PEToolBench ours_memory adapter smoke test."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
PETOOLBENCH_DIR = REPO_ROOT / "external_benchmark" / "PEToolBench"


def run(cmd):
    print("+ " + " ".join(str(part) for part in cmd))
    subprocess.run([str(part) for part in cmd], check=True)


def line_count(path: Path) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate PEToolBench adapter smoke path.")
    parser.add_argument("--work-dir", required=True)
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    normalized_paths = {}
    for history_type in ["p", "r", "c"]:
        input_path = PETOOLBENCH_DIR / "dataset_test" / f"user_entries_test_{history_type}.json"
        output_path = work_dir / f"normalized_{history_type}.jsonl"
        run(
            [
                sys.executable,
                THIS_DIR / "prepare_petoolbench.py",
                "--input",
                input_path,
                "--history_type",
                history_type,
                "--limit",
                "2",
                "--output",
                output_path,
            ]
        )
        assert line_count(output_path) == 2, f"expected 2 rows in {output_path}"
        normalized_paths[history_type] = output_path

    smoke_input = normalized_paths["p"]
    memory_path = work_dir / "memory.jsonl"
    predictions_path = work_dir / "predictions.json"
    summary_path = work_dir / "summary.json"

    run(
        [
            sys.executable,
            THIS_DIR / "build_memory.py",
            "--input",
            smoke_input,
            "--output",
            memory_path,
            "--mock",
        ]
    )
    assert line_count(memory_path) == 2, "expected 2 memory records"

    run(
        [
            sys.executable,
            THIS_DIR / "infer.py",
            "--input",
            smoke_input,
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
            THIS_DIR / "evaluate.py",
            "--predictions",
            predictions_path,
            "--output_summary",
            summary_path,
        ]
    )

    summary = json.load(open(summary_path, "r", encoding="utf-8"))
    assert summary["n"] == 2, summary
    assert summary["tool_accuracy"] == 1.0, summary
    assert summary["parameter_accuracy"] == 1.0, summary

    result = {
        "passed": True,
        "work_dir": str(work_dir),
        "normalized_counts": {
            history_type: line_count(path) for history_type, path in normalized_paths.items()
        },
        "memory_records": line_count(memory_path),
        "summary": summary,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
