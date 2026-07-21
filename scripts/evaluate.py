#!/usr/bin/env python3
"""Evaluate one canonical run without provider or network access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exp7.evaluation import EvaluationError, SUPPORTED_VARIANTS, evaluate_run


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Strictly evaluate predictions against manifest-bound prepared truth."
    )
    value.add_argument("--run-dir", type=Path, required=True)
    value.add_argument("--variant", choices=SUPPORTED_VARIANTS, required=True)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        summary = evaluate_run(run_dir=args.run_dir, variant=args.variant)
    except EvaluationError as exc:
        print(
            json.dumps(
                {"error": str(exc), "status": "error"},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "conditions": summary.condition_count,
                "output": str(summary.output_path),
                "parse_failures": summary.parse_failure_count,
                "row_errors": summary.row_error_count,
                "rows": summary.row_count,
                "status": "complete",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
