#!/usr/bin/env python3
"""Materialize deterministic prepared datasets without running inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exp7.datasets.pipeline import DatasetPreparationError, prepare_dataset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare all six mix600-v1 turn/difficulty datasets."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/datasets/mix600-v1.json"),
        help="Dataset config path, relative to the experiments7 root by default.",
    )
    parser.add_argument("--mix600", type=Path, help="Override the external mix600 source path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override the safe default artifacts/prepared/mix600-v1 directory.",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Replace only an existing directory whose manifest identifies the same dataset.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = prepare_dataset(
            repo_root=ROOT,
            config_path=args.config,
            mix600_path=args.mix600,
            output_dir=args.output_dir,
            replace_existing=args.replace_existing,
        )
    except DatasetPreparationError as exc:
        print(f"prepare failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest["summary"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
