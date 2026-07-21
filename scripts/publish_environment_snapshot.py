#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment_env.snapshot import publish_snapshot  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish an exact, no-replace experiments7 environment snapshot")
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument(
        "--spec",
        type=Path,
        default=ROOT / "configs" / "environment" / "exp6-base-v1.snapshot.json",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = publish_snapshot(args.repo_root, args.spec)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(result["destination"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
