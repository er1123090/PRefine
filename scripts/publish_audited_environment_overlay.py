#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment_env.audited import publish_audited_overlay


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish the exact, no-replace experiments4/5 audited overlay")
    parser.add_argument(
        "--spec",
        type=Path,
        default=ROOT / "configs" / "environment" / "exp45-audited-overlay-v2.snapshot.json",
    )
    args = parser.parse_args()
    result = publish_audited_overlay(ROOT, args.spec)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
