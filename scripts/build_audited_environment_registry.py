#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment_env.audited import generate_audited_configs


DEFAULT_AUDIT = ROOT / "configs" / "environment" / "audit-overlay-782df2d84077f93b69aed4c79b334643f7ef5cb33470fe5f38fc412434f90b6b.json"
DEFAULT_SOURCE_PRE = ROOT / "paper_outputs" / "strict-runs" / "exp7-strict-v6-20260718T091006Z-e9ff93d0c9fe0d6f15fb03c1214dd6e2" / "manifests" / "source-pre.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the exact experiments4/5 audited variant registry")
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--source-pre", type=Path, default=DEFAULT_SOURCE_PRE)
    args = parser.parse_args()
    result = generate_audited_configs(ROOT, args.audit, args.source_pre)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
