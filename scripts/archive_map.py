#!/usr/bin/env python3
"""Generate or verify the non-destructive experiments7 archive map."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from exp7.provenance.archive_map import ArchiveMapError, check_archive_map, write_archive_map

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try: value = check_archive_map(args.root) if args.check else write_archive_map(args.root)
    except (ArchiveMapError, OSError) as exc:
        print(str(exc), file=sys.stderr); return 2
    print(json.dumps({"status": "pass", "mode": "check" if args.check else "generate", "schema": value["schema"], "record_count": value["record_count"], "planned_only": True, "no_mutations": True}, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
