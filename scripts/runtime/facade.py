#!/usr/bin/env python3
"""Only supported user entrypoint for the experiments7 neutral facade."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from facade.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(root=ROOT))
