#!/usr/bin/env python3
"""Fail-closed tombstone for the retired, unconfined G0 wrapper.

Protected source and paper reads are available only through the strict V6 G0
child. Keeping this legacy path executable as a manifest launcher would allow
an older frozen argv or a path-carried producer binding to bypass that child.
"""
from __future__ import annotations

import json
import sys


def main() -> None:
    """Reject every legacy command before parsing paths or opening files."""
    print(
        json.dumps(
            {
                "status": "BLOCKED",
                "reason": (
                    "legacy protected-read wrapper is disabled; use the strict "
                    "V6 G0 child with stdin-bound producer evidence"
                ),
            },
            sort_keys=True,
        ),
        file=sys.stderr,
    )
    raise SystemExit(2)


if __name__ == "__main__":
    main()
