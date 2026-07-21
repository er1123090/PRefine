"""Compatibility imports from the existing PEToolBench normalization adapter."""

from __future__ import annotations

import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
EXTERNAL_BENCHMARK_DIR = THIS_DIR.parent
PETOOLBENCH_OURS_MEMORY_DIR = EXTERNAL_BENCHMARK_DIR / "petoolbench_ours_memory"
REPO_ROOT = EXTERNAL_BENCHMARK_DIR.parent

if str(PETOOLBENCH_OURS_MEMORY_DIR) not in sys.path:
    sys.path.insert(0, str(PETOOLBENCH_OURS_MEMORY_DIR))

from common import (  # noqa: E402
    first_json_object,
    iter_jsonl,
    read_json,
    tool_call_to_text,
    write_json,
    write_jsonl,
)


__all__ = [
    "EXTERNAL_BENCHMARK_DIR",
    "PETOOLBENCH_OURS_MEMORY_DIR",
    "REPO_ROOT",
    "first_json_object",
    "iter_jsonl",
    "read_json",
    "tool_call_to_text",
    "write_json",
    "write_jsonl",
]

