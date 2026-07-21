#!/usr/bin/env python3

from __future__ import annotations

import importlib
import sys

from plot_session_memory_curves_singleturn import plot_metric


def _consume_task_arg(argv: list[str]) -> tuple[str, list[str]]:
    task = "singleturn"
    remaining: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--task":
            if i + 1 >= len(argv):
                raise SystemExit("--task requires a value: singleturn or multiturn")
            task = argv[i + 1]
            i += 2
            continue
        if arg.startswith("--task="):
            task = arg.split("=", 1)[1]
            i += 1
            continue
        remaining.append(arg)
        i += 1
    return task, remaining


def _load_task_module(task: str):
    if task not in {"singleturn", "multiturn"}:
        raise SystemExit(f"Unsupported task: {task}")
    return importlib.import_module(f"plot_session_memory_curves_{task}")


def main() -> None:
    task, remaining = _consume_task_arg(sys.argv[1:])
    module = _load_task_module(task)
    sys.argv = [sys.argv[0]] + remaining
    module.main()


if __name__ == "__main__":
    main()
