"""One consistent inference entrypoint for all experiment8 methods."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_PYTHON = ROOT.parent / ".venvs/experiment8/bin/python"
DEFAULT_PYTHON = (
    EXPERIMENT_PYTHON if EXPERIMENT_PYTHON.is_file() else Path(sys.executable)
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument(
        "--method",
        required=True,
        choices=[
            "vanilla_llm",
            "ours_memory",
            "rag",
            "mem0",
            "langmem",
            "amem",
        ],
    )
    result.add_argument("--turn", choices=["single", "multi"], required=True)
    result.add_argument("--query", choices=["hint", "nohint"], default="hint")
    result.add_argument("--schema", choices=["easy", "all"], default="easy")
    result.add_argument(
        "--pref_type", choices=["easy", "medium", "hard"], required=True
    )
    result.add_argument("--model", required=True)
    result.add_argument(
        "--provider",
        choices=["auto", "openrouter"],
        default="auto",
        help="Provider shortcut for OpenRouter-compatible endpoint."
    )
    result.add_argument(
        "--input_path",
        default=str(ROOT / "data/MPT_v2_mix600.json"),
    )
    result.add_argument("--context_type", default=None)
    result.add_argument("--memory_path", default=None)
    result.add_argument("--db_path", default=str(ROOT / "outputs/rag/chroma"))
    result.add_argument("--output_path", default=None)
    result.add_argument("--log_path", default=None)
    result.add_argument("--concurrency", type=int, default=20)
    result.add_argument("--reasoning_effort", default=None)
    result.add_argument("--base_url", default=None)
    result.add_argument("--api_key", default=None)
    result.add_argument("--embedding_model", default=None)
    result.add_argument("--embedding_base_url", default=None)
    result.add_argument("--embedding_api_key", default=None)
    result.add_argument("--memory_top_k", type=int, default=10)
    result.add_argument("--linked_neighbor_limit", type=int, default=10)
    result.add_argument("--request_timeout_seconds", type=float, default=None)
    result.add_argument("--client_max_retries", type=int, default=None)
    result.add_argument("--max_queries", type=int, default=None)
    result.add_argument("--python", default=str(DEFAULT_PYTHON))
    result.add_argument("--dry_run", action="store_true")
    return result


def add_if(command: List[str], flag: str, value: object | None) -> None:
    if value is not None and value != "":
        command.extend([flag, str(value)])


def build_command(args: argparse.Namespace) -> List[str]:
    query_path = ROOT / f"config/query_{args.turn}turn_{args.query}.json"
    schema_path = ROOT / f"config/schema_{args.schema}.json"
    model_safe = args.model.replace("/", "__")
    run_dir = (
        ROOT
        / "outputs"
        / args.method
        / args.turn
        / args.query
        / args.schema
        / args.pref_type
        / model_safe
    )
    output_path = Path(args.output_path) if args.output_path else run_dir / "predictions.json"
    log_path = Path(args.log_path) if args.log_path else run_dir / "inference.jsonl"

    shared = [
        "--input_path",
        args.input_path,
        "--pref_list_path",
        str(ROOT / "config/pref_list.json"),
        "--pref_group_path",
        str(ROOT / "config/pref_group.json"),
        "--tools_schema_path",
        str(schema_path),
        "--pref_type",
        args.pref_type,
        "--model_name",
        args.model,
        "--output_path",
        str(output_path),
        "--log_path",
        str(log_path),
        "--concurrency",
        str(args.concurrency),
    ]

    if args.method in {"vanilla_llm", "rag", "langmem", "amem"}:
        script = ROOT / f"methods/{args.method}/inference.py"
        command = [
            args.python,
            str(script),
            "--turn",
            args.turn,
            "--query_path",
            str(query_path),
            *shared,
        ]
        if args.method == "rag":
            command.extend(["--db_path", args.db_path])
            command.extend(
                ["--embedding_model", args.embedding_model or "text-embedding-3-small"]
            )
            add_if(command, "--embedding_base_url", args.embedding_base_url)
            add_if(command, "--embedding_api_key", args.embedding_api_key)
        if args.method in {"langmem", "amem"}:
            if not args.memory_path:
                raise ValueError(f"--memory_path is required for {args.method}")
            command.extend(["--memory_path", args.memory_path])
            if args.method == "langmem":
                command.extend(
                    [
                        "--embedding_model",
                        args.embedding_model or "text-embedding-3-small",
                    ]
                )
                add_if(command, "--embedding_base_url", args.embedding_base_url)
                add_if(command, "--embedding_api_key", args.embedding_api_key)
            else:
                command.extend(
                    [
                        "--embedding_model",
                        args.embedding_model or "all-MiniLM-L6-v2",
                    ]
                )
                command.extend(["--memory_top_k", str(args.memory_top_k)])
                command.extend(
                    [
                        "--linked_neighbor_limit",
                        str(args.linked_neighbor_limit),
                    ]
                )
        context_default = "diag-apilist" if args.method == "vanilla_llm" else "memory_api"
        command.extend(["--context_type", args.context_type or context_default])
        command.extend(["--provider", args.provider])
        add_if(command, "--reasoning_effort", args.reasoning_effort)
        add_if(command, "--base_url", args.base_url)
        add_if(command, "--api_key", args.api_key)
        if args.method == "vanilla_llm":
            add_if(command, "--request_timeout_seconds", args.request_timeout_seconds)
            add_if(command, "--client_max_retries", args.client_max_retries)
        add_if(command, "--max_queries", args.max_queries)
        return command

    suffix = "singleturn" if args.turn == "single" else "multiturn"
    script = ROOT / f"methods/{args.method}/inference_{suffix}.py"
    query_flag = "--query_path" if args.turn == "single" else "--multiturn_path"
    command = [args.python, str(script), query_flag, str(query_path), *shared]
    command.extend(["--context_type", args.context_type or "memory_api"])
    command.extend(["--provider", args.provider])
    if args.method == "ours_memory":
        if not args.memory_path:
            raise ValueError("--memory_path is required for ours_memory")
        command.extend(["--memory_path", args.memory_path])
        add_if(command, "--base_url", args.base_url)
        add_if(command, "--api_key", args.api_key)
        add_if(command, "--request_timeout_seconds", args.request_timeout_seconds)
        add_if(command, "--client_max_retries", args.client_max_retries)
        add_if(command, "--max_queries", args.max_queries)
    if args.method == "mem0":
        add_if(command, "--base_url", args.base_url)
        add_if(command, "--api_key", args.api_key)
    add_if(command, "--reasoning_effort", args.reasoning_effort)
    return command


def main() -> None:
    args = parser().parse_args()
    command = build_command(args)
    print(shlex.join(command))
    if not args.dry_run:
        Path(command[command.index("--output_path") + 1]).parent.mkdir(
            parents=True, exist_ok=True
        )
        Path(command[command.index("--log_path") + 1]).parent.mkdir(
            parents=True, exist_ok=True
        )
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
