#!/usr/bin/env python3
"""Run the canonical six-condition VanillaLLM suite."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exp7.experiments import ExperimentConfig, load_experiment_config
from exp7.provenance.run_artifacts import run_suite


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Run singleturn and multiturn easy/medium/hard into one "
            "manifest-bound artifact directory."
        )
    )
    value.add_argument(
        "--prepared-root",
        type=Path,
        default=ROOT / "artifacts/prepared/mix600-v1",
    )
    value.add_argument("--run-dir", "--output-root", dest="run_dir", type=Path)
    value.add_argument("--model", default=os.environ.get("MODEL", "gpt-5-mini"))
    value.add_argument("--reasoning-effort")
    value.add_argument("--base-url")
    value.add_argument("--api-key-env", default="OPENAI_API_KEY")
    value.add_argument("--python-bin", default=os.environ.get("PYTHON_BIN", sys.executable))
    value.add_argument("--resume", action="store_true")
    value.add_argument("--dry-run", action="store_true")
    value.add_argument(
        "runner_args",
        nargs=argparse.REMAINDER,
        help="Extra scripts/run.py options after --.",
    )
    return value


def _default_run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "artifacts/runs" / f"vanilla-{stamp}"


def _run_legacy(arguments: list[str]) -> int:
    args = parser().parse_args(arguments)
    if args.resume and args.run_dir is None:
        raise ValueError("--resume requires an explicit --run-dir")
    run_dir = args.run_dir or _default_run_dir()
    extra_args = list(args.runner_args)
    if extra_args[:1] == ["--"]:
        extra_args.pop(0)
    summary = run_suite(
        repo_root=ROOT,
        prepared_root=args.prepared_root,
        run_dir=run_dir,
        model=args.model,
        python_bin=args.python_bin,
        reasoning_effort=args.reasoning_effort,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        extra_args=extra_args,
        resume=args.resume,
        dry_run=args.dry_run,
        output=sys.stdout,
    )
    if not args.dry_run:
        print(
            json.dumps(
                {
                    "completed_conditions": summary.completed_conditions,
                    "run_dir": str(summary.run_dir),
                    "status": summary.status,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    return 0


def _config_parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Run one strict, versioned VanillaLLM experiment config."
    )
    value.add_argument("--config", required=True)
    value.add_argument("--run-dir", type=Path)
    value.add_argument("--allow-external-output", action="store_true")
    value.add_argument("--resume", action="store_true")
    value.add_argument("--dry-run", action="store_true")
    return value


def _uses_experiment_config(arguments: list[str]) -> bool:
    return any(
        argument == "--config" or argument.startswith("--config=")
        for argument in arguments
    )


def _configured_run_dir(config: ExperimentConfig) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return config.output.root / f"{config.output.run_name_prefix}-{stamp}"


def _run_config(arguments: list[str]) -> int:
    config_parser = _config_parser()
    args = config_parser.parse_args(arguments)
    if args.run_dir is not None and not args.allow_external_output:
        config_parser.error(
            "config-mode --run-dir requires --allow-external-output"
        )
    if args.allow_external_output and args.run_dir is None:
        config_parser.error("--allow-external-output requires --run-dir")
    config = load_experiment_config(ROOT, args.config)
    if args.resume and args.run_dir is None:
        raise ValueError("--resume requires an explicit --run-dir")
    run_dir = args.run_dir or _configured_run_dir(config)
    summary = run_suite(
        repo_root=ROOT,
        prepared_root=config.dataset.prepared_root,
        run_dir=run_dir,
        model=config.provider.model,
        python_bin=config.python_bin,
        reasoning_effort=config.provider.reasoning_effort,
        base_url=config.provider.base_url,
        api_key_env=config.provider.api_key_env,
        extra_args=config.runner_extra_args(),
        experiment_config_bytes=config.config_bytes,
        experiment_config_sha256=config.config_sha256,
        experiment_config_relative_path=config.config_relative_path,
        evaluator_variant=config.evaluator_variant,
        output_policy=(
            "external_opt_in" if args.allow_external_output else "config_root"
        ),
        resume=args.resume,
        dry_run=args.dry_run,
        output=sys.stdout,
    )
    if not args.dry_run:
        print(
            json.dumps(
                {
                    "completed_conditions": summary.completed_conditions,
                    "run_dir": str(summary.run_dir),
                    "status": summary.status,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    return 1 if summary.status == "all_provider_errors" else 0


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if _uses_experiment_config(arguments):
        return _run_config(arguments)
    return _run_legacy(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
