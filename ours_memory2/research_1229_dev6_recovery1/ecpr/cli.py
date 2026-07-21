"""Command-line entry points for the standalone package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .contracts import CandidatePolicy
from .evaluate import evaluate_paired
from .final_protocol import authorize_final_cli
from .firewall import validate_sanitized_history
from .inference import run_inference
from .integrity import ATTEMPT_LOCK, IMPLEMENTATION_MANIFEST
from .io import load_json, sha256_file, write_json, write_jsonl_once
from .ledger import (
    ECPR_MEMORY,
    PREFINE_MEMORY,
    PROVIDER_JOURNAL,
    ProviderJournal,
    enforce_final_build_memory_arguments,
    enforce_final_evaluation_arguments,
    enforce_unsealed_output_arguments,
    final_paths,
    publish_memory_manifests,
)
from .memory import build_memory_records, write_memory_records
from .prepare import prepare_bundle
from .provider import LoopbackChatClient


ROOT = Path(__file__).resolve().parents[1]


def _emit(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ecpr", description="Standalone ECPR research package")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="build gold-free tasks and sealed evaluator artifacts")
    prepare.add_argument("--root", type=Path, default=ROOT)
    prepare.add_argument("--preregistration", type=Path, default=ROOT / "preregistration.json")

    memory = sub.add_parser("build-memory", help="build PREFINE + typed ECPR memory")
    memory.add_argument("--root", type=Path, default=ROOT)
    memory.add_argument("--preregistration", type=Path, default=ROOT / "preregistration.json")
    memory.add_argument("--history", type=Path, default=ROOT / "artifacts/history.sanitized.jsonl")
    memory.add_argument("--preference-slots", type=Path, default=ROOT / "configs/preference_slots.json")
    memory.add_argument("--output", type=Path, default=ROOT / "artifacts/memory.ecpr.jsonl")
    memory.add_argument("--final-attempt-id")
    memory.add_argument("--runner-nonce-fd", type=int)
    source = memory.add_mutually_exclusive_group(required=True)
    source.add_argument("--base-url", help="loopback OpenAI-compatible server used to build latent PREFINE memory")
    source.add_argument("--latent-path", type=Path, help="standalone prebuilt latent JSONL")

    infer = sub.add_parser("infer", help="run one paired action arm without access to gold")
    infer.add_argument("--root", type=Path, default=ROOT)
    infer.add_argument("--preregistration", type=Path, default=ROOT / "preregistration.json")
    infer.add_argument("--arm", choices=("baseline", "candidate"), required=True)
    infer.add_argument("--base-url", required=True)
    infer.add_argument("--output", type=Path, required=True)
    infer.add_argument("--ablation", action="append", default=[])
    infer.add_argument("--final-attempt-id")
    infer.add_argument("--runner-nonce-fd", type=int)

    evaluate = sub.add_parser("evaluate", help="run sealed paired evaluator")
    evaluate.add_argument("--root", type=Path, default=ROOT)
    evaluate.add_argument("--preregistration", type=Path, default=ROOT / "preregistration.json")
    evaluate.add_argument("--baseline", type=Path, required=True)
    evaluate.add_argument("--candidate", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, default=ROOT / "reports/summary.json")
    evaluate.add_argument("--final-attempt-id", required=True)
    evaluate.add_argument("--runner-nonce-fd", type=int, required=True)
    evaluate.add_argument("--gold-open-fd", type=int, required=True)
    return parser


def _final_contract(
    root: Path, attempt_id: str, runner_nonce_fd: int | None
) -> tuple[dict[str, Any], bytes]:
    _attempt, implementation, runner_nonce = authorize_final_cli(
        root,
        attempt_id,
        runner_nonce_fd if runner_nonce_fd is not None else -1,
    )
    return implementation["expected_contract"], runner_nonce


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        root = args.root.resolve()
        if (root / IMPLEMENTATION_MANIFEST).exists() or (root / ATTEMPT_LOCK).exists():
            raise ValueError("prepare is forbidden after implementation sealing or attempt acquisition")
        _emit(prepare_bundle(args.root, args.preregistration))
        return 0
    if args.command == "build-memory":
        root = args.root.resolve()
        journal = None
        checkpoint = None
        provider_timeout_seconds = 120.0
        if args.final_attempt_id:
            contract, _runner_nonce = _final_contract(
                root, args.final_attempt_id, args.runner_nonce_fd
            )
            provider_timeout_seconds = float(
                contract["provider_timeout_seconds"]
            )
            enforce_final_build_memory_arguments(
                root,
                contract,
                preregistration=args.preregistration,
                history=args.history,
                preference_slots=args.preference_slots,
                output=args.output,
                base_url=args.base_url,
                latent_path=args.latent_path,
            )
            journal = ProviderJournal(root / PROVIDER_JOURNAL, args.final_attempt_id)
            checkpoint = journal.checkpoint()
        else:
            enforce_unsealed_output_arguments(root, args.output)
        validate_sanitized_history(args.history)
        prereg = load_json(args.preregistration)
        policy = CandidatePolicy(**prereg["candidate_parameters"])
        if not policy.recency_tiebreak_only or policy.raw_api_history_in_candidate_prompt:
            raise ValueError("candidate policy violates the preregistered non-cheating contract")
        client = (
            LoopbackChatClient(
                args.base_url,
                prereg["model"]["id"],
                timeout_seconds=provider_timeout_seconds,
                journal=journal,
            )
            if args.base_url
            else None
        )
        records = build_memory_records(
            args.history,
            load_json(args.preference_slots),
            policy,
            client=client,
            seed=int(prereg["inference_budget"]["seed"]),
            latent_path=args.latent_path,
        )
        write_memory_records(
            args.output,
            records,
            commit_once=bool(args.final_attempt_id),
        )
        if args.final_attempt_id:
            paths = final_paths(root)
            if args.output.resolve() != (root / ECPR_MEMORY).resolve():
                raise AssertionError("final memory output escaped its fixed path")
            write_jsonl_once(
                root / PREFINE_MEMORY,
                [
                    {
                        "example_id": record["example_id"],
                        "latent_abstraction": record["latent_abstraction"],
                        "method": "prefine_v1",
                    }
                    for record in records
                ],
            )
            generation_slice = journal.slice_from(checkpoint, "memory_generation")
            manifest = publish_memory_manifests(
                root,
                args.final_attempt_id,
                generation_slice,
            )
            _emit(manifest)
            return 0
        manifest = {
            "schema_version": 1,
            "record_count": len(records),
            "history_sha256": sha256_file(args.history),
            "memory_sha256": sha256_file(args.output),
            "latent_source": "loopback_prefine" if args.base_url else "standalone_jsonl",
        }
        write_json(str(args.output) + ".manifest.json", manifest)
        _emit(manifest)
        return 0
    if args.command == "infer":
        if not args.final_attempt_id:
            enforce_unsealed_output_arguments(args.root.resolve(), args.output)
        else:
            _final_contract(
                args.root.resolve(), args.final_attempt_id, args.runner_nonce_fd
            )
        _emit(
            run_inference(
                root=args.root,
                arm=args.arm,
                output=args.output,
                base_url=args.base_url,
                preregistration=args.preregistration,
                ablations=set(args.ablation),
                final_attempt_id=args.final_attempt_id,
            )
        )
        return 0
    if args.command == "evaluate":
        root = args.root.resolve()
        _contract, runner_nonce = _final_contract(
            root, args.final_attempt_id, args.runner_nonce_fd
        )
        enforce_final_evaluation_arguments(
            root,
            preregistration=args.preregistration,
            baseline=args.baseline,
            candidate=args.candidate,
            output=args.output,
        )
        summary = evaluate_paired(
            root=root,
            baseline_path=args.baseline,
            candidate_path=args.candidate,
            output=args.output,
            preregistration=args.preregistration,
            final_attempt_id=args.final_attempt_id,
            runner_nonce=runner_nonce,
            gold_open_fd=args.gold_open_fd,
        )
        _emit({"case_count": summary["case_count"], "pass": summary["pass"], "gates": summary["gates"]})
        # A valid sealed evaluation completed even when the preregistered
        # performance gates fail.  The top-level runner maps that axis to 1.
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
