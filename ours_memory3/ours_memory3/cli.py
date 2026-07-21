"""Public-only command line interface for the ours_memory3 overlay."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .contracts import OverlayPolicy, PublicInputError
from .jsonl import read_jsonl, write_jsonl_once
from .overlay import build_overlay


_FORBIDDEN_KEYS = frozenset(
    {
        "answer",
        "answers",
        "gold",
        "golden",
        "ground_truth",
        "groundtruth",
        "label",
        "labels",
        "prediction",
        "predictions",
        "reference_answer",
        "reference_ground_truth",
        "expected_output",
        "expected_calls",
    }
)


def assert_public_only(value: object, path: str = "row") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise PublicInputError(f"{path} has a non-string key")
            if key.casefold() in _FORBIDDEN_KEYS:
                raise PublicInputError(f"{path}.{key} is not allowed in public input")
            assert_public_only(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_public_only(item, f"{path}[{index}]")


def _policy(path: Path | None) -> OverlayPolicy:
    if path is None:
        return OverlayPolicy()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicInputError("policy must be a valid JSON object") from exc
    if not isinstance(value, dict):
        raise PublicInputError("policy must be a JSON object")
    allowed = set(OverlayPolicy.__dataclass_fields__)
    if set(value) - allowed:
        raise PublicInputError("policy has an unknown field")
    return OverlayPolicy(**value)


def build(input_path: Path, output_path: Path, policy_path: Path | None) -> dict[str, int]:
    policy = _policy(policy_path)
    rows = read_jsonl(input_path)
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        assert_public_only(row, f"rows[{index}]")
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise PublicInputError("case_id must be a unique nonempty string")
        seen.add(case_id)
        decision = build_overlay(
            baseline_memory=row.get("baseline_memory"),
            history=row.get("history"),
            query=row.get("query"),
            mode=row.get("mode", "singleturn"),
            preference_slots=row.get("preference_slots"),
            schema=row.get("schema"),
            schema_domain=row.get("schema_domain"),
            domain_aliases=row.get("domain_aliases"),
            slot_aliases=row.get("slot_aliases"),
            policy=policy,
        )
        output.append(
            {
                "case_id": case_id,
                "memory": decision.memory,
                "selected_domain": decision.selected_domain,
                "facts": [fact.as_public_dict() for fact in decision.facts],
                "explicit_slots": list(decision.explicit_slots),
                "baseline_unchanged": decision.baseline_unchanged,
                "reason": decision.reason,
            }
        )
    write_jsonl_once(output_path, output)
    return {"cases": len(output), "overlays": sum(not row["baseline_unchanged"] for row in output)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="build public-only evidence-calibrated memory overlays")
    subparsers = parser.add_subparsers(dest="command", required=True)
    overlay = subparsers.add_parser("overlay")
    overlay_sub = overlay.add_subparsers(dest="overlay_command", required=True)
    build_parser = overlay_sub.add_parser("build")
    build_parser.add_argument("--input", type=Path, required=True)
    build_parser.add_argument("--output", type=Path, required=True)
    build_parser.add_argument("--policy", type=Path)
    args = parser.parse_args(argv)
    try:
        result = build(args.input, args.output, args.policy)
    except PublicInputError as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True))
    return 0
