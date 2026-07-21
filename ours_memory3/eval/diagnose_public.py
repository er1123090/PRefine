"""Aggregate-only public coverage diagnostics for the sealed ours_memory3 evaluator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if str(PACKAGE_ROOT / "eval") not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT / "eval"))

import run_paired
from ours_memory3.contracts import OverlayPolicy
from ours_memory3.overlay import build_overlay


def diagnose(root: Path) -> dict[str, object]:
    manifest, tasks, histories, schemas, ontology, slots = run_paired._runtime(root.resolve())
    policy = OverlayPolicy(**manifest["overlay_policy"])
    aliases = run_paired._domain_aliases(ontology)
    by_mode: dict[str, dict[str, int]] = {
        "singleturn": {"cases": 0, "overlay_cases": 0, "facts": 0, "explicit_masked": 0},
        "multiturn": {"cases": 0, "overlay_cases": 0, "facts": 0, "explicit_masked": 0},
    }
    domain_counts: dict[str, int] = {}
    for task in tasks:
        decision = build_overlay(
            baseline_memory="PUBLIC_DIAGNOSTIC_BASELINE",
            history=histories[task["example_id"]],
            query=task["query"],
            mode=task["mode"],
            preference_slots=slots,
            schema=schemas[task["schema_key"]],
            domain_aliases=aliases,
            policy=policy,
        )
        bucket = by_mode[task["mode"]]
        bucket["cases"] += 1
        bucket["facts"] += len(decision.facts)
        if decision.facts:
            bucket["overlay_cases"] += 1
        if decision.explicit_slots:
            bucket["explicit_masked"] += 1
        if decision.selected_domain is not None:
            domain_counts[decision.selected_domain] = domain_counts.get(decision.selected_domain, 0) + 1
    return {
        "status": "PUBLIC_DIAGNOSTIC_COMPLETE",
        "gold_opened": False,
        "prediction_opened": False,
        "by_mode": by_mode,
        "selected_domain_case_counts": dict(sorted(domain_counts.items())),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="aggregate-only ours_memory3 public coverage diagnostic")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(diagnose(args.root), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

