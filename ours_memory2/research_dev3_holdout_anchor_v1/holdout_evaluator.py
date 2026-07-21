"""Single-open, aggregate-only paired evaluator for the fresh holdout."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from ecpr.contracts import FORBIDDEN_KEYS, GOLD_FIELDS, PREDICTION_FIELDS
from ecpr.independent_audit import assert_registered_agreement, audit_paired_rows
from ecpr.io import canonical_json, load_json, sha256_bytes, sha256_file, unique_by, write_json_once
from ecpr.parsing import slot_value_map
from ecpr.statistics import f1, paired_cluster_statistics


ROOT = Path(__file__).resolve().parent
Count = tuple[int, int, int]


def _add(left: Count, right: Count) -> Count:
    return left[0] + right[0], left[1] + right[1], left[2] + right[2]


def _counts(
    expected: dict[tuple[str, str], set[str]],
    actual: dict[tuple[str, str], set[str]],
) -> Count:
    true_positive = sum(
        1 for key, allowed in expected.items() if actual.get(key, set()) & allowed
    )
    false_negative = sum(
        1 for key, allowed in expected.items() if not actual.get(key, set()) & allowed
    )
    false_positive = sum(
        1 for key, values in actual.items() if key not in expected or not values & expected[key]
    )
    return true_positive, false_positive, false_negative


def _filter_slots(
    values: dict[tuple[str, str], set[str]],
    preference_slots: dict[str, list[str]],
    preference: bool,
) -> dict[tuple[str, str], set[str]]:
    return {
        key: allowed
        for key, allowed in values.items()
        if ((key[1] in preference_slots.get(key[0], [])) is preference)
    }


def _metric(counts: Count) -> dict[str, Any]:
    true_positive, false_positive, false_negative = counts
    return {
        "precision": true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0,
        "recall": true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0,
        "f1": f1(counts),
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
    }


def _validate_prediction(row: dict[str, Any], arm: str) -> None:
    if set(row) != PREDICTION_FIELDS or row.get("arm") != arm:
        raise ValueError(f"{arm} prediction field contract violation")
    if {str(key).casefold() for key in row} & FORBIDDEN_KEYS:
        raise ValueError("prediction leaked an evaluator-only field")


def compute_registered_summary(
    *,
    gold_rows: Iterable[dict[str, Any]],
    baseline_rows: Iterable[dict[str, Any]],
    candidate_rows: Iterable[dict[str, Any]],
    preference_slots: dict[str, list[str]],
    preregistration: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    gold = unique_by(gold_rows, "case_key")
    expected_keys = set(gold)
    predictions = {
        "baseline": unique_by(baseline_rows, "case_key"),
        "candidate": unique_by(candidate_rows, "case_key"),
    }
    for target in gold.values():
        if set(target) != GOLD_FIELDS:
            raise ValueError("gold field contract violation")
    coverage: dict[str, float] = {}
    for arm, rows in predictions.items():
        for row in rows.values():
            _validate_prediction(row, arm)
        if set(rows) - expected_keys:
            raise ValueError(f"{arm} predictions exceed frozen public key universe")
        coverage[arm] = len(set(rows) & expected_keys) / len(expected_keys) if expected_keys else 0.0

    names = ("singleturn", "multiturn", "preference", "nonpreference")
    totals: dict[str, dict[str, Count]] = {
        arm: {name: (0, 0, 0) for name in names}
        for arm in predictions
    }
    parse_failures = {arm: 0 for arm in predictions}
    clusters: dict[str, dict[str, dict[str, Count]]] = defaultdict(
        lambda: {
            arm: {"singleturn": (0, 0, 0), "multiturn": (0, 0, 0)}
            for arm in predictions
        }
    )
    for case_key, target in gold.items():
        expected = slot_value_map(target["reference_ground_truth"])
        mode = str(target["mode"])
        example_id = str(target["example_id"])
        for arm, rows in predictions.items():
            row = rows.get(case_key)
            if row is not None and (
                row.get("example_id") != example_id or row.get("mode") != mode
            ):
                raise ValueError(f"{arm} prediction identity differs from evaluator row")
            output = "" if row is None or row.get("status") != "ok" else str(row.get("llm_output", ""))
            actual = slot_value_map(output)
            if not actual:
                parse_failures[arm] += 1
            counts = _counts(expected, actual)
            totals[arm][mode] = _add(totals[arm][mode], counts)
            for name, preference in (("preference", True), ("nonpreference", False)):
                totals[arm][name] = _add(
                    totals[arm][name],
                    _counts(
                        _filter_slots(expected, preference_slots, preference),
                        _filter_slots(actual, preference_slots, preference),
                    ),
                )
            clusters[example_id][arm][mode] = _add(
                clusters[example_id][arm][mode], counts
            )

    metrics = {
        arm: {name: _metric(count) for name, count in values.items()}
        for arm, values in totals.items()
    }
    for arm in metrics:
        metrics[arm]["bmf1"] = 0.5 * metrics[arm]["singleturn"]["f1"] + 0.5 * metrics[arm]["multiturn"]["f1"]
        metrics[arm]["parse_failure_rate"] = parse_failures[arm] / len(gold) if gold else 1.0
    deltas = {
        name: metrics["candidate"][name]["f1"] - metrics["baseline"][name]["f1"]
        for name in names
    }
    deltas["bmf1"] = metrics["candidate"]["bmf1"] - metrics["baseline"]["bmf1"]
    deltas["parse_failure_rate"] = (
        metrics["candidate"]["parse_failure_rate"] - metrics["baseline"]["parse_failure_rate"]
    )
    statistics = preregistration["statistics"]
    paired = paired_cluster_statistics(
        dict(clusters),
        bootstrap_draws=int(statistics["bootstrap_draws"]),
        bootstrap_seed=int(statistics["bootstrap_seed"]),
        randomization_draws=int(statistics["randomization_draws"]),
        randomization_seed=int(statistics["randomization_seed"]),
    )
    guardrails = preregistration["guardrails"]
    pass_spec = preregistration["pass"]
    gates = {
        "minimum_delta_bmf1": deltas["bmf1"] >= float(pass_spec["minimum_delta_bmf1"]),
        "bootstrap_ci_lower_positive": paired["bootstrap_ci_95"][0] > 0.0,
        "randomization_p": paired["randomization_p_one_sided"] < float(pass_spec["maximum_p_value_exclusive"]),
        "each_task_delta": all(
            deltas[name] >= float(guardrails["minimum_each_task_delta_f1"])
            for name in ("singleturn", "multiturn")
        ),
        "preference_drop": deltas["preference"] >= -float(guardrails["maximum_preference_f1_drop"]),
        "nonpreference_drop": deltas["nonpreference"] >= -float(guardrails["maximum_nonpreference_f1_drop"]),
        "parse_failure_increase": deltas["parse_failure_rate"] <= float(guardrails["maximum_parse_failure_rate_increase"]),
        "coverage": all(value == float(guardrails["required_coverage"]) for value in coverage.values()),
        "equal_action_budget": provenance.get("equal_action_budget") is True,
    }
    return {
        "schema_version": 1,
        "case_count": len(gold),
        "coverage": coverage,
        "equal_action_budget": provenance.get("equal_action_budget") is True,
        "metrics": metrics,
        "deltas": deltas,
        "paired_statistics": paired,
        "gates": gates,
        "pass": all(gates.values()),
        "provenance": provenance,
    }


def _read_jsonl_once(path: Path) -> tuple[list[dict[str, Any]], str]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    rows: list[dict[str, Any]] = []
    for line in payload.decode("utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("sealed JSONL row must be an object")
        rows.append(value)
    return rows, sha256_bytes(payload)


def evaluate(root: Path) -> dict[str, Any]:
    root = root.resolve()
    receipt_path = root / "artifacts/pre_gold_receipt.json"
    if not receipt_path.is_file():
        raise ValueError("missing pre-gold receipt")
    receipt = load_json(receipt_path)
    prereg = load_json(root / "preregistration.json")
    seal = load_json(root / "evaluator_vault/sealed_manifest.json")
    if sha256_file(root / "preregistration.json") != seal["preregistration_sha256"]:
        raise ValueError("preregistration no longer matches evaluator seal")
    baseline_path = root / "artifacts/predictions.baseline.jsonl"
    candidate_path = root / "artifacts/predictions.candidate.jsonl"
    for arm, path in (("baseline", baseline_path), ("candidate", candidate_path)):
        if receipt.get("outputs", {}).get(arm, {}).get("sha256") != sha256_file(path):
            raise ValueError(f"{arm} prediction does not match pre-gold receipt")
    if receipt.get("equal_action_budget") is not True:
        raise ValueError("pre-gold receipt did not establish equal action budget")
    if sha256_file(root / "artifacts/tasks.jsonl") != seal["task_sha256"]:
        raise ValueError("frozen public task identity changed")

    # This is intentionally the first and only evaluator open of gold bytes.
    gold_rows, gold_sha256 = _read_jsonl_once(root / "evaluator_vault/gold.jsonl")
    if gold_sha256 != seal["gold_sha256"]:
        raise ValueError("sealed gold hash mismatch")
    baseline_rows, baseline_sha256 = _read_jsonl_once(baseline_path)
    candidate_rows, candidate_sha256 = _read_jsonl_once(candidate_path)
    preference_slots = load_json(root / "configs/preference_slots.json")
    if not isinstance(preference_slots, dict):
        raise ValueError("preference slots must be an object")
    provenance = {
        "equal_action_budget": True,
        "pre_gold_receipt_sha256": sha256_file(receipt_path),
        "gold_opened_once": True,
    }
    summary = compute_registered_summary(
        gold_rows=gold_rows,
        baseline_rows=baseline_rows,
        candidate_rows=candidate_rows,
        preference_slots=preference_slots,
        preregistration=prereg,
        provenance=provenance,
    )
    independent = audit_paired_rows(
        gold_rows=gold_rows,
        baseline_rows=baseline_rows,
        candidate_rows=candidate_rows,
        preference_slots=preference_slots,
        preregistration=prereg,
        provenance=provenance,
    )
    assert_registered_agreement(summary, independent)
    output = root / "reports/summary.json"
    evidence = root / "reports/evaluation_evidence.json"
    if output.exists() or evidence.exists():
        raise FileExistsError("evaluator output already exists")
    write_json_once(output, summary)
    write_json_once(
        evidence,
        {
            "schema_version": 1,
            "kind": "single_open_dual_metric_evidence_v1",
            "gold_sha256": gold_sha256,
            "baseline_predictions_sha256": baseline_sha256,
            "candidate_predictions_sha256": candidate_sha256,
            "registered_summary_sha256": sha256_file(output),
            "independent_audit": independent,
            "agreement": True,
            "gold_opened_once": True,
        },
    )
    return {
        "case_count": summary["case_count"],
        "pass": summary["pass"],
        "summary_sha256": sha256_file(output),
        "evaluation_evidence_sha256": sha256_file(evidence),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="sealed aggregate-only fresh-holdout evaluator")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    print(json.dumps(evaluate(args.root), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
