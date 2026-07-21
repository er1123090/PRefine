"""Independent paired-metric audit used inside the single-open evaluator child.

The registered evaluator and this module intentionally have separate counting,
aggregation, randomization, and gate implementations.  They share only the
public row contracts and action-call parser.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Iterable

from .contracts import FORBIDDEN_KEYS, GOLD_FIELDS, PREDICTION_FIELDS
from .parsing import slot_value_map


Count = tuple[int, int, int]
INDEPENDENCE_BOUNDARY = {
    "execution": "same_process_same_identity_bound_input_bytes",
    "shared_components": [
        "ecpr.parsing.slot_value_map",
        "public_row_and_field_contracts",
    ],
    "independent_components": [
        "row_indexing",
        "count_aggregation",
        "metric_formulas",
        "bootstrap",
        "paired_randomization",
        "gate_recomputation",
    ],
    "claim": "independent_metric_implementation_not_process_or_parser_independence",
}


def _index(rows: Iterable[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row.get(key, ""))
        if not value or value in result:
            raise ValueError(f"independent audit duplicate or missing {key}")
        result[value] = row
    return result


def _plus(left: Count, right: Count) -> Count:
    return tuple(left[index] + right[index] for index in range(3))  # type: ignore[return-value]


def _score(expected: dict[tuple[str, str], set[str]], actual: dict[tuple[str, str], set[str]]) -> Count:
    true_positive = sum(
        1 for key, allowed in expected.items() if actual.get(key, set()) & allowed
    )
    false_negative = sum(
        1 for key, allowed in expected.items() if not actual.get(key, set()) & allowed
    )
    false_positive = sum(
        1
        for key, values in actual.items()
        if key not in expected or not values & expected[key]
    )
    return true_positive, false_positive, false_negative


def _slice(
    values: dict[tuple[str, str], set[str]],
    preference_slots: dict[str, list[str]],
    preference: bool,
) -> dict[tuple[str, str], set[str]]:
    return {
        key: allowed
        for key, allowed in values.items()
        if ((key[1] in preference_slots.get(key[0], [])) is preference)
    }


def _f1(count: Count) -> float:
    denominator = 2 * count[0] + count[1] + count[2]
    return 2 * count[0] / denominator if denominator else 0.0


def _metric(count: Count) -> dict[str, Any]:
    tp, fp, fn = count
    return {
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": _f1(count),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _balanced(counts: dict[str, Count]) -> float:
    return 0.5 * _f1(counts["singleturn"]) + 0.5 * _f1(counts["multiturn"])


def _aggregate(
    clusters: dict[str, dict[str, dict[str, Count]]],
    ids: list[str],
    arm: str,
) -> dict[str, Count]:
    result = {"singleturn": (0, 0, 0), "multiturn": (0, 0, 0)}
    for cluster_id in ids:
        for mode in result:
            result[mode] = _plus(result[mode], clusters[cluster_id][arm][mode])
    return result


def _statistics(
    clusters: dict[str, dict[str, dict[str, Count]]], spec: dict[str, Any]
) -> dict[str, Any]:
    ids = sorted(clusters)
    if not ids:
        raise ValueError("independent audit has no paired clusters")
    observed = _balanced(_aggregate(clusters, ids, "candidate")) - _balanced(
        _aggregate(clusters, ids, "baseline")
    )
    bootstrap_draws = int(spec["bootstrap_draws"])
    bootstrap = random.Random(int(spec["bootstrap_seed"]))
    deltas: list[float] = []
    for _ in range(bootstrap_draws):
        sampled = [ids[bootstrap.randrange(len(ids))] for _ in ids]
        deltas.append(
            _balanced(_aggregate(clusters, sampled, "candidate"))
            - _balanced(_aggregate(clusters, sampled, "baseline"))
        )
    deltas.sort()
    randomization_draws = int(spec["randomization_draws"])
    randomization = random.Random(int(spec["randomization_seed"]))
    at_least = 0
    for _ in range(randomization_draws):
        totals = {
            arm: {mode: (0, 0, 0) for mode in ("singleturn", "multiturn")}
            for arm in ("baseline", "candidate")
        }
        for cluster_id in ids:
            swap = bool(randomization.getrandbits(1))
            for mode in ("singleturn", "multiturn"):
                baseline = clusters[cluster_id]["baseline"][mode]
                candidate = clusters[cluster_id]["candidate"][mode]
                if swap:
                    baseline, candidate = candidate, baseline
                totals["baseline"][mode] = _plus(totals["baseline"][mode], baseline)
                totals["candidate"][mode] = _plus(totals["candidate"][mode], candidate)
        if _balanced(totals["candidate"]) - _balanced(totals["baseline"]) >= observed - 1e-15:
            at_least += 1
    return {
        "delta_bmf1": observed,
        "bootstrap_ci_95": [
            deltas[max(0, int(0.025 * bootstrap_draws))],
            deltas[min(bootstrap_draws - 1, int(0.975 * bootstrap_draws))],
        ],
        "bootstrap_draws": bootstrap_draws,
        "bootstrap_seed": int(spec["bootstrap_seed"]),
        "randomization_p_one_sided": (at_least + 1) / (randomization_draws + 1),
        "randomization_draws": randomization_draws,
        "randomization_seed": int(spec["randomization_seed"]),
        "cluster_count": len(ids),
    }


def audit_paired_rows(
    *,
    gold_rows: Iterable[dict[str, Any]],
    baseline_rows: Iterable[dict[str, Any]],
    candidate_rows: Iterable[dict[str, Any]],
    preference_slots: dict[str, list[str]],
    preregistration: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    gold = _index(gold_rows, "case_key")
    predictions = {
        "baseline": _index(baseline_rows, "case_key"),
        "candidate": _index(candidate_rows, "case_key"),
    }
    for row in gold.values():
        if set(row) != GOLD_FIELDS:
            raise ValueError("independent gold field contract violation")
    for arm, rows in predictions.items():
        for row in rows.values():
            if (
                set(row) != PREDICTION_FIELDS
                or row.get("arm") != arm
                or {str(key).casefold() for key in row} & FORBIDDEN_KEYS
            ):
                raise ValueError(f"independent {arm} prediction contract violation")
        if set(rows) - set(gold):
            raise ValueError(f"independent {arm} predictions exceed gold universe")
        for case_key, row in rows.items():
            target = gold[case_key]
            if (
                row.get("example_id") != target.get("example_id")
                or row.get("mode") != target.get("mode")
            ):
                raise ValueError(
                    f"independent {arm} prediction identity differs from gold"
                )

    names = ("singleturn", "multiturn", "preference", "nonpreference")
    totals = {
        arm: {name: (0, 0, 0) for name in names}
        for arm in ("baseline", "candidate")
    }
    parse_failures = {"baseline": 0, "candidate": 0}
    clusters: dict[str, dict[str, dict[str, Count]]] = defaultdict(
        lambda: {
            arm: {mode: (0, 0, 0) for mode in ("singleturn", "multiturn")}
            for arm in ("baseline", "candidate")
        }
    )
    for case_key, target in gold.items():
        expected = slot_value_map(target["reference_ground_truth"])
        for arm in ("baseline", "candidate"):
            row = predictions[arm].get(case_key)
            text = "" if row is None or row.get("status") != "ok" else row.get("llm_output", "")
            actual = slot_value_map(text)
            if not actual:
                parse_failures[arm] += 1
            count = _score(expected, actual)
            mode = str(target["mode"])
            totals[arm][mode] = _plus(totals[arm][mode], count)
            for name, preference in (("preference", True), ("nonpreference", False)):
                sliced = _score(
                    _slice(expected, preference_slots, preference),
                    _slice(actual, preference_slots, preference),
                )
                totals[arm][name] = _plus(totals[arm][name], sliced)
            cluster = str(target["example_id"])
            clusters[cluster][arm][mode] = _plus(clusters[cluster][arm][mode], count)

    metrics = {
        arm: {name: _metric(count) for name, count in arm_totals.items()}
        for arm, arm_totals in totals.items()
    }
    for arm in metrics:
        metrics[arm]["bmf1"] = 0.5 * metrics[arm]["singleturn"]["f1"] + 0.5 * metrics[arm]["multiturn"]["f1"]
        metrics[arm]["parse_failure_rate"] = parse_failures[arm] / len(gold) if gold else 1.0
    deltas = {
        name: metrics["candidate"][name]["f1"] - metrics["baseline"][name]["f1"]
        for name in names
    }
    deltas["bmf1"] = metrics["candidate"]["bmf1"] - metrics["baseline"]["bmf1"]
    deltas["parse_failure_rate"] = metrics["candidate"]["parse_failure_rate"] - metrics["baseline"]["parse_failure_rate"]
    statistics = _statistics(dict(clusters), preregistration["statistics"])
    coverage = {
        arm: len(set(rows) & set(gold)) / len(gold) if gold else 0.0
        for arm, rows in predictions.items()
    }
    guard = preregistration["guardrails"]
    pass_spec = preregistration["pass"]
    gates = {
        "minimum_delta_bmf1": deltas["bmf1"] >= float(pass_spec["minimum_delta_bmf1"]),
        "bootstrap_ci_lower_positive": statistics["bootstrap_ci_95"][0] > 0.0,
        "randomization_p": statistics["randomization_p_one_sided"] < float(pass_spec["maximum_p_value_exclusive"]),
        "each_task_delta": all(deltas[name] >= float(guard["minimum_each_task_delta_f1"]) for name in ("singleturn", "multiturn")),
        "preference_drop": deltas["preference"] >= -float(guard["maximum_preference_f1_drop"]),
        "nonpreference_drop": deltas["nonpreference"] >= -float(guard["maximum_nonpreference_f1_drop"]),
        "parse_failure_increase": deltas["parse_failure_rate"] <= float(guard["maximum_parse_failure_rate_increase"]),
        "coverage": all(value == float(guard["required_coverage"]) for value in coverage.values()),
        "equal_action_budget": provenance.get("equal_action_budget") is True,
    }
    return {
        "schema_version": 1,
        "case_count": len(gold),
        "coverage": coverage,
        "equal_action_budget": provenance.get("equal_action_budget") is True,
        "metrics": metrics,
        "deltas": deltas,
        "paired_statistics": statistics,
        "gates": gates,
        "pass": all(gates.values()),
    }


def assert_registered_agreement(registered: dict[str, Any], independent: dict[str, Any]) -> None:
    for key in (
        "schema_version",
        "case_count",
        "coverage",
        "equal_action_budget",
        "metrics",
        "deltas",
        "paired_statistics",
        "gates",
        "pass",
    ):
        if registered.get(key) != independent.get(key):
            raise ValueError(f"registered and independent evaluator disagree on {key}")
