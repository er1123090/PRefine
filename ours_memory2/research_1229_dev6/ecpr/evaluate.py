"""Fail-closed paired evaluation over the sealed key universe."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .contracts import FORBIDDEN_KEYS, GOLD_FIELDS, PREDICTION_FIELDS
from .final_protocol import (
    EVALUATION_EVIDENCE,
    commit_stage_once,
    consume_gold_open_capability,
    open_regular_bytes_once,
    parse_jsonl_bytes,
    validate_stage_chain,
    verify_runner_nonce,
    validate_final_attempt_v2,
)
from .independent_audit import (
    INDEPENDENCE_BOUNDARY,
    audit_paired_rows,
    assert_registered_agreement,
)
from .integrity import validate_preference_slots
from .io import (
    iter_jsonl,
    load_json,
    sha256_bytes,
    sha256_file,
    unique_by,
    write_json_once,
)
from .ledger import enforce_final_evaluation_arguments, validate_run_dag
from .parsing import slot_value_map
from .statistics import f1, paired_cluster_statistics


Count = tuple[int, int, int]


def _add(left: Count, right: Count) -> Count:
    return left[0] + right[0], left[1] + right[1], left[2] + right[2]


def _counts(gt: dict[tuple[str, str], set[str]], pred: dict[tuple[str, str], set[str]]) -> Count:
    tp = fp = fn = 0
    for key, allowed in gt.items():
        predicted = pred.get(key, set())
        if predicted & allowed:
            tp += 1
        else:
            fn += 1
    for key, predicted in pred.items():
        if key not in gt or not (predicted & gt[key]):
            fp += 1
    return tp, fp, fn


def _filter_slots(
    mapping: dict[tuple[str, str], set[str]], preference_slots: dict[str, list[str]], preference: bool
) -> dict[tuple[str, str], set[str]]:
    return {
        key: values
        for key, values in mapping.items()
        if ((key[1] in preference_slots.get(key[0], [])) is preference)
    }


def _metric(counts: Count) -> dict[str, Any]:
    tp, fp, fn = counts
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": precision, "recall": recall, "f1": f1(counts), "tp": tp, "fp": fp, "fn": fn}


def _validate_prediction(row: dict[str, Any], arm: str) -> None:
    if set(row) != PREDICTION_FIELDS:
        raise ValueError(f"{arm} prediction field contract violation")
    if row.get("arm") != arm:
        raise ValueError(f"wrong arm label in {arm} predictions")
    if {str(key).casefold() for key in row} & FORBIDDEN_KEYS:
        raise ValueError("prediction contains forbidden fields")


def _keyed_rows(
    rows: Iterable[dict[str, Any]] | dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return dict(rows) if isinstance(rows, dict) else unique_by(rows, "case_key")


def compute_paired_summary(
    *,
    gold_rows: Iterable[dict[str, Any]] | dict[str, dict[str, Any]],
    baseline_rows: Iterable[dict[str, Any]] | dict[str, dict[str, Any]],
    candidate_rows: Iterable[dict[str, Any]] | dict[str, dict[str, Any]],
    preference_slots: dict[str, list[str]],
    preregistration: dict[str, Any],
    provenance: dict[str, Any] | None = None,
    input_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Pure metric core: it has no filesystem or evaluator-vault capability."""
    gold = _keyed_rows(gold_rows)
    for row in gold.values():
        if set(row) != GOLD_FIELDS:
            raise ValueError("gold field contract violation")
    supplied = {"baseline": baseline_rows, "candidate": candidate_rows}
    predictions: dict[str, dict[str, dict[str, Any]]] = {}
    coverage: dict[str, float] = {}
    expected_keys = set(gold)
    for arm, source in supplied.items():
        rows = _keyed_rows(source)
        for row in rows.values():
            _validate_prediction(row, arm)
        extras = set(rows) - expected_keys
        if extras:
            raise ValueError(f"{arm} predictions contain keys outside frozen universe")
        predictions[arm] = rows
        coverage[arm] = len(set(rows) & expected_keys) / len(expected_keys) if expected_keys else 0.0

    totals: dict[str, dict[str, Count]] = {
        arm: {
            "singleturn": (0, 0, 0),
            "multiturn": (0, 0, 0),
            "preference": (0, 0, 0),
            "nonpreference": (0, 0, 0),
        }
        for arm in supplied
    }
    parse_failures = {arm: 0 for arm in supplied}
    clusters: dict[str, dict[str, dict[str, Count]]] = defaultdict(
        lambda: {
            "baseline": {"singleturn": (0, 0, 0), "multiturn": (0, 0, 0)},
            "candidate": {"singleturn": (0, 0, 0), "multiturn": (0, 0, 0)},
        }
    )
    budget_equal = provenance["equal_action_budget"] if provenance else True
    for case_key, gold_row in gold.items():
        mode = str(gold_row["mode"])
        example_id = str(gold_row["example_id"])
        gt_map = slot_value_map(gold_row["reference_ground_truth"])
        per_arm_rows = {arm: predictions[arm].get(case_key) for arm in supplied}
        for arm, row in per_arm_rows.items():
            if row is not None and (
                row.get("example_id") != example_id or row.get("mode") != mode
            ):
                raise ValueError(f"{arm} prediction identity differs from gold")
        if provenance is None:
            if None in per_arm_rows.values():
                budget_equal = False
            elif any(
                per_arm_rows["baseline"][field] != per_arm_rows["candidate"][field]
                for field in ("example_id", "mode", "model_snapshot", "seed", "temperature", "max_tokens", "calls", "schema_hash")
            ):
                budget_equal = False
        for arm, row in per_arm_rows.items():
            output_value = "" if row is None or row.get("status") != "ok" else row.get("llm_output", "")
            pred_map = slot_value_map(output_value)
            if not pred_map:
                parse_failures[arm] += 1
            counts = _counts(gt_map, pred_map)
            totals[arm][mode] = _add(totals[arm][mode], counts)
            pref_counts = _counts(
                _filter_slots(gt_map, preference_slots, True),
                _filter_slots(pred_map, preference_slots, True),
            )
            nonpref_counts = _counts(
                _filter_slots(gt_map, preference_slots, False),
                _filter_slots(pred_map, preference_slots, False),
            )
            totals[arm]["preference"] = _add(totals[arm]["preference"], pref_counts)
            totals[arm]["nonpreference"] = _add(totals[arm]["nonpreference"], nonpref_counts)
            clusters[example_id][arm][mode] = _add(clusters[example_id][arm][mode], counts)

    statistics = preregistration["statistics"]
    paired_stats = paired_cluster_statistics(
        dict(clusters),
        bootstrap_draws=int(statistics["bootstrap_draws"]),
        bootstrap_seed=int(statistics["bootstrap_seed"]),
        randomization_draws=int(statistics["randomization_draws"]),
        randomization_seed=int(statistics["randomization_seed"]),
    )
    metrics = {arm: {name: _metric(value) for name, value in arm_totals.items()} for arm, arm_totals in totals.items()}
    for arm in supplied:
        metrics[arm]["bmf1"] = 0.5 * metrics[arm]["singleturn"]["f1"] + 0.5 * metrics[arm]["multiturn"]["f1"]
        metrics[arm]["parse_failure_rate"] = parse_failures[arm] / len(gold) if gold else 1.0

    deltas = {
        name: metrics["candidate"][name]["f1"] - metrics["baseline"][name]["f1"]
        for name in ("singleturn", "multiturn", "preference", "nonpreference")
    }
    deltas["bmf1"] = metrics["candidate"]["bmf1"] - metrics["baseline"]["bmf1"]
    deltas["parse_failure_rate"] = metrics["candidate"]["parse_failure_rate"] - metrics["baseline"]["parse_failure_rate"]
    guard = preregistration["guardrails"]
    pass_spec = preregistration["pass"]
    gates = {
        "minimum_delta_bmf1": deltas["bmf1"] >= float(pass_spec["minimum_delta_bmf1"]),
        "bootstrap_ci_lower_positive": paired_stats["bootstrap_ci_95"][0] > 0.0,
        "randomization_p": paired_stats["randomization_p_one_sided"] < float(pass_spec["maximum_p_value_exclusive"]),
        "each_task_delta": all(deltas[name] >= float(guard["minimum_each_task_delta_f1"]) for name in ("singleturn", "multiturn")),
        "preference_drop": deltas["preference"] >= -float(guard["maximum_preference_f1_drop"]),
        "nonpreference_drop": deltas["nonpreference"] >= -float(guard["maximum_nonpreference_f1_drop"]),
        "parse_failure_increase": deltas["parse_failure_rate"] <= float(guard["maximum_parse_failure_rate_increase"]),
        "coverage": all(value == float(guard["required_coverage"]) for value in coverage.values()),
        "equal_action_budget": budget_equal,
    }
    summary = {
        "schema_version": 1,
        "case_count": len(gold),
        "coverage": coverage,
        "equal_action_budget": budget_equal,
        "metrics": metrics,
        "deltas": deltas,
        "paired_statistics": paired_stats,
        "gates": gates,
        "pass": all(gates.values()),
        "provenance": provenance,
        "input_hashes": dict(input_hashes or {}),
    }
    return summary


def evaluate_paired(
    *,
    root: str | Path,
    baseline_path: str | Path,
    candidate_path: str | Path,
    output: str | Path,
    preregistration: str | Path,
    final_attempt_id: str,
    runner_nonce: bytes | None = None,
    gold_open_fd: int | None = None,
) -> dict[str, Any]:
    """Official evaluator capability; a committed matching attempt is mandatory."""
    root = Path(root).resolve()
    attempt, implementation, _ = validate_final_attempt_v2(root)
    if attempt["attempt_id"] != final_attempt_id:
        raise ValueError("evaluator attempt ID does not match the sealed final attempt")
    enforce_final_evaluation_arguments(
        root,
        preregistration=Path(preregistration),
        baseline=Path(baseline_path),
        candidate=Path(candidate_path),
        output=Path(output),
    )
    provenance = validate_run_dag(
        root,
        final_attempt_id,
        Path(baseline_path),
        Path(candidate_path),
    )

    # This is deliberately before any gold row is opened or any metric slice is built.
    preference_slots = validate_preference_slots(root)
    preference_path = root / "configs/preference_slots.json"
    if sha256_file(preference_path) != implementation["preference_slots_sha256"]:
        raise ValueError("preference-slot config no longer matches the immutable seal")

    prereg = load_json(preregistration)
    sealed = load_json(root / "evaluator_vault/sealed_manifest.json")
    gold_path = root / "evaluator_vault/gold.jsonl"
    tasks_path = root / "artifacts/tasks.jsonl"
    if sha256_file(preregistration) != sealed["preregistration_sha256"]:
        raise ValueError("preregistration hash no longer matches sealed manifest")
    if sha256_file(tasks_path) != sealed["task_sha256"]:
        raise ValueError("task hash mismatch")

    if runner_nonce is None or gold_open_fd is None:
        raise ValueError("v2 evaluator requires both anonymous-pipe capabilities")
    verify_runner_nonce(attempt, runner_nonce)
    stages = validate_stage_chain(root)
    if list(stages) != ["runtime", "pre_gold"]:
        raise ValueError("gold evaluator requires the sealed pre_gold checkpoint")
    pre_gold_outputs = stages["pre_gold"]["payload"]["official_outputs"]
    prediction_rows: dict[str, list[dict[str, Any]]] = {}
    prediction_hashes: dict[str, str] = {}
    for arm, path in {
        "baseline": Path(baseline_path),
        "candidate": Path(candidate_path),
    }.items():
        payload, _identity = open_regular_bytes_once(path)
        digest = sha256_bytes(payload)
        expected = pre_gold_outputs[arm]
        if (
            expected["path"] != str(path.resolve().relative_to(root))
            or expected["sha256"] != digest
        ):
            raise ValueError(f"{arm} prediction bytes differ from pre_gold receipt")
        prediction_rows[arm] = parse_jsonl_bytes(
            payload, f"sealed {arm} predictions"
        )
        prediction_hashes[arm] = digest
    consume_gold_open_capability(gold_open_fd, stages["runtime"]["payload"])
    commit_stage_once(
        root,
        final_attempt_id,
        "gold_open",
        {
            "declared_gold_sha256": sealed["gold_sha256"],
            "pre_gold_record_sha256": stages["pre_gold"]["record_sha256"],
            "capability_transport": "anonymous_pipe_fd",
            "capability_secret_persisted": False,
            "capability_digest_persisted": False,
        },
        runner_nonce,
    )

    # The hash and parser below consume the exact bytes from one O_NOFOLLOW FD.
    gold_bytes, gold_stat = open_regular_bytes_once(gold_path)
    gold_sha256 = sha256_bytes(gold_bytes)
    if gold_sha256 != sealed["gold_sha256"]:
        raise ValueError("sealed gold hash mismatch")
    gold_rows = parse_jsonl_bytes(gold_bytes, "sealed gold")
    baseline_rows = prediction_rows["baseline"]
    candidate_rows = prediction_rows["candidate"]
    summary = compute_paired_summary(
        gold_rows=gold_rows,
        baseline_rows=baseline_rows,
        candidate_rows=candidate_rows,
        preference_slots=preference_slots,
        preregistration=prereg,
        provenance=provenance,
        input_hashes={
            "gold": gold_sha256,
            "baseline_predictions": prediction_hashes["baseline"],
            "candidate_predictions": prediction_hashes["candidate"],
        },
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
    write_json_once(output, summary)
    evidence = {
        "schema_version": 1,
        "kind": "single_fd_dual_evaluator_evidence",
        "attempt_id": final_attempt_id,
        "gold_fd_identity": {
            "device": int(gold_stat.st_dev),
            "inode": int(gold_stat.st_ino),
            "size": int(gold_stat.st_size),
            "sha256": gold_sha256,
            "opened_once": True,
            "nofollow": True,
        },
        "registered_summary_sha256": sha256_file(output),
        "independence_boundary": INDEPENDENCE_BOUNDARY,
        "independent_audit": independent,
        "agreement": True,
        "axes": {
            "integrity": "PASS",
            "execution": "COMPLETE",
            "performance": "PASS" if independent["pass"] else "FAIL",
        },
    }
    write_json_once(root / EVALUATION_EVIDENCE, evidence)
    return summary
