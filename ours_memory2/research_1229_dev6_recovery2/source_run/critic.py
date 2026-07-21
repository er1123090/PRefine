#!/usr/bin/env python3
"""Independent fail-closed auditor for the ECPR package and final paired run."""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import stat
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ecpr.contracts import FORBIDDEN_KEYS, GOLD_FIELDS, PREDICTION_FIELDS, TASK_FIELDS  # noqa: E402
from ecpr.firewall import validate_sanitized_history  # noqa: E402
from ecpr.final_protocol import (  # noqa: E402
    EVALUATION_EVIDENCE,
    classify_protocol_state,
    commit_stage_once,
    open_regular_bytes_once,
    parse_jsonl_bytes,
    parse_json_object_bytes,
    read_secret_fd,
    validate_stage_chain,
)
from ecpr.independent_audit import INDEPENDENCE_BOUNDARY, assert_registered_agreement  # noqa: E402
from ecpr.io import canonical_json, iter_jsonl, load_json, sha256_bytes, sha256_file, unique_by  # noqa: E402
from ecpr.integrity import (  # noqa: E402
    ATTEMPT_LOCK,
    EXPECTED_RUNTIME_CONTRACT_V3,
    IMPLEMENTATION_MANIFEST,
    LATENT_TRAIT_ONTOLOGY_V3,
    R3_ACTION_PROMPT_AMENDMENT,
    R3_METHOD_SCOPE_CLARIFICATION,
    TARGET_FREE_VLT3_REPORT,
    VLT3_AUDIT_AMENDMENT,
    validate_expected_runtime_contract,
    validate_final_attempt,
    validate_frozen_external_inputs,
    validate_implementation_manifest,
    validate_preference_slots,
    validate_registration_chain,
)
from ecpr.latent_firewall import (  # noqa: E402
    ACTIVE_CANDIDATE_REVISION,
    VLT2_CANDIDATE_REVISION,
    validate_latent_attestation,
)
from ecpr.ledger import (  # noqa: E402
    PROVIDER_JOURNAL,
    final_paths,
    validate_journal_records,
    validate_run_dag,
)
from ecpr.parsing import normalize_value, slot_value_map  # noqa: E402
from ecpr.prompts import candidate_memory_block  # noqa: E402


class AuditFailure(RuntimeError):
    pass


Count = tuple[int, int, int]
_RAW_CALL = re.compile(r"\b[A-Za-z_]\w*\s*\([^()]*=.+\)")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditFailure(message)


def _walk_strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _walk_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_strings(child)


def _json_scalar_type(value: Any) -> str:
    if value is None:
        return "null"
    if type(value) is bool:
        return "boolean"
    if type(value) is int:
        return "integer"
    if type(value) is float:
        return "number"
    if type(value) is str:
        return "string"
    raise AuditFailure("typed source literal must be an exact JSON scalar")


def _source_audit(root: Path = ROOT) -> dict[str, Any]:
    files = sorted((root / "ecpr").glob("*.py")) + [
        root / "critic.py",
        root / "run_gpu.py",
        root / "target_free_vlt3_diagnostics.py",
    ]
    require(files, "no implementation files found")
    for path in files:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            require(
                not any("ours_memory" in module or "experiments5" in module for module in modules),
                f"external method dependency imported by {path.name}",
            )
    inference_tree = ast.parse(
        (root / "ecpr/inference.py").read_text(encoding="utf-8")
    )
    literals = {
        node.value.casefold()
        for node in ast.walk(inference_tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    require(
        not any("evaluator_vault/" in value for value in literals),
        "inference contains evaluator vault path",
    )

    prompts_tree = ast.parse(
        (root / "ecpr/prompts.py").read_text(encoding="utf-8")
    )
    constructors = [
        node
        for node in prompts_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "build_action_prompt"
    ]
    require(len(constructors) == 1, "action prompt constructor is not unique")
    constructor = constructors[0]
    require(
        len(constructor.args.args) == 3
        and not constructor.args.kwonlyargs
        and constructor.args.vararg is None
        and constructor.args.kwarg is None,
        "action prompt constructor exposes an alternate arm-specific path",
    )
    action_tree = ast.parse(
        (root / "ecpr/action_case.py").read_text(encoding="utf-8")
    )
    prompt_calls = [
        node
        for node in ast.walk(action_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "build_action_prompt"
    ]
    require(
        len(prompt_calls) == 2
        and all(len(call.args) == 3 and not call.keywords for call in prompt_calls),
        "both arms do not use the same three-argument action prompt",
    )
    replay_source = (root / "ecpr/replay.py").read_text(encoding="utf-8")
    require(
        "_mint_replay_capability" not in replay_source,
        "raw replay surface can mint live authority",
    )

    raw_literal = "EcoNoMy"
    projected = candidate_memory_block(
        [
            {
                "domain": "GetRestaurants",
                "slot": "price_range",
                "value": normalize_value(raw_literal),
                "value_type": "string",
                "source_literal": {"type": "string", "value": raw_literal},
                "support": 2,
                "counterevidence": 0,
                "confidence": 1.0,
                "last_seen": 1,
                "provenance": [],
            }
        ],
        384,
        selected_domain="GetRestaurants",
    )
    require(
        "source_literal" not in projected
        and "value_type" not in projected
        and raw_literal not in projected,
        "raw typed source literal reaches the candidate action prompt",
    )

    symlinks = [
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_symlink()
    ]
    require(
        not symlinks,
        f"symlinks are not allowed in standalone package: {symlinks}",
    )
    return {
        "implementation_files": len(files),
        "external_method_imports": 0,
        "symlinks": 0,
        "shared_action_prompt_skeleton": True,
        "raw_source_literal_prompt_exposure": 0,
        "official_ledger_only_replay_authority": True,
    }


def _prereg_audit(root: Path = ROOT) -> dict[str, Any]:
    prereg_path = root / "preregistration.json"
    prereg = load_json(prereg_path)
    require(
        prereg.get("status") == "locked_before_target_metrics",
        "preregistration is not locked",
    )
    declarations = validate_frozen_external_inputs(root, prereg)
    require(
        all(
            value.get("runtime_dependency") is False
            and value.get("external_file_opened") is False
            for value in declarations.values()
        ),
        "raw external input was treated as a runtime dependency",
    )
    return {
        "preregistration_sha256": sha256_file(prereg_path),
        "frozen_input_declarations": len(declarations),
        "external_files_opened": 0,
    }


def static_audit(root: Path = ROOT) -> dict[str, Any]:
    result = {"source": _source_audit(root), "preregistration": _prereg_audit(root)}
    history = root / "artifacts/history.sanitized.jsonl"
    if history.exists():
        result["sanitized_history_records"] = validate_sanitized_history(history)
    tasks = root / "artifacts/tasks.jsonl"
    if tasks.exists():
        rows = unique_by(iter_jsonl(tasks), "case_key")
        for row in rows.values():
            require(set(row) == TASK_FIELDS, "task field contract violation")
            require(not ({str(key).casefold() for key in row} & FORBIDDEN_KEYS), "forbidden task field")
        result["task_records"] = len(rows)
    memory = root / "artifacts/memory.ecpr.jsonl"
    if memory.exists():
        rows = unique_by(iter_jsonl(memory), "example_id")
        for row in rows.values():
            require(
                set(row)
                == {
                    "example_id",
                    "latent_abstraction",
                    "latent_attestation",
                    "typed_hypotheses",
                    "method",
                    "candidate_revision",
                },
                "memory field contract violation",
            )
            revision = row["candidate_revision"]
            require(
                revision in {VLT2_CANDIDATE_REVISION, ACTIVE_CANDIDATE_REVISION},
                "memory revision mismatch",
            )
            validate_latent_attestation(
                row["latent_abstraction"], row["latent_attestation"]
            )
            require("api_calls" not in row and "sessions" not in row, "raw history embedded in candidate memory")
            for hypothesis in row["typed_hypotheses"]:
                if revision == ACTIVE_CANDIDATE_REVISION:
                    require(
                        set(hypothesis)
                        == {
                            "domain",
                            "slot",
                            "value",
                            "value_type",
                            "source_literal",
                            "support",
                            "counterevidence",
                            "confidence",
                            "last_seen",
                            "provenance",
                        },
                        "VLT3 typed hypothesis field contract violation",
                    )
                    source_literal = hypothesis["source_literal"]
                    require(
                        isinstance(source_literal, dict)
                        and set(source_literal) == {"type", "value"},
                        "VLT3 source literal field contract violation",
                    )
                    literal_type = _json_scalar_type(source_literal["value"])
                    require(
                        hypothesis["value_type"]
                        == source_literal["type"]
                        == literal_type,
                        "VLT3 source literal exact JSON type mismatch",
                    )
                    require(
                        hypothesis["value"]
                        == normalize_value(source_literal["value"]),
                        "VLT3 normalized value/source literal mismatch",
                    )
                else:
                    require(
                        set(hypothesis)
                        == {
                            "domain",
                            "slot",
                            "value",
                            "support",
                            "counterevidence",
                            "confidence",
                            "last_seen",
                            "provenance",
                        },
                        "legacy VLT2 typed hypothesis field contract violation",
                    )
                for provenance in hypothesis["provenance"]:
                    require(set(provenance) == {"session_index", "call_index", "call_digest"}, "provenance leaks raw evidence")
            require(not any(_RAW_CALL.search(text) for text in _walk_strings(row)), "raw API call leaked into candidate memory")
        result["memory_records"] = len(rows)
    for manifest_path in sorted((root / "artifacts").glob("predictions.*.jsonl.manifest.json")):
        payload = manifest_path.read_text(encoding="utf-8").casefold()
        require("evaluator_vault" not in payload and "reference_ground_truth" not in payload, "inference manifest leaks evaluator paths")
        require("/ours_memory/" not in payload, "inference manifest depends on ours_memory")
    return result


def _add(left: Count, right: Count) -> Count:
    return left[0] + right[0], left[1] + right[1], left[2] + right[2]


def _f1(value: Count) -> float:
    denominator = 2 * value[0] + value[1] + value[2]
    return 2 * value[0] / denominator if denominator else 0.0


def _counts(gt: dict[tuple[str, str], set[str]], pred: dict[tuple[str, str], set[str]]) -> Count:
    tp = sum(1 for key, allowed in gt.items() if pred.get(key, set()) & allowed)
    fn = sum(1 for key, allowed in gt.items() if not (pred.get(key, set()) & allowed))
    fp = sum(1 for key, values in pred.items() if key not in gt or not (values & gt[key]))
    return tp, fp, fn


def _filter_slots(
    mapping: dict[tuple[str, str], set[str]],
    preference_slots: dict[str, list[str]],
    preference: bool,
) -> dict[tuple[str, str], set[str]]:
    return {
        key: values
        for key, values in mapping.items()
        if ((key[1] in preference_slots.get(key[0], [])) is preference)
    }


def _metric(value: Count) -> dict[str, Any]:
    tp, fp, fn = value
    return {
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": _f1(value),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _bmf1(value: dict[str, Count]) -> float:
    return 0.5 * _f1(value["singleturn"]) + 0.5 * _f1(value["multiturn"])


def _aggregate(clusters: dict[str, dict[str, dict[str, Count]]], ids: list[str], arm: str) -> dict[str, Count]:
    result = {"singleturn": (0, 0, 0), "multiturn": (0, 0, 0)}
    for cluster_id in ids:
        for mode in result:
            result[mode] = _add(result[mode], clusters[cluster_id][arm][mode])
    return result


def _independent_stats(clusters: dict[str, dict[str, dict[str, Count]]], spec: dict[str, Any]) -> dict[str, Any]:
    ids = sorted(clusters)
    observed = _bmf1(_aggregate(clusters, ids, "candidate")) - _bmf1(_aggregate(clusters, ids, "baseline"))
    draws = int(spec["bootstrap_draws"])
    rng = random.Random(int(spec["bootstrap_seed"]))
    boot: list[float] = []
    for _ in range(draws):
        sample = [ids[rng.randrange(len(ids))] for _ in ids]
        boot.append(_bmf1(_aggregate(clusters, sample, "candidate")) - _bmf1(_aggregate(clusters, sample, "baseline")))
    boot.sort()
    random_draws = int(spec["randomization_draws"])
    rng = random.Random(int(spec["randomization_seed"]))
    extreme = 0
    for _ in range(random_draws):
        totals = {
            arm: {mode: (0, 0, 0) for mode in ("singleturn", "multiturn")}
            for arm in ("baseline", "candidate")
        }
        for cluster_id in ids:
            swap = bool(rng.getrandbits(1))
            for mode in ("singleturn", "multiturn"):
                baseline = clusters[cluster_id]["baseline"][mode]
                candidate = clusters[cluster_id]["candidate"][mode]
                if swap:
                    baseline, candidate = candidate, baseline
                totals["baseline"][mode] = _add(totals["baseline"][mode], baseline)
                totals["candidate"][mode] = _add(totals["candidate"][mode], candidate)
        if _bmf1(totals["candidate"]) - _bmf1(totals["baseline"]) >= observed - 1e-15:
            extreme += 1
    return {
        "delta": observed,
        "lower": boot[max(0, int(0.025 * draws))],
        "upper": boot[min(draws - 1, int(0.975 * draws))],
        "p": (extreme + 1) / (random_draws + 1),
    }


def _close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def _axes(
    integrity_status: str,
    execution_status: str,
    performance_status: str,
    *,
    integrity: dict[str, Any] | None = None,
    execution: dict[str, Any] | None = None,
    performance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "integrity": {"status": integrity_status, **(integrity or {})},
        "execution": {"status": execution_status, **(execution or {})},
        "performance": {"status": performance_status, **(performance or {})},
    }


def classify_completed_summary(
    *,
    static: dict[str, Any],
    strict: dict[str, Any],
    reported_pass: Any,
    recomputed_gates: dict[str, bool],
    mismatch: str | None = None,
) -> dict[str, Any]:
    recomputed_pass = all(recomputed_gates.values())
    performance = {
        "reported_pass": reported_pass,
        "recomputed_pass": recomputed_pass,
        "gates": recomputed_gates,
    }
    if mismatch is not None or not isinstance(reported_pass, bool) or reported_pass != recomputed_pass:
        return _axes(
            "FAIL",
            "COMPLETE",
            "PASS" if recomputed_pass else "FAIL",
            integrity={"reason": mismatch or "reported pass differs from independently recomputed gates"},
            execution={"detail": strict},
            performance=performance,
        )
    return _axes(
        "PASS",
        "COMPLETE",
        "PASS" if recomputed_pass else "FAIL",
        integrity={"static": static, "strict": strict},
        execution={"detail": strict},
        performance=performance,
    )


def _load_gold(path: Path) -> dict[str, dict[str, Any]]:
    return unique_by(iter_jsonl(path), "case_key")


def _completed_audit(
    root: Path,
    static: dict[str, Any],
    attempt: dict[str, Any],
    implementation: dict[str, Any],
    gold_loader,
) -> dict[str, Any]:
    required = {
        "gold": root / "evaluator_vault/gold.jsonl",
        "sealed": root / "evaluator_vault/sealed_manifest.json",
        "baseline": root / "artifacts/predictions.baseline.jsonl",
        "candidate": root / "artifacts/predictions.candidate.jsonl",
        "summary": root / "reports/summary.json",
        "memory": root / "artifacts/memory.ecpr.jsonl",
    }
    for name, path in required.items():
        require(path.is_file(), f"strict audit requires {name}: {path}")
    provenance = validate_run_dag(
        root,
        attempt["attempt_id"],
        required["baseline"],
        required["candidate"],
    )
    require(provenance["equal_action_budget"], "authoritative action contracts differ")
    preference_slots = validate_preference_slots(root)
    require(
        sha256_file(root / "configs/preference_slots.json")
        == implementation["preference_slots_sha256"],
        "preference-slot seal mismatch",
    )
    prereg = load_json(root / "preregistration.json")
    sealed = load_json(required["sealed"])
    require(sealed["preregistration_sha256"] == sha256_file(root / "preregistration.json"), "sealed preregistration hash mismatch")
    require(sealed["gold_sha256"] == sha256_file(required["gold"]), "sealed gold hash mismatch")

    gold = gold_loader(required["gold"])
    for row in gold.values():
        require(set(row) == GOLD_FIELDS, "sealed gold field contract violation")
    predictions: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in ("baseline", "candidate"):
        rows = unique_by(iter_jsonl(required[arm]), "case_key")
        require(set(rows) == set(gold), f"{arm} coverage is not exactly 100%")
        for row in rows.values():
            require(set(row) == PREDICTION_FIELDS, f"{arm} prediction field contract violation")
            require(row["arm"] == arm, f"{arm} label mismatch")
            require(not ({str(key).casefold() for key in row} & FORBIDDEN_KEYS), "forbidden prediction field")
        manifest = load_json(str(required[arm]) + ".manifest.json")
        require(
            manifest["predictions"]["sha256"] == sha256_file(required[arm]),
            f"{arm} manifest hash mismatch",
        )
        predictions[arm] = rows

    totals = {
        arm: {name: (0, 0, 0) for name in ("singleturn", "multiturn", "preference", "nonpreference")}
        for arm in ("baseline", "candidate")
    }
    clusters = defaultdict(
        lambda: {
            arm: {mode: (0, 0, 0) for mode in ("singleturn", "multiturn")}
            for arm in ("baseline", "candidate")
        }
    )
    parse_failures = {"baseline": 0, "candidate": 0}
    for case_key, target in gold.items():
        gt = slot_value_map(target["reference_ground_truth"])
        baseline = predictions["baseline"][case_key]
        candidate = predictions["candidate"][case_key]
        for arm, row in (("baseline", baseline), ("candidate", candidate)):
            value = row["llm_output"] if row["status"] == "ok" else ""
            pred = slot_value_map(value)
            if not pred:
                parse_failures[arm] += 1
            count = _counts(gt, pred)
            mode = target["mode"]
            totals[arm][mode] = _add(totals[arm][mode], count)
            totals[arm]["preference"] = _add(
                totals[arm]["preference"],
                _counts(
                    _filter_slots(gt, preference_slots, True),
                    _filter_slots(pred, preference_slots, True),
                ),
            )
            totals[arm]["nonpreference"] = _add(
                totals[arm]["nonpreference"],
                _counts(
                    _filter_slots(gt, preference_slots, False),
                    _filter_slots(pred, preference_slots, False),
                ),
            )
            clusters[str(target["example_id"])][arm][mode] = _add(clusters[str(target["example_id"])][arm][mode], count)

    stats = _independent_stats(dict(clusters), prereg["statistics"])
    summary = load_json(required["summary"])
    metrics = {
        arm: {name: _metric(value) for name, value in arm_totals.items()}
        for arm, arm_totals in totals.items()
    }
    for arm in ("baseline", "candidate"):
        metrics[arm]["bmf1"] = 0.5 * metrics[arm]["singleturn"]["f1"] + 0.5 * metrics[arm]["multiturn"]["f1"]
        metrics[arm]["parse_failure_rate"] = parse_failures[arm] / len(gold) if gold else 1.0
    deltas = {
        name: metrics["candidate"][name]["f1"] - metrics["baseline"][name]["f1"]
        for name in ("singleturn", "multiturn", "preference", "nonpreference")
    }
    deltas["bmf1"] = metrics["candidate"]["bmf1"] - metrics["baseline"]["bmf1"]
    deltas["parse_failure_rate"] = metrics["candidate"]["parse_failure_rate"] - metrics["baseline"]["parse_failure_rate"]
    guard = prereg["guardrails"]
    pass_spec = prereg["pass"]
    recomputed_gates = {
        "minimum_delta_bmf1": deltas["bmf1"] >= float(pass_spec["minimum_delta_bmf1"]),
        "bootstrap_ci_lower_positive": stats["lower"] > 0.0,
        "randomization_p": stats["p"] < float(pass_spec["maximum_p_value_exclusive"]),
        "each_task_delta": all(deltas[name] >= float(guard["minimum_each_task_delta_f1"]) for name in ("singleturn", "multiturn")),
        "preference_drop": deltas["preference"] >= -float(guard["maximum_preference_f1_drop"]),
        "nonpreference_drop": deltas["nonpreference"] >= -float(guard["maximum_nonpreference_f1_drop"]),
        "parse_failure_increase": deltas["parse_failure_rate"] <= float(guard["maximum_parse_failure_rate_increase"]),
        "coverage": all(value == float(guard["required_coverage"]) for value in summary.get("coverage", {}).values()),
        "equal_action_budget": provenance["equal_action_budget"],
    }
    mismatch: str | None = None
    try:
        require(summary["case_count"] == len(gold), "summary case count mismatch")
        require(summary.get("equal_action_budget") is provenance["equal_action_budget"], "summary action-budget mismatch")
        require(summary.get("provenance") == provenance, "summary DAG provenance mismatch")
        for arm in ("baseline", "candidate"):
            for name in ("singleturn", "multiturn", "preference", "nonpreference"):
                reported = summary["metrics"][arm][name]
                expected = metrics[arm][name]
                require((reported["tp"], reported["fp"], reported["fn"]) == totals[arm][name], f"{arm}/{name} counts mismatch")
                for field in ("precision", "recall", "f1"):
                    require(_close(reported[field], expected[field]), f"{arm}/{name}/{field} mismatch")
            require(_close(summary["metrics"][arm]["bmf1"], metrics[arm]["bmf1"]), f"{arm} BMF1 mismatch")
            require(_close(summary["metrics"][arm]["parse_failure_rate"], metrics[arm]["parse_failure_rate"]), f"{arm} parse failure mismatch")
        for name, value in deltas.items():
            require(_close(summary["deltas"][name], value), f"{name} delta mismatch")
        reported_stats = summary["paired_statistics"]
        require(_close(reported_stats["delta_bmf1"], stats["delta"]), "delta BMF1 mismatch")
        require(_close(reported_stats["bootstrap_ci_95"][0], stats["lower"]), "bootstrap lower mismatch")
        require(_close(reported_stats["bootstrap_ci_95"][1], stats["upper"]), "bootstrap upper mismatch")
        require(_close(reported_stats["randomization_p_one_sided"], stats["p"]), "randomization p mismatch")
        require(summary.get("gates") == recomputed_gates, "reported gates differ from independent recomputation")
    except (AuditFailure, KeyError, TypeError, ValueError) as exc:
        mismatch = str(exc)
    strict = {"case_count": len(gold), "cluster_count": len(clusters), "provenance": provenance}
    return classify_completed_summary(
        static=static,
        strict=strict,
        reported_pass=summary.get("pass"),
        recomputed_gates=recomputed_gates,
        mismatch=mismatch,
    )


def _completed_evidence_audit(
    root: Path,
    static: dict[str, Any],
    attempt: dict[str, Any],
    implementation: dict[str, Any],
) -> dict[str, Any]:
    """Validate sealed single-FD evidence without reopening evaluator gold."""
    stages = validate_stage_chain(root)
    require(
        list(stages)
        in (
            ["runtime", "pre_gold", "gold_open", "result"],
            ["runtime", "pre_gold", "gold_open", "result", "critic"],
        ),
        "completed protocol chain is incomplete",
    )
    result = stages["result"]["payload"]
    require(result["execution_status"] == "COMPLETE", "result is not complete")
    summary_path = root / result["summary"]["path"]
    evidence_path = root / result["evaluation_evidence"]["path"]
    require(summary_path == root / "reports/summary.json", "result summary path mismatch")
    require(evidence_path == root / EVALUATION_EVIDENCE, "result evidence path mismatch")
    summary_bytes, _summary_stat = open_regular_bytes_once(summary_path)
    evidence_bytes, _evidence_stat = open_regular_bytes_once(evidence_path)
    summary_sha256 = sha256_bytes(summary_bytes)
    evidence_sha256 = sha256_bytes(evidence_bytes)
    require(summary_sha256 == result["summary"]["sha256"], "result summary digest mismatch")
    require(evidence_sha256 == result["evaluation_evidence"]["sha256"], "result evidence digest mismatch")

    dag = validate_run_dag(
        root,
        attempt["attempt_id"],
        root / "artifacts/predictions.baseline.jsonl",
        root / "artifacts/predictions.candidate.jsonl",
    )
    pre_gold = stages["pre_gold"]["payload"]
    require(pre_gold["run_dag"] == dag, "pre_gold DAG differs from independent replay")
    require(
        pre_gold["implementation_manifest_sha256"]
        == attempt["binding"]["implementation_manifest_sha256"],
        "pre_gold implementation seal mismatch",
    )
    require(
        pre_gold["runtime_record_sha256"] == stages["runtime"]["record_sha256"],
        "pre_gold runtime receipt mismatch",
    )
    for identity in pre_gold["official_outputs"].values():
        path = root / identity["path"]
        payload, _path_stat = open_regular_bytes_once(path)
        require(sha256_bytes(payload) == identity["sha256"], "pre_gold output digest mismatch")

    journal_seal = pre_gold["provider_journal_seal"]
    journal_path = root / journal_seal["path"]
    require(
        journal_path == root / PROVIDER_JOURNAL,
        "pre_gold provider-journal path mismatch",
    )
    journal_bytes, journal_stat = open_regular_bytes_once(journal_path)
    journal_records = validate_journal_records(
        parse_jsonl_bytes(journal_bytes, "sealed provider journal"),
        attempt["attempt_id"],
    )
    require(
        sha256_bytes(journal_bytes) == journal_seal["sha256"]
        and int(journal_stat.st_dev) == journal_seal["device"]
        and int(journal_stat.st_ino) == journal_seal["inode"]
        and int(journal_stat.st_size) == journal_seal["size"]
        and stat.S_IMODE(journal_stat.st_mode) == 0o400
        and journal_seal["mode"] == "0400",
        "provider-journal same-FD identity or 0400 seal mismatch",
    )
    require(
        len(journal_records)
        == journal_seal["record_count"]
        == dag["journal_record_count"]
        and journal_seal["final_sequence"]
        == dag["final_journal_sequence"]
        == journal_records[-1]["sequence"]
        and journal_seal["final_record_sha256"]
        == dag["final_journal_record_sha256"]
        == journal_records[-1]["record_sha256"],
        "provider-journal seal differs from the materialized run DAG",
    )

    summary = parse_json_object_bytes(summary_bytes, "sealed evaluator summary")
    evidence = parse_json_object_bytes(
        evidence_bytes, "sealed evaluator evidence"
    )
    require(
        set(evidence)
        == {
            "schema_version",
            "kind",
            "attempt_id",
            "gold_fd_identity",
            "registered_summary_sha256",
            "independence_boundary",
            "independent_audit",
            "agreement",
            "axes",
        },
        "dual-evaluator evidence field contract violation",
    )
    require(evidence["kind"] == "single_fd_dual_evaluator_evidence", "dual-evaluator evidence kind mismatch")
    require(evidence["attempt_id"] == attempt["attempt_id"], "dual-evaluator attempt mismatch")
    require(evidence["registered_summary_sha256"] == summary_sha256, "dual-evaluator summary binding mismatch")
    require(
        evidence["independence_boundary"] == INDEPENDENCE_BOUNDARY,
        "dual-evaluator independence boundary mismatch",
    )
    require(evidence["agreement"] is True, "registered/independent evaluators disagree")
    gold_identity = evidence["gold_fd_identity"]
    require(
        isinstance(gold_identity, dict)
        and set(gold_identity) == {"device", "inode", "size", "sha256", "opened_once", "nofollow"}
        and gold_identity["opened_once"] is True
        and gold_identity["nofollow"] is True,
        "single-FD gold evidence contract violation",
    )
    require(
        gold_identity["sha256"]
        == stages["gold_open"]["payload"]["declared_gold_sha256"]
        == attempt["binding"]["declared_gold_sha256"],
        "single-FD gold digest does not match the sealed declaration",
    )
    require(summary.get("input_hashes", {}).get("gold") == gold_identity["sha256"], "summary gold digest mismatch")
    require(summary.get("provenance") == dag, "summary DAG provenance mismatch")
    assert_registered_agreement(summary, evidence["independent_audit"])
    performance = "PASS" if summary.get("pass") is True else "FAIL"
    require(result["performance_status"] == performance, "result performance mismatch")
    require(
        evidence["axes"]
        == {"integrity": "PASS", "execution": "COMPLETE", "performance": performance},
        "dual-evaluator axes mismatch",
    )
    return _axes(
        "PASS",
        "COMPLETE",
        performance,
        integrity={"static": static, "gold_reopened_by_critic": False},
        execution={"attempt_id": attempt["attempt_id"], "dag": dag},
        performance={"reported_pass": summary["pass"], "gates": summary["gates"]},
    )


def unsealed_vlt_audit(root: Path = ROOT) -> dict[str, Any]:
    """Strict static checkpoint that never opens external, target, or metric data."""
    root = root.resolve()
    require(
        not (root / IMPLEMENTATION_MANIFEST).exists(),
        "unsealed VLT audit refuses an implementation seal",
    )
    require(
        not (root / ATTEMPT_LOCK).exists(),
        "unsealed VLT audit refuses a final-attempt lock",
    )
    source = _source_audit(root)
    chain = validate_registration_chain(root)
    preregistration = load_json(root / "preregistration.json")
    runtime = validate_expected_runtime_contract(root, preregistration)
    require(len(chain) == 9, "VLT3 append-only registration chain is incomplete")
    audit = chain[8]
    require(
        runtime.get("candidate_revision") == ACTIVE_CANDIDATE_REVISION,
        "active runtime candidate revision mismatch",
    )
    require(
        runtime.get("vlt3_audit_amendment_sha256")
        == sha256_file(root / VLT3_AUDIT_AMENDMENT),
        "active runtime does not bind the VLT3 amendment",
    )
    ontology_path = root / str(runtime.get("latent_trait_ontology_path", ""))
    require(
        runtime.get("latent_trait_ontology_sha256")
        == sha256_file(ontology_path)
        == audit.get("latent_trait_ontology_vlt3_sha256"),
        "active runtime/audit ontology binding mismatch",
    )
    require(
        ontology_path == root / LATENT_TRAIT_ONTOLOGY_V3
        and runtime.get("r3_method_scope_clarification_sha256")
        == sha256_file(root / R3_METHOD_SCOPE_CLARIFICATION)
        and runtime.get("r3_action_prompt_amendment_sha256")
        == sha256_file(root / R3_ACTION_PROMPT_AMENDMENT)
        and runtime.get("target_free_vlt3_report_sha256")
        == sha256_file(root / TARGET_FREE_VLT3_REPORT),
        "active runtime R3/VLT3 append-only binding mismatch",
    )
    return _axes(
        "PASS",
        "NOT_RUN",
        "NOT_RUN",
        integrity={
            "mode": "strict_unsealed_vlt3_static_only",
            "source": source,
            "candidate": chain[2]["candidate_id"],
            "candidate_revision": runtime["candidate_revision"],
            "safe_transfer_amendment_v1_sha256": runtime[
                "safe_transfer_amendment_sha256"
            ],
            "vlt3_audit_amendment_sha256": runtime[
                "vlt3_audit_amendment_sha256"
            ],
            "active_runtime_contract_sha256": sha256_file(
                root / EXPECTED_RUNTIME_CONTRACT_V3
            ),
            "r3_method_scope_clarification_sha256": runtime[
                "r3_method_scope_clarification_sha256"
            ],
            "r3_action_prompt_amendment_sha256": runtime[
                "r3_action_prompt_amendment_sha256"
            ],
            "target_free_vlt3_report_sha256": runtime[
                "target_free_vlt3_report_sha256"
            ],
            "active_ontology_sha256": runtime[
                "latent_trait_ontology_sha256"
            ],
            "external_inputs_opened": 0,
            "target_tasks_opened": 0,
            "sanitized_history_opened": 0,
            "gold_rows_opened": 0,
            "metrics_computed": 0,
            "seal": "ABSENT",
            "attempt_lock": "ABSENT",
        },
    )


def audit_axes(
    root: Path = ROOT,
    *,
    gold_loader=_load_gold,
    allow_pending_critic: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    try:
        static = static_audit(root)
        seal_state = "NOT_SEALED"
        if (root / IMPLEMENTATION_MANIFEST).is_file():
            validate_implementation_manifest(root)
            seal_state = "PASS"
    except Exception as exc:
        return _axes("FAIL", "NOT_RUN", "NOT_RUN", integrity={"reason": str(exc)})
    if not (root / ATTEMPT_LOCK).is_file():
        protocol = classify_protocol_state(root)
        if protocol["integrity"] != "PASS":
            return _axes("FAIL", "NOT_RUN", "NOT_RUN", integrity={"reason": protocol.get("reason"), "static": static})
        return _axes(
            "PASS",
            "NOT_RUN",
            "NOT_RUN",
            integrity={"static": static, "seal": seal_state},
        )
    try:
        attempt, implementation, _ = validate_final_attempt(root)
    except Exception as exc:
        return _axes("FAIL", "NOT_RUN", "NOT_RUN", integrity={"reason": str(exc), "static": static})
    if attempt.get("schema_version") == 2:
        try:
            protocol = classify_protocol_state(root)
        except Exception as exc:
            return _axes("FAIL", "CONSUMED_INCOMPLETE", "NOT_RUN", integrity={"reason": str(exc), "static": static})
        if protocol["execution"] == "CONSUMED_INCOMPLETE":
            stages = validate_stage_chain(root)
            if (
                allow_pending_critic
                and list(stages) == ["runtime", "pre_gold", "gold_open", "result"]
                and stages["result"]["payload"]["execution_status"] == "COMPLETE"
            ):
                try:
                    return _completed_evidence_audit(
                        root, static, attempt, implementation
                    )
                except Exception as exc:
                    return _axes(
                        "FAIL",
                        "COMPLETE",
                        "NOT_RUN",
                        integrity={"reason": str(exc), "static": static},
                    )
            return _axes(
                "PASS",
                "CONSUMED_INCOMPLETE",
                "NOT_RUN",
                integrity={"static": static, "seal": "PASS", "attempt_id": attempt["attempt_id"]},
            )
        if protocol["execution"] == "FAILED":
            return _axes(
                protocol["integrity"],
                "FAILED",
                "NOT_RUN",
                integrity={"static": static, "attempt_id": attempt["attempt_id"]},
            )
        try:
            return _completed_evidence_audit(root, static, attempt, implementation)
        except Exception as exc:
            return _axes("FAIL", "COMPLETE", "NOT_RUN", integrity={"reason": str(exc), "static": static})

    required_outputs = list(final_paths(root).values())
    if any(not path.is_file() for path in required_outputs):
        return _axes(
            "PASS",
            "NOT_RUN",
            "NOT_RUN",
            integrity={"static": static, "seal": "PASS", "attempt_id": attempt["attempt_id"]},
        )
    try:
        return _completed_audit(root, static, attempt, implementation, gold_loader)
    except Exception as exc:
        return _axes(
            "FAIL",
            "COMPLETE",
            "NOT_RUN",
            integrity={"reason": str(exc), "static": static},
        )


def critic_exit_code(
    axes: dict[str, Any], *, allow_not_run: bool = False
) -> int:
    if axes["integrity"]["status"] != "PASS":
        return 1
    execution = axes["execution"]["status"]
    if allow_not_run and execution == "NOT_RUN":
        return 0
    return 0 if (
        execution == "COMPLETE"
        and axes["performance"]["status"] == "PASS"
    ) else 1


def validate_committed_critic_receipt(
    root: Path, axes: dict[str, Any]
) -> dict[str, Any]:
    """Recompute and bind strict axes to the immutable critic/source receipt."""
    root = root.resolve()
    stages = validate_stage_chain(root)
    require("critic" in stages, "strict final audit requires a critic receipt")
    receipt = stages["critic"]["payload"]
    attempt, implementation, _implementation_sha256 = validate_final_attempt(root)
    source_path = root / "critic.py"
    source_identity = implementation.get("files", {}).get("critic.py")
    require(
        not source_path.is_symlink()
        and source_path.is_file()
        and isinstance(source_identity, dict)
        and receipt["implementation_manifest_sha256"]
        == attempt["binding"]["implementation_manifest_sha256"]
        and receipt["critic_source_sha256"] == sha256_file(source_path)
        and receipt["critic_source_sha256"] == source_identity.get("sha256"),
        "committed critic source/implementation binding mismatch",
    )
    require(
        receipt["audit_axes_sha256"]
        == sha256_bytes(canonical_json(axes).encode("utf-8")),
        "committed critic audit-axis digest mismatch",
    )
    return receipt


def strict_audit(*, allow_pending_critic: bool = False) -> dict[str, Any]:
    axes = audit_axes(ROOT, allow_pending_critic=allow_pending_critic)
    if (
        not allow_pending_critic
        and (ROOT / ATTEMPT_LOCK).is_file()
        and (ROOT / "artifacts/final_test.critic.json").is_file()
    ):
        validate_committed_critic_receipt(ROOT, axes)
    return axes


def commit_final_critic_receipt(
    root: Path,
    *,
    final_attempt_id: str,
    runner_nonce: bytes | bytearray,
    axes: dict[str, Any],
) -> dict[str, Any]:
    stages = validate_stage_chain(root)
    require(
        list(stages) == ["runtime", "pre_gold", "gold_open", "result"],
        "critic receipt requires one pending complete result",
    )
    result = stages["result"]
    result_payload = result["payload"]
    attempt, implementation, _implementation_sha256 = validate_final_attempt(
        root
    )
    require(
        attempt["attempt_id"] == final_attempt_id,
        "critic receipt final-attempt ID mismatch",
    )
    require(
        result_payload["execution_status"] == "COMPLETE",
        "critic receipt cannot follow a failed result",
    )
    critic_source_sha256 = sha256_file(root / "critic.py")
    require(
        implementation.get("files", {}).get("critic.py", {}).get("sha256")
        == critic_source_sha256,
        "critic source differs from the immutable implementation manifest",
    )
    audit_pass = (
        axes.get("integrity", {}).get("status") == "PASS"
        and axes.get("execution", {}).get("status") == "COMPLETE"
        and axes.get("performance", {}).get("status")
        == result_payload["performance_status"]
    )
    payload = {
        "kind": "strict_final_critic_receipt",
        "audit_status": "PASS" if audit_pass else "FAIL",
        "integrity_status": "PASS" if audit_pass else "FAIL",
        "execution_status": "COMPLETE",
        "performance_status": (
            result_payload["performance_status"] if audit_pass else "NOT_RUN"
        ),
        "result_record_sha256": result["record_sha256"],
        "implementation_manifest_sha256": attempt["binding"][
            "implementation_manifest_sha256"
        ],
        "critic_source_sha256": critic_source_sha256,
        "summary_sha256": result_payload["summary"]["sha256"],
        "evaluation_evidence_sha256": result_payload["evaluation_evidence"][
            "sha256"
        ],
        "audit_axes_sha256": sha256_bytes(
            canonical_json(axes).encode("utf-8")
        ),
        "detail": (
            "strict_critic_audit_passed"
            if audit_pass
            else "strict_critic_audit_failed_no_retry"
        ),
    }
    return commit_stage_once(
        root,
        final_attempt_id,
        "critic",
        payload,
        runner_nonce,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--unsealed-vlt", action="store_true")
    parser.add_argument("--commit-final-receipt", action="store_true")
    parser.add_argument("--final-attempt-id")
    parser.add_argument("--runner-nonce-fd", type=int)
    args = parser.parse_args()
    if args.unsealed_vlt and not args.strict:
        parser.error("--unsealed-vlt requires --strict")
    if args.commit_final_receipt and (
        not args.strict
        or args.unsealed_vlt
        or not args.final_attempt_id
        or args.runner_nonce_fd is None
    ):
        parser.error(
            "--commit-final-receipt requires --strict, --final-attempt-id, "
            "and --runner-nonce-fd without --unsealed-vlt"
        )
    try:
        result = (
            unsealed_vlt_audit()
            if args.unsealed_vlt
            else strict_audit(allow_pending_critic=args.commit_final_receipt)
            if args.strict
            else static_audit()
        )
    except Exception as exc:
        print(json.dumps({"audit": "FAIL", "reason": str(exc)}, sort_keys=True))
        return 1
    if args.strict:
        if args.commit_final_receipt:
            nonce = bytearray(
                read_secret_fd(args.runner_nonce_fd, "critic runner nonce")
            )
            try:
                commit_final_critic_receipt(
                    ROOT,
                    final_attempt_id=args.final_attempt_id,
                    runner_nonce=nonce,
                    axes=result,
                )
            finally:
                for index in range(len(nonce)):
                    nonce[index] = 0
        print(json.dumps(result, sort_keys=True))
        return critic_exit_code(result, allow_not_run=args.unsealed_vlt)
    print(json.dumps({"audit": "PASS", "result": result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
