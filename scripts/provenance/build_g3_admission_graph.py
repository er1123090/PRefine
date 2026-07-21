#!/usr/bin/env python3
"""Build the fail-closed paper-result -> aggregate -> raw admission graph.

This program is deliberately read-only with respect to experiments4/5/6.  It
reads authenticated G1 inventories and protected source aggregates, then writes
only new G3 artifacts under experiments7/manifests/admission.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.util
import itertools
import json
import os
import re
import sys
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


ROOT7 = Path("/data/minseo/experiments7")
EXP4 = Path("/data/minseo/experiments4")
SOURCE_PRE = ROOT7 / "manifests/source-pre.jsonl"
PAPER_RESULTS = ROOT7 / "paper_outputs/inventory/results.jsonl"
ALIASES = ROOT7 / "paper_outputs/inventory/aliases.jsonl"
REGISTRY = ROOT7 / "variants/registry.json"
CODE_LINEAGE = ROOT7 / "lineage/code.jsonl"
CONFIG_LINEAGE = ROOT7 / "lineage/config.jsonl"
CANDIDATE_JSON = Path("/data/minseo/.omx/tmp/g3-path-hash-candidate-map.json")
CANDIDATE_MD = Path("/data/minseo/.omx/tmp/g3-path-hash-candidate-map.md")
PROVENANCE_OUTPUT = ROOT7 / "paper_outputs/provenance"
DEFAULT_OUTPUT = ROOT7 / "paper_outputs/admission/precopy"
CONTRACT_INPUT_PATHS = {
    "inventory_structure": "paper_outputs/inventory/structure.jsonl",
    "inventory_results": "paper_outputs/inventory/results.jsonl",
    "inventory_aliases": "paper_outputs/inventory/aliases.jsonl",
    "provenance_nodes": "paper_outputs/provenance/nodes.jsonl",
    "provenance_edges": "paper_outputs/provenance/edges.jsonl",
    "provenance_recomputations": "paper_outputs/provenance/recomputations.jsonl",
    "source_pre": "manifests/source-pre.jsonl",
    "registry": "variants/registry.json",
    "lineage_code": "lineage/code.jsonl",
    "lineage_config": "lineage/config.jsonl",
}

CORE_HASHES = {
    SOURCE_PRE: "6ab9fd4b02a55b2c59889ee2034f42d8162bca99a6abbfc35a0d65ca02d63bbc",
    PAPER_RESULTS: "02dfad2e5787e994f22f15c3b1b1b405c1577e25506b8d1976a53f492ed865d9",
    ALIASES: "2b2e3281712dc2b858038cf93395e01fb32b74848e3c73c4a33542b5e4591841",
    REGISTRY: "7126e93f2bfdfff49f250f5d1308df0de736fa10fe54959b49f3a4582ce24b38",
    CODE_LINEAGE: "ce0d6a295fbc91f655b1a0abda439ec8d50e7645012b2ceb7a31089f6d44dc7a",
    CONFIG_LINEAGE: "71d64df6953be1ef359c06be57d4e3f300e6003a5d31f22e34a9e4c1615433ec",
    CANDIDATE_JSON: "76f3acaf34a62b801233c1fcccc8718ba7475070d6174889edb0184567226e80",
    CANDIDATE_MD: "133e10bfbed63217641225414282984923210073947b3ac0aaa2daa6a8e334d8",
}

TABLE_FILES = {
    "t3_witness": EXP4 / "_paper/appendix_table3_combined_with_sources.csv",
    "t11_witness": EXP4 / "_paper/table11_context_guided_base_prefine_gemma_gpt4o.csv",
    "t12_witness": EXP4 / "_paper/table12_context_guided_prefine_reasoning.csv",
    "t13_witness": EXP4 / "_paper/table13_context_free_prefine.csv",
    "t14_langmem_single": EXP4 / "langmem/inference_single/gpt-4o-mini/singleturn_metrics.csv",
    "t14_langmem_multi": EXP4 / "langmem/inference_multi_minimal/gpt-4o-mini/memory_api/multiturn_metrics.csv",
    "t14_mem0_single": EXP4 / "evaluation/eval_results/results_mem0_single_0309.csv",
    "t14_mem0_multi": EXP4 / "evaluation/eval_results/results_mem0_0309_multi_gpt5.csv",
    "t14_rag_single": EXP4 / "evaluation/eval_results/results_rag_gpt5_single2.csv",
    "t14_rag_multi": EXP4 / "evaluation/eval_results/results_rag_gpt5_multi2.csv",
    "f4_base_single": EXP4 / "evaluation/eval_results/results_slotcount_1229-3_single1.csv",
    "f4_base_multi": EXP4 / "evaluation/eval_results/results_slotcount_1230-1_multi1.csv",
    "f4_pref_single": EXP4 / "evaluation/eval_results/results_slotcount_1231_memory3_inference1_single.csv",
    "f4_pref_multi": EXP4 / "evaluation/eval_results/results_slotcount_1231_memory3_inference2_multi.csv",
    "f4_evaluator": EXP4 / "evaluation/evaluation_slot_count.py",
    "vanilla_multi": EXP4 / "evaluation/eval_results/results_vanillaLLM_1230-1_multi2_parse.csv",
    "memory_multi3": EXP4 / "evaluation/eval_results/results_memory_multiturn3.csv",
    "memory_multi4": EXP4 / "evaluation/eval_results/results_memory_multiturn4.csv",
    "memory_multi_gpt5": EXP4 / "evaluation/eval_results/results_memory_multiturn_0306_gpt5.csv",
    "memory_single2": EXP4 / "evaluation/eval_results/results_memory_singleturn2.csv",
    "memory_single_gpt5": EXP4 / "evaluation/eval_results/results_memory_singleturn_0306_gpt5.csv",
    "t15_summary": EXP4 / "ours_memory/iteration_cap_comparison/summary.md",
    "t15_refinement_by_model": EXP4 / "ours_memory/iteration_cap_comparison/refinement_by_model.csv",
    "t15_refinement_totals": EXP4 / "ours_memory/iteration_cap_comparison/refinement_totals.csv",
    "t15_multi_by_pref": EXP4 / "ours_memory/iteration_cap_comparison/multiturn_by_pref.csv",
    "t15_single_by_pref": EXP4 / "ours_memory/iteration_cap_comparison/singleturn_by_pref.csv",
    "t15_multi_common": EXP4 / "ours_memory/iteration_cap_comparison/multiturn_common_subset.csv",
    "t15_single_common": EXP4 / "ours_memory/iteration_cap_comparison/singleturn_common_subset.csv",
    "t15_multi_a_detail": EXP4 / "ours_memory/inference/0312_MEMORY1_0317_MEM-DIAG/analysis/multiturn_detailed.csv",
    "t15_single_a_detail": EXP4 / "ours_memory/inference/0312_MEMORY1_0317_MEM-DIAG/analysis/singleturn_detailed.csv",
    "t15_multi_b": EXP4 / "evaluation/eval_results/results_memory_multiturn2.csv",
    "t15_single_b": EXP4 / "evaluation/eval_results/results_memory_singleturn2.csv",
    "t15_builder": EXP4 / "ours_memory/compare_iteration_caps.py",
    "t16_single": EXP4 / "extended_schema/singleturn_comparison.csv",
    "t16_multi": EXP4 / "extended_schema/multiturn_comparison.csv",
    "t16_builder": EXP4 / "extended_schema/build_fixed_400_comparison_csvs.py",
}

EXPECTED_RECORD_IDS = {
    "t15_summary": "a2240612d54f2db3fa27ba82d09e18084c953eaa5b4005ad5b7bdb4f950c8357",
    "t15_refinement_by_model": "fae3001f38975618d1273499146f0b929513f915a99c3ecf67453d3ab3c3b05a",
    "t15_refinement_totals": "8be175fe594d2df55654fe1a6ff9ee205d902467ca27c9fe5c3bb2dd20a0b825",
    "t15_multi_a_detail": "0e0a67bfc8855a94088afa76f6d69c18f5bdae38f918811a4eb866d046e1f57a",
    "t15_single_a_detail": "d813744381f4a5b143f36046547455706d43969986e1b06f34b9e38ead7fa9b9",
    "t15_builder": "e96f7fe0c7edad2b26e0db66da95afcca95926b372e7892504e934f1b2b6e29a",
    "t16_single": "87871b6bc48b289295253ac0a8941fe18813cd6531ebbbc5e74dab7832164128",
    "t16_multi": "06e915dce3b25043a1eadbdad7478d4f3c85ff143b200749c47f66f02e3b7fba",
    "t16_builder": "aba3dfcbac1cd5665cda1c952715db63524992687261b0531414aae54bea345e",
}

MEMORY_NAME_MAP_A = {
    "Qwen_Qwen3-8B": "Qwen3-8B",
    "deepseek-ai_DeepSeek-R1-0528-Qwen3-8B": "DeepSeek-R1-0528-Qwen3-8B",
    "deepseek-ai_DeepSeek-R1-Distill-Llama-8B": "DeepSeek-R1-Distill-Llama-8B",
    "google_gemma-3-12b-it": "gemma-3-12b-it",
    "gpt-4o-mini": "gpt-4o-mini",
}
MEMORY_NAME_MAP_B_SINGLE = {
    "Qwen_Qwen3-8B": "Qwen3-8B",
    "deepseek-ai_DeepSeek-R1-0528-Qwen3-8B": "DeepSeek-R1-0528-Qwen3-8B",
    "deepseek-ai_DeepSeek-R1-Distill-Llama-8B": "DeepSeek-R1-Distill-Llama-8B",
    "google_gemma-3-12b-it": "gemma-3-12b-it",
}

RECOMPUTED_T13_ALIASES = {
    str(EXP4 / "ours_memory/inference/1231_MEMORY3_inference1_single/google_gemma-3-12b-it/memory_api/easy/gemini-3-flash-preview/implicit_zs/0102_test1.json"): {
        "alias_record_id": "7256690e94a024871c4da7173d98eb5d0c5c52bf1855db45454a076a81cf3f47",
        "expected": {"tp": "313", "fp": "113", "fn": "48", "precision": "0.7347417840375586", "recall": "0.8670360110803325", "f1": "0.795425667090216", "parsing_fail_count": "14"},
    },
    str(EXP4 / "ours_memory/inference/1231_MEMORY3_inference1_single/google_gemma-3-12b-it/memory_api/medium/gemini-3-flash-preview/implicit_zs/0102_test1.json"): {
        "alias_record_id": "e5a628f9636c79f1196073b267be210ef9e8be56b6aad5dd0c15905cd7c73aff",
        "expected": {"tp": "248", "fp": "236", "fn": "45", "precision": "0.512396694214876", "recall": "0.8464163822525598", "f1": "0.6383526383526383", "parsing_fail_count": "5"},
    },
    str(EXP4 / "ours_memory/inference/1231_MEMORY3_inference1_single/google_gemma-3-12b-it/memory_api/hard/gemini-3-flash-preview/implicit_zs/0102_test1.json"): {
        "alias_record_id": "6f8ef2076f821842f7f531145575587d88dddf3aaf7b14e5772fb4187b3c2070",
        "expected": {"tp": "173", "fp": "425", "fn": "299", "precision": "0.28929765886287623", "recall": "0.3665254237288136", "f1": "0.3233644859813084", "parsing_fail_count": "19"},
    },
}
T13_EVALUATOR_RECORD = "9a257b17b8920084c5f5e07ab826d3f86b18f404ee8baab17342e0a991a0ff5e"


class AdmissionError(RuntimeError):
    pass


def apply_protected_source_envelope() -> None:
    """Activate the canonical G0 read-only guard before protected-source I/O."""
    for directory in (PROVENANCE_OUTPUT, DEFAULT_OUTPUT):
        directory.mkdir(parents=True, exist_ok=True)
    provider_path = ROOT7 / "scripts/g0/provider.py"
    spec = importlib.util.spec_from_file_location("experiments7_g0_provider", provider_path)
    if spec is None or spec.loader is None:
        raise AdmissionError(f"cannot load protected-source provider: {provider_path}")
    provider = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(provider)
    provider.apply_readonly_envelope(["/tmp", str(PROVENANCE_OUTPUT), str(DEFAULT_OUTPUT)])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise AdmissionError(f"invalid JSONL {path}:{number}: {exc}") from exc
    return rows


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def decode_rel(record: Mapping[str, Any]) -> str:
    return base64.b64decode(record["relative_path_b64"], validate=True).decode("utf-8")


def public_record(record: Mapping[str, Any], rel: str | None = None) -> dict[str, Any]:
    return {
        "source_manifest_record_id": record["record_id"],
        "root_id": record["root_id"],
        "relative_path": rel if rel is not None else decode_rel(record),
        "sha256": record.get("sha256"),
        "size": record.get("size"),
        "type": record["type"],
    }


def abs_for_record(record: Mapping[str, Any], rel: str) -> str:
    roots = {"experiments4": "/data/minseo/experiments4", "experiments5": "/data/minseo/experiments5", "experiments6": "/data/minseo/experiments6"}
    return str(Path(roots[record["root_id"]]) / rel)


def source_index(records: Sequence[dict[str, Any]]) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, tuple[dict[str, Any], str]]]:
    by_path: dict[tuple[str, str], dict[str, Any]] = {}
    by_id: dict[str, tuple[dict[str, Any], str]] = {}
    for record in records:
        rel = decode_rel(record)
        by_path[(record["root_id"], rel)] = record
        by_id[record["record_id"]] = (record, rel)
    return by_path, by_id


def exp4_rel(path: str | Path) -> str:
    resolved = Path(path)
    try:
        return resolved.relative_to(EXP4).as_posix()
    except ValueError as exc:
        raise AdmissionError(f"protected source path escaped experiments4: {path}") from exc


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def d(value: Any) -> Decimal:
    text = str(value).strip().replace("%", "").lstrip("+")
    try:
        return Decimal(text)
    except InvalidOperation as exc:
        raise AdmissionError(f"not decimal: {value!r}") from exc


def rounded(value: Any, scale: Decimal, precision: int) -> Decimal:
    quantum = Decimal(1).scaleb(-precision)
    return (d(value) * scale).quantize(quantum, rounding=ROUND_HALF_UP)


def paper_decimal(result: Mapping[str, Any]) -> Decimal:
    return d(result["numeric_value"])


def norm(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def canonical_model(text: Any) -> str:
    value = norm(text)
    if "codegemma" in value:
        return "codegemma"
    if "gemini" in value:
        return "gemini"
    if "distillllama" in value:
        return "r1_llama"
    if "distillqwen" in value:
        return "r1_qwen"
    if "0528qwen3" in value:
        return "r1_qwen"
    if "gpt5mini" in value:
        return "gpt5mini"
    if "gpt5" in value:
        return "gpt5"
    if "gpt4o" in value:
        return "gpt4o"
    if "gemma" in value:
        return "gemma"
    if "qwen3" in value:
        return "qwen3"
    return value


def row_action(row: Mapping[str, str]) -> str:
    for key in ("model_name", "action_model_base", "action_model", "action_model_raw", "inference_model"):
        if row.get(key):
            return canonical_model(row[key])
    return ""


def row_context(row: Mapping[str, str]) -> str:
    return canonical_model(row.get("context", row.get("memory_model", "")))


def row_pref(row: Mapping[str, str]) -> str:
    pref_type = str(row.get("pref_type", "")).lower()
    if pref_type in {"easy", "medium", "hard"}:
        return pref_type
    for key in ("difficulty", "query_turns", "group_key", "json_path"):
        value = str(row.get(key, "")).lower()
        for token in ("easy", "medium", "hard"):
            if re.search(rf"(^|[^a-z]){token}([^a-z]|$)", value):
                return token
    return pref_type


def pref_code(preference: Any) -> str:
    return {"Recall": "easy", "Induction": "medium", "Transfer": "hard"}.get(str(preference), str(preference).lower())


def paper_stub(result: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("result_id", "evidence_id", "page", "section", "setting", "method", "model", "preference_type", "metric", "numeric_value", "displayed_value", "displayed_marker", "result_kind", "derived_from")
    return {key: result.get(key) for key in keys if key in result}


def new_admission(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "experiments7-paper-result-admission/v1",
        "paper_result": paper_stub(result),
        "status": "unresolved",
        "admission_scope": "none",
        "reason_codes": ["no_exact_content_provenance_binding"],
        "aggregate_bindings": [],
        "raw_bindings": [],
        "root_bindings": [],
        "supporting_evidence": [],
    }


def lineage_index(paths: Sequence[Path]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        for row in load_jsonl(path):
            index[row["source_manifest_record_id"]].append({
                "lineage_id": row["lineage_id"], "stage": row["stage"], "family": row["family"], "variant_id": row["variant_id"], "origin_sha256": row["origin_sha256"]
            })
    return index


def variant_labels(record: Mapping[str, Any], lineage: Mapping[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    exact = lineage.get(record["record_id"], [])
    if exact:
        return exact
    return [{
        "variant_id": f"unregistered:{record['root_id']}:{record['record_id'][:12]}",
        "stage": "unregistered",
        "family": "unregistered",
        "origin_sha256": record.get("sha256"),
        "lineage_id": None,
    }]


def make_aggregate(path: Path, source_by_path: Mapping[tuple[str, str], dict[str, Any]], lineage: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    rel = exp4_rel(path)
    record = source_by_path.get(("experiments4", rel))
    if record is None or record.get("type") != "regular":
        raise AdmissionError(f"aggregate absent from source-pre: {path}")
    actual = sha256_file(path)
    if actual != record.get("sha256"):
        raise AdmissionError(f"aggregate hash drift: {path}")
    rows = csv_rows(path)
    return {"path": str(path), "rel": rel, "record": record, "rows": rows, "variants": variant_labels(record, lineage)}


def candidates(aggregate: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [{"aggregate": aggregate, "row": row, "row_number": index + 2} for index, row in enumerate(aggregate["rows"])]


def metric_exact(cells: Sequence[Mapping[str, Any]], row: Mapping[str, str], mapping: Mapping[str, str], scale: Decimal = Decimal(100)) -> bool:
    for cell in cells:
        column = mapping.get(cell["metric"])
        if not column or not row.get(column):
            return False
        precision = int(cell.get("rounding_rule", {}).get("display_precision", 2))
        if rounded(row[column], scale, precision) != paper_decimal(cell):
            return False
    return True


def aggregate_binding(candidate: Mapping[str, Any], cells: Sequence[Mapping[str, Any]], mapping: Mapping[str, str], relation: str = "exact_rounded_metric_match", scale: Decimal = Decimal(100)) -> dict[str, Any]:
    aggregate = candidate["aggregate"]
    record = aggregate["record"]
    metric_columns = {cell["result_id"]: mapping[cell["metric"]] for cell in cells}
    values = {result_id: candidate["row"][column] for result_id, column in metric_columns.items()}
    return {
        "aggregate_node_id": f"aggregate:{record['record_id']}:row:{candidate['row_number']}",
        **public_record(record, aggregate["rel"]),
        "protected_path": aggregate["path"],
        "row_number": candidate["row_number"],
        "relation": relation,
        "metric_columns": metric_columns,
        "aggregate_values": values,
        "scale": str(scale),
        "variant_labels": aggregate["variants"],
    }


def resolve_raw(candidate: Mapping[str, Any], source_by_path: Mapping[tuple[str, str], dict[str, Any]], source_by_id: Mapping[str, tuple[dict[str, Any], str]]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    row = candidate["row"]
    raw_path = row.get("json_path") or row.get("output_path")
    if not raw_path:
        return None, None
    rel = exp4_rel(raw_path)
    record = source_by_path.get(("experiments4", rel))
    if record is not None and record.get("type") == "regular":
        return {"record": record, "rel": rel, "path": raw_path, "relation": "direct_json_path_reference"}, None
    assertion = RECOMPUTED_T13_ALIASES.get(raw_path)
    if assertion is None:
        return None, {"missing_path": raw_path, "reason": "direct_raw_path_absent_from_source_pre"}
    for key, expected in assertion["expected"].items():
        if str(row.get(key, "")) != expected:
            raise AdmissionError(f"T13 alias assertion aggregate metric drift for {raw_path}: {key}")
    alias_record, alias_rel = source_by_id[assertion["alias_record_id"]]
    evaluator, evaluator_rel = source_by_id[T13_EVALUATOR_RECORD]
    evidence = {
        "kind": "recomputed_exact_alias",
        "stale_aggregate_json_path": raw_path,
        "same_config_alias_path": abs_for_record(alias_record, alias_rel),
        "exact_recomputed_fields": assertion["expected"],
        "evaluator": public_record(evaluator, evaluator_rel),
        "evaluator_variant_id": "report4_session_memory",
    }
    return {"record": alias_record, "rel": alias_rel, "path": abs_for_record(alias_record, alias_rel), "relation": "recomputed_exact_alias", "recomputation_evidence": evidence}, None


def raw_binding(raw: Mapping[str, Any], via: Sequence[str]) -> dict[str, Any]:
    result = {
        "raw_node_id": f"raw:{raw['record']['record_id']}",
        **public_record(raw["record"], raw["rel"]),
        "protected_path": raw["path"],
        "relation": raw["relation"],
        "via_aggregate_node_ids": sorted(set(via)),
    }
    if raw.get("recomputation_evidence"):
        result["recomputation_evidence"] = raw["recomputation_evidence"]
    return result


def assign_exact_group(
    cells: Sequence[dict[str, Any]],
    pool: Sequence[dict[str, Any]],
    mapping: Mapping[str, str],
    predicate: Callable[[Mapping[str, str]], bool],
    admissions: Mapping[str, dict[str, Any]],
    source_by_path: Mapping[tuple[str, str], dict[str, Any]],
    source_by_id: Mapping[str, tuple[dict[str, Any], str]],
    scale: Decimal = Decimal(100),
) -> None:
    matched = [candidate for candidate in pool if predicate(candidate["row"]) and metric_exact(cells, candidate["row"], mapping, scale)]
    resolved: list[tuple[dict[str, Any], dict[str, Any]]] = []
    missing: list[dict[str, Any]] = []
    for candidate in matched:
        raw, issue = resolve_raw(candidate, source_by_path, source_by_id)
        if raw:
            resolved.append((candidate, raw))
        elif issue:
            missing.append(issue)
    by_raw: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for candidate, raw in resolved:
        by_raw[raw["record"]["record_id"]].append((candidate, raw))
    if len(by_raw) != 1:
        reason = "no_exact_aggregate_row" if not matched else "ambiguous_exact_raw_candidates" if len(by_raw) > 1 else "raw_path_not_source_pre_bound"
        for cell in cells:
            admission = admissions[cell["result_id"]]
            admission["reason_codes"] = [reason]
            admission["candidate_count"] = len(matched)
            admission["distinct_bound_raw_count"] = len(by_raw)
            if missing:
                admission["unbound_candidates"] = missing
        return
    selected = next(iter(by_raw.values()))
    bindings = [aggregate_binding(candidate, cells, mapping, scale=scale) for candidate, _ in selected]
    raw = selected[0][1]
    raw_ref = raw_binding(raw, [binding["aggregate_node_id"] for binding in bindings])
    status = "admitted_recomputed_exact_alias" if raw["relation"] == "recomputed_exact_alias" else "admitted_exact_raw"
    for cell in cells:
        admission = admissions[cell["result_id"]]
        admission.update({"status": status, "admission_scope": "raw_file", "reason_codes": [], "aggregate_bindings": bindings, "raw_bindings": [raw_ref]})


def grouped(rows: Iterable[dict[str, Any]], keys: Sequence[str]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    output: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        output[tuple(row.get(key) for key in keys)].append(row)
    return output


MULTI7 = {
    "P-EM": "pref_em", "EA-Precision": "nonpref_precision", "EA-Recall": "nonpref_recall", "EA-F1": "nonpref_f1",
    "OA-Precision": "overall_precision", "OA-Recall": "overall_recall", "OA-F1": "overall_f1",
}
SINGLE3 = {"Precision": "precision", "Recall": "recall", "F1": "f1"}
TABLE3_MULTI = {"P-EM": "pref_em", "EA-F1": "nonpref_f1", "OA-F1": "overall_f1"}


def bind_tables_11_13(results: Sequence[dict[str, Any]], admissions: Mapping[str, dict[str, Any]], aggs: Mapping[str, dict[str, Any]], source_by_path: Mapping[tuple[str, str], dict[str, Any]], source_by_id: Mapping[str, tuple[dict[str, Any], str]]) -> None:
    base_pool = candidates(aggs["vanilla_multi"])
    pref_pool = candidates(aggs["memory_multi3"]) + candidates(aggs["memory_multi4"]) + candidates(aggs["memory_multi_gpt5"])
    single_pool = candidates(aggs["memory_single2"]) + candidates(aggs["memory_single_gpt5"])
    for evidence_id in ("table:11", "table:12"):
        rows = [row for row in results if row["evidence_id"] == evidence_id]
        for (method, setting, model, preference), cells in grouped(rows, ("method", "setting", "model", "preference_type")).items():
            action = canonical_model(model)
            pref = pref_code(preference)
            block = str(setting).split(":", 1)[1] if ":" in str(setting) else "base"
            if method == "Base Prompting":
                pool = base_pool
                predicate = lambda row, a=action, p=pref: row_action(row) == a and row_pref(row) == p
            else:
                context = canonical_model(block)
                pool = pref_pool
                predicate = lambda row, a=action, p=pref, c=context: row_action(row) == a and row_pref(row) == p and row_context(row) == c
            assign_exact_group(cells, pool, MULTI7, predicate, admissions, source_by_path, source_by_id)
    rows = [row for row in results if row["evidence_id"] == "table:13"]
    for (_, setting, model, preference), cells in grouped(rows, ("method", "setting", "model", "preference_type")).items():
        action = canonical_model(model)
        context = canonical_model(str(setting).split(":", 1)[1])
        pref = pref_code(preference)
        predicate = lambda row, a=action, p=pref, c=context: row_action(row) == a and row_pref(row) == p and row_context(row) == c
        assign_exact_group(cells, single_pool, SINGLE3, predicate, admissions, source_by_path, source_by_id)


def table3_method_key(method: str) -> str:
    return {"Base Prompting": "base", "PREFINE": "prefine", "RAG(Top-5)": "rag", "Mem0": "mem0", "LangMem": "langmem"}.get(method, method)


def assign_table3_langmem_gemini(
    cells: Sequence[dict[str, Any]],
    pool: Sequence[dict[str, Any]],
    preference: str,
    admissions: Mapping[str, dict[str, Any]],
    source_by_path: Mapping[tuple[str, str], dict[str, Any]],
    source_by_id: Mapping[str, tuple[dict[str, Any], str]],
) -> None:
    """Bind the paper's mixed LangMem/Gemini row cell by cell.

    The paper's Recall cells are the minimal run shifted by exactly -10pp.
    Induction/Transfer P-EM comes from the main run while EA/OA comes from the
    minimal run.  Treating the three cells as one aggregate tuple would invent
    a run that does not exist.
    """
    pref = pref_code(preference)
    semantic = [
        candidate for candidate in pool
        if row_action(candidate["row"]) == "gemini" and row_pref(candidate["row"]) == pref
        and "inference_multi" in candidate["aggregate"]["path"]
    ]
    main = [candidate for candidate in semantic if "inference_multi_minimal" not in candidate["aggregate"]["path"]]
    minimal = [candidate for candidate in semantic if "inference_multi_minimal" in candidate["aggregate"]["path"]]
    if len(main) != 1 or len(minimal) != 1:
        for cell in cells:
            admissions[cell["result_id"]].update({
                "status": "unresolved", "reason_codes": ["langmem_gemini_main_or_minimal_row_not_unique"],
                "candidate_count": {"main": len(main), "minimal": len(minimal)},
            })
        return
    for cell in cells:
        use_minimal = pref == "easy" or cell["metric"] != "P-EM"
        candidate = minimal[0] if use_minimal else main[0]
        actual = rounded(candidate["row"][TABLE3_MULTI[cell["metric"]]], Decimal(100), 2)
        expected = paper_decimal(cell)
        is_drift = pref == "easy"
        if (not is_drift and actual != expected) or (is_drift and actual - expected != Decimal("10.00")):
            admissions[cell["result_id"]].update({
                "status": "unresolved", "reason_codes": ["langmem_gemini_cellwise_value_did_not_reproduce"],
                "aggregate_value": str(actual), "paper_value": str(expected),
            })
            continue
        raw, issue = resolve_raw(candidate, source_by_path, source_by_id)
        if raw is None:
            admissions[cell["result_id"]].update({
                "status": "unresolved", "reason_codes": ["langmem_gemini_raw_path_not_source_pre_bound"],
                "unbound_candidates": [issue] if issue else [],
            })
            continue
        binding = aggregate_binding(candidate, [cell], TABLE3_MULTI, relation="cellwise_mixed_run_binding")
        admission = admissions[cell["result_id"]]
        admission.update({
            "status": "admitted_direct_raw_with_declared_drift" if is_drift else "admitted_exact_raw",
            "admission_scope": "raw_file",
            "reason_codes": ["paper_langmem_gemini_recall_systematic_minus_10pp"] if is_drift else [],
            "aggregate_bindings": [binding],
            "raw_bindings": [raw_binding(raw, [binding["aggregate_node_id"]])],
        })
        if is_drift:
            admission["declared_drift"] = {
                "metric": cell["metric"], "paper": str(expected), "aggregate": str(actual),
                "delta_percentage_points": str(actual - expected),
                "classification": "systematic_paper_minus_10pp_display_drift",
            }


def bind_table3(results: Sequence[dict[str, Any]], admissions: Mapping[str, dict[str, Any]], witness: Mapping[str, Any], source_by_path: Mapping[tuple[str, str], dict[str, Any]], source_by_id: Mapping[str, tuple[dict[str, Any], str]], lineage: Mapping[str, list[dict[str, Any]]]) -> None:
    upstreams: dict[str, set[Path]] = defaultdict(set)
    for row in witness["rows"]:
        section = row["Section"]
        label = row["Method/Base LLM"]
        if section == "BASE PROMPTING": key = "base"
        elif section.startswith("PREFINE"): key = "prefine"
        elif label.startswith("RAG"): key = "rag"
        elif label == "Mem0": key = "mem0"
        elif label == "LangMem": key = "langmem"
        else: continue
        for item in row["upstream_source_files"].split(";"):
            relative = item.strip()
            if not relative.startswith("experiments4/"):
                raise AdmissionError(f"unexpected Table3 upstream path: {relative}")
            upstreams[key].add(Path("/data/minseo") / relative)
    pools: dict[str, list[dict[str, Any]]] = {}
    for key, paths in upstreams.items():
        pool: list[dict[str, Any]] = []
        for path in sorted(paths):
            pool.extend(candidates(make_aggregate(path, source_by_path, lineage)))
        pools[key] = pool
    rows = [row for row in results if row["evidence_id"] == "table:3"]
    for (method, model, setting, preference), cells in grouped(rows, ("method", "model", "setting", "preference_type")).items():
        if model in {"Average", "Average Gain"} or preference in {None, "Average"}:
            for cell in cells:
                admissions[cell["result_id"]].update({"status": "aggregate_only_derived", "admission_scope": "aggregate", "reason_codes": ["paper_average_or_gain_has_no_cell_minimal_raw_binding"], "supporting_evidence": [{"kind": "paper_witness", **public_record(witness["record"], witness["rel"]), "protected_path": witness["path"]}]})
            continue
        action = canonical_model(model)
        pref = pref_code(preference)
        mapping = TABLE3_MULTI if setting == "context-guided" else SINGLE3
        if method == "RAG(Top-5)":
            stamp = "20260217_092315" if setting == "context-guided" else "20260120_124137"
            branch = "inference_multi" if setting == "context-guided" else "inference_single"
            expected_path = str(EXP4 / f"RAG/{branch}/gemini-3-flash/{stamp}_output_rag_eval_{pref}.json")
            predicate = lambda row, path=expected_path: row.get("json_path") == path
            assign_exact_group(cells, pools["rag"], mapping, predicate, admissions, source_by_path, source_by_id)
            continue
        if method == "Mem0":
            if setting == "context-guided":
                expected_path = str(EXP4 / f"mem0/inference/multiturn/{pref}/0217_gemini-3-flash-high_test2.json")
            else:
                effort = "high" if pref == "hard" else "hard"
                expected_path = str(EXP4 / f"mem0/inference/singleturn/{pref}/0210_gemini3-flash-{effort}_test1.json")
            predicate = lambda row, path=expected_path: row.get("json_path") == path
            assign_exact_group(cells, pools["mem0"], mapping, predicate, admissions, source_by_path, source_by_id)
            continue
        if method == "LangMem" and action == "gemini" and setting == "context-guided":
            assign_table3_langmem_gemini(cells, pools["langmem"], preference, admissions, source_by_path, source_by_id)
            continue
        predicate = lambda row, a=action, p=pref: row_action(row) == a and row_pref(row) == p
        assign_exact_group(cells, pools[table3_method_key(method)], mapping, predicate, admissions, source_by_path, source_by_id)


def derived_from_parent_cells(result: Mapping[str, Any], parents: Sequence[dict[str, Any]], admission: dict[str, Any], relation: str) -> None:
    if not parents or any(not parent["status"].startswith("admitted") or not parent["raw_bindings"] for parent in parents):
        admission.update({"status": "unresolved_derived", "admission_scope": "none", "reason_codes": ["derived_parent_result_unresolved_or_without_raw"]})
        return
    raw_refs = {raw["raw_node_id"]: raw for parent in parents for raw in parent["raw_bindings"]}
    aggregate_refs = {binding["aggregate_node_id"]: binding for parent in parents for binding in parent["aggregate_bindings"]}
    if not raw_refs:
        admission.update({"status": "unresolved_derived", "admission_scope": "none", "reason_codes": ["derived_parent_raw_union_empty"]})
        return
    values: list[Decimal] = []
    for parent in parents:
        parent_id = parent["paper_result"]["result_id"]
        candidates = {
            d(binding["aggregate_values"][parent_id]) * d(binding.get("scale", "1"))
            for binding in parent["aggregate_bindings"]
            if parent_id in binding.get("aggregate_values", {})
        }
        if len(candidates) == 1:
            values.append(next(iter(candidates)))
        else:
            values.append(paper_decimal(parent["paper_result"]))
    actual = (sum(values) / Decimal(len(values))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    expected = paper_decimal(result)
    drift = actual != expected
    admission.update({
        "status": "admitted_derived_raw_with_declared_drift" if drift else "admitted_derived_exact_raw_set",
        "admission_scope": "derived_raw_file_set",
        "reason_codes": ["producer_declared_arithmetic_mean_display_drift"] if drift else [],
        "aggregate_bindings": list(aggregate_refs.values()), "raw_bindings": list(raw_refs.values()),
        "supporting_evidence": [{"kind": "derived_from_paper_result", "result_id": parent["paper_result"]["result_id"]} for parent in parents],
        "derivation": {"relation": relation, "parent_count": len(parents), "full_precision_parent_mean": str(actual), "paper_value": str(expected)},
    })
    if drift:
        admission["declared_drift"] = {
            "paper": str(expected), "aggregate": str(actual), "delta_percentage_points": str(actual - expected),
            "classification": "producer_declared_arithmetic_mean_display_drift",
        }


def bind_table3_prefine_parents(results: Sequence[dict[str, Any]], admissions: Mapping[str, dict[str, Any]]) -> None:
    parent_rows = [row for row in results if row["evidence_id"] in {"table:11", "table:12", "table:13"}]
    for result in (row for row in results if row["evidence_id"] == "table:3" and row.get("method") == "PREFINE" and row.get("model") not in {"Average", "Average Gain"} and row.get("preference_type") != "Average"):
        model = canonical_model(result["model"])
        if result["setting"] == "context-guided":
            parents = [
                admissions[row["result_id"]] for row in parent_rows
                if row["evidence_id"] in {"table:11", "table:12"} and row.get("method") == "PREFINE"
                and canonical_model(row.get("model")) == model and row.get("preference_type") == result.get("preference_type")
                and row.get("metric") == result.get("metric")
            ]
            expected_parent_count = 4
            relation = "producer_declared_mean_across_table11_table12_memory_models"
        else:
            parents = [
                admissions[row["result_id"]] for row in parent_rows
                if row["evidence_id"] == "table:13" and canonical_model(row.get("model")) == model
                and row.get("preference_type") == result.get("preference_type") and row.get("metric") == result.get("metric")
            ]
            expected_parent_count = 4
            relation = "producer_declared_mean_across_table13_memory_models"
        admission = admissions[result["result_id"]]
        if len(parents) != expected_parent_count:
            admission.update({"status": "unresolved_derived", "admission_scope": "none", "reason_codes": ["table3_prefine_parent_cardinality_mismatch"], "parent_count": len(parents)})
            continue
        derived_from_parent_cells(result, parents, admission, relation)


def bind_table3_prefine_cg_full_precision(
    results: Sequence[dict[str, Any]], admissions: Mapping[str, dict[str, Any]], aggs: Mapping[str, dict[str, Any]],
    source_by_path: Mapping[tuple[str, str], dict[str, Any]], source_by_id: Mapping[str, tuple[dict[str, Any], str]],
) -> None:
    """Reproduce main-PDF PREFINE CG from five full-precision context rows."""
    primary = candidates(aggs["memory_multi4"])
    fallback = candidates(aggs["memory_multi3"])
    gpt5_pool = candidates(aggs["memory_multi_gpt5"])
    # The main-PDF Table 3 is a later five-context revision than the older
    # four-context appendix witness: it includes both Qwen_Qwen3-8B and
    # DeepSeek-R1-0528-Qwen3-8B.  This set reproduces all 69 available PDF
    # cells exactly at 2dp; Gemini/Transfer lacks the GPT-4o parent.
    allowed_contexts = {"qwen3", "r1_qwen", "r1_llama", "gemma", "gpt4o"}
    rows = [row for row in results if row["evidence_id"] == "table:3" and row.get("method") == "PREFINE" and row.get("setting") == "context-guided" and row.get("model") not in {"Average", "Average Gain"} and row.get("preference_type") != "Average"]
    for (_, model, _, preference), cells in grouped(rows, ("method", "model", "setting", "preference_type")).items():
        action = canonical_model(model)
        pref = pref_code(preference)
        pool = gpt5_pool if action == "gpt5" else primary
        selected_by_context: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for candidate in pool:
            context = row_context(candidate["row"])
            if row_action(candidate["row"]) == action and row_pref(candidate["row"]) == pref and context in allowed_contexts:
                selected_by_context[context].append(candidate)
        if action != "gpt5":
            for candidate in fallback:
                context = row_context(candidate["row"])
                if context not in selected_by_context and row_action(candidate["row"]) == action and row_pref(candidate["row"]) == pref and context in allowed_contexts:
                    selected_by_context[context].append(candidate)
        if set(selected_by_context) != allowed_contexts or any(len(value) != 1 for value in selected_by_context.values()):
            for cell in cells:
                admissions[cell["result_id"]].update({
                    "status": "unresolved_derived", "admission_scope": "none",
                    "reason_codes": ["table3_prefine_cg_full_precision_parent_set_not_unique"],
                    "context_candidate_counts": {key: len(value) for key, value in sorted(selected_by_context.items())},
                })
            continue
        selected = [selected_by_context[key][0] for key in sorted(allowed_contexts)]
        aggregate_refs = [aggregate_binding(candidate, cells, TABLE3_MULTI, relation="producer_declared_full_precision_mean_component") for candidate in selected]
        raw_refs: list[dict[str, Any]] = []
        raw_failure: list[str] = []
        for candidate, binding in zip(selected, aggregate_refs):
            raw, issue = resolve_raw(candidate, source_by_path, source_by_id)
            if raw is None:
                raw_failure.append(str(issue))
            else:
                raw_refs.append(raw_binding(raw, [binding["aggregate_node_id"]]))
        if raw_failure or len({raw["raw_node_id"] for raw in raw_refs}) != len(allowed_contexts):
            for cell in cells:
                admissions[cell["result_id"]].update({
                    "status": "unresolved_derived", "admission_scope": "none",
                    "reason_codes": ["table3_prefine_cg_raw_parent_set_not_fully_bound"], "unbound_candidates": raw_failure,
                })
            continue
        for cell in cells:
            column = TABLE3_MULTI[cell["metric"]]
            actual = (sum(d(candidate["row"][column]) * Decimal(100) for candidate in selected) / Decimal(len(selected))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            expected = paper_decimal(cell)
            delta = actual - expected
            admission = admissions[cell["result_id"]]
            if abs(delta) > Decimal("0.01"):
                admission.update({
                    "status": "unresolved_derived", "admission_scope": "none",
                    "reason_codes": ["table3_prefine_cg_full_precision_value_drift_exceeds_declared_tolerance"],
                    "producer_value": str(actual), "paper_value": str(expected), "delta_percentage_points": str(delta),
                })
                continue
            drift = delta != 0
            admission.update({
                "status": "admitted_derived_raw_with_declared_drift" if drift else "admitted_derived_exact_raw_set",
                "admission_scope": "derived_raw_file_set", "reason_codes": ["producer_declared_arithmetic_mean_display_drift"] if drift else [],
                "aggregate_bindings": aggregate_refs, "raw_bindings": raw_refs,
                "supporting_evidence": [{"kind": "older_four_context_appendix_witness_revision_conflict", **public_record(aggs["t3_witness"]["record"], aggs["t3_witness"]["rel"]), "protected_path": aggs["t3_witness"]["path"]}],
                "derivation": {"relation": "main_pdf_full_precision_mean_across_five_observed_contexts", "producer_value": str(actual), "paper_value": str(expected)},
            })
            if drift:
                admission["declared_drift"] = {
                    "paper": str(expected), "aggregate": str(actual), "delta_percentage_points": str(delta),
                    "classification": "producer_declared_arithmetic_mean_display_drift",
                }


def bind_table3_prefine_cf_full_precision(
    results: Sequence[dict[str, Any]], admissions: Mapping[str, dict[str, Any]], aggs: Mapping[str, dict[str, Any]],
    source_by_path: Mapping[tuple[str, str], dict[str, Any]], source_by_id: Mapping[str, tuple[dict[str, Any], str]],
) -> None:
    """Reproduce main-PDF PREFINE CF from the unique five-context raw set.

    The main PDF is a later five-context revision than the four-context Table
    13 witness. Some producer CSV groups retain multiple model variants or
    original/``_format`` rows. Solve over every five-context combination and
    admit only a unique exact raw set, falling back to a unique <=0.01pp
    display-drift set. Ambiguity and larger revision drift remain unresolved.
    """
    standard_pool = candidates(aggs["memory_single2"])
    gpt5_pool = candidates(aggs["memory_single_gpt5"])
    allowed_contexts = {"qwen3", "r1_qwen", "r1_llama", "gemma", "gpt4o"}
    context_order = sorted(allowed_contexts)
    witness_support = [{
        "kind": "older_four_context_appendix_witness_revision_conflict",
        **public_record(aggs["t3_witness"]["record"], aggs["t3_witness"]["rel"]),
        "protected_path": aggs["t3_witness"]["path"],
    }]
    rows = [
        row for row in results
        if row["evidence_id"] == "table:3" and row.get("method") == "PREFINE"
        and row.get("setting") == "context-free" and row.get("model") not in {"Average", "Average Gain"}
        and row.get("preference_type") != "Average"
    ]

    def reset(admission: dict[str, Any]) -> None:
        admission.update({
            "admission_scope": "none", "aggregate_bindings": [], "raw_bindings": [],
            "root_bindings": [], "supporting_evidence": witness_support,
        })
        for key in ("declared_drift", "producer_value", "paper_value", "delta_percentage_points"):
            admission.pop(key, None)

    for (_, model, _, preference), cells in grouped(rows, ("method", "model", "setting", "preference_type")).items():
        action = canonical_model(model)
        pref = pref_code(preference)
        pool = gpt5_pool if action == "gpt5" else standard_pool
        selected_by_context: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for candidate in pool:
            context = row_context(candidate["row"])
            if row_action(candidate["row"]) == action and row_pref(candidate["row"]) == pref and context in allowed_contexts:
                selected_by_context[context].append(candidate)
        if set(selected_by_context) != allowed_contexts:
            counts = {key: len(selected_by_context.get(key, [])) for key in context_order}
            for cell in cells:
                admission = admissions[cell["result_id"]]
                reset(admission)
                admission.update({
                    "status": "unresolved_derived",
                    "reason_codes": ["table3_prefine_cf_five_context_parent_set_incomplete"],
                    "context_candidate_counts": counts,
                })
            continue

        combinations: list[dict[str, Any]] = []
        for selected_tuple in itertools.product(*(selected_by_context[key] for key in context_order)):
            values = {
                metric: (
                    sum(d(candidate["row"][column]) * Decimal(100) for candidate in selected_tuple)
                    / Decimal(len(context_order))
                ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                for metric, column in SINGLE3.items()
            }
            deltas = {cell["metric"]: values[cell["metric"]] - paper_decimal(cell) for cell in cells}
            combinations.append({"selected": selected_tuple, "values": values, "deltas": deltas})
        exact = [item for item in combinations if all(delta == 0 for delta in item["deltas"].values())]
        tolerance = [item for item in combinations if all(abs(delta) <= Decimal("0.01") for delta in item["deltas"].values())]
        qualifying = exact if exact else tolerance
        match_kind = "exact" if exact else "display_drift_tolerance"

        resolved: list[dict[str, Any]] = []
        unbound_qualifying = 0
        for item in qualifying:
            raw_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
            failed = False
            for candidate in item["selected"]:
                raw, _ = resolve_raw(candidate, source_by_path, source_by_id)
                if raw is None:
                    failed = True
                    break
                raw_pairs.append((candidate, raw))
            if failed:
                unbound_qualifying += 1
                continue
            item = dict(item)
            item["raw_pairs"] = raw_pairs
            item["raw_set_key"] = tuple(sorted(raw["record"]["record_id"] for _, raw in raw_pairs))
            resolved.append(item)

        raw_sets = {item["raw_set_key"] for item in resolved}
        if len(qualifying) != 1 or unbound_qualifying or len(raw_sets) != 1 or len(resolved) != 1:
            reason = (
                "table3_prefine_cf_full_precision_value_not_reproduced"
                if not qualifying else "table3_prefine_cf_full_precision_raw_set_ambiguous_or_unbound"
            )
            best_delta = None
            if combinations:
                best_delta = min(max(abs(delta) for delta in item["deltas"].values()) for item in combinations)
            for cell in cells:
                admission = admissions[cell["result_id"]]
                reset(admission)
                admission.update({
                    "status": "unresolved_derived", "reason_codes": [reason],
                    "context_candidate_counts": {key: len(selected_by_context[key]) for key in context_order},
                    "exact_combination_count": len(exact), "tolerance_combination_count": len(tolerance),
                    "fully_bound_qualifying_combination_count": len(resolved),
                    "unbound_qualifying_combination_count": unbound_qualifying,
                    "best_max_delta_percentage_points": str(best_delta) if best_delta is not None else None,
                })
            continue

        chosen = resolved[0]
        aggregate_refs = [
            aggregate_binding(candidate, cells, SINGLE3, relation="main_pdf_full_precision_mean_component")
            for candidate, _ in chosen["raw_pairs"]
        ]
        raw_refs = [
            raw_binding(raw, [binding["aggregate_node_id"]])
            for (_, raw), binding in zip(chosen["raw_pairs"], aggregate_refs)
        ]
        if len({raw["raw_node_id"] for raw in raw_refs}) != len(context_order):
            for cell in cells:
                admission = admissions[cell["result_id"]]
                reset(admission)
                admission.update({
                    "status": "unresolved_derived",
                    "reason_codes": ["table3_prefine_cf_five_context_raw_ids_not_distinct"],
                })
            continue
        for cell in cells:
            admission = admissions[cell["result_id"]]
            reset(admission)
            actual = chosen["values"][cell["metric"]]
            expected = paper_decimal(cell)
            delta = chosen["deltas"][cell["metric"]]
            drift = delta != 0
            admission.update({
                "status": "admitted_derived_raw_with_declared_drift" if drift else "admitted_derived_exact_raw_set",
                "admission_scope": "derived_raw_file_set",
                "reason_codes": ["producer_declared_arithmetic_mean_display_drift"] if drift else [],
                "aggregate_bindings": aggregate_refs, "raw_bindings": raw_refs,
                "supporting_evidence": witness_support,
                "derivation": {
                    "relation": "main_pdf_full_precision_mean_across_five_observed_contexts",
                    "selection": "unique_exact_raw_set" if match_kind == "exact" else "unique_within_0.01pp_raw_set",
                    "producer_value": str(actual), "paper_value": str(expected),
                },
            })
            if drift:
                admission["declared_drift"] = {
                    "paper": str(expected), "aggregate": str(actual),
                    "delta_percentage_points": str(delta),
                    "classification": "producer_declared_arithmetic_mean_display_drift",
                }


def bind_table3_higher_derivations(results: Sequence[dict[str, Any]], admissions: Mapping[str, dict[str, Any]]) -> None:
    table3 = [row for row in results if row["evidence_id"] == "table:3"]
    # Per-model averages over Recall/Induction/Transfer.
    for result in (row for row in table3 if row.get("preference_type") == "Average"):
        parents = [
            admissions[row["result_id"]] for row in table3
            if row.get("method") == result.get("method") and row.get("model") == result.get("model")
            and row.get("setting") == result.get("setting") and row.get("metric") == result.get("metric")
            and row.get("preference_type") in {"Recall", "Induction", "Transfer"}
        ]
        if len(parents) == 3:
            derived_from_parent_cells(result, parents, admissions[result["result_id"]], "mean_across_preference_types")
    # Paper Average rows over action/base models.
    for result in (row for row in table3 if row.get("model") == "Average"):
        parents = [
            admissions[row["result_id"]] for row in table3
            if row.get("method") == result.get("method") and row.get("model") not in {"Average", "Average Gain"}
            and row.get("setting") == result.get("setting") and row.get("preference_type") == result.get("preference_type")
            and row.get("metric") == result.get("metric")
        ]
        derived_from_parent_cells(result, parents, admissions[result["result_id"]], "mean_across_action_models")
    # Average Gain rows inherit the Base/PREFINE Average raw union.  The paper
    # stores a difference rather than a mean, so keep the exact paper parent
    # relation without relabeling it as an exact aggregate-row match.
    for result in (row for row in table3 if row.get("model") == "Average Gain"):
        parents = [
            admissions[row["result_id"]] for row in table3
            if row.get("model") == "Average" and row.get("setting") == result.get("setting")
            and row.get("preference_type") == result.get("preference_type") and row.get("metric") == result.get("metric")
            and row.get("method") in {"Base Prompting", "PREFINE"}
        ]
        admission = admissions[result["result_id"]]
        if len(parents) != 2 or any(not parent["status"].startswith("admitted") for parent in parents):
            admission.update({"status": "unresolved_derived", "admission_scope": "none", "reason_codes": ["average_gain_parent_unresolved"]})
            continue
        base = next(parent for parent in parents if parent["paper_result"]["method"] == "Base Prompting")
        pref = next(parent for parent in parents if parent["paper_result"]["method"] == "PREFINE")
        actual = (paper_decimal(pref["paper_result"]) - paper_decimal(base["paper_result"])).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        if actual != paper_decimal(result):
            admission.update({"status": "unresolved_derived", "admission_scope": "none", "reason_codes": ["average_gain_parent_difference_mismatch"]})
            continue
        raw_refs = {raw["raw_node_id"]: raw for parent in parents for raw in parent["raw_bindings"]}
        aggregate_refs = {binding["aggregate_node_id"]: binding for parent in parents for binding in parent["aggregate_bindings"]}
        admission.update({
            "status": "admitted_derived_exact_raw_set", "admission_scope": "derived_raw_file_set", "reason_codes": [],
            "raw_bindings": list(raw_refs.values()), "aggregate_bindings": list(aggregate_refs.values()),
            "supporting_evidence": [{"kind": "derived_from_paper_result", "result_id": parent["paper_result"]["result_id"]} for parent in parents],
            "derivation": {"relation": "prefine_average_minus_base_average", "paper_value": result["numeric_value"]},
        })


def row_node(aggregate: Mapping[str, Any], row_number: int, relation: str) -> dict[str, Any]:
    return {"aggregate_node_id": f"aggregate:{aggregate['record']['record_id']}:row:{row_number}", **public_record(aggregate["record"], aggregate["rel"]), "protected_path": aggregate["path"], "row_number": row_number, "relation": relation, "variant_labels": aggregate["variants"]}


def key3(memory: str, pref: str, action: str) -> tuple[str, str, str]:
    return (memory, pref, action)


def bind_table15(results: Sequence[dict[str, Any]], admissions: Mapping[str, dict[str, Any]], aggs: Mapping[str, dict[str, Any]], source_by_path: Mapping[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    multi_common = aggs["t15_multi_common"]
    single_common = aggs["t15_single_common"]
    multi_a = aggs["t15_multi_a_detail"]
    single_a = aggs["t15_single_a_detail"]
    multi_b = aggs["t15_multi_b"]
    single_b = aggs["t15_single_b"]
    a_multi_index: dict[tuple[str, str, str], tuple[int, dict[str, str]]] = {}
    for number, row in enumerate(multi_a["rows"], 2):
        memory = MEMORY_NAME_MAP_A.get(row["memory_model"])
        if memory and row["status"] == "ok": a_multi_index[key3(memory, row["pref_type"], row["action_model"])] = (number, row)
    a_single_index: dict[tuple[str, str, str], tuple[int, dict[str, str]]] = {}
    for number, row in enumerate(single_a["rows"], 2):
        memory = MEMORY_NAME_MAP_A.get(row["memory_model"])
        if memory and row["status"] == "ok": a_single_index[key3(memory, row["pref_type"], row["action_model"])] = (number, row)
    b_multi_index: dict[tuple[str, str, str], tuple[int, dict[str, str]]] = {}
    for number, row in enumerate(multi_b["rows"], 2):
        if row["pref_type"] == "memory_api" and row["prompt_type"] == "implicit_zs" and row["status"] == "ok":
            b_multi_index[key3(row["context"], row["query_turns"], row["model_name"])] = (number, row)
    b_single_candidates: dict[tuple[str, str, str], list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for number, row in enumerate(single_b["rows"], 2):
        if row["file_name"].endswith("_format.json") or row["pref_type"] != "memory_api" or row["prompt_type"] != "implicit_zs" or row["status"] != "ok": continue
        memory = MEMORY_NAME_MAP_B_SINGLE.get(row["context"], row["context"])
        b_single_candidates[key3(memory, row["query_turns"], row["model_name"])].append((number, row))
    b_single_index = {key: sorted(items, key=lambda item: item[1]["file_name"])[0] for key, items in b_single_candidates.items()}

    sets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for setting, common, a_index, b_index, a_agg, b_agg in (
        ("context-guided", multi_common, a_multi_index, b_multi_index, multi_a, multi_b),
        ("context-free", single_common, a_single_index, b_single_index, single_a, single_b),
    ):
        for label, index, aggregate in (("10-iterations", a_index, a_agg), ("3-iterations", b_index, b_agg)):
            raw_set: list[dict[str, Any]] = []
            for common_number, common_row in enumerate(common["rows"], 2):
                key = key3(common_row["memory_model_norm"], common_row["pref_type"], common_row["action_model"])
                if key not in index:
                    raise AdmissionError(f"Table15 missing semantic key {setting}/{label}: {key}")
                row_number, source_row = index[key]
                raw_path = source_row["json_path"]
                record = source_by_path.get(("experiments4", exp4_rel(raw_path)))
                if record is None or record.get("type") != "regular":
                    raise AdmissionError(f"Table15 raw is not source-pre-bound: {raw_path}")
                raw_set.append({
                    "raw_node_id": f"raw:{record['record_id']}", **public_record(record, exp4_rel(raw_path)), "protected_path": raw_path,
                    "relation": "direct_json_path_reference_via_common_subset_semantic_key",
                    "semantic_key": {"memory_model_norm": key[0], "pref_type": key[1], "action_model": key[2]},
                    "via_aggregate_node_ids": [f"aggregate:{common['record']['record_id']}:row:{common_number}", f"aggregate:{aggregate['record']['record_id']}:row:{row_number}"],
                })
            expected = 45 if setting == "context-guided" else 48
            if len(raw_set) != expected or len({row["raw_node_id"] for row in raw_set}) != expected:
                raise AdmissionError(f"Table15 direct lineage cardinality drift for {setting}/{label}")
            sets[(setting, label)] = raw_set

    by_pref = {"context-guided": aggs["t15_multi_by_pref"], "context-free": aggs["t15_single_by_pref"]}
    for result in (row for row in results if row["evidence_id"] == "table:15"):
        setting = result["setting"]
        pref = pref_code(result["preference_type"])
        aggregate = by_pref[setting]
        aggregate_row_number, aggregate_row = next((number, row) for number, row in enumerate(aggregate["rows"], 2) if row["pref_type"] == pref)
        column = result["locator"]["column"]
        if column == "10-iterations": source_column, labels = ("f1_0312_MEMORY1_max10", ["10-iterations"])
        elif column == "3-iterations": source_column, labels = ("f1_1231_MEMORY3_max3", ["3-iterations"])
        else: source_column, labels = (None, ["10-iterations", "3-iterations"])
        if source_column:
            value = rounded(aggregate_row[source_column], Decimal(1), 3)
        else:
            value = (d(aggregate_row["f1_0312_MEMORY1_max10"]) - d(aggregate_row["f1_1231_MEMORY3_max3"])).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
        if value != paper_decimal(result):
            raise AdmissionError(f"Table15 paper/aggregate drift: {result['result_id']}")
        raw_refs: list[dict[str, Any]] = []
        for label in labels: raw_refs.extend(sets[(setting, label)])
        admission = admissions[result["result_id"]]
        admission.update({
            "status": "admitted_exact_aggregate_raw_set", "admission_scope": "raw_file_set", "reason_codes": [],
            "aggregate_bindings": [{**row_node(aggregate, aggregate_row_number, "exact_rounded_metric_match" if source_column else "exact_rounded_difference"), "metric_column": source_column, "paper_value": result["numeric_value"]}],
            "raw_bindings": raw_refs,
            "supporting_evidence": [
                {"kind": "comparison_builder", **public_record(aggs["t15_builder"]["record"], aggs["t15_builder"]["rel"]), "protected_path": aggs["t15_builder"]["path"]},
                {"kind": "summary_direct_root_and_output_declaration", **public_record(aggs["t15_summary_record"], exp4_rel(TABLE_FILES["t15_summary"])), "protected_path": str(TABLE_FILES["t15_summary"])},
            ],
        })
    return {"raw_links": sum(len(value) for value in sets.values()), "unique_raw": len({raw["raw_node_id"] for values in sets.values() for raw in values})}


def row_difficulty(row: Mapping[str, str]) -> str:
    for key in ("pref_type", "query_turns", "prompt_type", "group_key", "file_name", "json_path"):
        value = str(row.get(key, "")).lower()
        for token in ("easy", "medium", "hard"):
            if re.search(rf"(^|[^a-z]){token}([^a-z]|$)", value):
                return token
    return ""


def bind_table14(results: Sequence[dict[str, Any]], admissions: Mapping[str, dict[str, Any]], aggs: Mapping[str, dict[str, Any]], source_by_path: Mapping[tuple[str, str], dict[str, Any]]) -> None:
    pools = {
        ("LangMem", "context-free"): aggs["t14_langmem_single"],
        ("LangMem", "context-guided"): aggs["t14_langmem_multi"],
        ("Mem0", "context-free"): aggs["t14_mem0_single"],
        ("Mem0", "context-guided"): aggs["t14_mem0_multi"],
        ("RAG", "context-free"): aggs["t14_rag_single"],
        ("RAG", "context-guided"): aggs["t14_rag_multi"],
    }
    rows = [row for row in results if row["evidence_id"] == "table:14"]
    raw_ids: set[str] = set()
    for (method, setting, preference), cells in grouped(rows, ("method", "setting", "preference_type")).items():
        aggregate = pools[(method, setting)]
        pref = pref_code(preference)
        mapping = SINGLE3 if setting == "context-free" else TABLE3_MULTI
        semantic = [
            (number, row) for number, row in enumerate(aggregate["rows"], 2)
            if row.get("status") == "ok" and row_difficulty(row) == pref and (method != "LangMem" or row_action(row) == "gpt5")
        ]
        exact = [(number, row) for number, row in semantic if metric_exact(cells, row, mapping)]
        declared_drift: dict[str, Any] | None = None
        if len(exact) == 1:
            number, row = exact[0]
        elif method == "RAG" and setting == "context-guided" and pref == "hard" and len(semantic) == 1:
            number, row = semantic[0]
            for cell in cells:
                if cell["metric"] != "P-EM" and not metric_exact([cell], row, mapping):
                    raise AdmissionError(f"Table14 RAG hard non-drift metric mismatch: {cell['result_id']}")
            pem = next(cell for cell in cells if cell["metric"] == "P-EM")
            actual = rounded(row[mapping["P-EM"]], Decimal(100), 2)
            expected = paper_decimal(pem)
            if expected != Decimal("0.87") or actual != Decimal("8.69"):
                raise AdmissionError(f"Table14 documented RAG P-EM drift changed: paper={expected}, aggregate={actual}")
            declared_drift = {"metric": "P-EM", "paper": str(expected), "aggregate": str(actual), "delta_percentage_points": str(actual - expected), "classification": "factor-ten paper display typo_or_historical_drift"}
        else:
            raise AdmissionError(f"Table14 row match is not unique: {method}/{setting}/{pref}, exact={len(exact)}, semantic={len(semantic)}")
        raw_path = row["json_path"]
        record = source_by_path.get(("experiments4", exp4_rel(raw_path)))
        if record is None or record.get("type") != "regular":
            raise AdmissionError(f"Table14 raw absent from source-pre: {raw_path}")
        raw_ids.add(record["record_id"])
        binding = {**row_node(aggregate, number, "exact_rounded_metric_tuple" if declared_drift is None else "direct_semantic_key_with_declared_display_drift"), "metric_columns": {cell["result_id"]: mapping[cell["metric"]] for cell in cells}}
        raw_ref = {"raw_node_id": f"raw:{record['record_id']}", **public_record(record, exp4_rel(raw_path)), "protected_path": raw_path, "relation": "direct_json_path_reference", "via_aggregate_node_ids": [binding["aggregate_node_id"]]}
        for cell in cells:
            is_drift = declared_drift is not None and cell["metric"] == "P-EM"
            admissions[cell["result_id"]].update({
                "status": "admitted_direct_raw_with_declared_drift" if is_drift else "admitted_exact_raw",
                "admission_scope": "raw_file", "reason_codes": ["paper_vs_aggregate_factor_ten_display_drift"] if is_drift else [],
                "aggregate_bindings": [binding], "raw_bindings": [raw_ref],
            })
            if is_drift:
                admissions[cell["result_id"]]["declared_drift"] = declared_drift
    if len(raw_ids) != 18:
        raise AdmissionError(f"Table14 raw cardinality drift: {len(raw_ids)} != 18")


def figure4_derived_parent_raw_union(
    parent_results: Sequence[Mapping[str, Any]],
    parent_admissions: Sequence[Mapping[str, Any]],
) -> tuple[bool, list[dict[str, Any]]]:
    """Validate Figure 4 parents and return inference-parent raw only.

    Ground Truth is an aggregate-only dataset statistic, so it is a valid
    non-copy parent. Every other Figure 4 parent is an inference result and
    must be admitted with a non-empty raw binding.
    """
    if len(parent_results) != len(parent_admissions) or not parent_results:
        return False, []
    inference_parents: list[Mapping[str, Any]] = []
    for result, admission in zip(parent_results, parent_admissions):
        if result.get("evidence_id") != "figure:4":
            return False, []
        if result.get("method") == "Ground Truth":
            if admission.get("status") != "aggregate_only_dataset_reference":
                return False, []
            continue
        raw_bindings = admission.get("raw_bindings")
        if not str(admission.get("status", "")).startswith("admitted") or not raw_bindings:
            return False, []
        if any(not isinstance(raw, Mapping) or not raw.get("raw_node_id") for raw in raw_bindings):
            return False, []
        inference_parents.append(admission)
    raw_refs = {
        raw["raw_node_id"]: dict(raw)
        for parent in inference_parents
        for raw in parent["raw_bindings"]
    }
    return True, [raw_refs[key] for key in sorted(raw_refs)]


def bind_figure4(results: Sequence[dict[str, Any]], admissions: Mapping[str, dict[str, Any]], aggs: Mapping[str, dict[str, Any]], source_by_path: Mapping[tuple[str, str], dict[str, Any]], source_by_id: Mapping[str, tuple[dict[str, Any], str]]) -> None:
    specs = {
        ("Base Prompting", "context-free"): (aggs["f4_base_single"], EXP4 / "vanillaLLM/inference/1229-3_output_singleturn"),
        ("Base Prompting", "context-guided"): (aggs["f4_base_multi"], EXP4 / "vanillaLLM/inference/1230-1_output_multiturn"),
        ("PREFINE", "context-free"): (aggs["f4_pref_single"], EXP4 / "ours_memory/inference/1231_MEMORY3_inference1_single"),
        ("PREFINE", "context-guided"): (aggs["f4_pref_multi"], EXP4 / "ours_memory/inference/1231_MEMORY3_inference2_multi"),
    }
    figure_rows = [row for row in results if row["evidence_id"] == "figure:4"]
    for result in figure_rows:
        setting = result["setting"]
        method = result["method"]
        if method == "Ground Truth":
            aggregate, _ = specs[("Base Prompting", setting)]
            selected = [(number, row) for number, row in enumerate(aggregate["rows"], 2) if row_action(row) == "r1_llama" and not row.get("error")]
            if setting == "context-guided":
                selected = [(number, row) for number, row in selected if row["file_name"].startswith("0103_test_deepseek2")]
            value = (sum(d(row["avg_gt_slots"]) for _, row in selected) / Decimal(len(selected))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            admission = admissions[result["result_id"]]
            if value == paper_decimal(result):
                admission.update({
                    "status": "aggregate_only_dataset_reference", "admission_scope": "aggregate", "reason_codes": ["ground_truth_dataset_statistic_not_inference_output"],
                    "aggregate_bindings": [{**row_node(aggregate, number, "exact_mean_component"), "metric_column": "avg_gt_slots"} for number, _ in selected],
                })
            else:
                admission.update({"status": "unresolved_derived", "reason_codes": ["ground_truth_slotcount_mean_did_not_reproduce"]})
            continue
        aggregate, raw_root = specs[(method, setting)]
        model = canonical_model(result["model"])
        selected = [(number, row) for number, row in enumerate(aggregate["rows"], 2) if row_action(row) == model and not row.get("error")]
        if method == "PREFINE":
            allowed_contexts = {"r1_qwen", "r1_llama", "gemma", "gpt4o"}
            selected = [(number, row) for number, row in selected if canonical_model(Path(row["rel_path"]).parts[0]) in allowed_contexts]
        if result.get("numeric_value") is not None and method == "Base Prompting" and setting == "context-guided" and model == "r1_llama":
            selected = [(number, row) for number, row in selected if row["file_name"].startswith("0103_test_deepseek2")]
        raw_refs: list[dict[str, Any]] = []
        aggregate_refs: list[dict[str, Any]] = []
        unbound_paths: list[str] = []
        for number, row in selected:
            raw_path = str(raw_root / row["rel_path"])
            rel = exp4_rel(raw_path)
            record = source_by_path.get(("experiments4", rel))
            relation = "direct_slotcount_rel_path_reference"
            recomputation = None
            if record is None and raw_path in RECOMPUTED_T13_ALIASES:
                assertion = RECOMPUTED_T13_ALIASES[raw_path]
                record, rel = source_by_id[assertion["alias_record_id"]]
                raw_path = abs_for_record(record, rel)
                relation = "recomputed_exact_alias"
                recomputation = {"stale_slotcount_rel_path": str(raw_root / row["rel_path"]), "same_config_alias_path": raw_path, "evaluator_record_id": T13_EVALUATOR_RECORD}
            if record is None or record.get("type") != "regular":
                unbound_paths.append(str(raw_root / row["rel_path"]))
                continue
            node = row_node(aggregate, number, "slotcount_semantic_membership" if result.get("numeric_value") is None else "exact_mean_component")
            node["metric_column"] = "avg_pred_slots"
            aggregate_refs.append(node)
            raw_ref = {"raw_node_id": f"raw:{record['record_id']}", **public_record(record, rel), "protected_path": raw_path, "relation": relation, "via_aggregate_node_ids": [node["aggregate_node_id"]]}
            if recomputation:
                raw_ref["recomputation_evidence"] = recomputation
            raw_refs.append(raw_ref)
        raw_refs = list({row["raw_node_id"]: row for row in raw_refs}.values())
        admission = admissions[result["result_id"]]
        if unbound_paths:
            admission.update({
                "status": "unresolved", "admission_scope": "none",
                "reason_codes": ["figure4_semantic_raw_set_not_fully_source_pre_bound"],
                "unbound_candidates": sorted(unbound_paths),
                "bound_candidate_count": len(raw_refs),
            })
            continue
        if not raw_refs:
            admission.update({"status": "unresolved", "reason_codes": ["figure4_slotcount_semantic_set_empty"]})
            continue
        if result.get("numeric_value") is not None:
            mean_value = (sum(d(row["avg_pred_slots"]) for _, row in selected) / Decimal(len(selected))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            if mean_value != paper_decimal(result):
                admission.update({"status": "unresolved_derived", "reason_codes": ["figure4_explicit_mean_did_not_reproduce"], "aggregate_bindings": aggregate_refs})
                continue
            status, relation_reason = "admitted_derived_exact_raw_set", []
        else:
            status, relation_reason = "admitted_semantic_raw_set_unlabeled_graphical_mark", ["graphical_mark_has_no_printed_numeric_value"]
        admission.update({"status": status, "admission_scope": "raw_file_set", "reason_codes": relation_reason, "aggregate_bindings": aggregate_refs, "raw_bindings": raw_refs})
    result_by_id = {row["result_id"]: row for row in results}
    for result in (row for row in results if row["evidence_id"] == "page:8" and row.get("derived_from")):
        parent_ids = list(result["derived_from"])
        parent_admissions = [admissions[parent] for parent in parent_ids if parent in admissions]
        parent_results = [result_by_id[parent] for parent in parent_ids if parent in result_by_id]
        if (
            len(parent_admissions) == len(parent_ids)
            and len(parent_results) == len(parent_ids)
            and all(parent["evidence_id"] == "figure:5" for parent in parent_results)
            and all(parent["status"].startswith("non_inference") for parent in parent_admissions)
        ):
            admissions[result["result_id"]].update({
                "status": "non_inference_memory_snapshot_derived_statistic",
                "admission_scope": "none",
                "reason_codes": ["derived_only_from_non_inference_memory_snapshot_statistics"],
                "supporting_evidence": [
                    {"kind": "derived_from_paper_result", "result_id": parent}
                    for parent in parent_ids
                ],
            })
            continue
        valid_parents, inference_raw_refs = figure4_derived_parent_raw_union(
            parent_results, parent_admissions
        )
        if len(parent_admissions) != len(parent_ids) or not valid_parents:
            admissions[result["result_id"]].update({
                "status": "unresolved_derived", "admission_scope": "none",
                "reason_codes": ["derived_from_unresolved_figure4_results"],
            })
            continue
        if not inference_raw_refs:
            admissions[result["result_id"]].update({
                "status": "unresolved_derived", "admission_scope": "none",
                "reason_codes": ["derived_figure4_parents_have_no_admitted_raw"],
            })
            continue
        admissions[result["result_id"]].update({
            "status": "admitted_derived_from_figure4_raw_sets", "admission_scope": "derived_raw_file_set", "reason_codes": [],
            "raw_bindings": inference_raw_refs, "supporting_evidence": [{"kind": "derived_from_paper_result", "result_id": parent} for parent in parent_ids],
        })


def bind_table10(results: Sequence[dict[str, Any]], admissions: Mapping[str, dict[str, Any]]) -> None:
    t3 = {}
    for result in (row for row in results if row["evidence_id"] == "table:3"):
        t3[(result["method"], result["model"], result["setting"], result["preference_type"], result["metric"])] = result
    for result in (row for row in results if row["evidence_id"] == "table:10"):
        suffix = (result["model"], result["setting"], result["preference_type"], result["metric"])
        base = t3.get(("Base Prompting", *suffix))
        pref = t3.get(("PREFINE", *suffix))
        admission = admissions[result["result_id"]]
        if base is None or pref is None:
            admission.update({"status": "unresolved_derived", "reason_codes": ["missing_table3_delta_parent"]})
            continue
        computed = (paper_decimal(pref) - paper_decimal(base)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        if computed != paper_decimal(result):
            admission.update({"status": "unresolved_derived", "reason_codes": ["table10_delta_does_not_equal_table3_prefine_minus_base"]})
            continue
        parents = [admissions[base["result_id"]], admissions[pref["result_id"]]]
        if any(
            not parent["status"].startswith("admitted") or not parent["raw_bindings"]
            for parent in parents
        ):
            admission.update({
                "status": "unresolved_derived",
                "admission_scope": "none",
                "reason_codes": ["table3_delta_parents_not_both_admitted_with_raw"],
            })
            continue
        raw_refs = {raw["raw_node_id"]: raw for parent in parents for raw in parent["raw_bindings"]}
        aggregate_refs = {binding["aggregate_node_id"]: binding for parent in parents for binding in parent["aggregate_bindings"]}
        if not raw_refs:
            admission.update({"status": "unresolved_derived", "reason_codes": ["table3_delta_parents_have_no_admitted_raw"]})
            continue
        admission.update({
            "status": "admitted_derived_delta_raw_set", "admission_scope": "derived_raw_file_set", "reason_codes": [],
            "aggregate_bindings": list(aggregate_refs.values()), "raw_bindings": list(raw_refs.values()),
            "supporting_evidence": [{"kind": "derived_from_paper_result", "result_id": base["result_id"]}, {"kind": "derived_from_paper_result", "result_id": pref["result_id"]}],
        })


def t16_method(method: str) -> str:
    return {"Base Prompting": "vanilla_llm", "RAG": "rag", "Mem0": "mem0", "LangMem": "langmem", "PREFINE": "ours_memory"}[method]


def bind_table16(results: Sequence[dict[str, Any]], admissions: Mapping[str, dict[str, Any]], aggs: Mapping[str, dict[str, Any]], source_by_path: Mapping[tuple[str, str], dict[str, Any]]) -> None:
    rows = [row for row in results if row["evidence_id"] == "table:16"]
    for (method, model), cells in grouped(rows, ("method", "model")).items():
        if all(cell.get("displayed_marker") is not None or cell.get("numeric_value") is None for cell in cells):
            for cell in cells:
                admissions[cell["result_id"]].update({"status": "not_applicable_marker", "admission_scope": "none", "reason_codes": ["paper_en_dash_has_no_inference_output"]})
            continue
        method_key = t16_method(method)
        model_key = canonical_model(model)
        for setting, aggregate, mapping in (
            ("dynamic-schema:context-free", aggs["t16_single"], SINGLE3),
            ("dynamic-schema:context-guided", aggs["t16_multi"], {"P-EM": "pref_em", "EA-F1": "nonpref_f1", "OA-F1": "overall_f1"}),
        ):
            subset = [cell for cell in cells if cell["setting"] == setting]
            semantic = [(number, row) for number, row in enumerate(aggregate["rows"], 2) if row["method"] == method_key and canonical_model(row["action_model_base"]) == model_key and row["status"] == "ok"]
            exact = [(number, row) for number, row in semantic if metric_exact(subset, row, mapping)]
            drift: dict[str, Any] | None = None
            if len(exact) == 1:
                number, row = exact[0]
            elif method_key == "rag" and setting == "dynamic-schema:context-guided":
                low = [(number, row) for number, row in semantic if row["reasoning_effort"] == "low"]
                if len(low) != 1: raise AdmissionError(f"Table16 RAG semantic row is not unique: {model}")
                number, row = low[0]
                drift = {}
                for cell in subset:
                    precision = int(cell["rounding_rule"]["display_precision"])
                    actual = rounded(row[mapping[cell["metric"]]], Decimal(100), precision)
                    expected = paper_decimal(cell)
                    if actual != expected: drift[cell["metric"]] = {"paper": str(expected), "aggregate": str(actual), "delta_percentage_points": str(actual - expected)}
            else:
                raise AdmissionError(f"Table16 row match is not unique: {method}/{model}/{setting}, exact={len(exact)}")
            raw_path = row["output_path"]
            record = source_by_path.get(("experiments4", exp4_rel(raw_path)))
            if record is None: raise AdmissionError(f"Table16 raw absent from source-pre: {raw_path}")
            binding = {**row_node(aggregate, number, "exact_rounded_metric_tuple" if drift is None else "direct_semantic_key_with_declared_display_eval_drift"), "metric_columns": {cell["result_id"]: mapping[cell["metric"]] for cell in subset}}
            raw_ref = {"raw_node_id": f"raw:{record['record_id']}", **public_record(record, exp4_rel(raw_path)), "protected_path": raw_path, "relation": "direct_output_path_reference", "via_aggregate_node_ids": [binding["aggregate_node_id"]]}
            for cell in subset:
                cell_drift = drift.get(cell["metric"]) if drift else None
                admissions[cell["result_id"]].update({
                    "status": "admitted_direct_raw_with_declared_drift" if cell_drift else "admitted_exact_raw",
                    "admission_scope": "raw_file", "reason_codes": ["paper_vs_current_aggregate_display_drift"] if cell_drift else [],
                    "aggregate_bindings": [binding], "raw_bindings": [raw_ref],
                    "supporting_evidence": [{"kind": "fixed400_generator", **public_record(aggs["t16_builder"]["record"], aggs["t16_builder"]["rel"]), "protected_path": aggs["t16_builder"]["path"]}],
                })
                if cell_drift: admissions[cell["result_id"]]["declared_drift"] = cell_drift


def classify_remaining(results: Sequence[dict[str, Any]], admissions: Mapping[str, dict[str, Any]]) -> None:
    for result in results:
        admission = admissions[result["result_id"]]
        if admission["status"] != "unresolved": continue
        if result["evidence_id"] in {"page:16", "page:17"}:
            admission.update({"status": "non_inference_human_study_metadata", "admission_scope": "none", "reason_codes": ["human_study_metadata_not_inference_output"]})
        elif result["evidence_id"] == "table:6":
            admission.update({"status": "non_inference_dataset_statistic", "admission_scope": "none", "reason_codes": ["dataset_composition_statistic_not_inference_output"]})
        elif result["evidence_id"] == "figure:5":
            admission.update({"status": "non_inference_memory_snapshot_statistic", "admission_scope": "none", "reason_codes": ["memory_size_snapshot_not_inference_output"]})
        elif result["evidence_id"] == "page:8" and result.get("derived_from"):
            admission.update({"status": "unresolved_derived", "admission_scope": "none", "reason_codes": ["derived_from_unresolved_figure4_results"]})
        elif result["result_kind"] == "derived_value":
            admission.update({"status": "unresolved_derived", "admission_scope": "none", "reason_codes": ["derived_value_lacks_admitted_producer_chain"]})


def attest_inputs() -> list[dict[str, Any]]:
    attestations = []
    for path, expected in CORE_HASHES.items():
        actual = sha256_file(path)
        if actual != expected: raise AdmissionError(f"authenticated input hash drift: {path}")
        attestations.append({"path": str(path), "sha256": actual})
    return attestations


def build() -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    input_attestations = attest_inputs()
    source_records = load_jsonl(SOURCE_PRE)
    source_by_path, source_by_id = source_index(source_records)
    lineage = lineage_index((CODE_LINEAGE, CONFIG_LINEAGE))
    results = load_jsonl(PAPER_RESULTS)
    if len(results) != 1908 or len({row["result_id"] for row in results}) != 1908:
        raise AdmissionError("paper inventory cardinality/uniqueness drift")
    admissions = {row["result_id"]: new_admission(row) for row in results}

    needed_csvs = [key for key, path in TABLE_FILES.items() if path.suffix == ".csv"]
    aggs = {key: make_aggregate(TABLE_FILES[key], source_by_path, lineage) for key in needed_csvs}
    for key in ("t15_builder", "t16_builder"):
        rel = exp4_rel(TABLE_FILES[key]); record = source_by_path[("experiments4", rel)]
        if sha256_file(TABLE_FILES[key]) != record["sha256"]: raise AdmissionError(f"builder hash drift: {TABLE_FILES[key]}")
        aggs[key] = {"path": str(TABLE_FILES[key]), "rel": rel, "record": record, "variants": variant_labels(record, lineage)}
    summary_rel = exp4_rel(TABLE_FILES["t15_summary"]); summary_record = source_by_path[("experiments4", summary_rel)]
    if sha256_file(TABLE_FILES["t15_summary"]) != summary_record["sha256"]: raise AdmissionError("Table15 summary hash drift")
    aggs["t15_summary_record"] = summary_record
    for key, expected_id in EXPECTED_RECORD_IDS.items():
        path = TABLE_FILES[key]; record = source_by_path[("experiments4", exp4_rel(path))]
        if record["record_id"] != expected_id: raise AdmissionError(f"record-id drift for {key}")

    bind_tables_11_13(results, admissions, aggs, source_by_path, source_by_id)
    bind_table3(results, admissions, aggs["t3_witness"], source_by_path, source_by_id, lineage)
    bind_table3_prefine_parents(results, admissions)
    bind_table3_prefine_cg_full_precision(results, admissions, aggs, source_by_path, source_by_id)
    bind_table3_prefine_cf_full_precision(results, admissions, aggs, source_by_path, source_by_id)
    bind_table3_higher_derivations(results, admissions)
    # Classify non-inference Figure 5 parents before resolving their Page 8
    # derived statistics. The final call remains for other still-unbound rows.
    classify_remaining(results, admissions)
    bind_figure4(results, admissions, aggs, source_by_path, source_by_id)
    bind_table10(results, admissions)
    bind_table14(results, admissions, aggs, source_by_path)
    t15_stats = bind_table15(results, admissions, aggs, source_by_path)
    bind_table16(results, admissions, aggs, source_by_path)
    classify_remaining(results, admissions)
    ordered = [admissions[row["result_id"]] for row in sorted(results, key=lambda item: item["result_id"])]

    evidence_counts = Counter(item["paper_result"]["evidence_id"] for item in ordered)
    expected_evidence = {"figure:4": 34, "figure:5": 45, "page:8": 9, "page:16": 1, "page:17": 9, "table:3": 434, "table:6": 8, "table:10": 144, "table:11": 504, "table:12": 336, "table:13": 252, "table:14": 54, "table:15": 18, "table:16": 60}
    if dict(evidence_counts) != expected_evidence: raise AdmissionError(f"paper evidence coverage drift: {dict(evidence_counts)}")
    status_counts = Counter(item["status"] for item in ordered)
    group_status: dict[str, dict[str, Any]] = {}
    for evidence, count in sorted(evidence_counts.items()):
        subset = [item for item in ordered if item["paper_result"]["evidence_id"] == evidence]
        counts = Counter(item["status"] for item in subset)
        if all(status.startswith("admitted") or status == "not_applicable_marker" for status in counts): state = "admitted_with_explicit_markers_or_drift"
        elif any(status.startswith("admitted") for status in counts): state = "partial"
        elif all(status.startswith("non_inference") for status in counts): state = "non_inference"
        else: state = "unresolved"
        group_status[evidence] = {"inventory_results": count, "state": state, "status_counts": dict(sorted(counts.items()))}
    raw_nodes = {binding["raw_node_id"] for item in ordered for binding in item["raw_bindings"]}
    aggregate_nodes = {binding["aggregate_node_id"] for item in ordered for binding in item["aggregate_bindings"]}
    unresolved = [item for item in ordered if item["status"].startswith("unresolved")]
    summary = {
        "schema": "experiments7-g3-admission-summary/v1",
        "policy": {
            "candidate_map_boundary": "metadata-only_not-admitted",
            "path_or_hash_similarity_alone_is_admission": False,
            "required_for_raw_admission": "exact aggregate metric/content evidence plus direct raw reference, or explicit recomputation proof",
            "protected_sources": ["/data/minseo/experiments4", "/data/minseo/experiments5", "/data/minseo/experiments6"],
            "protected_sources_mutated": False,
            "raw_payloads_copied": False,
        },
        "input_attestations": input_attestations,
        "inventory_results": len(ordered),
        "status_counts": dict(sorted(status_counts.items())),
        "group_status": group_status,
        "graph_counts": {"paper_result_nodes": len(ordered), "aggregate_row_nodes": len(aggregate_nodes), "raw_file_nodes": len(raw_nodes), "unresolved_result_nodes": len(unresolved)},
        "table15_verified_direct_lineage": t15_stats,
        "variant_rule": "registered lineage variant_id where exact; otherwise unregistered:<root>:<record-prefix>. Distinct eval4 parser outputs remain distinct.",
    }
    return ordered, summary, unresolved


def jsonl_payload(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8")


def artifact_claim(relative: str, payload: bytes, records: int | None = None) -> dict[str, Any]:
    claim: dict[str, Any] = {"path": relative, "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
    if records is not None:
        claim["records"] = records
    return claim


def contract_state(admission: Mapping[str, Any]) -> str:
    legacy = admission["status"]
    if legacy.startswith("admitted") and admission["raw_bindings"]:
        if any(token in legacy for token in ("derived", "raw_set", "semantic_raw_set")):
            return "DERIVED_FROM_ADMITTED_RAW"
        return "ADMITTED_FOR_COPY"
    if legacy.startswith("non_inference") or legacy == "aggregate_only_dataset_reference":
        return "NON_INFERENCE"
    if legacy == "not_applicable_marker":
        return "NOT_APPLICABLE"
    return "UNRESOLVED"


UNRESOLVED_DIAGNOSTIC_FIELDS = (
    "candidate_count",
    "distinct_bound_raw_count",
    "unbound_candidates",
    "bound_candidate_count",
    "context_candidate_counts",
    "exact_combination_count",
    "tolerance_combination_count",
    "fully_bound_qualifying_combination_count",
    "unbound_qualifying_combination_count",
    "best_max_delta_percentage_points",
    "aggregate_value",
    "producer_value",
    "paper_value",
    "delta_percentage_points",
    "parent_count",
)


def unresolved_contract_diagnostics(
    admission: Mapping[str, Any], candidate_raw_ids: Sequence[str]
) -> dict[str, Any]:
    """Serialize candidate diagnostics without promoting candidates to raw admission."""
    diagnostics = {
        key: admission[key]
        for key in UNRESOLVED_DIAGNOSTIC_FIELDS
        if key in admission
    }
    candidate_ids = sorted(set(candidate_raw_ids))
    if candidate_ids:
        diagnostics["candidate_raw_artifact_ids"] = candidate_ids
    return diagnostics


def precopy_gate(state_counts: Mapping[str, int], result_count: int) -> tuple[str, bool]:
    """Return the strict fail-closed V6 precopy publication decision.

    Historical G3 states such as DERIVED_FROM_ADMITTED_RAW and NON_INFERENCE
    are intentionally not success states.  A strict producer must normalize
    every closed result, including zero-raw non-inference results, to
    ADMITTED_FOR_COPY before this gate can pass.
    """
    if any(
        not isinstance(count, int) or isinstance(count, bool) or count < 0
        for count in state_counts.values()
    ):
        return "BLOCKED", False
    illegal_nonzero = {
        state
        for state, count in state_counts.items()
        if count and state not in {"ADMITTED_FOR_COPY", "UNRESOLVED"}
    }
    complete = (
        result_count == 1908
        and sum(state_counts.values()) == result_count
        and not illegal_nonzero
        and state_counts.get("ADMITTED_FOR_COPY", 0) == result_count
        and state_counts.get("UNRESOLVED", 0) == 0
    )
    return ("PASS" if complete else "BLOCKED", complete)


def edge_record(source: str, target: str, relation: str, result_id: str) -> dict[str, Any]:
    material = {"source_node_id": source, "target_node_id": target, "relation": relation, "paper_result_id": result_id}
    digest = hashlib.sha256(canonical_json(material).encode("utf-8")).hexdigest()
    return {"schema": "experiments7-provenance-edge/v1", "edge_id": f"edge:{digest}", **material}


def compose_contract(admissions: Sequence[dict[str, Any]], internal_summary: Mapping[str, Any]) -> tuple[dict[Path, bytes], dict[str, Any], dict[str, Any]]:
    source_records = load_jsonl(SOURCE_PRE)
    source_by_id = {row["record_id"]: row for row in source_records}
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[str, dict[str, Any]] = {}
    recomputations: dict[str, dict[str, Any]] = {}

    def add_edge(source: str, target: str, relation: str, result_id: str, bucket: list[str]) -> None:
        edge = edge_record(source, target, relation, result_id)
        edges[edge["edge_id"]] = edge
        bucket.append(edge["edge_id"])

    for admission in admissions:
        result = admission["paper_result"]
        paper_node_id = f"paper:{result['result_id']}"
        nodes[paper_node_id] = {
            "schema": "experiments7-provenance-node/v1", "node_id": paper_node_id,
            "node_type": "paper_result", "paper_result": result,
        }

    result_rows: list[dict[str, Any]] = []
    state_counts = {state: 0 for state in ("ADMITTED_FOR_COPY", "DERIVED_FROM_ADMITTED_RAW", "NON_INFERENCE", "NOT_APPLICABLE", "UNRESOLVED")}
    evidence_state_counts: dict[str, Counter[str]] = defaultdict(Counter)
    basis_counts: Counter[str] = Counter()
    raw_copy_ids: set[str] = set()
    for admission in admissions:
        result = admission["paper_result"]
        result_id = result["result_id"]
        paper_node_id = f"paper:{result_id}"
        validating_nodes = [paper_node_id]
        validating_edges: list[str] = []
        for binding in admission["aggregate_bindings"]:
            node_id = binding["aggregate_node_id"]
            nodes[node_id] = {
                "schema": "experiments7-provenance-node/v1", "node_id": node_id,
                "node_type": "aggregate_row", "source_manifest_record_id": binding["source_manifest_record_id"],
                "root_id": binding["root_id"], "relative_path_b64": base64.b64encode(binding["relative_path"].encode("utf-8")).decode("ascii"),
                "sha256": binding["sha256"], "size": binding["size"], "row_number": binding["row_number"],
                "relation": binding["relation"], "metric_columns": binding.get("metric_columns", {}),
                "aggregate_values": binding.get("aggregate_values", {}), "scale": binding.get("scale"),
                "variant_labels": binding.get("variant_labels", []),
            }
            validating_nodes.append(node_id)
            add_edge(node_id, paper_node_id, "aggregate_supports_paper_result", result_id, validating_edges)
        raw_ids: list[str] = []
        for binding in admission["raw_bindings"]:
            node_id = binding["raw_node_id"]
            manifest = source_by_id.get(binding["source_manifest_record_id"])
            if manifest is None or manifest.get("type") != "regular":
                raise AdmissionError(f"raw node lost source-pre binding: {node_id}")
            relation_types = {binding["relation"]}
            if node_id in nodes:
                relation_types.update(nodes[node_id].get("relation_types", []))
            nodes[node_id] = {
                "schema": "experiments7-provenance-node/v1", "node_id": node_id,
                "node_type": "raw_artifact", "raw_kind": "raw_inference",
                "source_manifest_record_id": manifest["record_id"], "root_id": manifest["root_id"],
                "relative_path_b64": manifest["relative_path_b64"], "sha256": manifest["sha256"], "size": manifest["size"],
                "relation_types": sorted(relation_types),
            }
            validating_nodes.append(node_id)
            raw_ids.append(node_id)
            add_edge(node_id, paper_node_id, binding["relation"], result_id, validating_edges)
            for aggregate_id in sorted(set(binding.get("via_aggregate_node_ids", [])) & set(validating_nodes)):
                add_edge(node_id, aggregate_id, "raw_supports_aggregate", result_id, validating_edges)
            if binding.get("recomputation_evidence"):
                key = hashlib.sha256(canonical_json(binding["recomputation_evidence"]).encode("utf-8")).hexdigest()
                row = recomputations.setdefault(key, {
                    "schema": "experiments7-provenance-recomputation/v1", "recomputation_id": f"recomputation:{key}",
                    "raw_node_id": node_id, "evidence": binding["recomputation_evidence"], "paper_result_ids": [],
                })
                row["paper_result_ids"].append(result_id)
        for support in admission.get("supporting_evidence", []):
            if support.get("kind") == "derived_from_paper_result" and support.get("result_id"):
                parent_id = f"paper:{support['result_id']}"
                if parent_id in nodes:
                    validating_nodes.append(parent_id)
                    add_edge(parent_id, paper_node_id, "paper_result_derives_paper_result", result_id, validating_edges)
                continue
            record_id = support.get("source_manifest_record_id")
            manifest = source_by_id.get(record_id)
            if manifest is None:
                continue
            node_id = f"support:{record_id}"
            nodes[node_id] = {
                "schema": "experiments7-provenance-node/v1", "node_id": node_id,
                "node_type": "supporting_artifact", "artifact_kind": support.get("kind", "support"),
                "source_manifest_record_id": record_id, "root_id": manifest["root_id"],
                "relative_path_b64": manifest["relative_path_b64"], "sha256": manifest.get("sha256"), "size": manifest.get("size"),
            }
            validating_nodes.append(node_id)
            add_edge(node_id, paper_node_id, "supporting_artifact_supports_paper_result", result_id, validating_edges)
        state = contract_state(admission)
        state_counts[state] += 1
        evidence_state_counts[result["evidence_id"]][state] += 1
        basis_counts[admission["status"]] += 1
        candidate_raw_ids = sorted(set(raw_ids))
        if state in {"ADMITTED_FOR_COPY", "DERIVED_FROM_ADMITTED_RAW"}:
            if not raw_ids:
                raise AdmissionError(f"copy-eligible result lacks raw binding: {result_id}")
            reasons: list[str] = []
            raw_copy_ids.update(raw_ids)
        else:
            raw_ids = []
            reasons = sorted(set(admission.get("reason_codes", []))) or [f"g3_{admission['status']}"]
        row: dict[str, Any] = {
            "schema": "experiments7-admission-result/v1", "result_id": result_id, "derived_state": state,
            "reason_codes": reasons, "raw_artifact_ids": sorted(set(raw_ids)),
            "validating_node_ids": sorted(set(validating_nodes)), "validating_edge_ids": sorted(set(validating_edges)),
            "evidence_id": result["evidence_id"], "classification_basis": admission["status"],
        }
        if admission.get("declared_drift"):
            row["declared_drift"] = admission["declared_drift"]
        if admission.get("derivation"):
            row["derivation"] = admission["derivation"]
        if state == "UNRESOLVED":
            row.update(unresolved_contract_diagnostics(admission, candidate_raw_ids))
        result_rows.append(row)

    for row in recomputations.values():
        row["paper_result_ids"] = sorted(set(row["paper_result_ids"]))
    node_rows = [nodes[key] for key in sorted(nodes)]
    edge_rows = [edges[key] for key in sorted(edges)]
    recomputation_rows = [recomputations[key] for key in sorted(recomputations)]
    result_rows.sort(key=lambda row: row["result_id"])
    nodes_payload = jsonl_payload(node_rows)
    edges_payload = jsonl_payload(edge_rows)
    recomputations_payload = jsonl_payload(recomputation_rows)
    results_payload = jsonl_payload(result_rows)

    produced_inputs = {
        "provenance_nodes": nodes_payload, "provenance_edges": edges_payload,
        "provenance_recomputations": recomputations_payload,
    }
    input_claims: dict[str, dict[str, Any]] = {}
    input_hashes: dict[str, str] = {}
    for key, relative in CONTRACT_INPUT_PATHS.items():
        payload = produced_inputs.get(key)
        if payload is None:
            payload = (ROOT7 / relative).read_bytes()
        records = len(payload.splitlines()) if relative.endswith(".jsonl") else None
        claim = artifact_claim(relative, payload, records)
        input_claims[key] = claim
        input_hashes[key] = claim["sha256"]

    sealed_run_id = "g3-precopy-" + hashlib.sha256(nodes_payload + edges_payload + results_payload).hexdigest()[:20]
    copy_eligible = state_counts["ADMITTED_FOR_COPY"] + state_counts["DERIVED_FROM_ADMITTED_RAW"]
    terminal_state, copy_admission_complete = precopy_gate(state_counts, len(result_rows))
    summary = {
        "schema": "experiments7-admission-summary/v1", "phase": "precopy", "terminal_state": terminal_state,
        "sealed_run_id": sealed_run_id, "ledger_coverage_pass": len(result_rows) == len({row["result_id"] for row in result_rows}) == 1908,
        "classification_complete": True, "copy_admission_complete": copy_admission_complete,
        "unresolved_count": state_counts["UNRESOLVED"], "total_result_count": len(result_rows),
        "admission_result_records": len(result_rows), "copy_eligible_result_count": copy_eligible,
        "results_with_raw_count": copy_eligible, "raw_artifact_count": len(raw_copy_ids),
        "result_state_counts": state_counts, "input_hashes": input_hashes,
        "evidence_state_counts": {key: dict(sorted(value.items())) for key, value in sorted(evidence_state_counts.items())},
        "classification_basis_counts": dict(sorted(basis_counts.items())),
        "declared_drift_result_count": sum(1 for row in result_rows if "declared_drift" in row),
        "graph_counts": {"nodes": len(node_rows), "edges": len(edge_rows), "recomputations": len(recomputation_rows)},
        "policy": internal_summary["policy"], "group_status": internal_summary["group_status"],
    }
    summary_payload = (json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    results_relative = "paper_outputs/admission/precopy/results.jsonl"
    summary_relative = "paper_outputs/admission/precopy/summary.json"
    seal = {
        "schema": "experiments7-admission-seal/v1", "phase": "precopy", "terminal_state": terminal_state,
        "sealed_run_id": sealed_run_id, "protected_envelope_id": "g0-provider-readonly-envelope-v1",
        "agent_type": "test-engineer", "admission_task_id": "/root/controller7_exact3_arch_review",
        "artifacts": {
            "results": artifact_claim(results_relative, results_payload, len(result_rows)),
            "summary": artifact_claim(summary_relative, summary_payload),
        },
        "inputs": input_claims,
    }
    seal_payload = (json.dumps(seal, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    readme_lines = [
        "# Paper-output provenance\n",
        "This directory records the fail-closed mapping from each paper result to aggregate evidence and raw inference artifacts.\n",
        "- `ADMITTED_FOR_COPY`: direct content/metric provenance to one or more raw inference files.\n",
        "- `DERIVED_FROM_ADMITTED_RAW`: a paper value/mark derived from an admitted raw set; copying is deduplicated by raw node.\n",
        "- Declared drift remains explicitly labeled in `results.jsonl`; it is not claimed as exact.\n",
        "- `NON_INFERENCE` and `NOT_APPLICABLE`: no raw inference file is copied.\n",
        "- `UNRESOLVED`: evidence was insufficient or ambiguous, so no raw file is admitted.\n",
        "Raw node `raw:<source_manifest_record_id>` maps to `paper_outputs/raw/verified/<root_id>/<decoded relative_path_b64>`.\n",
        "## Counts by evidence\n",
    ]
    for evidence, counts in sorted(summary["evidence_state_counts"].items()):
        readme_lines.append(f"- `{evidence}`: " + ", ".join(f"{state}={count}" for state, count in sorted(counts.items())) + "\n")
    files = {
        PROVENANCE_OUTPUT / "nodes.jsonl": nodes_payload,
        PROVENANCE_OUTPUT / "edges.jsonl": edges_payload,
        PROVENANCE_OUTPUT / "recomputations.jsonl": recomputations_payload,
        PROVENANCE_OUTPUT / "README.md": "".join(readme_lines).encode("utf-8"),
        DEFAULT_OUTPUT / "results.jsonl": results_payload,
        DEFAULT_OUTPUT / "summary.json": summary_payload,
        DEFAULT_OUTPUT / "seal.json": seal_payload,
    }
    return files, summary, seal


def publish(files: Mapping[Path, bytes]) -> None:
    """No-replace, idempotent publication; the seal is linked last."""
    ordered = sorted((item for item in files.items() if item[0].name != "seal.json"), key=lambda item: str(item[0]))
    ordered.extend(item for item in files.items() if item[0].name == "seal.json")
    for path, payload in ordered:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if not path.is_file() or path.is_symlink() or path.read_bytes() != payload:
                raise AdmissionError(f"existing publication artifact differs: {path}")
            continue
        temp = path.with_name(f".{path.name}.g3-{os.getpid()}.tmp")
        try:
            with temp.open("xb") as handle:
                handle.write(payload); handle.flush(); os.fsync(handle.fileno())
            try:
                os.link(temp, path)
            except FileExistsError:
                if not path.is_file() or path.is_symlink() or path.read_bytes() != payload:
                    raise AdmissionError(f"publication race produced different bytes: {path}")
            directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try: os.fsync(directory_fd)
            finally: os.close(directory_fd)
        finally:
            if temp.exists(): temp.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--publish", action="store_true", help="write new G3 admission artifacts")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output != DEFAULT_OUTPUT:
        raise AdmissionError(f"output is restricted to {DEFAULT_OUTPUT}")
    apply_protected_source_envelope()
    admissions, internal_summary, unresolved = build()
    files, summary, seal = compose_contract(admissions, internal_summary)
    if args.publish:
        if summary["terminal_state"] != "PASS":
            raise AdmissionError(
                f"precopy publication blocked: unresolved_count={summary['unresolved_count']}"
            )
        publish(files)
    seal_payload = files[DEFAULT_OUTPUT / "seal.json"]
    print(json.dumps({"ok": summary["terminal_state"] == "PASS", "published": args.publish, "inventory_results": len(admissions), "unresolved_results": len(unresolved), "result_state_counts": summary["result_state_counts"], "raw_artifact_count": summary["raw_artifact_count"], "graph_counts": summary["graph_counts"], "seal_sha256": hashlib.sha256(seal_payload).hexdigest(), "table15": internal_summary["table15_verified_direct_lineage"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AdmissionError, KeyError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
