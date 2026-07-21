#!/usr/bin/env python3

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


EXTENDED_ROOT = Path("/data/minseo/experiments4/extended_schema")
EVAL_ROOT = Path("/data/minseo/experiments4/ours_memory/session_memory_eval_0312")
PREF_LIST_PATH = Path("/data/minseo/experiments4/pref_list_extended.json")

if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from evaluate_session_memory_results_multiturn import (  # type: ignore
    evaluate_examples as evaluate_multiturn_examples,
    load_multiturn_eval_module,
)
from evaluate_session_memory_results_singleturn import (  # type: ignore
    eval_one_file as eval_singleturn_file,
    extract_examples as extract_singleturn_examples,
    load_json as load_singleturn_json,
)


METHOD_ORDER = {
    "vanilla_llm": 0,
    "rag": 1,
    "mem0": 2,
    "langmem": 3,
    "ours_memory": 4,
}

EFFORT_SUFFIXES = ("minimal", "low", "medium", "high", "default")

SINGLE_FIELDNAMES = [
    "method",
    "source",
    "memory_model",
    "context_type",
    "query_variant",
    "pref_type",
    "action_model_raw",
    "action_model_base",
    "reasoning_effort",
    "prompt_name",
    "file_name",
    "group_key",
    "query_count",
    "status",
    "error",
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1",
    "parsing_fail_count",
    "output_path",
    "rel_path",
]

MULTI_FIELDNAMES = [
    "method",
    "source",
    "memory_model",
    "context_type",
    "query_variant",
    "pref_type",
    "action_model_raw",
    "action_model_base",
    "reasoning_effort",
    "prompt_name",
    "file_name",
    "group_key",
    "query_count",
    "status",
    "error",
    "overall_tp",
    "overall_fp",
    "overall_fn",
    "overall_precision",
    "overall_recall",
    "overall_f1",
    "pref_em",
    "pref_em_count",
    "n_examples",
    "nonpref_tp",
    "nonpref_fp",
    "nonpref_fn",
    "nonpref_precision",
    "nonpref_recall",
    "nonpref_f1",
    "parsing_fail_count",
    "output_path",
    "rel_path",
]


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def as_posix(path: Path) -> str:
    return path.as_posix()


def split_action_model(raw: str) -> tuple[str, str]:
    for suffix in EFFORT_SUFFIXES:
        for sep in ("_", "-"):
            token = f"{sep}{suffix}"
            if raw.endswith(token):
                return raw[: -len(token)], suffix
    return raw, ""


def iter_final_jsons(root: Path) -> List[Path]:
    files: List[Path] = []
    if not root.exists():
        return files

    for json_path in sorted(root.rglob("*.json")):
        rel_parts = json_path.relative_to(root).parts
        if any(part.startswith(".") for part in rel_parts):
            continue
        if "fixed_400" not in json_path.name:
            continue
        files.append(json_path)
    return files


def parse_single_meta(method: str, task_root: Path, json_path: Path) -> Dict[str, str]:
    rel_parts = json_path.relative_to(task_root).parts

    if method in {"vanilla_llm", "rag"}:
        if len(rel_parts) != 7:
            raise ValueError(f"Unexpected {method} singleturn layout: {json_path}")
        source, context_type, pref_type, query_variant, action_model_raw, prompt_name, file_name = rel_parts
        memory_model = ""
    elif method in {"langmem", "ours_memory"}:
        if len(rel_parts) != 7:
            raise ValueError(f"Unexpected {method} singleturn layout: {json_path}")
        source, memory_model, context_type, pref_type, action_model_raw, prompt_name, file_name = rel_parts
        query_variant = "singleturn"
    elif method == "mem0":
        if len(rel_parts) != 6:
            raise ValueError(f"Unexpected mem0 singleturn layout: {json_path}")
        source, context_type, pref_type, action_model_raw, prompt_name, file_name = rel_parts
        memory_model = ""
        query_variant = "singleturn"
    else:
        raise ValueError(f"Unsupported method: {method}")

    action_model_base, reasoning_effort = split_action_model(action_model_raw)
    rel_path = json_path.relative_to(EXTENDED_ROOT)
    return {
        "method": method,
        "source": source,
        "memory_model": memory_model,
        "context_type": context_type,
        "query_variant": query_variant,
        "pref_type": pref_type,
        "action_model_raw": action_model_raw,
        "action_model_base": action_model_base,
        "reasoning_effort": reasoning_effort,
        "prompt_name": prompt_name,
        "file_name": file_name,
        "group_key": as_posix(rel_path.parent),
        "output_path": str(json_path),
        "rel_path": as_posix(rel_path),
    }


def parse_multi_meta(method: str, task_root: Path, json_path: Path) -> Dict[str, str]:
    rel_parts = json_path.relative_to(task_root).parts

    if method in {"vanilla_llm", "rag"}:
        if len(rel_parts) != 7:
            raise ValueError(f"Unexpected {method} multiturn layout: {json_path}")
        source, context_type, pref_type, query_variant, action_model_raw, prompt_name, file_name = rel_parts
        memory_model = ""
    elif method in {"langmem", "ours_memory"}:
        if len(rel_parts) != 7:
            raise ValueError(f"Unexpected {method} multiturn layout: {json_path}")
        source, memory_model, context_type, pref_type, action_model_raw, prompt_name, file_name = rel_parts
        query_variant = "multiturn"
    elif method == "mem0":
        if len(rel_parts) != 6:
            raise ValueError(f"Unexpected mem0 multiturn layout: {json_path}")
        source, context_type, pref_type, action_model_raw, prompt_name, file_name = rel_parts
        memory_model = ""
        query_variant = "multiturn"
    else:
        raise ValueError(f"Unsupported method: {method}")

    action_model_base, reasoning_effort = split_action_model(action_model_raw)
    rel_path = json_path.relative_to(EXTENDED_ROOT)
    return {
        "method": method,
        "source": source,
        "memory_model": memory_model,
        "context_type": context_type,
        "query_variant": query_variant,
        "pref_type": pref_type,
        "action_model_raw": action_model_raw,
        "action_model_base": action_model_base,
        "reasoning_effort": reasoning_effort,
        "prompt_name": prompt_name,
        "file_name": file_name,
        "group_key": as_posix(rel_path.parent),
        "output_path": str(json_path),
        "rel_path": as_posix(rel_path),
    }


def evaluate_single_method(method: str, task_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for json_path in iter_final_jsons(task_root):
        meta = parse_single_meta(method, task_root, json_path)
        try:
            data = load_singleturn_json(json_path)
            query_count = len(extract_singleturn_examples(data))
            metrics, parsing_fail_count = eval_singleturn_file(str(json_path))
            status = "ok"
            error = ""
        except Exception as exc:
            query_count = 0
            metrics = None
            parsing_fail_count = 0
            status = "error"
            error = repr(exc)

        rows.append(
            {
                **meta,
                "query_count": query_count,
                "status": status,
                "error": error,
                "tp": 0 if metrics is None else metrics.tp,
                "fp": 0 if metrics is None else metrics.fp,
                "fn": 0 if metrics is None else metrics.fn,
                "precision": 0.0 if metrics is None else metrics.precision,
                "recall": 0.0 if metrics is None else metrics.recall,
                "f1": 0.0 if metrics is None else metrics.f1,
                "parsing_fail_count": parsing_fail_count,
            }
        )
    return rows


def evaluate_multi_method(method: str, task_root: Path, module: Any, pref_map: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for json_path in iter_final_jsons(task_root):
        meta = parse_multi_meta(method, task_root, json_path)
        try:
            data = module.load_json(json_path)
            examples = module.extract_examples(data)
            query_count = len(examples)
            overall_prf, pref_emr, nonpref_prf, parsing_fail_count = evaluate_multiturn_examples(
                module,
                examples,
                pref_map,
            )
            status = "ok"
            error = ""
        except Exception as exc:
            query_count = 0
            overall_prf = module.PRF(0.0, 0.0, 0.0, 0, 0, 0)
            pref_emr = module.EMR(0.0, 0, 0)
            nonpref_prf = module.PRF(0.0, 0.0, 0.0, 0, 0, 0)
            parsing_fail_count = 0
            status = "error"
            error = repr(exc)

        rows.append(
            {
                **meta,
                "query_count": query_count,
                "status": status,
                "error": error,
                "overall_tp": overall_prf.tp,
                "overall_fp": overall_prf.fp,
                "overall_fn": overall_prf.fn,
                "overall_precision": overall_prf.precision,
                "overall_recall": overall_prf.recall,
                "overall_f1": overall_prf.f1,
                "pref_em": pref_emr.em,
                "pref_em_count": pref_emr.em_count,
                "n_examples": pref_emr.n,
                "nonpref_tp": nonpref_prf.tp,
                "nonpref_fp": nonpref_prf.fp,
                "nonpref_fn": nonpref_prf.fn,
                "nonpref_precision": nonpref_prf.precision,
                "nonpref_recall": nonpref_prf.recall,
                "nonpref_f1": nonpref_prf.f1,
                "parsing_fail_count": parsing_fail_count,
            }
        )
    return rows


def build_singleturn_rows() -> List[Dict[str, Any]]:
    configs = [
        ("vanilla_llm", EXTENDED_ROOT / "vanillaLLM" / "output" / "singleturn"),
        ("rag", EXTENDED_ROOT / "RAG" / "output" / "singleturn"),
        ("mem0", EXTENDED_ROOT / "mem0" / "output" / "singleturn"),
        ("langmem", EXTENDED_ROOT / "langmem" / "output" / "singleturn"),
        ("ours_memory", EXTENDED_ROOT / "ours_memory" / "output" / "singleturn"),
    ]
    rows: List[Dict[str, Any]] = []
    for method, task_root in configs:
        rows.extend(evaluate_single_method(method, task_root))
    rows.sort(
        key=lambda row: (
            -float(row["f1"]),
            METHOD_ORDER.get(str(row["method"]), 999),
            str(row["source"]),
            str(row["action_model_base"]),
            str(row["reasoning_effort"]),
            str(row["memory_model"]),
        )
    )
    return rows


def build_multiturn_rows() -> List[Dict[str, Any]]:
    module = load_multiturn_eval_module()
    pref_map = module.load_pref_list(str(PREF_LIST_PATH))
    configs = [
        ("vanilla_llm", EXTENDED_ROOT / "vanillaLLM" / "output" / "multiturn"),
        ("rag", EXTENDED_ROOT / "RAG" / "output" / "multiturn"),
        ("mem0", EXTENDED_ROOT / "mem0" / "output" / "multiturn"),
        ("langmem", EXTENDED_ROOT / "langmem" / "output" / "multiturn"),
        ("ours_memory", EXTENDED_ROOT / "ours_memory" / "output" / "multiturn"),
    ]
    rows: List[Dict[str, Any]] = []
    for method, task_root in configs:
        rows.extend(evaluate_multi_method(method, task_root, module, pref_map))
    rows.sort(
        key=lambda row: (
            -float(row["overall_f1"]),
            METHOD_ORDER.get(str(row["method"]), 999),
            str(row["source"]),
            str(row["action_model_base"]),
            str(row["reasoning_effort"]),
            str(row["memory_model"]),
        )
    )
    return rows


def main() -> None:
    single_rows = build_singleturn_rows()
    multi_rows = build_multiturn_rows()

    single_csv = EXTENDED_ROOT / "singleturn_comparison.csv"
    multi_csv = EXTENDED_ROOT / "multiturn_comparison.csv"
    write_csv(single_csv, SINGLE_FIELDNAMES, single_rows)
    write_csv(multi_csv, MULTI_FIELDNAMES, multi_rows)

    print(f"[DONE] singleturn rows={len(single_rows)} -> {single_csv}")
    print(f"[DONE] multiturn rows={len(multi_rows)} -> {multi_csv}")


if __name__ == "__main__":
    main()
