#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import importlib.util
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_ROOT_DIR = Path("/data/minseo/experiments6/ours_memory/inference/1231_MEMORY3_inference2_multi")
DEFAULT_PREF_LIST_PATH = Path("/data/minseo/experiments6/pref_list.json")
EVAL_SCRIPT_PATH = Path("/data/minseo/experiments6/evaluation/evaluation_multiturn-f1-parse-aggregate.py")
PREF_ORDER = ["easy", "medium", "hard"]


DETAIL_FIELDNAMES = [
    "run_tag",
    "memory_model",
    "context_type",
    "pref_type",
    "action_model",
    "prompt_name",
    "file_name",
    "json_path",
    "rel_path",
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
]


SUMMARY_FIELDNAMES = [
    "run_tag",
    "memory_model",
    "context_type",
    "pref_type",
    "action_model",
    "prompt_name",
    "files_in_group",
    "ok_files",
    "error_files",
    "status",
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
]


ACTION_TABLE_FIELDNAMES = [
    "action_model",
    "context_type",
    "prompt_name",
    "overall_precision",
    "overall_recall",
    "overall_f1",
    "easy_precision",
    "easy_recall",
    "easy_f1",
    "medium_precision",
    "medium_recall",
    "medium_f1",
    "hard_precision",
    "hard_recall",
    "hard_f1",
    "overall_pref_em",
    "easy_pref_em",
    "medium_pref_em",
    "hard_pref_em",
    "overall_nonpref_precision",
    "overall_nonpref_recall",
    "overall_nonpref_f1",
    "parsing_fail_count",
    "n_examples",
    "memory_model_count",
]


MEMORY_TABLE_FIELDNAMES = [
    "memory_model",
    "context_type",
    "prompt_name",
    "overall_precision",
    "overall_recall",
    "overall_f1",
    "easy_precision",
    "easy_recall",
    "easy_f1",
    "medium_precision",
    "medium_recall",
    "medium_f1",
    "hard_precision",
    "hard_recall",
    "hard_f1",
    "overall_pref_em",
    "easy_pref_em",
    "medium_pref_em",
    "hard_pref_em",
    "overall_nonpref_precision",
    "overall_nonpref_recall",
    "overall_nonpref_f1",
    "parsing_fail_count",
    "n_examples",
    "action_model_count",
]


@dataclass
class MetricCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def safe_name(value: Any) -> str:
    text = str(value).strip()
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)
    return cleaned.strip("_") or "unnamed"


def write_csv(path: str | Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_multiturn_eval_module() -> Any:
    spec = importlib.util.spec_from_file_location("multiturn_eval_module", EVAL_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load evaluation module from {EVAL_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default=str(DEFAULT_ROOT_DIR))
    parser.add_argument("--analysis_dir", type=str, default=None)
    parser.add_argument("--pref_list_path", type=str, default=str(DEFAULT_PREF_LIST_PATH))
    return parser.parse_args()


def parse_relpath(path: Path, root_dir: Path) -> Dict[str, str]:
    parts = path.relative_to(root_dir).parts
    if len(parts) < 6:
        padded = ("",) * (6 - len(parts)) + parts
        memory_model, context_type, pref_type, action_model, prompt_name, file_name = padded[-6:]
    else:
        memory_model, context_type, pref_type, action_model, prompt_name, file_name = parts[-6:]
    return {
        "memory_model": memory_model,
        "context_type": context_type,
        "pref_type": pref_type,
        "action_model": action_model,
        "prompt_name": prompt_name,
        "file_name": file_name,
    }


def iter_result_jsons(root_dir: Path) -> List[Path]:
    return sorted(path for path in root_dir.rglob("*.json") if "logs" not in path.parts)


def evaluate_one_file(module: Any, json_path: Path, pref_map: Dict[str, set[str]]) -> Dict[str, Any]:
    data = module.load_json(str(json_path))
    examples = module.extract_examples(data)
    overall_prf = module.micro_f1_slot_and_value_or(examples)
    pref_emr = module.pref_em_rate(examples, pref_map)
    nonpref_prf = module.micro_f1_with_filter(examples, pref_map, want_pref=False)
    parsing_fail_count = sum(1 for ex in examples if module.is_parsing_failed_pred(ex)[0])
    return {
        "overall_prf": overall_prf,
        "pref_emr": pref_emr,
        "nonpref_prf": nonpref_prf,
        "parsing_fail_count": parsing_fail_count,
    }


def build_detailed_rows(root_dir: Path, module: Any, pref_map: Dict[str, set[str]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    run_tag = root_dir.name
    for json_path in iter_result_jsons(root_dir):
        meta = parse_relpath(json_path, root_dir)
        try:
            result = evaluate_one_file(module, json_path, pref_map)
            overall_prf = result["overall_prf"]
            pref_emr = result["pref_emr"]
            nonpref_prf = result["nonpref_prf"]
            parsing_fail_count = int(result["parsing_fail_count"])
            status = "ok"
            error = ""
        except Exception as exc:
            overall_prf = module.PRF(0.0, 0.0, 0.0, 0, 0, 0)
            pref_emr = module.EMR(0.0, 0, 0)
            nonpref_prf = module.PRF(0.0, 0.0, 0.0, 0, 0, 0)
            parsing_fail_count = 0
            status = "error"
            error = repr(exc)

        rows.append(
            {
                "run_tag": run_tag,
                **meta,
                "json_path": str(json_path),
                "rel_path": str(json_path.relative_to(root_dir)),
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

    rows.sort(
        key=lambda row: (
            row["run_tag"],
            row["action_model"],
            row["memory_model"],
            row["context_type"],
            row["pref_type"],
            row["prompt_name"],
            row["file_name"],
        )
    )
    return rows


def summarize_rows(rows: Sequence[Dict[str, Any]], module: Any) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, str, str, str], Dict[str, Any]] = {}
    for row in rows:
        key = (
            row["run_tag"],
            row["memory_model"],
            row["context_type"],
            row["pref_type"],
            row["action_model"],
            row["prompt_name"],
        )
        if key not in grouped:
            grouped[key] = {
                "run_tag": row["run_tag"],
                "memory_model": row["memory_model"],
                "context_type": row["context_type"],
                "pref_type": row["pref_type"],
                "action_model": row["action_model"],
                "prompt_name": row["prompt_name"],
                "files_in_group": 0,
                "ok_files": 0,
                "error_files": 0,
                "overall_tp": 0,
                "overall_fp": 0,
                "overall_fn": 0,
                "pref_em_count": 0,
                "n_examples": 0,
                "nonpref_tp": 0,
                "nonpref_fp": 0,
                "nonpref_fn": 0,
                "parsing_fail_count": 0,
            }

        group = grouped[key]
        group["files_in_group"] += 1
        if row["status"] == "ok":
            group["ok_files"] += 1
            group["overall_tp"] += int(row["overall_tp"])
            group["overall_fp"] += int(row["overall_fp"])
            group["overall_fn"] += int(row["overall_fn"])
            group["pref_em_count"] += int(row["pref_em_count"])
            group["n_examples"] += int(row["n_examples"])
            group["nonpref_tp"] += int(row["nonpref_tp"])
            group["nonpref_fp"] += int(row["nonpref_fp"])
            group["nonpref_fn"] += int(row["nonpref_fn"])
            group["parsing_fail_count"] += int(row["parsing_fail_count"])
        else:
            group["error_files"] += 1

    summary_rows: List[Dict[str, Any]] = []
    for group in grouped.values():
        overall_prf = module.prf_from_counts(group["overall_tp"], group["overall_fp"], group["overall_fn"])
        nonpref_prf = module.prf_from_counts(group["nonpref_tp"], group["nonpref_fp"], group["nonpref_fn"])
        pref_em = (group["pref_em_count"] / group["n_examples"]) if group["n_examples"] > 0 else 0.0
        status = "ok" if group["error_files"] == 0 else ("error" if group["ok_files"] == 0 else "partial")
        summary_rows.append(
            {
                **group,
                "status": status,
                "overall_precision": overall_prf.precision,
                "overall_recall": overall_prf.recall,
                "overall_f1": overall_prf.f1,
                "pref_em": pref_em,
                "nonpref_precision": nonpref_prf.precision,
                "nonpref_recall": nonpref_prf.recall,
                "nonpref_f1": nonpref_prf.f1,
            }
        )

    summary_rows.sort(
        key=lambda row: (
            row["run_tag"],
            row["action_model"],
            row["memory_model"],
            row["pref_type"],
            row["context_type"],
            row["prompt_name"],
        )
    )
    return summary_rows


def aggregate_metric_counts(rows: Iterable[Dict[str, Any]], prefix: str) -> MetricCounts:
    counts = MetricCounts()
    for row in rows:
        counts.tp += int(row[f"{prefix}_tp"])
        counts.fp += int(row[f"{prefix}_fp"])
        counts.fn += int(row[f"{prefix}_fn"])
    return counts


def build_pref_slice(rows: Sequence[Dict[str, Any]], pref_type: str) -> List[Dict[str, Any]]:
    return [row for row in rows if row["pref_type"] == pref_type]


def build_wide_table(
    summary_rows: Sequence[Dict[str, Any]],
    group_key_name: str,
    group_value_name: str,
    fieldnames: Sequence[str],
    module: Any,
) -> List[Dict[str, Any]]:
    ok_rows = [row for row in summary_rows if row["status"] == "ok"]
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in ok_rows:
        key = (row[group_key_name], row["context_type"], row["prompt_name"])
        grouped[key].append(row)

    table_rows: List[Dict[str, Any]] = []
    other_axis_name = "memory_model_count" if group_key_name == "action_model" else "action_model_count"
    other_axis_field = "memory_model" if group_key_name == "action_model" else "action_model"

    for (group_value, context_type, prompt_name), rows_in_group in sorted(grouped.items()):
        overall_counts = aggregate_metric_counts(rows_in_group, "overall")
        overall_prf = module.prf_from_counts(overall_counts.tp, overall_counts.fp, overall_counts.fn)
        nonpref_counts = aggregate_metric_counts(rows_in_group, "nonpref")
        nonpref_prf = module.prf_from_counts(nonpref_counts.tp, nonpref_counts.fp, nonpref_counts.fn)
        pref_em_count = sum(int(row["pref_em_count"]) for row in rows_in_group)
        n_examples = sum(int(row["n_examples"]) for row in rows_in_group)
        parsing_fail_count = sum(int(row["parsing_fail_count"]) for row in rows_in_group)
        row_out: Dict[str, Any] = {
            group_value_name: group_value,
            "context_type": context_type,
            "prompt_name": prompt_name,
            "overall_precision": overall_prf.precision,
            "overall_recall": overall_prf.recall,
            "overall_f1": overall_prf.f1,
            "overall_pref_em": (pref_em_count / n_examples) if n_examples > 0 else 0.0,
            "overall_nonpref_precision": nonpref_prf.precision,
            "overall_nonpref_recall": nonpref_prf.recall,
            "overall_nonpref_f1": nonpref_prf.f1,
            "parsing_fail_count": parsing_fail_count,
            "n_examples": n_examples,
            other_axis_name: len({row[other_axis_field] for row in rows_in_group}),
        }

        for pref_type in PREF_ORDER:
            pref_rows = build_pref_slice(rows_in_group, pref_type)
            pref_counts = aggregate_metric_counts(pref_rows, "overall")
            pref_prf = module.prf_from_counts(pref_counts.tp, pref_counts.fp, pref_counts.fn)
            pref_em_count = sum(int(row["pref_em_count"]) for row in pref_rows)
            pref_examples = sum(int(row["n_examples"]) for row in pref_rows)
            row_out[f"{pref_type}_precision"] = pref_prf.precision
            row_out[f"{pref_type}_recall"] = pref_prf.recall
            row_out[f"{pref_type}_f1"] = pref_prf.f1
            row_out[f"{pref_type}_pref_em"] = (pref_em_count / pref_examples) if pref_examples > 0 else 0.0

        table_rows.append({field: row_out.get(field, "") for field in fieldnames})

    return table_rows


def display_label(name: str) -> str:
    label = str(name)
    if "/" in label:
        label = label.split("/")[-1]
    if "_" in label:
        prefix, rest = label.split("_", 1)
        if prefix.lower() in {"deepseek-ai", "google", "meta-llama"}:
            return rest
    return label


def plot_action_aggregate(table_rows: Sequence[Dict[str, Any]], metric_name: str, ylabel: str, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(12, 6))
    x_labels = [display_label(row["action_model"]) for row in table_rows]
    x_positions = list(range(len(table_rows)))
    series = [
        ("overall", f"overall_{metric_name}"),
        ("easy", f"easy_{metric_name}"),
        ("medium", f"medium_{metric_name}"),
        ("hard", f"hard_{metric_name}"),
    ]

    for label, key in series:
        y_values = [float(row[key]) for row in table_rows]
        ax.plot(x_positions, y_values, marker="o", linewidth=2, markersize=5, label=label.capitalize())

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=20, ha="right")
    ax.set_xlabel("Action model")
    ax.set_ylabel(ylabel)
    if metric_name in {"precision", "recall", "f1"}:
        ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_detailed_by_action(
    summary_rows: Sequence[Dict[str, Any]],
    metric_key: str,
    ylabel: str,
    output_dir: Path,
) -> List[Path]:
    ensure_dir(output_dir)
    ok_rows = [row for row in summary_rows if row["status"] == "ok"]
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in ok_rows:
        grouped[(row["action_model"], row["context_type"], row["prompt_name"])].append(row)

    saved_paths: List[Path] = []
    for (action_model, context_type, prompt_name), rows_in_group in sorted(grouped.items()):
        memory_models = sorted({row["memory_model"] for row in rows_in_group})
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=(metric_key != "parsing_fail_count"))

        for ax, pref_type in zip(axes, PREF_ORDER):
            pref_rows = [row for row in rows_in_group if row["pref_type"] == pref_type]
            values_by_memory = {row["memory_model"]: row[metric_key] for row in pref_rows}
            x_positions = list(range(len(memory_models)))
            y_values = [float(values_by_memory.get(memory_model, 0.0) or 0.0) for memory_model in memory_models]
            ax.bar(x_positions, y_values, color="#4C78A8")
            ax.set_title(pref_type.capitalize())
            ax.set_xticks(x_positions)
            ax.set_xticklabels([display_label(name) for name in memory_models], rotation=20, ha="right")
            ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
            if metric_key in {"overall_precision", "overall_recall", "overall_f1"}:
                ax.set_ylim(0.0, 1.0)
            if ax is axes[0]:
                ax.set_ylabel(ylabel)

        fig.suptitle(
            f"{display_label(action_model)} | {context_type} | {prompt_name} | {ylabel} by memory model",
            y=0.98,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        output_path = output_dir / f"{safe_name(action_model)}__{safe_name(context_type)}__{safe_name(prompt_name)}__{safe_name(metric_key)}.png"
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        saved_paths.append(output_path)

    return saved_paths


def validate_outputs(root_dir: Path, detailed_rows: Sequence[Dict[str, Any]], outputs: Sequence[Path]) -> None:
    expected_files = len(iter_result_jsons(root_dir))
    if expected_files != len(detailed_rows):
        raise ValueError(f"Expected {expected_files} evaluated files, got {len(detailed_rows)} rows")
    missing = [path for path in outputs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing expected output files: {missing}")


def print_report(
    analysis_dir: Path,
    detailed_rows: Sequence[Dict[str, Any]],
    summary_rows: Sequence[Dict[str, Any]],
    action_table_rows: Sequence[Dict[str, Any]],
    outputs: Sequence[Path],
    module: Any,
) -> None:
    ok_rows = [row for row in detailed_rows if row["status"] == "ok"]
    error_rows = [row for row in detailed_rows if row["status"] == "error"]
    overall_counts = aggregate_metric_counts(ok_rows, "overall")
    overall_prf = module.prf_from_counts(overall_counts.tp, overall_counts.fp, overall_counts.fn)
    print(f"[ANALYSIS DIR] {analysis_dir}")
    print(f"[DETAILED ROWS] {len(detailed_rows)}")
    print(f"[SUMMARY ROWS] {len(summary_rows)}")
    print(f"[ACTION TABLE ROWS] {len(action_table_rows)}")
    print(f"[ERROR ROWS] {len(error_rows)}")
    print(
        f"[OVERALL] TP={overall_prf.tp} FP={overall_prf.fp} FN={overall_prf.fn} "
        f"P={overall_prf.precision:.4f} R={overall_prf.recall:.4f} F1={overall_prf.f1:.4f}"
    )
    print(f"[PARSING FAILURES] {sum(int(row['parsing_fail_count']) for row in ok_rows)}")
    for path in outputs:
        print(f"[OUTPUT] {path}")


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir).resolve()
    analysis_dir = Path(args.analysis_dir).resolve() if args.analysis_dir else root_dir / "analysis"
    ensure_dir(analysis_dir)

    module = load_multiturn_eval_module()
    pref_map = module.load_pref_list(args.pref_list_path)
    detailed_rows = build_detailed_rows(root_dir, module, pref_map)
    summary_rows = summarize_rows(detailed_rows, module)

    detailed_csv = analysis_dir / "multiturn_query_metrics_detailed.csv"
    summary_csv = analysis_dir / "multiturn_query_metrics_summary.csv"
    action_table_csv = analysis_dir / "multiturn_query_action_table.csv"
    memory_table_csv = analysis_dir / "multiturn_query_memory_table.csv"

    write_csv(detailed_csv, DETAIL_FIELDNAMES, detailed_rows)
    write_csv(summary_csv, SUMMARY_FIELDNAMES, summary_rows)

    action_table_rows = build_wide_table(summary_rows, "action_model", "action_model", ACTION_TABLE_FIELDNAMES, module)
    memory_table_rows = build_wide_table(summary_rows, "memory_model", "memory_model", MEMORY_TABLE_FIELDNAMES, module)
    write_csv(action_table_csv, ACTION_TABLE_FIELDNAMES, action_table_rows)
    write_csv(memory_table_csv, MEMORY_TABLE_FIELDNAMES, memory_table_rows)

    outputs: List[Path] = [detailed_csv, summary_csv, action_table_csv, memory_table_csv]
    for metric_name, ylabel in [("f1", "Overall F1"), ("precision", "Overall Precision"), ("recall", "Overall Recall")]:
        plot_path = analysis_dir / "plots" / "action_aggregate" / f"overall_{metric_name}.png"
        plot_action_aggregate(action_table_rows, metric_name, ylabel, plot_path)
        outputs.append(plot_path)

    outputs.extend(
        plot_detailed_by_action(summary_rows, "overall_f1", "Overall F1", analysis_dir / "plots" / "detailed" / "f1")
    )
    outputs.extend(
        plot_detailed_by_action(summary_rows, "overall_precision", "Overall Precision", analysis_dir / "plots" / "detailed" / "precision")
    )
    outputs.extend(
        plot_detailed_by_action(summary_rows, "overall_recall", "Overall Recall", analysis_dir / "plots" / "detailed" / "recall")
    )

    validate_outputs(root_dir, detailed_rows, outputs)
    print_report(analysis_dir, detailed_rows, summary_rows, action_table_rows, outputs, module)


if __name__ == "__main__":
    main()
