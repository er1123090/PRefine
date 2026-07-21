#!/usr/bin/env python3
"""Build Table 3-style conflict-majority comparisons split by code path."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable


ROOT = Path("/data/minseo/experiments5")

DEFAULT_EXP4_BASE = (
    ROOT
    / "results"
    / "vanilla_llm_exp4"
    / "conflict_majority_exp4_vanilla_llm_freesia23_20260529_1500"
    / "per_file_metrics.csv"
)
DEFAULT_EXP5_BASE = (
    ROOT
    / "results"
    / "vanilla_llm"
    / "conflict_majority_vanilla_llm_freesia23_20260529_130444"
    / "per_file_metrics.csv"
)
DEFAULT_MEMORY = (
    ROOT
    / "results"
    / "our_memory"
    / "conflict_majority_freesia23_20260529_103605"
    / "per_file_metrics.csv"
)
DEFAULT_OUT_DIR = ROOT / "results" / "conflict_majority_table3_style_by_codepath_20260529"

MODEL_ORDER = [
    "google/codegemma-7b-it",
    "google/gemma-3-12b-it",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
]
MODEL_LABELS = {
    "google/codegemma-7b-it": "CodeGemma-7B",
    "google/gemma-3-12b-it": "Gemma-3-12B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "R1-Distill-Llama-8B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": "R1-Distill-Qwen-7B",
}
CONFLICT_ORDER = ["ordered", "random"]
TURN_ORDER = ["singleturn", "multiturn"]
DIFFICULTY_ORDER = ["easy", "medium", "hard"]
SECTION_ORDER = ["BASE PROMPTING", "OURS_MEMORY"]
CODE_PATH_ORDER = ["EXP4-compatible", "EXP5-native"]
CODE_PATH_SAFE = {
    "EXP4-compatible": "exp4_compatible",
    "EXP5-native": "exp5_native",
}

COUNT_FIELDS = [
    "n_examples",
    "parse_failures",
    "overall_tp",
    "overall_fp",
    "overall_fn",
    "pref_em_count",
    "pref_em_total",
    "nonpref_tp",
    "nonpref_fp",
    "nonpref_fn",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def as_int(row: dict[str, str], field: str) -> int:
    value = row.get(field, "")
    if value in ("", None):
        return 0
    return int(float(value))


def add_common(
    rows: Iterable[dict[str, str]],
    *,
    code_path: str,
    section: str,
) -> list[dict[str, str]]:
    normalized = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        out = dict(row)
        out["code_path"] = code_path
        out["section"] = section
        normalized.append(out)
    return normalized


def load_rows(exp4_base: Path, exp5_base: Path, memory: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    rows.extend(
        add_common(
            read_csv(exp4_base),
            code_path="EXP4-compatible",
            section="BASE PROMPTING",
        )
    )
    rows.extend(
        add_common(
            read_csv(exp5_base),
            code_path="EXP5-native",
            section="BASE PROMPTING",
        )
    )

    for row in read_csv(memory):
        pipeline = row.get("pipeline")
        if pipeline == "e4":
            code_path = "EXP4-compatible"
        elif pipeline == "exp5":
            code_path = "EXP5-native"
        else:
            continue
        if row.get("status") != "ok":
            continue
        out = dict(row)
        out["code_path"] = code_path
        out["section"] = "OURS_MEMORY"
        rows.append(out)
    return rows


def empty_counts() -> dict[str, int]:
    return {field: 0 for field in COUNT_FIELDS}


def aggregate(rows: Iterable[dict[str, str]]) -> dict[str, int]:
    counts = empty_counts()
    for row in rows:
        for field in COUNT_FIELDS:
            counts[field] += as_int(row, field)
    return counts


def pct(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value * 100:.2f}"


def div(num: int, den: int) -> float | None:
    if den == 0:
        return 0.0
    return num / den


def f1(tp: int, fp: int, fn: int) -> float | None:
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 0.0
    return (2 * tp) / denom


def overall_precision(counts: dict[str, int]) -> float | None:
    return div(counts["overall_tp"], counts["overall_tp"] + counts["overall_fp"])


def overall_recall(counts: dict[str, int]) -> float | None:
    return div(counts["overall_tp"], counts["overall_tp"] + counts["overall_fn"])


def overall_f1(counts: dict[str, int]) -> float | None:
    return f1(counts["overall_tp"], counts["overall_fp"], counts["overall_fn"])


def pref_em(counts: dict[str, int]) -> float | None:
    return div(counts["pref_em_count"], counts["pref_em_total"])


def nonpref_f1(counts: dict[str, int]) -> float | None:
    return f1(counts["nonpref_tp"], counts["nonpref_fp"], counts["nonpref_fn"])


def parse_fail_rate(counts: dict[str, int]) -> float | None:
    return div(counts["parse_failures"], counts["n_examples"])


def row_for(
    rows: list[dict[str, str]],
    *,
    code_path: str,
    conflict: str,
    turn: str,
    section: str,
    model: str | None,
) -> dict[str, str]:
    selected = [
        row
        for row in rows
        if row.get("code_path") == code_path
        and row.get("conflict_slug") == conflict
        and row.get("turn") == turn
        and row.get("section") == section
        and (model is None or row.get("inference_model") == model)
    ]
    all_counts = aggregate(selected)

    if model is None:
        label = "Average"
    else:
        label = MODEL_LABELS.get(model, model)

    out = {
        "code_path": code_path,
        "conflict": f"conflict_{conflict}",
        "turn": turn,
        "section": section,
        "inference_llm": label,
    }

    for difficulty in DIFFICULTY_ORDER:
        counts = aggregate(row for row in selected if row.get("difficulty") == difficulty)
        prefix = difficulty.capitalize()
        if turn == "singleturn":
            out[f"{prefix} Prec."] = pct(overall_precision(counts))
            out[f"{prefix} Rec."] = pct(overall_recall(counts))
            out[f"{prefix} F1"] = pct(overall_f1(counts))
        else:
            out[f"{prefix} P-EM"] = pct(pref_em(counts))
            out[f"{prefix} EA-F1"] = pct(nonpref_f1(counts))
            out[f"{prefix} OA-F1"] = pct(overall_f1(counts))

    if turn == "singleturn":
        out["Avg F1"] = pct(overall_f1(all_counts))
    else:
        out["Avg OA-F1"] = pct(overall_f1(all_counts))
    out["Parse fail %"] = pct(parse_fail_rate(all_counts))
    return out


def table_rows(
    rows: list[dict[str, str]],
    *,
    code_path: str,
    conflict: str,
    turn: str,
) -> list[dict[str, str]]:
    out = []
    for section in SECTION_ORDER:
        for model in MODEL_ORDER:
            out.append(
                row_for(
                    rows,
                    code_path=code_path,
                    conflict=conflict,
                    turn=turn,
                    section=section,
                    model=model,
                )
            )
        out.append(
            row_for(
                rows,
                code_path=code_path,
                conflict=conflict,
                turn=turn,
                section=section,
                model=None,
            )
        )
    return out


def columns_for(turn: str) -> list[str]:
    base = ["code_path", "conflict", "turn", "section", "inference_llm"]
    if turn == "singleturn":
        metrics = [
            "Easy Prec.",
            "Easy Rec.",
            "Easy F1",
            "Medium Prec.",
            "Medium Rec.",
            "Medium F1",
            "Hard Prec.",
            "Hard Rec.",
            "Hard F1",
            "Avg F1",
            "Parse fail %",
        ]
    else:
        metrics = [
            "Easy P-EM",
            "Easy EA-F1",
            "Easy OA-F1",
            "Medium P-EM",
            "Medium EA-F1",
            "Medium OA-F1",
            "Hard P-EM",
            "Hard EA-F1",
            "Hard OA-F1",
            "Avg OA-F1",
            "Parse fail %",
        ]
    return base + metrics


def markdown_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row.get(col, "") for col in columns) + " |")
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def latex_escape(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


def write_latex(path: Path, sections: list[tuple[str, list[dict[str, str]], list[str]]]) -> None:
    lines = [
        "% Auto-generated conflict-majority Table 3-style comparison split by code path.",
        "% Values are percentages.",
    ]
    for title, rows, columns in sections:
        align = "l" * len(columns)
        lines.extend(
            [
                "",
                f"\\subsection*{{{latex_escape(title)}}}",
                f"\\begin{{tabular}}{{{align}}}",
                "\\hline",
                " & ".join(latex_escape(col) for col in columns) + r" \\",
                "\\hline",
            ]
        )
        for row in rows:
            lines.append(" & ".join(latex_escape(row.get(col, "")) for col in columns) + r" \\")
        lines.extend(["\\hline", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def build(args: argparse.Namespace) -> Path:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.exp4_base, args.exp5_base, args.memory)
    sections: list[tuple[str, list[dict[str, str]], list[str]]] = []
    long_rows: list[dict[str, str]] = []
    all_columns = list(
        dict.fromkeys(columns_for("singleturn") + columns_for("multiturn"))
    )

    md_lines = [
        "# Conflict Majority Table 3-style Comparison by Code Path",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "EXP4-compatible uses exp4 vanilla/base prompting and ours_memory pipeline=e4, both evaluated with the experiments4 evaluator.",
        "EXP5-native uses exp5 vanilla/base prompting and ours_memory pipeline=exp5, evaluated with the experiments5 evaluator.",
        "Rows are grouped by inference LLM. OURS_MEMORY model rows micro-average over the 3 memory-construction models.",
        "Singleturn is context-free query with Prec. / Rec. / F1. Multiturn is context-guided query with P-EM / EA-F1 / OA-F1.",
        "Avg F1 / Avg OA-F1 and Parse fail % are micro-aggregated over Easy, Medium, and Hard.",
        "",
    ]

    for conflict in CONFLICT_ORDER:
        for code_path in CODE_PATH_ORDER:
            for turn in TURN_ORDER:
                title = f"conflict_{conflict} / {code_path} / {turn}"
                part_rows = table_rows(
                    rows,
                    code_path=code_path,
                    conflict=conflict,
                    turn=turn,
                )
                columns = columns_for(turn)
                sections.append((title, part_rows, columns))
                long_rows.extend(part_rows)
                md_lines.extend([f"## {title}", "", markdown_table(part_rows, columns), ""])

                filename = (
                    f"table3_style_conflict_{conflict}_{CODE_PATH_SAFE[code_path]}_"
                    f"{turn}.csv"
                )
                write_csv(out_dir / filename, part_rows, columns)

    write_csv(out_dir / "table3_style_by_codepath_long.csv", long_rows, all_columns)
    write_csv(
        out_dir / "table3_style_by_codepath_average_rows.csv",
        [row for row in long_rows if row.get("inference_llm") == "Average"],
        all_columns,
    )
    (out_dir / "table3_style_by_codepath.md").write_text("\n".join(md_lines))
    write_latex(out_dir / "table3_style_by_codepath.tex", sections)
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp4-base", type=Path, default=DEFAULT_EXP4_BASE)
    parser.add_argument("--exp5-base", type=Path, default=DEFAULT_EXP5_BASE)
    parser.add_argument("--memory", type=Path, default=DEFAULT_MEMORY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    out_dir = build(parse_args())
    print(out_dir)


if __name__ == "__main__":
    main()
