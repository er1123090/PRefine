#!/usr/bin/env python3
"""Build an appendix memory-3 matched ablation comparison table."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Iterable


ROOT = Path("/data/minseo")
PAPER_DIR = ROOT / "experiments4" / "_paper"
RESULT_DIR = (
    ROOT
    / "experiments5"
    / "results"
    / "our_memory"
    / "1229_dev6_true_blind_api_history_ablation_e4_ours_memory_eval_20260529"
)
SPEC_DIR = ROOT / ".omx" / "specs" / "autoresearch-appendix-memory3-table"

ABLATION_TABLE = RESULT_DIR / "table3_style_true_blind_api_history_vs_ours.csv"
TABLE11 = PAPER_DIR / "table11_context_guided_base_prefine_gemma_gpt4o.csv"
TABLE12 = PAPER_DIR / "table12_context_guided_prefine_reasoning.csv"
TABLE13 = PAPER_DIR / "table13_context_free_prefine.csv"
TABLE3_APPENDIX = PAPER_DIR / "table3_recomputed_from_appendix.csv"

OUT_CSV = RESULT_DIR / "table3_style_appendix_memory3_matched_ablation.csv"
OUT_MD = RESULT_DIR / "table3_style_appendix_memory3_matched_ablation.md"
OUT_SOURCES = RESULT_DIR / "table3_style_appendix_memory3_matched_sources.csv"
OUT_VALIDATION = SPEC_DIR / "result.json"

METHOD_LABELS = {
    "vanilla_llm": "VANILLA_LLM",
    "gen_only": "GEN_ONLY",
    "gen_only_accum": "GEN_ONLY_ACCUM",
    "refine1": "REFINE-1",
    "refine2": "REFINE-2",
    "refine3": "REFINE-3",
    "ours_memory": "OURS_MEMORY",
}

METHOD_ORDER = [
    "vanilla_llm",
    "gen_only",
    "gen_only_accum",
    "refine1",
    "refine2",
    "refine3",
    "ours_memory",
]

ABLATION_METHODS = [
    "gen_only",
    "gen_only_accum",
    "refine1",
    "refine2",
    "refine3",
]

INFERENCE_MODELS = {
    "CodeGemma-7B": {
        "table11_12": "CodeGemma-7B-Instruct",
        "table13": "CodeGemma-7B-it",
        "table3": "CodeGemma-7B",
    },
    "Gemma-3-12B": {
        "table11_12": "Gemma-3-12B-Instruct",
        "table13": "Gemma-3-12b-it",
        "table3": "Gemma-3-12B",
    },
    "R1-Distill-Llama-8B": {
        "table11_12": "R1-Distill-Llama-8B",
        "table13": "R1-Distill-Llama-8B",
        "table3": "R1-Distill-Llama-8B",
    },
    "R1-Distill-Qwen-7B": {
        "table11_12": "R1-Distill-Qwen-7B",
        "table13": "R1-Distill-Qwen-7B",
        "table3": "R1-Distill-Qwen-7B",
    },
}

APPENDIX_MEMORY_MODELS = {
    "Gemma-3-12B-it": {
        "table11_12": "Gemma-3-12B-it",
        "table13_prefix": "gemma_3_12b_it",
        "experiment_model": "google/gemma-3-12b-it",
        "match_type": "exact_normalized",
    },
    "R1-Distill-Llama-8B": {
        "table11_12": "R1-Distill-Llama-8B",
        "table13_prefix": "r1_llama_8b",
        "experiment_model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "match_type": "exact_normalized",
    },
    "R1-Distill-Qwen-7B": {
        "table11_12": "R1-Distill-Qwen-7B",
        "table13_prefix": "r1_qwen_7b",
        "experiment_model": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        "match_type": "paper_available_qwen_family_substitute",
    },
}

DIFFICULTY_BY_MODELING_TYPE = {
    "Preference Recall": "easy",
    "Preference Induction": "medium",
    "Preference Transfer": "hard",
}

GUIDED_COLS = {
    "easy": {
        "p_em": "guided_easy_p_em",
        "ea_f1": "guided_easy_ea_f1",
        "oa_f1": "guided_easy_oa_f1",
    },
    "medium": {
        "p_em": "guided_medium_p_em",
        "ea_f1": "guided_medium_ea_f1",
        "oa_f1": "guided_medium_oa_f1",
    },
    "hard": {
        "p_em": "guided_hard_p_em",
        "ea_f1": "guided_hard_ea_f1",
        "oa_f1": "guided_hard_oa_f1",
    },
}

FREE_COLS = {
    "easy": {
        "precision": "free_easy_precision",
        "recall": "free_easy_recall",
        "f1": "free_easy_f1",
    },
    "medium": {
        "precision": "free_medium_precision",
        "recall": "free_medium_recall",
        "f1": "free_medium_f1",
    },
    "hard": {
        "precision": "free_hard_precision",
        "recall": "free_hard_recall",
        "f1": "free_hard_f1",
    },
}

OUTPUT_FIELDS = [
    "method_key",
    "method",
    "guided_easy_p_em",
    "guided_easy_ea_f1",
    "guided_easy_oa_f1",
    "guided_medium_p_em",
    "guided_medium_ea_f1",
    "guided_medium_oa_f1",
    "guided_hard_p_em",
    "guided_hard_ea_f1",
    "guided_hard_oa_f1",
    "guided_avg_oa_f1",
    "free_easy_precision",
    "free_easy_recall",
    "free_easy_f1",
    "free_medium_precision",
    "free_medium_recall",
    "free_medium_f1",
    "free_hard_precision",
    "free_hard_recall",
    "free_hard_f1",
    "free_avg_f1",
]

SOURCE_FIELDS = [
    "target_method_key",
    "target_metric_group",
    "appendix_table",
    "section",
    "turn",
    "difficulty",
    "metric",
    "paper_memory_model",
    "experiment_memory_model",
    "memory_match_type",
    "paper_inference_model",
    "experiment_inference_model",
    "value",
    "source_file",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fnum(value: str | float) -> float:
    return float(value)


def avg(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("cannot average an empty value list")
    return mean(values)


def blank_result(method_key: str) -> dict[str, object]:
    return {
        field: "" for field in OUTPUT_FIELDS
    } | {"method_key": method_key, "method": METHOD_LABELS[method_key]}


def set_avg_columns(row: dict[str, object]) -> None:
    row["guided_avg_oa_f1"] = avg(
        fnum(row[GUIDED_COLS[difficulty]["oa_f1"]]) for difficulty in ["easy", "medium", "hard"]
    )
    row["free_avg_f1"] = avg(
        fnum(row[FREE_COLS[difficulty]["f1"]]) for difficulty in ["easy", "medium", "hard"]
    )


def build_lookup(rows: list[dict[str, str]], *keys: str) -> dict[tuple[str, ...], dict[str, str]]:
    out: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        out[tuple(row[k] for k in keys)] = row
    return out


def appendix_vanilla_row(
    table11_rows: list[dict[str, str]],
    table3_rows: list[dict[str, str]],
    source_rows: list[dict[str, object]],
) -> dict[str, object]:
    row = blank_result("vanilla_llm")

    base_lookup = build_lookup(
        [r for r in table11_rows if r["section"] == "BASE PROMPTING"],
        "modeling_type",
        "inference_model",
    )
    for modeling_type, difficulty in DIFFICULTY_BY_MODELING_TYPE.items():
        matched = [
            base_lookup[(modeling_type, info["table11_12"])]
            for info in INFERENCE_MODELS.values()
        ]
        for metric in ["p_em", "ea_f1", "oa_f1"]:
            row[GUIDED_COLS[difficulty][metric]] = avg(fnum(r[metric]) for r in matched)
            for r, info in zip(matched, INFERENCE_MODELS.values(), strict=True):
                source_rows.append(
                    {
                        "target_method_key": "vanilla_llm",
                        "target_metric_group": "context_guided",
                        "appendix_table": r["appendix_table"],
                        "section": r["section"],
                        "turn": "multiturn",
                        "difficulty": difficulty,
                        "metric": metric,
                        "paper_memory_model": "",
                        "experiment_memory_model": "",
                        "memory_match_type": "not_applicable_base_prompting",
                        "paper_inference_model": r["inference_model"],
                        "experiment_inference_model": info["table3"],
                        "value": r[metric],
                        "source_file": str(TABLE11),
                    }
                )

    table3_lookup = build_lookup(
        [r for r in table3_rows if r["Section"] == "BASE PROMPTING"],
        "Method/Base LLM",
    )
    for difficulty in ["easy", "medium", "hard"]:
        for metric, out_col in FREE_COLS[difficulty].items():
            table3_col = {
                ("easy", "precision"): "CF Recall Prec.",
                ("easy", "recall"): "CF Recall Rec.",
                ("easy", "f1"): "CF Recall F1",
                ("medium", "precision"): "CF Induction Prec.",
                ("medium", "recall"): "CF Induction Rec.",
                ("medium", "f1"): "CF Induction F1",
                ("hard", "precision"): "CF Transfer Prec.",
                ("hard", "recall"): "CF Transfer Rec.",
                ("hard", "f1"): "CF Transfer F1",
            }[(difficulty, metric)]
            matched = [table3_lookup[(info["table3"],)] for info in INFERENCE_MODELS.values()]
            row[out_col] = avg(fnum(r[table3_col]) for r in matched)
            for r, info in zip(matched, INFERENCE_MODELS.values(), strict=True):
                source_rows.append(
                    {
                        "target_method_key": "vanilla_llm",
                        "target_metric_group": "context_free",
                        "appendix_table": "Table 3 recomputed from appendix values",
                        "section": r["Section"],
                        "turn": "singleturn",
                        "difficulty": difficulty,
                        "metric": metric,
                        "paper_memory_model": "",
                        "experiment_memory_model": "",
                        "memory_match_type": "not_applicable_base_prompting",
                        "paper_inference_model": r["Method/Base LLM"],
                        "experiment_inference_model": info["table3"],
                        "value": r[table3_col],
                        "source_file": str(TABLE3_APPENDIX),
                    }
                )

    set_avg_columns(row)
    return row


def appendix_ours_row(
    table11_rows: list[dict[str, str]],
    table12_rows: list[dict[str, str]],
    table13_rows: list[dict[str, str]],
    source_rows: list[dict[str, object]],
) -> dict[str, object]:
    row = blank_result("ours_memory")

    guided_rows = [r for r in table11_rows + table12_rows if r["section"] == "PREFINE"]
    guided_lookup = build_lookup(
        guided_rows,
        "memory_construction_model",
        "modeling_type",
        "inference_model",
    )
    for modeling_type, difficulty in DIFFICULTY_BY_MODELING_TYPE.items():
        for metric in ["p_em", "ea_f1", "oa_f1"]:
            values = []
            for memory_name, memory_info in APPENDIX_MEMORY_MODELS.items():
                for inference_info in INFERENCE_MODELS.values():
                    r = guided_lookup[
                        (
                            memory_info["table11_12"],
                            modeling_type,
                            inference_info["table11_12"],
                        )
                    ]
                    values.append(fnum(r[metric]))
                    source_rows.append(
                        {
                            "target_method_key": "ours_memory",
                            "target_metric_group": "context_guided",
                            "appendix_table": r["appendix_table"],
                            "section": r["section"],
                            "turn": "multiturn",
                            "difficulty": difficulty,
                            "metric": metric,
                            "paper_memory_model": memory_name,
                            "experiment_memory_model": memory_info["experiment_model"],
                            "memory_match_type": memory_info["match_type"],
                            "paper_inference_model": r["inference_model"],
                            "experiment_inference_model": inference_info["table3"],
                            "value": r[metric],
                            "source_file": str(TABLE11 if r["appendix_table"] == "Table 11" else TABLE12),
                        }
                    )
            row[GUIDED_COLS[difficulty][metric]] = avg(values)

    free_lookup = build_lookup(table13_rows, "modeling_type", "inference_model")
    for modeling_type, difficulty in DIFFICULTY_BY_MODELING_TYPE.items():
        for metric in ["precision", "recall", "f1"]:
            values = []
            for memory_name, memory_info in APPENDIX_MEMORY_MODELS.items():
                source_col = f"{memory_info['table13_prefix']}_{metric}"
                for inference_info in INFERENCE_MODELS.values():
                    r = free_lookup[(modeling_type, inference_info["table13"])]
                    values.append(fnum(r[source_col]))
                    source_rows.append(
                        {
                            "target_method_key": "ours_memory",
                            "target_metric_group": "context_free",
                            "appendix_table": r["appendix_table"],
                            "section": r["section"],
                            "turn": "singleturn",
                            "difficulty": difficulty,
                            "metric": metric,
                            "paper_memory_model": memory_name,
                            "experiment_memory_model": memory_info["experiment_model"],
                            "memory_match_type": memory_info["match_type"],
                            "paper_inference_model": r["inference_model"],
                            "experiment_inference_model": inference_info["table3"],
                            "value": r[source_col],
                            "source_file": str(TABLE13),
                        }
                    )
            row[FREE_COLS[difficulty][metric]] = avg(values)

    set_avg_columns(row)
    return row


def format_number(value: object) -> str:
    return f"{float(value):.2f}"


def rank_format(rows: list[dict[str, object]], columns: list[str]) -> list[dict[str, str]]:
    out = [{k: str(v) for k, v in row.items()} for row in rows]
    for col in columns:
        values = sorted({round(float(row[col]), 12) for row in rows}, reverse=True)
        best = values[0]
        second = values[1] if len(values) > 1 else None
        for src, dest in zip(rows, out, strict=True):
            text = format_number(src[col])
            value = round(float(src[col]), 12)
            if value == best:
                text = f"**{text}**"
            elif second is not None and value == second:
                text = f"<u>{text}</u>"
            dest[col] = text
    return out


def markdown_table(rows: list[dict[str, str]], columns: list[tuple[str, str]]) -> list[str]:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    align = "| " + " | ".join("---" if i == 0 else "---:" for i, _ in enumerate(columns)) + " |"
    lines = [header, align]
    for row in rows:
        lines.append("| " + " | ".join(row[key] for key, _ in columns) + " |")
    return lines


def write_markdown(rows: list[dict[str, object]], validation: dict[str, object]) -> None:
    guided_cols = [
        ("method", "Method"),
        ("guided_easy_p_em", "Recall/Easy P-EM"),
        ("guided_easy_ea_f1", "Recall/Easy EA-F1"),
        ("guided_easy_oa_f1", "Recall/Easy OA-F1"),
        ("guided_medium_p_em", "Induction/Medium P-EM"),
        ("guided_medium_ea_f1", "Induction/Medium EA-F1"),
        ("guided_medium_oa_f1", "Induction/Medium OA-F1"),
        ("guided_hard_p_em", "Transfer/Hard P-EM"),
        ("guided_hard_ea_f1", "Transfer/Hard EA-F1"),
        ("guided_hard_oa_f1", "Transfer/Hard OA-F1"),
        ("guided_avg_oa_f1", "Avg OA-F1"),
    ]
    free_cols = [
        ("method", "Method"),
        ("free_easy_precision", "Recall/Easy Prec."),
        ("free_easy_recall", "Recall/Easy Rec."),
        ("free_easy_f1", "Recall/Easy F1"),
        ("free_medium_precision", "Induction/Medium Prec."),
        ("free_medium_recall", "Induction/Medium Rec."),
        ("free_medium_f1", "Induction/Medium F1"),
        ("free_hard_precision", "Transfer/Hard Prec."),
        ("free_hard_recall", "Transfer/Hard Rec."),
        ("free_hard_f1", "Transfer/Hard F1"),
        ("free_avg_f1", "Avg F1"),
    ]
    guided_ranked = rank_format(rows, [key for key, _ in guided_cols if key != "method"])
    free_ranked = rank_format(rows, [key for key, _ in free_cols if key != "method"])

    lines = [
        "# Table 3-style ablation with appendix memory-3 matched vanilla/ours",
        "",
        "Source policy:",
        "- Ablation rows (`GEN_ONLY`, `GEN_ONLY_ACCUM`, `REFINE-1/2/3`) are copied from the current ablation experiment table.",
        "- `VANILLA_LLM` uses appendix/base-prompting values for the four matched inference models. Base prompting has no memory-construction axis; context-free values come from the appendix-recomputed Table 3 source artifact.",
        "- `OURS_MEMORY` uses appendix `PREFINE` values averaged over three appendix memory-construction rows and four matched inference models.",
        "- Appendix has no exact `DeepSeek-R1-0528-Qwen3-8B` memory-construction row; `R1-Distill-Qwen-7B` is used as the paper-available Qwen-family row and is flagged in the source CSV/validation JSON.",
        "",
        "Matched inference models: "
        + ", ".join(INFERENCE_MODELS.keys())
        + ".",
        "Appendix memory rows used for `OURS_MEMORY`: "
        + ", ".join(APPENDIX_MEMORY_MODELS.keys())
        + ".",
        "",
        "## Context-Guided Query / Multiturn",
        "",
        *markdown_table(guided_ranked, guided_cols),
        "",
        "## Context-Free Query / Singleturn",
        "",
        *markdown_table(free_ranked, free_cols),
        "",
        "## Validation Summary",
        "",
        f"- Status: `{validation['status']}`",
        f"- Output CSV: `{OUT_CSV}`",
        f"- Source rows CSV: `{OUT_SOURCES}`",
        f"- Validation artifact: `{OUT_VALIDATION}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")


def build() -> dict[str, object]:
    table11_rows = read_csv(TABLE11)
    table12_rows = read_csv(TABLE12)
    table13_rows = read_csv(TABLE13)
    table3_rows = read_csv(TABLE3_APPENDIX)
    ablation_rows = read_csv(ABLATION_TABLE)
    ablation_by_method = {r["method_key"]: r for r in ablation_rows}

    source_rows: list[dict[str, object]] = []
    output_rows: list[dict[str, object]] = []
    output_rows.append(appendix_vanilla_row(table11_rows, table3_rows, source_rows))
    for method in ABLATION_METHODS:
        source = ablation_by_method[method]
        output_rows.append({field: source[field] for field in OUTPUT_FIELDS})
    output_rows.append(appendix_ours_row(table11_rows, table12_rows, table13_rows, source_rows))

    output_rows = sorted(output_rows, key=lambda row: METHOD_ORDER.index(str(row["method_key"])))
    write_csv(OUT_CSV, output_rows, OUTPUT_FIELDS)
    write_csv(OUT_SOURCES, source_rows, SOURCE_FIELDS)

    validation = validate_rows(output_rows, source_rows, ablation_by_method)
    write_markdown(output_rows, validation)
    OUT_VALIDATION.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    return validation


def validate_rows(
    output_rows: list[dict[str, object]],
    source_rows: list[dict[str, object]],
    ablation_by_method: dict[str, dict[str, str]],
) -> dict[str, object]:
    errors: list[str] = []
    warnings: list[str] = []
    row_by_method = {str(row["method_key"]): row for row in output_rows}

    if [str(row["method_key"]) for row in output_rows] != METHOD_ORDER:
        errors.append("Output method order does not match expected Table 3 order.")

    for method in ABLATION_METHODS:
        for field in OUTPUT_FIELDS:
            if str(row_by_method[method][field]) != ablation_by_method[method][field]:
                errors.append(f"Ablation value changed unexpectedly: {method}.{field}")

    counts: dict[str, int] = {}
    for source in source_rows:
        key = f"{source['target_method_key']}:{source['target_metric_group']}"
        counts[key] = counts.get(key, 0) + 1

    expected_counts = {
        "vanilla_llm:context_guided": 4 * 3 * 3,
        "vanilla_llm:context_free": 4 * 3 * 3,
        "ours_memory:context_guided": 3 * 4 * 3 * 3,
        "ours_memory:context_free": 3 * 4 * 3 * 3,
    }
    for key, expected in expected_counts.items():
        if counts.get(key) != expected:
            errors.append(f"Source value count mismatch for {key}: got {counts.get(key)}, expected {expected}")

    qwen_substitutes = [
        row for row in source_rows
        if row["memory_match_type"] == "paper_available_qwen_family_substitute"
    ]
    if len(qwen_substitutes) != 72:
        errors.append(f"Expected 72 Qwen-family substitute source values, found {len(qwen_substitutes)}")
    else:
        warnings.append(
            "Appendix lacks exact DeepSeek-R1-0528-Qwen3-8B memory rows; R1-Distill-Qwen-7B was used for the Qwen-family appendix memory row."
        )

    numeric_columns = [field for field in OUTPUT_FIELDS if field not in {"method_key", "method"}]
    for method in METHOD_ORDER:
        for field in numeric_columns:
            try:
                float(row_by_method[method][field])
            except Exception as exc:  # noqa: BLE001
                errors.append(f"Non-numeric output value for {method}.{field}: {exc}")

    status = "passed" if not errors else "failed"
    return {
        "status": status,
        "passed": not errors,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": (
            "Generated appendix memory-3 matched ablation table with documented Qwen-family memory caveat."
            if not errors
            else "Validation failed."
        ),
        "output_artifact_path": str(OUT_MD),
        "csv_artifact_path": str(OUT_CSV),
        "source_rows_path": str(OUT_SOURCES),
        "source_value_counts": counts,
        "expected_source_value_counts": expected_counts,
        "warnings": warnings,
        "errors": errors,
        "matched_inference_models": list(INFERENCE_MODELS.keys()),
        "appendix_memory_models_used": list(APPENDIX_MEMORY_MODELS.keys()),
        "experiment_memory_models_requested": [
            info["experiment_model"] for info in APPENDIX_MEMORY_MODELS.values()
        ],
    }


def validate_existing() -> dict[str, object]:
    if not OUT_CSV.exists() or not OUT_MD.exists() or not OUT_SOURCES.exists():
        return build()
    rows = read_csv(OUT_CSV)
    source_rows = read_csv(OUT_SOURCES)
    ablation_by_method = {r["method_key"]: r for r in read_csv(ABLATION_TABLE)}
    validation = validate_rows(rows, source_rows, ablation_by_method)
    OUT_VALIDATION.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    return validation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true", help="Validate existing outputs, rebuilding if needed.")
    args = parser.parse_args()

    validation = validate_existing() if args.validate else build()
    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0 if validation["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
