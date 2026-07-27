#!/usr/bin/env python3
"""Export one workbook comparing vanilla 5 and ours-memory 3x5 runs."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from openpyxl import Workbook, load_workbook
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_local_full_inference as local_runtime  # noqa: E402
import run_vanilla_batch_sample as population_runtime  # noqa: E402


EXPERIMENT_ROOT = (
    ROOT / "outputs" / "mpt_v2_0725_hint_no_easy_conflict"
)
DEFAULT_OUTPUT = (
    EXPERIMENT_ROOT / "unified_vanilla5_ours3x5_performance.xlsx"
)
SAMPLE_SIZE = 4000
SAMPLE_ROOT = (
    EXPERIMENT_ROOT
    / "ours_memory_batch"
    / f"stratified_sample_{SAMPLE_SIZE}_seed_20260727"
)
SAMPLE_MANIFEST = SAMPLE_ROOT / "sample_manifest.jsonl"


@dataclass(frozen=True)
class RunSpec:
    key: str
    method: str
    memory: str
    inference_model: str
    setting: str
    provider: str
    scope: str
    directory: Path


RUNS: tuple[RunSpec, ...] = (
    RunSpec(
        "vanilla_gpt5",
        "Vanilla LLM",
        "None",
        "GPT-5",
        "reasoning=minimal",
        "OpenAI Batch API",
        "full",
        EXPERIMENT_ROOT / "vanilla_llm" / "openai_gpt-5_minimal",
    ),
    RunSpec(
        "vanilla_claude",
        "Vanilla LLM",
        "None",
        "Claude Haiku 4.5",
        "thinking=off",
        "Anthropic Batch API",
        "full",
        EXPERIMENT_ROOT
        / "vanilla_llm"
        / "anthropic_claude-haiku-4-5_non_reasoning",
    ),
    RunSpec(
        "vanilla_qwen",
        "Vanilla LLM",
        "None",
        "Qwen3-8B",
        "thinking=off",
        "OpenRouter",
        "full",
        EXPERIMENT_ROOT
        / "vanilla_llm"
        / "qwen_qwen3-8b_thinking_off_optimized",
    ),
    RunSpec(
        "vanilla_gptoss",
        "Vanilla LLM",
        "None",
        "GPT-OSS-20B",
        "thinking=off, reasoning=low",
        "OpenRouter",
        "full",
        EXPERIMENT_ROOT
        / "vanilla_llm"
        / "openai_gpt-oss-20b_thinking_off_reasoning_low_optimized",
    ),
    RunSpec(
        "vanilla_gemma",
        "Vanilla LLM",
        "None",
        "Gemma4-12B",
        "thinking=off",
        "GPU / vLLM",
        "full",
        EXPERIMENT_ROOT
        / "vanilla_llm"
        / "google_gemma-4-12b-it_thinking_off_optimized",
    ),
    RunSpec(
        "qwenmem_qwen",
        "Ours-memory",
        "Qwen3-8B",
        "Qwen3-8B",
        "reasoning=high",
        "OpenRouter",
        "full",
        EXPERIMENT_ROOT
        / "ours_memory"
        / "qwen3_8b_memory__to__openrouter_qwen3_8b_high_reasoning",
    ),
    RunSpec(
        "qwenmem_gptoss",
        "Ours-memory",
        "Qwen3-8B",
        "GPT-OSS-20B",
        "reasoning=high",
        "OpenRouter",
        "full",
        EXPERIMENT_ROOT
        / "ours_memory"
        / "qwen3_8b_memory__to__openrouter_gpt_oss_20b_high_reasoning",
    ),
    RunSpec(
        "qwenmem_gemma",
        "Ours-memory",
        "Qwen3-8B",
        "Gemma4-12B",
        "reasoning=high",
        "GPU / vLLM",
        "full",
        EXPERIMENT_ROOT
        / "ours_memory"
        / "qwen3_8b_memory__to__gemma4_12b_it_high_reasoning_optimized",
    ),
    RunSpec(
        "qwenmem_gpt5",
        "Ours-memory",
        "Qwen3-8B",
        "GPT-5",
        "reasoning=medium",
        "OpenAI Batch API",
        "sample",
        SAMPLE_ROOT
        / "jobs"
        / "qwen3_8b_memory__to__openai_gpt5_medium",
    ),
    RunSpec(
        "qwenmem_claude",
        "Ours-memory",
        "Qwen3-8B",
        "Claude Haiku 4.5",
        "thinking budget=2048",
        "Anthropic Batch API",
        "sample",
        SAMPLE_ROOT
        / "jobs"
        / "qwen3_8b_memory__to__anthropic_claude_haiku4_5_thinking2048",
    ),
    RunSpec(
        "gemmamem_qwen",
        "Ours-memory",
        "Gemma4-12B",
        "Qwen3-8B",
        "reasoning=high",
        "OpenRouter",
        "full",
        EXPERIMENT_ROOT
        / "ours_memory"
        / "gemma4_12b_it_memory__to__openrouter_qwen3_8b_high_reasoning",
    ),
    RunSpec(
        "gemmamem_gptoss",
        "Ours-memory",
        "Gemma4-12B",
        "GPT-OSS-20B",
        "reasoning=high",
        "OpenRouter",
        "full",
        EXPERIMENT_ROOT
        / "ours_memory"
        / "gemma4_12b_it_memory__to__openrouter_gpt_oss_20b_high_reasoning",
    ),
    RunSpec(
        "gemmamem_gemma",
        "Ours-memory",
        "Gemma4-12B",
        "Gemma4-12B",
        "reasoning=high",
        "GPU / vLLM",
        "full",
        EXPERIMENT_ROOT
        / "ours_memory"
        / "gemma4_12b_it_memory__to__gemma4_12b_it_high_reasoning_optimized",
    ),
    RunSpec(
        "gemmamem_gpt5",
        "Ours-memory",
        "Gemma4-12B",
        "GPT-5",
        "reasoning=medium",
        "OpenAI Batch API",
        "sample",
        SAMPLE_ROOT
        / "jobs"
        / "gemma4_12b_it_memory__to__openai_gpt5_medium",
    ),
    RunSpec(
        "gemmamem_claude",
        "Ours-memory",
        "Gemma4-12B",
        "Claude Haiku 4.5",
        "thinking budget=2048",
        "Anthropic Batch API",
        "sample",
        SAMPLE_ROOT
        / "jobs"
        / "gemma4_12b_it_memory__to__anthropic_claude_haiku4_5_thinking2048",
    ),
    RunSpec(
        "gptossmem_qwen",
        "Ours-memory",
        "GPT-OSS-20B",
        "Qwen3-8B",
        "reasoning=high",
        "OpenRouter",
        "full",
        EXPERIMENT_ROOT
        / "ours_memory"
        / "gpt_oss_20b_memory__to__openrouter_qwen3_8b_high_reasoning",
    ),
    RunSpec(
        "gptossmem_gptoss",
        "Ours-memory",
        "GPT-OSS-20B",
        "GPT-OSS-20B",
        "reasoning=high",
        "GPU / vLLM",
        "full",
        EXPERIMENT_ROOT
        / "ours_memory"
        / "gpt_oss_20b_memory__to__gpt_oss_20b_high_reasoning_optimized",
    ),
    RunSpec(
        "gptossmem_gemma",
        "Ours-memory",
        "GPT-OSS-20B",
        "Gemma4-12B",
        "reasoning=high",
        "GPU / vLLM",
        "full",
        EXPERIMENT_ROOT
        / "ours_memory"
        / "gpt_oss_20b_memory__to__gemma4_12b_it_high_reasoning_optimized",
    ),
    RunSpec(
        "gptossmem_gpt5",
        "Ours-memory",
        "GPT-OSS-20B",
        "GPT-5",
        "reasoning=medium",
        "OpenAI Batch API",
        "sample",
        SAMPLE_ROOT
        / "jobs"
        / "gpt_oss_20b_memory__to__openai_gpt5_medium",
    ),
    RunSpec(
        "gptossmem_claude",
        "Ours-memory",
        "GPT-OSS-20B",
        "Claude Haiku 4.5",
        "thinking budget=2048",
        "Anthropic Batch API",
        "sample",
        SAMPLE_ROOT
        / "jobs"
        / "gpt_oss_20b_memory__to__anthropic_claude_haiku4_5_thinking2048",
    ),
)


METRIC_HEADERS = (
    "Run ID",
    "Method",
    "Memory Source",
    "Inference Model",
    "Inference Setting",
    "Provider / Backend",
    "Evaluation Scope",
    "N",
    "API Errors",
    "Parsing Failures",
    "Parsing Failure Rate",
    "Overall Precision",
    "Overall Recall",
    "Overall F1",
    "Overall F1 Rank",
    "Preference Precision",
    "Preference Recall",
    "Preference F1",
    "Preference Exact Match",
    "Preference EM Count",
    "Preference EM Rank",
    "Non-preference Precision",
    "Non-preference Recall",
    "Non-preference F1",
    "Input Tokens",
    "Output Tokens",
    "Reasoning Tokens",
    "Total Tokens",
    "Result Directory",
)

PERCENT_COLUMNS = {
    "Parsing Failure Rate",
    "Overall Precision",
    "Overall Recall",
    "Overall F1",
    "Preference Precision",
    "Preference Recall",
    "Preference F1",
    "Preference Exact Match",
    "Non-preference Precision",
    "Non-preference Recall",
    "Non-preference F1",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def metric_values(
    spec: RunSpec,
    report: Mapping[str, Any],
    scope_label: str,
) -> dict[str, Any]:
    overall = report["overall"]
    token_usage = report.get("token_usage") or {}
    return {
        "Run ID": spec.key,
        "Method": spec.method,
        "Memory Source": spec.memory,
        "Inference Model": spec.inference_model,
        "Inference Setting": spec.setting,
        "Provider / Backend": spec.provider,
        "Evaluation Scope": scope_label,
        "N": int(overall["n"]),
        "API Errors": int(overall["api_errors"]),
        "Parsing Failures": int(overall["parsing_failures"]),
        "Parsing Failure Rate": float(overall["parsing_failure_rate"]),
        "Overall Precision": float(overall["overall"]["precision"]),
        "Overall Recall": float(overall["overall"]["recall"]),
        "Overall F1": float(overall["overall"]["f1"]),
        "Overall F1 Rank": None,
        "Preference Precision": float(overall["pref"]["precision"]),
        "Preference Recall": float(overall["pref"]["recall"]),
        "Preference F1": float(overall["pref"]["f1"]),
        "Preference Exact Match": float(overall["pref"]["exact_match_rate"]),
        "Preference EM Count": int(overall["pref"]["exact_match_count"]),
        "Preference EM Rank": None,
        "Non-preference Precision": float(
            overall["nonpref"]["precision"]
        ),
        "Non-preference Recall": float(overall["nonpref"]["recall"]),
        "Non-preference F1": float(overall["nonpref"]["f1"]),
        "Input Tokens": int(token_usage.get("input_tokens", 0) or 0),
        "Output Tokens": int(token_usage.get("output_tokens", 0) or 0),
        "Reasoning Tokens": int(
            token_usage.get("reasoning_tokens", 0) or 0
        ),
        "Total Tokens": int(token_usage.get("total_tokens", 0) or 0),
        "Result Directory": str(spec.directory),
    }


def add_dense_ranks(rows: list[dict[str, Any]]) -> None:
    for metric, rank_name in (
        ("Overall F1", "Overall F1 Rank"),
        ("Preference Exact Match", "Preference EM Rank"),
    ):
        values = sorted({float(row[metric]) for row in rows}, reverse=True)
        ranks = {value: rank for rank, value in enumerate(values, start=1)}
        for row in rows:
            row[rank_name] = ranks[float(row[metric])]


def load_predictions(
    spec: RunSpec,
    population: Sequence[Mapping[str, Any]],
    sample_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if spec.scope == "sample":
        native = json.loads(
            (spec.directory / "predictions.json").read_text(encoding="utf-8")
        )
        ids = {str(row["sample_id"]) for row in native}
        if ids != sample_ids:
            raise RuntimeError(
                f"{spec.key}: sampled prediction IDs do not match manifest"
            )
        return native, native

    records = local_runtime.read_checkpoint(
        spec.directory / "inference.jsonl"
    )
    if len(records) != len(population):
        raise RuntimeError(
            f"{spec.key}: expected {len(population)} records, got {len(records)}"
        )
    native = local_runtime.materialize_predictions(population, records)
    comparable = [
        row for row in native if str(row["sample_id"]) in sample_ids
    ]
    if len(comparable) != len(sample_ids):
        raise RuntimeError(
            f"{spec.key}: expected {len(sample_ids)} comparable rows, "
            f"got {len(comparable)}"
        )
    return native, comparable


def evaluate_runs() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    population = population_runtime.build_population(
        query="hint",
        context_type="diag-apilist",
        input_path=str(ROOT / "data" / "MPT_v2_0725.json"),
        exclude_easy_conflict=True,
    )
    if len(population) != 4695:
        raise RuntimeError(
            f"Expected 4695 population rows, got {len(population)}"
        )
    manifest = read_jsonl(SAMPLE_MANIFEST)
    sample_ids = {str(row["sample_id"]) for row in manifest}
    if len(sample_ids) != 1000:
        raise RuntimeError(
            f"Expected 1000 sampled IDs, got {len(sample_ids)}"
        )

    common_rows: list[dict[str, Any]] = []
    native_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    for index, spec in enumerate(RUNS, start=1):
        print(f"[{index:02d}/{len(RUNS)}] evaluating {spec.key}", flush=True)
        native_predictions, common_predictions = load_predictions(
            spec, population, sample_ids
        )
        native_report = population_runtime.evaluation_report(
            native_predictions
        )
        common_report = (
            native_report
            if spec.scope == "sample"
            else population_runtime.evaluation_report(common_predictions)
        )
        native_scope = (
            "Full population (4,695)"
            if spec.scope == "full"
            else "Stratified sample (1,000)"
        )
        native_rows.append(metric_values(spec, native_report, native_scope))
        common_rows.append(
            metric_values(
                spec,
                common_report,
                "Common stratified sample (1,000)",
            )
        )

        native_overall = native_report["overall"]
        target = 4695 if spec.scope == "full" else 1000
        clean = (
            int(native_overall["n"]) == target
            and int(native_overall["api_errors"]) == 0
            and int(native_overall["parsing_failures"]) == 0
        )
        status_rows.append(
            {
                "Run ID": spec.key,
                "Method": spec.method,
                "Memory Source": spec.memory,
                "Inference Model": spec.inference_model,
                "Target": target,
                "Final IDs": int(native_overall["n"]),
                "API Errors": int(native_overall["api_errors"]),
                "Parsing Failures": int(
                    native_overall["parsing_failures"]
                ),
                "Status": (
                    "Clean complete"
                    if clean
                    else "Rows complete; cleanup pending"
                ),
                "Result Directory": str(spec.directory),
            }
        )
        del native_predictions, common_predictions
        gc.collect()

    add_dense_ranks(common_rows)
    add_dense_ranks(native_rows)
    return common_rows, native_rows, status_rows


def style_metric_sheet(
    workbook: Workbook,
    name: str,
    title: str,
    note: str,
    rows: Sequence[Mapping[str, Any]],
    table_name: str,
) -> None:
    sheet = workbook.create_sheet(name)
    sheet.sheet_view.showGridLines = False
    sheet.merge_cells(
        start_row=1,
        start_column=1,
        end_row=1,
        end_column=len(METRIC_HEADERS),
    )
    sheet.cell(1, 1, title)
    sheet.cell(1, 1).font = Font(size=16, bold=True, color="FFFFFF")
    sheet.cell(1, 1).fill = PatternFill("solid", fgColor="17365D")
    sheet.cell(1, 1).alignment = Alignment(vertical="center")
    sheet.row_dimensions[1].height = 26
    sheet.merge_cells(
        start_row=2,
        start_column=1,
        end_row=2,
        end_column=len(METRIC_HEADERS),
    )
    sheet.cell(2, 1, note)
    sheet.cell(2, 1).alignment = Alignment(wrap_text=True, vertical="top")
    sheet.cell(2, 1).fill = PatternFill("solid", fgColor="D9EAF7")
    sheet.row_dimensions[2].height = 36

    header_row = 4
    for column, header in enumerate(METRIC_HEADERS, start=1):
        cell = sheet.cell(header_row, column, header)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="4472C4")
        cell.alignment = Alignment(
            horizontal="center", vertical="center", wrap_text=True
        )
    for row_index, values in enumerate(rows, start=header_row + 1):
        for column, header in enumerate(METRIC_HEADERS, start=1):
            cell = sheet.cell(row_index, column, values[header])
            cell.alignment = Alignment(
                vertical="top",
                wrap_text=header in {
                    "Inference Setting",
                    "Evaluation Scope",
                    "Result Directory",
                },
            )
            if header in PERCENT_COLUMNS:
                cell.number_format = "0.00%"
            elif header.endswith("Tokens") or header in {
                "N",
                "API Errors",
                "Parsing Failures",
                "Preference EM Count",
                "Overall F1 Rank",
                "Preference EM Rank",
            }:
                cell.number_format = "#,##0"

    last_row = header_row + len(rows)
    last_column = len(METRIC_HEADERS)
    table = Table(
        displayName=table_name,
        ref=f"A{header_row}:{get_column_letter(last_column)}{last_row}",
    )
    table.tableStyleInfo = TableStyleInfo(
        name="TableStyleMedium2",
        showFirstColumn=False,
        showLastColumn=False,
        showRowStripes=True,
        showColumnStripes=False,
    )
    sheet.add_table(table)
    sheet.freeze_panes = "H5"
    sheet.auto_filter.ref = table.ref

    header_to_column = {
        header: index for index, header in enumerate(METRIC_HEADERS, start=1)
    }
    for metric in ("Overall F1", "Preference Exact Match"):
        column = get_column_letter(header_to_column[metric])
        sheet.conditional_formatting.add(
            f"{column}{header_row + 1}:{column}{last_row}",
            ColorScaleRule(
                start_type="min",
                start_color="F8696B",
                mid_type="percentile",
                mid_value=50,
                mid_color="FFEB84",
                end_type="max",
                end_color="63BE7B",
            ),
        )

    widths = {
        "Run ID": 23,
        "Method": 15,
        "Memory Source": 16,
        "Inference Model": 20,
        "Inference Setting": 24,
        "Provider / Backend": 22,
        "Evaluation Scope": 28,
        "Result Directory": 55,
    }
    for index, header in enumerate(METRIC_HEADERS, start=1):
        sheet.column_dimensions[get_column_letter(index)].width = widths.get(
            header, 18
        )


def style_status_sheet(
    workbook: Workbook, rows: Sequence[Mapping[str, Any]]
) -> None:
    headers = (
        "Run ID",
        "Method",
        "Memory Source",
        "Inference Model",
        "Target",
        "Final IDs",
        "API Errors",
        "Parsing Failures",
        "Status",
        "Result Directory",
    )
    sheet = workbook.create_sheet("Completion_Status")
    sheet.sheet_view.showGridLines = False
    for column, header in enumerate(headers, start=1):
        cell = sheet.cell(1, column, header)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="4472C4")
        cell.alignment = Alignment(horizontal="center", wrap_text=True)
    for row_index, row in enumerate(rows, start=2):
        for column, header in enumerate(headers, start=1):
            cell = sheet.cell(row_index, column, row[header])
            cell.alignment = Alignment(
                vertical="top",
                wrap_text=header in {"Status", "Result Directory"},
            )
            if header in {
                "Target",
                "Final IDs",
                "API Errors",
                "Parsing Failures",
            }:
                cell.number_format = "#,##0"
        status_cell = sheet.cell(row_index, headers.index("Status") + 1)
        status_cell.fill = PatternFill(
            "solid",
            fgColor=(
                "E2F0D9"
                if row["Status"] == "Clean complete"
                else "FCE4D6"
            ),
        )
    last_row = len(rows) + 1
    table = Table(
        displayName="CompletionStatus",
        ref=f"A1:J{last_row}",
    )
    table.tableStyleInfo = TableStyleInfo(
        name="TableStyleMedium2",
        showRowStripes=True,
    )
    sheet.add_table(table)
    sheet.freeze_panes = "E2"
    widths = (23, 15, 16, 20, 12, 12, 12, 16, 30, 60)
    for column, width in enumerate(widths, start=1):
        sheet.column_dimensions[get_column_letter(column)].width = width


def style_readme(workbook: Workbook, output_path: Path) -> None:
    sheet = workbook.create_sheet("README")
    sheet.sheet_view.showGridLines = False
    rows = (
        ("Workbook", output_path.name),
        ("Generated", datetime.now(timezone.utc).isoformat()),
        ("Dataset", str(ROOT / "data" / "MPT_v2_0725.json")),
        ("Population", "hint query; easy-conflict excluded; N=4,695"),
        ("Comparable sample", str(SAMPLE_MANIFEST)),
        (
            "Primary sheet",
            "Comparable_1000 evaluates all 20 runs on the same 1,000 sample IDs.",
        ),
        (
            "Native sheet",
            "Native_Scope preserves each run's actual scope: 4,695 or 1,000.",
        ),
        (
            "Failure handling",
            "API and parsing failures remain in the denominator and are scored as failures.",
        ),
        (
            "Checkpoint rule",
            "The latest JSONL row for each sample_id is used.",
        ),
        (
            "Important setting difference",
            "Vanilla GPT-5 uses minimal reasoning and vanilla Claude has thinking off; "
            "ours-memory uses GPT-5 medium and Claude thinking budget 2048.",
        ),
        (
            "Ranking",
            "Dense ranks are computed separately within each performance sheet.",
        ),
    )
    for row_index, (key, value) in enumerate(rows, start=1):
        sheet.cell(row_index, 1, key).font = Font(bold=True)
        sheet.cell(row_index, 1).fill = PatternFill(
            "solid", fgColor="D9EAF7"
        )
        sheet.cell(row_index, 2, value).alignment = Alignment(
            wrap_text=True, vertical="top"
        )
    sheet.column_dimensions["A"].width = 28
    sheet.column_dimensions["B"].width = 120
    sheet.freeze_panes = "A1"


def build_workbook(output_path: Path) -> dict[str, Any]:
    common_rows, native_rows, status_rows = evaluate_runs()
    workbook = Workbook()
    workbook.remove(workbook.active)
    style_metric_sheet(
        workbook,
        "Comparable_1000",
        "Vanilla 5 + Ours-memory 3×5 Performance — Common 1,000",
        "All 20 runs are re-evaluated on the identical proportional stratified sample. "
        "Use this sheet for direct model-to-model comparison.",
        common_rows,
        "ComparablePerformance",
    )
    style_metric_sheet(
        workbook,
        "Native_Scope",
        "Vanilla 5 + Ours-memory 3×5 Performance — Native Scope",
        "Full runs use all 4,695 instances; GPT-5/Claude ours-memory runs use the "
        "planned 1,000-instance sample. Use Comparable_1000 for fair comparisons.",
        native_rows,
        "NativeScopePerformance",
    )
    style_status_sheet(workbook, status_rows)
    style_readme(workbook, output_path)
    workbook.active = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output_path)

    verified = load_workbook(output_path, read_only=False, data_only=False)
    expected_sheets = {
        "Comparable_1000",
        "Native_Scope",
        "Completion_Status",
        "README",
    }
    if set(verified.sheetnames) != expected_sheets:
        raise RuntimeError(
            f"Workbook sheet mismatch: {verified.sheetnames}"
        )
    if verified["Comparable_1000"].max_row != 24:
        raise RuntimeError("Comparable_1000 does not contain 20 result rows")
    if verified["Native_Scope"].max_row != 24:
        raise RuntimeError("Native_Scope does not contain 20 result rows")
    if verified["Completion_Status"].max_row != 21:
        raise RuntimeError("Completion_Status does not contain 20 result rows")
    verified.close()

    top_overall = sorted(
        common_rows, key=lambda row: row["Overall F1"], reverse=True
    )[:5]
    top_pref = sorted(
        common_rows,
        key=lambda row: row["Preference Exact Match"],
        reverse=True,
    )[:5]
    return {
        "output": str(output_path),
        "run_count": len(common_rows),
        "sheets": sorted(expected_sheets),
        "top_overall_f1": [
            {
                "run_id": row["Run ID"],
                "overall_f1": row["Overall F1"],
            }
            for row in top_overall
        ],
        "top_preference_em": [
            {
                "run_id": row["Run ID"],
                "preference_em": row["Preference Exact Match"],
            }
            for row in top_pref
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = build_workbook(args.output.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
