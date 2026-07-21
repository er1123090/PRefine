"""Pure CP1 inventory construction from a captured, immutable paper layout.

This module accepts only already-captured page text.  It never opens a PDF,
legacy inventory, protected experiment root, or external output path.
"""
from __future__ import annotations

from decimal import Decimal
import hashlib
import json
import re
from typing import Any

TABLE_PAGES = {
    1: 5, 2: 6, 3: 7, 4: 14, 5: 14, 6: 15, 7: 15, 8: 17,
    9: 18, 10: 19, 11: 20, 12: 21, 13: 22, 14: 22, 15: 23, 16: 23,
}
FIGURE_PAGES = {1: 1, 2: 4, 3: 5, 4: 8, 5: 9, 6: 15, 7: 16, 8: 16, 9: 24, 10: 25}
FORBIDDEN_INPUT_FIELDS = {"status", "verified", "derived_state"}
VALUE_2_RE = re.compile(r"(?:N/A|[-+]?(?:100\.00|\d{1,2}\.\d{2}))")
VALUE_3_RE = re.compile(r"[-+]?\d+\.\d{3}")
PERCENT_2_RE = re.compile(r"(?:100|\d{1,2})\.\d{2}%")
EXPECTED_LAYOUT_SHA256 = ""


class Blocked(RuntimeError):
    """A fail-closed captured-layout inventory condition."""


def stable_id(prefix: str, key: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", key.lower()).strip("-")
    if prefix == "result" and key.startswith("f5:panel-a:"):
        readable = "paper.f05.a." + normalized.removeprefix("f5-panel-a-")
    elif prefix == "result" and key.startswith("f5:panel-b:"):
        readable = "paper.f05.b." + normalized.removeprefix("f5-panel-b-")
    else:
        readable = normalized[:72]
    suffix = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}:{readable}:{suffix}"


def assert_no_forbidden_fields(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        forbidden = sorted(FORBIDDEN_INPUT_FIELDS.intersection(value))
        if forbidden:
            raise Blocked(f"FORBIDDEN_INPUT_FIELD:{path}:{forbidden[0]}")
        for key, child in value.items():
            assert_no_forbidden_fields(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_no_forbidden_fields(child, f"{path}[{index}]")


def canonical_locator(record: dict[str, Any]) -> str:
    return json.dumps(record["locator"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def page_line(pages: list[str], page: int, line: int) -> str:
    lines = pages[page - 1].splitlines()
    if line < 1 or line > len(lines):
        raise Blocked(f"LINE_OUT_OF_RANGE:{page}:{line}")
    return lines[line - 1]


def require_text(text: str, needle: str, reason: str) -> None:
    if needle not in text:
        raise Blocked(f"TEXT_ANCHOR_MISSING:{reason}:{needle}")


def fixed_values_after(text: str, anchor: str, expected: int, pattern: re.Pattern[str] = VALUE_2_RE) -> list[str]:
    position = text.rfind(anchor)
    if position < 0:
        raise Blocked(f"ROW_ANCHOR_MISSING:{anchor}")
    values = pattern.findall(text[position + len(anchor):])
    if len(values) != expected:
        raise Blocked(f"VALUE_COUNT_MISMATCH:{anchor}:{len(values)}:{expected}")
    return values


def decimal_text(value: str) -> str:
    cleaned = value.replace(",", "").replace("%", "")
    return format(Decimal(cleaned), "f")

class InventoryBuilder:
    def __init__(self) -> None:
        self.results: list[dict[str, Any]] = []
        self.aliases: list[dict[str, Any]] = []
        self.by_key: dict[str, dict[str, Any]] = {}
        self.alias_keys: set[str] = set()
        self.source_counts: dict[str, int] = {}

    def add_result(
        self,
        key: str,
        *,
        source: str,
        result_kind: str,
        page: int,
        section: str,
        locator: dict[str, Any],
        value: str | None,
        value_type: str,
        method: str | None = None,
        model: str | None = None,
        dataset: str | None = "MPT",
        setting: str | None = None,
        preference_type: str | None = None,
        metric: str | None = None,
        precision: int | None = 2,
        displayed_marker: str | None = None,
        source_lexeme: str | None = None,
        numeric_value: str | None = None,
        derived_from: list[str] | None = None,
    ) -> str:
        if key in self.by_key:
            raise Blocked(f"DUPLICATE_RESULT_KEY:{key}")
        result_id = stable_id("result", key)
        if value_type in {"numeric", "derived"} and numeric_value is None:
            if value is None:
                raise Blocked(f"NUMERIC_VALUE_MISSING:{key}")
            numeric_value = decimal_text(value)
        record: dict[str, Any] = {
            "schema": "paper-result-v1",
            "result_id": result_id,
            "result_kind": result_kind,
            "page": page,
            "section": section,
            "locator": locator,
            "method": method,
            "model": model,
            "dataset": dataset,
            "setting": setting,
            "preference_type": preference_type,
            "metric": metric,
            "displayed_value": value,
            "numeric_value": numeric_value,
            "value_type": value_type,
            "displayed_marker": displayed_marker,
            "source_lexeme": source_lexeme if source_lexeme is not None else value,
            "rounding_rule": {
                "display_precision": precision,
                "mode": "not_inferred_from_layout",
                "evidence_requirement": "producer_binding",
            },
            "evidence_id": locator.get("evidence_id"),
        }
        if derived_from is not None:
            record["derived_from"] = list(derived_from)
        self.results.append(record)
        self.by_key[key] = record
        self.source_counts[source] = self.source_counts.get(source, 0) + 1
        return result_id

    def result(self, key: str) -> dict[str, Any]:
        try:
            return self.by_key[key]
        except KeyError as exc:
            raise Blocked(f"ALIAS_TARGET_KEY_MISSING:{key}") from exc

    def add_alias(
        self,
        key: str,
        *,
        alias_of_key: str,
        page: int,
        section: str,
        locator: dict[str, Any],
        displayed_lexeme: str,
        reason: str,
    ) -> str:
        if key in self.alias_keys:
            raise Blocked(f"DUPLICATE_ALIAS_KEY:{key}")
        alias_id = stable_id("alias", key)
        target = self.result(alias_of_key)
        self.aliases.append({
            "schema": "paper-alias-v1",
            "alias_id": alias_id,
            "alias_of": target["result_id"],
            "page": page,
            "section": section,
            "locator": locator,
            "displayed_lexeme": displayed_lexeme,
            "reason": reason,
        })
        self.alias_keys.add(key)
        return alias_id


def build_structure(pages: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for page_number, text in enumerate(pages, start=1):
        records.append({
            "schema": "paper-structure-v1",
            "structure_id": f"page:{page_number}",
            "kind": "page",
            "page": page_number,
            "line_start": 1,
            "line_end": len(text.splitlines()),
            "caption": None,
            "text_sha256": EXPECTED_LAYOUT_SHA256,
        })
    for kind, page_map in (("table", TABLE_PAGES), ("figure", FIGURE_PAGES)):
        for number, page_number in sorted(page_map.items()):
            pattern = re.compile(rf"{kind}\s*{number}\s*:", re.IGNORECASE)
            matches = [(index, line) for index, line in enumerate(pages[page_number - 1].splitlines(), start=1) if pattern.search(line)]
            if len(matches) != 1:
                raise Blocked(f"STRUCTURE_CAPTION_COUNT:{kind}:{number}:{len(matches)}")
            line_number, caption = matches[0]
            records.append({
                "schema": "paper-structure-v1",
                "structure_id": f"{kind}:{number}",
                "kind": kind,
                "page": page_number,
                "line_start": line_number,
                "line_end": line_number,
                "caption": caption.strip(),
                "text_sha256": EXPECTED_LAYOUT_SHA256,
            })
    return records


def t3_columns(include_averages: bool) -> list[dict[str, str]]:
    guided: list[dict[str, str]] = []
    free: list[dict[str, str]] = []
    for preference in ("Recall", "Induction", "Transfer"):
        for metric in ("P-EM", "EA-F1", "OA-F1"):
            guided.append({
                "id": f"context-guided:{preference}:{metric}",
                "query_setting": "context-guided",
                "preference_type": preference,
                "metric": metric,
            })
        for metric in ("Precision", "Recall", "F1"):
            free.append({
                "id": f"context-free:{preference}:{metric}",
                "query_setting": "context-free",
                "preference_type": preference,
                "metric": metric,
            })
    if not include_averages:
        return guided + free
    return guided + [{
        "id": "context-guided:Average:OA-F1",
        "query_setting": "context-guided",
        "preference_type": "Average",
        "metric": "OA-F1",
    }] + free + [{
        "id": "context-free:Average:F1",
        "query_setting": "context-free",
        "preference_type": "Average",
        "metric": "F1",
    }]


def parse_table3(pages: list[str], inventory: InventoryBuilder) -> None:
    rows = [
        (75, "base:CodeGemma-7B", "CodeGemma-7B", "Base Prompting", "CodeGemma-7B", 20),
        (76, "base:Gemma-3-12B", "Gemma-3-12B", "Base Prompting", "Gemma-3-12B", 20),
        (77, "base:R1-Distill-Llama-8B", "R1-Distill-Llama-8B", "Base Prompting", "R1-Distill-Llama-8B", 20),
        (78, "base:R1-Distill-Qwen-7B", "R1-Distill-Qwen-7B", "Base Prompting", "R1-Distill-Qwen-7B", 20),
        (79, "base:GPT-4o-mini", "GPT-4o-mini", "Base Prompting", "GPT-4o-mini", 20),
        (80, "base:GPT-5-mini", "GPT-5-mini", "Base Prompting", "GPT-5-mini", 20),
        (81, "base:GPT-5", "GPT-5", "Base Prompting", "GPT-5", 20),
        (82, "base:Gemini-3-Flash", "Gemini-3-Flash", "Base Prompting", "Gemini-3-Flash", 20),
        (83, "base:Average", "Average", "Base Prompting", "Average", 18),
        (85, "memory:RAG(Top-5)", "RAG(Top-5)", "RAG(Top-5)", "Gemini-3-Flash", 20),
        (86, "memory:Mem0", "Mem0", "Mem0", "Gemini-3-Flash", 20),
        (87, "memory:LangMem", "LangMem", "LangMem", "Gemini-3-Flash", 20),
        (89, "prefine:CodeGemma-7B", "CodeGemma-7B", "PREFINE", "CodeGemma-7B", 20),
        (90, "prefine:Gemma-3-12B", "Gemma-3-12B", "PREFINE", "Gemma-3-12B", 20),
        (91, "prefine:R1-Distill-Llama-8B", "R1-Distill-Llama-8B", "PREFINE", "R1-Distill-Llama-8B", 20),
        (92, "prefine:R1-Distill-Qwen-7B", "R1-Distill-Qwen-7B", "PREFINE", "R1-Distill-Qwen-7B", 20),
        (93, "prefine:GPT-4o-mini", "GPT-4o-mini", "PREFINE", "GPT-4o-mini", 20),
        (94, "prefine:GPT-5-mini", "GPT-5-mini", "PREFINE", "GPT-5-mini", 20),
        (95, "prefine:GPT-5", "GPT-5", "PREFINE", "GPT-5", 20),
        (96, "prefine:Gemini-3-Flash", "Gemini-3-Flash", "PREFINE", "Gemini-3-Flash", 20),
        (97, "prefine:Avg.Gain", "Avg.Gain(%p)", "PREFINE", "Average Gain", 18),
        (98, "prefine:Average", "Average", "PREFINE", "Average", 18),
    ]
    for line_number, row_key, anchor, method, model, expected in rows:
        text = page_line(pages, 7, line_number)
        values = fixed_values_after(text, anchor, expected)
        columns = t3_columns(expected == 20)
        if len(columns) != len(values):
            raise Blocked(f"TABLE3_COLUMN_COUNT:{row_key}")
        for column, value in zip(columns, values):
            column_id = column["id"]
            inventory.add_result(
                f"t3:{row_key}:{column_id}",
                source="table3",
                result_kind="table_cell",
                page=7,
                section="7.1",
                locator={
                    "evidence_id": "table:3",
                    "table": 3,
                    "line": line_number,
                    "row": row_key,
                    "column": column["id"],
                    "extraction_rule": "fixed_two_decimal_after_row_anchor",
                },
                value=value,
                value_type="numeric",
                method=method,
                model=model,
                setting=column["query_setting"],
                preference_type=column["preference_type"],
                metric=column["metric"],
            )


def parse_table10(pages: list[str], inventory: InventoryBuilder) -> None:
    models = [
        ("CodeGemma-7B", "Non-Reasoning"),
        ("Gemma-3-12B", "Non-Reasoning"),
        ("R1-Distill-Llama-8B", "Reasoning"),
        ("R1-Distill-Qwen-7B", "Reasoning"),
        ("GPT-4o-mini", "Proprietary"),
        ("GPT-5-mini", "Proprietary"),
        ("Gemini-3-Flash", "Proprietary"),
        ("GPT-5", "Proprietary"),
    ]
    for query_setting, first_line in (("context-guided", 21), ("context-free", 31)):
        metrics = ("P-EM", "EA-F1", "OA-F1") if query_setting == "context-guided" else ("Precision", "Recall", "F1")
        columns = [(preference, metric) for preference in ("Recall", "Induction", "Transfer") for metric in metrics]
        for offset, (model, model_type) in enumerate(models):
            line_number = first_line + offset
            values = fixed_values_after(page_line(pages, 19, line_number), model, 9)
            for (preference, metric), value in zip(columns, values):
                column = f"{query_setting}:{preference}:{metric}"
                inventory.add_result(
                    f"t10:{query_setting}:{model}:{preference}:{metric}",
                    source="table10",
                    result_kind="table_cell",
                    page=19,
                    section="C.1",
                    locator={
                        "evidence_id": "table:10",
                        "table": 10,
                        "line": line_number,
                        "row": model,
                        "column": column,
                        "model_type": model_type,
                        "statistic": "PREFINE_minus_base",
                        "extraction_rule": "fixed_two_decimal_after_model_anchor",
                    },
                    value=value,
                    value_type="numeric",
                    method="PREFINE minus Base Prompting",
                    model=model,
                    setting=query_setting,
                    preference_type=preference,
                    metric=metric,
                )


CONTEXT_GUIDED_MODEL_ANCHORS = (
    "CodeGemma-7B-Instruct",
    "Gemma-3-12B-Instruct",
    "R1-distill-Llama-8B",
    "R1-distill-Qwen-7B",
    "Gemini-3-Flash [high]",
    "GPT-5-mini [high]",
    "GPT-4o-mini",
    "GPT-5 [high]",
)


def model_anchor_in(text: str) -> str:
    matches = [model for model in CONTEXT_GUIDED_MODEL_ANCHORS if model in text]
    if len(matches) != 1:
        raise Blocked(f"MODEL_ANCHOR_COUNT:{len(matches)}:{text[:120]}")
    return matches[0]


def parse_context_guided_block(
    pages: list[str],
    inventory: InventoryBuilder,
    *,
    table: int,
    page: int,
    first_line: int,
    source: str,
    method: str,
    memory_backbone: str | None,
) -> None:
    metrics = ("P-EM", "EA-Precision", "EA-Recall", "EA-F1", "OA-Precision", "OA-Recall", "OA-F1")
    for offset in range(24):
        line_number = first_line + offset
        text = page_line(pages, page, line_number)
        model_anchor = model_anchor_in(text)
        values = fixed_values_after(text, model_anchor, 7)
        preference = ("Recall", "Induction", "Transfer")[offset // 8]
        normalized_model = model_anchor.replace(" [high]", "")
        for metric, value in zip(metrics, values):
            block_name = memory_backbone if memory_backbone is not None else "base"
            inventory.add_result(
                f"t{table}:{block_name}:{preference}:{normalized_model}:{metric}",
                source=source,
                result_kind="table_cell",
                page=page,
                section="C.2",
                locator={
                    "evidence_id": f"table:{table}",
                    "table": table,
                    "line": line_number,
                    "row": normalized_model,
                    "column": metric,
                    "block": block_name,
                    "extraction_rule": "seven_fixed_two_decimal_values_after_model_anchor",
                },
                value=value,
                value_type="numeric",
                method=method,
                model=normalized_model,
                setting=f"context-guided:{block_name}",
                preference_type=preference,
                metric=metric,
            )


def parse_table11(pages: list[str], inventory: InventoryBuilder) -> None:
    parse_context_guided_block(
        pages,
        inventory,
        table=11,
        page=20,
        first_line=8,
        source="table11",
        method="Base Prompting",
        memory_backbone=None,
    )
    parse_context_guided_block(
        pages,
        inventory,
        table=11,
        page=20,
        first_line=35,
        source="table11",
        method="PREFINE",
        memory_backbone="Gemma-3-12B-it",
    )
    parse_context_guided_block(
        pages,
        inventory,
        table=11,
        page=20,
        first_line=62,
        source="table11",
        method="PREFINE",
        memory_backbone="GPT-4o-mini",
    )


def parse_table12(pages: list[str], inventory: InventoryBuilder) -> None:
    parse_context_guided_block(
        pages,
        inventory,
        table=12,
        page=21,
        first_line=24,
        source="table12",
        method="PREFINE",
        memory_backbone="R1-Distill-Llama-8B",
    )
    parse_context_guided_block(
        pages,
        inventory,
        table=12,
        page=21,
        first_line=51,
        source="table12",
        method="PREFINE",
        memory_backbone="R1-Distill-Qwen-7B",
    )


TABLE13_MODEL_ANCHORS = (
    "R1-Distill-Llama-8B",
    "R1-Distill-Qwen-7B",
    "Gemini-3-Flash[high]",
    "CodeGemma-7B-it",
    "Gemma-3-12b-it",
    "GPT-5-mini[high]",
    "GPT-4o-mini",
)


def parse_table13(pages: list[str], inventory: InventoryBuilder) -> None:
    backbones = ("Gemma-3-12B-it", "GPT-4o-mini", "R1-Distill-Llama-8B", "R1-Distill-Qwen-7B")
    columns = [(backbone, metric) for backbone in backbones for metric in ("Precision", "Recall", "F1")]
    for offset in range(21):
        line_number = 74 + offset
        text = page_line(pages, 22, line_number)
        model_anchor = TABLE13_MODEL_ANCHORS[offset % 7]
        position = text.rfind(model_anchor)
        if position < 0:
            raise Blocked(f"TABLE13_MODEL_ANCHOR_MISSING:{line_number}:{model_anchor}")
        candidates = VALUE_2_RE.findall(text[position + len(model_anchor):])
        expected_candidates = 27 if line_number in {81, 82, 83, 84, 86, 87} else 12
        if len(candidates) != expected_candidates:
            raise Blocked(f"TABLE13_CANDIDATE_COUNT:{line_number}:{len(candidates)}:{expected_candidates}")
        if expected_candidates == 27:
            table13_indices = (0, 3, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21)
            values = [candidates[index] for index in table13_indices]
        else:
            values = candidates
        preference = ("Recall", "Induction", "Transfer")[offset // 7]
        normalized_model = model_anchor.replace("[high]", "")
        for (backbone, metric), value in zip(columns, values):
            inventory.add_result(
                f"t13:{preference}:{normalized_model}:{backbone}:{metric}",
                source="table13",
                result_kind="table_cell",
                page=22,
                section="C.3",
                locator={
                    "evidence_id": "table:13",
                    "table": 13,
                    "line": line_number,
                    "row": normalized_model,
                    "column": f"{backbone}:{metric}",
                    "extraction_rule": "first_twelve_fixed_two_decimal_values_after_last_model_anchor",
                },
                value=value,
                value_type="numeric",
                method="PREFINE",
                model=normalized_model,
                setting=f"context-free:{backbone}",
                preference_type=preference,
                metric=metric,
            )


def parse_table14(pages: list[str], inventory: InventoryBuilder) -> None:
    rows = (
        (81, "Mem0", "Gemini-3-Flash"),
        (82, "Mem0", "GPT-5"),
        (83, "RAG", "Gemini-3-Flash"),
        (84, "RAG", "GPT-5"),
        (86, "LangMem", "Gemini-3-Flash"),
        (87, "LangMem", "GPT-5"),
    )
    columns = t3_columns(False)
    memory_target = {"Mem0": "Mem0", "RAG": "RAG(Top-5)", "LangMem": "LangMem"}
    for line_number, method, model in rows:
        text = page_line(pages, 22, line_number)
        anchor = f"{method}({model})"
        require_text(text, anchor, f"table14:{line_number}")
        all_values = VALUE_2_RE.findall(text)
        if len(all_values) != 30:
            raise Blocked(f"TABLE14_CONTAMINATED_LINE_VALUE_COUNT:{line_number}:{len(all_values)}")
        table14_indices = (0, 1, 2, 4, 5, 7, 8, 9, 10, 11, 12, 13, 20, 25, 26, 27, 28, 29)
        values = [all_values[index] for index in table14_indices]
        if len(values) != 18:
            raise Blocked(f"TABLE14_VALUE_COUNT:{line_number}:{len(values)}")
        for column, value in zip(columns, values):
            column_id = column["id"]
            location = {
                "evidence_id": "table:14",
                "table": 14,
                "line": line_number,
                "row": anchor,
                "column": column["id"],
                "extraction_rule": "first_three_plus_last_fifteen_of_thirty_fixed_two_decimal_tokens",
            }
            if model == "Gemini-3-Flash":
                target_key = f"t3:memory:{memory_target[method]}:{column_id}"
                target = inventory.result(target_key)
                if Decimal(target["numeric_value"]) != Decimal(value):
                    raise Blocked(f"TABLE14_ALIAS_VALUE_MISMATCH:{method}:{column_id}:{value}")
                inventory.add_alias(
                    f"t14:{method}:{model}:{column_id}",
                    alias_of_key=target_key,
                    page=22,
                    section="C.4",
                    locator=location,
                    displayed_lexeme=value,
                    reason="table14_repeated_gemini_location",
                )
            else:
                inventory.add_result(
                    f"t14:{method}:{model}:{column_id}",
                    source="table14",
                    result_kind="table_cell",
                    page=22,
                    section="C.4",
                    locator=location,
                    value=value,
                    value_type="numeric",
                    method=method,
                    model=model,
                    setting=column["query_setting"],
                    preference_type=column["preference_type"],
                    metric=column["metric"],
                )


def parse_table15(pages: list[str], inventory: InventoryBuilder) -> None:
    rows = (
        (9, "context-guided", "Recall"),
        (10, "context-guided", "Induction"),
        (11, "context-guided", "Transfer"),
        (12, "context-free", "Recall"),
        (13, "context-free", "Induction"),
        (14, "context-free", "Transfer"),
    )
    delta_keys: dict[tuple[str, str], str] = {}
    for line_number, query_setting, preference in rows:
        text = page_line(pages, 23, line_number)
        values = fixed_values_after(text, preference, 3, VALUE_3_RE)
        score10_key = f"t15:{query_setting}:{preference}:score10"
        score3_key = f"t15:{query_setting}:{preference}:score3"
        common_locator = {"evidence_id": "table:15", "table": 15, "line": line_number, "row": f"{query_setting}:{preference}"}
        score10_id = inventory.add_result(
            score10_key,
            source="table15",
            result_kind="table_cell",
            page=23,
            section="C.5",
            locator={**common_locator, "column": "10-iterations", "extraction_rule": "fixed_three_decimal"},
            value=values[0],
            value_type="numeric",
            method="PREFINE",
            setting=query_setting,
            preference_type=preference,
            metric="aggregate_score",
            precision=3,
        )
        score3_id = inventory.add_result(
            score3_key,
            source="table15",
            result_kind="table_cell",
            page=23,
            section="C.5",
            locator={**common_locator, "column": "3-iterations", "extraction_rule": "fixed_three_decimal"},
            value=values[1],
            value_type="numeric",
            method="PREFINE",
            setting=query_setting,
            preference_type=preference,
            metric="aggregate_score",
            precision=3,
        )
        delta_key = f"t15:{query_setting}:{preference}:delta"
        inventory.add_result(
            delta_key,
            source="table15",
            result_kind="derived_value",
            page=23,
            section="C.5",
            locator={**common_locator, "column": "delta_10_minus_3", "extraction_rule": "fixed_three_decimal"},
            value=values[2],
            value_type="numeric",
            method="PREFINE",
            setting=query_setting,
            preference_type=preference,
            metric="delta_10_minus_3",
            precision=3,
            derived_from=[score10_id, score3_id],
        )
        delta_keys[(query_setting, preference)] = delta_key
    for preference, displayed, line_number in (
        ("Recall", "−0.006", 58),
        ("Induction", "−0.006", 58),
        ("Transfer", "+0.034", 60),
    ):
        require_text(page_line(pages, 22, line_number), displayed, f"table15-prose:{preference}")
        inventory.add_alias(
            f"t15-prose:context-guided:{preference}",
            alias_of_key=delta_keys[("context-guided", preference)],
            page=22,
            section="C.5",
            locator={"evidence_id": "page:22", "line": line_number, "semantic_target": f"context-guided:{preference}"},
            displayed_lexeme=displayed,
            reason="table15_prose_repetition",
        )
    for preference, displayed in (("Induction", "+0.002"), ("Recall", "+0.004")):
        require_text(page_line(pages, 22, 59), displayed, f"table15-context-free-endpoint:{preference}")
        inventory.add_alias(
            f"t15-prose:context-free:{preference}",
            alias_of_key=delta_keys[("context-free", preference)],
            page=22,
            section="C.5",
            locator={
                "evidence_id": "page:22",
                "line": 59,
                "semantic_target": f"context-free:{preference}",
            },
            displayed_lexeme=displayed,
            reason="table15_prose_repetition",
        )


def parse_table16(pages: list[str], inventory: InventoryBuilder) -> None:
    rows = (
        (33, "Base Prompting", "Gemini-3-Flash"),
        (34, "Base Prompting", "GPT-5"),
        (35, "RAG", "Gemini-3-Flash"),
        (36, "RAG", "GPT-5"),
        (37, "Mem0", "Gemini-3-Flash"),
        (38, "Mem0", "GPT-5"),
        (39, "LangMem", "Gemini-3-Flash"),
        (40, "LangMem", "GPT-5"),
        (41, "PREFINE", "Gemini-3-Flash"),
        (42, "PREFINE", "GPT-5"),
    )
    metrics = ("P-EM", "EA-F1", "OA-F1", "Precision", "Recall", "F1")
    for line_number, method, model in rows:
        text = page_line(pages, 23, line_number)
        if method == "Mem0":
            if text.count("–") != 6:
                raise Blocked(f"TABLE16_NA_MARKER_COUNT:{line_number}:{text.count(chr(0x2013))}")
            values: list[str | None] = [None] * 6
        else:
            extracted = PERCENT_2_RE.findall(text)
            if len(extracted) != 6:
                raise Blocked(f"TABLE16_PERCENT_COUNT:{line_number}:{len(extracted)}")
            values = list(extracted)
        for index, (metric, value) in enumerate(zip(metrics, values)):
            query_setting = "context-guided" if index < 3 else "context-free"
            inventory.add_result(
                f"t16:{method}:{model}:{metric}",
                source="table16",
                result_kind="table_cell",
                page=23,
                section="C.6",
                locator={
                    "evidence_id": "table:16",
                    "table": 16,
                    "line": line_number,
                    "row": f"{method}:{model}",
                    "column": metric,
                    "extraction_rule": "six_fixed_percent_values_or_six_en_dash_markers",
                },
                value=value if value is not None else "N/A",
                value_type="numeric" if value is not None else "not_applicable",
                method=method,
                model=model,
                setting=f"dynamic-schema:{query_setting}",
                metric=metric,
                displayed_marker=None if value is not None else "N/A",
                source_lexeme=value if value is not None else "–",
                numeric_value=decimal_text(value) if value is not None else None,
            )
    repeated = (
        ("Base Prompting", "P-EM", "3.75%"),
        ("PREFINE", "P-EM", "47.00%"),
        ("Base Prompting", "F1", "36.39%"),
        ("PREFINE", "F1", "51.45%"),
    )
    for location_name, page, line, reason in (
        ("main", 9, 55, "table16_main_text_repetition"),
        ("appendix", 23, 25, "table16_appendix_text_repetition"),
    ):
        combined = page_line(pages, page, line)
        if location_name == "appendix":
            combined += " " + page_line(pages, 23, 26)
        for method, metric, displayed in repeated:
            require_text(combined, displayed, f"table16:{location_name}:{method}:{metric}")
            inventory.add_alias(
                f"t16-{location_name}:{method}:{metric}",
                alias_of_key=f"t16:{method}:GPT-5:{metric}",
                page=page,
                section="7.5" if location_name == "main" else "C.6",
                locator={"evidence_id": f"page:{page}", "line": line, "semantic_target": f"{method}:GPT-5:{metric}"},
                displayed_lexeme=displayed,
                reason=reason,
            )


def parse_figure4(pages: list[str], inventory: InventoryBuilder) -> None:
    require_text(page_line(pages, 8, 20), "Figure4:", "figure4-caption")
    require_text(page_line(pages, 9, 37), "3.34", "figure4-r1-base")
    require_text(page_line(pages, 9, 37), "2.85", "figure4-r1-prefine")
    models = (
        "CodeGemma-7B",
        "Gemma-3-12B",
        "R1-Distill-Llama-8B",
        "R1-Distill-Qwen-7B",
        "GPT-4o-mini",
        "GPT-5-mini",
        "GPT-5",
        "Gemini-3-Flash",
    )
    exact = {
        ("context-guided", "R1-Distill-Llama-8B", "Base Prompting"): "3.34",
        ("context-guided", "R1-Distill-Llama-8B", "PREFINE"): "2.85",
    }
    mark_keys: dict[tuple[str, str, str], str] = {}
    for query_setting in ("context-guided", "context-free"):
        for model in models:
            for method in ("Base Prompting", "PREFINE"):
                key = f"f4:mark:{query_setting}:{model}:{method}"
                value = exact.get((query_setting, model, method))
                inventory.add_result(
                    key,
                    source="figure4",
                    result_kind="figure_mark",
                    page=8,
                    section="7.2",
                    locator={
                        "evidence_id": "figure:4",
                        "figure": 4,
                        "panel": query_setting,
                        "mark": f"{model}:{method}",
                        "extraction_rule": "explicit_narrative_binding" if value is not None else "unlabeled_graphical_mark_no_geometry_estimate",
                    },
                    value=value,
                    value_type="numeric" if value is not None else "unlabeled_graphical_mark",
                    method=method,
                    model=model,
                    setting=query_setting,
                    metric="average_predicted_api_arguments",
                    displayed_marker=None if value is not None else "UNLABELED_GRAPHICAL_MARK",
                    source_lexeme=value,
                )
                mark_keys[(query_setting, model, method)] = key
    reference_keys: dict[str, str] = {}
    for query_setting, value in (("context-guided", "3.84"), ("context-free", "1.11")):
        key = f"f4:reference:{query_setting}"
        inventory.add_result(
            key,
            source="figure4",
            result_kind="figure_mark",
            page=8,
            section="7.2",
            locator={
                "evidence_id": "figure:4",
                "figure": 4,
                "panel": query_setting,
                "mark": "ground-truth-reference",
                "extraction_rule": "printed_reference_value",
            },
            value=value,
            value_type="numeric",
            method="Ground Truth",
            setting=query_setting,
            metric="average_ground_truth_arguments",
        )
        reference_keys[query_setting] = key
    for method, displayed in (("Base Prompting", "3.34"), ("PREFINE", "2.85")):
        inventory.add_alias(
            f"f4-prose:r1:{method}",
            alias_of_key=mark_keys[("context-guided", "R1-Distill-Llama-8B", method)],
            page=9,
            section="7.4",
            locator={"evidence_id": "page:9", "line": 37, "semantic_target": f"R1-Distill-Llama-8B:{method}"},
            displayed_lexeme=displayed,
            reason="figure4_prose_mark_repetition",
        )
    deviations = {
        "context-guided": ("0.77", "0.56", "28.1"),
        "context-free": ("1.08", "0.77", "28.7"),
    }
    for query_setting, (base_value, prefine_value, reduction_value) in deviations.items():
        line = 50 if query_setting == "context-guided" else 51
        combined = " ".join(page_line(pages, 8, current) for current in range(line, 53))
        for displayed in (base_value, prefine_value, reduction_value):
            require_text(combined, displayed, f"figure4-mad:{query_setting}:{displayed}")
        source_ids = [
            inventory.result(mark_keys[(query_setting, model, method)])["result_id"]
            for model in models
            for method in ("Base Prompting", "PREFINE")
        ] + [inventory.result(reference_keys[query_setting])["result_id"]]
        base_key = f"f4-mad:{query_setting}:base"
        prefine_key = f"f4-mad:{query_setting}:prefine"
        base_id = inventory.add_result(
            base_key,
            source="narrative_figure4",
            result_kind="derived_value",
            page=8,
            section="7.2",
            locator={
                "evidence_id": "page:8",
                "line": line,
                "semantic_target": "Base Prompting",
                "extraction_rule": "explicit_narrative_aggregate",
            },
            value=base_value,
            value_type="derived",
            method="Base Prompting",
            setting=query_setting,
            metric="mean_absolute_deviation_argument_count",
            precision=2,
            derived_from=source_ids,
        )
        prefine_id = inventory.add_result(
            prefine_key,
            source="narrative_figure4",
            result_kind="derived_value",
            page=8,
            section="7.2",
            locator={
                "evidence_id": "page:8",
                "line": line,
                "semantic_target": "PREFINE",
                "extraction_rule": "explicit_narrative_aggregate",
            },
            value=prefine_value,
            value_type="derived",
            method="PREFINE",
            setting=query_setting,
            metric="mean_absolute_deviation_argument_count",
            precision=2,
            derived_from=source_ids,
        )
        inventory.add_result(
            f"f4-mad:{query_setting}:reduction",
            source="narrative_figure4",
            result_kind="derived_value",
            page=8,
            section="7.2",
            locator={"evidence_id": "page:8", "line": 52, "semantic_target": query_setting, "extraction_rule": "explicit_narrative_percentage"},
            value=reduction_value + "%",
            value_type="derived",
            method="PREFINE versus Base Prompting",
            setting=query_setting,
            metric="mean_absolute_deviation_reduction_pct",
            precision=1,
            derived_from=[base_id, prefine_id],
        )


def exact_occurrences(pages: list[str], needle: str) -> list[tuple[int, int]]:
    return [
        (page_number, line_number)
        for page_number, page in enumerate(pages, start=1)
        for line_number, line in enumerate(page.splitlines(), start=1)
        if needle in line
    ]


def parse_figure5(pages: list[str], inventory: InventoryBuilder) -> None:
    require_text(page_line(pages, 9, 20), "Figure 5:", "figure5-caption")
    panel_a = (
        ("Base Prompting", "1883.57"),
        ("LangMem", "209.22"),
        ("RAG(Top-5)", "133.58"),
        ("Mem0", "119.87"),
        ("PREFINE", "23.28"),
    )
    panel_a_keys: dict[str, str] = {}
    for method, value in panel_a:
        key = f"f5:panel-a:{method}"
        inventory.add_result(
            key,
            source="figure5",
            result_kind="figure_mark",
            page=9,
            section="7.3",
            locator={
                "evidence_id": "figure:5",
                "figure": 5,
                "panel": "a",
                "mark": method,
                "asset": "diagnostics/figure5-source.png",
                "extraction_rule": "pinned_asset_exact_label_transcription",
            },
            value=value,
            value_type="numeric",
            method=method,
            setting="test-time-retrieval",
            metric="average_retrieved_tokens",
        )
        panel_a_keys[method] = key
    session10 = {
        "Base Prompting": "2812.40",
        "LangMem": "2792.35",
        "Mem0": "626.28",
        "PREFINE": "24.79",
    }
    panel_b_ids: list[str] = []
    for method in ("Base Prompting", "LangMem", "Mem0", "PREFINE"):
        for session in range(1, 11):
            key = f"f5:panel-b:{method}:session-{session}"
            value = session10.get(method) if session == 10 else None
            result_id = inventory.add_result(
                key,
                source="figure5",
                result_kind="figure_mark",
                page=9,
                section="7.3",
                locator={
                    "evidence_id": "figure:5",
                    "figure": 5,
                    "panel": "b",
                    "mark": f"{method}:session-{session}",
                    "asset": "diagnostics/figure5-source.png",
                    "extraction_rule": "pinned_asset_exact_endpoint_label" if value is not None else "unlabeled_graphical_mark_no_geometry_estimate",
                },
                value=value,
                value_type="numeric" if value is not None else "unlabeled_graphical_mark",
                method=method,
                setting=f"accumulated-session-{session}",
                metric="memory_tokens",
                displayed_marker=None if value is not None else "UNLABELED_GRAPHICAL_MARK",
                source_lexeme=value,
            )
            panel_b_ids.append(result_id)
    require_text(page_line(pages, 8, 62), "23.28", "figure5-prose-prefine")
    inventory.add_alias(
        "f5-prose:23.28",
        alias_of_key=panel_a_keys["PREFINE"],
        page=8,
        section="7.3",
        locator={"evidence_id": "page:8", "line": 62, "semantic_target": "PREFINE:panel-a"},
        displayed_lexeme="23.28",
        reason="figure5_prose_mark_repetition",
    )
    require_text(page_line(pages, 8, 63), "1.24%", "figure5-ratio")
    ratio_key = "f5-derived:prefine-over-base-ratio"
    inventory.add_result(
        ratio_key,
        source="narrative_figure5",
        result_kind="derived_value",
        page=8,
        section="7.3",
        locator={"evidence_id": "page:8", "line": 63, "extraction_rule": "explicit_narrative_ratio"},
        value="1.24%",
        value_type="derived",
        method="PREFINE versus Base Prompting",
        setting="test-time-retrieval",
        metric="retrieved_token_ratio_pct",
        precision=2,
        derived_from=[
            inventory.result(panel_a_keys["PREFINE"])["result_id"],
            inventory.result(panel_a_keys["Base Prompting"])["result_id"],
        ],
    )
    occurrences = exact_occurrences(pages, "1.24%")
    if occurrences != [(1, 25), (2, 29), (8, 63)]:
        raise Blocked(f"FIGURE5_RATIO_OCCURRENCES:{occurrences}")
    for page, line in occurrences[:2]:
        inventory.add_alias(
            f"f5-ratio-repeat:{page}:{line}",
            alias_of_key=ratio_key,
            page=page,
            section="Abstract" if page == 1 else "1",
            locator={"evidence_id": f"page:{page}", "line": line, "semantic_target": "PREFINE_over_full_history_ratio"},
            displayed_lexeme="1.24%",
            reason="figure5_abstract_intro_repetition",
        )
    require_text(page_line(pages, 8, 63), "80%", "figure5-reduction-range")
    inventory.add_result(
        "f5-derived:baseline-reduction-range",
        source="narrative_figure5",
        result_kind="derived_value",
        page=8,
        section="7.3",
        locator={"evidence_id": "page:8", "line": 63, "extraction_rule": "explicit_narrative_lower_bound"},
        value=">80%",
        value_type="range",
        method="PREFINE versus baseline memory methods",
        setting="test-time-retrieval",
        metric="retrieved_token_reduction_pct",
        precision=None,
        derived_from=[inventory.result(key)["result_id"] for key in panel_a_keys.values()],
    )
    require_text(page_line(pages, 8, 64), "20–25", "figure5-session-range")
    inventory.add_result(
        "f5-derived:session-token-range",
        source="narrative_figure5",
        result_kind="derived_value",
        page=8,
        section="7.3",
        locator={"evidence_id": "page:8", "line": 64, "extraction_rule": "explicit_narrative_approximate_range"},
        value="approximately 20–25",
        value_type="range",
        method="PREFINE",
        setting="accumulated-sessions",
        metric="memory_tokens_range",
        precision=None,
        derived_from=panel_b_ids,
    )


def parse_table6(pages: list[str], inventory: InventoryBuilder) -> None:
    rows = (
        (14, "multi-session-dialogues", "265", "multi_session_dialogues", 0),
        (15, "sessions", "2,020", "sessions", 0),
        (16, "turns", "39,884", "turns", 0),
        (17, "avg-sessions-dialogue", "7.6", "average_sessions_per_dialogue", 1),
        (18, "avg-turns-session", "19.7", "average_turns_per_session", 1),
        (19, "preference-recall", "332", "preference_recall_instances", 0),
        (20, "preference-induction", "293", "preference_induction_instances", 0),
        (21, "preference-transfer", "472", "preference_transfer_instances", 0),
    )
    for line_number, row, displayed, metric, precision in rows:
        require_text(page_line(pages, 15, line_number), displayed, f"table6:{row}")
        inventory.add_result(
            f"t6:{row}",
            source="table6",
            result_kind="table_cell",
            page=15,
            section="A.3",
            locator={
                "evidence_id": "table:6",
                "table": 6,
                "line": line_number,
                "row": row,
                "column": "count",
                "extraction_rule": "exact_printed_dataset_statistic",
            },
            value=displayed,
            value_type="numeric",
            setting="dataset-statistics",
            metric=metric,
            precision=precision,
        )
    main_aliases = (
        (52, "multi-session-dialogues", "265"),
        (52, "sessions", "2,020"),
        (52, "turns", "39,884"),
        (53, "avg-sessions-dialogue", "7.6"),
        (53, "avg-turns-session", "19.7"),
        (53, "preference-recall", "332"),
        (54, "preference-induction", "293"),
        (54, "preference-transfer", "472"),
    )
    for line_number, row, displayed in main_aliases:
        require_text(page_line(pages, 4, line_number), displayed, f"table6-main:{row}")
        inventory.add_alias(
            f"t6-main:{row}",
            alias_of_key=f"t6:{row}",
            page=4,
            section="4",
            locator={"evidence_id": "page:4", "line": line_number, "semantic_target": row},
            displayed_lexeme=displayed,
            reason="table6_main_text_repetition",
        )


def parse_human_annotation(pages: list[str], inventory: InventoryBuilder) -> None:
    annotations = (
        (
            "annotators",
            16,
            63,
            "19",
            "19",
            "annotator_count",
            0,
            "numeric",
            "human-study-setup",
        ),
        (
            "budget-slot-values",
            17,
            10,
            "27",
            "27",
            "budget_slot_value_count",
            0,
            "numeric",
            "human-study-setup",
        ),
        (
            "api-domains",
            17,
            10,
            "12",
            "12",
            "api_domain_count",
            0,
            "numeric",
            "human-study-setup",
        ),
        (
            "travel-slot-values",
            17,
            11,
            "4",
            "4",
            "travel_slot_value_count",
            0,
            "numeric",
            "human-study-setup",
        ),
        (
            "budget-agreement",
            17,
            15,
            "89.7%",
            "89.7",
            "budget_group_agreement_pct",
            1,
            "numeric",
            "human-study-results",
        ),
        (
            "travel-agreement",
            17,
            15,
            "97.4%",
            "97.4",
            "travel_group_agreement_pct",
            1,
            "numeric",
            "human-study-results",
        ),
        (
            "budget-kappa",
            17,
            16,
            "0.701",
            "0.701",
            "budget_fleiss_kappa",
            3,
            "numeric",
            "human-study-results",
        ),
        (
            "travel-kappa",
            17,
            17,
            "0.880",
            "0.880",
            "travel_fleiss_kappa",
            3,
            "numeric",
            "human-study-results",
        ),
        (
            "travel-name-confirmation",
            17,
            18,
            "19/19",
            "100",
            "travel_group_name_confirmation_pct",
            0,
            "numeric",
            "human-study-results",
        ),
        (
            "budget-name-confirmation",
            17,
            19,
            "16/19 (84%)",
            "84",
            "budget_group_name_confirmation_pct",
            0,
            "numeric",
            "human-study-results",
        ),
    )
    anchors = {
        "annotators": "19",
        "budget-slot-values": "27",
        "api-domains": "12",
        "travel-slot-values": "4",
        "budget-agreement": "89.7%",
        "travel-agreement": "97.4%",
        "budget-kappa": "0.701",
        "travel-kappa": "0.880",
        "travel-name-confirmation": "All19",
        "budget-name-confirmation": "16of19(84%)",
    }
    for key, page, line_number, displayed, numeric, metric, precision, value_type, setting in annotations:
        text = page_line(pages, page, line_number)
        if key == "annotators":
            text += " " + page_line(pages, 16, 64)
        require_text(text, anchors[key], f"human:{key}")
        inventory.add_result(
            f"human:{key}",
            source="human_annotation",
            result_kind="narrative_value",
            page=page,
            section="A.6",
            locator={"evidence_id": f"page:{page}", "line": line_number, "semantic_target": key, "extraction_rule": "exact_human_annotation_statement"},
            value=displayed,
            value_type=value_type,
            setting=setting,
            metric=metric,
            precision=precision,
            numeric_value=numeric,
            source_lexeme=displayed if key in {"travel-name-confirmation", "budget-name-confirmation"} else anchors[key],
        )
    main_aliases = (
        (11, "annotators", "19"),
        (13, "budget-agreement", "89.7%"),
        (14, "travel-agreement", "97.4%"),
    )
    for line_number, key, displayed in main_aliases:
        require_text(page_line(pages, 4, line_number), displayed, f"human-main:{key}")
        inventory.add_alias(
            f"human-main:{key}",
            alias_of_key=f"human:{key}",
            page=4,
            section="4",
            locator={"evidence_id": "page:4", "line": line_number, "semantic_target": key},
            displayed_lexeme=displayed,
            reason="human_main_text_repetition",
        )
    for key, displayed in (("budget-kappa", "0.701"), ("travel-kappa", "0.880")):
        require_text(page_line(pages, 17, 24), displayed, f"human-discussion:{key}")
        inventory.add_alias(
            f"human-discussion:{key}",
            alias_of_key=f"human:{key}",
            page=17,
            section="A.6",
            locator={"evidence_id": "page:17", "line": 24, "semantic_target": key},
            displayed_lexeme=displayed,
            reason="human_discussion_repetition",
        )
    ethics_text = page_line(pages, 10, 16)
    require_text(ethics_text, "19 human annotators", "human-ethics:annotators")
    inventory.add_alias(
        "human-ethics:annotators",
        alias_of_key="human:annotators",
        page=10,
        section="Ethics Statement",
        locator={"evidence_id": "page:10", "line": 16, "semantic_target": "annotators"},
        displayed_lexeme="19",
        reason="human_ethics_repetition",
    )


def counts_by(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(record[field])
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def validate_inventory(structure: list[dict[str, Any]], inventory: InventoryBuilder) -> None:
    structure_kinds = counts_by(structure, "kind")
    if structure_kinds != {"figure": 10, "page": 25, "table": 16} or len(structure) != 51:
        raise Blocked(f"STRUCTURE_MATRIX_MISMATCH:{structure_kinds}:{len(structure)}")
    if any(record["text_sha256"] != EXPECTED_LAYOUT_SHA256 for record in structure):
        raise Blocked("STRUCTURE_LAYOUT_HASH_MISMATCH")
    expected_sources = {
        "figure4": 34,
        "figure5": 45,
        "human_annotation": 10,
        "narrative_figure4": 6,
        "narrative_figure5": 3,
        "table3": 434,
        "table6": 8,
        "table10": 144,
        "table11": 504,
        "table12": 336,
        "table13": 252,
        "table14": 54,
        "table15": 18,
        "table16": 60,
    }
    if dict(sorted(inventory.source_counts.items())) != expected_sources:
        raise Blocked(f"RESULT_SOURCE_MATRIX_MISMATCH:{inventory.source_counts}")
    if len(inventory.results) != 1908:
        raise Blocked(f"RESULT_TOTAL_MISMATCH:{len(inventory.results)}")
    expected_value_types = {
        "derived": 7,
        "not_applicable": 12,
        "numeric": 1821,
        "range": 2,
        "unlabeled_graphical_mark": 66,
    }
    value_type_counts = counts_by(inventory.results, "value_type")
    if value_type_counts != expected_value_types:
        raise Blocked(f"VALUE_TYPE_MATRIX_MISMATCH:{value_type_counts}")
    expected_alias_reasons = {
        "figure4_prose_mark_repetition": 2,
        "figure5_abstract_intro_repetition": 2,
        "figure5_prose_mark_repetition": 1,
        "human_discussion_repetition": 2,
        "human_ethics_repetition": 1,
        "human_main_text_repetition": 3,
        "table14_repeated_gemini_location": 54,
        "table15_prose_repetition": 5,
        "table16_appendix_text_repetition": 4,
        "table16_main_text_repetition": 4,
        "table6_main_text_repetition": 8,
    }
    alias_reason_counts = counts_by(inventory.aliases, "reason")
    if alias_reason_counts != expected_alias_reasons or len(inventory.aliases) != 86:
        raise Blocked(f"ALIAS_MATRIX_MISMATCH:{alias_reason_counts}:{len(inventory.aliases)}")
    required_result_fields = {
        "schema",
        "result_id",
        "result_kind",
        "page",
        "section",
        "locator",
        "method",
        "model",
        "dataset",
        "setting",
        "preference_type",
        "metric",
        "displayed_value",
        "numeric_value",
        "value_type",
        "displayed_marker",
        "source_lexeme",
        "rounding_rule",
        "evidence_id",
    }
    result_ids: set[str] = set()
    locations: dict[tuple[int, str, str], str] = {}
    for record in inventory.results:
        keys = set(record)
        if not required_result_fields <= keys or keys - required_result_fields - {"derived_from"}:
            raise Blocked("RESULT_FIELDS_MISMATCH:{}".format(record.get("result_id")))
        result_id = record["result_id"]
        if result_id in result_ids:
            raise Blocked(f"DUPLICATE_RESULT_ID:{result_id}")
        result_ids.add(result_id)
        location = (record["page"], record["section"], canonical_locator(record))
        if location in locations:
            raise Blocked("DUPLICATE_PAPER_LOCATION:{}:{}".format(locations[location], result_id))
        locations[location] = result_id
    alias_ids: set[str] = set()
    for record in inventory.aliases:
        alias_id = record["alias_id"]
        if alias_id in alias_ids or alias_id in result_ids:
            raise Blocked(f"DUPLICATE_OR_COLLIDING_ALIAS_ID:{alias_id}")
        alias_ids.add(alias_id)
        if record["alias_of"] not in result_ids:
            raise Blocked("ALIAS_TARGET_NOT_RESULT:{}:{}".format(alias_id, record["alias_of"]))
        location = (record["page"], record["section"], canonical_locator(record))
        if location in locations:
            raise Blocked("DUPLICATE_PAPER_LOCATION:{}:{}".format(locations[location], alias_id))
        locations[location] = alias_id
    for record in inventory.results:
        for target in record.get("derived_from", []):
            if record["result_kind"] != "derived_value" or target not in result_ids or target == record["result_id"]:
                raise Blocked("INVALID_DERIVATION:{}:{}".format(record["result_id"], target))
    table_results: dict[int, int] = {}
    table_numeric: dict[int, int] = {}
    for record in inventory.results:
        table = record["locator"].get("table")
        if isinstance(table, int):
            table_results[table] = table_results.get(table, 0) + 1
            if record["value_type"] == "numeric":
                table_numeric[table] = table_numeric.get(table, 0) + 1
    for table, expected in ((3, 434), (10, 144), (11, 504), (12, 336), (13, 252), (15, 18)):
        if table_results.get(table) != expected or table_numeric.get(table) != expected:
            raise Blocked(f"TABLE_MATRIX_MISMATCH:{table}:{table_results.get(table)}:{table_numeric.get(table)}")
    table14_aliases = sum(record["locator"].get("table") == 14 for record in inventory.aliases)
    if table_results.get(14) != 54 or table14_aliases != 54:
        raise Blocked(f"TABLE14_MATRIX_MISMATCH:{table_results.get(14)}:{table14_aliases}")
    table16 = [record for record in inventory.results if record["locator"].get("table") == 16]
    if len(table16) != 60 or sum(record["value_type"] == "numeric" for record in table16) != 48 or sum(record["value_type"] == "not_applicable" for record in table16) != 12:
        raise Blocked("TABLE16_MATRIX_MISMATCH")
    figure4 = [record for record in inventory.results if record["locator"].get("figure") == 4 and record["result_kind"] == "figure_mark"]
    figure4_references = [record for record in figure4 if record["method"] == "Ground Truth"]
    if len(figure4) != 34 or len(figure4_references) != 2:
        raise Blocked(f"FIGURE4_MATRIX_MISMATCH:{len(figure4)}:{len(figure4_references)}")
    figure5 = [record for record in inventory.results if record["locator"].get("figure") == 5 and record["result_kind"] == "figure_mark"]
    figure5_a = [record for record in figure5 if ".f05.a." in record["result_id"]]
    figure5_b = [record for record in figure5 if ".f05.b." in record["result_id"]]
    if len(figure5) != 45 or len(figure5_a) != 5 or len(figure5_b) != 40:
        raise Blocked(f"FIGURE5_MATRIX_MISMATCH:{len(figure5_a)}:{len(figure5_b)}:{len(figure5)}")
    assert_no_forbidden_fields(structure)
    assert_no_forbidden_fields(inventory.results)
    assert_no_forbidden_fields(inventory.aliases)


def build_inventory(pages: list[str]) -> tuple[list[dict[str, Any]], InventoryBuilder]:
    structure = build_structure(pages)
    inventory = InventoryBuilder()
    parse_table3(pages, inventory)
    parse_table10(pages, inventory)
    parse_table11(pages, inventory)
    parse_table12(pages, inventory)
    parse_table13(pages, inventory)
    parse_table14(pages, inventory)
    parse_table15(pages, inventory)
    parse_table16(pages, inventory)
    parse_figure4(pages, inventory)
    parse_figure5(pages, inventory)
    # Figure 6 characterizes dataset distributions and is structure-only, not an inference-output surface.
    parse_table6(pages, inventory)
    parse_human_annotation(pages, inventory)
    inventory.results.sort(key=lambda record: record["result_id"])
    inventory.aliases.sort(key=lambda record: record["alias_id"])
    validate_inventory(structure, inventory)
    return structure, inventory




def build_inventory_from_layout(
    pages: list[str], layout_sha256: str
) -> tuple[list[dict[str, Any]], InventoryBuilder]:
    """Build only from a 25-page immutable captured layout and its digest."""
    if (
        not isinstance(layout_sha256, str)
        or len(layout_sha256) != 64
        or any(character not in "0123456789abcdef" for character in layout_sha256)
    ):
        raise Blocked("LAYOUT_SHA256_INVALID")
    if len(pages) != 25 or any(not isinstance(page, str) or not page.strip() for page in pages):
        raise Blocked("CAPTURED_LAYOUT_PAGE_SET_INVALID")
    global EXPECTED_LAYOUT_SHA256
    previous = EXPECTED_LAYOUT_SHA256
    EXPECTED_LAYOUT_SHA256 = layout_sha256
    try:
        return build_inventory(pages)
    finally:
        EXPECTED_LAYOUT_SHA256 = previous

