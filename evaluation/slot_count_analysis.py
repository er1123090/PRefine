"""
Slot count analysis: compare average GT slot count vs. predicted slot count per output file.

Walks a directory of result JSON files (or a single file) and produces a CSV with:
  - avg_gt_slots   : average number of ground-truth slots per example
  - avg_pred_slots : average number of predicted slots per example
  - diff_avg_slots : pred - gt (positive = over-prediction, negative = under-prediction)

Usage:
    # Single file
    python evaluation/slot_count_analysis.py \
        --json_path outputs/vanilla_llm/.../result.json \
        --out_csv results/slot_count.csv

    # Directory (recursive)
    python evaluation/slot_count_analysis.py \
        --root_dir outputs/ \
        --out_csv results/slot_count.csv
"""

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Parsing helpers (same logic as evaluation metrics)
# ---------------------------------------------------------------------------

_CALL_RE = re.compile(r"(?:\{?)([A-Za-z_]\w*)(?:\}?)\s*\((.*?)\)")


def _strip_think_tags(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL | re.IGNORECASE)
    return re.sub(r"</think>", "", s, flags=re.IGNORECASE).strip()


def _strip_quotes(s: str) -> str:
    s = str(s).strip()
    if len(s) >= 2 and s[0] in ('"', "'") and s[-1] == s[0]:
        return s[1:-1]
    return s


def _process_regex_value(val_str: str) -> str:
    s = val_str.strip()
    if len(s) >= 2 and s[0] in ('"', "'") and s[-1] == s[0]:
        return s[1:-1]
    if s.lower() == "true":
        return "True"
    if s.lower() == "false":
        return "False"
    return s


def _split_args(arg_str: str) -> List[str]:
    parts, buf = [], []
    in_quote: Optional[str] = None
    escape = False
    for ch in arg_str:
        if escape:
            buf.append(ch)
            escape = False
            continue
        if ch == "\\":
            buf.append(ch)
            escape = True
            continue
        if in_quote:
            buf.append(ch)
            if ch == in_quote:
                in_quote = None
            continue
        if ch in ("'", '"'):
            in_quote = ch
            buf.append(ch)
            continue
        if ch == ",":
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf = []
        else:
            buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_regex_call(call: str) -> List[Tuple[str, str, str]]:
    m = _CALL_RE.fullmatch(call.strip())
    if not m:
        return []
    domain = m.group(1).strip()
    out = []
    for part in _split_args(m.group(2).strip()):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        out.append((domain, k.strip(), _process_regex_value(v)))
    return out


def _extract_from_json(text: str) -> List[Tuple[str, str, str]]:
    cleaned = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        s, e = cleaned.find("{"), cleaned.rfind("}")
        if s != -1 and e != -1:
            try:
                data = json.loads(cleaned[s: e + 1])
            except Exception:
                return []
        else:
            return []

    results = []
    items = [data] if isinstance(data, dict) else (data if isinstance(data, list) else [])
    _SKIP = {"reasoning", "raw_content", "model_name", "evaluation_result", "reasoning_tokens"}

    for item in items:
        if not isinstance(item, dict):
            continue
        target = None
        if "function" in item and "name" in item.get("function", {}):
            target = item["function"]
        elif "name" in item and "parameters" in item:
            target = item
        if target and "parameters" in target:
            for k, v in target["parameters"].items():
                val = str(v) if not isinstance(v, (list, dict)) else json.dumps(v)
                results.append((str(target["name"]), str(k), _strip_quotes(val)))
        elif not target:
            for k, v in item.items():
                if k in _SKIP:
                    continue
                if isinstance(v, dict):
                    for sk, sv in v.items():
                        val = str(sv) if not isinstance(sv, (list, dict)) else json.dumps(sv)
                        results.append((str(k), str(sk), _strip_quotes(val)))
    return results


def extract_slot_values(x: Union[str, List[str], None]) -> List[Tuple[str, str, str]]:
    if x is None:
        return []
    if isinstance(x, list):
        out = []
        for i in x:
            out.extend(extract_slot_values(i))
        return out
    if isinstance(x, str):
        s = _strip_think_tags(x).strip()
        if not s:
            return []
        j = _extract_from_json(s)
        if j:
            return j
        calls = [m.group(0) for m in _CALL_RE.finditer(s)]
        if not calls and _CALL_RE.fullmatch(s):
            calls = [s]
        out = []
        for c in calls:
            out.extend(_parse_regex_call(c))
        return out
    return []


# ---------------------------------------------------------------------------
# Slot statistics
# ---------------------------------------------------------------------------

@dataclass
class SlotStats:
    total_examples: int
    avg_gt_slots: float
    avg_pred_slots: float
    diff_avg_slots: float


def calculate_slot_stats(json_path: str) -> SlotStats:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples = data
    if isinstance(data, dict):
        for key in ("data", "examples", "items"):
            if key in data and isinstance(data[key], list):
                examples = data[key]
                break

    if not isinstance(examples, list):
        return SlotStats(0, 0.0, 0.0, 0.0)

    total_gt, total_pred, count = 0, 0, 0
    for ex in examples:
        gt_tuples = extract_slot_values(ex.get("reference_ground_truth"))
        pred_tuples = extract_slot_values(ex.get("llm_output"))
        total_gt += len(gt_tuples)
        total_pred += len(pred_tuples)
        count += 1

    avg_gt = total_gt / count if count else 0.0
    avg_pred = total_pred / count if count else 0.0
    return SlotStats(count, avg_gt, avg_pred, avg_pred - avg_gt)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _iter_json_files(root_dir: str) -> List[str]:
    out = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith(".json"):
                out.append(os.path.join(dirpath, fn))
    return sorted(out)


def _parse_model_name(rel_path: str) -> str:
    parts = rel_path.split(os.sep)
    return parts[-3] if len(parts) >= 3 else "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Slot count analysis for inference outputs.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--root_dir", help="Root directory to search for JSON output files.")
    group.add_argument("--json_path", help="Single JSON output file.")
    parser.add_argument("--out_csv", required=True, help="Path to write the CSV summary.")
    args = parser.parse_args()

    files = _iter_json_files(args.root_dir) if args.root_dir else [args.json_path]
    print(f"Files to process: {len(files)}")

    rows = []
    for fpath in files:
        try:
            stats = calculate_slot_stats(fpath)
            rel = os.path.relpath(fpath, args.root_dir) if args.root_dir else os.path.basename(fpath)
            rows.append({
                "rel_path": rel,
                "file_name": os.path.basename(fpath),
                "model_name": _parse_model_name(rel),
                "total_examples": stats.total_examples,
                "avg_gt_slots": round(stats.avg_gt_slots, 4),
                "avg_pred_slots": round(stats.avg_pred_slots, 4),
                "diff_avg_slots": round(stats.diff_avg_slots, 4),
                "error": "",
            })
        except Exception as e:
            print(f"[Error] {fpath}: {e}")
            rows.append({
                "rel_path": fpath,
                "file_name": os.path.basename(fpath),
                "model_name": "",
                "total_examples": 0,
                "avg_gt_slots": "",
                "avg_pred_slots": "",
                "diff_avg_slots": "",
                "error": str(e),
            })

    if not rows:
        print("No results.")
        return

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    fieldnames = [
        "rel_path", "file_name", "model_name",
        "total_examples", "avg_gt_slots", "avg_pred_slots", "diff_avg_slots", "error",
    ]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"Saved -> {args.out_csv}")


if __name__ == "__main__":
    main()
