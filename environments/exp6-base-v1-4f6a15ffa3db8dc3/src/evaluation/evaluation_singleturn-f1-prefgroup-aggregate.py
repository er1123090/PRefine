#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple, Union, Optional
from collections import defaultdict
import argparse

# =========================================================
# Preference Group Definitions (From Reference Code)
# =========================================================

PREF_GROUP_DEFINITIONS = {
  "low_cost": {
    "group_preference": "budget_conscious",
    "rules": [
      { "domain": "GetRestaurants", "slot": "price_range", "value": "cheap" },
      { "domain": "GetRentalCars", "slot": "car_type", "value": "Compact" },
      { "domain": "GetHotels", "slot": "average_star", "value": 1 },
      { "domain": "GetHotels", "slot": "average_star", "value": 2 },
      { "domain": "GetRideSharing", "slot": "shared_ride", "value": True },
      { "domain": "GetTravel", "slot": "free_entry", "value": True },
      { "domain": "GetFlights", "slot": "flight_class", "value": "Economy" }
    ]
  },
  "high_cost": {
    "group_preference": "budget_conscious",
    "rules": [
      { "domain": "GetRestaurants", "slot": "price_range", "value": "pricey" },
      { "domain": "GetRentalCars", "slot": "car_type", "value": "Full-size" },
      { "domain": "GetHotels", "slot": "average_star", "value": 4 },
      { "domain": "GetHotels", "slot": "average_star", "value": 5 }
    ]
  },
  "solo_usage": {
    "group_preference": "solo_travel",
    "rules": [
      { "domain": "GetBuses", "slot": "group_size", "value": 1 },
      { "domain": "GetFlights", "slot": "passengers", "value": 1 },
      { "domain": "GetRideSharing", "slot": "number_of_seats", "value": 1 },
      { "domain": "GetEvents", "slot": "number_of_tickets", "value": 1 },
      { "domain": "GetRestaurants", "slot": "number_of_seats", "value": 1 }
    ]
  }
}

def normalize_val(v: Any) -> str:
    """JSON raw 값을 파서의 문자열 포맷으로 정규화 (Boolean, Int 처리)"""
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, (int, float)):
        return str(v)
    return str(v)

class GroupMatcher:
    def __init__(self, definitions: Dict[str, Any]):
        # (domain, slot, normalized_value) -> list of group_names
        self.rule_map = defaultdict(list)
        self.all_group_names = sorted(list(definitions.keys()))
        
        for g_name, content in definitions.items():
            for rule in content.get("rules", []):
                d = rule["domain"]
                s = rule["slot"]
                v = rule["value"]
                # 파싱된 결과와 매칭하기 위해 값을 문자열로 정규화
                norm_v = normalize_val(v)
                self.rule_map[(d, s, norm_v)].append(g_name)

    def get_groups(self, domain: str, slot: str, value: str) -> List[str]:
        return self.rule_map.get((domain, slot, value), [])

# =========================================================
# Robust Parsing Logic
# =========================================================

_CALL_RE = re.compile(r"([A-Za-z_]\w*)\s*\((.*?)\)")

def _strip_think_tags(s: str) -> str:
    """Removes <think>...</think> blocks."""
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"</think>", "", s, flags=re.IGNORECASE)
    return s.strip()

def _strip_quotes(s: str) -> str:
    """단순 따옴표 제거"""
    s = str(s).strip()
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        return s[1:-1]
    return s

def _process_regex_value(val_str: str) -> str:
    """Regex 파싱 전용 값 처리 함수"""
    s = val_str.strip()
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        return s[1:-1]
    if s.lower() == "true":
        return "True"
    if s.lower() == "false":
        return "False"
    return s

def _split_args(arg_str: str) -> List[str]:
    """Splits arguments properly handling quotes and escapes."""
    parts, buf = [], []
    in_quote: Optional[str] = None
    escape = False
    for ch in arg_str:
        if escape:
            buf.append(ch); escape = False; continue
        if ch == "\\":
            buf.append(ch); escape = True; continue
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

def parse_regex_call(call: str) -> List[Tuple[str, str, str]]:
    """정규표현식 기반 파싱"""
    call = call.strip()
    m = _CALL_RE.fullmatch(call)
    if not m:
        return []
    domain = m.group(1).strip()
    args_str = m.group(2).strip()
    if not args_str:
        return []
    out = []
    for part in _split_args(args_str):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        slot = k.strip()
        val = _process_regex_value(v)
        out.append((domain, slot, val))
    return out

def extract_from_json(text: str) -> List[Tuple[str, str, str]]:
    """다양한 JSON 포맷 파싱 지원"""
    cleaned = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find('{')
        end = cleaned.rfind('}')
        if start != -1 and end != -1:
            try:
                data = json.loads(cleaned[start:end+1])
            except json.JSONDecodeError:
                return []
        else:
            return []

    results = []
    if isinstance(data, dict):
        items = [data]
    elif isinstance(data, list):
        items = data
    else:
        return []

    for item in items:
        if not isinstance(item, dict):
            continue
        
        matched_strategy = False
        # Strategy 1: OpenAI Function Call
        if "function" in item and isinstance(item["function"], dict):
            f = item["function"]
            if "name" in f and "parameters" in f and isinstance(f["parameters"], dict):
                domain = f["name"]
                params = f["parameters"]
                for slot, val in params.items():
                    val_str = str(val) if not isinstance(val, (list, dict)) else json.dumps(val)
                    results.append((str(domain), str(slot), _strip_quotes(val_str)))
                matched_strategy = True
        
        # Strategy 2: Flat Format
        if not matched_strategy:
            if "name" in item and "parameters" in item and isinstance(item["parameters"], dict):
                domain = item["name"]
                params = item["parameters"]
                for slot, val in params.items():
                    val_str = str(val) if not isinstance(val, (list, dict)) else json.dumps(val)
                    results.append((str(domain), str(slot), _strip_quotes(val_str)))
                matched_strategy = True

        # Strategy 3: Key-as-Function
        if not matched_strategy:
            for key, val in item.items():
                if key in {"reasoning", "raw_content", "model_name", "evaluation_result", "reasoning_tokens"}:
                    continue
                if isinstance(val, dict):
                    domain = key
                    params = val
                    for slot, v in params.items():
                        val_str = str(v) if not isinstance(v, (list, dict)) else json.dumps(v)
                        results.append((str(domain), str(slot), _strip_quotes(val_str)))
    return results

def extract_all_slot_values(x: Union[str, List[str], None]) -> List[Tuple[str, str, str]]:
    """Main Parsing Entry Point"""
    if x is None:
        return []
    
    if isinstance(x, list):
        all_res = []
        for item in x:
            if isinstance(item, str):
                res = extract_all_slot_values(item)
                all_res.extend(res)
            elif isinstance(item, dict):
                res = extract_from_json(json.dumps(item))
                all_res.extend(res)
        return all_res

    if isinstance(x, str):
        s = _strip_think_tags(x).strip()
        if not s:
            return []
        
        json_res = extract_from_json(s)
        if json_res:
            return json_res
        
        calls = [m.group(0) for m in _CALL_RE.finditer(s)]
        if not calls and _CALL_RE.fullmatch(s):
            calls = [s]
        
        regex_res = []
        for c in calls:
            regex_res.extend(parse_regex_call(c))
        return regex_res
        
    return []

# =========================================================
# Metric Logic
# =========================================================

@dataclass
class PRF:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int

def prf_from_counts(tp: int, fp: int, fn: int) -> PRF:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return PRF(p, r, f1, tp, fp, fn)

def build_gt_allowed_map(gt_field: Any) -> Dict[Tuple[str, str], Set[str]]:
    allowed = defaultdict(set)
    for d, s, v in extract_all_slot_values(gt_field):
        allowed[(d, s)].add(v)
    return dict(allowed)

def build_pred_map(pred_field: Any) -> Dict[Tuple[str, str], Set[str]]:
    pred = defaultdict(set)
    for d, s, v in extract_all_slot_values(pred_field):
        pred[(d, s)].add(v)
    return dict(pred)

def counts_slot_and_value_or(
    gt_allowed: Dict[Tuple[str, str], Set[str]],
    pred_vals: Dict[Tuple[str, str], Set[str]],
) -> Tuple[int, int, int]:
    tp = fp = fn = 0
    for key, allowed_vals in gt_allowed.items():
        pv = pred_vals.get(key, set())
        if pv and (pv & allowed_vals):
            tp += 1
        else:
            fn += 1
    for key, pv in pred_vals.items():
        if key not in gt_allowed:
            fp += 1
        else:
            allowed_vals = gt_allowed[key]
            if not (pv & allowed_vals):
                fp += 1
    return tp, fp, fn

def micro_f1_slot_and_value_or(
    examples: List[Dict[str, Any]],
    gt_key: str = "reference_ground_truth",
    pred_key: str = "llm_output",
) -> PRF:
    TP = FP = FN = 0
    for ex in examples:
        gt_allowed = build_gt_allowed_map(ex.get(gt_key))
        pred_vals = build_pred_map(ex.get(pred_key))
        tp, fp, fn = counts_slot_and_value_or(gt_allowed, pred_vals)
        TP += tp; FP += fp; FN += fn
    return prf_from_counts(TP, FP, FN)

# =========================================================
# New: Evaluate by Preference Groups
# =========================================================

def evaluate_by_groups(
    examples: List[Dict[str, Any]],
    gt_key: str = "reference_ground_truth",
    pred_key: str = "llm_output",
) -> Dict[str, PRF]:
    matcher = GroupMatcher(PREF_GROUP_DEFINITIONS)
    
    # group_name -> {tp, fp, fn}
    stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    
    for ex in examples:
        gt_allowed = build_gt_allowed_map(ex.get(gt_key))
        pred_vals = build_pred_map(ex.get(pred_key))
        
        # 1. Analyze GT items (Check for TP and FN)
        for (d, s), allowed_set in gt_allowed.items():
            pv = pred_vals.get((d, s), set())
            
            # OR-case 지원: allowed_set 중 하나라도 매칭되면 HIT
            hit_val = None
            intersection = pv & allowed_set
            if intersection:
                hit_val = list(intersection)[0] # 매칭된 값 중 하나 선택
            
            if hit_val:
                # HIT (TP): 매칭된 값이 속한 그룹에 TP 부여
                groups = matcher.get_groups(d, s, hit_val)
                for g in groups:
                    stats[g]["tp"] += 1
            else:
                # MISS (FN): GT에 존재했으나 예측하지 못한 값들이 속한 그룹에 FN 부여
                missed_groups = set()
                for val in allowed_set:
                    gs = matcher.get_groups(d, s, val)
                    missed_groups.update(gs)
                
                for g in missed_groups:
                    stats[g]["fn"] += 1

        # 2. Analyze Pred items (Check for FP)
        for (d, s), p_set in pred_vals.items():
            allowed_set = gt_allowed.get((d, s), set())
            
            for p_val in p_set:
                if p_val in allowed_set:
                    continue # 정답인 경우는 위에서 TP로 처리됨
                
                # 정답이 아닌 예측값 (FP)
                # 이 잘못된 값이 어떤 그룹 규칙에 해당한다면 해당 그룹의 FP 증가
                groups = matcher.get_groups(d, s, p_val)
                for g in groups:
                    stats[g]["fp"] += 1

    # Convert counts to PRF objects
    results = {}
    for g_name in matcher.all_group_names:
        s = stats[g_name]
        results[g_name] = prf_from_counts(s["tp"], s["fp"], s["fn"])
    
    return results

# =========================================================
# Logging helpers
# =========================================================

def format_or_case_block(
    ex: Dict[str, Any],
    gt_allowed: Dict[Tuple[str, str], Set[str]],
    pred_vals: Dict[Tuple[str, str], Set[str]],
    tp: int, fp: int, fn: int
) -> str:
    or_slots = {k: v for k, v in gt_allowed.items() if len(v) >= 2}
    if not or_slots:
        return ""

    ex_id = ex.get("example_id_sub") or ex.get("example_id") or "NA"
    utt = ex.get("user_utterance", "")
    gt_raw = ex.get("reference_ground_truth", None)
    pred_raw = ex.get("llm_output", None)
    m = prf_from_counts(tp, fp, fn)

    lines = []
    lines.append("=" * 80)
    lines.append(f"EXAMPLE: {ex_id}")
    if utt:
        lines.append(f"USER_UTTERANCE: {utt}")
    lines.append(f"GT_RAW: {gt_raw}")
    lines.append(f"PRED_RAW: {pred_raw}")
    lines.append("")
    lines.append("OR SLOTS (same domain+slot, multiple acceptable values):")
    for (d, s), allowed_vals in sorted(or_slots.items(), key=lambda x: (x[0][0], x[0][1])):
        pv = pred_vals.get((d, s), set())
        hit = bool(pv & allowed_vals)
        lines.append(f"- ({d}, {s}) allowed={sorted(list(allowed_vals))}")
        lines.append(f"  pred={sorted(list(pv)) if pv else []}  -> {'HIT' if hit else 'MISS'}")

    lines.append("")
    lines.append(f"COUNTS: TP={tp} FP={fp} FN={fn}")
    lines.append(f"PRF: P={m.precision:.4f} R={m.recall:.4f} F1={m.f1:.4f}")
    lines.append("")
    return "\n".join(lines)

def write_or_case_log(
    examples: List[Dict[str, Any]],
    out_path: str,
    gt_key: str = "reference_ground_truth",
    pred_key: str = "llm_output",
) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        wrote_any = False
        for ex in examples:
            gt_allowed = build_gt_allowed_map(ex.get(gt_key))
            pred_vals = build_pred_map(ex.get(pred_key))
            tp, fp, fn = counts_slot_and_value_or(gt_allowed, pred_vals)
            block = format_or_case_block(ex, gt_allowed, pred_vals, tp, fp, fn)
            if block:
                f.write(block)
                f.write("\n")
                wrote_any = True
        if not wrote_any:
            f.write("No OR cases found.\n")

def is_parsing_failed_pred(ex: Dict[str, Any], pred_key: str = "llm_output") -> Tuple[bool, str]:
    pred_raw = ex.get(pred_key)
    if not pred_raw:
        return True, "empty_raw_output"
    s = str(pred_raw).strip()
    if not s or s.lower() == "none":
        return True, "empty_raw_output"
    pred_slots = extract_all_slot_values(pred_raw)
    if not pred_slots:
        return True, "no_valid_structure_found"
    return False, "ok"

def write_parsing_failures(
    examples: List[Dict[str, Any]],
    out_path: str,
    pred_key: str = "llm_output",
) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        n_fail = 0
        for ex in examples:
            failed, reason = is_parsing_failed_pred(ex, pred_key=pred_key)
            if failed:
                ex_id = ex.get("example_id_sub") or ex.get("example_id") or "NA"
                f.write(f"{ex_id}\t{reason}\n")
                n_fail += 1
        f.write(f"\n# total_failures={n_fail} / total_examples={len(examples)}\n")

# =========================================================
# Execution & JSON Load
# =========================================================

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_examples(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ["data", "examples", "items"]:
            if key in data and isinstance(data[key], list):
                return data[key]
    raise ValueError("Unsupported JSON structure")

def eval_one_file(
    json_path: str,
    or_log_path: Optional[str] = None,
    parsing_fail_path: Optional[str] = None,
) -> Tuple[PRF, Dict[str, PRF]]:
    data = load_json(json_path)
    examples = extract_examples(data)

    # 1. Overall Metrics
    overall = micro_f1_slot_and_value_or(examples)

    # 2. Group Metrics
    group_results = evaluate_by_groups(examples)

    if or_log_path:
        os.makedirs(os.path.dirname(or_log_path), exist_ok=True)
        write_or_case_log(examples, or_log_path)

    if parsing_fail_path:
        os.makedirs(os.path.dirname(parsing_fail_path), exist_ok=True)
        write_parsing_failures(examples, parsing_fail_path)

    return overall, group_results

def iter_json_files(root_dir: str) -> List[str]:
    out = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith(".json"):
                out.append(os.path.join(dirpath, fn))
    out.sort()
    return out

def ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def parse_hparams_from_relpath(rel_path: str) -> Dict[str, str]:
    parts = rel_path.split(os.sep)
    if len(parts) >= 6:
        context, pref_type, query_turns, model_name, prompt_type, file_name = parts[-6:]
    else:
        padded = [""] * (6 - len(parts)) + parts
        context, pref_type, query_turns, model_name, prompt_type, file_name = padded[-6:]
    return {
        "context": context,
        "pref_type": pref_type,
        "query_turns": query_turns,
        "model_name": model_name,
        "prompt_type": prompt_type,
        "file_name": file_name,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default=None, help="Root directory for recursive scan.")
    parser.add_argument("--json_path", type=str, default=None, help="Single file path.")
    parser.add_argument("--out_csv", type=str, required=True, help="Output CSV path.")
    parser.add_argument("--save_or_logs", action="store_true", help="Save OR-case logs.")
    parser.add_argument("--save_parsing_failures", action="store_true", help="Save parsing failure logs.")
    parser.add_argument("--logs_dir", type=str, default=None)

    args = parser.parse_args()

    if args.root_dir:
        targets = iter_json_files(args.root_dir)
        if not targets:
            raise FileNotFoundError(f"No .json files found under {args.root_dir}")
    elif args.json_path:
        targets = [args.json_path]
    else:
        raise ValueError("Provide --root_dir or --json_path")

    out_csv = args.out_csv
    ensure_parent_dir(out_csv)
    logs_dir = args.logs_dir or os.path.join(os.path.dirname(out_csv) or ".", "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Collect all possible group names for CSV header
    all_group_names = sorted(list(PREF_GROUP_DEFINITIONS.keys()))

    rows = []
    for jp in targets:
        if args.root_dir:
            rel = os.path.relpath(jp, args.root_dir)
            h = parse_hparams_from_relpath(rel)
            group_key = os.path.join(h["context"], h["pref_type"], h["query_turns"], h["model_name"], h["prompt_type"])
        else:
            rel = os.path.basename(jp)
            h = {"context": "", "pref_type": "", "query_turns": "", "model_name": "", "prompt_type": "", "file_name": os.path.basename(jp)}
            group_key = ""

        stem = rel.replace(os.sep, "__") 
        or_log_path = os.path.join(logs_dir, f"{stem}.or_cases.txt") if args.save_or_logs else None
        parsing_fail_path = os.path.join(logs_dir, f"{stem}.parsing_failures.txt") if args.save_parsing_failures else None

        try:
            overall, group_metrics = eval_one_file(jp, or_log_path, parsing_fail_path)
            status = "ok"
            err = ""
        except Exception as e:
            overall = PRF(0.0, 0.0, 0.0, 0, 0, 0)
            group_metrics = {g: PRF(0.0, 0.0, 0.0, 0, 0, 0) for g in all_group_names}
            status = "error"
            err = repr(e)

        row = {
            "group_key": group_key,
            "context": h["context"],
            "pref_type": h["pref_type"],
            "query_turns": h["query_turns"],
            "model_name": h["model_name"],
            "prompt_type": h["prompt_type"],
            "file_name": h["file_name"],
            "json_path": jp,
            "rel_path": rel,
            "status": status,
            "error": err,
            "tp": overall.tp,
            "fp": overall.fp,
            "fn": overall.fn,
            "precision": overall.precision,
            "recall": overall.recall,
            "f1": overall.f1,
            "or_log_path": or_log_path or "",
            "parsing_fail_path": parsing_fail_path or "",
        }

        # Add group specific metrics to the row
        for g_name in all_group_names:
            g_prf = group_metrics.get(g_name, PRF(0.0, 0.0, 0.0, 0, 0, 0))
            row[f"group_{g_name}_f1"] = g_prf.f1
            row[f"group_{g_name}_p"] = g_prf.precision
            row[f"group_{g_name}_r"] = g_prf.recall
            row[f"group_{g_name}_tp"] = g_prf.tp
            row[f"group_{g_name}_fp"] = g_prf.fp
            row[f"group_{g_name}_fn"] = g_prf.fn

        rows.append(row)

    # Base fieldnames
    fieldnames = [
        "group_key",
        "context", "pref_type", "query_turns", "model_name", "prompt_type", "file_name",
        "json_path", "rel_path", "status", "error",
        "tp", "fp", "fn", "precision", "recall", "f1",
        "or_log_path", "parsing_fail_path",
    ]
    # Add dynamic group fieldnames
    for g_name in all_group_names:
        fieldnames.extend([
            f"group_{g_name}_f1",
            f"group_{g_name}_p",
            f"group_{g_name}_r",
            f"group_{g_name}_tp",
            f"group_{g_name}_fp",
            f"group_{g_name}_fn"
        ])

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    ok_rows = [r for r in rows if r["status"] == "ok"]
    if ok_rows:
        TP = sum(int(r["tp"]) for r in ok_rows)
        FP = sum(int(r["fp"]) for r in ok_rows)
        FN = sum(int(r["fn"]) for r in ok_rows)
        agg = prf_from_counts(TP, FP, FN)
        print(f"[DONE] files={len(rows)} ok={len(ok_rows)} error={len(rows)-len(ok_rows)}")
        print(f"[AGG Overall] TP={agg.tp} FP={agg.fp} FN={agg.fn} P={agg.precision:.4f} R={agg.recall:.4f} F1={agg.f1:.4f}")
        print(f"[CSV Saved] {out_csv}")
    else:
        print(f"[DONE] files={len(rows)} ok=0 error={len(rows)}")
        print(f"[CSV Saved] {out_csv}")

if __name__ == "__main__":
    main()