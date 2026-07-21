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
# Robust Parsing Logic
# =========================================================

_CALL_RE = re.compile(r"(?:\{?)([A-Za-z_]\w*)(?:\}?)\s*\((.*?)\)")

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
    """
    다양한 JSON 포맷 파싱 지원
    [UPDATED] Flat JSON Structure (domain + sibling keys) 및 Case-insensitive key 처리 추가
    """
    # 1. 마크다운 코드 블록 제거 및 클리닝
    cleaned = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    
    # 2. JSON 파싱 시도
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

        # --- Helper: 대소문자 무시하고 키 찾기 ---
        keys_lower = {k.lower(): k for k in item.keys()}
        
        # Strategy 1: OpenAI Function Call Format
        if "function" in item and isinstance(item["function"], dict):
            f = item["function"]
            # function 내부도 대소문자 체크를 원하면 복잡해지므로, 표준 포맷(name, parameters)만 가정
            if "name" in f and "parameters" in f and isinstance(f["parameters"], dict):
                domain = f["name"]
                params = f["parameters"]
                for slot, val in params.items():
                    val_str = str(val) if not isinstance(val, (list, dict)) else json.dumps(val)
                    results.append((str(domain), str(slot), _strip_quotes(val_str)))
                matched_strategy = True
        
        # Strategy 2 & 2b: Explicit or Flat 'domain'/'name' field
        if not matched_strategy:
            # domain, name, function, tool 등의 키가 있는지 확인
            func_key_candidates = ["domain", "name", "tool", "function"]
            found_func_key = None
            for cand in func_key_candidates:
                if cand in keys_lower:
                    found_func_key = keys_lower[cand]
                    break
            
            if found_func_key:
                domain = item[found_func_key]
                
                # Parameters 찾기 (arguments, args, parameters)
                param_key_candidates = ["parameters", "arguments", "args"]
                found_param_key = None
                for cand in param_key_candidates:
                    if cand in keys_lower:
                        found_param_key = keys_lower[cand]
                        break
                
                # Case 2A: Nested Parameters (e.g. {"domain": "...", "parameters": {...}})
                if found_param_key and isinstance(item[found_param_key], dict):
                    params = item[found_param_key]
                    for slot, val in params.items():
                        val_str = str(val) if not isinstance(val, (list, dict)) else json.dumps(val)
                        results.append((str(domain), str(slot), _strip_quotes(val_str)))
                    matched_strategy = True
                
                # Case 2B: Flat Parameters (e.g. {"domain": "...", "slot": "val"})
                else:
                    # 메타데이터 키 제외하고 나머지를 모두 파라미터로 간주
                    skip_keys = set(func_key_candidates + param_key_candidates + 
                                    ["reasoning", "raw_content", "model_name", "evaluation_result", "reasoning_tokens"])
                    
                    for k, v in item.items():
                        if k.lower() in skip_keys:
                            continue
                        val_str = str(v) if not isinstance(v, (list, dict)) else json.dumps(v)
                        results.append((str(domain), str(k), _strip_quotes(val_str)))
                    
                    # 파라미터가 아예 없는 경우(0-arg function)에도 구조는 파싱된 것으로 간주
                    matched_strategy = True

        # Strategy 3: Key-as-Function Format (e.g. {"GetHotels": {"loc": "Seoul"}})
        if not matched_strategy:
            for key, val in item.items():
                # Skip known metadata keys to avoid false positives
                # UPDATED: domain, name 등도 최상위 키로 올 수 있는 후보에서 제외 (이미 위에서 처리했거나 메타데이터임)
                if key.lower() in ["reasoning", "raw_content", "model_name", "evaluation_result", "reasoning_tokens", 
                                   "domain", "name", "function", "tool", "parameters", "args", "arguments"]:
                    continue
                
                if isinstance(val, dict):
                    domain = key
                    params = val
                    for slot, v in params.items():
                        val_str = str(v) if not isinstance(v, (list, dict)) else json.dumps(v)
                        results.append((str(domain), str(slot), _strip_quotes(val_str)))
    
    return results

def extract_all_slot_values(x: Union[str, List[str], None]) -> List[Tuple[str, str, str]]:
    """
    Main Parsing Entry Point
    """
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
        # 1. Clean <think> tags first
        s = _strip_think_tags(x).strip()
        if not s:
            return []
        
        # 2. Try JSON
        json_res = extract_from_json(s)
        if json_res:
            return json_res
        
        # 3. Try Regex
        calls = [m.group(0) for m in _CALL_RE.finditer(s)]
        if not calls and _CALL_RE.fullmatch(s):
            calls = [s]
        
        regex_res = []
        for c in calls:
            regex_res.extend(parse_regex_call(c))
        
        return regex_res
        
    return []

# =========================================================
# Metric
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
# Logging
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
            f.write("No OR cases found (no (domain,slot) with >=2 acceptable values).\n")

def is_parsing_failed_pred(ex: Dict[str, Any], pred_key: str = "llm_output") -> Tuple[bool, str]:
    pred_raw = ex.get(pred_key)
    
    # 1. Check raw content existence
    if not pred_raw:
        return True, "empty_raw_output"
        
    s = str(pred_raw).strip()
    if not s or s.lower() == "none":
        return True, "empty_raw_output"

    # 2. Check if parsing yields any slots
    pred_slots = extract_all_slot_values(pred_raw)
    
    # If parsing result is empty, but raw output exists, it is a failure
    if not pred_slots:
        # Note: If the function has NO parameters, this will be empty list.
        # This might count as a failure depending on your definition.
        # But if valid JSON was parsed and it just had no params, 
        # extract_from_json returns [] which is correct behavior for no params.
        # To avoid flagging valid 0-param calls as failures, we could update logic,
        # but currently the metric is slot-based, so 0-param calls are invisible.
        return True, "no_valid_structure_found"

    return False, "ok"

def write_parsing_failures(
    examples: List[Dict[str, Any]],
    out_path: str,
    pred_key: str = "llm_output",
) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        n_fail = 0
        f.write("Example_ID\tFailure_Reason\tRaw_LLM_Output\n")
        
        for ex in examples:
            failed, reason = is_parsing_failed_pred(ex, pred_key=pred_key)
            if failed:
                ex_id = ex.get("example_id_sub") or ex.get("example_id") or "NA"
                raw_val = ex.get(pred_key, "")
                safe_raw = json.dumps(str(raw_val), ensure_ascii=False)
                
                f.write(f"{ex_id}\t{reason}\t{safe_raw}\n")
                n_fail += 1
        f.write(f"\n# total_failures={n_fail} / total_examples={len(examples)}\n")

# =========================================================
# Execution & File I/O
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
    raise ValueError("Unsupported JSON structure: expected list or dict with data/examples/items list.")

def eval_one_file(
    json_path: str,
    or_log_path: Optional[str] = None,
    parsing_fail_path: Optional[str] = None,
) -> Tuple[PRF, int]:
    """
    Returns: (PRF metrics object, number of parsing failures)
    """
    data = load_json(json_path)
    examples = extract_examples(data)

    # 1. Calculate Standard Metrics
    overall = micro_f1_slot_and_value_or(examples)

    # 2. Count Parsing Failures
    n_failures = 0
    for ex in examples:
        failed, _ = is_parsing_failed_pred(ex)
        if failed:
            n_failures += 1

    # 3. Optional Logging
    if or_log_path:
        os.makedirs(os.path.dirname(or_log_path), exist_ok=True)
        write_or_case_log(examples, or_log_path)

    if parsing_fail_path:
        os.makedirs(os.path.dirname(parsing_fail_path), exist_ok=True)
        write_parsing_failures(examples, parsing_fail_path)

    return overall, n_failures

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
    parser.add_argument("--root_dir", type=str, default=None,
                        help="Recursively evaluate all .json files under this directory.")
    parser.add_argument("--json_path", type=str, default=None,
                        help="Evaluate a single json file (ignored if --root_dir is set).")

    parser.add_argument("--out_csv", type=str, required=True,
                        help="Output CSV path (per-file metrics).")

    parser.add_argument("--save_or_logs", action="store_true",
                        help="Save OR-case logs per json file.")
    parser.add_argument("--save_parsing_failures", action="store_true",
                        help="Save parsing-failure example_id_sub list per json file.")

    parser.add_argument("--logs_dir", type=str, default=None,
                        help="Directory to place logs. Default: alongside out_csv in <out_csv_dir>/logs")

    args = parser.parse_args()

    if args.root_dir:
        targets = iter_json_files(args.root_dir)
        if not targets:
            raise FileNotFoundError(f"No .json files found under root_dir={args.root_dir}")
    elif args.json_path:
        targets = [args.json_path]
    else:
        raise ValueError("Provide either --root_dir or --json_path")

    out_csv = args.out_csv
    ensure_parent_dir(out_csv)

    logs_dir = args.logs_dir or os.path.join(os.path.dirname(out_csv) or ".", "logs")
    os.makedirs(logs_dir, exist_ok=True)

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
        or_log_path = None
        parsing_fail_path = None
        if args.save_or_logs:
            or_log_path = os.path.join(logs_dir, f"{stem}.or_cases.txt")
        if args.save_parsing_failures:
            parsing_fail_path = os.path.join(logs_dir, f"{stem}.parsing_failures.txt")

        try:
            prf, n_fail_count = eval_one_file(jp, or_log_path=or_log_path, parsing_fail_path=parsing_fail_path)
            status = "ok"
            err = ""
        except Exception as e:
            prf = PRF(0.0, 0.0, 0.0, 0, 0, 0)
            n_fail_count = 0
            status = "error"
            err = repr(e)

        rows.append({
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
            "tp": prf.tp,
            "fp": prf.fp,
            "fn": prf.fn,
            "precision": prf.precision,
            "recall": prf.recall,
            "f1": prf.f1,
            "parsing_fail_count": n_fail_count,
            "or_log_path": or_log_path or "",
            "parsing_fail_path": parsing_fail_path or "",
        })

    fieldnames = [
        "group_key",
        "context", "pref_type", "query_turns", "model_name", "prompt_type", "file_name",
        "json_path", "rel_path", "status", "error",
        "tp", "fp", "fn", "precision", "recall", "f1",
        "parsing_fail_count",
        "or_log_path", "parsing_fail_path",
    ]

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
        TOTAL_FAILS = sum(int(r["parsing_fail_count"]) for r in ok_rows)
        
        overall = prf_from_counts(TP, FP, FN)
        print(f"[DONE] files={len(rows)} ok={len(ok_rows)} error={len(rows)-len(ok_rows)}")
        print(f"[AGG over all files] TP={overall.tp} FP={overall.fp} FN={overall.fn}  "
              f"P={overall.precision:.4f} R={overall.recall:.4f} F1={overall.f1:.4f}")
        print(f"[AGG Parsing Failures] Total Failures={TOTAL_FAILS}")
        print(f"[CSV] {out_csv}")
    else:
        print(f"[DONE] files={len(rows)} ok=0 error={len(rows)}")
        print(f"[CSV] {out_csv}")

if __name__ == "__main__":
    main()