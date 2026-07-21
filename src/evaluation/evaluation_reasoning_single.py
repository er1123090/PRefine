#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union, Optional
from collections import defaultdict
import argparse

# =========================================================
# 1. Parsing Logic (Recall=1 구분을 위해 내부적으로 필요함)
# =========================================================

_CALL_RE = re.compile(r"(?:\{?)([A-Za-z_]\w*)(?:\}?)\s*\((.*?)\)")

def _strip_think_tags(s: str) -> str:
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"</think>", "", s, flags=re.IGNORECASE)
    return s.strip()

def _strip_quotes(s: str) -> str:
    s = str(s).strip()
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        return s[1:-1]
    return s

def _process_regex_value(val_str: str) -> str:
    s = val_str.strip()
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        return s[1:-1]
    if s.lower() == "true": return "True"
    if s.lower() == "false": return "False"
    return s

def _split_args(arg_str: str) -> List[str]:
    parts, buf = [], []
    in_quote: Optional[str] = None
    escape = False
    for ch in arg_str:
        if escape: buf.append(ch); escape = False; continue
        if ch == "\\": buf.append(ch); escape = True; continue
        if in_quote:
            buf.append(ch)
            if ch == in_quote: in_quote = None
            continue
        if ch in ("'", '"'):
            in_quote = ch
            buf.append(ch)
            continue
        if ch == ",":
            part = "".join(buf).strip()
            if part: parts.append(part)
            buf = []
        else:
            buf.append(ch)
    tail = "".join(buf).strip()
    if tail: parts.append(tail)
    return parts

def parse_regex_call(call: str) -> List[Tuple[str, str, str]]:
    call = call.strip()
    m = _CALL_RE.fullmatch(call)
    if not m: return []
    domain = m.group(1).strip()
    args_str = m.group(2).strip()
    if not args_str: return []
    out = []
    for part in _split_args(args_str):
        if "=" not in part: continue
        k, v = part.split("=", 1)
        out.append((domain, k.strip(), _process_regex_value(v)))
    return out

def extract_from_json(text: str) -> List[Tuple[str, str, str]]:
    cleaned = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find('{')
        end = cleaned.rfind('}')
        if start != -1 and end != -1:
            try: data = json.loads(cleaned[start:end+1])
            except: return []
        else: return []

    results = []
    items = [data] if isinstance(data, dict) else (data if isinstance(data, list) else [])
    
    for item in items:
        if not isinstance(item, dict): continue
        matched = False
        
        # Strategy 1 & 2 combined
        target = None
        if "function" in item and "name" in item["function"]: target = item["function"]
        elif "name" in item and "parameters" in item: target = item
        
        if target and "parameters" in target:
            domain = target["name"]
            for k, v in target["parameters"].items():
                val_str = str(v) if not isinstance(v, (list, dict)) else json.dumps(v)
                results.append((str(domain), str(k), _strip_quotes(val_str)))
            matched = True
        
        # Strategy 3
        if not matched:
            for k, v in item.items():
                if k in {"reasoning", "raw_content", "model_name", "evaluation_result", "reasoning_tokens"}: continue
                if isinstance(v, dict):
                    for sk, sv in v.items():
                        val_str = str(sv) if not isinstance(sv, (list, dict)) else json.dumps(sv)
                        results.append((str(k), str(sk), _strip_quotes(val_str)))
    return results

def extract_all_slot_values(x: Union[str, List[str], None]) -> List[Tuple[str, str, str]]:
    if x is None: return []
    if isinstance(x, list):
        out = []
        for i in x: out.extend(extract_all_slot_values(i))
        return out
    if isinstance(x, str):
        s = _strip_think_tags(x).strip()
        if not s: return []
        j = extract_from_json(s)
        if j: return j
        calls = [m.group(0) for m in _CALL_RE.finditer(s)]
        if not calls and _CALL_RE.fullmatch(s): calls = [s]
        out = []
        for c in calls: out.extend(parse_regex_call(c))
        return out
    return []

# =========================================================
# 2. Simplified Evaluation & Stats Logic
# =========================================================

@dataclass
class FileTokenStats:
    total_avg: float
    recall_1_avg: float
    recall_less_1_avg: float
    total_count: int
    recall_1_count: int
    recall_less_1_count: int

def calculate_token_stats(json_path: str) -> FileTokenStats:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # JSON 구조 유연하게 처리
    examples = data
    if isinstance(data, dict):
        for k in ["data", "examples", "items"]:
            if k in data and isinstance(data[k], list):
                examples = data[k]
                break
    
    # Aggregators
    sum_r1 = 0.0
    cnt_r1 = 0
    sum_r_less = 0.0
    cnt_r_less = 0

    for ex in examples:
        # 1. Ground Truth & Prediction 파싱
        gt_data = ex.get("reference_ground_truth")
        pred_data = ex.get("llm_output")
        
        gt_tuples = extract_all_slot_values(gt_data)
        pred_tuples = extract_all_slot_values(pred_data)
        
        # 2. GT Map 생성 (Multi-value 지원)
        gt_map = defaultdict(set)
        for d, s, v in gt_tuples:
            gt_map[(d, s)].add(v)
        
        # 3. Pred Map 생성
        pred_map = defaultdict(set)
        for d, s, v in pred_tuples:
            pred_map[(d, s)].add(v)
            
        # 4. False Negative 계산 (Recall 1.0 여부 판단용)
        fn = 0
        for key, allowed_vals in gt_map.items():
            pv = pred_map.get(key, set())
            if not (pv & allowed_vals): # 교집합이 없으면 FN 발생
                fn += 1
        
        # 5. Reasoning Token 가져오기
        try:
            tokens = float(ex.get("reasoning_token_count", 0))
        except (ValueError, TypeError):
            tokens = 0.0
            
        # 6. 그룹별 합산
        if fn == 0:
            sum_r1 += tokens
            cnt_r1 += 1
        else:
            sum_r_less += tokens
            cnt_r_less += 1
            
    # 평균 계산
    total_sum = sum_r1 + sum_r_less
    total_cnt = cnt_r1 + cnt_r_less
    
    avg_total = total_sum / total_cnt if total_cnt > 0 else 0.0
    avg_r1 = sum_r1 / cnt_r1 if cnt_r1 > 0 else 0.0
    avg_r_less = sum_r_less / cnt_r_less if cnt_r_less > 0 else 0.0
    
    return FileTokenStats(avg_total, avg_r1, avg_r_less, total_cnt, cnt_r1, cnt_r_less)

# =========================================================
# 3. Main Execution
# =========================================================

def iter_json_files(root_dir: str) -> List[str]:
    out = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith(".json"):
                out.append(os.path.join(dirpath, fn))
    out.sort()
    return out

def parse_filename_metadata(rel_path: str) -> Dict[str, str]:
    """경로에서 메타데이터 추출 (필요 없으면 비워둬도 무방)"""
    parts = rel_path.split(os.sep)
    # 예: context/type/turns/model/prompt/file.json 구조 가정
    # 구조가 맞지 않으면 파일명만 남김
    keys = ["context", "pref_type", "query_turns", "model_name", "prompt_type", "file_name"]
    meta = {k: "" for k in keys}
    
    if len(parts) >= 6:
        meta["context"] = parts[-6]
        meta["pref_type"] = parts[-5]
        meta["query_turns"] = parts[-4]
        meta["model_name"] = parts[-3]
        meta["prompt_type"] = parts[-2]
    
    meta["file_name"] = os.path.basename(rel_path)
    return meta

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, help="검색할 루트 디렉토리")
    parser.add_argument("--json_path", type=str, help="단일 JSON 파일 경로")
    parser.add_argument("--out_csv", type=str, required=True, help="저장할 CSV 파일 경로")
    args = parser.parse_args()

    # 대상 파일 수집
    if args.root_dir:
        files = iter_json_files(args.root_dir)
    elif args.json_path:
        files = [args.json_path]
    else:
        print("Error: --root_dir 또는 --json_path를 지정하세요.")
        return

    print(f"Total files to process: {len(files)}")
    
    results = []
    for fpath in files:
        try:
            stats = calculate_token_stats(fpath)
            
            # 메타데이터 (경로 기반)
            rel_path = os.path.relpath(fpath, args.root_dir) if args.root_dir else os.path.basename(fpath)
            meta = parse_filename_metadata(rel_path)
            
            row = {
                "rel_path": rel_path,
                "file_name": meta["file_name"],
                "model_name": meta["model_name"], # 경로에서 추출된 모델명
                # --- 핵심 데이터 ---
                "avg_tokens_total": round(stats.total_avg, 2),
                "avg_tokens_recall_1": round(stats.recall_1_avg, 2),
                "avg_tokens_recall_less_1": round(stats.recall_less_1_avg, 2),
                "count_total": stats.total_count,
                "count_recall_1": stats.recall_1_count,
                "count_recall_less_1": stats.recall_less_1_count,
            }
            results.append(row)
            
        except Exception as e:
            print(f"[Error] {fpath}: {e}")
            results.append({
                "rel_path": fpath,
                "file_name": os.path.basename(fpath),
                "error": str(e)
            })

    # CSV 저장
    if results:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        # 컬럼 순서 정의
        fieldnames = [
            "rel_path", "file_name", "model_name",
            "avg_tokens_total", 
            "avg_tokens_recall_1", 
            "avg_tokens_recall_less_1",
            "count_total", "count_recall_1", "count_recall_less_1",
            "error"
        ]
        
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            w.writeheader()
            w.writerows(results)
            
        print(f"Successfully saved stats to: {args.out_csv}")

if __name__ == "__main__":
    main()