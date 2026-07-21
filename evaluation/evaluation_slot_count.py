#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union, Optional
import argparse

# =========================================================
# 1. Parsing Logic (기존 로직 그대로 재사용)
# =========================================================

_CALL_RE = re.compile(r"(?:\{?)([A-Za-z_]\w*)(?:\}?)\s*\((.*?)\)")

def _strip_think_tags(s: str) -> str:
    if not isinstance(s, str): return ""
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
        
        target = None
        if "function" in item and "name" in item["function"]: target = item["function"]
        elif "name" in item and "parameters" in item: target = item
        
        if target and "parameters" in target:
            domain = target["name"]
            for k, v in target["parameters"].items():
                val_str = str(v) if not isinstance(v, (list, dict)) else json.dumps(v)
                results.append((str(domain), str(k), _strip_quotes(val_str)))
            matched = True
        
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
# 2. Slot Counting Logic
# =========================================================

@dataclass
class SlotStats:
    total_examples: int
    avg_gt_slots: float
    avg_pred_slots: float
    # 분석 편의를 위해 추가: GT와 Pred의 총합
    sum_gt_slots: int
    sum_pred_slots: int

def calculate_slot_stats(json_path: str) -> SlotStats:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # JSON 구조 유연하게 처리 (list, dict keys...)
    examples = data
    if isinstance(data, dict):
        for k in ["data", "examples", "items"]:
            if k in data and isinstance(data[k], list):
                examples = data[k]
                break
    
    if not isinstance(examples, list):
        # 데이터가 리스트가 아니면 빈 통계 반환
        return SlotStats(0, 0.0, 0.0, 0, 0)

    total_gt_slots = 0
    total_pred_slots = 0
    count = 0

    for ex in examples:
        # 1. Ground Truth & Prediction 원문 가져오기
        gt_raw = ex.get("reference_ground_truth")
        pred_raw = ex.get("llm_output")
        
        # 2. 파싱하여 튜플 리스트로 변환 [(domain, slot, value), ...]
        gt_tuples = extract_all_slot_values(gt_raw)
        pred_tuples = extract_all_slot_values(pred_raw)
        
        # 3. 개수 합산
        total_gt_slots += len(gt_tuples)
        total_pred_slots += len(pred_tuples)
        count += 1
            
    # 평균 계산
    avg_gt = total_gt_slots / count if count > 0 else 0.0
    avg_pred = total_pred_slots / count if count > 0 else 0.0
    
    return SlotStats(count, avg_gt, avg_pred, total_gt_slots, total_pred_slots)

# =========================================================
# 3. Main Execution (File Walking & CSV Writing)
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
    """경로에서 메타데이터 추출 (필요 시 수정 가능)"""
    parts = rel_path.split(os.sep)
    # 예: context/type/turns/model/prompt/file.json 등의 구조를 가정하고 추출 시도
    # 구조가 맞지 않아도 에러 없이 처리
    meta = {"model_name": ""}
    
    # 예시: 뒤에서 3번째 폴더명이 모델명인 경우
    if len(parts) >= 3:
        meta["model_name"] = parts[-3]
    else:
        meta["model_name"] = "unknown"
        
    return meta

def main():
    parser = argparse.ArgumentParser(description="Calculate average slot counts for GT and LLM output.")
    parser.add_argument("--root_dir", type=str, help="Root directory containing JSON files to search.")
    parser.add_argument("--json_path", type=str, help="Specific single JSON file path.")
    parser.add_argument("--out_csv", type=str, required=True, help="Path to save the output CSV.")
    args = parser.parse_args()

    # 대상 파일 수집
    if args.root_dir:
        files = iter_json_files(args.root_dir)
    elif args.json_path:
        files = [args.json_path]
    else:
        print("Error: Please provide either --root_dir or --json_path.")
        return

    print(f"Total files to process: {len(files)}")
    
    results = []
    for fpath in files:
        try:
            stats = calculate_slot_stats(fpath)
            
            # 메타데이터 및 경로 정보
            rel_path = os.path.relpath(fpath, args.root_dir) if args.root_dir else os.path.basename(fpath)
            meta = parse_filename_metadata(rel_path)
            
            row = {
                "rel_path": rel_path,
                "file_name": os.path.basename(fpath),
                "model_name": meta.get("model_name", ""),
                
                # --- 요청한 데이터 ---
                "total_examples": stats.total_examples,
                "avg_gt_slots": round(stats.avg_gt_slots, 4),    # Ground Truth 평균 슬롯 수
                "avg_pred_slots": round(stats.avg_pred_slots, 4), # LLM Output 평균 슬롯 수
                "diff_avg_slots": round(stats.avg_pred_slots - stats.avg_gt_slots, 4) # (Pred - GT) 차이
            }
            results.append(row)
            
        except Exception as e:
            print(f"[Error processing] {fpath}: {e}")
            results.append({
                "rel_path": fpath,
                "file_name": os.path.basename(fpath),
                "error": str(e)
            })

    # CSV 저장
    if results:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        # 컬럼 순서
        fieldnames = [
            "rel_path", "file_name", "model_name",
            "total_examples", 
            "avg_gt_slots", 
            "avg_pred_slots", 
            "diff_avg_slots", # 양수면 LLM이 더 많이 생성, 음수면 더 적게 생성
            "error"
        ]
        
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            w.writeheader()
            w.writerows(results)
            
        print(f"Successfully saved slot stats to: {args.out_csv}")

if __name__ == "__main__":
    main()