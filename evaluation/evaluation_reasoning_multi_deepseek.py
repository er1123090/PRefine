#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import json
import re
import sys
import dateparser
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union, Optional
from collections import defaultdict
import argparse

# Transformers 라이브러리 로드 (설치되어 있지 않은 경우 처리)
try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

# =========================================================
# 0. Tokenizer Manager (New)
# =========================================================

class TokenizerManager:
    """
    파일 경로에 따라 적절한 토크나이저(Llama/Qwen/DeepSeek)를 로드하고 캐싱하는 클래스
    """
    def __init__(self, llama_path: str, qwen_path: str, deepseek_path: str):
        self.model_paths = {
            "llama": llama_path,
            "qwen": qwen_path,
            "deepseek": deepseek_path
        }
        self.cache = {}

    def get_tokenizer(self, file_path: str):
        """
        파일 경로를 분석하여 적절한 토크나이저를 반환합니다.
        DeepSeek의 경우 Distill 버전에 따라 Llama/Qwen으로 분기 처리합니다.
        """
        if not AutoTokenizer:
            return None

        path_lower = file_path.lower()
        target_key = None
        
        # [Smart Detection Logic]
        # 1. DeepSeek가 포함된 경우 (Distill 모델 구분)
        if "deepseek" in path_lower:
            if "llama" in path_lower:
                # DeepSeek-R1-Distill-Llama -> Llama Tokenizer 사용
                target_key = "llama"
            elif "qwen" in path_lower:
                # DeepSeek-R1-Distill-Qwen -> Qwen Tokenizer 사용
                target_key = "qwen"
            else:
                # 순수 DeepSeek (V2, V3 등) -> DeepSeek Tokenizer 사용
                target_key = "deepseek"
        
        # 2. 그 외 일반적인 경우
        elif "llama" in path_lower:
            target_key = "llama"
        elif "qwen" in path_lower:
            target_key = "qwen"
        
        # 매칭되는 키워드가 없으면 토크나이저 사용 안 함
        if not target_key:
            return None

        # 캐시에 있으면 반환
        if target_key in self.cache:
            return self.cache[target_key]

        # 없으면 로드 시도
        model_id = self.model_paths[target_key]
        print(f"[Info] Detected '{target_key}' type in filename. Loading tokenizer: {model_id} ...")
        
        try:
            # trust_remote_code=True는 Qwen, DeepSeek 등 일부 모델에 필수
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            self.cache[target_key] = tokenizer
            return tokenizer
        except Exception as e:
            print(f"[Warning] Failed to load {target_key} tokenizer from {model_id}.")
            print(f"          Reason: {e}")
            print(f"          (Tip: If this is a gated model like Llama-3, ensure you are logged in via 'huggingface-cli login' or use a non-gated model path.)")
            self.cache[target_key] = None # 실패 기록 (재시도 방지)
            return None

# =========================================================
# 1. Parsing Logic
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

def _normalize_date_time_value(slot: str, value: str) -> str:
    """
    슬롯 이름에 따라 날짜/시간 값을 정규화합니다.
    """
    slot_lower = slot.lower()
    
    # 파싱 설정을 통해 미래 우선, 비엄격 모드 적용
    settings = {
        'PREFER_DATES_FROM': 'future', 
        'STRICT_PARSING': False
    }

    try:
        # 1. 시간 (Time) 처리
        if "time" in slot_lower:
            # 값에 숫자가 없으면(예: "evening", "now") 파싱하지 않고 원본 반환
            if not any(char.isdigit() for char in value):
                return value
                
            dt = dateparser.parse(value, settings=settings)
            if dt:
                return dt.strftime("%H:%M") # 예: 18:15

        # 2. 날짜 (Date) 처리 (date, day 등이 포함된 슬롯)
        elif "date" in slot_lower or "day" in slot_lower:
            dt = dateparser.parse(value, settings=settings)
            if dt:
                # 연도는 제외하고 월-일만 반환
                return dt.strftime("%m-%d") # 예: 03-05
                
    except Exception:
        return value

    return value

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
        
        slot = k.strip()
        val_processed = _process_regex_value(v)
        val_normalized = _normalize_date_time_value(slot, val_processed)
        
        out.append((domain, slot, val_normalized))
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
                
                slot = str(k)
                val_stripped = _strip_quotes(val_str)
                val_normalized = _normalize_date_time_value(slot, val_stripped)
                
                results.append((str(domain), slot, val_normalized))
            matched = True
        
        if not matched:
            for k, v in item.items():
                if k in {"reasoning", "raw_content", "model_name", "evaluation_result", "reasoning_tokens", "reasoning_content"}: continue
                if isinstance(v, dict):
                    for sk, sv in v.items():
                        val_str = str(sv) if not isinstance(sv, (list, dict)) else json.dumps(sv)
                        
                        slot = str(sk)
                        val_stripped = _strip_quotes(val_str)
                        val_normalized = _normalize_date_time_value(slot, val_stripped)
                        
                        results.append((str(k), slot, val_normalized))
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

def calculate_token_stats(json_path: str, tokenizer=None) -> FileTokenStats:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    examples = data
    if isinstance(data, dict):
        for k in ["data", "examples", "items"]:
            if k in data and isinstance(data[k], list):
                examples = data[k]
                break
    
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
        
        # 2. GT Map
        gt_map = defaultdict(set)
        for d, s, v in gt_tuples:
            gt_map[(d, s)].add(v)
        
        # 3. Pred Map
        pred_map = defaultdict(set)
        for d, s, v in pred_tuples:
            pred_map[(d, s)].add(v)
            
        # 4. False Negative
        fn = 0
        for key, allowed_vals in gt_map.items():
            pv = pred_map.get(key, set())
            if not (pv & allowed_vals):
                fn += 1
        
        # 5. Reasoning Token Logic (Updated)
        tokens = 0.0
        
        # A. JSON에 명시된 숫자 카운트 (reasoning_token_count) 우선 확인
        raw_count = ex.get("reasoning_token_count")
        if raw_count is not None:
            try:
                tokens = float(raw_count)
            except (ValueError, TypeError):
                tokens = 0.0

        # B. 카운트가 없으면(0.0), reasoning_tokens 필드 확인
        if tokens == 0.0:
            val_rt = ex.get("reasoning_tokens")
            
            # case 1: 숫자가 들어있는 경우 (문자열 형태 숫자 포함)
            if val_rt is not None:
                try:
                    tokens = float(val_rt)
                except (ValueError, TypeError):
                    # case 2: 텍스트 컨텐츠인 경우 -> 토크나이징 수행
                    if tokenizer and isinstance(val_rt, str):
                        encoded = tokenizer.encode(val_rt, add_special_tokens=False)
                        tokens = float(len(encoded))
            
            # C. 여전히 0이면, reasoning_content 또는 reasoning 필드 확인
            if tokens == 0.0 and tokenizer:
                content = ex.get("reasoning_content") or ex.get("reasoning")
                if content and isinstance(content, str):
                    encoded = tokenizer.encode(content, add_special_tokens=False)
                    tokens = float(len(encoded))
            
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
    parts = rel_path.split(os.sep)
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
    
    # [NEW] 모델별 토크나이저 경로 설정
    parser.add_argument("--llama_path", type=str, default="meta-llama/Llama-3.1-8B", help="Llama 토크나이저 경로 (또는 DeepSeek-Distill-Llama)")
    parser.add_argument("--qwen_path", type=str, default="Qwen/Qwen2.5-Math-7B", help="Qwen 토크나이저 경로")
    parser.add_argument("--deepseek_path", type=str, default="deepseek-ai/deepseek-llm-7b-chat", help="DeepSeek 전용 토크나이저 경로")
    
    args = parser.parse_args()

    # 대상 파일 수집
    if args.root_dir:
        files = iter_json_files(args.root_dir)
    elif args.json_path:
        files = [args.json_path]
    else:
        print("Error: --root_dir 또는 --json_path를 지정하세요.")
        return

    # [NEW] Tokenizer Manager 초기화
    manager = None
    if AutoTokenizer:
        manager = TokenizerManager(args.llama_path, args.qwen_path, args.deepseek_path)
    else:
        print("[Warning] 'transformers' library not found. Token counting via tokenizer will be disabled.")

    print(f"Total files to process: {len(files)}")
    
    results = []
    for fpath in files:
        try:
            # [NEW] 파일명에 맞는 토크나이저 자동 선택
            current_tokenizer = None
            if manager:
                current_tokenizer = manager.get_tokenizer(fpath)

            stats = calculate_token_stats(fpath, tokenizer=current_tokenizer)
            
            rel_path = os.path.relpath(fpath, args.root_dir) if args.root_dir else os.path.basename(fpath)
            meta = parse_filename_metadata(rel_path)
            
            row = {
                "rel_path": rel_path,
                "file_name": meta["file_name"],
                "model_name": meta["model_name"],
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