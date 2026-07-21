import tqdm
import os
import json
import argparse
import pandas as pd
import math
from openai import OpenAI
import re
import copy

# (Prompt Import 부분은 동일)
from prompt_conf import (
    RECURSIVE_MEMORY_UPDATE_PROMPT_V1, 
    RECURSIVE_MEMORY_UPDATE_PROMPT_V2, 
    RECURSIVE_MEMORY_UPDATE_PROMPT_V3, 
    RECURSIVE_MEMORY_UPDATE_PROMPT_V4,
    RECURSIVE_MEMORY_UPDATE_CONF_PROMPT_V1,
    RECURSIVE_MEMORY_UPDATE_CONF_PROMPT_V2,
    RECURSIVE_MEMORY_UPDATE_CONF_PROMPT_V5
)

PROMPT_MAP = {
    "v1": RECURSIVE_MEMORY_UPDATE_PROMPT_V1,
    "v2": RECURSIVE_MEMORY_UPDATE_PROMPT_V2,
    "v3": RECURSIVE_MEMORY_UPDATE_PROMPT_V3,
    "v4": RECURSIVE_MEMORY_UPDATE_PROMPT_V4,
    "conf-v1": RECURSIVE_MEMORY_UPDATE_CONF_PROMPT_V1,
    "conf-v2": RECURSIVE_MEMORY_UPDATE_CONF_PROMPT_V2,
    "conf-v5": RECURSIVE_MEMORY_UPDATE_CONF_PROMPT_V5
}

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# (check_if_testable 함수는 동일)
def check_if_testable(pref_list_path, example, query_map):
    # ... (기존 코드 유지) ...
    # 편의상 생략, 기존 코드와 동일하게 사용
    return True

# ---------------------------------------------------------
# Memory System Class (Updated for Logprobs)
# ---------------------------------------------------------
class RecursiveMemorySystem:
    def __init__(self, client, prompt_template, model="gpt-4o-mini-2024-07-18"):
        self.client = client
        self.model = model
        self.prompt_template = prompt_template
        self.explicit_pref = [] 
        self.implicit_pref = {} # [변경] 초기값을 빈 딕셔너리나 빈 리스트로 유연하게 처리하겠지만, 구조상 dict가 됨
        self.accumulated_api_list = []
        self.t = 0

    def _format_pref_to_string(self, pref_data):
        """
        메모리 데이터를 프롬프트 주입용 문자열로 변환
        (리스트 형태와 딕셔너리 형태 모두 지원하도록 수정됨)
        """
        if not pref_data:
            return "None"
            
        lines = []
        
        # [Case 1] Dict 형태 (새로운 Implicit Pref 구조: Traits, Principles)
        if isinstance(pref_data, dict):
            for category, items in pref_data.items():
                # 카테고리 헤더 추가 (예: [Behavioral Traits])
                header = category.replace('_', ' ').title()
                lines.append(f"[{header}]")
                
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict):
                            content = item.get("content", "")
                            conf = item.get("confidence")
                            conf_str = f"{conf:.2f}" if isinstance(conf, (float, int)) else "N/A"
                            lines.append(f"  - {content} (Confidence: {conf_str})")
                        else:
                            lines.append(f"  - {item}")
                else:
                    lines.append(f"  - {items}")
            return "\n".join(lines)

        # [Case 2] List 형태 (Explicit Pref 또는 구버전 Implicit Pref)
        elif isinstance(pref_data, list):
            for item in pref_data:
                if isinstance(item, dict):
                    content = item.get("content", "")
                    conf = item.get("confidence")
                    conf_str = f"{conf:.2f}" if isinstance(conf, (float, int)) else "N/A"
                    lines.append(f"- {content} (Confidence: {conf_str})")
                else:
                    lines.append(f"- {item}")
            return "\n".join(lines)
            
        return str(pref_data)

    def _calculate_span_confidence(self, full_text, target_text, token_logprobs):
        # (이 함수는 기존과 동일하게 유지)
        if not target_text or target_text not in full_text:
            return 0.0

        start_idx = full_text.find(target_text)
        end_idx = start_idx + len(target_text)

        relevant_logprobs = []
        current_cursor = 0
        
        for token_data in token_logprobs:
            token_str = token_data.token
            token_len = len(token_str)
            token_start = current_cursor
            token_end = current_cursor + token_len
            
            if max(start_idx, token_start) < min(end_idx, token_end):
                if token_data.logprob is not None:
                    relevant_logprobs.append(token_data.logprob)
            
            current_cursor += token_len
            if current_cursor > end_idx:
                break
        
        if not relevant_logprobs:
            return 0.0
            
        avg_logprob = sum(relevant_logprobs) / len(relevant_logprobs)
        confidence = math.exp(avg_logprob)
        return confidence

    def f_reason(self, h_t: str, s_t: str):
        self.t += 1
        
        prev_explicit_str = self._format_pref_to_string(self.explicit_pref)
        prev_implicit_str = self._format_pref_to_string(self.implicit_pref)

        formatted_prompt = self.prompt_template.format(
            prev_explicit=prev_explicit_str,
            prev_implicit=prev_implicit_str,
            h_t=h_t if h_t.strip() else "No dialogue",
            s_t=s_t if s_t.strip() else "No API calls"
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": formatted_prompt}],
                temperature=0,
                response_format={"type": "json_object"},
                logprobs=True 
            )
            
            full_content = response.choices[0].message.content
            token_logprobs = response.choices[0].logprobs.content
            res_json = json.loads(full_content)
            
            # -------------------------------------------------------
            # 1. Explicit Pref 업데이트 (기존 로직 유지: 리스트)
            # -------------------------------------------------------
            raw_explicit = res_json.get("explicit_pref", [])
            updated_explicit = []
            
            # 방어 코드: 가끔 dict가 아닌 list of strings로 올 때 대비
            if isinstance(raw_explicit, list):
                for item in raw_explicit:
                    if isinstance(item, dict):
                        content = item.get("content", "")
                    else:
                        content = str(item)
                    
                    if content:
                        conf_score = self._calculate_span_confidence(full_content, content, token_logprobs)
                        updated_explicit.append({"content": content, "confidence": conf_score})
            
            self.explicit_pref = updated_explicit

            # -------------------------------------------------------
            # 2. Implicit Pref 업데이트 (수정됨: 딕셔너리 구조 처리)
            # -------------------------------------------------------
            raw_implicit = res_json.get("implicit_pref", {})
            
            # Case A: 새로운 프롬프트 (Dict 반환: traits, principles)
            if isinstance(raw_implicit, dict):
                updated_implicit = {}
                # 우리가 관심 있는 키들만 순회하거나, 모든 키를 순회
                target_keys = ["behavioral_traits", "decision_principles"]
                
                for key, items in raw_implicit.items():
                    # 만약 프롬프트가 다른 키를 뱉더라도 처리하도록 유연하게
                    if not isinstance(items, list): 
                        continue
                        
                    key_updated_list = []
                    for item in items:
                        if isinstance(item, dict):
                            content = item.get("content", "")
                        else:
                            content = str(item)
                            
                        if content:
                            conf_score = self._calculate_span_confidence(full_content, content, token_logprobs)
                            key_updated_list.append({"content": content, "confidence": conf_score})
                    
                    updated_implicit[key] = key_updated_list
                
                self.implicit_pref = updated_implicit

            # Case B: 구버전 프롬프트 (List 반환) - 하위 호환성 유지
            elif isinstance(raw_implicit, list):
                updated_implicit = []
                for item in raw_implicit:
                    content = item.get("content", "") if isinstance(item, dict) else str(item)
                    if content:
                        conf_score = self._calculate_span_confidence(full_content, content, token_logprobs)
                        updated_implicit.append({"content": content, "confidence": conf_score})
                self.implicit_pref = updated_implicit

            if s_t.strip():
                self.accumulated_api_list.append(f"[Session {self.t}] {s_t}")
                
        except Exception as e:
            print(f"[Error t={self.t}] {e}")
            # 에러 발생 시 로그 출력 (기존 데이터 유지)
# ---------------------------------------------------------
# Main Generation Logic (JSONL & Per-Session Snapshot)
# ---------------------------------------------------------
def generate_memory_file(args):
    # 0. Prompt Selection Validation
    selected_version = args.prompt_version.lower()
    if selected_version not in PROMPT_MAP:
        raise ValueError(f"Invalid prompt version: {selected_version}. Choose from {list(PROMPT_MAP.keys())}")
    
    selected_prompt = PROMPT_MAP[selected_version]
    print(f"Using Prompt Version: {selected_version.upper()}")

    # Load Resources
    if args.input_path.endswith('.jsonl'):
        df = pd.read_json(args.input_path, lines=True)
    else:
        with open(args.input_path, "r", encoding="utf-8") as f:
            df = pd.DataFrame(json.load(f))

    with open(args.query_path, "r", encoding="utf-8") as f:
        query_map = json.load(f)

    print(f"Loaded {len(df)} dialogues.")
    print("Checking evaluability and generating memories...")
    
    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(args.memory_output_path), exist_ok=True)

    # 기존 파일이 있다면 초기화
    if os.path.exists(args.memory_output_path):
        os.remove(args.memory_output_path)
    
    generated_count = 0
    skipped_count = 0

    with open(args.memory_output_path, "w", encoding="utf-8") as f_out:
        
        for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc=f"Gen Memory ({selected_version.upper()})"):
            example = row.to_dict()
            ex_id = str(example.get("example_id")) 
            
            # 1. Pre-check
            is_useful = check_if_testable(args.pref_list_path, example, query_map)
            if not is_useful:
                skipped_count += 1
                continue 

            # 2. Generate Memory
            generated_count += 1
            
            # 시스템 초기화
            memory_sys = RecursiveMemorySystem(client, prompt_template=selected_prompt)
            
            history_sessions = example.get("all_standing_instructions", [])
            api_sessions = example.get("api_calls_all", [])
            
            session_history_log = []
            
            for t_idx in range(len(history_sessions)):
                # Dialogue Extract
                turns = history_sessions[t_idx].get("generated_dialogue", [])
                h_t = "\n".join([f"{t.get('role','').capitalize()}: {t.get('message','')}" for t in turns])
                
                # API Extract
                if t_idx < len(api_sessions):
                    raw_apis = api_sessions[t_idx].get("api_call", [])
                    s_t = ", ".join(raw_apis) if isinstance(raw_apis, list) else str(raw_apis)
                else:
                    s_t = ""
                
                # Update Memory (Confidence 포함하여 업데이트됨)
                memory_sys.f_reason(h_t, s_t)
                
                snapshot = {
                    "session_idx": t_idx + 1,
                    # 이제 리스트(구조화된 딕셔너리) 형태 그대로 저장됨
                    "explicit_pref": memory_sys.explicit_pref,
                    "implicit_pref": memory_sys.implicit_pref,
                    "accumulated_api_list": list(memory_sys.accumulated_api_list) 
                }
                session_history_log.append(snapshot)
            
            # 3. Save to JSONL
            final_record = {
                "example_id": ex_id,
                "prompt_version": selected_version,
                "final_explicit_pref": memory_sys.explicit_pref,
                "final_implicit_pref": memory_sys.implicit_pref,
                "final_api_list": memory_sys.accumulated_api_list,
                "memory_evolution_history": session_history_log
            }
            
            f_out.write(json.dumps(final_record, ensure_ascii=False) + "\n")
            f_out.flush()

    print(f"Generation Summary:")
    print(f" - Generated: {generated_count}")
    print(f" - Skipped: {skipped_count}")
    print(f"Saved to {args.memory_output_path} (JSONL format)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments/data/sgd_converted_dev_mapped_grouped_with_pref_with_constraints.json")
    parser.add_argument("--memory_output_path", type=str, required=True, help="Path to save .jsonl file")
    
    # 프롬프트 버전 선택 인자
    parser.add_argument("--prompt_version", type=str, required=True, choices=["v1", "v2", "v3", "v4","conf-v5", "conf-v1", "conf-v2"], help="Select prompt version (v1, v2, v3, v4, conf-v5, conf-v1, conf-v2)")
    
    parser.add_argument("--query_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments/ours_memory/temp_queries.json")
    parser.add_argument("--pref_list_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments/ours_memory/pref_list.json")
    
    args = parser.parse_args()
    generate_memory_file(args)