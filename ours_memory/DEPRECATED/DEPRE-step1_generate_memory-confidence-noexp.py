import tqdm
import os
import json
import argparse
import pandas as pd
import math
from openai import OpenAI
import re
import copy

# ---------------------------------------------------------
# [Prompt Import]
# ---------------------------------------------------------
# prompt_conf.py에 V7 프롬프트가 추가되어 있어야 합니다.
from prompt_conf import (
    RECURSIVE_MEMORY_UPDATE_PROMPT_V1, 
    RECURSIVE_MEMORY_UPDATE_PROMPT_V2, 
    RECURSIVE_MEMORY_UPDATE_PROMPT_V3, 
    RECURSIVE_MEMORY_UPDATE_PROMPT_V4,
    RECURSIVE_MEMORY_UPDATE_CONF_PROMPT_V1,
    RECURSIVE_MEMORY_UPDATE_CONF_PROMPT_V2,
    RECURSIVE_MEMORY_UPDATE_CONF_PROMPT_V5,
    RECURSIVE_MEMORY_UPDATE_IMPLICIT_RATIONALE_PROMPT_V6,
    RECURSIVE_MEMORY_UPDATE_WITH_TAGS_PROMPT_V7  # [NEW] V7 추가
)

PROMPT_MAP = {
    "v1": RECURSIVE_MEMORY_UPDATE_PROMPT_V1,
    "v2": RECURSIVE_MEMORY_UPDATE_PROMPT_V2,
    "v3": RECURSIVE_MEMORY_UPDATE_PROMPT_V3,
    "v4": RECURSIVE_MEMORY_UPDATE_PROMPT_V4,
    "conf-v1": RECURSIVE_MEMORY_UPDATE_CONF_PROMPT_V1,
    "conf-v2": RECURSIVE_MEMORY_UPDATE_CONF_PROMPT_V2,
    "conf-v5": RECURSIVE_MEMORY_UPDATE_CONF_PROMPT_V5,
    "conf-v6": RECURSIVE_MEMORY_UPDATE_IMPLICIT_RATIONALE_PROMPT_V6,
    "conf-v7": RECURSIVE_MEMORY_UPDATE_WITH_TAGS_PROMPT_V7 # [NEW] Mapping 추가
}

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# (check_if_testable 함수는 동일)
def check_if_testable(pref_list_path, example, query_map):
    # ... (기존 코드와 동일하여 생략) ...
    return True

# ---------------------------------------------------------
# Memory System Class (Updated for Tags, Rationale & Logprobs)
# ---------------------------------------------------------
class RecursiveMemorySystem:
    def __init__(self, client, prompt_template, model="gpt-4o-mini-2024-07-18"):
        self.client = client
        self.model = model
        self.prompt_template = prompt_template
        self.explicit_pref = [] 
        self.implicit_pref = {} 
        self.accumulated_api_list = []
        self.t = 0

    def _format_pref_to_string(self, pref_data):
        """
        메모리 데이터를 프롬프트 주입용 문자열로 변환
        NOTE: Tags와 Rationale은 입력 프롬프트에는 포함하지 않고(토큰 절약),
        Content와 Confidence 위주로 구성합니다.
        """
        if not pref_data:
            return "None"
            
        lines = []
        
        # [Case 1] Dict 형태 (Implicit Pref 구조)
        if isinstance(pref_data, dict):
            for category, items in pref_data.items():
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

        # [Case 2] List 형태 (Explicit Pref)
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
        # (기존 로직 동일)
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
        
        # Explicit Pref 제외 (V6, V7 기준)
        prev_implicit_str = self._format_pref_to_string(self.implicit_pref)

        formatted_prompt = self.prompt_template.format(
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
            # 1. Implicit Pref 업데이트 (Rationale + Tags 파싱)
            # -------------------------------------------------------
            raw_implicit = res_json.get("implicit_pref", {})
            
            if isinstance(raw_implicit, dict):
                updated_implicit = {}
                # 타겟 키 (Traits, Principles 등)
                
                for key, items in raw_implicit.items():
                    if not isinstance(items, list): 
                        continue
                        
                    key_updated_list = []
                    for item in items:
                        if isinstance(item, dict):
                            content = item.get("content", "")
                            rationale = item.get("rationale", None)
                            
                            # [NEW] Tag 파싱 (V7 이상에서 동작, 없으면 None 또는 기본값)
                            # 프롬프트가 Tags를 뱉지 않는 버전(V6 등)과의 호환성을 위해 get 사용
                            tags = item.get("tags", None) 
                        else:
                            content = str(item)
                            rationale = None
                            tags = None
                            
                        if content:
                            conf_score = self._calculate_span_confidence(full_content, content, token_logprobs)
                            
                            # [NEW] 구조에 tags 필드 추가
                            key_updated_list.append({
                                "content": content, 
                                "confidence": conf_score,
                                "rationale": rationale,
                                "tags": tags
                            })
                    
                    updated_implicit[key] = key_updated_list
                
                self.implicit_pref = updated_implicit

            # -------------------------------------------------------
            # 2. API History 누적
            # -------------------------------------------------------
            if s_t.strip():
                self.accumulated_api_list.append(f"[Session {self.t}] {s_t}")
                
        except Exception as e:
            print(f"[Error t={self.t}] {e}")

# ---------------------------------------------------------
# Main Generation Logic (동일)
# ---------------------------------------------------------
def generate_memory_file(args):
    # Prompt Version Check
    selected_version = args.prompt_version.lower()
    if selected_version not in PROMPT_MAP:
        raise ValueError(f"Invalid prompt version: {selected_version}. Choose from {list(PROMPT_MAP.keys())}")
    
    selected_prompt = PROMPT_MAP[selected_version]
    print(f"Using Prompt Version: {selected_version.upper()}")

    # Load Data
    if args.input_path.endswith('.jsonl'):
        df = pd.read_json(args.input_path, lines=True)
    else:
        with open(args.input_path, "r", encoding="utf-8") as f:
            df = pd.DataFrame(json.load(f))

    with open(args.query_path, "r", encoding="utf-8") as f:
        query_map = json.load(f)

    print(f"Loaded {len(df)} dialogues.")
    
    # Init Output File
    os.makedirs(os.path.dirname(args.memory_output_path), exist_ok=True)
    if os.path.exists(args.memory_output_path):
        os.remove(args.memory_output_path)
    
    generated_count = 0
    skipped_count = 0

    with open(args.memory_output_path, "w", encoding="utf-8") as f_out:
        
        for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc=f"Gen Memory ({selected_version.upper()})"):
            example = row.to_dict()
            ex_id = str(example.get("example_id")) 
            
            # Check Testability
            is_useful = check_if_testable(args.pref_list_path, example, query_map)
            if not is_useful:
                skipped_count += 1
                continue 

            generated_count += 1
            
            # Init System
            memory_sys = RecursiveMemorySystem(client, prompt_template=selected_prompt)
            
            history_sessions = example.get("all_standing_instructions", [])
            api_sessions = example.get("api_calls_all", [])
            
            session_history_log = []
            
            # Session Loop
            for t_idx in range(len(history_sessions)):
                turns = history_sessions[t_idx].get("generated_dialogue", [])
                h_t = "\n".join([f"{t.get('role','').capitalize()}: {t.get('message','')}" for t in turns])
                
                if t_idx < len(api_sessions):
                    raw_apis = api_sessions[t_idx].get("api_call", [])
                    s_t = ", ".join(raw_apis) if isinstance(raw_apis, list) else str(raw_apis)
                else:
                    s_t = ""
                
                # Update Memory (Logic inside handles tags now)
                memory_sys.f_reason(h_t, s_t)
                
                # Snapshot
                snapshot = {
                    "session_idx": t_idx + 1,
                    "explicit_pref": memory_sys.explicit_pref,
                    "implicit_pref": memory_sys.implicit_pref, # Includes 'tags' and 'rationale'
                    "accumulated_api_list": list(memory_sys.accumulated_api_list) 
                }
                session_history_log.append(snapshot)
            
            # Final Record
            final_record = {
                "example_id": ex_id,
                "prompt_version": selected_version,
                "final_explicit_pref": memory_sys.explicit_pref,
                "final_implicit_pref": memory_sys.implicit_pref, # Includes 'tags' and 'rationale'
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
    
    # [NEW] conf-v7 추가
    parser.add_argument("--prompt_version", type=str, required=True, 
                        choices=["v1", "v2", "v3", "v4","conf-v5", "conf-v1", "conf-v2", "conf-v6", "conf-v7"], 
                        help="Select prompt version including 'conf-v7' for tags")
    
    parser.add_argument("--query_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments/ours_memory/temp_queries.json")
    parser.add_argument("--pref_list_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments/ours_memory/pref_list.json")
    
    args = parser.parse_args()
    generate_memory_file(args)