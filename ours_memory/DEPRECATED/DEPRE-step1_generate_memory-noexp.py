import tqdm
import os
import json
import argparse
import pandas as pd
from openai import OpenAI
import re
import copy


#####api list rule로 넣는 코드 수정 필요 ########

# ---------------------------------------------------------
# [Prompt Import]
# ---------------------------------------------------------
# prompt.py 파일에 해당 변수들이 정의되어 있어야 합니다.
from prompt import (
    RECURSIVE_MEMORY_UPDATE_PROMPT_V1, 
    RECURSIVE_MEMORY_UPDATE_PROMPT_V2, 
    RECURSIVE_MEMORY_UPDATE_PROMPT_V3, 
    RECURSIVE_MEMORY_UPDATE_PROMPT_V4,
    RECURSIVE_MEMORY_UPDATE_PROMPT_IMPLICIT_ONLY_V4 # [Prompt.py에 추가 확인 필요]
)

# 프롬프트 매핑 딕셔너리 정의
PROMPT_MAP = {
    "v1": RECURSIVE_MEMORY_UPDATE_PROMPT_V1,
    "v2": RECURSIVE_MEMORY_UPDATE_PROMPT_V2,
    "v3": RECURSIVE_MEMORY_UPDATE_PROMPT_V3,
    "v4": RECURSIVE_MEMORY_UPDATE_PROMPT_V4,
    "imp-v4": RECURSIVE_MEMORY_UPDATE_PROMPT_IMPLICIT_ONLY_V4
}

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ---------------------------------------------------------
# Utility: Pre-check Logic
# ---------------------------------------------------------
def check_if_testable(pref_list_path, example, query_map):
    prefs = example.get("api_calls_pref", [])
    if isinstance(prefs, list) and prefs:
        for pref in prefs:
            for ev in pref.get("evidence", []):
                if ev.get("domain") in query_map:
                    return True 

    if os.path.exists(pref_list_path):
        with open(pref_list_path, "r", encoding="utf-8") as f:
            pref_list = json.load(f)
        
        api_calls = example.get("api_calls", [])
        if isinstance(api_calls, list):
            for call_str in api_calls:
                if "(" in call_str:
                    domain = call_str.split("(")[0].strip()
                    try: args = call_str.split("(", 1)[1].rsplit(")", 1)[0]
                    except: continue
                else: domain = call_str.strip(); args = ""
                
                if domain in query_map and domain in pref_list:
                    pattern = r'(\w+)=["\']([^"\']+)["\']'
                    matches = re.findall(pattern, args)
                    target_slots = pref_list.get(domain, [])
                    if any(slot in target_slots for slot, _ in matches):
                        return True
    return False

# ---------------------------------------------------------
# Memory System Class
# ---------------------------------------------------------
class RecursiveMemorySystem:
    def __init__(self, client, prompt_template, prompt_version_name, model="gpt-4o-mini-2024-07-18"):
        self.client = client
        self.model = model
        self.prompt_template = prompt_template
        self.prompt_version_name = prompt_version_name 
        
        self.explicit_pref = "None"
        self.implicit_pref = "None" # implicit 모드일 때 메인 저장소
        
        self.accumulated_api_list = []
        self.t = 0

    def f_reason(self, h_t: str, s_t: str):
        self.t += 1
        
        # [수정됨] imp-v4 로직을 타도록 조건 추가
        if self.prompt_version_name in ["imp-v4"]:
            # 입력: prev_preference (기존의 implicit_pref 변수를 넣음)
            formatted_prompt = self.prompt_template.format(
                prev_preference=self.implicit_pref, 
                h_t=h_t if h_t.strip() else "No dialogue",
                s_t=s_t if s_t.strip() else "No API calls"
            )
        else:
            # 기존 V1~V4 방식 (Explicit/Implicit 분리)
            formatted_prompt = self.prompt_template.format(
                prev_explicit=self.explicit_pref,
                prev_implicit=self.implicit_pref,
                h_t=h_t if h_t.strip() else "No dialogue",
                s_t=s_t if s_t.strip() else "No API calls"
            )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": formatted_prompt}],
                temperature=0,
                response_format={"type": "json_object"}
            )
            res_json = json.loads(response.choices[0].message.content)

            # [수정됨] imp-v4도 단일 preference 키를 파싱하도록 조건 추가
            if self.prompt_version_name in ["imp-v4"]:
                # 'preference' 키의 값을 implicit_pref 변수에 저장
                self.implicit_pref = res_json.get("preference", self.implicit_pref)
            else:
                self.explicit_pref = res_json.get("explicit_pref", self.explicit_pref)
                self.implicit_pref = res_json.get("implicit_pref", self.implicit_pref)
            
            if s_t.strip():
                self.accumulated_api_list.append(f"[Session {self.t}] {s_t}")
        except Exception as e:
            print(f"[Error t={self.t}] {e}")

# ---------------------------------------------------------
# Main Generation Logic
# ---------------------------------------------------------
def generate_memory_file(args):
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
    
    os.makedirs(os.path.dirname(args.memory_output_path), exist_ok=True)
    if os.path.exists(args.memory_output_path):
        os.remove(args.memory_output_path)
    
    generated_count = 0
    skipped_count = 0

    with open(args.memory_output_path, "w", encoding="utf-8") as f_out:
        
        for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc=f"Gen Memory ({selected_version.upper()})"):
            example = row.to_dict()
            ex_id = str(example.get("example_id")) 
            
            is_useful = check_if_testable(args.pref_list_path, example, query_map)
            if not is_useful:
                skipped_count += 1
                continue 

            generated_count += 1
            
            memory_sys = RecursiveMemorySystem(
                client, 
                prompt_template=selected_prompt, 
                prompt_version_name=selected_version
            )
            
            history_sessions = example.get("sessions", [])
            api_sessions = example.get("api_calls_all", [])
            
            session_history_log = []
            
            for t_idx in range(len(history_sessions)):
                turns = history_sessions[t_idx].get("dialogue", [])
                h_t = "\n".join([f"{t.get('role','').capitalize()}: {t.get('message','')}" for t in turns])
                
                if t_idx < len(api_sessions):
                    raw_apis = api_sessions[t_idx].get("api_call", [])
                    s_t = ", ".join(raw_apis) if isinstance(raw_apis, list) else str(raw_apis)
                else:
                    s_t = ""
                
                memory_sys.f_reason(h_t, s_t)
                
                snapshot = {
                    "session_idx": t_idx + 1,
                    "explicit_pref": memory_sys.explicit_pref,
                    "implicit_pref": memory_sys.implicit_pref,
                    "accumulated_api_list": list(memory_sys.accumulated_api_list) 
                }
                session_history_log.append(snapshot)
            
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
    parser.add_argument("--input_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments2/data/dev_3.json")
    parser.add_argument("--memory_output_path", type=str, required=True, help="Path to save .jsonl file")
    
    # [수정됨] choices에 'imp-v4' 추가
    parser.add_argument("--prompt_version", type=str, default="v3", 
                        choices=["v1", "v2", "v3", "v4", "imp-v4"], 
                        help="Select prompt version")
    
    parser.add_argument("--query_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments2/ours_memory/temp_queries.json")
    parser.add_argument("--pref_list_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments2/ours_memory/pref_list.json")
    
    args = parser.parse_args()
    generate_memory_file(args)