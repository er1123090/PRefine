import tqdm
import os
import json
import argparse
import pandas as pd
from typing import List, Dict, Any, Tuple
from datetime import datetime
from openai import OpenAI
import re
import copy

# ---------------------------------------------------------
# [Import Prompts]
# prompt.py 파일에서 필요한 템플릿들을 가져옵니다.
# ---------------------------------------------------------
from prompt import (
        RECURSIVE_MEMORY_UPDATE_PROMPT,
        EXPLICIT_ZS_PROMPT_TEMPLATE,
        IMPLICIT_ZS_PROMPT_TEMPLATE,
        IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE # 새로 추가된 프롬프트
    )

# API Client Initialization
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ---------------------------------------------------------
# 1. Recursive Memory System Class
# ---------------------------------------------------------
class RecursiveMemorySystem:
    def __init__(self, client, model="gpt-4o-mini-2024-07-18"):
        self.client = client
        self.model = model
        # t=0 초기 상태
        self.explicit_pref = "None"
        self.implicit_pref = "None"
        self.accumulated_api_list = []  # API Log History (Append only)
        self.t = 0

    def f_reason(self, h_t: str, s_t: str):
        """
        [Reasoning Core]
        Updates (E_t, I_t) using (H_t, S_t) and Previous State.
        """
        self.t += 1
        
        # prompt.py에서 가져온 영어 템플릿에 데이터 주입
        formatted_prompt = RECURSIVE_MEMORY_UPDATE_PROMPT.format(
            prev_explicit=self.explicit_pref,
            prev_implicit=self.implicit_pref,
            h_t=h_t if h_t.strip() else "No dialogue in this session.",
            s_t=s_t if s_t.strip() else "No API calls in this session."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": formatted_prompt}],
                temperature=0,
                response_format={"type": "json_object"} # JSON 모드 강제
            )
            
            content = response.choices[0].message.content
            res_json = json.loads(content)
            
            # 1. State Update (Overwrite)
            self.explicit_pref = res_json.get("explicit_pref", self.explicit_pref)
            self.implicit_pref = res_json.get("implicit_pref", self.implicit_pref)
            
            # 2. List Update (Append)
            if s_t.strip():
                # 나중에 구분하기 쉽도록 세션 번호 태그 추가
                self.accumulated_api_list.append(f"[Session {self.t}] {s_t}")

        except Exception as e:
            print(f"[Error in f_reason at t={self.t}] {e}")
            # 에러 발생 시 업데이트하지 않고 기존 상태 유지

    def get_final_output(self, user_query: str):
        """
        Generates response using the final accumulated memory.
        Uses IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE.
        """
        # 1. 메모리 정보 종합 (retrieved_memories 구성)
        api_history_str = "\n".join(self.accumulated_api_list) if self.accumulated_api_list else "No past API records."
        
        retrieved_memories_block = f"""
        [Explicit Preferences]:
        {self.explicit_pref}

        [Implicit Preferences]:
        {self.implicit_pref}

        [Past API Call History]:
        {api_history_str}
        """

        # 2. 프롬프트 포맷팅
        # dialogue_history는 현재 세션의 문맥을 의미하나, 
        # 이 평가 루프에서는 test_utterance가 단독으로 주어지므로 "Current Turn" 등으로 처리하거나 비워둡니다.
        final_prompt = IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE.format(
            retrieved_memories=retrieved_memories_block,
            dialogue_history="[Start of new turn based on long-term memory]", 
            user_utterance=user_query
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": final_prompt}],
                temperature=0.0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"FINAL_GENERATION_ERROR: {e}"

# ---------------------------------------------------------
# 2. Helper Functions
# ---------------------------------------------------------
def load_dataset(fpath: str) -> pd.DataFrame:
    if fpath.endswith('.jsonl'):
        return pd.read_json(fpath, lines=True)
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            return pd.DataFrame(json.load(f))
    except ValueError:
        # Fallback for JSON Lines if not strictly named .jsonl
        return pd.read_json(fpath, lines=True)

def load_json_map(fpath: str) -> Dict:
    if not os.path.exists(fpath): return {}
    with open(fpath, "r", encoding="utf-8") as f:
        return json.load(f)

def write_log(log_path: str, record: Dict[str, Any]):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# ---------------------------------------------------------
# [UPDATED] assign_user_utterances
# ---------------------------------------------------------
def assign_user_utterances(pref_list_path: str, example: Dict[str, Any], query_map: Dict[str, str], use_rule_imp: bool = False) -> List[Tuple[str, str]]:
    """
    Returns:
        List of (user_utterance, ground_truth_label) tuples.
    """
    results = []

    # 1. --pref_type rule_pref mode (Parse api_calls + Intersection Check)
    if use_rule_imp:
        if not os.path.exists(pref_list_path):
            return []
            
        with open(pref_list_path, "r", encoding="utf-8") as f:
            pref_list = json.load(f)

        api_calls = example.get("api_calls", [])
        if isinstance(api_calls, list):
            for call_str in api_calls:
                # --- [Step 1] Extract Domain & Args ---
                if "(" in call_str:
                    domain = call_str.split("(")[0].strip()
                    try:
                        args_content = call_str.split("(", 1)[1].rsplit(")", 1)[0]
                    except IndexError:
                        continue 
                else:
                    domain = call_str.strip()
                    args_content = ""

                # --- [Step 2] Filter Domain ---
                if domain not in query_map:
                    continue
                if domain not in pref_list:
                    continue

                # --- [Step 3] Parse arguments ---
                pattern = r'(\w+)=["\']([^"\']+)["\']'
                matches = re.findall(pattern, args_content) # List of (slot, value)

                # ------------------------------------------------------------------
                # [Logic Update] 
                # 1. Check Intersection: 해당 API Call이 pref_list의 슬롯을 하나라도 포함하는가?
                # 2. Construct GT: 포함한다면, Ground Truth는 (pref 여부와 상관없이) 모든 슬롯을 다 넣는다.
                # ------------------------------------------------------------------

                # A. 해당 도메인에서 우리가 관심있는(pref_list에 있는) 슬롯 목록
                target_pref_slots = pref_list.get(domain, [])

                # B. 현재 API Call에 있는 슬롯 중 하나라도 target_pref_slots에 포함되는지 확인
                has_target_slot = False
                for slot, _ in matches:
                    if slot in target_pref_slots:
                        has_target_slot = True
                        break
                
                # C. [Filter] 관심있는 슬롯이 하나도 없으면 이 케이스는 건너뜀
                if not has_target_slot:
                    continue

                # D. [Construct GT] 통과했다면, 모든 슬롯을 사용하여 GT 생성
                filtered_slots = []
                for slot, value in matches:
                    # 모든 슬롯 포함 (pref_list에 없더라도)
                    filtered_slots.append(f'{slot}="{value}"')

                if filtered_slots:
                    new_ground_truth = f"{domain}({', '.join(filtered_slots)})"
                    results.append((query_map[domain], new_ground_truth))
        
        return results

    # 2. --pref_type multi_pref_medium mode
    else:
        prefs = example.get("api_calls_pref", [])
        if not isinstance(prefs, list) or not prefs:
            return []

        for pref in prefs:
            evidence_list = pref.get("evidence", [])
            if not isinstance(evidence_list, list):
                continue
            
            for evidence in evidence_list:
                domain = evidence.get("domain")
                
                # Check if domain exists in query_map
                if domain and domain in query_map:
                    # Construct GT: domain(slot="value")
                    if 'api_call' in evidence:
                        ground_truth_str = evidence['api_call']
                    else:
                        ground_truth_str = f'{domain}({evidence["slot"]}="{evidence["value"]}")'
                    
                    results.append((query_map[domain], ground_truth_str))
        
        return results

# ---------------------------------------------------------
# 3. Main Pipeline
# ---------------------------------------------------------
def process_memory_pipeline(args):
    # Load Data
    df = load_dataset(args.input_path)
    query_map = load_json_map(args.query_path)
    
    print(f"Dataset Loaded: {len(df)} examples.")
    print(f"Config: PrefType={args.pref_type}")

    processed_data = []
    
    # Iterate over each user/dialogue example
    for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Processing"):
        original_ex = row.to_dict()
        
        # -------------------------------------------------------------
        # Phase 1: Build Memory from History (Recursive Update)
        # -------------------------------------------------------------
        
        # 1. Initialize Memory System for this user
        memory_sys = RecursiveMemorySystem(client)
        
        # 2. Extract Session Data
        history_sessions = original_ex.get("all_standing_instructions", [])
        api_sessions = original_ex.get("api_calls_all", [])
        
        # 3. Recursive Update Loop (t = 1 to N)
        num_sessions = len(history_sessions)
        
        for t_idx in range(num_sessions):
            # Construct Dialogue History (H_t)
            turns = history_sessions[t_idx].get("generated_dialogue", [])
            h_t_lines = []
            for turn in turns:
                role = turn.get('role', 'unknown').capitalize()
                msg = turn.get('message') or turn.get('content') or ""
                h_t_lines.append(f"{role}: {msg}")
            h_t = "\n".join(h_t_lines)
            
            # Construct API History (S_t)
            if t_idx < len(api_sessions):
                raw_apis = api_sessions[t_idx].get("api_call", [])
                s_t = ", ".join(raw_apis) if isinstance(raw_apis, list) else str(raw_apis)
            else:
                s_t = ""
            
            # --- [CORE] Call f_reason ---
            memory_sys.f_reason(h_t, s_t)

        # -------------------------------------------------------------
        # Phase 2: Evaluation (Query Injection & Response Generation)
        # -------------------------------------------------------------

        # 4. Extract Target Query and Ground Truth
        use_rule_imp = (args.pref_type == "rule_pref")
        eval_pairs = assign_user_utterances(args.pref_list_path, original_ex, query_map, use_rule_imp)
        
        if not eval_pairs:
            continue

        for sub_idx, (utterance, ground_truth) in enumerate(eval_pairs):
            # 5. Generate Final Response using Memory
            # 여기서 memory_sys는 이미 위 반복문을 통해 학습된 상태입니다.
            # IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE를 사용하여 답변 생성
            gpt_output = memory_sys.get_final_output(utterance)
            
            # Prepare Record
            result_record = copy.deepcopy(original_ex)
            result_record["example_id_sub"] = f"{result_record.get('example_id', 'unknown')}_{sub_idx}"
            result_record["test_utterance"] = utterance
            result_record["reference_ground_truth"] = ground_truth
            result_record["gpt_output"] = gpt_output
            
            # Save Memory State for Analysis (최종 메모리 상태 기록)
            result_record["final_memory_state"] = {
                "explicit_pref": memory_sys.explicit_pref,
                "implicit_pref": memory_sys.implicit_pref,
                "api_history_count": len(memory_sys.accumulated_api_list)
            }
            
            processed_data.append(result_record)
            
            # Logging
            write_log(args.log_path, {
                "id": result_record["example_id_sub"],
                "memory_explicit": memory_sys.explicit_pref,
                "memory_implicit": memory_sys.implicit_pref,
                "user_query": utterance,
                "ground_truth": ground_truth,
                "model_output": gpt_output
            })

    # Save Final Output
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=4, ensure_ascii=False)
    
    print(f"Done. Saved to {args.output_path}")

# ---------------------------------------------------------
# 4. Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Path Arguments
    parser.add_argument("--input_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments/data/sgd_converted_dev_mapped_grouped_with_pref_with_constraints.json", help="Path to input dataset (.json or .jsonl)")
    parser.add_argument("--output_path", type=str, default="./output_memory.json")
    parser.add_argument("--log_path", type=str, default="./process_memory.log")
    parser.add_argument("--query_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments/ours_memory/temp_queries.json", help="Path to query mapping json")
    
    # pref_list_path는 required에 경로값이 아닌 boolean이 와야 하므로 default로 변경
    parser.add_argument("--pref_list_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments/ours_memory/pref_list.json", help="Path to preference list json")
    
    # Logic Arguments
    parser.add_argument("--pref_type", type=str, choices=["multi_pref_medium", "rule_pref"], required=True)
    
    args = parser.parse_args()
    
    process_memory_pipeline(args)