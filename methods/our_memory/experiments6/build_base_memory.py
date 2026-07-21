import tqdm
import os
import json
import argparse
import pandas as pd
from openai import AsyncOpenAI  # [변경] 비동기 클라이언트
import re
import asyncio  # [변경] asyncio 모듈 추가

# ---------------------------------------------------------
# [Prompt Definition]
# 사용자가 제공한 원본 프롬프트 (Explicit 관련 내용 없음)
# ---------------------------------------------------------
# 프롬프트 매핑 딕셔너리
from prompt import RECURSIVE_MEMORY_UPDATE_PROMPT_V1
PROMPT_MAP = {
    "v1": RECURSIVE_MEMORY_UPDATE_PROMPT_V1,
}


def check_if_testable(pref_list_path, example, query_map):
    # (이전과 동일한 로직)
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
                    except: args = ""
                else: 
                    domain = call_str.strip(); args = ""
                
                if domain in query_map and domain in pref_list:
                    pattern = r'(\w+)=["\']([^"\']+)["\']'
                    matches = re.findall(pattern, args)
                    target_slots = pref_list.get(domain, [])
                    if any(slot in target_slots for slot, _ in matches):
                        return True
    return False

# ---------------------------------------------------------
# Memory System Class (Async Version)
# ---------------------------------------------------------
class RecursiveMemorySystem:
    def __init__(self, client, prompt_template, model="gpt-4o-mini-2024-07-18"):
        self.client = client
        self.model = model
        self.prompt_template = prompt_template
        self.explicit_pref = "None"
        self.implicit_pref = "None"
        self.accumulated_api_list = []
        self.t = 0

    # [변경] async def로 변경하여 비동기 실행 가능하게 함
    async def f_reason(self, h_t: str, s_t: str):
        self.t += 1
        
        formatted_prompt = self.prompt_template.format(
            prev_implicit=self.implicit_pref,
            h_t=h_t if h_t.strip() else "No dialogue",
            s_t=s_t if s_t.strip() else "No API calls"
        )

        try:
            # [변경] await 키워드 사용
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": formatted_prompt}],
                temperature=0,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content.strip()
            
            try:
                res_json = json.loads(content)
            except json.JSONDecodeError:
                match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
                if match:
                    res_json = json.loads(match.group(1))
                else:
                    match_brace = re.search(r"\{.*\}", content, re.DOTALL)
                    if match_brace:
                        res_json = json.loads(match_brace.group(0))
                    else:
                        raise ValueError("No JSON found")

            self.implicit_pref = res_json.get("implicit_pref", self.implicit_pref)
            
            if s_t.strip():
                self.accumulated_api_list.append(f"[Session {self.t}] {s_t}")
                
        except Exception as e:
            print(f"[Error t={self.t}] {e}")

# ---------------------------------------------------------
# Single Dialogue Processor (Async Unit)
# ---------------------------------------------------------
async def process_single_dialogue(sem, client, row, args, query_map, selected_prompt):
    """
    하나의 Dialogue(Example)를 처리하는 비동기 함수.
    내부의 Session들은 '순차적'으로 실행됩니다 (await를 loop 안에서 사용).
    """
    example = row.to_dict()
    ex_id = str(example.get("example_id"))

    # 1. Pre-check
    if os.path.exists(args.pref_list_path):
        is_useful = check_if_testable(args.pref_list_path, example, query_map)
        if not is_useful:
            return None  # Skip

    # Semaphore를 사용하여 동시 실행 수 제한 (API Rate Limit 방지)
    async with sem:
        memory_sys = RecursiveMemorySystem(client, prompt_template=selected_prompt)
        sessions = example.get("sessions", [])
        session_history_log = []

        # [중요] Dialogue 내부의 Session은 순차적(Sequential)이어야 함
        # for loop 안에서 await를 쓰면 순서대로 실행됨
        for t_idx, session in enumerate(sessions):
            turns = session.get("dialogue", [])
            h_t = "\n".join([f"{t.get('role','User')}: {t.get('message','')}" for t in turns])
            
            raw_apis = session.get("api_call", [])
            s_t = ", ".join(raw_apis) if isinstance(raw_apis, list) else str(raw_apis)
            
            # 여기서 await를 하므로 앞 세션이 끝나야 다음 세션으로 넘어감
            await memory_sys.f_reason(h_t, s_t)
            
            snapshot = {
                "session_idx": t_idx + 1,
                "explicit_pref": memory_sys.explicit_pref,
                "implicit_pref": memory_sys.implicit_pref,
                "accumulated_api_list": list(memory_sys.accumulated_api_list) 
            }
            session_history_log.append(snapshot)

        return {
            "example_id": ex_id,
            "prompt_version": args.prompt_version,
            "final_explicit_pref": memory_sys.explicit_pref,
            "final_implicit_pref": memory_sys.implicit_pref,
            "final_api_list": memory_sys.accumulated_api_list,
            "memory_evolution_history": session_history_log
        }

# ---------------------------------------------------------
# Main Async Generation Logic
# ---------------------------------------------------------
async def run_async_generation(args):
    # 0. Setup
    selected_version = args.prompt_version.lower()
    selected_prompt = PROMPT_MAP.get(selected_version, RECURSIVE_MEMORY_UPDATE_PROMPT_V1)
    
    # [변경] AsyncClient 생성
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Load Data
    if args.input_path.endswith('.jsonl'):
        df = pd.read_json(args.input_path, lines=True)
    else:
        with open(args.input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and "dataset" in data:
                df = pd.DataFrame(data["dataset"])
            else:
                df = pd.DataFrame(data)

    # Load Query Map
    if os.path.exists(args.query_path):
        with open(args.query_path, "r", encoding="utf-8") as f:
            query_map = json.load(f)
    else:
        query_map = {}

    print(f"Loaded {len(df)} dialogues. Starting Async Processing...")
    
    # [변경] 동시 요청 제한 (Rate Limit 고려하여 10~20 정도로 설정)
    sem = asyncio.Semaphore(10) 
    
    # Task 생성
    tasks = []
    for _, row in df.iterrows():
        task = asyncio.create_task(
            process_single_dialogue(sem, client, row, args, query_map, selected_prompt)
        )
        tasks.append(task)

    # Output Init
    os.makedirs(os.path.dirname(args.memory_output_path), exist_ok=True)
    
    generated_count = 0
    skipped_count = 0

    # 결과 실시간 저장 (as_completed)
    with open(args.memory_output_path, "w", encoding="utf-8") as f_out:
        # tqdm으로 진행상황 표시
        for future in tqdm.tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Async Processing"):
            result = await future
            
            if result is None:
                skipped_count += 1
            else:
                generated_count += 1
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                f_out.flush() # 바로바로 저장

    print(f"Generation Summary:")
    print(f" - Generated: {generated_count}")
    print(f" - Skipped: {skipped_count}")
    print(f"Saved to {args.memory_output_path}")

    await client.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="/data/minseo/experiments6/data/1229_dev_6.json", help="Path to input JSON/JSONL dataset")
    parser.add_argument("--memory_output_path", type=str, required=True, help="Path to save .jsonl file")
    parser.add_argument("--prompt_version", type=str, default="v1", choices=["v1"])
    parser.add_argument("--query_path", type=str, default="/data/minseo/experiments6/query_singleturn.json")
    parser.add_argument("--pref_list_path", type=str, default="/data/minseo/experiments6/pref_list.json")
    
    args = parser.parse_args()
    
    # [변경] asyncio.run으로 비동기 함수 실행
    asyncio.run(run_async_generation(args))