import json
import re
import asyncio
import os
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any, Optional
from tqdm import tqdm

# OpenAI Import
from openai import AsyncOpenAI

# Google Import (Optional checking to prevent immediate crash if not installed)
try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

# prompt.py에서 프롬프트 템플릿 임포트
try:
    from prompt_update3 import (
        LATENT_PREF_SYSTEM_PROMPT,
        LATENT_PREF_INITIAL_PROMPT,
        LATENT_PREF_VERIFIER_PROMPT,
        LATENT_PREF_REFINEMENT_PROMPT
    )
except ImportError:
    # prompt_update3가 없을 경우를 대비한 더미 템플릿 (테스트용)
    LATENT_PREF_SYSTEM_PROMPT = "Context: {prev_implicit}\nDialogue: {full_dialogue}\nAPIs: {full_api_calls}"
    LATENT_PREF_INITIAL_PROMPT = "Extract user preferences in JSON."
    LATENT_PREF_REFINEMENT_PROMPT = "Refine this draft: {previous_draft}\nFeedback: {feedback}"
    LATENT_PREF_VERIFIER_PROMPT = "Verify this preference: {candidate_pref}\nDialogue: {full_dialogue}\nAPIs: {full_api_calls}"
    print("[Warning] 'prompt_update3.py' not found. Using dummy prompts.")

# -------------------------------------------------------------------------
# 1. Memory State
# -------------------------------------------------------------------------
@dataclass
class MemoryState:
    implicit_pref: str = "{}" 
    accumulated_dialogue: str = "" 
    accumulated_api_calls: List[str] = field(default_factory=list)
    session_count: int = 0
    evolution_log: List[Dict] = field(default_factory=list)

# -------------------------------------------------------------------------
# 2. Preference Aggregator (Multi-Provider Support)
# -------------------------------------------------------------------------
class PreferenceAggregator:
    # [변경] max_retries를 인자로 받도록 수정
    def __init__(self, model: str, provider: str, api_key: str, max_retries: int = 3, api_base: Optional[str] = None):
        self.model = model
        self.provider = provider.lower()
        self.api_key = api_key
        self.api_base = api_base
        self.max_retries = max_retries  # [변경] 입력받은 값으로 설정

        # Client Setup
        if self.provider == "openai":
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.api_base)
        elif self.provider == "google":
            if not GOOGLE_AVAILABLE:
                raise ImportError("google-generativeai library is required for Google provider.")
            genai.configure(api_key=self.api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    async def _call_llm(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        """
        OpenAI와 Google API 호출을 추상화하여 처리하는 메서드
        """
        # --- 1. OpenAI Logic ---
        if self.provider == "openai":
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature,
                    max_tokens=2048,
                    response_format={"type": "json_object"}
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"[OpenAI Error] {e}")
                return "{}"

        # --- 2. Google Gemini Logic ---
        elif self.provider == "google":
            try:
                # Gemini는 System Instruction을 모델 생성 시 주입
                model = genai.GenerativeModel(
                    model_name=self.model,
                    system_instruction=system_prompt
                )
                
                # JSON 모드 설정 (Gemini 1.5 Pro/Flash 등 지원)
                generation_config = genai.types.GenerationConfig(
                    candidate_count=1,
                    temperature=temperature,
                    max_output_tokens=2048,
                    response_mime_type="application/json"
                )

                response = await model.generate_content_async(
                    user_prompt,
                    generation_config=generation_config
                )
                return response.text
            except Exception as e:
                print(f"[Google Error] {e}")
                return "{}"
        
        return "{}"

    async def update_memory(self, current_state: MemoryState, session_dialogue: str, session_api_calls: List[str]) -> MemoryState:
        session_idx = current_state.session_count + 1
        session_header = f"\n=== Session {session_idx} ===\n"
        full_dialogue = current_state.accumulated_dialogue + session_header + session_dialogue
        
        current_session_apis = [f"[Session {session_idx}] {api}" for api in session_api_calls]
        full_api_list = current_state.accumulated_api_calls + current_session_apis
        full_api_str = "\n".join(full_api_list) if full_api_list else "No API calls recorded."

        candidate_pref = current_state.implicit_pref
        if candidate_pref == "{}": 
            candidate_pref = "None"

        feedback = ""
        updated_log = list(current_state.evolution_log)
        session_attempts_log = []
        
        candidate_pref_str = candidate_pref if isinstance(candidate_pref, str) else json.dumps(candidate_pref)

        # [참고] max_retries가 0이면 루프를 돌지 않아 메모리 업데이트가 일어나지 않음 (Baseline 역할)
        for i in range(self.max_retries):
            # Step A: Generate / Refine
            candidate_pref_json = await self._generate_preference(
                prev_implicit=current_state.implicit_pref,
                full_dialogue=full_dialogue,
                full_api_calls=full_api_str,
                feedback=feedback,
                previous_draft=candidate_pref if i > 0 else None
            )
            
            if not candidate_pref_json:
                continue

            candidate_pref_str = json.dumps(candidate_pref_json, ensure_ascii=False, indent=2)

            # Step B: Verify
            is_valid, new_feedback, verifier_input, verifier_output_json = await self._verify_preference(
                full_dialogue=full_dialogue,
                full_api_calls=full_api_str,
                candidate_pref=candidate_pref_str
            )

            attempt_record = {
                "step": i + 1,
                "draft_preference": candidate_pref_json,
                "is_valid": is_valid,
                "verifier_feedback": new_feedback,
                "verifier_input": verifier_input,
                "verifier_output": verifier_output_json
            }
            session_attempts_log.append(attempt_record)

            if is_valid:
                candidate_pref = candidate_pref_str
                break
            else:
                feedback = new_feedback 
                candidate_pref = candidate_pref_str

        updated_log.append({
            "session_index": session_idx,
            "refinement_process": session_attempts_log,
            "final_preference_at_session": json.loads(candidate_pref) if isinstance(candidate_pref, str) and candidate_pref not in ["None", "{}"] else {}
        })

        updated_state = MemoryState(
            implicit_pref=candidate_pref,
            accumulated_dialogue=full_dialogue,
            accumulated_api_calls=full_api_list,
            session_count=session_idx,
            evolution_log=updated_log
        )
        return updated_state

    async def _generate_preference(self, prev_implicit: str, full_dialogue: str, full_api_calls: str, feedback: str = "", previous_draft: str = None) -> dict:
        safe_prev_implicit = prev_implicit if prev_implicit else "None"
        
        context_prompt = LATENT_PREF_SYSTEM_PROMPT.format(
            prev_implicit=safe_prev_implicit,
            full_dialogue=full_dialogue,
            full_api_calls=full_api_calls
        )
        
        if previous_draft and feedback:
            task_prompt = LATENT_PREF_REFINEMENT_PROMPT.format(
                previous_draft=previous_draft,
                feedback=feedback
            )
        else:
            task_prompt = LATENT_PREF_INITIAL_PROMPT
        
        # 통합된 LLM 호출 메서드 사용 (Temperature 0.4)
        content = await self._call_llm(context_prompt, task_prompt, temperature=0.4)
        return self._parse_json(content)

    async def _verify_preference(self, full_dialogue: str, full_api_calls: str, candidate_pref: str) -> Tuple[bool, str, str, Dict[str, Any]]:
        prompt = LATENT_PREF_VERIFIER_PROMPT.format(
            full_dialogue=full_dialogue,
            full_api_calls=full_api_calls,
            candidate_pref=candidate_pref
        )
        
        system_msg = "You are a Preference Verification Module. Output JSON only."
        
        # 통합된 LLM 호출 메서드 사용 (Verifier는 결정적이어야 하므로 Temp 0.0)
        content = await self._call_llm(system_msg, prompt, temperature=0.0)
        res_json = self._parse_json(content)
        
        if not res_json:
            return False, "Failed to parse verifier output", prompt, {}

        return res_json.get("valid", False), res_json.get("feedback", ""), prompt, res_json

    def _parse_json(self, content: str) -> dict:
        """
        LLM 응답에서 JSON 추출 (Markdown Block 처리 및 Trailing comma 보정)
        """
        if not content:
            return {}
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Markdown 코드 블록 제거
        if "```json" in content:
            content = re.sub(r"```json\s*", "", content)
            content = re.sub(r"\s*```", "", content)
        elif "```" in content:
            content = content.replace("```", "")

        # 가장 바깥쪽 중괄호 찾기
        start_idx = content.find('{')
        end_idx = content.rfind('}')

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = content[start_idx : end_idx + 1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # Trailing comma 제거 시도
                try:
                    cleaned = re.sub(r",\s*([\]}])", r"\1", json_str)
                    return json.loads(cleaned)
                except:
                    pass
        
        return {}

# -------------------------------------------------------------------------
# 3. Main Logic
# -------------------------------------------------------------------------
def format_dialogue(dialogue_list: List[Dict]) -> str:
    formatted = []
    for turn in dialogue_list:
        role = turn.get('role', 'Unknown')
        msg = turn.get('message', '')
        formatted.append(f"{role}: {msg}")
    return "\n".join(formatted)

def load_dataset(filepath: str) -> List[Dict]:
    if not os.path.exists(filepath):
        print(f"[Error] File not found: {filepath}")
        return []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            else:
                return [data]
    except Exception as e:
        print(f"[Error] Failed to load JSON: {e}")
        return []

async def process_single_user(example: Dict, aggregator: PreferenceAggregator, file_lock: asyncio.Lock, 
                              output_file: str, verifier_file: str, refinement_file: str, 
                              semaphore: asyncio.Semaphore, pbar: tqdm):
    async with semaphore:
        example_id = example.get("example_id", "unknown")
        sessions = example.get("sessions", [])
        
        current_state = MemoryState()
        
        for session in sessions:
            dialogue_text = format_dialogue(session.get("dialogue", []))
            api_calls = session.get("api_call", [])
            
            current_state = await aggregator.update_memory(
                current_state, 
                dialogue_text, 
                api_calls
            )
        
        # 1. Main Result Record
        result_record = {
            "example_id": example_id,
            "final_implicit_preference": current_state.implicit_pref,
            "final_accumulated_api_calls": current_state.accumulated_api_calls,
            "total_sessions_processed": current_state.session_count,
            "preference_evolution_history": current_state.evolution_log 
        }

        # 2. Extract Separate Logs
        refinement_logs = []
        verifier_logs = []

        for log_entry in current_state.evolution_log:
            session_idx = log_entry.get("session_index")
            for step_info in log_entry.get("refinement_process", []):
                
                base_meta = {
                    "example_id": example_id,
                    "session_index": session_idx,
                    "step": step_info.get("step")
                }

                # Refinement Log
                r_log = base_meta.copy()
                r_log["draft_preference"] = step_info.get("draft_preference")
                refinement_logs.append(json.dumps(r_log, ensure_ascii=False))

                # Verifier Log
                v_log = base_meta.copy()
                v_log["is_valid"] = step_info.get("is_valid")
                v_log["verifier_input"] = step_info.get("verifier_input")
                v_log["verifier_output"] = step_info.get("verifier_output")
                
                verifier_logs.append(json.dumps(v_log, ensure_ascii=False))

        # 3. Write to Files
        async with file_lock:
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result_record, ensure_ascii=False) + "\n")
            
            if refinement_logs:
                with open(refinement_file, "a", encoding="utf-8") as f:
                    for line in refinement_logs:
                        f.write(line + "\n")
            
            if verifier_logs:
                with open(verifier_file, "a", encoding="utf-8") as f:
                    for line in verifier_logs:
                        f.write(line + "\n")
        
        pbar.update(1)

async def process_dataset_concurrently(args):
    # API Key Handling based on Provider
    api_key = args.api_key
    
    if args.provider == "openai":
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("[Info] No OpenAI API Key provided. Using 'EMPTY' for vLLM compatibility.")
            api_key = "EMPTY"
    elif args.provider == "google":
        if not api_key:
            api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("[Error] Google API Key is missing. Please set GOOGLE_API_KEY env var or pass --api_key.")
            return

    dataset = load_dataset(args.input)
    if not dataset:
        return

    print(f"=== Initializing Client ===")
    print(f"  - Provider: {args.provider}")
    print(f"  - Model: {args.model}")
    print(f"  - Max Retries: {args.max_retries}") # [변경] 로그 출력 추가
    if args.provider == "openai":
        print(f"  - Base URL: {args.api_base}")

    # Aggregator 생성 시 Provider 정보 전달
    try:
        aggregator = PreferenceAggregator(
            model=args.model,
            provider=args.provider,
            api_key=api_key,
            api_base=args.api_base,
            max_retries=args.max_retries # [변경] Argument 전달
        )
    except Exception as e:
        print(f"[Initialization Error] {e}")
        return
    
    semaphore = asyncio.Semaphore(args.concurrency)
    file_lock = asyncio.Lock()

    # Clear output files
    for fpath in [args.output, args.verifier_output, args.refinement_output]:
        with open(fpath, "w", encoding="utf-8") as f:
            pass

    print(f"=== Processing {len(dataset)} examples concurrently (Max: {args.concurrency}) ===")
    
    pbar = tqdm(total=len(dataset), desc="Processing Users", unit="user")

    tasks = []
    for example in dataset:
        task = asyncio.create_task(
            process_single_user(
                example, aggregator, file_lock, 
                args.output, args.verifier_output, args.refinement_output, 
                semaphore, pbar
            )
        )
        tasks.append(task)
    
    await asyncio.gather(*tasks)
    
    pbar.close()

    print(f"\n=== Processing Complete ===")
    print(f"  > Main Result: {args.output}")
    print(f"  > Verifier Log: {args.verifier_output}")
    print(f"  > Refinement Log: {args.refinement_output}")
    
    if args.provider == "openai":
        await aggregator.client.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process latent preferences using vLLM(OpenAI) or Google Gemini.")
    
    parser.add_argument("--input", type=str, default="/data/minseo/experiments4/data/1229_dev_6.json", help="Path to input JSON file")
    
    parser.add_argument("--output", type=str, default="result.jsonl", help="Main result output file")
    parser.add_argument("--verifier_output", type=str, default="verifier_logs.jsonl", help="Verifier details log")
    parser.add_argument("--refinement_output", type=str, default="refinement_logs.jsonl", help="Refinement process log")
    
    # Provider Argument Added
    parser.add_argument("--provider", type=str, default="openai", choices=["openai", "google"], help="Model provider: 'openai' or 'google'")
    
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model name (e.g., gpt-4o, gemini-1.5-pro)")
    parser.add_argument("--api_base", type=str, default=None, help="API Base URL (Only for OpenAI/vLLM)")
    parser.add_argument("--api_key", type=str, default=None, help="API Key (Optional if env var is set)")
    
    parser.add_argument("--concurrency", type=int, default=30, help="Number of concurrent users to process")
    
    # [변경] Max Retries Argument 추가
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum number of verifier loops (0 to N)")
    
    args = parser.parse_args()
    
    asyncio.run(process_dataset_concurrently(args))