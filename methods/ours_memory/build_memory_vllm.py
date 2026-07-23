import json
import re
import asyncio
import os
import argparse
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any
from openai import AsyncOpenAI
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.construction_usage import (
    begin_usage_collection,
    end_usage_collection,
    record_response_usage,
    set_usage_session,
)
from src.token_measurement import count_json_tokens, encoding_metadata

# prompt.py에서 프롬프트 템플릿 임포트
try:
    from prompt_update3 import (
        LATENT_PREF_SYSTEM_PROMPT,
        LATENT_PREF_INITIAL_PROMPT,
        LATENT_PREF_VERIFIER_PROMPT,
        LATENT_PREF_REFINEMENT_PROMPT
    )
except ImportError:
    print("[Error] 'prompt.py' file not found. Please ensure it is in the same directory.")
    exit(1)

# -------------------------------------------------------------------------
# [NEW] Helper: DeepSeek Reasoning Parser
# -------------------------------------------------------------------------
def parse_reasoning(raw_content: str) -> Dict[str, str]:
    """
    Parses DeepSeek style <think> tags to separate reasoning from final output.
    """
    if not raw_content:
        return {
            "llm_output": "",
            "reasoning_content": ""
        }

    # DeepSeek <think> Token Parsing Logic
    end_tag = "</think>"
    end_idx = raw_content.rfind(end_tag)

    if end_idx != -1:
        # </think> exists: split reasoning and content
        reasoning_part = raw_content[:end_idx]
        # Remove start tag if present (it should be at the start)
        reasoning_content = reasoning_part.replace("<think>", "").strip()
        # Extract content after end tag
        clean_content = raw_content[end_idx + len(end_tag):].strip()
    else:
        # No tag: treat everything as output
        reasoning_content = ""
        clean_content = raw_content.strip()

    return {
        "llm_output": clean_content,
        "reasoning_content": reasoning_content
    }

# -------------------------------------------------------------------------
# 1. Memory State
# -------------------------------------------------------------------------
@dataclass
class MemoryState:
    # implicit_pref는 이제 단순 문자열이 아니라 JSON 구조를 가진 문자열(혹은 dict)이 됩니다.
    implicit_pref: str = "{}" 
    accumulated_dialogue: str = "" 
    accumulated_api_calls: List[str] = field(default_factory=list)
    session_count: int = 0
    evolution_log: List[Dict] = field(default_factory=list)

# -------------------------------------------------------------------------
# 2. Preference Aggregator
# -------------------------------------------------------------------------
class PreferenceAggregator:
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        reasoning_effort: str = None,
        generation_max_tokens: int = 4096,
        verification_max_tokens: int = 2048,
    ):
        self.client = client
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.generation_max_tokens = generation_max_tokens
        self.verification_max_tokens = verification_max_tokens
        self.max_retries = 10    # 최대 재시도 횟수

    async def update_memory(self, current_state: MemoryState, session_dialogue: str, session_api_calls: List[str]) -> MemoryState:
        session_idx = current_state.session_count + 1
        set_usage_session(session_idx)
        session_header = f"\n=== Session {session_idx} ===\n"
        full_dialogue = current_state.accumulated_dialogue + session_header + session_dialogue
        
        current_session_apis = [f"[Session {session_idx}] {api}" for api in session_api_calls]
        full_api_list = current_state.accumulated_api_calls + current_session_apis
        full_api_str = "\n".join(full_api_list) if full_api_list else "No API calls recorded."

        candidate_pref = current_state.implicit_pref
        if candidate_pref == "{}": 
            candidate_pref = "None"

        feedback = ""
        # 초기화
        updated_log = list(current_state.evolution_log)
        session_attempts_log = []
        
        # 기본적으로 실패 시 이전 상태 유지를 위해 candidate_pref_str 초기화
        candidate_pref_str = candidate_pref if isinstance(candidate_pref, str) else json.dumps(candidate_pref)

        for i in range(self.max_retries):
            # Step A: Generate / Refine
            # [수정됨] reasoning content도 함께 반환받음
            candidate_pref_json, gen_reasoning = await self._generate_preference(
                prev_implicit=current_state.implicit_pref,
                full_dialogue=full_dialogue,
                full_api_calls=full_api_str,
                feedback=feedback,
                previous_draft=candidate_pref if i > 0 else None
            )
            
            # JSON이 비어있으면(파싱 실패 등) 재시도
            if not candidate_pref_json:
                continue

            candidate_pref_str = json.dumps(candidate_pref_json, ensure_ascii=False, indent=2)

            # Step B: Verify
            # [수정됨] verifier의 reasoning content도 반환받음
            is_valid, new_feedback, verifier_input, verifier_output_json, ver_reasoning = await self._verify_preference(
                full_dialogue=full_dialogue,
                full_api_calls=full_api_str,
                candidate_pref=candidate_pref_str
            )

            attempt_record = {
                "step": i + 1,
                "draft_preference": candidate_pref_json,
                "gen_reasoning": gen_reasoning,  # [추가됨] 생성 단계의 생각 과정
                "is_valid": is_valid,
                "verifier_feedback": new_feedback,
                "verifier_input": verifier_input,
                "verifier_output": verifier_output_json,
                "ver_reasoning": ver_reasoning   # [추가됨] 검증 단계의 생각 과정
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
            "final_preference_at_session": json.loads(candidate_pref) if isinstance(candidate_pref, str) and candidate_pref not in ["None", "{}"] else {},
            "stored_memory_tokens_after_session": count_json_tokens(
                json.loads(candidate_pref)
                if isinstance(candidate_pref, str) and candidate_pref not in ["None", "{}"]
                else {}
            ),
        })

        updated_state = MemoryState(
            implicit_pref=candidate_pref,
            accumulated_dialogue=full_dialogue,
            accumulated_api_calls=full_api_list,
            session_count=session_idx,
            evolution_log=updated_log
        )
        return updated_state

    async def _generate_preference(self, prev_implicit: str, full_dialogue: str, full_api_calls: str, feedback: str = "", previous_draft: str = None) -> Tuple[dict, str]:
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
        
        try:
            request_kwargs = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": context_prompt},
                    {"role": "user", "content": task_prompt}
                ],
                "temperature": 0.4,
                "max_tokens": self.generation_max_tokens,
                "response_format": {"type": "json_object"} if "gpt" in self.model else None,
            }
            if self.reasoning_effort:
                request_kwargs["reasoning_effort"] = self.reasoning_effort

            response = await self.client.chat.completions.create(
                **request_kwargs
            )
            record_response_usage(
                response,
                component="refiner" if previous_draft else "generator",
                provider="openai",
                model=self.model,
            )
            
            message = response.choices[0].message
            raw_content = message.content or ""
            
            # [NEW] Reasoning Parsing
            parsed = parse_reasoning(raw_content)
            clean_content = parsed["llm_output"]
            reasoning_content = (
                getattr(message, "reasoning_content", None)
                or parsed["reasoning_content"]
            )

            return self._parse_json(clean_content), reasoning_content

        except Exception as e:
            print(f"[Gen Error] {e}")
            return {}, ""

    async def _verify_preference(self, full_dialogue: str, full_api_calls: str, candidate_pref: str) -> Tuple[bool, str, str, Dict[str, Any], str]:
        prompt = LATENT_PREF_VERIFIER_PROMPT.format(
            full_dialogue=full_dialogue,
            full_api_calls=full_api_calls,
            candidate_pref=candidate_pref
        )

        try:
            request_kwargs = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a Preference Verification Module. Output JSON only."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0,
                "max_tokens": self.verification_max_tokens,
                "response_format": {"type": "json_object"} if "gpt" in self.model else None,
            }
            if self.reasoning_effort:
                request_kwargs["reasoning_effort"] = self.reasoning_effort

            response = await self.client.chat.completions.create(**request_kwargs)
            record_response_usage(
                response,
                component="verifier",
                provider="openai",
                model=self.model,
            )
            
            message = response.choices[0].message
            raw_content = message.content or ""
            
            # [NEW] Reasoning Parsing
            parsed = parse_reasoning(raw_content)
            clean_content = parsed["llm_output"]
            reasoning_content = (
                getattr(message, "reasoning_content", None)
                or parsed["reasoning_content"]
            )

            res_json = self._parse_json(clean_content)
            
            # JSON 파싱 실패 시 기본값 처리
            if not res_json:
                return False, "Failed to parse verifier output", prompt, {}, reasoning_content

            return res_json.get("valid", False), res_json.get("feedback", ""), prompt, res_json, reasoning_content
            
        except Exception as e:
            print(f"[Ver Error] {e}")
            return False, str(e), prompt, {"error": str(e)}, ""

    def _parse_json(self, content: str) -> dict:
        """
        LLM의 응답에서 JSON 객체를 추출하는 강화된 함수.
        """
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        if "```json" in content:
            content = re.sub(r"```json\s*", "", content)
            content = re.sub(r"\s*```", "", content)
        elif "```" in content:
            content = content.replace("```", "")

        start_idx = content.find('{')
        end_idx = content.rfind('}')

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = content[start_idx : end_idx + 1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
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
        usage_token = begin_usage_collection()

        try:
            for session in sessions:
                dialogue_text = format_dialogue(session.get("dialogue", []))
                api_calls = session.get("api_call", [])

                current_state = await aggregator.update_memory(
                    current_state,
                    dialogue_text,
                    api_calls,
                )
        finally:
            construction_token_usage = end_usage_collection(usage_token)
        
        # 1. Main Result Record
        result_record = {
            "example_id": example_id,
            "method": "ours_memory",
            "reasoning_effort": aggregator.reasoning_effort,
            "final_implicit_preference": current_state.implicit_pref,
            "final_accumulated_api_calls": current_state.accumulated_api_calls,
            "total_sessions_processed": current_state.session_count,
            "preference_evolution_history": current_state.evolution_log,
            "memory_mode": "verified_refine",
            "construction_token_usage": construction_token_usage,
            "token_counts": construction_token_usage["summary"],
            **encoding_metadata(),
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

                # Refinement Log (Reasoning 포함)
                r_log = base_meta.copy()
                r_log["draft_preference"] = step_info.get("draft_preference")
                r_log["reasoning_content"] = step_info.get("gen_reasoning") # [추가됨]
                refinement_logs.append(json.dumps(r_log, ensure_ascii=False))

                # Verifier Log (Reasoning 포함)
                v_log = base_meta.copy()
                v_log["is_valid"] = step_info.get("is_valid")
                v_log["verifier_input"] = step_info.get("verifier_input")
                v_log["verifier_output"] = step_info.get("verifier_output")
                v_log["reasoning_content"] = step_info.get("ver_reasoning") # [추가됨]
                
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
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[Info] No API Key provided. Using 'EMPTY' for vLLM compatibility.")
        api_key = "EMPTY"

    dataset = load_dataset(args.input)
    if not dataset:
        return

    print(f"=== Initializing Client ===")
    print(f"  - Base URL: {args.api_base}")
    print(f"  - Model: {args.model}")
    print(f"  - Reasoning Effort: {args.reasoning_effort or 'default'}")
    
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=args.api_base
    )
    
    aggregator = PreferenceAggregator(
        client,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        generation_max_tokens=args.generation_max_tokens,
        verification_max_tokens=args.verification_max_tokens,
    )
    
    semaphore = asyncio.Semaphore(args.concurrency)
    file_lock = asyncio.Lock()

    # Prepare output files and optionally resume completed dialogues.
    for fpath in [args.output, args.verifier_output, args.refinement_output]:
        parent = os.path.dirname(os.path.abspath(fpath))
        os.makedirs(parent, exist_ok=True)

    completed_ids = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    completed_ids.add(str(json.loads(line).get("example_id")))
                except json.JSONDecodeError:
                    continue
        dataset = [
            example for example in dataset
            if str(example.get("example_id")) not in completed_ids
        ]
        print(f"  - Resume: skipping {len(completed_ids)} completed dialogues")
    else:
        for fpath in [args.output, args.verifier_output, args.refinement_output]:
            with open(fpath, "w", encoding="utf-8"):
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
    
    await client.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process latent preferences using vLLM or OpenAI.")
    
    parser.add_argument("--input", type=str, default="/data/minseo/experiment8/data/MPT_v2_mix600.json", help="Path to input JSON file")
    
    parser.add_argument("--output", type=str, default="result.jsonl", help="Main result output file")
    parser.add_argument("--verifier_output", type=str, default="verifier_logs.jsonl", help="Verifier details log")
    parser.add_argument("--refinement_output", type=str, default="refinement_logs.jsonl", help="Refinement process log")
    
    parser.add_argument("--model", type=str, default="deepseek-r1", help="Model name (e.g. deepseek-r1, gpt-4o)")
    parser.add_argument("--api_base", type=str, default="http://localhost:8001/v1", help="API Base URL")
    parser.add_argument("--api_key", type=str, default=None, help="API Key")
    
    parser.add_argument("--concurrency", type=int, default=100, help="Number of concurrent users to process")
    parser.add_argument("--reasoning_effort", choices=["low", "medium", "high"], default=None)
    parser.add_argument("--generation_max_tokens", type=int, default=4096)
    parser.add_argument("--verification_max_tokens", type=int, default=2048)
    parser.add_argument("--resume", action="store_true", help="Skip example_ids already present in --output")
    
    args = parser.parse_args()
    
    asyncio.run(process_dataset_concurrently(args))
