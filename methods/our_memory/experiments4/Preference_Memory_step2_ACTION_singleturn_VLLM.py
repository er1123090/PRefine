import tqdm
import os
import json
import argparse
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
import re
import copy
import asyncio
import itertools
from openai import AsyncOpenAI

# Gemini Library Import
try:
    from google import genai
    from google.genai import types
except ImportError:
    print("[Warning] google-genai library not found. Gemini models will not work.")

# ---------------------------------------------------------
# [Prompt Template]
# ---------------------------------------------------------
from prompt_inference import IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE

# ---------------------------------------------------------
# 1. Helpers & Loaders
# ---------------------------------------------------------

def parse_reasoning(raw_content: str) -> Dict[str, str]:
    """
    [New] DeepSeek 등 Reasoning 모델의 출력에서 <think> 태그를 파싱하여
    reasoning_content와 실제 답변(llm_output)을 분리합니다.
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
        # </think> 태그가 있는 경우: 앞부분은 reasoning, 뒷부분은 output
        reasoning_part = raw_content[:end_idx]
        # <think> 시작 태그 제거 및 공백 정리
        reasoning_content = reasoning_part.replace("<think>", "").strip()
        # </think> 태그 이후 내용만 추출
        clean_content = raw_content[end_idx + len(end_tag):].strip()
    else:
        # 태그가 없는 경우: 전체가 output이고 reasoning은 없음
        reasoning_content = ""
        clean_content = raw_content.strip()

    return {
        "llm_output": clean_content,
        "reasoning_content": reasoning_content
    }

def load_tools_from_file(file_path: str) -> List[Dict]:
    if not os.path.exists(file_path):
        return []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tools = json.load(f)
        return tools
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return []

def load_chains_dataset(fpath: str) -> pd.DataFrame:
    try:
        df = pd.read_json(fpath, lines=True)
        return df
    except ValueError:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "dataset" in data:
            return pd.DataFrame(data["dataset"])
        return pd.DataFrame(data)

def load_query_map(fpath: str) -> Dict[str, str]:
    if not os.path.exists(fpath):
        return {}
    with open(fpath, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if isinstance(raw_data, dict):
        return raw_data

    if isinstance(raw_data, list):
        query_map = {}
        for item in raw_data:
            if not isinstance(item, dict):
                continue

            domain = None
            targets = item.get("target") or []
            if isinstance(targets, list):
                for target in targets:
                    if isinstance(target, dict) and target.get("domain"):
                        domain = str(target["domain"])
                        break

            if not domain:
                api_calls = item.get("api_call") or []
                if isinstance(api_calls, list):
                    for api_call in api_calls:
                        if isinstance(api_call, str) and "(" in api_call:
                            domain = api_call.split("(", 1)[0].strip()
                            break

            if not domain:
                continue

            utterance = None
            query_turns = item.get("query") or []
            if isinstance(query_turns, list):
                for turn in query_turns:
                    if not isinstance(turn, dict):
                        continue
                    role = str(turn.get("role", "")).lower()
                    message = turn.get("message") or turn.get("content")
                    if role == "user" and message:
                        utterance = str(message).strip()
                        break

            if utterance:
                query_map[domain] = utterance

        return query_map

    return {}

def load_memory_file(fpath: str) -> Dict[str, Any]:
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Memory file not found: {fpath}")
    
    memory_storage = {}
    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                record = json.loads(line)
                rec_id = str(record.get("example_id"))
                memory_storage[rec_id] = record
            except json.JSONDecodeError:
                continue
    print(f"[Info] Loaded {len(memory_storage)} memory records from {fpath}")
    return memory_storage

# ---------------------------------------------------------
# 2. Logic to Assign User Utterance
# ---------------------------------------------------------
def generate_func_strings(domain: str, slot_values_map: Dict[str, List[str]]) -> List[str]:
    if not slot_values_map:
        return []
    sorted_keys = sorted(slot_values_map.keys())
    values_lists = [slot_values_map[k] for k in sorted_keys]
    combinations = list(itertools.product(*values_lists))
    results = []
    for combo in combinations:
        args_parts = []
        for key, val in zip(sorted_keys, combo):
            args_parts.append(f'{key}="{val}"')
        args_str = ", ".join(args_parts)
        results.append(f"{domain}({args_str})")
    return results

def assign_user_utterances(
    pref_list_path: str, 
    example: Dict[str, Any], 
    query_map: Dict[str, str], 
    pref_type: str, 
    pref_group_path: str = None
) -> List[Tuple[str, List[str]]]: 
    
    results = []
    def to_str(val):
        if isinstance(val, bool): return "True" if val else "False"
        return str(val)

    # [CASE 1] easy
    if pref_type == "easy":
        if not os.path.exists(pref_list_path): return []
        with open(pref_list_path, "r", encoding="utf-8") as f: pref_list = json.load(f)

        api_calls = example.get("api_calls", [])
        if isinstance(api_calls, list):
            for call_str in api_calls:
                if "(" in call_str:
                    domain = call_str.split("(")[0].strip()
                    try: args_content = call_str.split("(", 1)[1].rsplit(")", 1)[0]
                    except IndexError: continue 
                else:
                    domain = call_str.strip(); args_content = ""

                if domain not in query_map or domain not in pref_list: continue

                pattern = r'(\w+)=["\']([^"\']+)["\']'
                matches = re.findall(pattern, args_content)
                target_pref_slots = pref_list.get(domain, [])
                
                current_slot_map = {}
                for slot, value in matches:
                    if slot in target_pref_slots:
                        current_slot_map[slot] = [to_str(value)]
                
                if current_slot_map:
                    ground_truth_strs = generate_func_strings(domain, current_slot_map)
                    results.append((query_map[domain], ground_truth_strs))
                    
        return results

    # [CASE 2] medium
    elif pref_type == "medium":
        prefs = example.get("api_calls_pref", [])
        if not isinstance(prefs, list) or not prefs: return []
        if not pref_group_path or not os.path.exists(pref_group_path): return []
        
        with open(pref_group_path, "r", encoding="utf-8") as f: pref_group_data = json.load(f)

        for pref in prefs:
            group_name = pref.get("value_group")
            if group_name not in pref_group_data: continue
            
            group_rules = pref_group_data[group_name].get("rules", [])
            domain_data_map = {} 

            for evidence in pref.get("evidence", []):
                e_domain = evidence.get("domain")
                e_slot = evidence.get("slot")
                if not e_domain or not e_slot or e_domain not in query_map: continue

                candidate_values = []
                for rule in group_rules:
                    if rule.get("domain") == e_domain and rule.get("slot") == e_slot:
                        candidate_values.append(to_str(rule.get("value")))
                
                if candidate_values:
                    if e_domain not in domain_data_map: domain_data_map[e_domain] = {}
                    if e_slot not in domain_data_map[e_domain]: domain_data_map[e_domain][e_slot] = set()
                    for v in candidate_values: domain_data_map[e_domain][e_slot].add(v)
            
            for domain, slot_map in domain_data_map.items():
                final_slot_map = {k: list(v) for k, v in slot_map.items()}
                ground_truth_strs = generate_func_strings(domain, final_slot_map)
                results.append((query_map[domain], ground_truth_strs))
                        
        return results
        
    # [CASE 3] hard
    elif pref_type == "hard":
        if not pref_group_path or not os.path.exists(pref_group_path): return []
        with open(pref_group_path, "r", encoding="utf-8") as f: pref_group_data = json.load(f)
        
        prefs = example.get("api_calls_pref", [])
        if not isinstance(prefs, list) or not prefs: return []

        for pref in prefs:
            current_group_name = pref.get("value_group")
            if not current_group_name or current_group_name not in pref_group_data: continue
            
            used_domains = {e.get("domain") for e in pref.get("evidence", []) if e.get("domain")}
            rules = pref_group_data[current_group_name].get("rules", [])
            
            candidate_domains = set()
            for rule in rules:
                d = rule.get("domain")
                if d and d in query_map and d not in used_domains:
                    candidate_domains.add(d)
            
            for cand_domain in candidate_domains:
                slot_values_map = {}
                for rule in rules:
                    if rule.get("domain") == cand_domain:
                        s = rule.get("slot"); v = rule.get("value")
                        if s and v is not None:
                            if s not in slot_values_map: slot_values_map[s] = []
                            val_str = to_str(v)
                            if val_str not in slot_values_map[s]: slot_values_map[s].append(val_str)
                
                if slot_values_map:
                    ground_truth_strs = generate_func_strings(cand_domain, slot_values_map)
                    results.append((query_map[cand_domain], ground_truth_strs))
        return results

    return results

# ---------------------------------------------------------
# 3. Memory Formatting & Prompt Building
# ---------------------------------------------------------
def format_memory_content(content: Any) -> str:
    if content is None:
        return "None"
    if isinstance(content, (dict, list)):
        return json.dumps(content, indent=2, ensure_ascii=False)
    return str(content).strip()

def format_api_calls(api_calls: List[str]) -> str:
    if not api_calls or not isinstance(api_calls, list):
        return "None"
    return "\n".join(api_calls)

def build_memory_input_prompt(
    example: Dict[str, Any],
    user_memory: Dict[str, Any],
    current_user_utterance: str,
    template: str,
    context_type: str,
    tools_schema: List[Dict] = None
) -> str:
    # 요청하신 대로 final_implicit_pref와 final_accumulated_api_calls 우선 사용
    raw_implicit_pref = user_memory.get('final_implicit_preference') or user_memory.get('final_implicit_pref')
    implicit_pref_str = format_memory_content(raw_implicit_pref)
    
    raw_api_calls = user_memory.get('final_accumulated_api_calls') or user_memory.get('final_api_list')
    api_history_str = format_api_calls(raw_api_calls)

    raw_explicit_pref = user_memory.get('final_explicit_preference') or user_memory.get('final_explicit_pref')
    explicit_pref_str = format_memory_content(raw_explicit_pref)

    sessions = example.get("sessions", [])
    if sessions:
        sessions_str = []
        for idx, instruction_data in enumerate(sessions, start=1):
            lines = [f"[Session {idx}]"]
            for turn in instruction_data.get("dialogue", []):
                role = turn.get("role", "").capitalize()
                content = turn.get("message") or turn.get("content") or ""
                if role and content: lines.append(f"{role}: {content}")
            sessions_str.append("\n".join(lines))
        dialogue_text = "\n\n".join(sessions_str)
    else:
        raw_turns = example.get("turns", [])
        dialogue_text = "\n".join([f"{t.get('speaker', 'User')}: {t.get('utterance', '')}" for t in raw_turns]) if raw_turns else "None"

    retrieved_memories_block = ""
    dialogue_history_input = ""

    if context_type == "api-only":
        retrieved_memories_block = f"[Past API History]:\n{api_history_str}"
        dialogue_history_input = "None"
    else:
        memory_block_parts = []
        if raw_explicit_pref:
            memory_block_parts.append(f"[Explicit Preferences]:\n{explicit_pref_str}")
        memory_block_parts.append(f"[Implicit Preferences]:\n{implicit_pref_str}")

        if context_type == "memory_only":
            retrieved_memories_block = "\n\n".join(memory_block_parts)
            dialogue_history_input = "None"
        elif context_type == "memory_api":
            memory_block_parts.append(f"[Past API History]:\n{api_history_str}")
            retrieved_memories_block = "\n\n".join(memory_block_parts)
            dialogue_history_input = "None"
        elif context_type == "memory_diag":
            retrieved_memories_block = "\n\n".join(memory_block_parts)
            dialogue_history_input = dialogue_text
        else:
            retrieved_memories_block = "\n\n".join(memory_block_parts)
            dialogue_history_input = dialogue_text

    schema_str = json.dumps(tools_schema, indent=2, ensure_ascii=False) if tools_schema else "No specific schema provided."

    prompt = template.format(
        preference_schema=schema_str,
        retrieved_memories=retrieved_memories_block,
        dialogue_history=dialogue_history_input,
        user_utterance=current_user_utterance.strip()
    )
    return prompt


def limit_prepared_items(
    prepared_items: List[Dict[str, Any]],
    max_queries: Optional[int],
) -> List[Dict[str, Any]]:
    if max_queries is None:
        return prepared_items
    if max_queries <= 0:
        raise ValueError(f"max_queries must be positive, got {max_queries}")

    total_items = len(prepared_items)
    if total_items <= max_queries:
        return prepared_items

    if max_queries == 1:
        selected_indices = [0]
    else:
        selected_indices = [
            ((total_items - 1) * idx) // (max_queries - 1)
            for idx in range(max_queries)
        ]

    limited_items = [prepared_items[idx] for idx in selected_indices]
    print(
        f"[Info] Limiting prepared items from {total_items} to {len(limited_items)} "
        f"with deterministic evenly spaced sampling (max_queries={max_queries})."
    )
    return limited_items

# ---------------------------------------------------------
# 4. Async LLM Call (Updated for vLLM Support)
# ---------------------------------------------------------
async def call_llm_api_async(
    prompt: str, 
    model_name: str, 
    openai_client: AsyncOpenAI = None, 
    tools_schema: List[Dict] = None,
    reasoning_effort: str = None
) -> str:
    try:
        if "gemini" in model_name.lower():
            if not os.environ.get("GOOGLE_API_KEY"): return "API_KEY_MISSING_GOOGLE"
            client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
            config_params = {"temperature": 0.0}

            if reasoning_effort and ("gemini-3" in model_name.lower() or "flash" in model_name.lower()):
                config_params["thinking_config"] = types.ThinkingConfig(
                    include_thoughts=True,
                    thinking_level=reasoning_effort.lower()
                )

            conf = types.GenerateContentConfig(**config_params)
            response = await client.aio.models.generate_content(
                model=model_name, contents=prompt, config=conf
            )
            return response.text.strip()

        else:
            # OpenAI / vLLM Logic
            if not openai_client: return "API_KEY_MISSING_OPENAI"
            
            kwargs = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0
            }
            max_tokens_env = os.environ.get("VLLM_MAX_TOKENS", "").strip()
            if max_tokens_env:
                try:
                    max_tokens = int(max_tokens_env)
                    if max_tokens > 0:
                        kwargs["max_tokens"] = max_tokens
                except ValueError:
                    pass
            
            # vLLM usually does not support reasoning_effort.
            # Only add it if we are targeting official OpenAI reasoning models
            is_openai_reasoning = any(k in model_name.lower() for k in ["o1", "o3"])
            if is_openai_reasoning and reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort.lower()
                if "temperature" in kwargs: del kwargs["temperature"] 

            # Make the call
            response = await openai_client.chat.completions.create(**kwargs)
            message = response.choices[0].message
            return message.content.strip()

    except Exception as e:
        print(f"LLM API Error ({model_name}): {e}")
        return f"API_ERROR: {str(e)}"

# ---------------------------------------------------------
# 5. Async Processing Pipeline
# ---------------------------------------------------------
async def process_single_item(
    original_ex: Dict[str, Any],
    user_memory: Dict[str, Any],
    utterance: str, 
    ground_truth: Any, 
    sub_idx: int,
    model_name: str,
    prompt_template: str,
    context_type: str,
    openai_client: AsyncOpenAI,
    tools_schema: List[Dict],
    log_path: str,
    semaphore: asyncio.Semaphore,
    file_lock: asyncio.Lock,
    pbar: tqdm.tqdm,
    reasoning_effort: str = None 
):
    async with semaphore:
        prompt = build_memory_input_prompt(
            example=original_ex,
            user_memory=user_memory,
            current_user_utterance=utterance,
            template=prompt_template,
            context_type=context_type,
            tools_schema=tools_schema
        )

        llm_output_raw = await call_llm_api_async(
            prompt, model_name, openai_client, tools_schema, reasoning_effort
        )

        # ---------------------------------------------------------
        # [NEW] Apply Reasoning Parsing (Formatting) Immediately
        # ---------------------------------------------------------
        parsed_result = parse_reasoning(llm_output_raw)
        clean_output = parsed_result["llm_output"]
        reasoning_content = parsed_result["reasoning_content"]

        log_record = {
            "timestamp": datetime.now().isoformat(),
            "example_id": original_ex.get("example_id"),
            "example_id_sub": f"{original_ex.get('example_id')}_{sub_idx}",
            "model_name": model_name,
            "context_type": context_type,
            "pref_type": "implicit",
            "injected_utterance": utterance,
            "reference_ground_truth": ground_truth,
            "model_input": prompt,
            "raw_model_output": llm_output_raw, # 원본 저장
            "model_output": clean_output,       # 정제된 출력 저장
            "reasoning_content": reasoning_content # 추론 과정 저장
        }
        
        async with file_lock:
            dirpath = os.path.dirname(log_path)
            if dirpath: os.makedirs(dirpath, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_record, ensure_ascii=False) + "\n")

        result_ex = copy.deepcopy(original_ex)
        result_ex["example_id_sub"] = f"{original_ex.get('example_id')}_{sub_idx}"
        result_ex["test_utterance"] = utterance
        result_ex["reference_ground_truth"] = ground_truth
        
        # 결과 객체에도 분리된 필드 추가
        result_ex["llm_output"] = clean_output
        result_ex["reasoning_content"] = reasoning_content
        result_ex["raw_content"] = llm_output_raw
        
        pbar.update(1)
        return result_ex

async def process_with_llm_async(
    input_path: str, memory_path: str, output_path: str, log_path: str, 
    query_map_path: str, pref_list_path: str, pref_group_path: str,
    tools_schema_path: str,
    prompt_template: str, context_type: str, pref_type: str,
    model_name: str, concurrency: int = 10,
    reasoning_effort: str = None,
    base_url: str = None,
    api_key: str = None,
    max_queries: Optional[int] = None,
):
    df = load_chains_dataset(input_path)
    memory_map = load_memory_file(memory_path)
    query_map = load_query_map(query_map_path)
    tools_schema = load_tools_from_file(tools_schema_path)
    print(f"[Info] Loaded {len(query_map)} query mappings from {query_map_path}")
    
    # -------------------------------------------------------------
    # Client Initialization (Modified for vLLM)
    # -------------------------------------------------------------
    openai_client = None
    
    # Check if we are using vLLM or OpenAI
    if "gemini" not in model_name.lower():
        client_args = {}
        
        # If base_url is provided (vLLM Case)
        if base_url:
            print(f"[Info] Using Custom/vLLM Base URL: {base_url}")
            client_args["base_url"] = base_url
            # vLLM usually needs a dummy API Key if one isn't set, usually 'EMPTY'
            client_args["api_key"] = api_key if api_key else "EMPTY"
        else:
            # Standard OpenAI Case
            if os.environ.get("OPENAI_API_KEY"):
                client_args["api_key"] = os.environ.get("OPENAI_API_KEY")
        
        if "api_key" in client_args:
            openai_client = AsyncOpenAI(**client_args)
    # -------------------------------------------------------------

    print(f"Starting ASYNC process... (Model: {model_name})")

    tasks = []
    semaphore = asyncio.Semaphore(concurrency)
    file_lock = asyncio.Lock()
    prepared_items = []
    skipped_count = 0
    
    for _, row in df.iterrows():
        original_ex = row.to_dict()
        ex_id = str(original_ex.get("example_id"))
        
        user_memory = memory_map.get(ex_id)
        if not user_memory:
            skipped_count += 1
            continue

        if pref_type == "easy":
            if not original_ex.get("api_calls"): skipped_count += 1; continue
        elif pref_type in ["medium", "hard"]:
            if not original_ex.get("api_calls_pref"): skipped_count += 1; continue

        pairs_list = assign_user_utterances(pref_list_path, original_ex, query_map, pref_type, pref_group_path)
        
        if not pairs_list:
            skipped_count += 1
            continue

        for sub_idx, (utterance, ground_truth) in enumerate(pairs_list):
            prepared_items.append({
                "original_ex": original_ex,
                "user_memory": user_memory,
                "utterance": utterance,
                "ground_truth": ground_truth,
                "sub_idx": sub_idx
            })

    prepared_items = limit_prepared_items(prepared_items, max_queries)
    total_tasks = len(prepared_items)
    print(f"Total tasks: {total_tasks}. Skipped dialogues: {skipped_count}")
    if total_tasks == 0:
        raise RuntimeError(
            "No tasks were prepared. "
            f"query_path={query_map_path} yielded {len(query_map)} query mappings, "
            f"dataset_rows={len(df)}, memory_records={len(memory_map)}, "
            f"pref_type={pref_type}, context_type={context_type}. "
            "This usually means the query file format does not match the loader."
        )

    pbar = tqdm.tqdm(total=total_tasks, desc="Processing Async")

    for item in prepared_items:
        task = asyncio.create_task(
            process_single_item(
                original_ex=item['original_ex'],
                user_memory=item['user_memory'],
                utterance=item['utterance'],
                ground_truth=item['ground_truth'],
                sub_idx=item['sub_idx'],
                model_name=model_name,
                prompt_template=prompt_template,
                context_type=context_type,
                openai_client=openai_client,
                tools_schema=tools_schema,
                log_path=log_path,
                semaphore=semaphore,
                file_lock=file_lock,
                pbar=pbar,
                reasoning_effort=reasoning_effort
            )
        )
        tasks.append(task)
    
    processed_data = await asyncio.gather(*tasks)
    pbar.close()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=4, ensure_ascii=False)
    print(f"Saved -> {output_path}")

# ---------------------------------------------------------
# 6. Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Paths
    parser.add_argument("--input_path", type=str, default="/data/minseo/experiments4/data/1229_dev_6.json")
    parser.add_argument("--memory_path", type=str, required=True, help="Path to generated memory jsonl")
    parser.add_argument("--output_path", type=str, default="output_memory_eval.json")
    parser.add_argument("--log_path", type=str, default="process_memory.log")
    parser.add_argument("--query_path", type=str, default="/data/minseo/experiments4/query_singleturn.json")
    parser.add_argument("--pref_list_path", type=str, default="/data/minseo/experiments4/pref_list.json")
    parser.add_argument("--pref_group_path", type=str, default="/data/minseo/experiments4/pref_group.json")
    parser.add_argument("--tools_schema_path", type=str, default="/data/minseo/experiments4/schema_easy.json")

    # Experiment Settings
    parser.add_argument("--pref_type", type=str, choices=["medium", "easy", "hard"], required=True)
    parser.add_argument("--context_type", type=str, 
                        choices=["memory_only", "memory_diag", "memory_api", "api-only"], 
                        required=True)
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model name as served in vLLM")
    parser.add_argument("--concurrency", type=int, default=50, help="Concurrency level (Higher for vLLM)")
    parser.add_argument("--reasoning_effort", type=str, choices=["minimal", 'low', "medium", "high"], default=None)
    parser.add_argument("--max_queries", type=int, default=None)

    # vLLM Specific Settings
    parser.add_argument("--base_url", type=str, default="http://localhost:8001/v1", help="vLLM server URL")
    parser.add_argument("--api_key", type=str, default="EMPTY", help="API Key for vLLM")

    args = parser.parse_args()

    asyncio.run(
        process_with_llm_async(
            input_path=args.input_path,
            memory_path=args.memory_path,
            output_path=args.output_path,
            log_path=args.log_path,
            query_map_path=args.query_path,
            pref_list_path=args.pref_list_path,
            pref_group_path=args.pref_group_path,
            tools_schema_path=args.tools_schema_path,
            prompt_template=IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE,
            context_type=args.context_type,
            pref_type=args.pref_type,
            model_name=args.model_name,
            concurrency=args.concurrency,
            reasoning_effort=args.reasoning_effort,
            base_url=args.base_url,
            api_key=args.api_key,
            max_queries=args.max_queries,
        )
    )
