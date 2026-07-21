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
from prompt import IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE
# ---------------------------------------------------------
# 1. Helpers & Loaders
# ---------------------------------------------------------
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
        # Handle dict wrapper if exists
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
    """Load memory jsonl file and map by example_id."""
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
# 3. Memory Formatting & Prompt Building (Updated)
# ---------------------------------------------------------
def build_memory_input_prompt(
    example: Dict[str, Any],
    user_memory: Dict[str, Any],
    current_user_utterance: str,
    template: str,
    context_type: str,
    tools_schema: List[Dict] = None
) -> str:
    # 1. Retrieve & Format Memory (JSON Dump directly)
    
    # Explicit Preference
    raw_explicit = user_memory.get('final_explicit_pref')
    if raw_explicit is None or raw_explicit == "None":
        explicit_pref = "None"
    else:
        # Dump any dict/list/string as formatted JSON
        explicit_pref = json.dumps(raw_explicit, ensure_ascii=False, indent=2)

    # Implicit Preference
    raw_implicit = user_memory.get('final_implicit_pref')
    if not raw_implicit or raw_implicit == "None":
        implicit_pref = "None"
    else:
        # Dump dict/list as formatted JSON
        implicit_pref = json.dumps(raw_implicit, ensure_ascii=False, indent=2)
    
    # API History
    # Try 'final_api_list' first, then fallback to 'final_accumulated_api_calls'
    raw_api = user_memory.get('final_api_list') or user_memory.get('final_accumulated_api_calls', [])
    
    if not raw_api or raw_api == "None":
        api_list_str = "None"
    elif isinstance(raw_api, list):
        # Join list with newlines for readability (or you can use json.dumps here too if preferred)
        api_list_str = "\n".join(raw_api)
    else:
        # Fallback for other types
        api_list_str = str(raw_api)

    # 2. Get Dialogue History
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

    # 3. Construct Context Blocks
    retrieved_memories_block = ""
    dialogue_history_input = ""

    if context_type == "memory_only":
        retrieved_memories_block = f"[Explicit Preferences]:\n{explicit_pref}\n\n[Implicit Preferences]:\n{implicit_pref}"
        dialogue_history_input = "None"
    
    elif context_type == "memory_api":
        retrieved_memories_block = f"[Explicit Preferences]:\n{explicit_pref}\n\n[Implicit Preferences]:\n{implicit_pref}\n\n[Past API History]:\n{api_list_str}"
        dialogue_history_input = "None"

    elif context_type == "memory_diag":
        retrieved_memories_block = f"[Explicit Preferences]:\n{explicit_pref}\n\n[Implicit Preferences]:\n{implicit_pref}"
        dialogue_history_input = dialogue_text
    
    else: # Default
        retrieved_memories_block = f"[Explicit Preferences]:\n{explicit_pref}\n\n[Implicit Preferences]:\n{implicit_pref}"
        dialogue_history_input = dialogue_text

    # 4. Schema Formatting
    schema_str = json.dumps(tools_schema, indent=2, ensure_ascii=False) if tools_schema else "No specific schema provided."

    # 5. Final Formatting
    prompt = template.format(
        preference_schema=schema_str,
        retrieved_memories=retrieved_memories_block,
        dialogue_history=dialogue_history_input,
        user_utterance=current_user_utterance.strip()
    )
    return prompt

# ---------------------------------------------------------
# 4. Async LLM Call
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
            if not openai_client: return "API_KEY_MISSING_OPENAI"
            
            kwargs = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0
            }
            
            is_reasoning_model = any(k in model_name.lower() for k in ["o1", "o3", "gpt-5"])
            if is_reasoning_model and reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort.lower()
                if "temperature" in kwargs: del kwargs["temperature"]

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
        # Build Prompt using updated logic
        prompt = build_memory_input_prompt(
            example=original_ex,
            user_memory=user_memory,
            current_user_utterance=utterance,
            template=prompt_template,
            context_type=context_type,
            tools_schema=tools_schema
        )

        llm_output = await call_llm_api_async(
            prompt, model_name, openai_client, tools_schema, reasoning_effort
        )

        # Logging
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
            "model_output": llm_output,
        }
        
        async with file_lock:
            dirpath = os.path.dirname(log_path)
            if dirpath: os.makedirs(dirpath, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_record, ensure_ascii=False) + "\n")

        # Result Object
        result_ex = copy.deepcopy(original_ex)
        result_ex["example_id_sub"] = f"{original_ex.get('example_id')}_{sub_idx}"
        result_ex["test_utterance"] = utterance
        result_ex["reference_ground_truth"] = ground_truth
        result_ex["llm_output"] = llm_output
        result_ex["used_memory"] = user_memory 
        
        pbar.update(1)
        return result_ex

async def process_with_llm_async(
    input_path: str, memory_path: str, output_path: str, log_path: str, 
    query_map_path: str, pref_list_path: str, pref_group_path: str,
    tools_schema_path: str,
    prompt_template: str, context_type: str, pref_type: str,
    model_name: str, concurrency: int = 10,
    reasoning_effort: str = None 
):
    # 1. Load Data & Memory
    df = load_chains_dataset(input_path)
    memory_map = load_memory_file(memory_path)
    query_map = load_query_map(query_map_path)
    tools_schema = load_tools_from_file(tools_schema_path)
    print(f"[Info] Loaded {len(query_map)} query mappings from {query_map_path}")
    
    openai_client = None
    if os.environ.get("OPENAI_API_KEY"):
        openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    print(f"Starting ASYNC process... (Model: {model_name})")

    tasks = []
    semaphore = asyncio.Semaphore(concurrency)
    file_lock = asyncio.Lock()
    prepared_items = []
    skipped_count = 0
    
    # 2. Prepare Tasks
    for _, row in df.iterrows():
        original_ex = row.to_dict()
        ex_id = str(original_ex.get("example_id"))
        
        # Memory Check
        user_memory = memory_map.get(ex_id)
        if not user_memory:
            skipped_count += 1
            continue

        # Filter Check
        if pref_type == "easy":
            if not original_ex.get("api_calls"): skipped_count += 1; continue
        elif pref_type in ["medium", "hard"]:
            if not original_ex.get("api_calls_pref"): skipped_count += 1; continue

        # Generate Test Pairs
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

    # 3. Create Async Tasks
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
    
    # 4. Gather Results
    processed_data = await asyncio.gather(*tasks)
    pbar.close()

    # 5. Save Output
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
    parser.add_argument("--input_path", type=str, default="/data/minseo/experiments6/data/1229_dev_6.json")
    parser.add_argument("--memory_path", type=str, required=True, help="Path to generated memory jsonl")
    parser.add_argument("--output_path", type=str, default="output_memory_eval.json")
    parser.add_argument("--log_path", type=str, default="process_memory.log")
    parser.add_argument("--query_path", type=str, default="/data/minseo/experiments6/query_singleturn.json")
    parser.add_argument("--pref_list_path", type=str, default="/data/minseo/experiments6/pref_list.json")
    parser.add_argument("--pref_group_path", type=str, default="/data/minseo/experiments6/pref_group.json")
    parser.add_argument("--tools_schema_path", type=str, default="/data/minseo/experiments6/schema_easy.json")

    # Experiment Settings
    parser.add_argument("--pref_type", type=str, choices=["medium", "easy", "hard"], required=True)
    parser.add_argument("--context_type", type=str, 
                        choices=["memory_only", "memory_diag", "memory_api"], 
                        required=True)
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--reasoning_effort", type=str, choices=["minimal", 'low', "medium", "high"], default=None)

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
            reasoning_effort=args.reasoning_effort
        )
    )
