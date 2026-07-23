import tqdm
import os
import json
import argparse
import sys
from pathlib import Path
try:
    import pandas as pd
except ImportError:
    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from src.exp4_runtime import tabular as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
import re
import copy
import asyncio
import itertools
from openai import AsyncOpenAI

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from src.token_measurement import count_text_tokens
try:
    from mem0 import MemoryClient
except ImportError:
    MemoryClient = None

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
        return json.load(f)

# ---------------------------------------------------------
# [New] Reasoning Parsing Logic
# ---------------------------------------------------------
def parse_deepseek_reasoning(raw_content: str) -> Dict[str, str]:
    """
    Parses <think> tags for models serving raw reasoning strings (e.g. vLLM DeepSeek).
    """
    if not raw_content:
        return {"llm_output": "", "reasoning_content": ""}

    end_tag = "</think>"
    end_idx = raw_content.rfind(end_tag)

    if end_idx != -1:
        # </think> exists: split reasoning and content
        reasoning_part = raw_content[:end_idx]
        # Remove start tag if present
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

# ---------------------------------------------------------
# 2. Logic to Assign User Utterance (Preserved from Code 1)
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
# 3. Memory Formatting & Prompt Building (Updated for mem0)
# ---------------------------------------------------------
def format_api_calls(api_calls: List[str]) -> str:
    if not api_calls or not isinstance(api_calls, list): return "None"
    return "\n".join(api_calls)

def format_retrieved_memory_text(retrieved_memories: List[Dict]) -> str:
    formatted_memories = []
    for mem in retrieved_memories or []:
        content = mem.get("memory", "") if isinstance(mem, dict) else ""
        if content:
            formatted_memories.append(f"- {content}")
    return (
        "\n".join(formatted_memories)
        if formatted_memories
        else "No relevant memories found."
    )

def build_mem0_input_prompt(
    example: Dict[str, Any],
    retrieved_memories: List[Dict], # List from mem0 search results
    current_user_utterance: str,
    template: str,
    context_type: str,
    tools_schema: List[Dict] = None
) -> str:
    
    # 1. Format Memories
    memories_str = format_retrieved_memory_text(retrieved_memories)

    # 2. Format Dialogue History
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

    # 3. Format API History (if needed)
    raw_api_calls = example.get("api_calls") or [] # Original dataset doesn't have 'accumulated' like memory file, using dataset field
    api_history_str = format_api_calls(raw_api_calls)

    # 4. Construct Context Blocks
    dialogue_history_input = ""
    
    if context_type == "memory_only":
        dialogue_history_input = "None"
    elif context_type == "memory_api":
        dialogue_history_input = f"[Past API History]:\n{api_history_str}"
    elif context_type == "memory_diag":
        dialogue_history_input = dialogue_text
    else: 
        dialogue_history_input = dialogue_text

    schema_str = json.dumps(tools_schema, indent=2, ensure_ascii=False) if tools_schema else "No specific schema provided."

    # 5. Final Prompt
    prompt = template.format(
        preference_schema=schema_str,
        retrieved_memories=memories_str,
        dialogue_history=dialogue_history_input,
        user_utterance=current_user_utterance.strip()
    )
    return prompt

# ---------------------------------------------------------
# 4. Async LLM Call (Preserved)
# ---------------------------------------------------------
async def call_llm_api_async(
    prompt: str, 
    model_name: str, 
    openai_client: AsyncOpenAI = None, 
    tools_schema: List[Dict] = None,
    reasoning_effort: str = None
) -> Dict[str, Any]:
    """
    Returns a dict with keys: 'content', 'reasoning', 'token_counts', 'error'
    """
    result = {
        "content": "",
        "reasoning": "",
        "token_counts": {},
        "error": None
    }

    try:
        # --- GEMINI ---
        if "gemini" in model_name.lower():
            if not os.environ.get("GOOGLE_API_KEY"):
                return {"error": "API_KEY_MISSING_GOOGLE"}
            
            client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
            config_params = {"temperature": 0.0}

            if reasoning_effort and ("gemini" in model_name.lower()):
                config_params["thinking_config"] = types.ThinkingConfig(
                    include_thoughts=True,
                    thinking_level=reasoning_effort.lower()
                )

            conf = types.GenerateContentConfig(**config_params)
            response = await client.aio.models.generate_content(
                model=model_name, contents=prompt, config=conf
            )
            
            # Extract Content & Reasoning
            thought_parts = []
            content_parts = []
            if response.candidates and response.candidates[0].content:
                for part in response.candidates[0].content.parts:
                    if not part.text: continue
                    if hasattr(part, 'thought') and part.thought:
                        thought_parts.append(part.text)
                    else:
                        content_parts.append(part.text)
            
            result["content"] = "\n".join(content_parts).strip()
            result["reasoning"] = "\n".join(thought_parts).strip()
            
            # Extract Token Counts
            if response.usage_metadata:
                result["token_counts"] = {
                    "total_tokens": response.usage_metadata.total_token_count,
                    "input_tokens": response.usage_metadata.prompt_token_count,
                    "output_tokens": response.usage_metadata.candidates_token_count,
                    "reasoning_tokens": getattr(response.usage_metadata, 'thoughts_token_count', 0)
                }
            return result

        # --- OPENAI / vLLM ---
        else:
            if not openai_client:
                return {"error": "API_KEY_MISSING_OPENAI"}
            
            kwargs = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
            }
            
            is_reasoning_model = any(k in model_name.lower() for k in ["o1", "o3", "gpt-5"])
            if is_reasoning_model and reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort.lower()
            else:
                pass # kwargs["temperature"] = 0.0

            response = await openai_client.chat.completions.create(**kwargs)
            message = response.choices[0].message
            raw_content = message.content or ""

            # 1. Attempt to get reasoning from message (DeepSeek API standard)
            reasoning_text = getattr(message, 'reasoning_content', "")
            
            # 2. If empty, parse <think> tags (vLLM/DeepSeek raw)
            if not reasoning_text:
                parsed = parse_deepseek_reasoning(raw_content)
                result["content"] = parsed["llm_output"]
                result["reasoning"] = parsed["reasoning_content"]
            else:
                result["content"] = raw_content
                result["reasoning"] = reasoning_text

            # 3. Token Counts (handling o1/o3 specifics)
            if response.usage:
                total_tokens = response.usage.total_tokens
                reasoning_tokens = 0
                
                details = getattr(response.usage, 'completion_tokens_details', None)
                if not details:
                    details = getattr(response.usage, 'output_tokens_details', None)
                
                if details:
                    reasoning_tokens = getattr(details, 'reasoning_tokens', 0)
                
                result["token_counts"] = {
                    "total_tokens": total_tokens,
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                    "reasoning_tokens": reasoning_tokens
                }

            return result

    except Exception as e:
        print(f"LLM API Error ({model_name}): {e}")
        return {"error": f"API_ERROR: {str(e)}"}

# ---------------------------------------------------------
# 5. Async Processing Pipeline (Updated for mem0)
# ---------------------------------------------------------
async def process_single_item(
    original_ex: Dict[str, Any],
    memory_client: MemoryClient, # Changed: Passed memory_client instance
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
        user_id = str(original_ex.get("example_id", "unknown_user"))
        
        # 0. Retrieve from mem0 (Run in executor because mem0 client is sync)
        # Note: mem0 search usually returns {'results': [...]} or a list directly depending on version
        loop = asyncio.get_running_loop()
        try:
            # [FIXED] Added filters={"user_id": user_id} to satisfy API requirements
            search_response = await loop.run_in_executor(
                None, 
                lambda: memory_client.search(
                    query=utterance, 
                    user_id=user_id,
                    filters={"user_id": user_id} 
                )
            )
            # Normalize response to list of dicts
            retrieved_memories = (
                search_response.get("results", [])
                if isinstance(search_response, dict)
                else search_response
            )
            retrieved_memories = [
                memory
                for memory in (retrieved_memories or [])
                if isinstance(memory, dict)
            ]
        except Exception as e:
            print(f"Mem0 Error for {user_id}: {e}")
            retrieved_memories = []

        retrieved_memory_text = format_retrieved_memory_text(retrieved_memories)
        retrieval_prompt_tokens = count_text_tokens(retrieved_memory_text)
        retrieved_memory_tokens = (
            retrieval_prompt_tokens if retrieved_memories else 0
        )

        # 1. Build Prompt with retrieved memories
        prompt = build_mem0_input_prompt(
            example=original_ex,
            retrieved_memories=retrieved_memories,
            current_user_utterance=utterance,
            template=prompt_template,
            context_type=context_type,
            tools_schema=tools_schema
        )

        # 2. Call LLM (Returns Dictionary now)
        llm_result = await call_llm_api_async(
            prompt, model_name, openai_client, tools_schema, reasoning_effort
        )

        # 3. Unpack Result
        if llm_result.get("error"):
            clean_output = llm_result["error"]
            reasoning_content = ""
            token_counts = {}
        else:
            clean_output = llm_result["content"]
            reasoning_content = llm_result["reasoning"]
            token_counts = llm_result["token_counts"]

        # 4. Logging
        log_record = {
            "timestamp": datetime.now().isoformat(),
            "example_id": original_ex.get("example_id"),
            "example_id_sub": f"{original_ex.get('example_id')}_{sub_idx}",
            "model_name": model_name,
            "method": "mem0",
            "context_type": context_type,
            "pref_type": "implicit_mem0",
            "injected_utterance": utterance,
            "retrieved_memories": [m.get("memory") for m in retrieved_memories],
            "retrieved_memory_count": len(retrieved_memories),
            "retrieved_memory_tokens": retrieved_memory_tokens,
            "retrieval_prompt_tokens": retrieval_prompt_tokens,
            "retrieval_token_encoding": "cl100k_base",
            "reference_ground_truth": ground_truth,
            "model_input": prompt,
            "model_output": clean_output,
            "reasoning_content": reasoning_content,
            "token_counts": token_counts
        }
        
        async with file_lock:
            dirpath = os.path.dirname(log_path)
            if dirpath: os.makedirs(dirpath, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_record, ensure_ascii=False) + "\n")

        # 5. Update Return Object
        result_ex = copy.deepcopy(original_ex)
        result_ex["example_id_sub"] = f"{original_ex.get('example_id')}_{sub_idx}"
        result_ex["method"] = "mem0"
        result_ex["test_utterance"] = utterance
        result_ex["reference_ground_truth"] = ground_truth
        result_ex["retrieved_memories"] = retrieved_memories
        result_ex["retrieved_memory_count"] = len(retrieved_memories)
        result_ex["retrieved_memory_tokens"] = retrieved_memory_tokens
        result_ex["retrieval_prompt_tokens"] = retrieval_prompt_tokens
        result_ex["retrieval_token_encoding"] = "cl100k_base"
        
        result_ex["llm_output"] = clean_output
        result_ex["reasoning_content"] = reasoning_content
        result_ex["token_counts"] = token_counts
        
        pbar.update(1)
        return result_ex

async def process_with_llm_async(
    input_path: str, output_path: str, log_path: str, 
    query_map_path: str, pref_list_path: str, pref_group_path: str,
    tools_schema_path: str,
    prompt_template: str, context_type: str, pref_type: str,
    model_name: str, concurrency: int = 10,
    reasoning_effort: str = None 
):
    df = load_chains_dataset(input_path)
    query_map = load_query_map(query_map_path)
    tools_schema = load_tools_from_file(tools_schema_path)
    
    # Initialize Clients
    openai_client = None
    if os.environ.get("OPENAI_API_KEY"):
        openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Initialize mem0 Client
    if MemoryClient is None:
        raise RuntimeError("mem0ai is required for Mem0 inference")
    memory_client = MemoryClient(api_key=os.environ.get("MEM0_API_KEY"))

    print(f"Starting ASYNC process... (Model: {model_name}) | Reasoning: {reasoning_effort} | Memory: mem0")

    tasks = []
    semaphore = asyncio.Semaphore(concurrency)
    file_lock = asyncio.Lock()
    prepared_items = []
    skipped_count = 0
    
    for _, row in df.iterrows():
        original_ex = row.to_dict()
        
        if pref_type == "easy":
            if not original_ex.get("api_calls"): skipped_count += 1; continue
        elif pref_type in ["medium", "hard"]:
            if not original_ex.get("api_calls_pref"): skipped_count += 1; continue

        # Generate Utterance/GT Pairs using logic from Code 1
        pairs_list = assign_user_utterances(
            pref_list_path, original_ex, query_map, pref_type, pref_group_path
        )
        
        if not pairs_list:
            skipped_count += 1
            continue

        for sub_idx, (utterance, ground_truth) in enumerate(pairs_list):
            prepared_items.append({
                "original_ex": original_ex,
                "utterance": utterance,
                "ground_truth": ground_truth,
                "sub_idx": sub_idx
            })

    total_tasks = len(prepared_items)
    print(f"Total tasks: {total_tasks}. Skipped dialogues: {skipped_count}")

    pbar = tqdm.tqdm(total=total_tasks, desc="Processing Async")

    for item in prepared_items:
        task = asyncio.create_task(
            process_single_item(
                original_ex=item['original_ex'],
                memory_client=memory_client, # Pass client here
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
    parser.add_argument("--input_path", type=str, default="/data/minseo/experiment8/data/MPT_v2_mix600.json")
    # memory_path REMOVED as we utilize mem0 API
    parser.add_argument("--output_path", type=str, default="output_mem0_eval.json")
    parser.add_argument("--log_path", type=str, default="process_mem0.log")
    parser.add_argument("--query_path", type=str, default="/data/minseo/experiment8/config/query_singleturn_hint.json")
    parser.add_argument("--pref_list_path", type=str, default="/data/minseo/experiment8/config/pref_list.json")
    parser.add_argument("--pref_group_path", type=str, default="/data/minseo/experiment8/config/pref_group.json")
    parser.add_argument("--tools_schema_path", type=str, default="/data/minseo/experiment8/config/schema_easy.json")

    # Experiment Settings
    parser.add_argument("--pref_type", type=str, choices=["medium", "easy", "hard"], required=True)
    parser.add_argument("--context_type", type=str, 
                        choices=["memory_only", "memory_diag", "memory_api"], 
                        required=True)
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--reasoning_effort", type=str, choices=["minimal", 'low', "medium", "high"], default=None)

    args = parser.parse_args()

    # Ensure API Keys are set
    if not os.environ.get("MEM0_API_KEY"):
        print("[Error] MEM0_API_KEY environment variable is not set.")
        exit(1)

    asyncio.run(
        process_with_llm_async(
            input_path=args.input_path,
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
