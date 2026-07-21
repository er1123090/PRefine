import tqdm
import os
import json 
import argparse
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from openai import OpenAI
import re
import copy
from mem0 import MemoryClient  # [New] mem0 import

# Initialize OpenAI API
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# [New] Initialize mem0 Client
# Ensure MEM0_API_KEY is set in your environment variables or pass it directly
memory_client = MemoryClient(api_key=os.environ.get("MEM0_API_KEY"))

# If prompt module is local, keep it; otherwise define dummy variables.
try:
    from prompt import (
        EXPLICIT_ZS_PROMPT_TEMPLATE, 
        EXPLICIT_FS_PROMPT_TEMPLATE, 
        IMPLICIT_ZS_PROMPT_TEMPLATE, 
        IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE
    )
except ImportError:
    # Dummy templates for testing - Updated to include memory section
    EXPLICIT_ZS_PROMPT_TEMPLATE = "Relevant Memories:\n{retrieved_memories}\n\nHistory:\n{dialogue_history}\nUser: {user_utterance}"
    EXPLICIT_FS_PROMPT_TEMPLATE = EXPLICIT_ZS_PROMPT_TEMPLATE
    IMPLICIT_ZS_PROMPT_TEMPLATE = EXPLICIT_ZS_PROMPT_TEMPLATE
    # Fallback memory template
    IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE = """You are an API selection assistant.
Relevant Memories (User Preferences & Constraints):
{retrieved_memories}

Dialogue History:
{dialogue_history}

User Utterance:
{user_utterance}

Output ONLY one API call.
"""

# ---------------------------------------------------------
# 1. Load Dataset & Queries
# ---------------------------------------------------------
def load_chains_dataset(fpath: str) -> pd.DataFrame:
    try:
        df = pd.read_json(fpath, lines=True)
        return df
    except ValueError:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)

def load_query_map(fpath: str) -> Dict[str, str]:
    if not os.path.exists(fpath):
        return {}
    with open(fpath, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------------------------------------------------------
# 2. Logic to Assign User Utterance
# ---------------------------------------------------------
def assign_user_utterances(pref_list_path: str, example: Dict[str, Any], query_map: Dict[str, str], use_rule_imp: bool = False) -> List[Tuple[str, str]]:
    results = []

    if use_rule_imp:
        if not os.path.exists(pref_list_path):
            return []
            
        with open(pref_list_path, "r", encoding="utf-8") as f:
            pref_list = json.load(f)

        api_calls = example.get("api_calls", [])
        if isinstance(api_calls, list):
            for call_str in api_calls:
                if "(" in call_str:
                    domain = call_str.split("(")[0].strip()
                    try:
                        args_content = call_str.split("(", 1)[1].rsplit(")", 1)[0]
                    except IndexError:
                        continue 
                else:
                    domain = call_str.strip()
                    args_content = ""

                if domain not in query_map:
                    continue
                if domain not in pref_list:
                    continue

                pattern = r'(\w+)=["\']([^"\']+)["\']'
                matches = re.findall(pattern, args_content)
                valid_slots_in_pref = pref_list[domain] 
                filtered_slots = []

                for slot, value in matches:
                    if slot in valid_slots_in_pref:
                        filtered_slots.append(f'{slot}="{value}"')

                if filtered_slots:
                    new_ground_truth = f"{domain}({', '.join(filtered_slots)})"
                    results.append((query_map[domain], new_ground_truth))
        return results

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
                if domain and domain in query_map:
                    slots_str_list = []
                    slots_str_list.append(f'{evidence["slot"]}="{evidence["value"]}"')
                    ground_truth_str = f"{domain}({', '.join(slots_str_list)})"
                    results.append((query_map[domain], ground_truth_str))
        
        return results

# ---------------------------------------------------------
# 3. Helpers for History & mem0 Conversion
# ---------------------------------------------------------
def get_api_calls_string(example: Dict[str, Any]) -> str:
    all_api_data = example.get("api_calls_all", [])
    collected_apis = []
    for item in all_api_data:
        calls = item.get("api_call", [])
        if isinstance(calls, list):
            collected_apis.extend(calls)
        elif isinstance(calls, str) and calls:
            collected_apis.append(calls)
    return "\n".join(collected_apis)

def get_dialogue_history_string(example: Dict[str, Any]) -> str:
    history_data = example.get("all_standing_instructions", [])
    sessions_str = []
    for idx, instruction_data in enumerate(history_data, start=1):
        turns = instruction_data.get("generated_dialogue", [])
        lines = [f"[Session {idx}]"]
        for turn in turns:
            role = turn.get("role", "").capitalize()
            content = turn.get("message") or turn.get("content") or ""
            if role and content:
                lines.append(f"{role}: {content}")
        sessions_str.append("\n".join(lines))
    return "\n\n".join(sessions_str)

def prepare_messages_for_mem0(example: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    [New] Converts the dataset's dialogue history into mem0 format.
    Format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    messages = []
    history_data = example.get("all_standing_instructions", [])
    
    # We flatten all sessions into a chronological list of messages for memory ingestion
    for instruction_data in history_data:
        turns = instruction_data.get("generated_dialogue", [])
        for turn in turns:
            role = turn.get("role", "").lower() # mem0 expects lowercase 'user'/'assistant'
            content = turn.get("message") or turn.get("content") or ""
            
            if role and content:
                # Map dataset roles to mem0 roles if necessary
                if role not in ["user", "assistant", "system"]:
                    role = "user" if role == "user" else "assistant"
                
                messages.append({"role": role, "content": content})
    return messages

# ---------------------------------------------------------
# 4. Build Input Prompt (Updated for mem0)
# ---------------------------------------------------------
def build_input_prompt(
    example: Dict[str, Any], 
    current_user_utterance: str, 
    mem0_results: List[Dict[str, Any]], 
    template: str, 
    include_api_call: bool = False, 
    include_both: bool = False
) -> str:
    
    # 1. Format Retrieved Memories
    memory_texts = []
    if mem0_results:
        for res in mem0_results:
            # res structure: {'id':..., 'memory': 'Allergic to nuts', 'score':...}
            mem_content = res.get("memory", "")
            if mem_content:
                memory_texts.append(f"- {mem_content}")
    
    memory_str = "\n".join(memory_texts) if memory_texts else "No relevant memories found."

    # 2. Existing Context Construction
    api_str = get_api_calls_string(example)
    history_str = get_dialogue_history_string(example)
    
    final_context = ""
    if include_both:
        final_context = f"--- Past API Calls ---\n{api_str}\n\n--- Dialogue History ---\n{history_str}"
    elif include_api_call:
        final_context = api_str
    else:
        final_context = history_str

    # 3. Inject into Template
    # Ensure your template has {retrieved_memories} placeholder, 
    # or simple concatenation if using strict templates.
    try:
        prompt = template.format(
            retrieved_memories=memory_str,
            dialogue_history=final_context,
            user_utterance=current_user_utterance.strip(),
        )
    except KeyError:
        # Fallback if template doesn't have {retrieved_memories} key
        prompt = (
            f"Context (Memories): {memory_str}\n\n" 
            + template.format(
                dialogue_history=final_context,
                user_utterance=current_user_utterance.strip()
            )
        )
        
    return prompt

# ---------------------------------------------------------
# 5. Call GPT API
# ---------------------------------------------------------
def call_gpt_api(prompt: str) -> str:
    baseline_prompt = "You are a helpful assistant. Use the provided memories and history to answer."
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18", 
            messages=[
                {"role": "system", "content": baseline_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        output = response.choices[0].message.content.strip()
        return output
    except Exception as e:
        print(f"GPT Error: {e}")
        return "API_ERROR"

# ---------------------------------------------------------
# 6. Pipeline (With mem0 Integration)
# ---------------------------------------------------------
def process_with_gpt(
    input_path: str,
    output_path: str,
    log_path: str,
    query_map_path: str,
    pref_list_path: str, 
    prompt_template: str,
    prompt_type_name: str,
    include_api_call: bool = False,
    include_both: bool = False,
    use_rule_imp: bool = False
):
    df = load_chains_dataset(input_path)
    query_map = load_query_map(query_map_path)
    
    processed_data = []
    skipped_count = 0
    total_generated_cases = 0

    mode_str = "Both (API + Dialogue)" if include_both else ("API Only" if include_api_call else "Dialogue Only")
    print(f"Starting process... (Type: {prompt_type_name}, Use Rule Imp: {use_rule_imp}, Input Content: {mode_str})")

    # Iterate through each user/example
    for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Processing Examples"):
        original_ex = row.to_dict()
        user_id = str(original_ex.get("example_id", "unknown_user")) # Treat example_id as user_id

        # --- [Step A] Pre-processing Checks ---
        if use_rule_imp:
            if not original_ex.get("api_calls"):
                skipped_count += 1
                continue
        else:
            if not original_ex.get("api_calls_pref"):
                skipped_count += 1
                continue

        pairs_list = assign_user_utterances(pref_list_path, original_ex, query_map, use_rule_imp=use_rule_imp)
        if not pairs_list:
            skipped_count += 1
            continue

        # --- [Step B] mem0 Integration: Add Memories ---
        # 1. Convert history to messages
        mem0_messages = prepare_messages_for_mem0(original_ex)
        
        # 2. Reset Memory for this user_id to ensure clean evaluation state 
        try:
            memory_client.delete_all(user_id=user_id)
        except Exception as e:
            # Ignore if user doesn't exist yet or other minor error
            pass

        # 3. Add current history to memory
        if mem0_messages:
            memory_client.add(mem0_messages, user_id=user_id)

        # --- [Step C] Process each query case ---
        for sub_idx, (utterance, ground_truth) in enumerate(pairs_list):
            total_generated_cases += 1
            
            current_ex = copy.deepcopy(original_ex)
            current_ex["user_utterance"] = utterance
            current_ex["reference_ground_truth"] = ground_truth
            current_ex["example_id_sub"] = f"{user_id}_{sub_idx}"

            # 4. Search Memory using the current query (utterance)
            # [FIXED] Updated to use 'filters' dictionary as per API spec
            search_results = memory_client.search(
                query=utterance, 
                filters={
                    "user_id": user_id
                }
            )
            
            # Extract actual result list (mem0 returns dict with 'results' key usually)
            memories = search_results.get("results", []) if isinstance(search_results, dict) else search_results

            # 5. Generate Prompt with Memories
            prompt = build_input_prompt(
                current_ex, 
                current_user_utterance=utterance,
                mem0_results=memories,  # Pass memories
                template=prompt_template, 
                include_api_call=include_api_call,
                include_both=include_both 
            )

            # 6. Call GPT
            gpt_output = call_gpt_api(prompt)

            # Log
            log_record = {
                "timestamp": datetime.now().isoformat(),
                "example_id": current_ex["example_id"],
                "example_id_sub": current_ex["example_id_sub"],
                "prompt_type": prompt_type_name,
                "input_mode": mode_str,
                "use_rule_imp": use_rule_imp,
                "injected_utterance": utterance,
                "retrieved_memories": [m.get('memory') for m in memories], # Log what was retrieved
                "reference_ground_truth": ground_truth,
                "model_input": prompt,
                "model_output": gpt_output,
            }
            write_log(log_path, log_record)

            current_ex["gpt_output"] = gpt_output
            processed_data.append(current_ex)

    print(f"Original examples skipped: {skipped_count}")
    print(f"Total test cases generated: {total_generated_cases}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=4, ensure_ascii=False)
    print(f"Saved -> {output_path}")

def write_log(log_path: str, record: Dict[str, Any]):
    dirpath = os.path.dirname(log_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# ---------------------------------------------------------
# 7. Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments/data/sgd_converted_dev_mapped_grouped_with_pref_with_constraints.json")
    parser.add_argument("--output_path", type=str, default="output_mem0.json")
    parser.add_argument("--log_path", type=str, default="process_mem0.log")
    parser.add_argument("--query_path", type=str, default="temp_queries.json")
    parser.add_argument("--pref_list_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments/pref_list.json")
    
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--api_call", action="store_true", help="Use ONLY the aggregated API calls.")
    input_group.add_argument("--include_both", action="store_true", help="Use BOTH API calls AND Dialogue History.")
    
    parser.add_argument("--rule_imp_pref", action="store_true", help="Rule based preference parsing.")
    parser.add_argument("--prompt_type", type=str, choices=["exp-zs", "exp-fs", "imp-zs", "imp-fs", "mem0"], default="mem0")

    args = parser.parse_args()

    # Select template (Templates now expect {retrieved_memories})
    if args.prompt_type == "exp-zs":
        selected_template = EXPLICIT_ZS_PROMPT_TEMPLATE
    elif args.prompt_type == "exp-fs":
        selected_template = EXPLICIT_FS_PROMPT_TEMPLATE
    elif args.prompt_type == "imp-zs":
        selected_template = IMPLICIT_ZS_PROMPT_TEMPLATE
    elif args.prompt_type == "mem0":
        selected_template = IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE
    else:
        # Default fallback
        selected_template = IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE

    process_with_gpt(
        input_path=args.input_path,
        output_path=args.output_path,
        log_path=args.log_path,
        query_map_path=args.query_path,
        pref_list_path=args.pref_list_path,
        prompt_template=selected_template,
        prompt_type_name=args.prompt_type,
        include_api_call=args.api_call,
        include_both=args.include_both, 
        use_rule_imp=args.rule_imp_pref
    )