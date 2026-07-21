# step2_evaluate.py
import tqdm
import os
import json
import argparse
import copy
from datetime import datetime
from openai import OpenAI
from mem0 import MemoryClient

# utils_mem0.py에서 함수 및 템플릿 임포트
from utils_mem0 import (
    load_chains_dataset, load_query_map, assign_user_utterances,
    get_api_calls_string, get_dialogue_history_string,
    EXPLICIT_ZS_PROMPT_TEMPLATE, EXPLICIT_FS_PROMPT_TEMPLATE,
    IMPLICIT_ZS_PROMPT_TEMPLATE, IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE
)

# Initialize APIs
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
memory_client = MemoryClient(api_key=os.environ.get("MEM0_API_KEY"))

def build_input_prompt(example, current_user_utterance, mem0_results, template, context_type):
    """
    context_type에 따라 프롬프트 구성을 다르게 합니다.
    - memory_only: 메모리 + 유저 발화 (Context 없음)
    - memory_diag: 메모리 + 대화 이력 + 유저 발화
    - memory_api: 메모리 + API 호출 이력 + 유저 발화
    """
    # 1. Format Retrieved Memories
    memory_texts = []
    if mem0_results:
        for res in mem0_results:
            mem_content = res.get("memory", "")
            if mem_content:
                memory_texts.append(f"- {mem_content}")
    
    memory_str = "\n".join(memory_texts) if memory_texts else "No relevant memories found."

    # 2. Context Construction based on context_type
    api_str = get_api_calls_string(example)
    history_str = get_dialogue_history_string(example)
    
    final_context = ""
    
    if context_type == "memory_diag":
        final_context = f"--- Dialogue History ---\n{history_str}"
    elif context_type == "memory_api":
        final_context = f"--- Past API Calls ---\n{api_str}"
    elif context_type == "memory_only":
        final_context = "" 

    # 3. Inject
    try:
        prompt = template.format(
            retrieved_memories=memory_str,
            dialogue_history=final_context,
            user_utterance=current_user_utterance.strip(),
        )
    except KeyError:
        prompt = f"Context (Memories): {memory_str}\n\n"
        if final_context:
            prompt += f"{final_context}\n\n"
        prompt += f"User: {current_user_utterance.strip()}"

    return prompt

def call_gpt_api(prompt: str) -> str:
    baseline_prompt_path = "/data/minseo/personal-tool/conv_api/experiments2/mem0/new_baseline_prompt_update.txt"
    
    with open(baseline_prompt_path, "r", encoding="utf-8") as f:
        baseline_prompt = f.read()

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18", 
            messages=[
                {"role": "system", "content": baseline_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"GPT Error: {e}")
        return "API_ERROR"

def write_log(log_path: str, record: dict):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def run_evaluation(args, selected_template):
    df = load_chains_dataset(args.input_path)
    query_map = load_query_map(args.query_path)
    
    processed_data = []
    skipped_count = 0
    total_generated_cases = 0

    print(f"Starting evaluation... (Prompt: {args.prompt_type}, Context: {args.context_type}, Pref: {args.pref_type})")

    for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        original_ex = row.to_dict()
        user_id = str(original_ex.get("example_id", "unknown_user"))

        # [Logic Change] pref_type에 따른 필터링 (easy vs medium/hard)
        if args.pref_type == "easy":
            # easy는 api_calls 데이터가 있어야 함
            if not original_ex.get("api_calls"):
                skipped_count += 1; continue
        elif args.pref_type in ["medium", "hard"]:
            # medium, hard는 api_calls_pref 데이터가 있어야 함
            if not original_ex.get("api_calls_pref"):
                skipped_count += 1; continue

        # [Logic Change] assign_user_utterances 호출 방식 변경
        # 기존 use_rule_imp 플래그 대신 pref_type과 pref_group_path 전달
        pairs_list = assign_user_utterances(
            pref_list_path=args.pref_list_path, 
            example=original_ex, 
            query_map=query_map, 
            pref_type=args.pref_type,
            pref_group_path=args.pref_group_path
        )
        
        if not pairs_list:
            skipped_count += 1; continue

        # [Evaluation Loop]
        for sub_idx, (utterance, ground_truth) in enumerate(pairs_list):
            total_generated_cases += 1
            current_ex = copy.deepcopy(original_ex)
            
            # 메타 데이터 저장
            current_ex["user_utterance"] = utterance
            current_ex["reference_ground_truth"] = ground_truth 
            current_ex["example_id_sub"] = f"{user_id}_{sub_idx}"

            # --- SEARCH MEMORY ---
            search_results = memory_client.search(
                query=utterance, 
                filters={"user_id": user_id}
            )
            memories = search_results.get("results", []) if isinstance(search_results, dict) else search_results

            # --- GENERATE & CALL ---
            prompt = build_input_prompt(
                current_ex, utterance, memories, selected_template,
                context_type=args.context_type
            )
            gpt_output = call_gpt_api(prompt)

            # --- LOG & SAVE ---
            log_record = {
                "timestamp": datetime.now().isoformat(),
                "example_id": current_ex["example_id"],
                "example_id_sub": current_ex["example_id_sub"],
                "prompt_type": args.prompt_type,
                "context_type": args.context_type,
                "pref_type": args.pref_type,
                "injected_utterance": utterance,
                "retrieved_memories": [m.get('memory') for m in memories],
                "reference_ground_truth": ground_truth,
                "model_input": prompt,
                "model_output": gpt_output,
            }
            write_log(args.log_path, log_record)
            
            current_ex["gpt_output"] = gpt_output
            processed_data.append(current_ex)

    print(f"Skipped: {skipped_count}, Total Cases: {total_generated_cases}")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=4, ensure_ascii=False)
    print(f"Saved -> {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments/data/sgd_converted_dev_mapped_grouped_with_pref_with_constraints.json")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--log_path", type=str, required=True)
    parser.add_argument("--query_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments/mem0/temp_queries.json")
    parser.add_argument("--pref_list_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments/mem0/pref_list.json")
    
    # [New Argument] Preference Group File Path (for hard)
    parser.add_argument("--pref_group_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments/mem0/pref_group.json", help="Path to pref_group.json for hard mode")

    # [New Arguments] Context Type Selection
    parser.add_argument(
        "--context_type", 
        type=str, 
        choices=["memory_only", "memory_diag", "memory_api"], 
        required=True,
        help="Choose what additional context to include besides memories."
    )
    
    # [Modified Choices] Preference Type Selection (easy/medium/hard)
    parser.add_argument(
        "--pref_type", 
        type=str, 
        choices=["medium", "easy", "hard"], 
        required=True,
        help="Select preference generation mode."
    )
    
    parser.add_argument("--prompt_type", type=str, choices=["imp-zs"], default="imp-zs")

    args = parser.parse_args()

    # Template Selector
    if args.prompt_type == "imp-zs":
        template = IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE

    run_evaluation(args, template)