import json
import argparse
import os
from typing import List, Dict, Any

def parse_reasoning(raw_content: str) -> Dict[str, str]:
    """
    레퍼런스 코드의 로직을 그대로 사용하여 raw_content에서
    reasoning_content와 clean_llm_output을 분리합니다.
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

def process_file(input_path: str, output_path: str):
    if not os.path.exists(input_path):
        print(f"[Error] Input file not found: {input_path}")
        return

    print(f"Loading data from {input_path}...")
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print("[Error] Failed to parse JSON. Please check if the file is a valid JSON list.")
        return

    processed_data = []
    
    # 데이터가 리스트가 아닌 경우 리스트로 감싸서 처리
    if isinstance(data, dict):
        data = [data]

    count = 0
    for item in data:
        # 기존 llm_output 가져오기 (없으면 빈 문자열)
        original_output = item.get("llm_output", "")
        
        # 1. raw_content로 백업
        item["raw_content"] = original_output
        
        # 2. 파싱 수행
        parsed = parse_reasoning(original_output)
        
        # 3. 새로운 필드 할당 및 llm_output 덮어쓰기
        item["reasoning_content"] = parsed["reasoning_content"]
        item["llm_output"] = parsed["llm_output"]
        
        processed_data.append(item)
        count += 1

    print(f"Processed {count} items.")
    print(f"Saving to {output_path}...")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=4, ensure_ascii=False)
    
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Separate reasoning content from llm_output in JSON data.")
    
    # 기본값은 예시 경로로 설정되어 있으니 실제 경로에 맞게 수정하거나 인자로 넘겨주세요.
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input JSON file")
    parser.add_argument("--output_path", type=str, default="output_separated.json", help="Path to save the processed JSON file")

    args = parser.parse_args()

    process_file(args.input_path, args.output_path)