import json
import random
import os

def sample_dataset(input_path, output_path, sample_size, seed=42):
    """
    JSON 데이터셋에서 지정된 개수만큼 무작위 샘플링하여 저장하는 함수
    """
    # 1. 재현성을 위해 시드 설정
    random.seed(seed)
    
    # 파일 존재 여부 확인
    if not os.path.exists(input_path):
        print(f"Error: 입력 파일 '{input_path}'을(를) 찾을 수 없습니다.")
        return

    print(f"Loading data from {input_path}...")
    
    # 2. 데이터 로드
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: JSON 형식이 올바르지 않습니다. {e}")
        return

    total_len = len(data)
    print(f"Total examples: {total_len}")

    # 3. 샘플링 개수 검증
    if sample_size > total_len:
        print(f"Warning: 요청한 샘플 수({sample_size})가 전체 데이터 수({total_len})보다 큽니다. 전체를 저장합니다.")
        sample_size = total_len

    # 4. 무작위 샘플링 (example_id 단위로 추출됨)
    sampled_data = random.sample(data, sample_size)
    print(f"Sampled {len(sampled_data)} examples.")

    # 5. 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        # indent=2: 보기 좋게 들여쓰기
        # ensure_ascii=False: 한글 및 특수문자 깨짐 방지
        json.dump(sampled_data, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully saved to {output_path}")

# --- 실행 설정 ---
if __name__ == "__main__":
    # 1. 입력 파일명 (실제 가지고 계신 파일명으로 변경)
    INPUT_FILE = "/data/minseo/experiments6/data/1229_dev_6.json"
    
    # 2. 저장할 파일명
    OUTPUT_FILE = "/data/minseo/experiments6/data/1229_dev_6_SAMPLED_100.json"
    
    # 3. 추출할 샘플 개수
    NUM_SAMPLES = 100
    
    sample_dataset(INPUT_FILE, OUTPUT_FILE, NUM_SAMPLES)