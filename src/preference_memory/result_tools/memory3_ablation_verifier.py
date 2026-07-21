import json
import os

def process_jsonl_to_json(input_file_path, output_file_path):
    """
    JSONL 파일을 읽어서 is_valid가 false인 항목만 필터링하고,
    verifier_input 필드를 제거한 뒤 표준 JSON 포맷(Array)으로 저장하는 함수
    """
    
    filtered_data = [] # 데이터를 모을 리스트 생성
    count = 0
    
    try:
        # 입력 파일 읽기
        with open(input_file_path, 'r', encoding='utf-8') as infile:
            for line in infile:
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    
                    # is_valid가 False인 경우만 처리
                    if data.get('is_valid') is False:
                        # verifier_input 필드 제거
                        data.pop('verifier_input', None)
                        
                        # 리스트에 추가
                        filtered_data.append(data)
                        count += 1
                        
                except json.JSONDecodeError:
                    print(f"JSON 파싱 에러 발생: {line[:50]}...")
                    continue
        
        # 출력 파일 쓰기 (표준 JSON 형식)
        # 디렉토리가 없으면 생성
        output_dir = os.path.dirname(output_file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            # indent=4 옵션을 주어 보기 좋게 정렬 (용량이 걱정되면 indent 옵션 제거)
            json.dump(filtered_data, outfile, ensure_ascii=False, indent=4)

        print(f"작업 완료! 총 {count}개의 항목이 저장되었습니다.")
        print(f"저장된 파일: {output_file_path}")

    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {input_file_path}")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")

# 실행 부분
if __name__ == "__main__":
    # 입력 파일 경로
    INPUT_FILE = "/data/minseo/experiments6/ours_memory/inference/1231_MEMORY3/gpt-4o-mini/_verifier_logs1.jsonl"
    
    # 출력 파일 경로 (확장자를 .json으로 변경하는 것을 권장)
    OUTPUT_FILE = "/data/minseo/experiments6/ours_memory/inference/1231_MEMORY3/gpt-4o-mini/_verifier_logs1_ablation.json" 

    process_jsonl_to_json(INPUT_FILE, OUTPUT_FILE)