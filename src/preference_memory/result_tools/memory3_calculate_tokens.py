import os
import json
import tiktoken
import glob

# specific directory
base_dir = "/data/minseo/experiments6/ours_memory/inference/1231_MEMORY3"
enc = tiktoken.get_encoding("cl100k_base")

def process_file(file_path):
    updated_lines = []
    total_tokens = 0
    count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                
                # Check directly for implicit_pref first, then try parsing final_implicit_preference
                implicit_pref_text = ""
                
                if "final_implicit_preference" in data:
                    fip = data["final_implicit_preference"]
                    if isinstance(fip, str):
                        try:
                            fip_json = json.loads(fip)
                            implicit_pref_text = fip_json.get("implicit_pref", "")
                        except json.JSONDecodeError:
                            # Fallback or maybe it's just a string? Unlikely based on file content sample
                            pass
                    elif isinstance(fip, dict):
                         implicit_pref_text = fip.get("implicit_pref", "")
                
                # If parsed successfully
                if implicit_pref_text:
                    tokens = enc.encode(implicit_pref_text)
                    token_count = len(tokens)
                    data["implicit_pref_token_count"] = token_count
                    total_tokens += token_count
                    count += 1
                else:
                    # If we can't find it, maybe set to 0 or skip?
                    # Based on user request, assume it exists. 
                    # If it doesn't exist, we just don't add the count or add 0.
                    data["implicit_pref_token_count"] = 0
                
                updated_lines.append(json.dumps(data))
                
            except json.JSONDecodeError:
                continue

    # Write back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in updated_lines:
            f.write(line + '\n')
            
    return total_tokens / count if count > 0 else 0

def main():
    results = {}
    
    # Iterate through directories
    for model_folder in os.listdir(base_dir):
        path = os.path.join(base_dir, model_folder)
        if os.path.isdir(path):
            jsonl_path = os.path.join(path, "_memory1.jsonl")
            if os.path.exists(jsonl_path):
                avg_tokens = process_file(jsonl_path)
                results[model_folder] = avg_tokens
    
    # Print results
    print("Average Token Counts per Model:")
    for model, avg in results.items():
        print(f"{model}: {avg:.2f}")

if __name__ == "__main__":
    main()
