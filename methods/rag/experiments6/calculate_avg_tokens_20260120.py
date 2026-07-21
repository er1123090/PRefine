import json
import glob
import tiktoken
import numpy as np

def count_tokens(text, enc):
    return len(enc.encode(text))

def main():
    target_pattern = "experiments6/RAG/inference_*/gemini-3-flash/*20260120*.log"
    files = glob.glob(target_pattern)
    
    if not files:
        print(f"No files found matching the pattern: {target_pattern}")
        debug_files = glob.glob("experiments6/RAG/**/*20260120*.log", recursive=True)
        if debug_files:
            print(f"Found files at: {debug_files}")
        return

    try:
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        print(f"Error loading tiktoken: {e}")
        return
    
    all_context_tokens = []

    print(f"{'File Path':<80} | {'Avg Tokens':<15} | {'Count':<10}")
    print("-" * 115)

    for file_path in sorted(files):
        context_tokens = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        content_to_measure = None
                        
                        # Check retrieved_context field first
                        if "retrieved_context" in data and data["retrieved_context"]:
                            content_to_measure = data["retrieved_context"]
                        
                        # Fallback to model_input if retrieved_context is missing
                        elif "model_input" in data:
                            model_input = data["model_input"]
                            header = "--- Relevant Past Context (Retrieved) ---"
                            if header in model_input:
                                parts = model_input.split(header)
                                if len(parts) > 1:
                                    section = parts[1]
                                    end_marker = "User Utterance:"
                                    if end_marker in section:
                                        content_to_measure = section.split(end_marker)[0].strip()
                                    else:
                                        content_to_measure = section.strip()

                        if content_to_measure:
                            tokens = count_tokens(content_to_measure, enc)
                            context_tokens.append(tokens)
                            
                    except json.JSONDecodeError: 
                        continue
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

        if context_tokens:
            avg = np.mean(context_tokens)
            print(f"{file_path:<80} | {avg:<15.2f} | {len(context_tokens):<10}")
            all_context_tokens.extend(context_tokens)
        else:
            print(f"{file_path:<80} | {'N/A':<15} | {0:<10}")

    print("-" * 115)
    if all_context_tokens:
        overall_avg = np.mean(all_context_tokens)
        print(f"Overall Average Token Count: {overall_avg:.2f}")
        print(f"Total samples processed: {len(all_context_tokens)}")
    else:
        print("No relevant context found in any files.")

if __name__ == "__main__":
    main()
