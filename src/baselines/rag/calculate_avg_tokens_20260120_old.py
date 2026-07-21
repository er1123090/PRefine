import os
import json
import glob
import tiktoken
import numpy as np

def count_tokens(text, enc):
    return len(enc.encode(text))

def main():
    target_pattern = "data/minseo/experiments6/RAG/inference_*/gemini-3-flash/*20260120*.log"
    files = glob.glob(target_pattern)
    
    if not files:
        print("No files found matching the pattern.")
        return

    enc = tiktoken.get_encoding("cl100k_base")
    
    all_context_tokens = []
    file_stats = {}

    for file_path in files:
        print(f"Processing: {file_path}")
        context_tokens = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        # Check for retrieved_context directly
                        if "retrieved_context" in data:
                            tokens = count_tokens(data["retrieved_context"], enc)
                            context_tokens.append(tokens)
                        else:
                             # Fallback: check model_input for the section
                            model_input = data.get("model_input", "")
                            if "Relevant Past Context (Retrieved)" in model_input:
                                # Extract content between the header and next section or end
                                start_marker = "--- Relevant Past Context (Retrieved) ---"
                                start_idx = model_input.find(start_marker)
                                if start_idx != -1:
                                    content_start = start_idx + len(start_marker)
                                    # Find next potential section header
                                    end_idx = model_input.find("\n\n", content_start) # Simple heuristic end
                                    if end_idx == -1:
                                        content = model_input[content_start:]
                                    else:
                                        # Heuristic: look for next major section if structured
                                        next_section = model_input.find("\nUser Utterance:", content_start) 
                                        if next_section != -1:
                                            content = model_input[content_start:next_section]
                                        else:
                                            # Default to rest of string or reasonable chunk
                                            content = model_input[content_start:]
                                    
                                    tokens = count_tokens(content.strip(), enc)
                                    context_tokens.append(tokens)
                    except json.JSONDecodeError: 
                        continue # Skip invalid lines
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            continue

        if context_tokens:
            avg = np.mean(context_tokens)
            file_stats[file_path] = avg
            all_context_tokens.extend(context_tokens)
            print(f"  - Average tokens: {avg:.2f} (count: {len(context_tokens)})")
        else:
             print(f"  - No relevant context found.")

    if all_context_tokens:
        overall_avg = np.mean(all_context_tokens)
        print("\nOverall Average Token Count for 'Relevant Past Context (Retrieved)':")
        print(f"{overall_avg:.2f}")
    else:
        print("\nNo context tokens collected across files.")

if __name__ == "__main__":
    main()
