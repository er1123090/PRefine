import json
import tiktoken
import numpy as np

def count_tokens(text, encoding):
    return len(encoding.encode(text))

def main():
    file_path = '/data/minseo/experiments4/data/1229_dev_6.json'
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return

    # Use cl100k_base encoding (common for GPT-3.5/4)
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        print(f"Error loading tiktoken encoding: {e}")
        return

    example_token_counts = []

    print(f"Processing {len(data)} examples...")

    for example in data:
        current_example_tokens = 0
        
        # Iterate through sessions if present (structure based on file snippet)
        sessions = example.get('sessions', [])
        
        for session in sessions:
            dialogue = session.get('dialogue', [])
            for turn in dialogue:
                message = turn.get('message', "")
                if message:
                    current_example_tokens += count_tokens(message, encoding)
        
        example_token_counts.append(current_example_tokens)

    if not example_token_counts:
        print("No examples found or processed.")
        return

    avg_tokens = np.mean(example_token_counts)
    median_tokens = np.median(example_token_counts)
    max_tokens = np.max(example_token_counts)
    min_tokens = np.min(example_token_counts)

    print(f"\nResults for {file_path}")
    print(f"Total examples: {len(example_token_counts)}")
    print(f"Average tokens per example_id: {avg_tokens:.2f}")
    print(f"Median tokens per example_id: {median_tokens:.2f}")
    print(f"Max tokens per example_id: {max_tokens}")
    print(f"Min tokens per example_id: {min_tokens}")

if __name__ == "__main__":
    main()
