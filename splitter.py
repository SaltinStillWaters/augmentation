import json
import random

def sample_jsonl(input_path, output_path, percentage, seed=42):
    random.seed(seed)

    # Load all lines from the input file
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = [json.loads(line) for line in f]

    # Shuffle and sample
    sample_size = int(len(lines) * percentage)
    sampled = random.sample(lines, sample_size)

    # Write sampled lines to output
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in sampled:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f">> Sampled {sample_size} of {len(lines)} lines to: {output_path}")

# Example usage
sample_jsonl('augmented/semantic/50.jsonl', 'augmented/semantic/10.jsonl', percentage=0.20)