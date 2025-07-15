import json
import re


# Input and output paths
input_file = "llms/zero_shot/data/gpt_results/results_prompt_1.txt"
output_file = "llms/zero_shot/data/gpt_results/results_prompt_1.jsonl"

input_file = "llms/few_shot/data/gpt_results/results_prompt_2_five_ex.txt"
output_file = "llms/few_shot/data/gpt_results/results_prompt_2_five_ex.jsonl"

# Read and split the file content
with open(input_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Split into individual JSON objects using double newlines
objects = content.strip().split('\n\n')

# Write to JSONL
with open(output_file, 'w', encoding='utf-8') as out:
    for obj_str in objects:
        obj = json.loads(obj_str)
        json.dump(obj, out, ensure_ascii=False)
        out.write('\n')

print(f"Converted {len(objects)} objects to JSONL format.")