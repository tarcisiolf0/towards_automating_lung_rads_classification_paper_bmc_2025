import json
import re

input_filename = "llms/zero_shot/data/llama_results/results_prompt_2.txt"
output_filename = "llms/zero_shot/data/llama_results/results_prompt_2.jsonl"

# input_filename = "llms/few_shot/data/llama_results/results_prompt_2_five_ex.txt"
# output_filename = "llms/few_shot/data/llama_results/results_prompt_2_five_ex.jsonl"

with open(input_filename, "r", encoding="utf-8") as infile, \
    open(output_filename, "w", encoding="utf-8") as outfile:
    for line in infile:
        try:
            # Attempt to load each line as a single JSON object
            data = json.loads(line.strip())
            # Write the JSON object to the output file, followed by a newline
            json.dump(data, outfile, ensure_ascii=False)
            outfile.write("\n")
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON line: {line.strip()}")