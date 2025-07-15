import json
import re

input_filename = "llms/zero_shot/data/gemini_results/results_prompt_2.txt"
output_filename = "llms/zero_shot/data/gemini_results/results_prompt_2.jsonl"

input_filename = "llms/few_shot/data/gemini_results/results_prompt_1_five_ex.txt"
output_filename = "llms/few_shot/data/gemini_results/results_prompt_1_five_ex.jsonl"

# # Read the file line by line
# with open(input_filename, "r", encoding="utf-8") as file:
#     lines = file.readlines()

# # Process each line
# with open(output_filename, "w", encoding="utf-8") as output_file:
#     for line in lines:
#         json_objects = re.findall(r'\{.*?\}', line)  # Extract JSON objects from the line
        
#         if len(json_objects) == 3:  # Ensure there are exactly three objects
#             grouped_array = [json.loads(obj) for obj in json_objects]  # Convert to JSON objects
            
#             # Write as a JSON array per line
#             output_file.write(json.dumps(grouped_array, ensure_ascii=False) + "\n")

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