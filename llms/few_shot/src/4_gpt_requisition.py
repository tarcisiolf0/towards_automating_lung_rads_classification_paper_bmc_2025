import time
from openai import OpenAI
import process_files as pf
import os
import logging
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def gpt_req(api_key, inputs, prompts):
    results = []
    total_exec_time = 0

    for i in range(len(inputs)):
        
        prompt = prompts[i]
        input = inputs[i]

        client = OpenAI(api_key=api_key)

        start_time = time.time()

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": input}
            ],
            seed=42,
            temperature=0
        )

        text_response = response.choices[0].message.content
        logging.info(f"Response: {text_response}")

        ## Tempo 
        end_time = time.time()
        exec_time = end_time - start_time
        total_exec_time += exec_time
        
        pf.print_execution_stats(i, exec_time, total_exec_time)
    
        results.append(text_response)

    return results
    

if __name__ == '__main__':
    # Load environment variables
    load_dotenv()
    # Configure the API key
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Prompt 1 - 5 Examples
    inputs = pf.read_input_file("llms/few_shot/data/inputs.txt")
    inputs = pf.pre_process_input_file(inputs)

    prompts_one_five_ex = pf.read_input_file("llms/few_shot/data/prompt_1_five_ex.txt")

    prompts_one_five_ex = pf.pre_process_input_file(prompts_one_five_ex)

    results_one_five_ex = gpt_req(openai_api_key, inputs, prompts_one_five_ex)
    pf.write_output_file("llms/few_shot/data/gpt_results/results_prompt_1_five_ex.txt", results_one_five_ex)

    # Prompt 2 - 5 Examples
    prompts_two_five_ex = pf.read_input_file("llms/few_shot/data/prompt_2_five_ex.txt")

    prompts_two_five_ex = pf.pre_process_input_file(prompts_two_five_ex)

    results_two_five_ex = gpt_req(openai_api_key, inputs, prompts_two_five_ex)
    pf.write_output_file("llms/few_shot/data/gpt_results/results_prompt_2_five_ex.txt", results_two_five_ex)
