import google.generativeai as genai
import os
import time
import process_files as pf
import os
import logging
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
def gemini_req(api_key, inputs, prompts):
    results = []
    total_exec_time = 0

    genai.configure(api_key=api_key)

    for i in range(len(inputs)):     
        input = inputs[i]

        prompt = prompts[i]

        model = genai.GenerativeModel('gemini-1.5-flash', generation_config={"response_mime_type": "application/json", "temperature" : 0})

        start_time = time.time()   

        final_prompt = ''
        final_prompt = input+"\n\n"+prompt

        response = model.generate_content(final_prompt)
        text_response = response.text
        logging.info(f"Response: {text_response}")

        ## Tempo 
        end_time = time.time()
        exec_time = end_time - start_time
        total_exec_time += exec_time
        
        pf.print_execution_stats(i, exec_time, total_exec_time)

        # Sleep to avoid hitting the rate limit
        time.sleep(5)
        
        results.append(text_response)
        
    return results


if __name__ == '__main__':
    # Load environment variables
    load_dotenv()
    # Configure the API key
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    # Prompt 1 - 5 Examples
    inputs = pf.read_input_file("llms/few_shot/data/inputs.txt")
    inputs = pf.pre_process_input_file(inputs)

    #prompts_one_five_ex = pf.read_input_file("llms/few_shot/data/prompt_1_five_ex.txt")
    #prompts_one_five_ex = pf.pre_process_input_file(prompts_one_five_ex)

    #results_one_five_ex = gemini_req(gemini_api_key, inputs, prompts_one_five_ex)
    #pf.write_output_file("llms/few_shot/data/gemini_results/results_prompt_1_five_ex.txt", results_one_five_ex)

    # Prompt 2 - 5 Examples
    prompts_two_five_ex = pf.read_input_file("llms/few_shot/data/prompt_2_five_ex.txt")
    prompts_two_five_ex = pf.pre_process_input_file(prompts_two_five_ex)

    results_two_five_ex = gemini_req(gemini_api_key, inputs, prompts_two_five_ex)
    pf.write_output_file("llms/few_shot/data/gemini_results/results_prompt_2_five_ex.txt", results_two_five_ex)

