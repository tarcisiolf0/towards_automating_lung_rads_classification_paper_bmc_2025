import time
from together import Together
import process_files as pf
from dotenv import load_dotenv
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def llama_req(api_key, inputs, prompt):
    results = []
    total_exec_time = 0

    for i in range(len(inputs)):
        input = inputs[i]      

        client = Together(api_key=api_key)

        start_time = time.time()

        response = client.chat.completions.create(
            model="meta-llama/Llama-3-70b-chat-hf",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": input}
            ],
            temperature=0
        )

        text_response = response.choices[0].message.content
        logging.info(f"Response: {text_response}")

        ## Tempo 
        end_time = time.time()
        exec_time = end_time - start_time
        total_exec_time += exec_time
        
        pf.print_execution_stats(i, exec_time, total_exec_time)
        
        print(f"{text_response}")
        results.append(text_response)

    return results
    

if __name__ == '__main__':
    # Load environment variables
    load_dotenv()
    # Configure the API key
    together_api_key = os.getenv("TOGETHER_API_KEY")

   # Prompt 1
    inputs = pf.read_input_file("llms/zero_shot/data/inputs.txt")
    prompt_1 = pf.read_prompt_file("llms/zero_shot/data/prompt_1.txt")

    inputs = pf.pre_process_input_file(inputs)

    results_1 = llama_req(together_api_key, inputs, prompt_1)
    pf.write_output_file("llms/zero_shot/data/llama_results/results_prompt_1.txt", results_1)

    #Prompt 2
    prompt_2 = pf.read_prompt_file("llms/zero_shot/data/prompt_2.txt")
    results_2 = llama_req(together_api_key, inputs, prompt_2)
    pf.write_output_file("llms/zero_shot/data/llama_results/results_prompt_2.txt", results_2)