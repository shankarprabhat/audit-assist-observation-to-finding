import requests
import pandas as pd
import traceback
import os
from dotenv import load_dotenv
import time

load_dotenv()  # Load variables from .env into the environment

llm_selection = "local_ollama"
# llm_selection = "hugging-face-qwen"

if llm_selection == 'flan-t5-large':
    token = os.environ.get("HF_TOKEN")
    TOKEN = "Bearer " + token
    headers = {"Authorization": TOKEN}
    base_llm_model_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"

elif llm_selection == 'hugging-face-qwen':
    base_llm_model_URL = "https://ytqej0aypbclmmsl.us-east-1.aws.endpoints.huggingface.cloud/v1/chat/completions"
    my_hf_model_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    headers = {
      'Authorization': 'Bearer ' + my_hf_model_token,
      'Content-Type': 'application/json'
    }

elif llm_selection == 'local_ollama':
    model_name = "qwen3:0.6b"
    model_name = "gemma3:1b"
    # model_name = "deepseek-r1:1.5b"
    base_llm_model_URL = "http://localhost:11434/api/generate" # For single-turn completion
    headers = {"Content-Type": "application/json"}
    print('select local ollama model')

def run_inference_query(payload):
    response = requests.post(base_llm_model_URL, headers=headers, json=payload)
    if response.status_code != 200:
        # raise Exception(f"Request failed with status code {response.status_code}: {response.text}")
        return "Request failed with status code {response.status_code}: {response.text}", response.status_code
    # Check if the response is JSON
    return response.json(), response.status_code

# 1. Load your CSV data (replace 'your_data.csv' with your file path)
def prepare_data():

    df = pd.read_csv('Audit Observations-Findings.csv')
    df = df.dropna()

    # Assuming observations are in column 2 and findings are in column 3 (index 0,1,2..)
    observations = df.iloc[:, 1].tolist()
    findings = df.iloc[:, 2].tolist()

    examples = []

    for ii in range(0,23):
        text = {"observation": observations[ii], "finding": findings[ii]}
        examples = examples + [text]


    # Format the examples into a string
    example_string = ""
    for example in examples:
        example_string += f"Observation: {example['observation']}\nFinding: {example['finding']}\n\n"

    return example_string

def prepare_context(input_payload):
    document_name = input_payload['document_name']
    doc_section = input_payload['document_section']

    obtained_context = "GCP: 4.1.1. The investigator(s) should be qualified by education, training," \
    " and experience to assume responsibility for the proper conduct of the trial, should meet all" \
    " the qualifications specified by the applicable regulatory requirement(s), and should provide " \
    "evidence of such qualifications through up-to-date curriculum vitae and/or other relevant" \
    " documentation requested by the sponsor, the IRB/IEC, and/or the regulatory authority(ies)." \
    " GCP: 4.2.3. The investigator should have available an adequate number of qualified staff" \
    " and adequate facilities for the foreseen duration of the trial to conduct the trial properly" \
    " and safely."

    return obtained_context

def prepare_prompt(example_string, input_observation, input_context):

    # Create the prompt
    promt_question = "You are an Auditor for analyzing clinical trials reports. You are being provided" \
    " an example of audit observation and its context, and are required to provide the finding for the real observation. One"\
    " example is given. Provide the findings for real observation using the real context provided."\
    " Try to find the implications of the observations on the clinical trial."

    prompt = f"{promt_question}:\n\n{example_string}\nReal Observation: {input_observation}\n Real Context: {input_context}"

    return prompt


# 1. Load your CSV data (replace 'your_data.csv' with your file path)
def prepare_example_data():
    """This code prepares the example data to be sent in the prompt"""

    df = pd.read_csv('Audit Observations-Findings_with_context.csv')

    # Assuming observations are in column 2 and findings are in column 3 (index 0,1,2..)
    observations = df.iloc[1,1]
    context = df.iloc[1,4]
    findings = df.iloc[1,5]

    examples = []

    text = {"observation": observations, "context": context, "finding": findings}
    examples = examples + [text]


    # Format the examples into a string
    example_string = ""
    for example in examples:
        example_string += f"Example Observation for learning : {example['observation']}\n Example Context for learning : {example['context']} \n Example Finding for learning: {example['finding']}\n\n"

    return example_string

def generate_audit_findings(input_payload,run_config=None):

    example_string = prepare_example_data()

    # print('example_string: ', example_string)

    input_observation = input_payload['short_observations']
    input_context = prepare_context(input_payload)

    prompt = prepare_prompt(example_string, input_observation, input_context)
    print(prompt)
    
    # If the run config is provided, get the model name from there
    if run_config:
        llm_selection = run_config['model_name']

    if llm_selection == "local_ollama":

        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False # Set to True for streaming responses
        }

        # Send the prompt to the model
        output = run_inference_query(payload)

    else:
        payload = {
          "model": "bartowski/Qwen2.5-Coder-32B-Instruct-GGUF",
          "messages": [
            {
              "role": "user",
              "content": prompt
            }
          ],
          "max_tokens": 150,
          "stream": False
        }
        headers = {
          'Authorization': 'Bearer ' + my_hf_model_token,
          'Content-Type': 'application/json'
        }
        # Send the prompt to the model
        output = run_inference_query(payload)

    return output

def bulk_output():
    inference_type = "local" # or online
    model_name = ["qwen3:0.6b","gemma3:1b","deepseek-r1:1.5b"]
    embedding_model = "BAAI/bge-large-en-v1.5"  # Example embedding model

    data_sheet = ["21CFR Part 312","21CFR Part 50","21CFR Part 56"]
    data_sheet = ["21CFR Part 50"]
    source_file = ["21 CFR Part 50.pdf","21 CFR Part 56.pdf","21 CFR Part 58.pdf","NITLT01.pdf","AZD9291.pdf","ich-gcp-r3.pdf"]

    run_config = {
        "embedding_model": embedding_model
    }

    # Read the source file for question and answer pairs
    for model_llm in model_name:
        run_config["model_name"] = model_llm
        for sheet in data_sheet:
            df = pd.read_excel('Observations To Finding Training Data.xlsx',sheet_name=sheet)
            
            # List to store dictionaries of questions and responses
            qa_pairs = [] 

            for ii in range(0,df.shape[0]):
                
                # Do the inference in try catch block for each observation
                try:
                    tic = time.time()
                    short_observation = df['Short Observation'][ii]
                    protocol = df['Protocol'][ii]
                    reference_document_name = df['Reference Document Name'][ii]
                    clause_number = df['Clause Number'][ii]
                    category_name = df['Category Name'][ii]
                    long_observation = df['Long Observation'][ii]
    
                    input_payload = {
                            "short_observations" : short_observation,
                            "protocol" : protocol,
                            "reference_document_name" : reference_document_name,
                            "clause_number" : clause_number,
                            "category_name" : category_name
                            }
    
                    print(f"\nRunning workflow with sheet: {sheet} model: {model_llm}")
                    output_observation = generate_audit_findings(input_payload,run_config)
                    toc = time.time()
                    # Add the question and full response to our list
                    qa_pairs.append({
                        "short_observation": short_observation,
                        "Ground_truth": long_observation,
                        "output_oesponse": output_observation,
                        # "Contexts": retrieved_contexts,
                        # "Retrieved_Files": unique_retrieved_filenames,
                        "time_taken": round(toc - tic,2)
                    })
                except:
                    print(f"Inference failed")
                    qa_pairs.append({
                        "short_observation": short_observation,
                        "Ground_truth": long_observation,
                        "output_oesponse": [],
                        # "Contexts": retrieved_contexts,
                        # "Retrieved_Files": unique_retrieved_filenames,
                        "time_taken": round(toc - tic,2)
                    })
            # All questions are finished, create the DataFrame
            qa_df = pd.DataFrame(qa_pairs)
            
            # Optional: Save the DataFrame to a CSV file
            csv_filename = "qa_responses.csv"
            csv_filename = os.path.join("output", f"{run_config['sheet_name']}_{run_config['model_name']}_qa_responses.csv")    
            # Replace colons in filename to avoid issues on some filesystems
            csv_filename = csv_filename.replace(":", "_")  
            qa_df.to_csv(csv_filename, index=False, encoding="utf-8")
            print(f"\nDataFrame saved to {csv_filename}")

    return

if __name__ == "__main__":

    run_type = 'bulk'

    if run_type == 'single':
        input_payload = {
            "input_observations" : "Some study personnel had no documented GCP training. A study coordinator, with no ECG certification, conducted ECG assessments.",
            "document_name" : "ich-gcp.pdf",
            "document_section" : "5.4.3"
            }

        tic = time.time()
        output_observation = generate_audit_findings(input_payload)
        toc = time.time()

        print(output_observation)

        # clean_string = output_observation[0]['response'].replace("\\n", " ")
        # print(clean_string)

        print(f"Time taken: {toc-tic} seconds" )
    else:
        bulk_output()