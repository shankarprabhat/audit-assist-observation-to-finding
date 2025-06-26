import requests
import pandas as pd
import traceback
import os
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env into the environment

llm_selection = "local_ollama"

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


# 1. Load your CSV data (replace 'your_data.csv' with your file path)
def prepare_example_data():
    """This code prepares the example data to be sent in the prompt"""
    
    df = pd.read_csv('Audit Observations-Findings_with_context.csv')
    # df = df.dropna()
    
    # Assuming observations are in column 2 and findings are in column 3 (index 0,1,2..)
    observations = df.iloc[1,1]
    context = df.iloc[1,4]
    findings = df.iloc[1,5]
    
    examples = []
    
    for ii in range(0,1):
        text = {"observation": observations, "context": context, "finding": findings}
        examples = examples + [text]
    
    
    # Format the examples into a string
    example_string = ""
    for example in examples:
        example_string += f"Example Observation: {example['observation']}\n Example Context: {example['context']} \n Example Finding: {example['finding']}\n\n"
    
    return example_string

def generate_audit_findings(input_observation,input_context,example_string):

    # Your question         
    question = f"Observation: {input_observation} \n Context: {input_context} \nFinding: "
    
    # Create the prompt
    promt_question = "You are an Auditor for analyzing clinical trials reports. You are being provided" \
    " an example of audit observation and its context, and are required to provide the finding. One"\
    " example is given. Provide the findings for new observation and context." 
    prompt = f"{promt_question}:\n\n{example_string}\nObservation: {input_observation}\n Context: {input_context}\n\n" \
    f"Finding:"

    print(prompt)
    import json
    
    if llm_selection == "local_ollama":
    
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False # Set to True for streaming responses
        }
    
        # Send the prompt to the model
        output = run_inference_query(payload)
    else:
        # Send the prompt to the model
        output = run_inference_query({
            "inputs": prompt
        })
    
    return output

if __name__ == "__main__":
    # example_string = prepare_data()
    example_string = prepare_example_data()
    
    print('example_string: ', example_string)

    input_observation = "Some study personnel had no documented GCP training. A study coordinator, with no ECG certification, conducted ECG assessments."
    
    input_context = "GCP: 4.1.1. The investigator(s) should be qualified by education, training," \
    " and experience to assume responsibility for the proper conduct of the trial, should meet all" \
    " the qualifications specified by the applicable regulatory requirement(s), and should provide " \
    "evidence of such qualifications through up-to-date curriculum vitae and/or other relevant" \
    " documentation requested by the sponsor, the IRB/IEC, and/or the regulatory authority(ies)." \
    " GCP: 4.2.3. The investigator should have available an adequate number of qualified staff" \
    " and adequate facilities for the foreseen duration of the trial to conduct the trial properly" \
    " and safely."   

    
    output_observation = generate_audit_findings(input_observation,input_context,example_string)
    print(output_observation)
