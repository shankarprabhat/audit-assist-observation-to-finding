import requests
import pandas as pd
import traceback
import os
from dotenv import load_dotenv
import time
import re
import json
import config

load_dotenv()  # Load variables from .env into the environment

llm_selection = "local_ollama"
# llm_selection = "hugging-face-qwen"

if llm_selection == 'flan-t5-large':
    token = os.environ.get("HF_TOKEN")
    TOKEN = "Bearer " + token
    headers = {"Authorization": TOKEN}
    base_llm_model_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"

elif llm_selection == 'hugging-face-qwen':
    base_llm_model_URL = os.environ.get("hugging_face_qwen")
    my_hf_model_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    headers = {
      'Authorization': 'Bearer ' + my_hf_model_token,
      'Content-Type': 'application/json'
    }
elif llm_selection == 'hugging-face-phi4':
    base_llm_model_URL = os.environ.get("hugging_face_phi4")
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

def get_clause_number(input_payload):

    # Get the protocol and regulatory clause
    clause_number = input_payload['clauseNumber']
    category_name = input_payload['categoryName']

    clause_protocol = []
    clause_regulatory = []

    # If both the regulatory clause and protocol clause are present
    if '\n' in clause_number and '\n' in category_name:
        split_clause = clause_number.split('\n')
        clause_regulatory = split_clause[1]

        clause_protocol = split_clause[0]
        clause_protocol = clause_protocol.replace('Protocol section ','')
        clause_protocol = clause_protocol.replace('Protocol section-','')
        clause_protocol = clause_protocol.replace('Protocol-section ','')
        clause_protocol = clause_protocol.strip()

    elif '\n' not in clause_number:
        clause_regulatory = clause_number
    else:
        print("Protocol clauses not provided")

    clause_regulatory = clause_regulatory.replace('21 CFR ','')
    clause_regulatory = clause_regulatory.strip()

    print('regulatory clause: ', clause_regulatory)
    print('protocol clause: ', clause_protocol)
    
    return clause_regulatory, clause_protocol

def prepare_context_section(input_payload, clause_regulatory, clause_protocol):
    # 1. Use semantic search on entire data to find the context
    # 2. Use semantic search on the section data to find the context
    # 3. Use the section data to extract the context
    # 3. For the regulatory document, extract the section
    regulatory_document_name = input_payload['regulatoryDocumentName']
    protocol_document_name = input_payload['protocolDocumentName']

    source = 'section_metadata/'
    regulatory_file = source + regulatory_document_name + '.json'
    # Load the json file
    with open(regulatory_file, 'r') as file:
        regulatory_data = json.load(file)

    protocol_data = []
    if protocol_document_name != 'None':
        protocol_file = source + protocol_document_name + '.json'
        # load protocol file
        with open(protocol_file,'r') as file:
            protocol_data = json.load(file)

    if len(clause_regulatory) > 0:
        regulatory_context = [data['content'] for data in regulatory_data["sections"] if clause_regulatory in data['title']]
        regulatory_context = ' '.join(regulatory_context)
        regulatory_context = regulatory_context[:200]
        print('\nregulatory_context: ', regulatory_context)
    else:
        regulatory_context = ""

    if len(clause_protocol)> 0 and len(protocol_data) > 0:
        protocol_context = [data['content'] for data in protocol_data["sections"] if clause_protocol in data['title']]
        protocol_context = ' '.join(protocol_context)
        protocol_context = protocol_context[:200]
        print('\nprotocol_context: ', protocol_context)
    else:
        protocol_context =""
    
    return regulatory_context, protocol_context

async def prepare_context(input_payload):
    
    context_fetch = input_payload['contextFetch']
    clause_regulatory, clause_protocol = get_clause_number(input_payload)
    protocol_context = ""
    
    if context_fetch == 'section':        
        regulatory_context, protocol_context = prepare_context_section(input_payload, clause_regulatory, clause_protocol)        
    
    if context_fetch == 'RAG':
        import RAG_retrieve as RAG
        embedding_model_choice = "BAAI/bge-large-en-v1.5"
        run_config_for_retrieval = {
            "question": "Informed consent did not mention the expected duration of the subject's participation",
            "embeddingModel": embedding_model_choice,
            "sourceFile": input_payload['regulatoryDocumentName'] + ".pdf",
            "useCloudEmbed": input_payload['useCloudEmbed']
        }
        if len(clause_regulatory) > 0:
            regulatory_context, success = await RAG.main_function(run_config_for_retrieval)
        
        if len(clause_protocol) > 0:
            run_config_for_retrieval['source_file'] = input_payload['protocolDocumentName'] + ".pdf"
            protocol_context, success = await RAG.main_function(run_config_for_retrieval)

    if len(regulatory_context) > 0:
        obtained_context = "Regulatory Context: " + regulatory_context + " Protocol Context: " + protocol_context
    else:
        # Extract the regulatory clause from the json data
        obtained_context = ""

        # obtained_context = "Regulatory Context: GCP: 4.1.1. The investigator(s) should be qualified by education, training," \
        # " and experience to assume responsibility for the proper conduct of the trial, should meet all" \
        # " the qualifications specified by the applicable regulatory requirement(s), and should provide " \
        # "evidence of such qualifications through up-to-date curriculum vitae and/or other relevant" \
        # " documentation requested by the sponsor, the IRB/IEC, and/or the regulatory authority(ies)." \
        # " GCP: 4.2.3. The investigator should have available an adequate number of qualified staff" \
        # " and adequate facilities for the foreseen duration of the trial to conduct the trial properly" \
        # " and safely. Protocol Context: None"

    print("\nobtained context: ", obtained_context)

    return obtained_context

def prepare_prompt(example_string, input_observation, input_context):

    # Create the prompt
    promt_question = "You are an Auditor for analyzing clinical trials reports. Based on an input short observation "\
    "and its context, your job is to come up with your findings. The context will contain"\
    " regulatory context, and may or may not contain protocol context. One example of audit observation, its context,"\
    " and corresponding findings are been provided. You need to come up with the findings for cases provided to you "\
    "based on thinking and logical observation, dont copy from the example given, come up with own findings."

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

async def generate_audit_findings(input_payload,run_config=None):

    example_string = prepare_example_data()

    # print('example_string: ', example_string)

    input_observation = input_payload['shortObservations']
    input_context = await prepare_context(input_payload)

    prompt = prepare_prompt(example_string, input_observation, input_context)
    # print(prompt)

    run_llm = False

    if run_llm is True:

        # If the run config is provided, get the model name from there
        if run_config:
            # llm_selection = run_config['model_name']
            model_name = run_config['model_name']

        if llm_selection in "local_ollama":

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
    else:
        output = []

    # print('input context: ', input_context)
    # print("output: ", output)

    return output, input_context

async def bulk_output():
    inference_type = "local" # or online
    embedding_model = "BAAI/bge-large-en-v1.5"  # Example embedding model
    context_fetch = "RAG"  # other option is RAG

    model_name = ["qwen3:0.6b","gemma3:1b","deepseek-r1:1.5b"]
    model_name = ["qwen3:0.6b"]

    data_sheet = ["21CFR Part 312","21CFR Part 50","21CFR Part 56"]
    data_sheet = ["21CFR Part 50"]

    regulatory_source_file = ["21 CFR Part 50","21 CFR Part 56","21 CFR Part 58","ich-gcp-r3"]
    regulatory_source_file = ["21 CFR Part 50"]

    protocol_document_list = {
        "NITLT01": "NITLT01",
        "FETONET": "FETONET",
        "AZD9291": "AZD9291",
        "DIAN-TU-001": "DIAN-TU-001"
        }

    run_config = {
        "embedding_model": embedding_model
    }

    # Read the source file for question and answer pairs
    for model_llm in model_name:
        run_config["model_name"] = model_llm
        count = 0
        for sheet in data_sheet:
            print(f"\nRunning workflow with model: {model_llm} sheet: {sheet}")

            df = pd.read_excel('Observations To Finding Training Data.xlsx',sheet_name=sheet)

            df = df.iloc[:3,:]

            # List to store dictionaries of questions and responses
            qa_pairs = []

            for ii in range(0,df.shape[0]):
                print(f"\nQuestion Number: {ii}")

                # Do the inference in try catch block for each observation
                try:
                    tic = time.time()
                    short_observation = df['Short Observation'][ii]
                    protocol = df['Protocol'][ii]
                    protocol_absent = pd.isna(df['Reference Document Number'][ii])

                    if protocol_absent is False:
                        try:
                            reference_document_name = df['Reference Document Number'][ii].strip()
                        except:
                            reference_document_name = 'None'
                    else:
                        reference_document_name = 'None'

                    clause_number = df['Clause Number'][ii]
                    category_name = df['Category Name'][ii]
                    long_observation = df['Long Observation'][ii]

                    input_payload = {
                            "shortObservations" : short_observation,
                            "protocol" : protocol,
                            "protocolDocumentName" : reference_document_name,
                            "regulatoryDocumentName": regulatory_source_file[count],
                            "clauseNumber" : clause_number,
                            "categoryName" : category_name,
                            "contextFetch": context_fetch
                            }

                    output_observation, context = await generate_audit_findings(input_payload,run_config)
                    print('\ninput context: ', context)
                    print("\noutput: ", output_observation)

                    if len(output_observation) > 0:
                        response = output_observation[0]['response']
                        clean_response = response.replace("\\n", " ")

                        # Using regular expressions to extract content
                        match = re.search(r'<think>(.*?)</think>(.*)', clean_response, re.DOTALL)

                        if match:
                            out_think = match.group(1).strip()
                            response = match.group(2).strip()
                        else:
                            out_think = ""
                            response = clean_response # If no <think> tag, the whole thing is the response
                    else:
                        response =[]
                        clean_response = []
                        out_think = []

                    # out_think = []
                    toc = time.time()
                    # Add the question and full response to our list
                    qa_pairs.append({
                        "short_observation": short_observation,
                        "ground_truth": long_observation,
                        "output_response": response,
                        "input_context": context,
                        "output_thinking": out_think,
                        # "Contexts": retrieved_contexts,
                        # "Retrieved_Files": unique_retrieved_filenames,
                        "time_taken": round(toc - tic,2)
                    })
                    print(f"Time Question: {toc-tic} seconds")
                except:
                    toc = time.time()
                    print(f"Inference failed")
                    traceback.print_exc()

                    qa_pairs.append({
                        "short_observation": short_observation,
                        "ground_truth": long_observation,
                        "output_response": [],
                        "input_context": context,
                        "output_thinking": [],
                        # "Contexts": retrieved_contexts,
                        # "Retrieved_Files": unique_retrieved_filenames,
                        "time_taken": round(toc - tic,2)
                    })
            # All questions are finished, create the DataFrame
            qa_df = pd.DataFrame(qa_pairs)

            # Optional: Save the DataFrame to a CSV file
            csv_filename = "qa_responses.csv"
            csv_filename = os.path.join("output", f"{sheet}_{run_config['model_name']}_qa_responses.csv")
            # Replace colons in filename to avoid issues on some filesystems
            csv_filename = csv_filename.replace(":", "_")
            qa_df.to_csv(csv_filename, index=False, encoding="utf-8")
            print(f"\nDataFrame saved to {csv_filename} time taken: {toc-tic} seconds")

        # Increase the count for next sheet
        count = count + 1
    return

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()

    run_type = 'bulk'
    context_fetch = "RAG"

    if run_type == 'single':
        input_payload = {
            "shortObservations" : "Some study personnel had no documented GCP training. A study coordinator, with no ECG certification, conducted ECG assessments.",
            "referenceDocumentName" : "ich-gcp.pdf",            
            "protocol" : "AZD9291",
            "protocolDocumentName" : "AZD9291",
            "regulatoryDocumentName": "21 CFR 50",
            "clauseNumber" : "Protocol section 17\n21 CFR  50.27",
            "categoryName" : "Administrative and Legal",
            "contextFetch": context_fetch,
            "useCloudEmbed": False
            }

        tic = time.time()
        output_observation = generate_audit_findings(input_payload)
        toc = time.time()

        # print(output_observation)

        # clean_string = output_observation[0]['response'].replace("\\n", " ")
        # print(clean_string)

        print(f"Time taken: {toc-tic} seconds" )
    else:
        import asyncio
        asyncio.run(bulk_output())