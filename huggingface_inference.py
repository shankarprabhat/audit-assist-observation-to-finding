import requests

API_URL = "https://api-inference.huggingface.co/models/your_username/your_model_name"
headers = {"Authorization": f"Bearer YOUR_API_TOKEN"} # Get your API Token from Hugging Face settings

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

output = query({
    "inputs": "observation: The payroll records were incomplete.",
})

print(output)