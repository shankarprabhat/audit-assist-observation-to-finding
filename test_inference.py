import requests
import json
from dotenv import load_dotenv
import os

load_dotenv()

url = "https://ytqej0aypbclmmsl.us-east-1.aws.endpoints.huggingface.cloud/v1/chat/completions"
my_hf_model_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

payload = json.dumps({
  "model": "bartowski/Qwen2.5-Coder-32B-Instruct-GGUF",
  "messages": [
    {
      "role": "user",
      "content": "What is deep learning?"
    }
  ],
  "max_tokens": 150,
  "stream": False
})
headers = {
  'Authorization': 'Bearer ' + my_hf_model_token,
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)
