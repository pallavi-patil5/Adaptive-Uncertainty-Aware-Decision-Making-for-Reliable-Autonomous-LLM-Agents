# src/llm_client.py
import os
from dotenv import load_dotenv
import ollama
from huggingface_hub import InferenceClient

load_dotenv()

hf_client = InferenceClient(api_key=os.environ["HF_TOKEN"])

def call_hf(prompt: str, model: str = "Qwen/Qwen2.5-72B-Instruct", temperature: float = 0.0) -> str:
    resp = hf_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content

def call_ollama(prompt: str, model: str = "llama3:8b", temperature: float = 0.0) -> str:
    resp = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature},
    )
    return resp["message"]["content"]

def call_llm(prompt: str, **kwargs) -> str:
    """Primary: Ollama llama3:8b."""
    return call_ollama(prompt, **kwargs)

if __name__ == "__main__":
    print(call_llm("What is the capital of France? Answer in one word."))