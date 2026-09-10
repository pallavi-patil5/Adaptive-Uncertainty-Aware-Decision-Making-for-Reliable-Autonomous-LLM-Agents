# src/llm_client.py
import os
from dotenv import load_dotenv
from groq import Groq
import ollama

load_dotenv()

groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

def call_groq(prompt: str, model: str = "llama-3.1-8b-instant", temperature: float = 0.0) -> str:
    resp = groq_client.chat.completions.create(
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
    """Primary: Groq. Falls back to Ollama on failure (rate limit, timeout, etc)."""
    try:
        return call_groq(prompt, **kwargs)
    except Exception as e:
        print(f"[llm_client] Groq failed ({e}), falling back to Ollama")
        return call_ollama(prompt, **kwargs)

if __name__ == "__main__":
    print(call_llm("What is the capital of France? Answer in one word."))