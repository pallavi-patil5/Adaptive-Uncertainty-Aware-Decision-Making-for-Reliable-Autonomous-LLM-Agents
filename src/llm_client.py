# src/llm_client.py
import os
from dotenv import load_dotenv

load_dotenv()


def call_groq(prompt: str, model: str = "llama-3.1-8b-instant", temperature: float = 0.0) -> str:
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content


def call_hf(prompt: str, model: str = "Qwen/Qwen2.5-72B-Instruct", temperature: float = 0.0) -> str:
    from huggingface_hub import InferenceClient
    client = InferenceClient(api_key=os.environ["HF_TOKEN"])
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content

def _ensure_ollama_running() -> None:
    import time, subprocess, ollama
    try:
        ollama.list()
        return
    except Exception:
        pass
    ollama_exe = r"C:\Users\Pallavi\AppData\Local\Programs\Ollama\ollama.exe"
    subprocess.Popen([ollama_exe, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):          # wait up to 60 s
        time.sleep(1)
        try:
            ollama.list()
            return
        except Exception:
            pass
    raise RuntimeError("Ollama did not start within 60 seconds.")


def call_ollama(prompt: str, model: str = "llama3:8b", temperature: float = 0.0) -> str:
    import time, subprocess, ollama
    _ensure_ollama_running()
    for attempt in range(2):
        try:
            resp = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": temperature, "num_ctx": 2048},
            )
            return resp["message"]["content"]
        except Exception as e:
            if attempt == 0 and ("terminated" in str(e) or "ResponseError" in type(e).__name__):
                # llama-server crashed — stop, restart, retry once
                subprocess.Popen(["ollama", "stop", model], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                time.sleep(3)
                _ensure_ollama_running()
            else:
                raise

def call_llm(prompt: str, **kwargs) -> str:
    """Primary: Ollama."""
    return call_ollama(prompt, **kwargs)

if __name__ == "__main__":
    print(call_llm("What is the capital of France? Answer in one word."))