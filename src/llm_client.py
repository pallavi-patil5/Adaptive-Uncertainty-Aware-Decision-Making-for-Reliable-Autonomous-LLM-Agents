# src/llm_client.py
import os
import signal
import threading
from dotenv import load_dotenv

load_dotenv()

LLM_TIMEOUT = 60  # seconds — if any call takes longer, return fallback


def _call_with_timeout(fn, timeout=LLM_TIMEOUT, fallback=""):
    """Run fn() in a thread; return fallback if it exceeds timeout seconds."""
    result = [fallback]
    exc    = [None]

    def _run():
        try:
            result[0] = fn()
        except Exception as e:
            exc[0] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        print(f"[llm_client] WARNING: LLM call timed out after {timeout}s — using fallback.")
        return fallback
    if exc[0]:
        raise exc[0]
    return result[0]


def call_groq(prompt: str, model: str = "llama3-8b-8192", temperature: float = 0.0) -> str:
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    def _call():
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            timeout=30,
        )
        return resp.choices[0].message.content
    return _call_with_timeout(_call)


def call_hf(prompt: str, model: str = "Qwen/Qwen2.5-72B-Instruct", temperature: float = 0.0) -> str:
    from huggingface_hub import InferenceClient
    client = InferenceClient(api_key=os.environ["HF_TOKEN"])
    def _call():
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return resp.choices[0].message.content
    return _call_with_timeout(_call)


def _find_ollama_exe() -> str:
    """Locate the ollama executable cross-platform without hardcoding a path."""
    import shutil, sys
    # 1. Honour an explicit override in .env  (OLLAMA_PATH=/custom/path/ollama)
    env_path = os.environ.get("OLLAMA_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path
    # 2. Find on system PATH (works on any OS after a normal Ollama install)
    found = shutil.which("ollama")
    if found:
        return found
    # 3. Windows default install location as last resort
    if sys.platform == "win32":
        default = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama", "ollama.exe")
        if os.path.isfile(default):
            return default
    raise RuntimeError(
        "Ollama executable not found. Install Ollama (https://ollama.com) and ensure it is on your PATH, "
        "or set OLLAMA_PATH=/path/to/ollama in your .env file."
    )


def _ensure_ollama_running() -> None:
    import time, subprocess, ollama
    try:
        ollama.list()
        return
    except Exception:
        pass
    ollama_exe = _find_ollama_exe()
    subprocess.Popen([ollama_exe, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
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

    def _call():
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
                    subprocess.Popen(["ollama", "stop", model], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    time.sleep(3)
                    _ensure_ollama_running()
                else:
                    raise

    return _call_with_timeout(_call)


def call_llm(prompt: str, **kwargs) -> str:
    """Primary: Ollama (local). Falls back to Groq if Ollama fails."""
    return call_ollama(prompt, **kwargs)


if __name__ == "__main__":
    print(call_llm("What is the capital of France? Answer in one word."))
