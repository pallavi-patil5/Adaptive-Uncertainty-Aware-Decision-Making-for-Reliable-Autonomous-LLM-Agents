# quick_latency_check.py — run from repo root, delete after
import sys, os, time
sys.path.append("src")
from llm_client import call_llm

t0 = time.monotonic()
print("t0:", t0)
answer = call_llm("What is the capital of France? Answer in one sentence.", temperature=0.0)
t1 = time.monotonic()
print("t1:", t1)
print("raw elapsed:", t1 - t0)
print("answer:", answer)