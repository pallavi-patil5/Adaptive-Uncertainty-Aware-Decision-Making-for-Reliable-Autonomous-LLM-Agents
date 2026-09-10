# src/load_datasets.py
from datasets import load_dataset
import json, os

os.makedirs("data/raw", exist_ok=True)

def save_raw(name, ds):
    ds.to_json(f"data/raw/{name}.jsonl")
    print(f"{name}: {len(ds)} examples saved")

# SimpleQA
simpleqa = load_dataset("basicv8vc/SimpleQA", split="test")
save_raw("simpleqa", simpleqa)

# SQuAD 2.0
squad2 = load_dataset("rajpurkar/squad_v2", split="validation")
save_raw("squad2", squad2)

# AmbigQA
ambigqa = load_dataset("sewon/ambig_qa", "light", split="validation")
save_raw("ambigqa", ambigqa)

# HotpotQA
hotpotqa = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
save_raw("hotpotqa", hotpotqa)

# HaluEval (qa subset)
halueval = load_dataset("pminervini/HaluEval", "qa", split="data")
save_raw("halueval_qa", halueval)

print("All datasets loaded. Custom Decision Dataset is authored manually — see Step 6.")