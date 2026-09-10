# src/end_to_end_smoke_test.py
from llm_client import call_llm
from vector_store import query

def trivial_pipeline(question: str) -> dict:
    retrieved = query(question, k=1)
    context = retrieved["documents"][0][0] if retrieved["documents"][0] else ""
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer concisely:"
    answer = call_llm(prompt)
    return {"question": question, "context_used": context, "answer": answer}

if __name__ == "__main__":
    result = trivial_pipeline("What is the capital of France?")
    print(result)