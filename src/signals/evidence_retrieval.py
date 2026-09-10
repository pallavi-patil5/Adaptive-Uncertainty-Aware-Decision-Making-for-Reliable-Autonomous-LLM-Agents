# src/signals/evidence_retrieval.py
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from vector_store import query

def retrieval_signals(question: str, k: int = 5, relevance_distance_threshold: float = 0.6) -> dict:
    """
    Returns:
      - top1_distance: distance of the single closest evidence doc (lower = more relevant)
      - avg_topk_distance: mean distance across top-k retrieved docs
      - evidence_coverage: fraction of top-k docs under the relevance threshold
      - retrieved_docs: raw retrieved text + metadata, for downstream cross-encoder scoring
    """
    result = query(question, k=k)
    distances = result["distances"][0] if result["distances"] else []
    documents = result["documents"][0] if result["documents"] else []
    metadatas = result["metadatas"][0] if result["metadatas"] else []

    if not distances:
        return {
            "top1_distance": None,
            "avg_topk_distance": None,
            "evidence_coverage": 0.0,
            "retrieved_docs": [],
        }

    top1_distance = distances[0]
    avg_topk_distance = sum(distances) / len(distances)
    relevant_count = sum(1 for d in distances if d < relevance_distance_threshold)
    evidence_coverage = relevant_count / len(distances)

    retrieved_docs = [
        {"text": doc, "distance": dist, "metadata": meta}
        for doc, dist, meta in zip(documents, distances, metadatas)
    ]

    return {
        "top1_distance": top1_distance,
        "avg_topk_distance": avg_topk_distance,
        "evidence_coverage": evidence_coverage,
        "retrieved_docs": retrieved_docs,
    }


if __name__ == "__main__":
    result = retrieval_signals("What is the capital of France?")
    print(f"top1_distance: {result['top1_distance']}")
    print(f"avg_topk_distance: {result['avg_topk_distance']}")
    print(f"evidence_coverage: {result['evidence_coverage']}")
    for doc in result["retrieved_docs"]:
        print(" -", doc["text"][:80], "| distance:", doc["distance"])