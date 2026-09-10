# src/vector_store.py
import chromadb
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
chroma_client = chromadb.PersistentClient(path="./data/processed/chroma_db")
collection = chroma_client.get_or_create_collection("evidence_store")

def embed(texts: list[str]) -> list[list[float]]:
    return embedder.encode(texts, normalize_embeddings=True).tolist()

def add_documents(docs: list[str], ids: list[str], extra_meta: list[dict] | None = None):
    collection.add(
        documents=docs,
        embeddings=embed(docs),
        ids=ids,
        metadatas=extra_meta or [{} for _ in docs],
    )

def query(text: str, k: int = 3, where: dict | None = None):
    q_emb = embed([text])
    return collection.query(query_embeddings=q_emb, n_results=k, where=where)

if __name__ == "__main__":
    add_documents(
        docs=["Paris is the capital of France.", "The Eiffel Tower is in Paris."],
        ids=["doc1", "doc2"],
        extra_meta=[{"source_dataset": "manual_test"}, {"source_dataset": "manual_test"}],
    )
    result = query("What is the capital of France?")
    print(result["documents"])