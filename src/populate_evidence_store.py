# src/populate_evidence_store.py
import sys, os, json
sys.path.append(os.path.join(os.path.dirname(__file__)))

from vector_store import add_documents

def extract_evidence_text(record: dict) -> str | None:
    """Flatten a record's evidence list into a single text string, or None if empty."""
    evidence = record.get("evidence", [])
    if not evidence:
        return None
    parts = []
    for item in evidence:
        if isinstance(item, dict):
            parts.append(" ".join(str(v) for v in item.values() if v))
        elif isinstance(item, str):
            parts.append(item)
    text = " ".join(p for p in parts if p).strip()
    return text if text else None


def populate(path="data/processed/unified_dataset.jsonl", batch_size=100):
    docs, ids, metas = [], [], []
    total_added = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            text = extract_evidence_text(record)
            if not text:
                continue
            docs.append(text)
            ids.append(f"evidence_{record['id']}")
            metas.append({
                "source_dataset": record["source_dataset"],
                "question_id": record["id"],
                "category": record.get("category", ""),
            })

            if len(docs) >= batch_size:
                add_documents(docs, ids, metas)
                total_added += len(docs)
                print(f"Added {total_added} evidence docs so far...")
                docs, ids, metas = [], [], []

    if docs:
        add_documents(docs, ids, metas)
        total_added += len(docs)

    print(f"Done. Total evidence documents indexed: {total_added}")


if __name__ == "__main__":
    populate()