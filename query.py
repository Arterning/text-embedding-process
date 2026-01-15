"""Query text chunks from Milvus vector database."""

import os
import sys

from dotenv import load_dotenv
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
MILVUS_DB_URI = os.getenv("MILVUS_DB_URI", "http://localhost:19530")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")


def query(question: str, top_k: int = 5) -> list[dict]:
    """Query similar text chunks from Milvus."""
    model = SentenceTransformer(EMBEDDING_MODEL)
    client = MilvusClient(uri=MILVUS_DB_URI)

    if not client.has_collection(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' does not exist.")
        print("Please run import_docs.py first to import documents.")
        return []

    query_embedding = model.encode([question])[0].tolist()

    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_embedding],
        limit=top_k,
        output_fields=["text", "header", "source"],
    )

    return results[0] if results else []


def main():
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Enter your question: ").strip()

    if not question:
        print("No question provided.")
        return

    print(f"\nSearching for: {question}\n")
    print("-" * 60)

    results = query(question)

    if not results:
        print("No results found.")
        return

    for i, result in enumerate(results, 1):
        entity = result["entity"]
        distance = result["distance"]
        print(f"\n[{i}] Score: {distance:.4f}")
        print(f"    Source: {entity['source']}")
        if entity["header"]:
            print(f"    Header: {entity['header']}")
        print(f"    Text: {entity['text'][:200]}...")
        print("-" * 60)


if __name__ == "__main__":
    main()
