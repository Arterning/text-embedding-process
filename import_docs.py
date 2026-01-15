"""Import markdown files from docs directory into Milvus vector database."""

import os
import re
from pathlib import Path

from dotenv import load_dotenv
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
MILVUS_DB_PATH = os.getenv("MILVUS_DB_PATH", "./milvus_data.db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")
DOCS_DIR = Path("docs")


def split_by_headers(content: str, file_path: str) -> list[dict]:
    """Split markdown content by headers."""
    lines = content.split("\n")
    chunks = []
    current_chunk = []
    current_header = ""

    for line in lines:
        header_match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if header_match:
            if current_chunk:
                text = "\n".join(current_chunk).strip()
                if text:
                    chunks.append({
                        "text": text,
                        "header": current_header,
                        "source": file_path,
                    })
            current_header = header_match.group(2)
            current_chunk = [line]
        else:
            current_chunk.append(line)

    if current_chunk:
        text = "\n".join(current_chunk).strip()
        if text:
            chunks.append({
                "text": text,
                "header": current_header,
                "source": file_path,
            })

    return chunks


def load_markdown_files(docs_dir: Path) -> list[dict]:
    """Load and split all markdown files from docs directory."""
    all_chunks = []

    if not docs_dir.exists():
        print(f"Directory {docs_dir} does not exist. Creating it...")
        docs_dir.mkdir(parents=True)
        return all_chunks

    md_files = list(docs_dir.glob("**/*.md"))
    if not md_files:
        print(f"No markdown files found in {docs_dir}")
        return all_chunks

    for md_file in md_files:
        print(f"Processing: {md_file}")
        content = md_file.read_text(encoding="utf-8")
        chunks = split_by_headers(content, str(md_file))
        all_chunks.extend(chunks)
        print(f"  -> {len(chunks)} chunks extracted")

    return all_chunks


def main():
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    dimension = model.get_sentence_embedding_dimension()
    print(f"Embedding dimension: {dimension}")

    chunks = load_markdown_files(DOCS_DIR)
    if not chunks:
        print("No chunks to import. Please add markdown files to the docs directory.")
        return

    print(f"\nTotal chunks to import: {len(chunks)}")

    print(f"\nConnecting to Milvus: {MILVUS_DB_PATH}")
    client = MilvusClient(MILVUS_DB_PATH)

    if client.has_collection(COLLECTION_NAME):
        print(f"Dropping existing collection: {COLLECTION_NAME}")
        client.drop_collection(COLLECTION_NAME)

    print(f"Creating collection: {COLLECTION_NAME}")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        dimension=dimension,
    )

    print("\nGenerating embeddings...")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    print("\nInserting data into Milvus...")
    data = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        data.append({
            "id": i,
            "vector": embedding.tolist(),
            "text": chunk["text"],
            "header": chunk["header"],
            "source": chunk["source"],
        })

    client.insert(collection_name=COLLECTION_NAME, data=data)
    print(f"\nSuccessfully imported {len(data)} chunks into Milvus.")


if __name__ == "__main__":
    main()
