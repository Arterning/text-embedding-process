
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

    
def main():
    print("Hello from text-process!")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
    model = SentenceTransformer(EMBEDDING_MODEL)
    dimension = model.get_sentence_embedding_dimension()
    print(f"Embedding dimension: {dimension}")

    texts = ["Hello, world!", "This is a test sentence."]
    embeddings = model.encode(texts, show_progress_bar=True)
    for text, embedding in zip(texts, embeddings):
        print(f"Text: {text}\nEmbedding: {embedding[:5]}... (dim: {len(embedding)})\n")


if __name__ == "__main__":
    main()
