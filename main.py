
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

    sentences = [
        "今天天气很好",
        "我喜欢自然语言处理",
        "文本向量化的核心是语义表示",
        "文本向量化的核心概念是语义表示"
    ]
    embeddings = model.encode(sentences, show_progress_bar=True)
    for text, embedding in zip(sentences, embeddings):
        print(f"Text: {text}\nEmbedding: {embedding[:5]}... (dim: {len(embedding)})\n")

    # 计算两个句子的语义相似度
    from sentence_transformers.util import cos_sim
    similarity = cos_sim(embeddings[3], embeddings[2])
    print(f"句子3和句子4的相似度：{similarity.item():.4f}")


if __name__ == "__main__":
    main()
