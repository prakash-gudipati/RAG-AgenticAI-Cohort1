import os
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
import chromadb
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_documents(docs_path="docs"):
    documents = []

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"{docs_path} does not exist")

    for filename in os.listdir(docs_path):
        if filename.endswith(".txt"):
            full_path = os.path.join(docs_path, filename)
            with open(full_path, "r", encoding="utf-8") as f:
                text = f.read()

            documents.append({
                "content": text,
                "metadata": {
                    "source": full_path
                }
            })

    if not documents:
        raise FileNotFoundError("No .txt files found")

    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=10):
    chunks = []

    for doc in documents:
        text = doc["content"]
        source = doc["metadata"]["source"]

        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            chunks.append({
                "content": chunk_text,
                "metadata": {"source": source}
            })

            start = end - chunk_overlap

    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    chroma_client = chromadb.PersistentClient(path=persist_directory)

    collection = chroma_client.get_or_create_collection(
        name="company_docs",
        metadata={"hnsw:space": "cosine"}
    )

    texts = [c["content"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [f"doc_{i}" for i in range(len(chunks))]

    embeddings = embed_texts(texts)

    collection.add(
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
        ids=ids
    )

    return collection

def embed_texts(texts, model="text-embedding-3-small"):
    response = client.embeddings.create(
        model=model,
        input=texts
    )
    return [item.embedding for item in response.data]

def visualize_embeddings(collection):
    data = collection.get(include=["embeddings", "metadatas"])

    vectors = np.array(data["embeddings"])
    sources = [m["source"] for m in data["metadatas"]]

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(vectors)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7)

    for i, src in enumerate(sources):
        plt.annotate(os.path.basename(src), (reduced[i, 0], reduced[i, 1]), fontsize=8)

    plt.title("Embedding Space Visualization (PCA)")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.grid(True)
    plt.show()

def main():
    print("=== RAG Ingestion ===")

    docs_path = "docs"
    persist_directory = "db/chroma_db"

    documents = load_documents(docs_path)
    print(f"Loaded {len(documents)} documents")

    chunks = split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    collection = create_vector_store(chunks, persist_directory)
    print(f"Stored {collection.count()} vectors in Chroma")

    print("\nVisualizing embedding space...")
    visualize_embeddings(collection)
    
    print("✅ Ingestion complete")


if __name__ == "__main__":
    main()