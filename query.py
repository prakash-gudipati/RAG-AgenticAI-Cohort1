import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed_query(query, model="text-embedding-3-small"):
    response = client.embeddings.create(
        model=model,
        input=[query]
    )
    return response.data[0].embedding

def query_vector_store(query, persist_directory="db/chroma_db", n_results=3):
    chroma_client = chromadb.PersistentClient(path=persist_directory)

    collection = chroma_client.get_or_create_collection(
        name="company_docs",
        metadata={"hnsw:space": "cosine"}
    )

    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    return results

def inspect_database(persist_directory="db/chroma_db"):
    chroma_client = chromadb.PersistentClient(path=persist_directory)

    collection = chroma_client.get_or_create_collection(
        name="company_docs",
        metadata={"hnsw:space": "cosine"}
    )

    print(f"\n=== Database Info ===")
    print(f"Collection name: {collection.name}")
    print(f"Total vectors: {collection.count()}")
    print(f"Metadata: {collection.metadata}")

    if collection.count() > 0:
        results = collection.get(limit=5)
        print(f"\n=== First 5 Documents ===")
        for i, (doc_id, doc, metadata) in enumerate(zip(results['ids'], results['documents'], results['metadatas'])):
            print(f"\n[{i+1}] ID: {doc_id}")
            print(f"Source: {metadata.get('source', 'N/A')}")
            print(f"Content preview: {doc[:200]}...")

def main():
    print("=== ChromaDB Query Tool ===\n")

    query = input("Enter your query: ").strip()
    n_results = input("Number of results (default 3): ").strip()
    n_results = int(n_results) if n_results else 3

    print(f"\n=== Searching for: '{query}' ===")
    results = query_vector_store(query, n_results=n_results)

    print(f"\nFound {len(results['ids'][0])} results:\n")
    for i, (doc_id, doc, metadata, distance) in enumerate(zip(
        results['ids'][0],
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    )):
        print(f"\n[Result {i+1}] (Distance: {distance:.4f})")
        print(f"ID: {doc_id}")
        print(f"Source: {metadata.get('source', 'N/A')}")
        print(f"Content:\n{doc}\n")
        print("-" * 80)

if __name__ == "__main__":
    main()
