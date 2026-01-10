import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

persist_directory = "db/chroma_db"
collection_name="company_docs"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o"
SCORE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

chroma_client = chromadb.PersistentClient(path=persist_directory)

collection = chroma_client.get_collection(name = collection_name)

count = collection.count()

print(f"Loaded collection count: {count}")

while True:
    query = input("Query> ").strip()

    if query.lower() in {"exit", "quit"}:
        print("Exiting retrieval.")
        break

    if not query:
        print("Please enter a non-empty query.\n")
        continue

    # Ask user for number of results
    top_k_input = input(
        f"Number of results (default {DEFAULT_TOP_K}): "
    ).strip()

    try:
        top_k = int(top_k_input) if top_k_input else DEFAULT_TOP_K
        if top_k <= 0:
            raise ValueError
    except ValueError:
        print("Invalid number. Using default value.\n")
        top_k = DEFAULT_TOP_K

    # =========================
    # EMBED QUERY
    # =========================
    query_embedding = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query
    ).data[0].embedding

    # =========================
    # VECTOR SEARCH
    # =========================
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    print("\n--- Retrieved Context ---")

    retrieved_chunks = []

    for doc, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        similarity = 1 - distance  # convert cosine distance → similarity

        if similarity >= SCORE_THRESHOLD:
            retrieved_chunks.append(doc)

            print(f"\nSource: {meta['source']}")
            print(f"Similarity: {similarity:.3f}")
            print("-" * 50)
            print(doc)

    if not retrieved_chunks:
        print("\nNo documents met the similarity threshold.")
        print("\n--- Generated Answer ---")
        print(
            "I don't have enough information to answer that question "
            "based on the provided documents."
        )
        print("\n" + "=" * 80 + "\n")

    context = "\n\n".join(
        f"- {chunk}" for chunk in retrieved_chunks
    )

    prompt = f"""
    You are a helpful assistant.

    Answer the question using ONLY the information from the documents below.
    If the answer is not present, say:
    "I don't have enough information to answer that question based on the provided documents."

    Question:
    {query}

    Documents:
    {context}
    """

    # =========================
    # LLM CALL
    # =========================
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    answer = response.choices[0].message.content

    print("\n--- Generated Answer ---")
    print(answer)
    print("\n" + "=" * 80 + "\n") 
       
    continue

