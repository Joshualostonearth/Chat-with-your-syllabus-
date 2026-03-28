import chromadb
from sentence_transformers import SentenceTransformer

def get_relevant_chunks(query, n_results=3):
    # Load ChromaDB
    client = chromadb.PersistentClient(path="./chroma_store")
    collection = client.get_collection("syllabus")

    # Convert query to embedding
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode(query).tolist()

    # Search for relevant chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "page": results["metadatas"][0][i]["page"]
        })
    return chunks

# Test it
if __name__ == "__main__":
    query = "What is machine learning?"
    chunks = get_relevant_chunks(query)
    print(f"Found {len(chunks)} relevant chunks:")
    for c in chunks:
        print(f"\nPage {c['page']}:\n{c['text'][:200]}")
