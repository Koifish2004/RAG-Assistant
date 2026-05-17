import chromadb

client = chromadb.PersistentClient(path="vectorstore/db")
collection = client.get_or_create_collection(name="study_agent")

def store(embedded_chunks):
    ids        = []
    embeddings = []
    metadatas  = []

    for chunk in embedded_chunks:
        source    = chunk["source"].split("/")[-1]
        page      = chunk["page"]
        chunk_idx = chunk["chunk_index"]

        chunk_id  = f"{source}-page{page}-chunk{chunk_idx}"

        ids.append(chunk_id)
        embeddings.append(chunk["embedding"])
        metadatas.append({
            "source":      source,
            "page":        page,
            "chunk_index": chunk_idx,
            "text":        chunk["text"]
        })

    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas
    )
    print(f"Stored {len(ids)} chunks in ChromaDB")


def query(query_vector, n_results=5):
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        include=["metadatas", "distances"]
    )
    return results["metadatas"][0], results["distances"][0]