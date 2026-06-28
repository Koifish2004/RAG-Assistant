import requests

OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL = "nomic-embed-text"


def embed_chunks(chunks):
    embedded = []

    batch_size = 64

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [chunk["text"] for chunk in batch]
        response = requests.post(OLLAMA_URL, json={"model": MODEL, "input": texts})

        vectors = response.json()["embeddings"]
        zipped = zip(batch, vectors)
        for chunk, vector in zipped:
            embedded.append({**chunk, "embedding": vector})
        print(f"{i} chunks embedded")

    return embedded


def embed_query(text):
    response = requests.post(OLLAMA_URL, json={"model": MODEL, "prompt": text})
    return response.json()["embedding"]
