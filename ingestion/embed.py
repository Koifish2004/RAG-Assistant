import requests

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL = "nomic-embed-text"

def embed_chunks(chunks):
    embedded = []
    for chunk in chunks:
        response = requests.post(OLLAMA_URL, json={
            "model": MODEL,
            "prompt": chunk["text"]
        })
        vector = response.json()["embedding"]
        embedded.append({
            **chunk,
            "embedding": vector
        })
    return embedded


def embed_query(text):
    response = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "prompt": text
    })
    return response.json()["embedding"]