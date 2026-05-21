import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "mistral"

def generate(question, chunks):
    context = "\n\n".join([chunk["text"] for chunk in chunks])

    response = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful study assistant. Answer questions using only the provided context. If the answer is not in the context, say so."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ]
    })

    return response.json()["message"]["content"]

def generate_with_history(messages):
    response = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "stream": False,
        "messages": messages
    })

    return response.json()["message"]["content"]