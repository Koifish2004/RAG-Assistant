from ingestion.embed import embed_query
from vectorstore.store import query
from retrieval.generate import generate_with_history

SYSTEM_PROMPT = """You are a helpful study assistant. 
Answer questions using only the provided context. 
If the answer is not in the context, say so."""

conversation_history = []

def chat(question):
    vector = embed_query(question)
    top_chunks, _ = query(vector)
    context = "\n\n".join([chunk["text"] for chunk in top_chunks])

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    messages += conversation_history
    messages.append({
        "role":"user",
        "content": f"Context:\n{context}\n\nQuestion: {question}"
    })

    answer = generate_with_history(messages)


    conversation_history.append({"role":"user", "content": question})
    conversation_history.append({"role":"assistant", "context": answer})

    return answer