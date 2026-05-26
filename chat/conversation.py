from ingestion.embed import embed_query
from vectorstore.store import query
from retrieval.generate import generate_with_history

from langchain_classic.memory import ConversationSummaryBufferMemory
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage

SYSTEM_PROMPT = """You are a helpful study assistant. 
Answer questions using only the provided context. 
If the answer is not in the context, say so."""

llm = ChatOllama(model="mistral")

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=500,
    return_messages=True
)

def chat(question):
    vector = embed_query(question)
    top_chunks, _ = query(vector, question)
    context = "\n\n".join([chunk["text"] for chunk in top_chunks])

    history = memory.load_memory_variables({})["history"]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for message in history:
        if isinstance(message, HumanMessage):
            messages.append({"role": "user", "content": message.content})
        elif isinstance(message, AIMessage):
            messages.append({"role": "assistant", "content": message.content})
    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {question}"
    })

    answer = generate_with_history(messages)

    # save to LangChain memory
    memory.save_context({"input": question}, {"output": answer})

    return answer