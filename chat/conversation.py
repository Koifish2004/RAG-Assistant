from ingestion.embed import embed_query
from vectorstore.store import query
from retrieval.generate import generate_with_history, classify

from langchain_classic.memory import ConversationSummaryBufferMemory
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
import time
import logging

SYSTEM_PROMPT = """You are a helpful study assistant. 
Answer questions using only the provided context. 
If the answer is not in the context, say so."""

llm = ChatOllama(model="mistral")

memory = ConversationSummaryBufferMemory(
    llm=llm, max_token_limit=500, return_messages=True
)

logger = logging.getLogger(__name__)
logging.basicConfig(filename="chatinfo.log", encoding="utf-8", level=logging.INFO)


def chat(question, language):

    # classification first

    label, confidence = classify(question)
    logging.info("User Query: %s", question)

    logging.info("Class: %s with confidence %s", label, confidence)

    if confidence < 0.5:
        token = "Could you reframe your question?"
        yield token

    elif label in {"CONCEPTUAL", "DOC_LOOKUP"}:

        t1 = time.time()
        vector = embed_query(question)
        print(f"Embed: {time.time() - t1:.2f}s")
        t2 = time.time()
        top_chunks, _ = query(vector, question, language)
        print(f"Retrieve: {time.time() - t2:.2f}s")
        context = "\n\n".join([chunk["text"] for chunk in top_chunks])

        history = memory.load_memory_variables({})["history"]

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for message in history:
            if isinstance(message, HumanMessage):
                messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                messages.append({"role": "assistant", "content": message.content})
        messages.append(
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        )
        t3 = time.time()
        answer = ""
        for token in generate_with_history(messages):
            answer += token
            yield token

        print(f"Generate: {time.time() - t3:.2f}s")

        # save to LangChain memory
        memory.save_context({"input": question}, {"output": answer})

    elif label in {"SOLUTION_SEEKING", "ADVERSARIAL"}:
        token = "I can explain the concepts involved but won't write the solution. What part are you unsure about?"
        yield token

    else:
        token = "I cannot answer off topic questions, Please ask relevant questions"
        yield token
