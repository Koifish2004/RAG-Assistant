from ragas import evaluate, EvaluationDataset, RunConfig
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ingestion.embed import embed_query
from vectorstore.store import query
from retrieval.generate import generate_with_history
import json
import pandas as pd

# wrap Ollama for RAGAS
eval_llm = LangchainLLMWrapper(ChatOllama(model="mistral-nemo:12b"))
eval_embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text"))

SYSTEM_PROMPT = """You are a helpful study assistant. 
Answer questions using only the provided context. 
If the answer is not in the context, say so."""

def run_pipeline(question, top_k=5):
    vector = embed_query(question)
    top_chunks, _ = query(vector, question, n_results=top_k)
    context = "\n\n".join([c["text"] for c in top_chunks])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]
    answer = generate_with_history(messages)
    contexts = [c["text"] for c in top_chunks]
    return answer, contexts

def run_evaluation(testset_path="evaluation/testset.json"):
    with open(testset_path) as f:
        testset = json.load(f)

    samples = []
    for item in testset:
        question = item["user_input"]
        ground_truth = item["reference"]

        print(f"Running: {question[:60]}...")
        answer, contexts = run_pipeline(question)

        samples.append({
            "user_input": question,
            "response": answer,
            "retrieved_contexts": contexts,
            "reference": ground_truth
        })

    dataset = EvaluationDataset.from_list(samples)

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=eval_llm,
        embeddings=eval_embeddings,
        run_config=RunConfig(timeout=300, max_retries=2, max_workers=1)    )

    df = result.to_pandas()
    print("\n--- RAGAS Evaluation Results ---")
    print(df[["user_input", "faithfulness", "answer_relevancy"]].to_string())
    print(f"\nMean scores:")
    print(df[["faithfulness", "answer_relevancy"]].mean())

    df.to_json("evaluation/results.json", orient="records", indent=2)
    print("\nFull results saved to evaluation/results.json")

if __name__ == "__main__":
    run_evaluation()