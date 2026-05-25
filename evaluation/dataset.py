from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document
from ingestion.parsers import parse_pdf
from ingestion.chunkers import chunk
import json


generator_llm = LangchainLLMWrapper(ChatOllama(model="mistral-nemo:12b",
        format="json", temperature=0))
embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text"))

def generate_testset(pdf_path, test_size=10, output_path="evaluation/testset.json"):
    pages = parse_pdf(pdf_path)
    chunks = chunk(pages)
    docs = [
        Document(
            page_content=c["text"],
            metadata={"source": c["source"], "page": c["page"]}
        )
        for c in chunks
    ]

    # generate testset
    generator = TestsetGenerator(llm=generator_llm, embedding_model=embeddings)
    testset = generator.generate_with_langchain_docs(
        docs,
        testset_size=test_size
    )

    # save to file
    df = testset.to_pandas()
    df.to_json(output_path, orient="records", indent=2)
    print(f"Generated {len(df)} test samples → saved to {output_path}")
    return df
if __name__ == "__main__":
    generate_testset("data/a.pdf", test_size=10)