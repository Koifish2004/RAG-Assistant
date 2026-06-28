from pathlib import Path
from ingestion.parsers import parse_devdocs
from ingestion.parsers import parse_pdf
from ingestion.chunkers import chunk
from ingestion.embed import embed_chunks
from vectorstore.store import store
import json

REGISTRY_PATH = Path("vectorstore/ingested.json")


def load_registry():
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            return set(json.load(f))
    return set()


def save_registry(ingested):
    REGISTRY_PATH.parent.mkdir(exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(list(ingested), f, indent=2)


def ingest_pdf():
    ingested = load_registry()
    pdf_files = list(Path("data").glob("*.pdf"))
    new_files = [f for f in pdf_files if f.name not in ingested]

    if not new_files:
        print("No new PDFs to ingest.")
        return

    for pdf_path in new_files:
        print(f"Ingesting {pdf_path.name}...")
        try:
            pages = parse_pdf(str(pdf_path))
            chunks = chunk(pages)
            embedded = embed_chunks(chunks)
            store(embedded)
            ingested.add(pdf_path.name)
            save_registry(ingested)
            print(f"Done: {len(chunks)} chunks from {pdf_path.name}")
        except Exception as e:
            print(f"Failed to ingest {pdf_path.name}: {e}")


def ingest():
    chunks = parse_devdocs("data")
    embedded = embed_chunks(chunks)
    store(embedded)

    print(f"Done : {len(chunks)} chunks stored in vectorDB")


if __name__ == "__main__":
    ingest()
