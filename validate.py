from ingestion.parsers import parse_pdf
from ingestion.chunkers import chunk
from ingestion.embed import embed_chunks, embed_query
from vectorstore.store import store, query

# 1. ingest a test file
pages    = parse_pdf("data/a.pdf")
chunks   = chunk(pages)
embedded = embed_chunks(chunks)
store(embedded)

# 2. query it with a question related to your PDF
question    = "what television series is this an episode of?"
vector      = embed_query(question)
top_chunks, distances = query(vector)

# 3. print results
for i, (chunk, distance) in enumerate(zip(top_chunks, distances)):
    print(f"\n--- Result {i+1} ---")
    print(f"Source: {chunk['source']} | Page: {chunk['page']}")
    print(f"Distance: {distance}")
    print(f"Text: {chunk['text']}")