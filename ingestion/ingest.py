from parsers import parse_pdf
from chunkers import chunk
from embed import embed_chunks

pages    = parse_pdf("data/a.pdf")
chunks   = chunk(pages)
embedded = embed_chunks(chunks)

print(f"Parsed {len(pages)} pages → {len(chunks)} chunks → {len(embedded)} embedded")
print(embedded[0])