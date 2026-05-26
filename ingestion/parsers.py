import pymupdf4llm

def parse_pdf(filepath):
    pages = pymupdf4llm.to_markdown(
        filepath,
        page_chunks=True,
        header=False,
        footer=False
    )
   
    return [
        {
            "text": page["text"],
            "source": filepath,
            "page": page["metadata"]["page_number"],
            "type": "text"
        }
        for page in pages
        if page["text"].strip()
    ]