import pymupdf

def parse_pdf(filepath):
    doc = pymupdf.open(filepath)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text", sort=True)

        if text.strip():
            pages.append({
                "text": text,
                "source": filepath,
                "page": i + 1,
                "type": "text"
            })
    return pages