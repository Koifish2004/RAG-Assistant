import pymupdf4llm
from bs4 import BeautifulSoup as soup
from pathlib import Path
import json


def parse_pdf(filepath):
    pages = pymupdf4llm.to_markdown(
        filepath, page_chunks=True, header=False, footer=False
    )

    return [
        {
            "text": page["text"],
            "source": filepath,
            "page": page["metadata"]["page_number"],
            "type": "text",
        }
        for page in pages
        if page["text"].strip()
    ]


def parse_devdocs(docs_path):

    chunks = []

    for folder in Path("data").iterdir():

        if folder.is_dir():
            result = {}
            meta_path = folder / "meta.json"
            db_path = folder / "db.json"
            index_path = folder / "index.json"
            with open(meta_path) as f:
                meta = json.load(f)
                result["slug"] = meta["slug"]
                result["name"] = meta["name"]
                print(f"Processing {meta['slug']}...")

            with open(db_path) as f:
                db = json.load(f)
                result["db"] = db

            with open(index_path) as f:
                index = json.load(f)
                result["entries"] = index["entries"]

            for entry in result["entries"]:
                parts = entry["path"].split("#")
                page_path = parts[0]
                anchor = parts[1] if len(parts) > 1 else None

                html = result["db"].get(page_path, "")
                parsed_html = soup(html, "html.parser")
                if anchor:
                    section = parsed_html.find(id=anchor)
                else:
                    section = parsed_html

                text = section.get_text(separator=" ", strip=True) if section else ""
                chunk = {
                    "text": text,
                    "language": result["slug"],
                    "name": entry["name"],
                    "type": entry["type"],
                    "path": entry["path"],
                    "source": result["slug"],
                }

                chunks.append(chunk)
    return chunks


if __name__ == "__main__":
    chunks = parse_devdocs("data")
    print(f"Total chunks: {len(chunks)}")
    print(chunks[0])
