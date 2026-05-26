"""
Parse 10-K filings, chunk text, embed with
sentence-transformers, and store in ChromaDB
"""

import os
import glob
import hashlib
import fitz  # PyMuPDF
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from html.parser import HTMLParser

RAW_DIR       = "data/raw"
CHROMA_DIR    = "data/chroma_db"
COLLECTION    = "jpm_10k"
EMBED_MODEL   = "all-MiniLM-L6-v2"
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 150


# Extract Text
def extract_pdf(path):
    doc = fitz.open(path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text


def extract_html(path):
    
    class _Extractor(HTMLParser):
        def __init__(self):
            super().__init__()
            self.texts, self._skip = [], False

        def handle_starttag(self, tag, attrs):
            if tag in ("script", "style"):
                self._skip = True

        def handle_endtag(self, tag):
            if tag in ("script", "style"):
                self._skip = False

        def handle_data(self, data):
            if not self._skip:
                s = data.strip()
                if s:
                    self.texts.append(s)

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()
    ex = _Extractor()
    ex.feed(html)
    return "\n".join(ex.texts)


def load_documents():
    docs = []
    patterns = [os.path.join(RAW_DIR, p) for p in ("*.pdf", "*.htm", "*.html")]
    files = [f for pat in patterns for f in glob.glob(pat)]

    if not files:
        raise FileNotFoundError(
            f"No files found in {RAW_DIR}/\n"
        )

    for path in files:
        fname = os.path.basename(path)
        # Parse the year
        parts = fname.replace(".", "_").split("_")
        year = next((p for p in parts if p.isdigit() and len(p) == 4), "unknown")
        print(f"  Extracting: {fname}")
        text = extract_pdf(path) if path.lower().endswith(".pdf") else extract_html(path)
        docs.append({"filename": fname, "year": year, "text": text})
    return docs


# Chunk Documents
def chunk_documents(docs) :
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for doc in docs:
        for i, text in enumerate(splitter.split_text(doc["text"])):
            cid = hashlib.md5(f"{doc['filename']}_{i}".encode()).hexdigest()
            chunks.append({
                "id": cid,
                "text": text,
                "filename": doc["filename"],
                "year": doc["year"],
                "chunk_index": i,
            })
    return chunks


# Embed and Store in Chroma DB
def embed_and_store(chunks: list[dict]):
    model = SentenceTransformer(EMBED_MODEL)

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    batch = 64

    for i in tqdm(range(0, len(chunks), batch)):
        b = chunks[i: i + batch]
        embeddings = model.encode([c["text"] for c in b], show_progress_bar=False).tolist()
        collection.add(
            ids=[c["id"] for c in b],
            embeddings=embeddings,
            documents=[c["text"] for c in b],
            metadatas=[
                {"filename": c["filename"], "year": c["year"], "chunk_index": c["chunk_index"]}
                for c in b
            ],
        )



def main():
    print("\nLoad Docs")
    docs = load_documents()

    print("\nChunk Docs")
    chunks = chunk_documents(docs)

    print(f"\nEmbed and Store Chunks")
    embed_and_store(chunks)
    print("\nCompleted")


if __name__ == "__main__":
    main()
