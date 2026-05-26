"""
Retriever class performs semantic search over ChromaDB
"""

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR  = "data/chroma_db"
COLLECTION  = "jpm_10k"
EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_K   = 5


class Retriever:
    def __init__(
        self,
        chroma_dir: str = CHROMA_DIR,
        collection_name: str = COLLECTION,
        embed_model: str = EMBED_MODEL,
        top_k: int = DEFAULT_K,
    ):
        self.model = SentenceTransformer(embed_model)
        client = chromadb.PersistentClient(path=chroma_dir)
        self.collection = client.get_collection(collection_name)
        self.top_k = top_k

    # Return the top-k relevant chunks for a query
    def retrieve(self, query, top_k):
        k = top_k or self.top_k
        embedding = self.model.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=embedding,
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # use cosine similarity for score
            chunks.append({
                "text": doc,
                "metadata": meta,
                "score": round(1 - dist, 4),  
            })
        return chunks

    # Format chunks into a single context block for the LLM
    def format_context(self, chunks):
        parts = []
        for i, c in enumerate(chunks, 1):
            year = c["metadata"].get("year", "?")
            parts.append(f"[Source {i} | Year: {year}]\n{c['text']}")
        return "\n\n---\n\n".join(parts)
