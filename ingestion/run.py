"""
Ingestion pipeline entrypoint.

Run this once (or on a schedule) to (re)build the knowledge base:

    python -m ingestion.run

Reads VECTOR_STORE_BACKEND env var to decide where to write (faiss for local
dev, pinecone for production) -- see vectorstore/__init__.py.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from ingestion.scrape import scrape_all
from ingestion.chunking import chunk_text
from ingestion.embed import Embedder, EMBEDDING_DIMENSION
from vectorstore import get_vector_store, Document


def run():
    print("[ingestion] scraping LangChain docs...")
    pages = scrape_all()
    print(f"[ingestion] fetched {len(pages)} pages")

    print("[ingestion] chunking...")
    all_chunks = []
    for page in pages:
        chunks = chunk_text(page["text"], source=page["url"])
        all_chunks.extend(chunks)
    print(f"[ingestion] produced {len(all_chunks)} chunks")

    print("[ingestion] embedding...")
    embedder = Embedder()
    texts = [c["text"] for c in all_chunks]
    embeddings = embedder.embed(texts)

    print("[ingestion] writing to vector store...")
    store = get_vector_store(dimension=EMBEDDING_DIMENSION)
    documents = [
        Document(id=c["id"], text=c["text"], metadata=c["metadata"])
        for c in all_chunks
    ]
    store.upsert(documents, embeddings)

    print(f"[ingestion] done. {len(documents)} chunks indexed.")


if __name__ == "__main__":
    run()
