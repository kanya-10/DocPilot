"""
Vector store factory. This is the one place that knows which backend to use.

Usage:
    from vectorstore import get_vector_store
    store = get_vector_store()   # reads VECTOR_STORE_BACKEND env var

Set VECTOR_STORE_BACKEND=pinecone for production, or leave unset / set to
"faiss" for local dev and tests (no API key required).
"""

import os

from .base import VectorStore, Document

__all__ = ["VectorStore", "Document", "get_vector_store"]


def get_vector_store(dimension: int = 1536) -> VectorStore:
    backend = os.environ.get("VECTOR_STORE_BACKEND", "faiss").lower()

    if backend == "pinecone":
        from .pinecone_store import PineconeStore
        return PineconeStore(dimension=dimension)

    if backend == "faiss":
        from .faiss_store import FaissStore
        return FaissStore(dimension=dimension)

    raise ValueError(f"Unknown VECTOR_STORE_BACKEND: {backend!r} (expected 'pinecone' or 'faiss')")
