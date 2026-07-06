"""
Vector store abstraction.

Why this exists: we don't want the agent or ingestion code to know or care
whether documents live in Pinecone (production) or FAISS (local dev/tests).
Swapping backends should mean changing one line of config, not rewriting
retrieval logic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Document:
    """A single retrievable chunk with its metadata."""
    id: str
    text: str
    metadata: dict
    score: float = 0.0


class VectorStore(ABC):
    """Common interface every vector store backend must implement."""

    @abstractmethod
    def upsert(self, documents: list[Document], embeddings: list[list[float]]) -> None:
        """Insert or update documents with their embeddings."""
        raise NotImplementedError

    @abstractmethod
    def query(self, embedding: list[float], top_k: int = 5) -> list[Document]:
        """Return the top_k most similar documents to the query embedding."""
        raise NotImplementedError

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Remove documents by id."""
        raise NotImplementedError
