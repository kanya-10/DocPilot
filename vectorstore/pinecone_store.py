"""
Pinecone-backed VectorStore. This is the production backend.

Requires env vars: PINECONE_API_KEY, PINECONE_INDEX_NAME
"""

import os
from pinecone import Pinecone, ServerlessSpec

from .base import VectorStore, Document


class PineconeStore(VectorStore):
    def __init__(self, index_name: str | None = None, dimension: int = 1536):
        api_key = os.environ["PINECONE_API_KEY"]
        self.index_name = index_name or os.environ.get("PINECONE_INDEX_NAME", "docpilot")
        self.client = Pinecone(api_key=api_key)

        existing = [idx["name"] for idx in self.client.list_indexes()]
        if self.index_name not in existing:
            self.client.create_index(
                name=self.index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        self.index = self.client.Index(self.index_name)

    def upsert(self, documents: list[Document], embeddings: list[list[float]]) -> None:
        vectors = [
            {
                "id": doc.id,
                "values": emb,
                "metadata": {**doc.metadata, "text": doc.text},
            }
            for doc, emb in zip(documents, embeddings)
        ]
        # Pinecone recommends batching upserts to avoid request size limits
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            self.index.upsert(vectors=vectors[i:i + batch_size])

    def query(self, embedding: list[float], top_k: int = 5) -> list[Document]:
        result = self.index.query(vector=embedding, top_k=top_k, include_metadata=True)
        docs = []
        for match in result["matches"]:
            meta = dict(match["metadata"])
            text = meta.pop("text", "")
            docs.append(Document(id=match["id"], text=text, metadata=meta, score=match["score"]))
        return docs

    def delete(self, ids: list[str]) -> None:
        self.index.delete(ids=ids)
