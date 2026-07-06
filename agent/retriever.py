"""
Retriever: embeds a query and fetches the most relevant chunks from the
vector store, formatted for injection into the agent's prompt.
"""

from ingestion.embed import Embedder, EMBEDDING_DIMENSION
from vectorstore import get_vector_store, Document

# Below this similarity score we treat retrieval as "nothing relevant found"
# rather than forcing a low-quality chunk into the prompt. Tuned empirically
# against the eval set in evals/ -- see evals/README for the sweep.
RELEVANCE_THRESHOLD = 0.35


class Retriever:
    def __init__(self):
        self.embedder = Embedder()
        self.store = get_vector_store(dimension=EMBEDDING_DIMENSION)

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        query_embedding = self.embedder.embed([query])[0]
        results = self.store.query(query_embedding, top_k=top_k)
        return [doc for doc in results if doc.score >= RELEVANCE_THRESHOLD]

    @staticmethod
    def format_context(docs: list[Document]) -> str:
        """Format retrieved docs as numbered sources for citation in the prompt."""
        if not docs:
            return "(no relevant documentation found)"
        parts = []
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "unknown")
            parts.append(f"[{i}] (source: {source})\n{doc.text}")
        return "\n\n".join(parts)
