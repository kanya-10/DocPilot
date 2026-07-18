"""
Retriever: embeds a query and fetches the most relevant chunks from the
vector store, formatted for injection into the agent's prompt.
"""

from ingestion.embed import Embedder, EMBEDDING_DIMENSION
from vectorstore import get_vector_store, Document

# Below this similarity score we treat retrieval as "nothing relevant found"
# rather than forcing a low-quality chunk into the prompt. Tuned empirically
# against real retrieval scores -- see evals/ for the harness this was
# validated against. Re-tune if you change embedding models, since different
# models produce different cosine-similarity score distributions.
RELEVANCE_THRESHOLD = 0.35


class Retriever:
    def __init__(self):
        self.embedder = Embedder()
        self.store = get_vector_store(dimension=EMBEDDING_DIMENSION)

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        # input_type="query" matters for e5-family models -- queries and
        # passages are embedded slightly differently by design.
        query_embedding = self.embedder.embed([query], input_type="query")[0]
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
