"""
Local embeddings via sentence-transformers -- no API key required.
"""

from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384 dimensions, fast, good quality/speed tradeoff
EMBEDDING_DIMENSION = 384


class Embedder:
    def __init__(self, model: str = EMBEDDING_MODEL):
        self.model = SentenceTransformer(model)

    def embed(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=False)
        return embeddings.tolist()
