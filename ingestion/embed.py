"""
Embeddings via Pinecone's hosted Inference API -- no local model, no PyTorch,
no large in-process memory footprint.
"""

import os

from pinecone import Pinecone

EMBEDDING_MODEL = "multilingual-e5-large"
EMBEDDING_DIMENSION = 1024

# Pinecone's Inference API caps input batch size per call for this model.
# Ingestion routinely sends hundreds of chunks at once, so we batch.
MAX_BATCH_SIZE = 90


class Embedder:
    def __init__(self, model: str = EMBEDDING_MODEL):
        self.pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        self.model = model

    def embed(self, texts: list[str], input_type: str = "passage") -> list[list[float]]:
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), MAX_BATCH_SIZE):
            batch = texts[i:i + MAX_BATCH_SIZE]
            result = self.pc.inference.embed(
                model=self.model,
                inputs=batch,
                parameters={"input_type": input_type, "truncate": "END"},
            )
            all_embeddings.extend(item["values"] for item in result)
        return all_embeddings