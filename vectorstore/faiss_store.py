"""
FAISS-backed VectorStore. Used for local development and tests so we don't
need a live Pinecone API key/account just to run the test suite or iterate
on retrieval logic.

Persistence: unlike Pinecone, FAISS has no server behind it -- the index
only exists in the process that created it. Ingestion runs in one process
and the API runs in another, so we persist the index + metadata to disk
after ingestion and load it back on API startup. Without this, retrieval
silently returns nothing every time.
"""

import os
import pickle

import faiss
import numpy as np

from .base import VectorStore, Document

DEFAULT_INDEX_DIR = "faiss_index"


class FaissStore(VectorStore):
    def __init__(self, dimension: int = 1536, index_dir: str = DEFAULT_INDEX_DIR):
        self.dimension = dimension
        self.index_dir = index_dir
        self._id_to_doc: dict[int, Document] = {}
        self._doc_id_to_faiss_id: dict[str, int] = {}
        self._next_id = 0

        if self._persisted_files_exist():
            self._load()
        else:
            self.index = faiss.IndexFlatIP(dimension)

    def _index_path(self) -> str:
        return os.path.join(self.index_dir, "index.faiss")

    def _meta_path(self) -> str:
        return os.path.join(self.index_dir, "meta.pkl")

    def _persisted_files_exist(self) -> bool:
        return os.path.exists(self._index_path()) and os.path.exists(self._meta_path())

    def _load(self) -> None:
        self.index = faiss.read_index(self._index_path())
        with open(self._meta_path(), "rb") as f:
            meta = pickle.load(f)
        self._id_to_doc = meta["id_to_doc"]
        self._doc_id_to_faiss_id = meta["doc_id_to_faiss_id"]
        self._next_id = meta["next_id"]

    def save(self) -> None:
        os.makedirs(self.index_dir, exist_ok=True)
        faiss.write_index(self.index, self._index_path())
        with open(self._meta_path(), "wb") as f:
            pickle.dump({
                "id_to_doc": self._id_to_doc,
                "doc_id_to_faiss_id": self._doc_id_to_faiss_id,
                "next_id": self._next_id,
            }, f)

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec if norm == 0 else vec / norm

    def upsert(self, documents: list[Document], embeddings: list[list[float]]) -> None:
        vectors = np.array([self._normalize(np.array(e, dtype="float32")) for e in embeddings])
        self.index.add(vectors)
        for doc in documents:
            self._id_to_doc[self._next_id] = doc
            self._doc_id_to_faiss_id[doc.id] = self._next_id
            self._next_id += 1
        self.save()

    def query(self, embedding: list[float], top_k: int = 5) -> list[Document]:
        vec = self._normalize(np.array(embedding, dtype="float32")).reshape(1, -1)
        scores, indices = self.index.search(vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx not in self._id_to_doc:
                continue
            doc = self._id_to_doc[idx]
            results.append(Document(id=doc.id, text=doc.text, metadata=doc.metadata, score=float(score)))
        return results

    def delete(self, ids: list[str]) -> None:
        for doc_id in ids:
            faiss_id = self._doc_id_to_faiss_id.pop(doc_id, None)
            if faiss_id is not None:
                self._id_to_doc.pop(faiss_id, None)
        self.save()