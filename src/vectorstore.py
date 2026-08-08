"""
Thin FAISS wrapper. Uses IndexFlatIP (inner product) over normalized
vectors, which is mathematically equivalent to cosine similarity —
free, exact, and more than fast enough for a single-document knowledge base.
"""

import pickle
from typing import List

import faiss
import numpy as np

from config import FAISS_INDEX_PATH, FAISS_META_PATH, EMBEDDING_DIM
from src.ingest import Chunk


class VectorStore:
    def __init__(self, dim: int = EMBEDDING_DIM):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: List[Chunk] = []

    def build(self, chunks: List[Chunk], embedder):
        self.chunks = chunks
        texts = [c.text for c in chunks]
        vecs = embedder.embed(texts)
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(np.ascontiguousarray(vecs, dtype="float32"))

    def search(self, query_vec: np.ndarray, top_k: int = 4):
        if self.index.ntotal == 0:
            return []
        query_vec = np.ascontiguousarray(query_vec.reshape(1, -1), dtype="float32")
        scores, idxs = self.index.search(query_vec, min(top_k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    def save(self, index_path: str = FAISS_INDEX_PATH, meta_path: str = FAISS_META_PATH):
        faiss.write_index(self.index, index_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self, index_path: str = FAISS_INDEX_PATH, meta_path: str = FAISS_META_PATH):
        self.index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            self.chunks = pickle.load(f)

    @property
    def is_ready(self) -> bool:
        return self.index is not None and self.index.ntotal > 0
