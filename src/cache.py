"""
Semantic cache used only in optimized mode.
Stores every answered query's embedding + answer. On a new query, checks
cosine similarity against everything cached so far; a hit above the
threshold returns the stored answer at ~zero cost and near-zero latency.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from config import DEFAULT_CACHE_THRESHOLD
from src.utils import cosine_sim_matrix


@dataclass
class CacheEntry:
    query: str
    embedding: np.ndarray
    answer: str
    model_used: str


class SemanticCache:
    def __init__(self, threshold: float = DEFAULT_CACHE_THRESHOLD):
        self.threshold = threshold
        self.entries: List[CacheEntry] = []

    def lookup(self, query: str, query_vec: np.ndarray) -> Optional[CacheEntry]:
        if not self.entries:
            return None
        matrix = np.stack([e.embedding for e in self.entries])
        sims = cosine_sim_matrix(query_vec, matrix)
        best_idx = int(np.argmax(sims))
        if sims[best_idx] >= self.threshold:
            return self.entries[best_idx]
        return None

    def store(self, query: str, query_vec: np.ndarray, answer: str, model_used: str):
        self.entries.append(CacheEntry(query=query, embedding=query_vec,
                                        answer=answer, model_used=model_used))

    def reset(self):
        self.entries = []

    def __len__(self):
        return len(self.entries)
