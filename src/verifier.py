"""
Free groundedness verifier for the cascade pipeline (Mode 3).

Decides whether the small model's answer should be trusted, using only the
local embedding model that's already loaded (no extra LLM call — that would
reintroduce the exact cost/latency problem the cascade is meant to solve).

Two cheap signals are combined:
  1. Embedding similarity between the generated answer and its strongest
     supporting retrieved chunk — low similarity suggests the answer isn't
     actually grounded in the retrieved context.
  2. A hedge-phrase check — if the small model itself says it couldn't find
     the answer, that's a direct, free signal to escalate.
"""

import numpy as np
from config import GROUNDEDNESS_THRESHOLD

HEDGE_PHRASES = [
    "cannot determine", "cannot be determined", "not stated", "not mentioned",
    "not provided", "insufficient information", "does not contain",
    "no information", "unable to answer", "i don't know", "i do not know",
    "unclear from the text", "the text does not", "context does not",
]


def _contains_hedge(answer_text: str) -> bool:
    lowered = (answer_text or "").lower()
    return any(phrase in lowered for phrase in HEDGE_PHRASES)


def groundedness_score(answer_text: str, context_chunks, embedder) -> float:
    """Cosine similarity between the answer and its strongest supporting
    context chunk. context_chunks: list of (Chunk, retrieval_score) tuples,
    the same structure VectorStore.search returns."""
    if not answer_text or not context_chunks:
        return 0.0
    answer_vec = embedder.embed(answer_text)
    chunk_texts = [c.text for c, _ in context_chunks]
    chunk_vecs = embedder.embed(chunk_texts)
    sims = chunk_vecs @ answer_vec
    return float(np.max(sims))


def verify_answer(answer_text: str, context_chunks, embedder,
                   threshold: float = GROUNDEDNESS_THRESHOLD):
    """Returns (is_grounded: bool, score: float, reason: str)."""
    if not answer_text or not answer_text.strip():
        return False, 0.0, "empty_answer"
    if _contains_hedge(answer_text):
        return False, 0.0, "hedge_phrase_detected"

    score = groundedness_score(answer_text, context_chunks, embedder)
    if score < threshold:
        return False, score, "low_groundedness"
    return True, score, "grounded"