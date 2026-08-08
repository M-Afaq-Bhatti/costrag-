"""
Query complexity classifier used by the optimized pipeline's router.

Rather than asking an LLM to self-report a difficulty score (expensive,
slow, and not really "classification"), this builds two centroids in
embedding space from a small set of generic example queries — one for
simple/factual phrasing, one for complex/multi-hop phrasing — and scores
new queries by which centroid they sit closer to. This is a legitimate,
lightweight few-shot classifier: no extra LLM call, sub-millisecond scoring.

A small keyword signal is blended in because certain comparison/causal
words ("compare", "why", "relationship between", "versus") are strong,
cheap, domain-agnostic markers of multi-hop reasoning that pure semantic
similarity sometimes under-weights.
"""

import numpy as np
from config import DEFAULT_COMPLEXITY_THRESHOLD

SIMPLE_EXAMPLES = [
    "What is the definition of X?",
    "What is the normal range for this value?",
    "What class of drug is this?",
    "What is the recommended dosage?",
    "What are the symptoms of this condition?",
    "What year was this guideline published?",
    "What is the primary function of this structure?",
    "List the diagnostic criteria for this condition.",
]

COMPLEX_EXAMPLES = [
    "Compare and contrast the treatment approaches for these two conditions.",
    "Explain why this mechanism leads to that clinical outcome.",
    "How does the pathophysiology described in one section relate to the "
    "management strategy described elsewhere?",
    "What is the relationship between these two risk factors and the "
    "resulting complication?",
    "Given the interaction between these two systems, why is this "
    "approach preferred over the alternative?",
    "Synthesize the findings from multiple sections to explain the "
    "overall progression of this disease.",
    "How do the guidelines differ between these two patient populations "
    "and what accounts for the difference?",
]

COMPLEXITY_KEYWORDS = [
    "compare", "versus", " vs ", "why", "relationship between",
    "difference between", "how does", "explain", "contrast",
    "synthesize", "impact of", "effect of", "interaction between",
]


class ComplexityClassifier:
    def __init__(self, embedder, threshold: float = DEFAULT_COMPLEXITY_THRESHOLD):
        self.embedder = embedder
        self.threshold = threshold
        simple_vecs = embedder.embed(SIMPLE_EXAMPLES)
        complex_vecs = embedder.embed(COMPLEX_EXAMPLES)
        self.simple_centroid = simple_vecs.mean(axis=0)
        self.complex_centroid = complex_vecs.mean(axis=0)
        self.simple_centroid /= np.linalg.norm(self.simple_centroid)
        self.complex_centroid /= np.linalg.norm(self.complex_centroid)

    def score(self, query: str, query_vec: np.ndarray = None) -> float:
        """Returns a signed complexity score. > threshold => complex/large model."""
        if query_vec is None:
            query_vec = self.embedder.embed(query)

        sim_simple = float(query_vec @ self.simple_centroid)
        sim_complex = float(query_vec @ self.complex_centroid)
        semantic_score = sim_complex - sim_simple

        lowered = query.lower()
        keyword_hits = sum(1 for kw in COMPLEXITY_KEYWORDS if kw in lowered)
        keyword_bonus = 0.05 * keyword_hits

        return semantic_score + keyword_bonus

    def classify(self, query: str, query_vec: np.ndarray = None) -> str:
        return "complex" if self.score(query, query_vec) > self.threshold else "simple"
