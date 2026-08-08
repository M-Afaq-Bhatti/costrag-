"""
Two pipelines sharing the same retrieval layer:

  BaselinePipeline  — always retrieves + always calls the large model.
                       No cache, no routing. This is the naive system
                       CostWise RAG is benchmarked against.

  OptimizedPipeline  — checks the semantic cache first (near-zero cost/
                       latency on a hit), otherwise classifies query
                       complexity and routes to the small or large model.

Both return a uniform metrics dict so the Streamlit dashboard can plot
baseline vs optimized on identical axes.
"""

from config import SMALL_MODEL, LARGE_MODEL, DEFAULT_TOP_K
from src.utils import now_ms

SYSTEM_PROMPT = (
    "You are a precise clinical reference assistant. Answer the user's "
    "question using ONLY the provided context excerpts. If the context "
    "does not contain the answer, say so explicitly rather than guessing. "
    "Be concise and factual."
)

USER_TEMPLATE = """CONTEXT:
{context}

QUESTION:
{question}

Answer using only the context above."""


def _format_context(retrieved) -> str:
    parts = []
    for chunk, score in retrieved:
        parts.append(f"[p.{chunk.page}] {chunk.text}")
    return "\n\n".join(parts)


class BaselinePipeline:
    """Naive RAG: retrieve -> always large model. No cache, no routing."""

    def __init__(self, vectorstore, embedder, llm_client, top_k: int = DEFAULT_TOP_K):
        self.vs = vectorstore
        self.embedder = embedder
        self.llm = llm_client
        self.top_k = top_k

    def answer(self, question: str) -> dict:
        t_start = now_ms()

        q_vec = self.embedder.embed(question)
        t_embed = now_ms()

        retrieved = self.vs.search(q_vec, top_k=self.top_k)
        t_retrieve = now_ms()
        context = _format_context(retrieved)

        gen = self.llm.generate(
            model=LARGE_MODEL,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=USER_TEMPLATE.format(context=context, question=question),
        )
        t_end = now_ms()

        return {
            "mode": "baseline",
            "question": question,
            "answer": gen["text"],
            "model_used": LARGE_MODEL,
            "cache_hit": False,
            "complexity_score": None,
            "cost_usd": gen["cost_usd"],
            "prompt_tokens": gen["prompt_tokens"],
            "completion_tokens": gen["completion_tokens"],
            "embed_latency_ms": t_embed - t_start,
            "retrieval_latency_ms": t_retrieve - t_embed,
            "generation_latency_ms": t_end - t_retrieve,
            "total_latency_ms": t_end - t_start,
            "retrieved_sources": [f"p.{c.page}" for c, _ in retrieved],
            "error": gen["error"],
        }


class OptimizedPipeline:
    """Cache -> complexity router -> small/large model."""

    def __init__(self, vectorstore, embedder, llm_client, cache, classifier,
                 top_k: int = DEFAULT_TOP_K):
        self.vs = vectorstore
        self.embedder = embedder
        self.llm = llm_client
        self.cache = cache
        self.classifier = classifier
        self.top_k = top_k

    def answer(self, question: str) -> dict:
        t_start = now_ms()

        q_vec = self.embedder.embed(question)
        t_embed = now_ms()

        cache_hit = self.cache.lookup(question, q_vec)
        t_cache = now_ms()

        if cache_hit is not None:
            t_end = now_ms()
            return {
                "mode": "optimized",
                "question": question,
                "answer": cache_hit.answer,
                "model_used": "semantic_cache",
                "cache_hit": True,
                "complexity_score": None,
                "cost_usd": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "embed_latency_ms": t_embed - t_start,
                "retrieval_latency_ms": 0.0,
                "generation_latency_ms": 0.0,
                "cache_check_latency_ms": t_cache - t_embed,
                "total_latency_ms": t_end - t_start,
                "retrieved_sources": [],
                "error": None,
            }

        complexity_score = self.classifier.score(question, q_vec)
        chosen_model = LARGE_MODEL if complexity_score > self.classifier.threshold else SMALL_MODEL

        retrieved = self.vs.search(q_vec, top_k=self.top_k)
        t_retrieve = now_ms()
        context = _format_context(retrieved)

        gen = self.llm.generate(
            model=chosen_model,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=USER_TEMPLATE.format(context=context, question=question),
        )
        t_end = now_ms()

        if gen["text"]:
            self.cache.store(question, q_vec, gen["text"], chosen_model)

        return {
            "mode": "optimized",
            "question": question,
            "answer": gen["text"],
            "model_used": chosen_model,
            "cache_hit": False,
            "complexity_score": complexity_score,
            "cost_usd": gen["cost_usd"],
            "prompt_tokens": gen["prompt_tokens"],
            "completion_tokens": gen["completion_tokens"],
            "embed_latency_ms": t_embed - t_start,
            "retrieval_latency_ms": t_retrieve - t_cache,
            "generation_latency_ms": t_end - t_retrieve,
            "cache_check_latency_ms": t_cache - t_embed,
            "total_latency_ms": t_end - t_start,
            "retrieved_sources": [f"p.{c.page}" for c, _ in retrieved],
            "error": gen["error"],
        }
