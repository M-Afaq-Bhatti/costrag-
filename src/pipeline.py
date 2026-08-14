"""
Three pipelines sharing the same retrieval layer:

  BaselinePipeline  — always retrieves + always calls the large model.
                       No cache, no routing. This is the naive system
                       CostWise RAG is benchmarked against.

  OptimizedPipeline  — checks the semantic cache first (near-zero cost/
                       latency on a hit), otherwise classifies query
                       complexity and routes to the small or large model
                       BEFORE generating. If the pre-classification is
                       wrong, there is no recovery.

  CascadePipeline    — Mode 3. Same cache check, but instead of trusting a
                       pre-classification, it verifies the actual generated
                       answer. Obviously-complex queries still skip straight
                       to the large model; everything else fires the small
                       AND large model in parallel, accepts the small
                       model's answer if a free groundedness check passes,
                       and falls back to the (already in-flight) large
                       model's answer if it doesn't.

All three return a uniform metrics dict so the Streamlit dashboard can plot
them on identical axes.
"""

import threading
from config import (
    SMALL_MODEL, LARGE_MODEL, DEFAULT_TOP_K,
    CASCADE_HIGH_COMPLEXITY_THRESHOLD, GROUNDEDNESS_THRESHOLD,
)
from src.utils import now_ms
from src.verifier import verify_answer

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


class CascadePipeline:
    """
    Mode 3: confidence-gated speculative cascade.

    Cost accounting note: a synchronous Groq HTTP call can't be reliably
    cancelled mid-flight once dispatched, so whenever the large model is
    fired speculatively it is billed once it completes, whether or not its
    answer ends up being used. `total_latency_ms` reflects only the time to
    a *verified* answer (what a user would actually wait for) — the
    background thread is joined afterward purely to capture its token/cost
    usage for accurate logging, which does not affect the reported latency.
    """

    def __init__(self, vectorstore, embedder, llm_client, cache, classifier,
                 top_k: int = DEFAULT_TOP_K,
                 high_complexity_threshold: float = CASCADE_HIGH_COMPLEXITY_THRESHOLD,
                 groundedness_threshold: float = GROUNDEDNESS_THRESHOLD):
        self.vs = vectorstore
        self.embedder = embedder
        self.llm = llm_client
        self.cache = cache
        self.classifier = classifier
        self.top_k = top_k
        self.high_complexity_threshold = high_complexity_threshold
        self.groundedness_threshold = groundedness_threshold

    def answer(self, question: str) -> dict:
        t_start = now_ms()

        q_vec = self.embedder.embed(question)
        t_embed = now_ms()

        cache_hit = self.cache.lookup(question, q_vec)
        t_cache = now_ms()

        if cache_hit is not None:
            t_end = now_ms()
            return {
                "mode": "cascade",
                "question": question,
                "answer": cache_hit.answer,
                "model_used": "semantic_cache",
                "cache_hit": True,
                "complexity_score": None,
                "cascade_path": False,
                "escalated": False,
                "groundedness_score": None,
                "cost_usd": 0.0,
                "large_call_wasted_cost": 0.0,
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
        retrieved = self.vs.search(q_vec, top_k=self.top_k)
        t_retrieve = now_ms()
        context = _format_context(retrieved)
        user_prompt = USER_TEMPLATE.format(context=context, question=question)

        # ── Obviously complex: skip the cascade, go straight to the large model ──
        if complexity_score > self.high_complexity_threshold:
            gen = self.llm.generate(model=LARGE_MODEL, system_prompt=SYSTEM_PROMPT,
                                     user_prompt=user_prompt)
            t_end = now_ms()
            if gen["text"]:
                self.cache.store(question, q_vec, gen["text"], LARGE_MODEL)
            return {
                "mode": "cascade",
                "question": question,
                "answer": gen["text"],
                "model_used": LARGE_MODEL,
                "cache_hit": False,
                "complexity_score": complexity_score,
                "cascade_path": False,
                "escalated": False,
                "groundedness_score": None,
                "cost_usd": gen["cost_usd"],
                "large_call_wasted_cost": 0.0,
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

        # ── Ambiguous / simple: fire small + large in parallel ──────────────
        large_holder = {}

        def _run_large():
            large_holder["result"] = self.llm.generate(
                model=LARGE_MODEL, system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt
            )

        large_thread = threading.Thread(target=_run_large, daemon=True)
        large_thread.start()

        small_gen = self.llm.generate(model=SMALL_MODEL, system_prompt=SYSTEM_PROMPT,
                                       user_prompt=user_prompt)
        t_small_done = now_ms()

        is_grounded, ground_score, reason = verify_answer(
            small_gen["text"], retrieved, self.embedder, threshold=self.groundedness_threshold
        )
        t_verify = now_ms()

        if is_grounded:
            # Accept the small model's answer without waiting for the large
            # model's thread. Reported latency stops here.
            user_latency_ms = t_verify - t_start

            large_thread.join(timeout=30)  # only to capture cost, doesn't affect latency above
            large_gen = large_holder.get("result")
            wasted_cost = large_gen["cost_usd"] if large_gen else 0.0

            if small_gen["text"]:
                self.cache.store(question, q_vec, small_gen["text"], SMALL_MODEL)

            return {
                "mode": "cascade",
                "question": question,
                "answer": small_gen["text"],
                "model_used": SMALL_MODEL,
                "cache_hit": False,
                "complexity_score": complexity_score,
                "cascade_path": True,
                "escalated": False,
                "groundedness_score": ground_score,
                "cost_usd": small_gen["cost_usd"] + wasted_cost,
                "large_call_wasted_cost": wasted_cost,
                "prompt_tokens": small_gen["prompt_tokens"],
                "completion_tokens": small_gen["completion_tokens"],
                "embed_latency_ms": t_embed - t_start,
                "retrieval_latency_ms": t_retrieve - t_cache,
                "generation_latency_ms": t_small_done - t_retrieve,
                "cache_check_latency_ms": t_cache - t_embed,
                "total_latency_ms": user_latency_ms,
                "retrieved_sources": [f"p.{c.page}" for c, _ in retrieved],
                "error": small_gen["error"],
            }

        else:
            # Not grounded: fall back to the large model's (already
            # in-flight) answer. Now the wait for it is unavoidable.
            large_thread.join(timeout=60)
            large_gen = large_holder.get("result") or {
                "text": "", "cost_usd": 0.0, "prompt_tokens": 0,
                "completion_tokens": 0, "error": "large_model_thread_timed_out",
            }
            t_end = now_ms()

            if large_gen["text"]:
                self.cache.store(question, q_vec, large_gen["text"], LARGE_MODEL)

            return {
                "mode": "cascade",
                "question": question,
                "answer": large_gen["text"],
                "model_used": LARGE_MODEL,
                "cache_hit": False,
                "complexity_score": complexity_score,
                "cascade_path": True,
                "escalated": True,
                "groundedness_score": ground_score,
                "cost_usd": small_gen["cost_usd"] + large_gen["cost_usd"],
                "large_call_wasted_cost": 0.0,
                "prompt_tokens": large_gen["prompt_tokens"],
                "completion_tokens": large_gen["completion_tokens"],
                "embed_latency_ms": t_embed - t_start,
                "retrieval_latency_ms": t_retrieve - t_cache,
                "generation_latency_ms": t_end - t_retrieve,
                "cache_check_latency_ms": t_cache - t_embed,
                "total_latency_ms": t_end - t_start,
                "retrieved_sources": [f"p.{c.page}" for c, _ in retrieved],
                "error": large_gen["error"],
            }
