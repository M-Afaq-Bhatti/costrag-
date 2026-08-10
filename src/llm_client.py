"""
Groq chat-completion wrapper.
Every call returns (text, prompt_tokens, completion_tokens, cost_usd, latency_ms)
so downstream pipeline code can log full cost/latency metrics per query.

Rate limiting uses a thread-safe sliding-window limiter rather than a fixed
minimum-interval throttle. This matters for the cascade pipeline (Mode 3),
which fires the small and large model calls concurrently from two threads:
a fixed-interval throttle is not thread-safe and would non-deterministically
delay whichever call loses the race, defeating the point of parallel
dispatch. A sliding window allows a legitimate burst of concurrent calls as
long as the rolling 60-second total stays under the configured budget.
"""

import time
import threading
from collections import deque

from groq import Groq, APIStatusError, RateLimitError

from config import MODEL_PRICING, MAX_ANSWER_TOKENS, GENERATION_TEMPERATURE
from src.utils import now_ms


class RateLimiter:
    """Thread-safe sliding-window rate limiter (allows bursts up to the limit)."""

    def __init__(self, requests_per_minute: int):
        self.limit = max(requests_per_minute, 1)
        self.window_s = 60.0
        self._calls = deque()
        self._lock = threading.Lock()

    def acquire(self):
        while True:
            with self._lock:
                now = time.time()
                while self._calls and now - self._calls[0] > self.window_s:
                    self._calls.popleft()
                if len(self._calls) < self.limit:
                    self._calls.append(now)
                    return
                wait = self.window_s - (now - self._calls[0]) + 0.05
            time.sleep(max(wait, 0.05))

    def update_limit(self, requests_per_minute: int):
        with self._lock:
            self.limit = max(requests_per_minute, 1)


class LLMClient:
    def __init__(self, api_key: str, requests_per_minute: int = 25):
        self.client = Groq(api_key=api_key)
        self.rate_limiter = RateLimiter(requests_per_minute)

    def set_rate(self, requests_per_minute: int):
        self.rate_limiter.update_limit(requests_per_minute)

    def _cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        pricing = MODEL_PRICING.get(model, {"input": 0.0, "output": 0.0})
        return (prompt_tokens / 1_000_000) * pricing["input"] + \
               (completion_tokens / 1_000_000) * pricing["output"]

    def generate(self, model: str, system_prompt: str, user_prompt: str,
                 max_tokens: int = MAX_ANSWER_TOKENS, max_retries: int = 3):
        """Thread-safe: safe to call from multiple threads concurrently
        (e.g. the cascade pipeline's speculative parallel dispatch)."""
        self.rate_limiter.acquire()
        t0 = now_ms()
        last_err = None

        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=GENERATION_TEMPERATURE,
                )
                latency_ms = now_ms() - t0
                text = resp.choices[0].message.content or ""
                usage = resp.usage
                prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                completion_tokens = getattr(usage, "completion_tokens", 0) or 0
                cost = self._cost(model, prompt_tokens, completion_tokens)
                return {
                    "text": text.strip(),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "cost_usd": cost,
                    "latency_ms": latency_ms,
                    "model": model,
                    "error": None,
                }
            except RateLimitError as e:
                last_err = e
                time.sleep(min(2 ** attempt * 1.5, 15))
                continue
            except APIStatusError as e:
                last_err = e
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(1.0)
                continue

        return {
            "text": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cost_usd": 0.0,
            "latency_ms": now_ms() - t0,
            "model": model,
            "error": str(last_err) if last_err else "unknown error",
        }