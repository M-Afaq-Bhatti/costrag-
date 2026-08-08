"""
Groq chat-completion wrapper.
Every call returns (text, prompt_tokens, completion_tokens, cost_usd, latency_ms)
so downstream pipeline code can log full cost/latency metrics per query.
Includes simple retry + backoff to survive Groq's free-tier rate limits
when running a full evaluation of many queries back to back.
"""

import time
from groq import Groq, APIStatusError, RateLimitError

from config import MODEL_PRICING, MAX_ANSWER_TOKENS, GENERATION_TEMPERATURE
from src.utils import now_ms


class LLMClient:
    def __init__(self, api_key: str, requests_per_minute: int = 25):
        self.client = Groq(api_key=api_key)
        self.min_interval_s = 60.0 / max(requests_per_minute, 1)
        self._last_call_ts = 0.0

    def _pace(self):
        elapsed = time.time() - self._last_call_ts
        wait = self.min_interval_s - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_call_ts = time.time()

    def _cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        pricing = MODEL_PRICING.get(model, {"input": 0.0, "output": 0.0})
        return (prompt_tokens / 1_000_000) * pricing["input"] + \
               (completion_tokens / 1_000_000) * pricing["output"]

    def generate(self, model: str, system_prompt: str, user_prompt: str,
                 max_tokens: int = MAX_ANSWER_TOKENS, max_retries: int = 3):
        self._pace()
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
