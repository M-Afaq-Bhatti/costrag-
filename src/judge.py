"""
Scores a generated answer against the golden reference answer using a
judge model that is DIFFERENT from both the small and large answer models,
which limits self-preference bias in the accuracy metric.
"""

from config import JUDGE_MODEL, MAX_JUDGE_TOKENS
from src.utils import safe_json_extract

JUDGE_SYSTEM_PROMPT = (
    "You are a strict grading assistant for a medical question-answering "
    "evaluation. You compare a MODEL ANSWER to a REFERENCE ANSWER for the "
    "same question and output ONLY a JSON object — no prose, no markdown "
    "fences. Grade on factual correctness and completeness relative to the "
    "reference, not writing style. Minor wording differences are fine; "
    "missing or contradicted facts are not."
)

JUDGE_USER_TEMPLATE = """QUESTION:
{question}

REFERENCE ANSWER:
{reference}

MODEL ANSWER:
{model_answer}

Score the MODEL ANSWER from 0.0 to 1.0:
- 1.0 = fully correct and complete relative to the reference
- 0.5 = partially correct, missing some key facts, or partially inaccurate
- 0.0 = incorrect, contradicts the reference, or fails to answer

Respond with exactly this JSON shape:
{{"score": <float 0.0-1.0>, "reasoning": "<one short sentence>"}}"""


def judge_answer(llm_client, question: str, reference: str, model_answer: str):
    user_prompt = JUDGE_USER_TEMPLATE.format(
        question=question, reference=reference, model_answer=model_answer or "(empty answer)"
    )
    result = llm_client.generate(
        model=JUDGE_MODEL,
        system_prompt=JUDGE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        max_tokens=MAX_JUDGE_TOKENS,
    )

    parsed = safe_json_extract(result["text"]) if result["text"] else None
    if parsed and "score" in parsed:
        try:
            score = float(parsed["score"])
            score = max(0.0, min(1.0, score))
        except (TypeError, ValueError):
            score = 0.0
        reasoning = parsed.get("reasoning", "")
    else:
        score = 0.0
        reasoning = "judge_parse_failed"

    return {
        "score": score,
        "reasoning": reasoning,
        "judge_cost_usd": result["cost_usd"],
        "judge_latency_ms": result["latency_ms"],
        "judge_error": result["error"],
    }
