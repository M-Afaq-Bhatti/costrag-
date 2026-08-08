import time
import json
import numpy as np


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def cosine_sim_matrix(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """query_vec: (d,) normalized. matrix: (n, d) normalized. Returns (n,) sims."""
    if matrix.shape[0] == 0:
        return np.array([])
    return matrix @ query_vec


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    norm[norm == 0] = 1e-9
    return vec / norm


def safe_json_extract(text: str):
    """Best-effort extraction of a JSON object from an LLM response that may
    include stray prose or markdown code fences around the JSON."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    candidate = text[start:end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def percentile(values, p):
    if not values:
        return 0.0
    return float(np.percentile(np.array(values), p))
