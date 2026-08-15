"""
Central configuration for CostWise RAG.
Change model names / pricing here if Groq updates its catalogue.
"""

import os

# ── Paths ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
DEFAULT_PDF_PATH = os.path.join(DATA_DIR, "source.pdf")
DEFAULT_DATASET_PATH = os.path.join(DATA_DIR, "golden_dataset.json")
FAISS_INDEX_PATH = os.path.join(STORAGE_DIR, "index.faiss")
FAISS_META_PATH = os.path.join(STORAGE_DIR, "meta.pkl")
INDEX_INFO_PATH = os.path.join(STORAGE_DIR, "index_info.json")

os.makedirs(STORAGE_DIR, exist_ok=True)

# ── Embedding model (FREE, runs locally, no API cost) ──────────────────
# fastembed (ONNX runtime, no PyTorch dependency) — small install footprint,
# well suited to Streamlit Community Cloud's free tier. 384-dim, ~67MB.
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

# ── Groq LLMs ────────────────────────────────────────────────────────
# SMALL_MODEL handles simple/factual queries routed by the classifier
# LARGE_MODEL handles complex/multi-hop queries + is always used in baseline
# JUDGE_MODEL scores answers against the golden reference (kept distinct
# from both answer models to reduce self-preference bias)
#
# NOTE (Aug 2026): llama-3.1-8b-instant and llama-3.3-70b-versatile were
# deprecated by Groq on June 17, 2026, with a hard shutdown on Aug 16, 2026.
# Migrated to Groq's official recommended replacements. If you see cost=$0
# and accuracy=0% across the board again, check console.groq.com/docs/deprecations
# first — it means a configured model has been retired, not a calculation bug.
SMALL_MODEL = "openai/gpt-oss-20b"
LARGE_MODEL = "openai/gpt-oss-120b"
JUDGE_MODEL = "qwen/qwen3-32b"

# Pricing in USD per 1,000,000 tokens (input, output).
# Verified against Groq's public rate card, Aug 2026. Update here if Groq
# changes pricing — the whole cost dashboard reads from this table.
MODEL_PRICING = {
    SMALL_MODEL: {"input": 0.075, "output": 0.30},
    LARGE_MODEL: {"input": 0.15, "output": 0.60},
    JUDGE_MODEL: {"input": 0.29, "output": 0.59},
}

# ── Retrieval ────────────────────────────────────────────────────────
CHUNK_SIZE_CHARS = 1400          # ≈ 300-350 words per chunk
CHUNK_OVERLAP_CHARS = 200
DEFAULT_TOP_K = 4

# ── Semantic cache ───────────────────────────────────────────────────
DEFAULT_CACHE_THRESHOLD = 0.92   # cosine similarity to count as a cache hit

# ── Complexity router ────────────────────────────────────────────────
DEFAULT_COMPLEXITY_THRESHOLD = 0.0   # score > threshold => route to LARGE_MODEL (Mode 2)

# ── Cascade mode (Mode 3) ────────────────────────────────────────────
# Classifier score above this is "confident enough" to skip the cascade
# entirely and go straight to the large model — avoids wasting a small-model
# attempt on a query that's very unlikely to pass verification anyway.
# Must be higher (stricter) than DEFAULT_COMPLEXITY_THRESHOLD.
CASCADE_HIGH_COMPLEXITY_THRESHOLD = 0.15

# Cosine similarity between the small model's answer and its best-matching
# retrieved chunk. Below this, the answer is treated as ungrounded and the
# (already in-flight) large model's answer is used instead.
GROUNDEDNESS_THRESHOLD = 0.55

# ── Rate limiting (Groq free tier ≈ 30 requests/min) ────────────────
DEFAULT_REQUESTS_PER_MINUTE = 25

# ── Generation ───────────────────────────────────────────────────────
MAX_ANSWER_TOKENS = 500
MAX_JUDGE_TOKENS = 250
GENERATION_TEMPERATURE = 0.1