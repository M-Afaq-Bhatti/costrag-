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
SMALL_MODEL = "llama-3.1-8b-instant"
LARGE_MODEL = "llama-3.3-70b-versatile"
JUDGE_MODEL = "openai/gpt-oss-120b"

# Pricing in USD per 1,000,000 tokens (input, output).
# Source: Groq public rate card, verified Aug 2026. Update here if Groq
# changes pricing — the whole cost dashboard reads from this table.
MODEL_PRICING = {
    SMALL_MODEL: {"input": 0.05, "output": 0.08},
    LARGE_MODEL: {"input": 0.59, "output": 0.79},
    JUDGE_MODEL: {"input": 0.15, "output": 0.60},
}

# ── Retrieval ────────────────────────────────────────────────────────
CHUNK_SIZE_CHARS = 1400          # ≈ 300-350 words per chunk
CHUNK_OVERLAP_CHARS = 200
DEFAULT_TOP_K = 4

# ── Semantic cache ───────────────────────────────────────────────────
DEFAULT_CACHE_THRESHOLD = 0.92   # cosine similarity to count as a cache hit

# ── Complexity router ────────────────────────────────────────────────
DEFAULT_COMPLEXITY_THRESHOLD = 0.0   # score > threshold => route to LARGE_MODEL

# ── Rate limiting (Groq free tier ≈ 30 requests/min) ────────────────
DEFAULT_REQUESTS_PER_MINUTE = 25

# ── Generation ───────────────────────────────────────────────────────
MAX_ANSWER_TOKENS = 500
MAX_JUDGE_TOKENS = 250
GENERATION_TEMPERATURE = 0.1
