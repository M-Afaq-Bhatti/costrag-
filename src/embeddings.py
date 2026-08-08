"""
Free, local embedding model (no API key, no per-call cost, no PyTorch).

Uses fastembed - an ONNX-runtime based embedding library. Chosen over
sentence-transformers specifically because it has no PyTorch dependency,
which keeps the install small and reliable on Streamlit Community Cloud's
free tier (torch installs alone can approach the free tier's memory ceiling).

Model: BAAI/bge-small-en-v1.5 - 384-dim, ~67MB, strong retrieval quality
for its size.
"""

import numpy as np
from fastembed import TextEmbedding

from config import EMBEDDING_MODEL_NAME
from src.utils import normalize


class Embedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self._model_name = model_name
        self.model = TextEmbedding(model_name=model_name)

    def embed(self, texts):
        """texts: str or list[str] -> normalized (n, d) numpy array (or (d,) for a single string)."""
        single = isinstance(texts, str)
        input_texts = [texts] if single else list(texts)

        vecs = np.array(list(self.model.embed(input_texts)), dtype="float32")
        vecs = normalize(vecs)
        return vecs[0] if single else vecs

    def get_sentence_embedding_dimension(self) -> int:
        """Kept for API parity with the rest of the codebase."""
        return int(self.embed("dimension probe").shape[0])
