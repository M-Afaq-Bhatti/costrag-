#!/usr/bin/env python3
"""
Build the FAISS knowledge base ONCE, from the command line, before running
the Streamlit app.

Run this whenever your source PDF changes. The app auto-loads the resulting
index on every `streamlit run app.py` after that — no re-embedding, no
"Build knowledge base" click, no waiting, on every single run.

Usage (from the project root):
    python build_index.py
    python build_index.py --pdf path/to/your.pdf

On Windows PowerShell:
    python build_index.py
    python build_index.py --pdf "D:/costwise-rag/data/cardiology.pdf"
"""

import argparse
import json
import os
import sys
import time

from config import (
    DEFAULT_PDF_PATH, FAISS_INDEX_PATH, FAISS_META_PATH,
    INDEX_INFO_PATH, EMBEDDING_MODEL_NAME,
)
from src.ingest import build_chunks_from_pdf
from src.embeddings import Embedder
from src.vectorstore import VectorStore


def main():
    parser = argparse.ArgumentParser(
        description="Build the CostWise RAG FAISS index once, ahead of running the Streamlit app."
    )
    parser.add_argument("--pdf", default=DEFAULT_PDF_PATH,
                         help=f"Path to source PDF (default: {DEFAULT_PDF_PATH})")
    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        print(f"ERROR: PDF not found at: {args.pdf}")
        print("Place your PDF there, or pass --pdf <path>.")
        sys.exit(1)

    print(f"[1/4] Loading embedding model ({EMBEDDING_MODEL_NAME}) — first run downloads ~67MB...")
    embedder = Embedder()

    print(f"[2/4] Extracting and chunking: {args.pdf}")
    t0 = time.time()
    chunks = build_chunks_from_pdf(args.pdf, source_name=os.path.basename(args.pdf))
    if not chunks:
        print("ERROR: no extractable text found in this PDF. Is it a scanned image PDF?")
        sys.exit(1)
    print(f"       -> {len(chunks)} chunks")

    print("[3/4] Embedding chunks and building the FAISS index...")
    vs = VectorStore(dim=embedder.get_sentence_embedding_dimension())
    vs.build(chunks, embedder)
    vs.save(FAISS_INDEX_PATH, FAISS_META_PATH)

    print("[4/4] Writing index metadata...")
    info = {
        "source_pdf": os.path.basename(args.pdf),
        "n_chunks": len(chunks),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(INDEX_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. Saved:")
    print(f"  {FAISS_INDEX_PATH}")
    print(f"  {FAISS_META_PATH}")
    print(f"  {INDEX_INFO_PATH}")
    print("\nYou can now run:  streamlit run app.py")
    print("It will load this index automatically — no rebuild step in the UI.")


if __name__ == "__main__":
    main()
