"""
CostWise RAG — Cost, Latency & Accuracy Optimized Retrieval Pipeline
======================================================================
A dual-mode RAG system (baseline vs cache+router optimized) benchmarked
on an identical golden query set, deployed as a single Streamlit app.

Run locally:   streamlit run app.py
Deploy:        push to GitHub, connect the repo on share.streamlit.io
"""

import os
import json
import time
import traceback

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from config import (
    SMALL_MODEL, LARGE_MODEL, JUDGE_MODEL, MODEL_PRICING,
    DEFAULT_TOP_K, DEFAULT_CACHE_THRESHOLD, DEFAULT_COMPLEXITY_THRESHOLD,
    DEFAULT_REQUESTS_PER_MINUTE, DEFAULT_DATASET_PATH, DEFAULT_PDF_PATH,
)
from src.embeddings import Embedder
from src.ingest import build_chunks_from_pdf
from src.vectorstore import VectorStore
from src.cache import SemanticCache
from src.classifier import ComplexityClassifier
from src.llm_client import LLMClient
from src.pipeline import BaselinePipeline, OptimizedPipeline
from src.judge import judge_answer
from src.metrics import aggregate, build_comparison_table

# ──────────────────────────────────────────────────────────────────────
# Page config + styling
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CostWise RAG",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.metric-card {
    background: var(--background-color, #f8f9fb);
    border: 1px solid rgba(49, 51, 63, 0.15);
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 8px;
}
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
}
.badge-cache { background: #E1F5EE; color: #085041; }
.badge-small { background: #E6F1FB; color: #0C447C; }
.badge-large { background: #FAECE7; color: #712B13; }
.badge-baseline { background: #F1EFE8; color: #444441; }
h1, h2, h3 { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.title("🩺 CostWise RAG")
st.caption(
    "A cost, latency, and accuracy optimized RAG pipeline — benchmarked head-to-head "
    "against a naive single-model baseline on an identical evaluation set."
)

# ──────────────────────────────────────────────────────────────────────
# Session state initialization
# ──────────────────────────────────────────────────────────────────────
defaults = {
    "vectorstore": None,
    "n_chunks": 0,
    "dataset": None,
    "cache": None,
    "classifier": None,
    "llm_client": None,
    "api_key_used": None,
    "eval_rows": [],
    "baseline_summary": {},
    "optimized_summary": {},
    "comparison_df": None,
    "live_log": [],
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


@st.cache_resource(show_spinner="Loading free local embedding model (first run only)...")
def get_embedder():
    return Embedder()


embedder = get_embedder()

if st.session_state.cache is None:
    st.session_state.cache = SemanticCache(threshold=DEFAULT_CACHE_THRESHOLD)
if st.session_state.classifier is None:
    st.session_state.classifier = ComplexityClassifier(
        embedder, threshold=DEFAULT_COMPLEXITY_THRESHOLD
    )

# ──────────────────────────────────────────────────────────────────────
# Sidebar — setup & configuration
# ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Setup")

    api_key = st.text_input(
        "Groq API key",
        type="password",
        value=os.environ.get("GROQ_API_KEY", ""),
        help="Free key from console.groq.com. Never stored — only kept for this session.",
    )
    if api_key and api_key != st.session_state.api_key_used:
        rpm = st.session_state.get("rpm_setting", DEFAULT_REQUESTS_PER_MINUTE)
        st.session_state.llm_client = LLMClient(api_key=api_key, requests_per_minute=rpm)
        st.session_state.api_key_used = api_key

    st.divider()
    st.subheader("📚 Knowledge base")

    pdf_file = st.file_uploader("Upload source PDF", type=["pdf"])
    use_bundled_pdf = os.path.exists(DEFAULT_PDF_PATH)
    if not pdf_file and use_bundled_pdf:
        st.caption(f"Bundled PDF found: `{os.path.basename(DEFAULT_PDF_PATH)}`")

    if st.button("🔨 Build knowledge base", use_container_width=True):
        source = pdf_file if pdf_file is not None else (DEFAULT_PDF_PATH if use_bundled_pdf else None)
        if source is None:
            st.error("Upload a PDF first, or place one at data/source.pdf in the repo.")
        else:
            try:
                with st.spinner("Extracting text, chunking, and embedding..."):
                    chunks = build_chunks_from_pdf(
                        source, source_name=(pdf_file.name if pdf_file else "bundled document")
                    )
                    if not chunks:
                        st.error("No extractable text found in this PDF. Is it a scanned image PDF?")
                    else:
                        vs = VectorStore(dim=embedder.get_sentence_embedding_dimension())
                        vs.build(chunks, embedder)
                        st.session_state.vectorstore = vs
                        st.session_state.n_chunks = len(chunks)
                        st.success(f"Indexed {len(chunks)} chunks.")
            except Exception as e:  # noqa: BLE001
                st.error(f"Failed to build knowledge base: {e}")

    if st.session_state.vectorstore is not None:
        st.success(f"✅ Knowledge base ready — {st.session_state.n_chunks} chunks indexed.")
    else:
        st.warning("⚠️ Knowledge base not built yet.")

    st.divider()
    st.subheader("📋 Golden evaluation dataset")

    dataset_file = st.file_uploader("Upload golden_dataset.json", type=["json"])
    use_bundled_dataset = os.path.exists(DEFAULT_DATASET_PATH)

    def _load_dataset(raw_bytes_or_path):
        if isinstance(raw_bytes_or_path, str):
            with open(raw_bytes_or_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = json.loads(raw_bytes_or_path.getvalue().decode("utf-8"))
        if not isinstance(data, list):
            raise ValueError("Dataset JSON must be a list of question objects.")
        cleaned = []
        for i, item in enumerate(data):
            if "question" not in item or "reference_answer" not in item:
                continue
            cleaned.append({
                "id": item.get("id", i + 1),
                "difficulty": item.get("difficulty", "unknown"),
                "question": item["question"],
                "reference_answer": item["reference_answer"],
            })
        return cleaned

    if dataset_file is not None:
        try:
            st.session_state.dataset = _load_dataset(dataset_file)
            st.success(f"Loaded {len(st.session_state.dataset)} queries from upload.")
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not parse dataset: {e}")
    elif use_bundled_dataset and st.session_state.dataset is None:
        try:
            st.session_state.dataset = _load_dataset(DEFAULT_DATASET_PATH)
            st.caption(f"Loaded bundled dataset — {len(st.session_state.dataset)} queries.")
        except Exception as e:  # noqa: BLE001
            st.caption(f"Bundled dataset present but failed to parse: {e}")

    if st.session_state.dataset:
        n_simple = sum(1 for d in st.session_state.dataset if d["difficulty"] == "simple")
        n_complex = sum(1 for d in st.session_state.dataset if d["difficulty"] == "complex")
        st.caption(f"{len(st.session_state.dataset)} queries loaded — {n_simple} simple, {n_complex} complex.")
    else:
        st.warning("⚠️ No golden dataset loaded yet.")

    st.divider()
    with st.expander("🎛️ Advanced settings"):
        top_k = st.slider("Retrieval top-k", 2, 8, DEFAULT_TOP_K)
        cache_threshold = st.slider("Semantic cache similarity threshold", 0.80, 0.99, DEFAULT_CACHE_THRESHOLD, 0.01)
        complexity_threshold = st.slider("Complexity routing threshold", -0.3, 0.3, DEFAULT_COMPLEXITY_THRESHOLD, 0.01)
        rpm = st.slider("Groq requests / minute (pacing)", 5, 30, DEFAULT_REQUESTS_PER_MINUTE,
                         help="Keep below your Groq tier's rate limit to avoid 429 errors.")
        st.session_state["rpm_setting"] = rpm

        st.session_state.cache.threshold = cache_threshold
        st.session_state.classifier.threshold = complexity_threshold
        if st.session_state.llm_client is not None:
            st.session_state.llm_client.min_interval_s = 60.0 / max(rpm, 1)

        if st.button("🗑️ Reset semantic cache", use_container_width=True):
            st.session_state.cache.reset()
            st.success("Cache cleared.")

    st.divider()
    with st.expander("🧠 Model configuration"):
        st.markdown(f"""
        | Role | Model | Input $/1M | Output $/1M |
        |---|---|---|---|
        | Small (simple queries) | `{SMALL_MODEL}` | ${MODEL_PRICING[SMALL_MODEL]['input']} | ${MODEL_PRICING[SMALL_MODEL]['output']} |
        | Large (complex + baseline) | `{LARGE_MODEL}` | ${MODEL_PRICING[LARGE_MODEL]['input']} | ${MODEL_PRICING[LARGE_MODEL]['output']} |
        | Judge (accuracy scoring) | `{JUDGE_MODEL}` | ${MODEL_PRICING[JUDGE_MODEL]['input']} | ${MODEL_PRICING[JUDGE_MODEL]['output']} |

        Embeddings: `all-MiniLM-L6-v2` — free, runs locally, no API cost.
        """)

ready = st.session_state.vectorstore is not None and st.session_state.llm_client is not None

# ──────────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────────
tab_live, tab_eval, tab_about = st.tabs(["💬 Live query", "📊 Full evaluation", "🏗️ Architecture"])

# ── TAB 1: Live query ───────────────────────────────────────────────
with tab_live:
    if not ready:
        st.info("Enter a Groq API key and build the knowledge base in the sidebar to get started.")
    else:
        col_mode, col_q = st.columns([1, 3])
        with col_mode:
            mode = st.radio("Pipeline mode", ["Optimized", "Baseline"], index=0)
        with col_q:
            question = st.text_input("Ask a question about the document", placeholder="e.g. What is the definition of unstable angina?")

        ask = st.button("Ask", type="primary")

        if ask and question.strip():
            baseline_pipe = BaselinePipeline(st.session_state.vectorstore, embedder,
                                              st.session_state.llm_client, top_k=top_k)
            optimized_pipe = OptimizedPipeline(st.session_state.vectorstore, embedder,
                                                st.session_state.llm_client, st.session_state.cache,
                                                st.session_state.classifier, top_k=top_k)
            try:
                with st.spinner("Running pipeline..."):
                    result = (optimized_pipe if mode == "Optimized" else baseline_pipe).answer(question)
                st.session_state.live_log.insert(0, result)

                st.markdown("#### Answer")
                st.write(result["answer"] if result["answer"] else "_No answer generated (see error below)._")
                if result["error"]:
                    st.error(f"Generation error: {result['error']}")

                st.markdown("#### Query metrics")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total latency", f"{result['total_latency_ms']:.0f} ms")
                m2.metric("Cost", f"${result['cost_usd']:.5f}")

                if result.get("cache_hit"):
                    badge = '<span class="badge badge-cache">Semantic cache hit</span>'
                elif result["model_used"] == LARGE_MODEL:
                    badge = '<span class="badge badge-large">Large model</span>'
                elif result["model_used"] == SMALL_MODEL:
                    badge = '<span class="badge badge-small">Small model</span>'
                else:
                    badge = '<span class="badge badge-baseline">Baseline (large only)</span>'
                m3.markdown(f"**Routing**<br>{badge}", unsafe_allow_html=True)
                m4.metric("Complexity score", f"{result['complexity_score']:.3f}" if result["complexity_score"] is not None else "—")

                stages = {
                    "Embed query": result.get("embed_latency_ms", 0),
                    "Cache check": result.get("cache_check_latency_ms", 0),
                    "Retrieval": result.get("retrieval_latency_ms", 0),
                    "Generation": result.get("generation_latency_ms", 0),
                }
                fig = go.Figure(go.Bar(
                    x=list(stages.values()), y=list(stages.keys()), orientation="h",
                    marker_color="#378ADD",
                ))
                fig.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10),
                                   xaxis_title="milliseconds")
                st.plotly_chart(fig, use_container_width=True)

                if result.get("retrieved_sources"):
                    st.caption("Retrieved from: " + ", ".join(result["retrieved_sources"]))
            except Exception as e:  # noqa: BLE001
                st.error(f"Something went wrong: {e}")
                st.code(traceback.format_exc())

        if st.session_state.live_log:
            with st.expander(f"Query history ({len(st.session_state.live_log)})"):
                hist_df = pd.DataFrame([{
                    "mode": r["mode"], "question": r["question"][:60],
                    "model": r["model_used"], "cache_hit": r["cache_hit"],
                    "latency_ms": round(r["total_latency_ms"], 1),
                    "cost_usd": round(r["cost_usd"], 5),
                } for r in st.session_state.live_log])
                st.dataframe(hist_df, use_container_width=True, hide_index=True)

# ── TAB 2: Full evaluation ──────────────────────────────────────────
with tab_eval:
    if not ready:
        st.info("Enter a Groq API key and build the knowledge base in the sidebar to get started.")
    elif not st.session_state.dataset:
        st.info("Upload or bundle a golden_dataset.json in the sidebar to run the benchmark.")
    else:
        n_available = len(st.session_state.dataset)
        n_eval = st.slider("Number of queries to evaluate (runs baseline + optimized on each)",
                            2, n_available, min(20, n_available), 1)
        est_calls = n_eval * 4
        st.caption(
            f"This will make up to {est_calls} Groq API calls (answer + judge, both modes) "
            f"at your configured pacing (~{60/rpm:.1f}s between calls) — "
            f"roughly {est_calls * 60 / rpm / 60:.1f} minutes."
        )

        run = st.button("▶️ Run full evaluation", type="primary")

        if run:
            subset = st.session_state.dataset[:n_eval]
            baseline_pipe = BaselinePipeline(st.session_state.vectorstore, embedder,
                                              st.session_state.llm_client, top_k=top_k)
            optimized_pipe = OptimizedPipeline(st.session_state.vectorstore, embedder,
                                                st.session_state.llm_client, st.session_state.cache,
                                                st.session_state.classifier, top_k=top_k)

            progress = st.progress(0.0)
            status = st.empty()
            rows = []

            for i, item in enumerate(subset):
                q, ref = item["question"], item["reference_answer"]
                status.text(f"[{i+1}/{len(subset)}] {q[:70]}")

                try:
                    b_res = baseline_pipe.answer(q)
                    b_judge = judge_answer(st.session_state.llm_client, q, ref, b_res["answer"])
                    b_res.update({"difficulty": item["difficulty"], "accuracy_score": b_judge["score"],
                                  "judge_error": b_judge["judge_error"]})
                    rows.append(b_res)
                except Exception as e:  # noqa: BLE001
                    rows.append({"mode": "baseline", "question": q, "difficulty": item["difficulty"],
                                 "answer": "", "model_used": LARGE_MODEL, "cache_hit": False,
                                 "complexity_score": None, "cost_usd": 0, "total_latency_ms": 0,
                                 "accuracy_score": None, "error": str(e)})

                try:
                    o_res = optimized_pipe.answer(q)
                    o_judge = judge_answer(st.session_state.llm_client, q, ref, o_res["answer"])
                    o_res.update({"difficulty": item["difficulty"], "accuracy_score": o_judge["score"],
                                  "judge_error": o_judge["judge_error"]})
                    rows.append(o_res)
                except Exception as e:  # noqa: BLE001
                    rows.append({"mode": "optimized", "question": q, "difficulty": item["difficulty"],
                                 "answer": "", "model_used": "error", "cache_hit": False,
                                 "complexity_score": None, "cost_usd": 0, "total_latency_ms": 0,
                                 "accuracy_score": None, "error": str(e)})

                progress.progress((i + 1) / len(subset))

            status.text("Done.")
            st.session_state.eval_rows = rows
            st.session_state.baseline_summary = aggregate(rows, "baseline")
            st.session_state.optimized_summary = aggregate(rows, "optimized")
            st.session_state.comparison_df = build_comparison_table(
                st.session_state.baseline_summary, st.session_state.optimized_summary,
                SMALL_MODEL, LARGE_MODEL,
            )

        if st.session_state.comparison_df is not None:
            st.markdown("### 📈 Baseline vs Optimized")
            st.dataframe(st.session_state.comparison_df, use_container_width=True, hide_index=True)

            b, o = st.session_state.baseline_summary, st.session_state.optimized_summary
            c1, c2, c3 = st.columns(3)

            with c1:
                fig = go.Figure(data=[
                    go.Bar(name="Baseline", x=["Total cost"], y=[b.get("total_cost_usd", 0)], marker_color="#888780"),
                    go.Bar(name="Optimized", x=["Total cost"], y=[o.get("total_cost_usd", 0)], marker_color="#1D9E75"),
                ])
                fig.update_layout(title="Cost (USD)", height=320, margin=dict(t=40))
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                fig = go.Figure(data=[
                    go.Bar(name="Baseline", x=["Avg", "p95"],
                           y=[b.get("avg_latency_ms", 0), b.get("p95_latency_ms", 0)], marker_color="#888780"),
                    go.Bar(name="Optimized", x=["Avg", "p95"],
                           y=[o.get("avg_latency_ms", 0), o.get("p95_latency_ms", 0)], marker_color="#378ADD"),
                ])
                fig.update_layout(title="Latency (ms)", height=320, margin=dict(t=40))
                st.plotly_chart(fig, use_container_width=True)

            with c3:
                fig = go.Figure(data=[
                    go.Bar(name="Baseline", x=["Accuracy"], y=[(b.get("avg_accuracy") or 0) * 100], marker_color="#888780"),
                    go.Bar(name="Optimized", x=["Accuracy"], y=[(o.get("avg_accuracy") or 0) * 100], marker_color="#D4537E"),
                ])
                fig.update_layout(title="Accuracy (%)", height=320, margin=dict(t=40), yaxis_range=[0, 100])
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("### 🔀 Routing breakdown (optimized mode)")
            opt_rows = [r for r in st.session_state.eval_rows if r["mode"] == "optimized"]
            cache_hits = sum(1 for r in opt_rows if r.get("cache_hit"))
            small_hits = sum(1 for r in opt_rows if r.get("model_used") == SMALL_MODEL)
            large_hits = sum(1 for r in opt_rows if r.get("model_used") == LARGE_MODEL)
            fig = go.Figure(data=[go.Pie(
                labels=["Semantic cache hit", "Routed to small model", "Routed to large model"],
                values=[cache_hits, small_hits, large_hits],
                marker_colors=["#5DCAA5", "#85B7EB", "#F0997B"], hole=0.5,
            )])
            fig.update_layout(height=350, margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("📄 Full per-query log"):
                log_df = pd.DataFrame(st.session_state.eval_rows)
                display_cols = [c for c in ["mode", "difficulty", "question", "model_used", "cache_hit",
                                             "complexity_score", "cost_usd", "total_latency_ms",
                                             "accuracy_score", "error"] if c in log_df.columns]
                st.dataframe(log_df[display_cols], use_container_width=True, hide_index=True)
                st.download_button(
                    "⬇️ Download full log (CSV)",
                    log_df.to_csv(index=False).encode("utf-8"),
                    file_name="costwise_rag_eval_log.csv",
                    mime="text/csv",
                )

# ── TAB 3: Architecture ─────────────────────────────────────────────
with tab_about:
    st.markdown("""
### How this system works

CostWise RAG runs every query through one of two pipelines that share the
same document retrieval layer, so the comparison isolates exactly what the
optimization changes.

**Baseline pipeline** — the naive system most RAG tutorials stop at:
retrieve the top-k chunks, always call the large model, no cache, no routing.

**Optimized pipeline:**
1. **Semantic cache** — the incoming query is embedded and compared against
   every previously-answered query in this session. A cosine similarity
   above the threshold returns the cached answer at ~$0 cost.
2. **Complexity router** — on a cache miss, an embedding-centroid classifier
   scores the query against example simple vs. complex phrasings (plus a
   small keyword signal for comparison/causal language) and decides which
   model tier to use — no extra LLM call needed to route.
3. **Model tier** — simple/factual queries go to a small, fast, cheap Groq
   model; complex/multi-hop queries go to the same large model the baseline
   always uses.

Every query — in both pipelines — is logged with cost, latency (broken down
by stage), and an independent LLM-judge accuracy score against a hand-written
reference answer, so the headline metrics above are measured, not estimated.

| Component | Role |
|---|---|
| `all-MiniLM-L6-v2` | Free local embeddings for retrieval, cache lookup, and routing |
| FAISS | Exact cosine-similarity vector search over the document |
| Semantic cache | Skips generation entirely for near-duplicate queries |
| Complexity classifier | Centroid-based router, no extra LLM call |
| """ + f"`{SMALL_MODEL}`" + """ | Handles simple/factual queries |
| """ + f"`{LARGE_MODEL}`" + """ | Handles complex/multi-hop queries + all baseline queries |
| """ + f"`{JUDGE_MODEL}`" + """ | Independent accuracy grader (different model family, reduces self-preference bias) |
    """)
