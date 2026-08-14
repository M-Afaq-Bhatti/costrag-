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
    SMALL_MODEL, LARGE_MODEL, JUDGE_MODEL, MODEL_PRICING, EMBEDDING_MODEL_NAME,
    DEFAULT_TOP_K, DEFAULT_CACHE_THRESHOLD, DEFAULT_COMPLEXITY_THRESHOLD,
    DEFAULT_REQUESTS_PER_MINUTE, DEFAULT_DATASET_PATH, DEFAULT_PDF_PATH,
    CASCADE_HIGH_COMPLEXITY_THRESHOLD, GROUNDEDNESS_THRESHOLD,
    FAISS_INDEX_PATH, FAISS_META_PATH, INDEX_INFO_PATH,
)
from src.embeddings import Embedder
from src.ingest import build_chunks_from_pdf
from src.vectorstore import VectorStore
from src.cache import SemanticCache
from src.classifier import ComplexityClassifier
from src.llm_client import LLMClient
from src.pipeline import BaselinePipeline, OptimizedPipeline, CascadePipeline
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
    "cascade_cache": None,
    "classifier": None,
    "llm_client": None,
    "api_key_used": None,
    "eval_rows": [],
    "baseline_summary": {},
    "optimized_summary": {},
    "cascade_summary": {},
    "comparison_df": None,
    "live_log": [],
    "index_info": {},
    "index_load_error": None,
    "_index_load_attempted": False,
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
if st.session_state.cascade_cache is None:
    # Separate cache instance so Routed and Cascade modes don't share hits —
    # keeps the comparison fair, each mode's cache reflects only its own queries.
    st.session_state.cascade_cache = SemanticCache(threshold=DEFAULT_CACHE_THRESHOLD)
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

    # Primary path: load the index built once from the command line via
    # build_index.py. No re-embedding on every Streamlit run.
    if st.session_state.vectorstore is None and not st.session_state.get("_index_load_attempted"):
        st.session_state["_index_load_attempted"] = True
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_META_PATH):
            try:
                vs = VectorStore(dim=embedder.get_sentence_embedding_dimension())
                vs.load(FAISS_INDEX_PATH, FAISS_META_PATH)
                st.session_state.vectorstore = vs
                st.session_state.n_chunks = vs.index.ntotal
                if os.path.exists(INDEX_INFO_PATH):
                    with open(INDEX_INFO_PATH, "r", encoding="utf-8") as f:
                        st.session_state.index_info = json.load(f)
            except Exception as e:  # noqa: BLE001
                st.session_state.index_load_error = str(e)

    info = st.session_state.get("index_info", {})
    if st.session_state.vectorstore is not None:
        st.success(f"✅ Knowledge base loaded — {st.session_state.n_chunks} chunks indexed.")
        if info:
            built_model = info.get("embedding_model")
            st.caption(f"Source: `{info.get('source_pdf', '?')}` · built {info.get('built_at', '?')}")
            if built_model and built_model != EMBEDDING_MODEL_NAME:
                st.warning(
                    f"⚠️ This index was built with `{built_model}`, but the app is "
                    f"currently configured for `{EMBEDDING_MODEL_NAME}`. Rebuild with "
                    f"`python build_index.py` to avoid mismatched embeddings."
                )
    elif st.session_state.get("index_load_error"):
        st.error("Found a saved index but couldn't load it — see details below.")
        st.caption(st.session_state.index_load_error)
    else:
        st.warning("⚠️ No prebuilt knowledge base found.")
        st.code("python build_index.py", language="powershell")
        st.caption(
            "Run that once from the project folder (with your PDF at "
            "`data/source.pdf`, or pass `--pdf <path>`) before starting Streamlit. "
            "The app will then load it automatically on every run — no rebuild."
        )

    with st.expander("🔧 Rebuild from a different PDF (optional, in-app)"):
        st.caption(
            "Only needed if you want to swap documents without using the CLI script. "
            "This also saves to disk, so it persists for future runs."
        )
        pdf_file = st.file_uploader("Upload source PDF", type=["pdf"])
        if st.button("🔨 Rebuild knowledge base now", use_container_width=True):
            if pdf_file is None:
                st.error("Upload a PDF first.")
            else:
                try:
                    with st.spinner("Extracting text, chunking, and embedding..."):
                        chunks = build_chunks_from_pdf(pdf_file, source_name=pdf_file.name)
                        if not chunks:
                            st.error("No extractable text found in this PDF. Is it a scanned image PDF?")
                        else:
                            vs = VectorStore(dim=embedder.get_sentence_embedding_dimension())
                            vs.build(chunks, embedder)
                            vs.save(FAISS_INDEX_PATH, FAISS_META_PATH)
                            new_info = {
                                "source_pdf": pdf_file.name,
                                "n_chunks": len(chunks),
                                "embedding_model": EMBEDDING_MODEL_NAME,
                                "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                            }
                            with open(INDEX_INFO_PATH, "w", encoding="utf-8") as f:
                                json.dump(new_info, f, indent=2)
                            st.session_state.vectorstore = vs
                            st.session_state.n_chunks = len(chunks)
                            st.session_state.index_info = new_info
                            st.session_state.index_load_error = None
                            st.success(f"Indexed {len(chunks)} chunks and saved to disk for future runs.")
                            st.rerun()
                except Exception as e:  # noqa: BLE001
                    st.error(f"Failed to build knowledge base: {e}")

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
        complexity_threshold = st.slider("Routed mode: complexity threshold", -0.3, 0.3, DEFAULT_COMPLEXITY_THRESHOLD, 0.01,
                                          help="Above this, Routed mode sends the query to the large model.")
        st.markdown("**Cascade mode (Mode 3)**")
        cascade_high_threshold = st.slider("Cascade: high-complexity skip threshold", 0.0, 0.5,
                                            CASCADE_HIGH_COMPLEXITY_THRESHOLD, 0.01,
                                            help="Above this, Cascade skips straight to the large model — "
                                                 "no small-model attempt, no verification.")
        groundedness_threshold = st.slider("Cascade: groundedness threshold", 0.0, 0.9,
                                            GROUNDEDNESS_THRESHOLD, 0.01,
                                            help="Minimum embedding similarity between the small model's answer "
                                                 "and its best-matching context chunk to be accepted.")
        rpm = st.slider("Groq requests / minute (pacing)", 5, 30, DEFAULT_REQUESTS_PER_MINUTE,
                         help="Keep below your Groq tier's rate limit to avoid 429 errors. Cascade mode "
                              "fires 2 calls at once, so the sliding-window limiter allows brief bursts "
                              "while keeping the rolling total under this budget.")
        st.session_state["rpm_setting"] = rpm

        st.session_state.cache.threshold = cache_threshold
        st.session_state.cascade_cache.threshold = cache_threshold
        st.session_state.classifier.threshold = complexity_threshold
        if st.session_state.llm_client is not None:
            st.session_state.llm_client.set_rate(rpm)

        col_reset1, col_reset2 = st.columns(2)
        if col_reset1.button("🗑️ Reset Routed cache", use_container_width=True):
            st.session_state.cache.reset()
            st.success("Routed mode cache cleared.")
        if col_reset2.button("🗑️ Reset Cascade cache", use_container_width=True):
            st.session_state.cascade_cache.reset()
            st.success("Cascade mode cache cleared.")

    st.divider()
    with st.expander("🧠 Model configuration"):
        st.markdown(f"""
        | Role | Model | Input $/1M | Output $/1M |
        |---|---|---|---|
        | Small (simple queries) | `{SMALL_MODEL}` | ${MODEL_PRICING[SMALL_MODEL]['input']} | ${MODEL_PRICING[SMALL_MODEL]['output']} |
        | Large (complex + baseline) | `{LARGE_MODEL}` | ${MODEL_PRICING[LARGE_MODEL]['input']} | ${MODEL_PRICING[LARGE_MODEL]['output']} |
        | Judge (accuracy scoring) | `{JUDGE_MODEL}` | ${MODEL_PRICING[JUDGE_MODEL]['input']} | ${MODEL_PRICING[JUDGE_MODEL]['output']} |

        Embeddings: `{EMBEDDING_MODEL_NAME}` — free, runs locally, no API cost.
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
            mode = st.radio("Pipeline mode", ["Cascade", "Optimized (Routed)", "Baseline"], index=0)
        with col_q:
            question = st.text_input("Ask a question about the document", placeholder="e.g. What is the definition of unstable angina?")

        ask = st.button("Ask", type="primary")

        if ask and question.strip():
            baseline_pipe = BaselinePipeline(st.session_state.vectorstore, embedder,
                                              st.session_state.llm_client, top_k=top_k)
            optimized_pipe = OptimizedPipeline(st.session_state.vectorstore, embedder,
                                                st.session_state.llm_client, st.session_state.cache,
                                                st.session_state.classifier, top_k=top_k)
            cascade_pipe = CascadePipeline(st.session_state.vectorstore, embedder,
                                            st.session_state.llm_client, st.session_state.cascade_cache,
                                            st.session_state.classifier, top_k=top_k,
                                            high_complexity_threshold=cascade_high_threshold,
                                            groundedness_threshold=groundedness_threshold)
            pipe_map = {"Cascade": cascade_pipe, "Optimized (Routed)": optimized_pipe, "Baseline": baseline_pipe}

            try:
                with st.spinner("Running pipeline..."):
                    result = pipe_map[mode].answer(question)
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
                elif result.get("escalated"):
                    badge = '<span class="badge badge-large">Escalated to large model</span>'
                elif result.get("cascade_path"):
                    badge = '<span class="badge badge-small">Small model (verified grounded)</span>'
                elif result["model_used"] == LARGE_MODEL:
                    badge = '<span class="badge badge-large">Large model</span>'
                elif result["model_used"] == SMALL_MODEL:
                    badge = '<span class="badge badge-small">Small model</span>'
                else:
                    badge = '<span class="badge badge-baseline">Baseline (large only)</span>'
                m3.markdown(f"**Routing**<br>{badge}", unsafe_allow_html=True)
                m4.metric("Complexity score", f"{result['complexity_score']:.3f}" if result["complexity_score"] is not None else "—")

                if result.get("cascade_path") and result.get("groundedness_score") is not None:
                    gc1, gc2 = st.columns(2)
                    gc1.metric("Groundedness score", f"{result['groundedness_score']:.3f}")
                    gc2.metric("Large-model call cost", f"${result.get('large_call_wasted_cost', 0) or result['cost_usd']:.5f}",
                               help="Cost of the speculative large-model call — wasted if the small model's "
                                    "answer was accepted, or the answer itself if escalation occurred.")

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
        n_eval = st.slider("Number of queries to evaluate (runs Baseline + Routed + Cascade on each)",
                            2, n_available, min(20, n_available), 1)
        est_calls = n_eval * 7  # rough upper bound: baseline(2) + routed(2) + cascade(up to 3)
        st.caption(
            f"This will make up to ~{est_calls} Groq API calls (answer + judge, all three modes; "
            f"Cascade may fire two generation calls per query) at your configured pacing — "
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
            cascade_pipe = CascadePipeline(st.session_state.vectorstore, embedder,
                                            st.session_state.llm_client, st.session_state.cascade_cache,
                                            st.session_state.classifier, top_k=top_k,
                                            high_complexity_threshold=cascade_high_threshold,
                                            groundedness_threshold=groundedness_threshold)

            pipelines = [
                ("baseline", baseline_pipe, LARGE_MODEL),
                ("optimized", optimized_pipe, "error"),
                ("cascade", cascade_pipe, "error"),
            ]

            progress = st.progress(0.0)
            status = st.empty()
            rows = []

            for i, item in enumerate(subset):
                q, ref = item["question"], item["reference_answer"]

                for mode_name, pipe, fallback_model in pipelines:
                    status.text(f"[{i+1}/{len(subset)}] {mode_name}: {q[:60]}")
                    try:
                        res = pipe.answer(q)
                        judge = judge_answer(st.session_state.llm_client, q, ref, res["answer"])
                        res.update({"difficulty": item["difficulty"], "accuracy_score": judge["score"],
                                    "judge_error": judge["judge_error"]})
                        rows.append(res)
                    except Exception as e:  # noqa: BLE001
                        rows.append({"mode": mode_name, "question": q, "difficulty": item["difficulty"],
                                     "answer": "", "model_used": fallback_model, "cache_hit": False,
                                     "complexity_score": None, "cascade_path": False, "escalated": False,
                                     "groundedness_score": None, "large_call_wasted_cost": 0.0,
                                     "cost_usd": 0, "total_latency_ms": 0,
                                     "accuracy_score": None, "error": str(e)})

                progress.progress((i + 1) / len(subset))

            status.text("Done.")
            st.session_state.eval_rows = rows
            st.session_state.baseline_summary = aggregate(rows, "baseline")
            st.session_state.optimized_summary = aggregate(rows, "optimized")
            st.session_state.cascade_summary = aggregate(rows, "cascade")
            st.session_state.comparison_df = build_comparison_table(
                st.session_state.baseline_summary, st.session_state.optimized_summary,
                st.session_state.cascade_summary,
            )

        if st.session_state.comparison_df is not None:
            st.markdown("### 📈 Baseline vs Routed vs Cascade")
            st.dataframe(st.session_state.comparison_df, use_container_width=True, hide_index=True)

            b = st.session_state.baseline_summary
            o = st.session_state.optimized_summary
            cs = st.session_state.cascade_summary
            c1, c2, c3 = st.columns(3)

            colors = {"Baseline": "#888780", "Routed": "#378ADD", "Cascade": "#1D9E75"}

            with c1:
                fig = go.Figure(data=[
                    go.Bar(name=k, x=["Total cost"], y=[v], marker_color=colors[k])
                    for k, v in [("Baseline", b.get("total_cost_usd", 0)),
                                 ("Routed", o.get("total_cost_usd", 0)),
                                 ("Cascade", cs.get("total_cost_usd", 0))]
                ])
                fig.update_layout(title="Cost (USD)", height=320, margin=dict(t=40))
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                fig = go.Figure(data=[
                    go.Bar(name="Baseline", x=["Avg", "p95"],
                           y=[b.get("avg_latency_ms", 0), b.get("p95_latency_ms", 0)], marker_color=colors["Baseline"]),
                    go.Bar(name="Routed", x=["Avg", "p95"],
                           y=[o.get("avg_latency_ms", 0), o.get("p95_latency_ms", 0)], marker_color=colors["Routed"]),
                    go.Bar(name="Cascade", x=["Avg", "p95"],
                           y=[cs.get("avg_latency_ms", 0), cs.get("p95_latency_ms", 0)], marker_color=colors["Cascade"]),
                ])
                fig.update_layout(title="Latency (ms)", height=320, margin=dict(t=40))
                st.plotly_chart(fig, use_container_width=True)

            with c3:
                fig = go.Figure(data=[
                    go.Bar(name="Baseline", x=["Accuracy"], y=[(b.get("avg_accuracy") or 0) * 100], marker_color=colors["Baseline"]),
                    go.Bar(name="Routed", x=["Accuracy"], y=[(o.get("avg_accuracy") or 0) * 100], marker_color=colors["Routed"]),
                    go.Bar(name="Cascade", x=["Accuracy"], y=[(cs.get("avg_accuracy") or 0) * 100], marker_color=colors["Cascade"]),
                ])
                fig.update_layout(title="Accuracy (%)", height=320, margin=dict(t=40), yaxis_range=[0, 100])
                st.plotly_chart(fig, use_container_width=True)

            col_route1, col_route2 = st.columns(2)

            with col_route1:
                st.markdown("#### 🔀 Routing breakdown — Routed mode")
                opt_rows = [r for r in st.session_state.eval_rows if r["mode"] == "optimized"]
                cache_hits = sum(1 for r in opt_rows if r.get("cache_hit"))
                small_hits = sum(1 for r in opt_rows if r.get("model_used") == SMALL_MODEL)
                large_hits = sum(1 for r in opt_rows if r.get("model_used") == LARGE_MODEL)
                fig = go.Figure(data=[go.Pie(
                    labels=["Cache hit", "Small model", "Large model"],
                    values=[cache_hits, small_hits, large_hits],
                    marker_colors=["#5DCAA5", "#85B7EB", "#F0997B"], hole=0.5,
                )])
                fig.update_layout(height=350, margin=dict(t=10))
                st.plotly_chart(fig, use_container_width=True)

            with col_route2:
                st.markdown("#### 🔀 Routing breakdown — Cascade mode")
                casc_rows = [r for r in st.session_state.eval_rows if r["mode"] == "cascade"]
                cache_hits_c = sum(1 for r in casc_rows if r.get("cache_hit"))
                shortcut_large = sum(1 for r in casc_rows if not r.get("cache_hit") and not r.get("cascade_path"))
                accepted_small = sum(1 for r in casc_rows if r.get("cascade_path") and not r.get("escalated"))
                escalated = sum(1 for r in casc_rows if r.get("cascade_path") and r.get("escalated"))
                fig = go.Figure(data=[go.Pie(
                    labels=["Cache hit", "Pre-filter shortcut (large)", "Small accepted (grounded)", "Escalated to large"],
                    values=[cache_hits_c, shortcut_large, accepted_small, escalated],
                    marker_colors=["#5DCAA5", "#F0997B", "#85B7EB", "#D4537E"], hole=0.5,
                )])
                fig.update_layout(height=350, margin=dict(t=10))
                st.plotly_chart(fig, use_container_width=True)

            st.caption(
                f"Cascade escalation rate: {(cs.get('pct_escalated') or 0) * 100:.1f}% of non-shortcut, "
                f"non-cached queries needed the large model after the small model's answer failed the "
                f"groundedness check. Avg wasted large-call cost per query: "
                f"${cs.get('avg_wasted_large_cost', 0):.5f}."
            )

            with st.expander("📄 Full per-query log"):
                log_df = pd.DataFrame(st.session_state.eval_rows)
                display_cols = [c for c in ["mode", "difficulty", "question", "model_used", "cache_hit",
                                             "complexity_score", "cascade_path", "escalated",
                                             "groundedness_score", "cost_usd", "large_call_wasted_cost",
                                             "total_latency_ms", "accuracy_score", "error"]
                                 if c in log_df.columns]
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

CostWise RAG runs every query through one of three pipelines that share the
same document retrieval layer, so the comparison isolates exactly what each
optimization changes.

**Baseline** — the naive system most RAG tutorials stop at: retrieve the
top-k chunks, always call the large model, no cache, no routing.

**Routed (Mode 2)**
1. **Semantic cache** — the incoming query is embedded and compared against
   every previously-answered query in this session. A cosine similarity
   above the threshold returns the cached answer at ~$0 cost.
2. **Complexity router** — on a cache miss, an embedding-centroid classifier
   scores the query and decides small vs. large model *before* generating.
3. If the classifier guesses wrong, there's no recovery — the answer is
   whatever that model tier produced.

**Cascade (Mode 3)** — built to fix Routed mode's main weakness: a
pre-classification the system never checks against the actual answer.
1. Same semantic cache check first.
2. Obviously-complex queries (classifier score above a stricter threshold)
   skip straight to the large model — no wasted small-model attempt.
3. Everything else fires the **small and large model in parallel**, in two
   threads, using a thread-safe sliding-window rate limiter so the burst
   doesn't get serialized or trip the free-tier limit.
4. The small model's answer is checked with a **free groundedness
   verifier** — cosine similarity between the answer and its best-matching
   retrieved chunk, plus a hedge-phrase check — using the same local
   embedding model, no extra LLM call.
5. Grounded → accept the small model's answer immediately, without waiting
   for the large model's (already in-flight) response. Not grounded → fall
   back to the large model's answer, which is likely already finished.

**Why parallel, not sequential:** if Cascade tried the small model, waited,
then only *then* called the large model on failure, the worst-case latency
for a misclassified query would be small-time + large-time stacked — worse
than the baseline it's meant to beat. Firing both at once bounds the worst
case at roughly `max(small_time, large_time)` instead. The trade-off: the
large model call is billed once fired, even when its answer goes unused —
tracked explicitly as "wasted large-call cost" so that cost is visible, not
hidden.

Every query, in all three pipelines, is logged with cost, latency (broken
down by stage), and an independent LLM-judge accuracy score against a
hand-written reference answer, so every metric above is measured, not
estimated.

| Component | Role |
|---|---|
| """ + f"`{EMBEDDING_MODEL_NAME}`" + """ | Free local embeddings for retrieval, cache lookup, routing, and groundedness verification |
| FAISS | Exact cosine-similarity vector search over the document |
| Semantic cache | Skips generation entirely for near-duplicate queries |
| Complexity classifier | Centroid-based router, no extra LLM call |
| Groundedness verifier | Free post-hoc answer check for Cascade mode, no extra LLM call |
| """ + f"`{SMALL_MODEL}`" + """ | Handles simple/factual queries |
| """ + f"`{LARGE_MODEL}`" + """ | Handles complex/multi-hop queries, all Baseline queries, and Cascade escalations |
| """ + f"`{JUDGE_MODEL}`" + """ | Independent accuracy grader (different model family, reduces self-preference bias) |
    """)
