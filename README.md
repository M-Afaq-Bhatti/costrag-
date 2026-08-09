# CostWise RAG

A dual-mode Retrieval-Augmented Generation system that benchmarks a **cache +
complexity-routing optimized pipeline** against a **naive single-model
baseline**, on an identical golden query set, with every query logged for
cost, latency, and LLM-judged accuracy. Built for a cardiology reference
document, but the pipeline is domain-agnostic.

**Live metrics, not claims.** The dashboard doesn't assert an optimization
worked — it runs both pipelines on the same queries and shows the numbers
side by side.

## How it works

| Mode | Behavior |
|---|---|
| **Baseline** | retrieve → always call the large model. No cache, no routing. |
| **Optimized** | check semantic cache → route by query complexity → small model for simple queries, large model for complex ones. |

Both modes share the same retrieval layer (FAISS + free local embeddings),
so the comparison isolates exactly what the optimization changes.

## Project structure

```
costwise-rag/
├── app.py                  # Streamlit dashboard (entry point)
├── config.py                # models, pricing, thresholds — edit here
├── requirements.txt
├── data/
│   ├── source.pdf                     # ← put your cardiology PDF here (optional; can also upload via UI)
│   ├── golden_dataset.json            # ← put your real 50-item dataset here (optional; can also upload via UI)
│   └── golden_dataset_template.json   # schema reference only
├── src/
│   ├── embeddings.py         # free local embeddings (fastembed, no PyTorch)
│   ├── ingest.py              # PDF → chunks
│   ├── vectorstore.py        # FAISS wrapper
│   ├── cache.py                # semantic cache
│   ├── classifier.py          # complexity router
│   ├── llm_client.py          # Groq wrapper + cost tracking + rate-limit pacing
│   ├── judge.py                # LLM-as-judge accuracy scoring
│   ├── pipeline.py            # BaselinePipeline / OptimizedPipeline
│   └── metrics.py             # aggregation + comparison table
└── .streamlit/config.toml    # theme
```

## 1. Get a free Groq API key

Sign up at [console.groq.com](https://console.groq.com) → API Keys → Create
key. No credit card required for the free tier.

## 2. Add your data

Either:
- Drop your PDF at `data/source.pdf` and your dataset at
  `data/golden_dataset.json`, **or**
- Just run the app and upload both files through the sidebar — nothing needs
  to be committed to the repo.

Your golden dataset JSON should look like `data/golden_dataset_template.json`:
a list of objects with `id`, `difficulty` (`"simple"` or `"complex"`),
`question`, and `reference_answer`.

## 3. Run locally

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`), paste your
Groq API key into the sidebar, click **Build knowledge base**, then either
try the **Live query** tab or run the **Full evaluation** tab.

## 4. Deploy to Streamlit Community Cloud

1. Push this folder to a public (or private) GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → point
   it at the repo, branch, and `app.py`.
3. Deploy. First load will take a minute or two while dependencies install
   and the embedding model downloads (~67MB, cached after that).
4. Anyone who opens the app link pastes their own Groq key in the sidebar —
   no server-side secret needed unless you want to hardcode one for a demo
   (Settings → Secrets → `GROQ_API_KEY = "..."`, then it auto-fills).

No AWS, no Docker, no server to manage — this is intentionally a pure
Streamlit deployment.

## Results

Benchmarked on a 50-query cardiology evaluation set (25 factual, 25 multi-hop)
run through both pipelines via the **Full evaluation** tab.

| Metric | Baseline | Optimized | Change |
|---|---|---|---|
| Total cost (50 queries) | $0.03 | $0.02 | -42.3% |
| Avg latency | 4,081 ms | 3,738 ms | -8.4% |
| p95 latency | 12,561 ms | 12,573 ms | +0.1% |
| Accuracy (LLM-judge score) | 42.0% | 47.0% | +11.9% (relative) |
| Cache hit rate | — | 20.0% | — |
| % routed to small model | — | 25.0% | — |

**Reading these honestly:**
- Cost and average latency dropped with *no* accuracy trade-off — routing
  45% of traffic away from the large model (20% served from cache, 25% sent
  to the small model) didn't cost correctness on this dataset.
- p95 latency barely moved. Caching and routing help the typical query, but
  the slowest queries are still bound by large-model generation time on
  genuinely hard multi-hop questions — tail latency needs a different lever
  (e.g. streaming, smaller context windows, or a faster large-model tier),
  not just routing.
- Accuracy at 42-47% is a modest absolute number worth investigating further
  rather than a metric to publish as a finished result — likely levers are
  judge rubric strictness, retrieval chunk size cutting off multi-hop
  context, and reference-answer granularity vs. what's actually retrievable
  from the source document.

Re-run **Full evaluation** any time you change `top_k`, chunk size, cache
threshold, or the routing threshold in the sidebar — the table above
recomputes from scratch each run, so it always reflects your current config,
not a cached claim.

## Notes on the free tier

- Groq's free tier is rate-limited (~30 requests/min). The sidebar's
  **Groq requests / minute** slider paces calls to stay under whatever limit
  you're on — lower it if you see 429 errors.
- A full evaluation run makes up to `4 × n_queries` calls (baseline answer +
  judge, optimized answer + judge). At 25 req/min, 20 queries ≈ 3-4 minutes.
- The embedding model and vector search are free and local — they don't
  count against any API quota.

## What to put in your portfolio writeup

The **Full evaluation** tab produces the comparison table and charts
directly — screenshot it, or use the CSV download for your own writeup.
Headline the total cost reduction, p95 latency improvement, and how close
optimized-mode accuracy stayed to baseline.