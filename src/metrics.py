"""
Aggregates per-query logs (produced by the pipelines + judge) into the
headline baseline-vs-optimized comparison table shown on the dashboard.
"""

import pandas as pd
from src.utils import percentile
from config import SMALL_MODEL


def aggregate(rows: list, mode: str) -> dict:
    df = pd.DataFrame([r for r in rows if r["mode"] == mode])
    if df.empty:
        return {}

    latencies = df["total_latency_ms"].tolist()
    accuracies = df["accuracy_score"].dropna().tolist() if "accuracy_score" in df else []

    result = {
        "mode": mode,
        "n_queries": len(df),
        "total_cost_usd": float(df["cost_usd"].sum()),
        "avg_cost_usd": float(df["cost_usd"].mean()),
        "avg_latency_ms": float(df["total_latency_ms"].mean()),
        "p95_latency_ms": percentile(latencies, 95),
        "median_latency_ms": percentile(latencies, 50),
        "avg_accuracy": float(sum(accuracies) / len(accuracies)) if accuracies else None,
    }

    if mode == "optimized":
        result["cache_hit_rate"] = float(df["cache_hit"].mean())
        non_cache = df[~df["cache_hit"]]
        if len(non_cache) > 0:
            result["pct_routed_small"] = float((non_cache["model_used"] == SMALL_MODEL).mean())
        else:
            result["pct_routed_small"] = None

    return result


def build_comparison_table(baseline_summary: dict, optimized_summary: dict,
                            small_model_name: str, large_model_name: str) -> pd.DataFrame:
    def pct_change(base, opt, lower_is_better=True):
        if not base:
            return "—"
        change = (opt - base) / base * 100
        sign = "-" if change < 0 else "+"
        return f"{sign}{abs(change):.1f}%"

    b = baseline_summary
    o = optimized_summary

    def routed_small_pct(summary):
        val = summary.get("pct_routed_small")
        if val is None:
            return "—"
        return f"{val * 100:.1f}%"

    rows = [
        {
            "Metric": f"Total cost ({o.get('n_queries', 0)} queries)",
            "Baseline": f"${b.get('total_cost_usd', 0):.4f}",
            "Optimized": f"${o.get('total_cost_usd', 0):.4f}",
            "Change": pct_change(b.get("total_cost_usd", 0), o.get("total_cost_usd", 0)),
        },
        {
            "Metric": "Avg latency",
            "Baseline": f"{b.get('avg_latency_ms', 0):.0f} ms",
            "Optimized": f"{o.get('avg_latency_ms', 0):.0f} ms",
            "Change": pct_change(b.get("avg_latency_ms", 0), o.get("avg_latency_ms", 0)),
        },
        {
            "Metric": "p95 latency",
            "Baseline": f"{b.get('p95_latency_ms', 0):.0f} ms",
            "Optimized": f"{o.get('p95_latency_ms', 0):.0f} ms",
            "Change": pct_change(b.get("p95_latency_ms", 0), o.get("p95_latency_ms", 0)),
        },
        {
            "Metric": "Accuracy (judge score)",
            "Baseline": f"{(b.get('avg_accuracy') or 0) * 100:.1f}%",
            "Optimized": f"{(o.get('avg_accuracy') or 0) * 100:.1f}%",
            "Change": pct_change(b.get("avg_accuracy", 0) or 0, o.get("avg_accuracy", 0) or 0),
        },
        {
            "Metric": "Cache hit rate",
            "Baseline": "—",
            "Optimized": f"{o.get('cache_hit_rate', 0) * 100:.1f}%",
            "Change": "—",
        },
        {
            "Metric": "% routed to small model",
            "Baseline": "—",
            "Optimized": routed_small_pct(o),
            "Change": "—",
        },
    ]
    return pd.DataFrame(rows)
