"""
Aggregates per-query logs (produced by the pipelines + judge) into the
headline Baseline / Routed / Cascade comparison table shown on the
dashboard. `cascade_summary` is optional so the table still works with
just Baseline + Routed if the cascade mode hasn't been run.
"""

from typing import Optional
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
        result["pct_routed_small"] = (
            float((non_cache["model_used"] == SMALL_MODEL).mean()) if len(non_cache) > 0 else None
        )

    elif mode == "cascade":
        result["cache_hit_rate"] = float(df["cache_hit"].mean())
        non_cache = df[~df["cache_hit"]]
        if len(non_cache) > 0:
            result["pct_accepted_small"] = float((non_cache["model_used"] == SMALL_MODEL).mean())
            result["avg_wasted_large_cost"] = float(non_cache["large_call_wasted_cost"].mean())
            cascade_rows = non_cache[non_cache["cascade_path"] == True]  # noqa: E712
            result["pct_escalated"] = (
                float(cascade_rows["escalated"].mean()) if len(cascade_rows) > 0 else None
            )
        else:
            result["pct_accepted_small"] = None
            result["avg_wasted_large_cost"] = 0.0
            result["pct_escalated"] = None

    return result


def _pct_change(base, new):
    if not base:
        return "—"
    change = (new - base) / base * 100
    sign = "-" if change < 0 else "+"
    return f"{sign}{abs(change):.1f}%"


def _fmt_pct(summary: dict, key: str) -> str:
    val = summary.get(key)
    return f"{val * 100:.1f}%" if val is not None else "—"


def build_comparison_table(baseline_summary: dict, routed_summary: dict,
                            cascade_summary: Optional[dict] = None) -> pd.DataFrame:
    """Builds the Metric / Baseline / Routed [/ Cascade] [/ Δ columns] table.
    Pass cascade_summary=None to get the original 2-way table."""
    b, r, c = baseline_summary, routed_summary, (cascade_summary or {})
    has_cascade = cascade_summary is not None

    def row(metric, b_val, r_val, c_val=None, r_delta="—", c_delta="—"):
        d = {"Metric": metric, "Baseline": b_val, "Routed": r_val, "Routed Δ": r_delta}
        if has_cascade:
            d["Cascade"] = c_val if c_val is not None else "—"
            d["Cascade Δ"] = c_delta
        return d

    n = r.get("n_queries", 0) or c.get("n_queries", 0)

    rows = [
        row(
            f"Total cost ({n} queries)",
            f"${b.get('total_cost_usd', 0):.4f}",
            f"${r.get('total_cost_usd', 0):.4f}",
            f"${c.get('total_cost_usd', 0):.4f}" if has_cascade else None,
            _pct_change(b.get("total_cost_usd", 0), r.get("total_cost_usd", 0)),
            _pct_change(b.get("total_cost_usd", 0), c.get("total_cost_usd", 0)) if has_cascade else "—",
        ),
        row(
            "Avg latency",
            f"{b.get('avg_latency_ms', 0):.0f} ms",
            f"{r.get('avg_latency_ms', 0):.0f} ms",
            f"{c.get('avg_latency_ms', 0):.0f} ms" if has_cascade else None,
            _pct_change(b.get("avg_latency_ms", 0), r.get("avg_latency_ms", 0)),
            _pct_change(b.get("avg_latency_ms", 0), c.get("avg_latency_ms", 0)) if has_cascade else "—",
        ),
        row(
            "p95 latency",
            f"{b.get('p95_latency_ms', 0):.0f} ms",
            f"{r.get('p95_latency_ms', 0):.0f} ms",
            f"{c.get('p95_latency_ms', 0):.0f} ms" if has_cascade else None,
            _pct_change(b.get("p95_latency_ms", 0), r.get("p95_latency_ms", 0)),
            _pct_change(b.get("p95_latency_ms", 0), c.get("p95_latency_ms", 0)) if has_cascade else "—",
        ),
        row(
            "Accuracy (judge score)",
            f"{(b.get('avg_accuracy') or 0) * 100:.1f}%",
            f"{(r.get('avg_accuracy') or 0) * 100:.1f}%",
            f"{(c.get('avg_accuracy') or 0) * 100:.1f}%" if has_cascade else None,
            _pct_change(b.get("avg_accuracy", 0) or 0, r.get("avg_accuracy", 0) or 0),
            _pct_change(b.get("avg_accuracy", 0) or 0, c.get("avg_accuracy", 0) or 0) if has_cascade else "—",
        ),
        row("Cache hit rate", "—", _fmt_pct(r, "cache_hit_rate"),
            _fmt_pct(c, "cache_hit_rate") if has_cascade else None),
        row("% served by small model", "—", _fmt_pct(r, "pct_routed_small"),
            _fmt_pct(c, "pct_accepted_small") if has_cascade else None),
    ]

    if has_cascade:
        rows.append(row("Escalation rate (cascade only)", "—", "—", _fmt_pct(c, "pct_escalated")))
        rows.append(row("Avg wasted large-call cost (cascade only)", "—", "—",
                         f"${c.get('avg_wasted_large_cost', 0):.5f}"))

    columns = ["Metric", "Baseline", "Routed", "Routed Δ"]
    if has_cascade:
        columns += ["Cascade", "Cascade Δ"]

    return pd.DataFrame(rows)[columns]
