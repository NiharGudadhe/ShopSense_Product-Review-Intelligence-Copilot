"""Summarise the API request log (data/logs/requests.jsonl): volume, route mix, refusals and latency.

Run from the project folder:   python -m shopsense.monitoring.log_report      (with PYTHONPATH=src, or from src/)
The model runs locally, so cost per request is 0; latency is the number that matters.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
LOG_FILE = ROOT / "data" / "logs" / "requests.jsonl"


def load_log(path=LOG_FILE):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=["time", "question", "asin", "route", "seconds", "refused"])
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)


def summarise(log):
    """One table: per route, request count, refusal share and latency (median and 95th percentile)."""
    if log.empty:
        return pd.DataFrame()
    grouped = log.groupby("route")
    table = pd.DataFrame({
        "requests": grouped.size(),
        "refused_%": (grouped["refused"].mean() * 100).round(1),
        "median_seconds": grouped["seconds"].median().round(1),
        "p95_seconds": grouped["seconds"].quantile(0.95).round(1),
    })
    total = pd.DataFrame({"requests": [len(log)], "refused_%": [round(100 * log["refused"].mean(), 1)],
                          "median_seconds": [round(log["seconds"].median(), 1)],
                          "p95_seconds": [round(log["seconds"].quantile(0.95), 1)]}, index=["ALL"])
    return pd.concat([table, total])


if __name__ == "__main__":
    result = summarise(load_log())
    print(result.to_string() if len(result) else "No requests logged yet.")
