"""
V3.1 sections C & D: outlier dependence and shock clustering.

Clustering rule (pre-declared, not selected for effect on results): two
events on the SAME symbol are part of the same independent shock cluster if
their entry timestamps are within CLUSTER_WINDOW_HOURS of each other.
Chosen to span a plausible single liquidation-cascade episode (crypto
cascades typically resolve within a day or so) without being so wide it
merges genuinely separate market events.
"""
import numpy as np
import pandas as pd

CLUSTER_WINDOW_HOURS = 24


def outlier_dependence(ledger, label):
    net = ledger["net_pnl"].dropna()
    total = net.sum()
    n = len(net)
    sorted_desc = net.sort_values(ascending=False)

    def stats_excluding(k=None, pct=None):
        if k is not None:
            excl = sorted_desc.head(k)
            remaining = net.drop(excl.index)
        else:
            k_n = max(1, int(round(n * pct)))
            excl = sorted_desc.head(k_n)
            remaining = net.drop(excl.index)
        ret = remaining.sum()
        wins = remaining[remaining > 0]
        losses = remaining[remaining <= 0]
        pf = (wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else None
        expectancy = remaining.mean() if len(remaining) else None
        sharpe = (remaining.mean() / remaining.std() * np.sqrt(len(remaining))) if remaining.std() else None
        return {"n_remaining": len(remaining), "total_pnl": float(ret), "profit_factor": pf,
                "expectancy": float(expectancy) if expectancy is not None else None,
                "sharpe_like": float(sharpe) if sharpe is not None else None,
                "pct_of_total_pnl_removed": float(excl.sum() / total * 100) if total else None}

    out = {"label": label, "n_trades": n, "total_pnl": float(total)}
    for k in (1, 3, 5, 10):
        if n >= k:
            out[f"excl_top_{k}"] = stats_excluding(k=k)
    for pct, name in ((0.01, "top1pct"), (0.05, "top5pct")):
        out[f"excl_{name}"] = stats_excluding(pct=pct)
    return out


def cluster_events(ledger, window_hours=CLUSTER_WINDOW_HOURS):
    """Assigns a cluster_id per symbol: consecutive events (sorted by entry
    time) within window_hours of the PREVIOUS event in the same symbol join
    the same cluster (chain rule -- a rolling cascade counts as one cluster
    even if individual gaps are each < window but the whole episode is
    longer)."""
    ledger = ledger.sort_values(["symbol", "entry_ts_ms"]).reset_index(drop=True)
    cluster_ids = []
    cluster_counter = 0
    last_ts_by_symbol = {}
    last_cluster_by_symbol = {}
    for _, row in ledger.iterrows():
        sym, ts = row["symbol"], row["entry_ts_ms"]
        last_ts = last_ts_by_symbol.get(sym)
        if last_ts is not None and (ts - last_ts) <= window_hours * 3_600_000:
            cid = last_cluster_by_symbol[sym]
        else:
            cluster_counter += 1
            cid = cluster_counter
        cluster_ids.append(cid)
        last_ts_by_symbol[sym] = ts
        last_cluster_by_symbol[sym] = cid
    ledger = ledger.copy()
    ledger["cluster_id"] = cluster_ids
    return ledger


def cluster_level_stats(ledger_with_clusters, label):
    g = ledger_with_clusters.groupby("cluster_id").agg(
        symbol=("symbol", "first"), n_trades=("net_pnl", "count"),
        net_pnl=("net_pnl", "sum"), start=("entry_ts_ms", "min"), end=("entry_ts_ms", "max"))
    n_clusters = len(g)
    n_trades = len(ledger_with_clusters)
    net = g["net_pnl"]
    wins = net[net > 0]
    losses = net[net <= 0]
    pf = (wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else None
    return {
        "label": label, "n_raw_trades": n_trades, "n_independent_clusters": n_clusters,
        "trades_per_cluster_mean": float(n_trades / n_clusters) if n_clusters else None,
        "cluster_expectancy": float(net.mean()) if n_clusters else None,
        "cluster_profit_factor": pf,
        "fraction_clusters_profitable": float((net > 0).mean()) if n_clusters else None,
        "top5_clusters_pct_of_total_pnl": float(
            net.sort_values(ascending=False).head(5).sum() / net.sum() * 100) if net.sum() else None,
    }
