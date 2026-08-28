"""
READ-ONLY forward-sample evaluation helper for the SOL Shock Continuation
V3.1 forward shadow. Never imported by the live runner (run_shadow.py) --
this module exists purely for evaluation/reporting, opens events.csv/
trades.csv for reading only, and never writes to them or to state.json.

The forward-epoch boundary is pinned in FORWARD_EPOCH_BOUNDARY.json
(committed, alongside this file) -- NOT derived from state.json's
epoch_start_iso, which is process-construction wall-clock time, not a
market-bar boundary, and must never be used here. This module does not
read state.json at all.
"""
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from research.outlier_and_clustering import cluster_events  # noqa: E402

BOUNDARY_PATH = os.path.join(HERE, "FORWARD_EPOCH_BOUNDARY.json")


def load_boundary(path=BOUNDARY_PATH):
    with open(path) as f:
        return json.load(f)


def forward_epoch_start_ts_ms(path=BOUNDARY_PATH):
    return load_boundary(path)["forward_epoch_start_ts_ms"]


def is_forward_event(row, boundary_ts_ms):
    """market_ts_ms >= boundary AND phase != 'warmup'. catchup bars at/after
    the boundary DO count -- they are genuine post-epoch-start market events
    that were only processed late after a restart, not pre-epoch history."""
    return row["market_ts_ms"] >= boundary_ts_ms and row["phase"] != "warmup"


def is_forward_trade(row, boundary_ts_ms):
    """signal_ts_ms >= boundary. Deliberately NEVER determined from a
    trade's exit phase -- a legitimate trade opened during phase=live may
    exit during phase=catchup after a restart, and must still count."""
    return row["signal_ts_ms"] >= boundary_ts_ms


def load_events(runtime_dir):
    path = os.path.join(runtime_dir, "events.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["market_ts_ms", "phase", "eligible"])
    return pd.read_csv(path)


def load_trades(runtime_dir):
    path = os.path.join(runtime_dir, "trades.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["signal_ts_ms"])
    return pd.read_csv(path)


def forward_events(events_df, boundary_ts_ms):
    if len(events_df) == 0:
        return events_df
    mask = (events_df["market_ts_ms"] >= boundary_ts_ms) & (events_df["phase"] != "warmup")
    return events_df[mask]


def forward_trades(trades_df, boundary_ts_ms):
    if len(trades_df) == 0:
        return trades_df
    return trades_df[trades_df["signal_ts_ms"] >= boundary_ts_ms]


def cluster_forward_events(events_df, boundary_ts_ms, window_hours=24):
    """Clusters ONLY events that already passed is_forward_event() --
    filtering happens BEFORE clustering, never after, so a warmup event can
    never join, seed, or otherwise influence a forward cluster."""
    fwd = forward_events(events_df, boundary_ts_ms)
    if len(fwd) == 0:
        return fwd
    fwd_eligible = fwd[fwd["eligible"] == True]  # noqa: E712 (pandas bool comparison)
    if len(fwd_eligible) == 0:
        return fwd_eligible
    adapted = fwd_eligible.copy()
    adapted["symbol"] = "SOL/USDT"
    adapted["entry_ts_ms"] = adapted["market_ts_ms"]
    return cluster_events(adapted, window_hours=window_hours)


def compute_counts(runtime_dir, boundary_ts_ms=None):
    """Read-only: opens events.csv/trades.csv, writes nothing, mutates
    nothing on disk."""
    if boundary_ts_ms is None:
        boundary_ts_ms = forward_epoch_start_ts_ms()
    events = load_events(runtime_dir)
    trades = load_trades(runtime_dir)

    warmup_eligible = 0
    forward_eligible = 0
    n_clusters = 0
    if len(events):
        warmup_eligible = int(((events["phase"] == "warmup") & (events["eligible"] == True)).sum())  # noqa: E712
        fwd = forward_events(events, boundary_ts_ms)
        forward_eligible = int((fwd["eligible"] == True).sum()) if len(fwd) else 0  # noqa: E712
        clustered = cluster_forward_events(events, boundary_ts_ms)
        n_clusters = int(clustered["cluster_id"].nunique()) if len(clustered) else 0

    n_forward_trades = int(len(forward_trades(trades, boundary_ts_ms))) if len(trades) else 0

    return {
        "boundary_ts_ms": boundary_ts_ms,
        "warmup_eligible_events": warmup_eligible,
        "forward_eligible_events": forward_eligible,
        "independent_forward_clusters": n_clusters,
        "forward_trades": n_forward_trades,
    }


if __name__ == "__main__":
    counts = compute_counts(HERE)
    for k, v in counts.items():
        print(f"{k}: {v}")
