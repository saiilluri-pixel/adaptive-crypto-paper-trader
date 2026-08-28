"""
READ-ONLY analysis of ws_capture.py's output. Computes data-delivery
latency percentiles (from streams that carry Binance's own event
timestamp) and an execution-realism comparison: the frozen strategy's
modeled fill (PaperEngine._fill()'s adverse-slippage formula) vs the real
best bid/ask observed at the same time, for both a hypothetical long entry
(compare against ask) and short entry (compare against bid).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
import config  # noqa: E402

IN_PATH = os.path.join(HERE, "ws_capture_output.jsonl")


def load(path=IN_PATH):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def latency_stats(rows):
    lat = []
    for r in rows:
        if r.get("exchange_event_ts") is not None:
            lat.append(r["local_receive_ts"] - r["exchange_event_ts"])
    if not lat:
        return None
    arr = np.array(lat)
    return {
        "n": len(arr), "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)), "p99_ms": float(np.percentile(arr, 99)),
        "max_ms": float(arr.max()), "min_ms": float(arr.min()),
    }


def execution_realism(rows):
    book = [r for r in rows if r["type"] == "bookTicker"]
    if not book:
        return None
    sample = book[len(book) // 2]  # a representative mid-capture snapshot
    bid, ask = sample["best_bid"], sample["best_ask"]
    mid = (bid + ask) / 2
    spread = ask - bid
    spread_pct = spread / mid * 100

    modeled_long_fill = mid * (1 + config.SLIPPAGE)   # PaperEngine._fill(), adverse_up=True for a long entry
    modeled_short_fill = mid * (1 - config.SLIPPAGE)  # adverse_up=False for a short entry

    long_vs_ask = modeled_long_fill - ask       # negative = modeled fill is BETTER than the real ask (optimistic)
    short_vs_bid = modeled_short_fill - bid     # positive = modeled fill is BETTER than the real bid (optimistic)

    spreads = [r["spread"] for r in book]
    return {
        "sample_bid": bid, "sample_ask": ask, "sample_mid": mid,
        "sample_spread": spread, "sample_spread_pct": spread_pct,
        "modeled_long_fill": modeled_long_fill, "real_ask": ask,
        "long_fill_vs_real_ask": long_vs_ask,
        "long_fill_vs_real_ask_pct": long_vs_ask / ask * 100,
        "modeled_short_fill": modeled_short_fill, "real_bid": bid,
        "short_fill_vs_real_bid": short_vs_bid,
        "short_fill_vs_real_bid_pct": short_vs_bid / bid * 100,
        "spread_median_over_capture": float(np.median(spreads)),
        "spread_pct_median_over_capture": float(np.median(spreads) / mid * 100),
        "spread_max_over_capture": float(np.max(spreads)),
        "n_bookTicker_samples": len(book),
    }


if __name__ == "__main__":
    rows = load()
    by_type = {}
    for r in rows:
        by_type.setdefault(r["type"], []).append(r)
    print("message counts:", {k: len(v) for k, v in by_type.items()})

    print("\n=== DATA LATENCY (local_receive_ts - exchange_event_ts), aggTrade + kline streams ===")
    stats = latency_stats(rows)
    print(json.dumps(stats, indent=2))

    print("\n=== EXECUTION REALISM: modeled PaperEngine fill vs real best bid/ask ===")
    er = execution_realism(rows)
    print(json.dumps(er, indent=2))

    closed_klines = [r for r in by_type.get("kline", []) if r.get("is_closed")]
    print(f"\nclosed klines observed during capture: {len(closed_klines)}")
    for k in closed_klines:
        print(f"  close_ts={k['kline_close_ts']} O={k['open']} H={k['high']} L={k['low']} C={k['close']} V={k['volume']}")
