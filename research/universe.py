"""
Phase 3: eligible liquid universe, for Hypothesis B (cross-sectional
momentum). Objective, backtest-blind rules -- computed and frozen BEFORE
any strategy is run against this universe, so no symbol is here because its
backtest looked good.

Eligibility rule (per symbol, per day, causal/point-in-time):
  - listed (has data) for at least MIN_HISTORY_DAYS before this date
  - trailing 30-day average daily quote volume (close * base volume) >=
    MIN_QUOTE_VOLUME
  - no gap of more than MAX_GAP_DAYS in its own daily series up to this date

A symbol's ELIGIBLE SET is the list of dates on which it passed all three.
Hypothesis B's cross-sectional ranking, at each rebalance date, only
considers symbols eligible ON THAT DATE -- never using knowledge of which
symbols end up liquid/listed later (no future-universe leakage).
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HERE = os.path.dirname(os.path.abspath(__file__))
MIN_QUOTE_VOLUME = 20_000_000   # $20M/day trailing 30d average -- genuinely liquid, not a micro-cap
MIN_HISTORY_DAYS = 60           # need 60 days of prior data before a symbol can be considered
MAX_GAP_DAYS = 3                # any single gap longer than this disqualifies the symbol entirely

CANDIDATES = ["BTC/USDT", "ETH/USDT", "SOL/USDT",   # anchors, always included in the candidate pool
              "XRP/USDT", "ZEC/USDT", "BNB/USDT", "DOGE/USDT", "TRX/USDT", "SUI/USDT",
              "PEPE/USDT", "LTC/USDT", "LINK/USDT", "ADA/USDT", "AVAX/USDT", "DOT/USDT"]


def _load_daily(sym):
    path = os.path.join(HERE, "data", f"{sym.replace('/', '')}_1d.csv")
    df = pd.read_csv(path).drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df


def build_universe():
    per_symbol = {}
    integrity = {}
    for sym in CANDIDATES:
        df = _load_daily(sym)
        gaps = df["ts"].diff().dropna()
        max_gap_days = (gaps.max() / 86_400_000) if len(gaps) else 0
        n_gaps_over_limit = int((gaps > MAX_GAP_DAYS * 86_400_000).sum())
        integrity[sym] = {"rows": len(df), "start": str(df["dt"].iloc[0]), "end": str(df["dt"].iloc[-1]),
                           "max_gap_days": float(max_gap_days), "gaps_over_limit": n_gaps_over_limit}

        if n_gaps_over_limit > 0:
            per_symbol[sym] = []   # disqualified entirely -- data integrity issue, not a liquidity judgment
            continue

        df["quote_vol"] = df["close"] * df["volume"]
        df["trail30_qvol"] = df["quote_vol"].rolling(30, min_periods=30).mean()
        eligible_mask = (df["trail30_qvol"] >= MIN_QUOTE_VOLUME) & (df.index >= MIN_HISTORY_DAYS)
        per_symbol[sym] = df.loc[eligible_mask, "ts"].tolist()

    return per_symbol, integrity


def main():
    per_symbol, integrity = build_universe()
    summary = {}
    for sym, ts_list in per_symbol.items():
        summary[sym] = {"eligible_days": len(ts_list),
                         "eligible_pct_of_history": round(len(ts_list) / integrity[sym]["rows"] * 100, 1),
                         "integrity": integrity[sym]}
        print(f"{sym:12s} eligible {len(ts_list):4d}/{integrity[sym]['rows']} days "
              f"({summary[sym]['eligible_pct_of_history']}%)  "
              f"max_gap={integrity[sym]['max_gap_days']:.1f}d")

    out = {"rule": {"MIN_QUOTE_VOLUME": MIN_QUOTE_VOLUME, "MIN_HISTORY_DAYS": MIN_HISTORY_DAYS,
                     "MAX_GAP_DAYS": MAX_GAP_DAYS},
           "summary": summary,
           "eligible_days_by_symbol": {s: v for s, v in per_symbol.items()}}
    os.makedirs(os.path.join(HERE, "reports"), exist_ok=True)
    with open(os.path.join(HERE, "reports", "universe.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("\nwrote research/reports/universe.json")


if __name__ == "__main__":
    main()
