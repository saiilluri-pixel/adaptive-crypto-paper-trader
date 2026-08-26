"""
Local historical-data cache for research (Phase 2, step 4).

Fetches Binance PUBLIC OHLCV via ccxt (no keys), paginated past ccxt's
1000-candle-per-request limit, and caches to CSV under research/data/ so
repeated research doesn't keep re-hitting Binance.

Integrity checks on every fetch/load:
  - duplicate timestamps (dropped, logged)
  - non-monotonic timestamps (raises -- never silently reordered)
  - gaps (missing candles at the expected step) -- LOGGED, never filled

Never silently fabricates missing market data: a gap is reported as a gap.

    python3 research/data_cache.py --days 365          # fetch/refresh all
    python3 research/data_cache.py --days 365 --symbol BTC/USDT
"""
import argparse
import os
import sys
import time

import ccxt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
TF_MS = {"5m": 300_000, "30m": 1_800_000}


def _cache_path(symbol, timeframe):
    return os.path.join(DATA_DIR, f"{symbol.replace('/', '')}_{timeframe}.csv")


def _fetch_paginated(ex, symbol, timeframe, since_ms, until_ms):
    ms = TF_MS[timeframe]
    rows = []
    since = since_ms
    while since < until_ms:
        batch = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not batch:
            break
        rows += batch
        new_since = batch[-1][0] + ms
        if new_since <= since:      # safety: guarantee forward progress
            break
        since = new_since
        time.sleep(ex.rateLimit / 1000)
        if len(batch) < 1000:
            break
    return rows


def _integrity_check(df, timeframe, symbol):
    ms = TF_MS[timeframe]
    issues = []
    dupes = df["ts"].duplicated().sum()
    if dupes:
        issues.append(f"{dupes} duplicate timestamp(s) dropped")
        df = df.drop_duplicates("ts", keep="first")
    df = df.sort_values("ts").reset_index(drop=True)
    diffs = df["ts"].diff().dropna()
    non_monotonic = (diffs <= 0).sum()
    if non_monotonic:
        raise ValueError(f"{symbol} {timeframe}: {non_monotonic} non-monotonic timestamp(s) "
                          f"after sort -- refusing to silently reorder/proceed")
    expected = diffs.mode()[0] if len(diffs) else ms
    gap_count = int(((diffs > expected) & (diffs != expected)).sum())
    gap_bars_missing = int(((diffs[diffs > expected] // expected) - 1).sum()) if gap_count else 0
    if gap_count:
        issues.append(f"{gap_count} gap(s) totalling ~{gap_bars_missing} missing bar(s) "
                       f"(NOT filled -- left as real gaps)")
    return df, issues


def fetch_and_cache(symbol, timeframe, days, force=False):
    path = _cache_path(symbol, timeframe)
    os.makedirs(DATA_DIR, exist_ok=True)
    ex = getattr(ccxt, config.EXCHANGE)({"enableRateLimit": True})
    until_ms = ex.milliseconds()
    since_ms = until_ms - days * 86_400_000

    if os.path.exists(path) and not force:
        existing = pd.read_csv(path)
        cached_start, cached_end = int(existing["ts"].min()), int(existing["ts"].max())
        if cached_start <= since_ms + TF_MS[timeframe] * 2 and cached_end >= until_ms - TF_MS[timeframe] * 5:
            print(f"  [{symbol} {timeframe}] cache covers requested range ({len(existing)} bars) -- reusing")
            df, issues = _integrity_check(existing, timeframe, symbol)
            for m in issues:
                print(f"    !! {m}")
            return df
        print(f"  [{symbol} {timeframe}] cache exists but doesn't cover requested range -- refetching")

    print(f"  [{symbol} {timeframe}] fetching {days}d from Binance (public data only)…")
    rows = _fetch_paginated(ex, symbol, timeframe, since_ms, until_ms)
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df, issues = _integrity_check(df, timeframe, symbol)
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.to_csv(path, index=False)
    print(f"  [{symbol} {timeframe}] cached {len(df)} bars -> {path}")
    for m in issues:
        print(f"    !! {m}")
    return df


def load_cached(symbol, timeframe):
    path = _cache_path(symbol, timeframe)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No cache for {symbol} {timeframe} -- run data_cache.py first")
    df = pd.read_csv(path)
    df, issues = _integrity_check(df, timeframe, symbol)
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--symbol", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    symbols = [args.symbol] if args.symbol else config.SYMBOLS
    for sym in symbols:
        for tf in (config.CHART_TF, config.STRUCTURE_TF):
            df = fetch_and_cache(sym, tf, args.days, force=args.force)
            span = f"{df['dt'].iloc[0]:%Y-%m-%d} -> {df['dt'].iloc[-1]:%Y-%m-%d}"
            print(f"    {sym} {tf}: {len(df)} bars, {span}")


if __name__ == "__main__":
    main()
