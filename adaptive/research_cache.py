"""
Local, extend-only historical candle cache for adaptive/research_service.py.
Never imported by adaptive/runner.py (the execution process) -- separate
process, separate concern, separate files. Extends via the SAME
cursor-discipline fetcher used for live trading (adaptive/cursor.py), so a
research-cache extension can never silently skip candles either: a gap is
detected and reported, not papered over.

CSV files, one per (symbol, timeframe), under a cache directory owned
entirely by the research process.
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from adaptive.cursor import advance  # noqa: E402
from adaptive.market_data import TF_MS, CANDLE_COLUMNS  # noqa: E402


class ResearchCache:
    def __init__(self, cache_dir: str, exchange):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.ex = exchange

    def _path(self, symbol: str, timeframe: str) -> str:
        safe = symbol.replace("/", "")
        return os.path.join(self.cache_dir, f"{safe}_{timeframe}.csv")

    def load(self, symbol: str, timeframe: str) -> pd.DataFrame:
        path = self._path(symbol, timeframe)
        if not os.path.exists(path):
            return pd.DataFrame(columns=CANDLE_COLUMNS)
        return pd.read_csv(path)

    def _fetch_since(self, symbol: str, timeframe: str, since_ms: int, limit: int) -> pd.DataFrame:
        raw = self.ex.fetch_ohlcv(symbol, timeframe, since=(since_ms or None), limit=limit)
        if not raw:
            return pd.DataFrame(columns=CANDLE_COLUMNS)
        df = pd.DataFrame(raw, columns=CANDLE_COLUMNS)
        now_ms = self.ex.milliseconds()
        tf_ms = TF_MS[timeframe]
        df = df[df["ts"] + tf_ms <= now_ms]  # never an unfinished/future candle
        return df.reset_index(drop=True)

    def extend(self, symbol: str, timeframe: str, limit: int = 1000):
        """Returns (updated_df, gap_detected). On a gap, the cache is
        extended only up to (not past) the hole -- callers should log the
        gap; research using this cache simply has less recent history
        until a future extend() call succeeds past it."""
        existing = self.load(symbol, timeframe)
        last_cursor = int(existing["ts"].iloc[-1]) if len(existing) else None
        result = advance(lambda since, lim: self._fetch_since(symbol, timeframe, since, lim),
                          last_cursor_ts=last_cursor, timeframe_ms=TF_MS[timeframe], limit=limit)
        if len(result.bars):
            merged = result.bars if len(existing) == 0 else pd.concat(
                [existing, result.bars], ignore_index=True)
            merged = merged.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
            merged.to_csv(self._path(symbol, timeframe), index=False)
            return merged, result.gap_detected
        return existing, result.gap_detected

    def extend_all(self, symbols, timeframes, limit: int = 1000):
        out = {}
        for sym in symbols:
            for tf in timeframes:
                df, gap = self.extend(sym, tf, limit=limit)
                out[(sym, tf)] = (df, gap)
        return out
