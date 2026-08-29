"""
REST-primary market data for BTC/USDT, ETH/USDT, SOL/USDT across
5m/15m/1h/4h timeframes, plus live best bid/ask. Public Binance Spot
endpoints only (ccxt.binance(), no keys, no authentication).

Honest scope note: adaptive spec section 5 states a WebSocket preference.
This module is REST-primary, not a maintained streaming connection. A
correct REST path built on adaptive/cursor.py's gap-integrity guarantee is
safer than a WebSocket path that risks reintroducing the exact defect
audited out of the SOL shadow (see cursor.py's docstring for the full
history) -- shipping REST-primary and saying so plainly, rather than
implying continuous streaming, is the honest tradeoff made here. Live
bid/ask are sourced from Binance's public ticker/order-book REST endpoints
at decision time, not a maintained in-memory order book.
"""
import os
import sys
from typing import Dict, Optional, Tuple

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from adaptive.cursor import advance, CursorResult  # noqa: E402

SYMBOLS = ("BTC/USDT", "ETH/USDT", "SOL/USDT")
TIMEFRAMES = ("5m", "15m", "1h", "4h")
TF_MS = {"5m": 300_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000}
CANDLE_COLUMNS = ["ts", "open", "high", "low", "close", "volume"]


def build_exchange():
    import ccxt
    ex = ccxt.binance({"enableRateLimit": True})
    ex.load_markets()
    return ex


class MarketData:
    def __init__(self, exchange=None):
        self.ex = exchange if exchange is not None else build_exchange()
        self.cursors: Dict[Tuple[str, str], Optional[int]] = {
            (s, tf): None for s in SYMBOLS for tf in TIMEFRAMES}
        self.candles: Dict[Tuple[str, str], pd.DataFrame] = {
            (s, tf): pd.DataFrame(columns=CANDLE_COLUMNS) for s in SYMBOLS for tf in TIMEFRAMES}
        self.gap_flags: Dict[Tuple[str, str], bool] = {
            (s, tf): False for s in SYMBOLS for tf in TIMEFRAMES}
        self.last_gap_detail: Dict[Tuple[str, str], Optional[str]] = {
            (s, tf): None for s in SYMBOLS for tf in TIMEFRAMES}

    def verify_spot_market(self, symbol: str) -> dict:
        m = self.ex.market(symbol)
        if not (m.get("spot") is True and m.get("contract") is False and m.get("swap") is False):
            raise RuntimeError(f"{symbol} did not resolve to a plain Spot market: {m}")
        return m

    def _fetch_since(self, symbol: str, timeframe: str, since_ms: int, limit: int) -> pd.DataFrame:
        raw = self.ex.fetch_ohlcv(symbol, timeframe, since=(since_ms or None), limit=limit)
        if not raw:
            return pd.DataFrame(columns=CANDLE_COLUMNS)
        df = pd.DataFrame(raw, columns=CANDLE_COLUMNS)
        now_ms = self.ex.milliseconds()
        tf_ms = TF_MS[timeframe]
        df = df[df["ts"] + tf_ms <= now_ms]  # keep only bars whose close time has passed
        return df.reset_index(drop=True)

    def poll(self, symbol: str, timeframe: str, limit: int = 500) -> CursorResult:
        key = (symbol, timeframe)
        result = advance(
            lambda since, lim: self._fetch_since(symbol, timeframe, since, lim),
            last_cursor_ts=self.cursors[key], timeframe_ms=TF_MS[timeframe], limit=limit)
        if len(result.bars):
            existing = self.candles[key]
            merged = result.bars if len(existing) == 0 else pd.concat(
                [existing, result.bars], ignore_index=True)
            self.candles[key] = merged.drop_duplicates("ts").tail(2000).reset_index(drop=True)
        if result.new_cursor_ts is not None:
            self.cursors[key] = result.new_cursor_ts
        self.gap_flags[key] = result.gap_detected
        self.last_gap_detail[key] = result.gap_detail
        return result

    def poll_all(self) -> Dict[Tuple[str, str], CursorResult]:
        return {(s, tf): self.poll(s, tf) for s in SYMBOLS for tf in TIMEFRAMES}

    def last_closed(self, symbol: str, timeframe: str) -> Optional[pd.Series]:
        df = self.candles[(symbol, timeframe)]
        return df.iloc[-1] if len(df) else None

    def best_bid_ask(self, symbol: str) -> Tuple[Optional[float], Optional[float]]:
        t = self.ex.fetch_ticker(symbol)
        bid, ask = t.get("bid"), t.get("ask")
        if bid is None or ask is None:
            ob = self.ex.fetch_order_book(symbol, limit=5)
            bid = ob["bids"][0][0] if ob.get("bids") else None
            ask = ob["asks"][0][0] if ob.get("asks") else None
        return bid, ask

    def data_is_stale(self, symbol: str, timeframe: str, max_age_ms: int) -> bool:
        df = self.candles[(symbol, timeframe)]
        if len(df) == 0:
            return True
        last_close = int(df["ts"].iloc[-1]) + TF_MS[timeframe]
        return (self.ex.milliseconds() - last_close) > max_age_ms

    def market_quality(self, symbol: str, primary_tf: str = "5m",
                        max_staleness_bars: float = 2.0, max_spread_pct: float = 0.5
                        ) -> Tuple[bool, list]:
        """Section 17 gate: blocks new ENTRIES only (callers must never
        route exit/stop logic through this check)."""
        reasons = []
        if self.gap_flags.get((symbol, primary_tf)):
            reasons.append(f"gap_detected:{self.last_gap_detail.get((symbol, primary_tf))}")
        if self.data_is_stale(symbol, primary_tf, int(TF_MS[primary_tf] * max_staleness_bars)):
            reasons.append("stale_data")
        try:
            bid, ask = self.best_bid_ask(symbol)
        except Exception as e:
            reasons.append(f"price_fetch_failed:{type(e).__name__}")
            bid = ask = None
        if bid is None or ask is None or bid <= 0 or ask <= 0:
            reasons.append("no_price_available")
        elif ask < bid:
            reasons.append("crossed_book")
        else:
            spread_pct = (ask - bid) / ((ask + bid) / 2) * 100
            if spread_pct > max_spread_pct:
                reasons.append(f"spread_too_wide:{spread_pct:.4f}%")
        return (len(reasons) == 0, reasons)
