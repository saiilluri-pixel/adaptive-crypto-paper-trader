"""
Binance public market-data feed via ccxt. No API keys — public endpoints only.

Provides closed OHLCV candles (chart + structure timeframes) and the live
last price for intrabar stop checks.
"""
import time
import ccxt
import pandas as pd

import config


class Feed:
    def __init__(self):
        self.ex = getattr(ccxt, config.EXCHANGE)({"enableRateLimit": True})

    def _ohlcv(self, symbol, timeframe, limit):
        raw = self.ex.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
        df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df

    def closed_candles(self, symbol, timeframe, limit):
        """Return only *closed* candles (drops the still-forming last bar)."""
        df = self._ohlcv(symbol, timeframe, limit + 2)
        # ccxt returns the in-progress candle as the last row; drop it.
        return df.iloc[:-1].reset_index(drop=True)

    def last_price(self, symbol):
        return float(self.ex.fetch_ticker(symbol)["last"])

    def retry(self, fn, *a, tries=5, wait=2, **k):
        for i in range(tries):
            try:
                return fn(*a, **k)
            except Exception as e:  # network blips, rate limits
                if i == tries - 1:
                    raise
                print(f"  [feed] {type(e).__name__}: {str(e)[:80]} — retry {i+1}/{tries}")
                time.sleep(wait)
