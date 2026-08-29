"""Deterministic tests for adaptive/research_cache.py."""
import os

import pandas as pd
import pytest

from adaptive.research_cache import ResearchCache
from adaptive.market_data import TF_MS

HOUR = TF_MS["1h"]
BASE = 1_800_000_000_000


class FakeExchange:
    def __init__(self, now_ms, rows):
        self._now_ms = now_ms
        self.rows = rows

    def milliseconds(self):
        return self._now_ms

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=1000):
        rows = self.rows
        if since is not None:
            rows = [r for r in rows if r[0] >= since]
        return rows[:limit]


def _rows(n, ts0=BASE, price=100.0):
    return [[ts0 + i * HOUR, price, price + 1, price - 1, price, 10.0] for i in range(n)]


def test_cache_extends_from_empty(tmp_path):
    now = BASE + 10 * HOUR
    ex = FakeExchange(now, _rows(10))
    cache = ResearchCache(str(tmp_path), ex)
    df, gap = cache.extend("BTC/USDT", "1h")
    assert len(df) == 10
    assert gap is False


def test_cache_persists_across_instances(tmp_path):
    now = BASE + 10 * HOUR
    ex = FakeExchange(now, _rows(10))
    cache1 = ResearchCache(str(tmp_path), ex)
    cache1.extend("BTC/USDT", "1h")

    cache2 = ResearchCache(str(tmp_path), ex)
    loaded = cache2.load("BTC/USDT", "1h")
    assert len(loaded) == 10


def test_cache_extends_incrementally_with_new_bars(tmp_path):
    now = BASE + 10 * HOUR
    ex = FakeExchange(now, _rows(10))
    cache = ResearchCache(str(tmp_path), ex)
    cache.extend("BTC/USDT", "1h")

    ex.rows = _rows(15)  # 5 more bars arrived
    ex._now_ms = BASE + 15 * HOUR
    df, gap = cache.extend("BTC/USDT", "1h")
    assert len(df) == 15
    assert gap is False


def test_cache_detects_gap_and_does_not_skip_past_it(tmp_path):
    now = BASE + 10 * HOUR
    ex = FakeExchange(now, _rows(10))
    cache = ResearchCache(str(tmp_path), ex)
    cache.extend("BTC/USDT", "1h")

    # simulate a gap: jump straight to bar 20 without 10-19 available
    ex.rows = [[BASE + 20 * HOUR, 100.0, 101.0, 99.0, 100.0, 10.0]]
    ex._now_ms = BASE + 21 * HOUR
    df, gap = cache.extend("BTC/USDT", "1h")
    assert gap is True
    assert len(df) == 10  # unchanged -- did not skip past the hole


def test_cache_never_includes_unfinished_candle(tmp_path):
    now = BASE + 5 * HOUR + 100  # bar 5 has JUST started, not closed
    rows = _rows(6)  # includes an in-progress 6th bar
    ex = FakeExchange(now, rows)
    cache = ResearchCache(str(tmp_path), ex)
    df, gap = cache.extend("BTC/USDT", "1h")
    assert len(df) == 5  # the still-forming bar excluded


def test_extend_all_covers_every_symbol_and_timeframe(tmp_path):
    now = BASE + 10 * HOUR
    ex = FakeExchange(now, _rows(10))
    cache = ResearchCache(str(tmp_path), ex)
    results = cache.extend_all(["BTC/USDT", "ETH/USDT"], ["1h", "4h"])
    assert set(results.keys()) == {("BTC/USDT", "1h"), ("BTC/USDT", "4h"),
                                     ("ETH/USDT", "1h"), ("ETH/USDT", "4h")}
