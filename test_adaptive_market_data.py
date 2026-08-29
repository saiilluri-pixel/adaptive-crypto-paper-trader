"""Deterministic tests for adaptive/market_data.py -- no network calls, uses
a fake ccxt-shaped exchange object."""
import pandas as pd
import pytest

from adaptive.market_data import MarketData, SYMBOLS, TIMEFRAMES, TF_MS

HOUR = TF_MS["1h"]
BASE = 1_800_000_000_000


class FakeExchange:
    def __init__(self, now_ms, candles_by_key=None, ticker=None, markets=None):
        self._now_ms = now_ms
        self.candles_by_key = candles_by_key or {}  # (symbol,timeframe) -> list of [ts,o,h,l,c,v]
        self.ticker = ticker or {}
        self._markets = markets or {}

    def milliseconds(self):
        return self._now_ms

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=500):
        rows = self.candles_by_key.get((symbol, timeframe), [])
        if since is not None:
            rows = [r for r in rows if r[0] >= since]
        return rows[:limit]

    def fetch_ticker(self, symbol):
        return self.ticker.get(symbol, {"bid": None, "ask": None})

    def fetch_order_book(self, symbol, limit=5):
        return {"bids": [[99.0, 1.0]], "asks": [[100.0, 1.0]]}

    def market(self, symbol):
        return self._markets.get(symbol, {"spot": True, "contract": False, "swap": False})


def _candle(ts, price=100.0, v=10.0):
    return [ts, price, price + 1, price - 1, price, v]


def test_poll_advances_cursor_and_stores_closed_candles():
    now = BASE + 3 * HOUR + 1000  # well past the 3rd bar's close
    rows = [_candle(BASE), _candle(BASE + HOUR), _candle(BASE + 2 * HOUR)]
    ex = FakeExchange(now, candles_by_key={("BTC/USDT", "1h"): rows})
    md = MarketData(exchange=ex)
    result = md.poll("BTC/USDT", "1h")
    assert result.gap_detected is False
    assert len(md.candles[("BTC/USDT", "1h")]) == 3
    assert md.cursors[("BTC/USDT", "1h")] == BASE + 2 * HOUR


def test_still_forming_candle_is_excluded():
    now = BASE + HOUR + 100  # only 1 full hour has elapsed since BASE
    rows = [_candle(BASE), _candle(BASE + HOUR)]  # second one is still forming
    ex = FakeExchange(now, candles_by_key={("BTC/USDT", "1h"): rows})
    md = MarketData(exchange=ex)
    result = md.poll("BTC/USDT", "1h")
    assert len(result.bars) == 1
    assert md.cursors[("BTC/USDT", "1h")] == BASE


def test_gap_flag_set_and_visible_via_market_quality():
    now = BASE + 5 * HOUR
    ex = FakeExchange(now, candles_by_key={("BTC/USDT", "5m"): [_candle(BASE)]},
                       ticker={"BTC/USDT": {"bid": 99.99, "ask": 100.01}})
    md = MarketData(exchange=ex)
    md.poll("BTC/USDT", "5m")  # cold start, cursor = BASE
    # now simulate a gap: next poll returns a bar far past BASE+5m
    ex.candles_by_key[("BTC/USDT", "5m")] = [_candle(BASE + 10 * TF_MS["5m"])]
    md.poll("BTC/USDT", "5m")
    assert md.gap_flags[("BTC/USDT", "5m")] is True
    ok, reasons = md.market_quality("BTC/USDT", primary_tf="5m")
    assert ok is False
    assert any("gap_detected" in r for r in reasons)


def test_best_bid_ask_from_ticker():
    ex = FakeExchange(BASE, ticker={"BTC/USDT": {"bid": 100.0, "ask": 100.05}})
    md = MarketData(exchange=ex)
    bid, ask = md.best_bid_ask("BTC/USDT")
    assert bid == 100.0 and ask == 100.05


def test_best_bid_ask_falls_back_to_order_book_when_ticker_missing():
    ex = FakeExchange(BASE, ticker={"BTC/USDT": {"bid": None, "ask": None}})
    md = MarketData(exchange=ex)
    bid, ask = md.best_bid_ask("BTC/USDT")
    assert bid == 99.0 and ask == 100.0  # from FakeExchange.fetch_order_book


def test_stale_data_blocks_market_quality():
    ex = FakeExchange(BASE + 100 * HOUR,  # long after any candle
                       candles_by_key={("BTC/USDT", "5m"): [_candle(BASE)]},
                       ticker={"BTC/USDT": {"bid": 99.99, "ask": 100.01}})
    md = MarketData(exchange=ex)
    md.poll("BTC/USDT", "5m")
    ok, reasons = md.market_quality("BTC/USDT", primary_tf="5m")
    assert ok is False
    assert "stale_data" in reasons


def test_wide_spread_blocks_market_quality():
    now = BASE + HOUR
    ex = FakeExchange(now, candles_by_key={("BTC/USDT", "5m"): [_candle(BASE)]},
                       ticker={"BTC/USDT": {"bid": 90.0, "ask": 110.0}})  # ~20% spread
    md = MarketData(exchange=ex)
    md.poll("BTC/USDT", "5m")
    ok, reasons = md.market_quality("BTC/USDT", primary_tf="5m", max_spread_pct=0.5)
    assert ok is False
    assert any("spread_too_wide" in r for r in reasons)


def test_clean_market_passes_quality_check():
    now = BASE + TF_MS["5m"] + 10_000
    ex = FakeExchange(now, candles_by_key={("BTC/USDT", "5m"): [_candle(BASE)]},
                       ticker={"BTC/USDT": {"bid": 99.99, "ask": 100.01}})
    md = MarketData(exchange=ex)
    md.poll("BTC/USDT", "5m")
    ok, reasons = md.market_quality("BTC/USDT", primary_tf="5m")
    assert ok is True
    assert reasons == []


def test_verify_spot_market_accepts_plain_spot():
    ex = FakeExchange(BASE, markets={"BTC/USDT": {"spot": True, "contract": False, "swap": False}})
    md = MarketData(exchange=ex)
    m = md.verify_spot_market("BTC/USDT")
    assert m["spot"] is True


def test_verify_spot_market_rejects_futures():
    ex = FakeExchange(BASE, markets={"BTC/USDT": {"spot": False, "contract": True, "swap": True}})
    md = MarketData(exchange=ex)
    with pytest.raises(RuntimeError):
        md.verify_spot_market("BTC/USDT")


def test_all_symbols_and_timeframes_initialized():
    md = MarketData(exchange=FakeExchange(BASE))
    for s in SYMBOLS:
        for tf in TIMEFRAMES:
            assert (s, tf) in md.candles
            assert (s, tf) in md.cursors
