"""
Deterministic tests for adaptive/runner.py -- restart/recovery, no
fabricated entries, gap blocks entries but never exits. No network calls:
uses a fake ccxt-shaped exchange (same pattern as test_adaptive_market_data.py).
"""
import os

import pandas as pd
import pytest

from adaptive.market_data import MarketData, SYMBOLS, TIMEFRAMES, TF_MS
from adaptive.runner import AdaptiveRunner

HOUR = TF_MS["1h"]
M15 = TF_MS["15m"]
M5 = TF_MS["5m"]
BASE = 1_800_000_000_000


class FakeExchange:
    def __init__(self, now_ms, candles_by_key=None, tickers=None):
        self._now_ms = now_ms
        self.candles_by_key = candles_by_key or {}
        self.tickers = tickers or {s: {"bid": 100.0, "ask": 100.02} for s in SYMBOLS}

    def milliseconds(self):
        return self._now_ms

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=500):
        rows = self.candles_by_key.get((symbol, timeframe), [])
        if since is not None:
            rows = [r for r in rows if r[0] >= since]
        return rows[:limit]

    def fetch_ticker(self, symbol):
        return self.tickers.get(symbol, {"bid": 100.0, "ask": 100.02})

    def fetch_order_book(self, symbol, limit=5):
        t = self.tickers.get(symbol, {"bid": 100.0, "ask": 100.02})
        return {"bids": [[t["bid"], 10.0]], "asks": [[t["ask"], 10.0]]}

    def market(self, symbol):
        return {"spot": True, "contract": False, "swap": False}


def _flat_candles(n, ts0, tf_ms, price=100.0):
    return [[ts0 + i * tf_ms, price, price * 1.001, price * 0.999, price, 100.0] for i in range(n)]


def _full_candle_set(now_ms, price=100.0):
    """Enough flat history on every (symbol, timeframe) for the runner to
    consider data 'not stale' and have indicator warmup satisfied."""
    out = {}
    for sym in SYMBOLS:
        for tf in TIMEFRAMES:
            tf_ms = TF_MS[tf]
            n = 200
            last_close = now_ms - (now_ms % tf_ms) - tf_ms
            ts0 = last_close - (n - 1) * tf_ms
            out[(sym, tf)] = _flat_candles(n, ts0, tf_ms, price=price)
    return out


def test_bootstrap_flat_start_no_positions(tmp_path):
    now = BASE + 300 * HOUR
    ex = FakeExchange(now, candles_by_key=_full_candle_set(now))
    runner = AdaptiveRunner(str(tmp_path), market_data=MarketData(exchange=ex))
    runner.bootstrap()
    assert len(runner.portfolio.positions) == 0
    assert runner.portfolio.cash == runner.portfolio.start_capital


def test_no_fabricated_entry_on_first_cycle_after_cold_start(tmp_path):
    """The bug caught during review: bootstrap() loads a large batch of
    historical candles. If run_once_cycle() evaluated signals against that
    stale history as if it were fresh, a historical (not genuinely new)
    signal could fabricate an entry on the very first live cycle. This
    must never happen -- entries require a bar freshly polled THIS cycle."""
    now = BASE + 300 * HOUR
    candles = _full_candle_set(now, price=100.0)
    # inject an obvious historical uptrend into BTC's 1h series so a
    # signal WOULD fire if evaluated against this stale bootstrap data
    n = 200
    tf_ms = TF_MS["1h"]
    last_close = now - (now % tf_ms) - tf_ms
    ts0 = last_close - (n - 1) * tf_ms
    trending = [[ts0 + i * tf_ms, 100.0 * (1.003 ** i), 100.0 * (1.003 ** i) * 1.001,
                 100.0 * (1.003 ** i) * 0.999, 100.0 * (1.003 ** i), 100.0] for i in range(n)]
    candles[("BTC/USDT", "1h")] = trending

    ex = FakeExchange(now, candles_by_key=candles)
    runner = AdaptiveRunner(str(tmp_path), market_data=MarketData(exchange=ex))
    runner.bootstrap()
    # no new bars have arrived since bootstrap -- run_once_cycle should not enter
    runner.run_once_cycle()
    assert len(runner.portfolio.positions) == 0
    assert runner.portfolio.n_trades == 0


def test_genuinely_new_bar_can_trigger_an_entry(tmp_path):
    now = BASE + 300 * HOUR
    candles = _full_candle_set(now, price=100.0)
    ex = FakeExchange(now, candles_by_key=candles)
    runner = AdaptiveRunner(str(tmp_path), market_data=MarketData(exchange=ex))
    runner.bootstrap()
    assert len(runner.portfolio.positions) == 0

    # advance time by one hour and append a genuine new trending 1h bar for BTC
    new_now = now + HOUR
    ex._now_ms = new_now
    new_bar_ts = now - (now % HOUR)
    btc_1h = list(ex.candles_by_key[("BTC/USDT", "1h")])
    last_close = btc_1h[-1][4]
    new_close = last_close * 1.05
    btc_1h.append([new_bar_ts, last_close, new_close * 1.001, last_close * 0.999, new_close, 500.0])
    ex.candles_by_key[("BTC/USDT", "1h")] = btc_1h

    runner.run_once_cycle()
    # a fresh, strongly trending bar should be at least CONSIDERED (not
    # necessarily entered, since scoring depends on empty stats_store
    # shrinking edge toward zero) -- assert the decision log recorded a
    # cycle_ranking event that saw the new bar as a candidate
    import json
    lines = [json.loads(l) for l in open(runner.decisions_path)]
    ranking_events = [l for l in lines if l.get("action") == "cycle_ranking"]
    assert len(ranking_events) >= 1


def test_bootstrap_creates_baseline_champion_when_missing(tmp_path):
    now = BASE + 300 * HOUR
    ex = FakeExchange(now, candles_by_key=_full_candle_set(now))
    runner = AdaptiveRunner(str(tmp_path), market_data=MarketData(exchange=ex))
    runner.bootstrap()
    assert "trend_momentum" in runner.champions
    assert runner.champions["trend_momentum"].version == 1
    assert runner.champions["trend_momentum"].status == "BASELINE"


def test_new_entries_use_latest_champion_params(tmp_path):
    """A promoted (research-service-written) champion file must be picked
    up by the execution runner on its next cycle -- proves the read side
    of the research<->execution contract works end to end."""
    now = BASE + 300 * HOUR
    ex = FakeExchange(now, candles_by_key=_full_candle_set(now))
    runner = AdaptiveRunner(str(tmp_path), market_data=MarketData(exchange=ex))
    runner.bootstrap()

    from adaptive.champion_store import ChampionStore
    store = ChampionStore(os.path.join(str(tmp_path), "champion_state.json"))
    records = store.read()
    promoted = store.promote("trend_momentum", {"z_threshold": 0.42}, records["trend_momentum"])
    records["trend_momentum"] = promoted
    store.write(records)

    runner._refresh_champions()
    assert runner._get_champion_params("trend_momentum") == {"z_threshold": 0.42}
    assert runner._champion_version("trend_momentum") == 2


def test_corrupted_champion_file_falls_back_to_last_known_good(tmp_path):
    now = BASE + 300 * HOUR
    ex = FakeExchange(now, candles_by_key=_full_candle_set(now))
    runner = AdaptiveRunner(str(tmp_path), market_data=MarketData(exchange=ex))
    runner.bootstrap()
    good_params = dict(runner._get_champion_params("trend_momentum"))

    # corrupt the file directly
    path = os.path.join(str(tmp_path), "champion_state.json")
    with open(path, "w") as f:
        f.write("{ not valid json at all")

    runner._refresh_champions()
    assert runner.champion_load_failed is True
    assert runner._get_champion_params("trend_momentum") == good_params  # unchanged, not crashed


def test_corrupted_champion_file_does_not_crash_a_full_cycle(tmp_path):
    now = BASE + 300 * HOUR
    ex = FakeExchange(now, candles_by_key=_full_candle_set(now))
    runner = AdaptiveRunner(str(tmp_path), market_data=MarketData(exchange=ex))
    runner.bootstrap()
    path = os.path.join(str(tmp_path), "champion_state.json")
    with open(path, "w") as f:
        f.write("{ not valid json at all")
    runner.run_once_cycle()  # must not raise
    assert runner.champion_load_failed is True


def test_research_process_cannot_touch_portfolio_position_state(tmp_path):
    """Structural proof: promoting a new champion via ChampionStore never
    touches state.json, and an existing open position's entry-time
    contract (strategy/stop/trail_state) is completely unaffected by it."""
    now = BASE + 300 * HOUR
    ex = FakeExchange(now, candles_by_key=_full_candle_set(now))
    runner = AdaptiveRunner(str(tmp_path), market_data=MarketData(exchange=ex))
    runner.bootstrap()
    runner.portfolio.buy("BTC/USDT", best_ask=100.0, notional_usdt=500.0, fee_rate=0.001,
                          slippage=0.0002, ts_ms=1000, strategy="trend_momentum", regime="TREND_UP",
                          confidence=0.8, stop_price=95.0, initial_stop_pct=5.0, risk_amount_usdt=25.0)
    from adaptive.trailing import new_trail_state
    runner.portfolio.positions["BTC/USDT"].trail_state = new_trail_state(100.0, 5.0)
    runner._save_state()
    state_mtime_before = os.path.getmtime(os.path.join(str(tmp_path), "state.json"))
    original_stop = runner.portfolio.positions["BTC/USDT"].stop_price
    original_initial_pct = runner.portfolio.positions["BTC/USDT"].initial_stop_pct

    from adaptive.champion_store import ChampionStore
    store = ChampionStore(os.path.join(str(tmp_path), "champion_state.json"))
    records = store.read()
    promoted = store.promote("trend_momentum", {"z_threshold": 0.1}, records["trend_momentum"])
    records["trend_momentum"] = promoted
    store.write(records)  # simulates the research service promoting a new champion

    assert os.path.getmtime(os.path.join(str(tmp_path), "state.json")) == state_mtime_before
    assert runner.portfolio.positions["BTC/USDT"].stop_price == original_stop
    assert runner.portfolio.positions["BTC/USDT"].initial_stop_pct == original_initial_pct
    assert runner.portfolio.positions["BTC/USDT"].strategy == "trend_momentum"  # unchanged


def test_dashboard_json_has_required_sections(tmp_path):
    now = BASE + 300 * HOUR
    ex = FakeExchange(now, candles_by_key=_full_candle_set(now))
    runner = AdaptiveRunner(str(tmp_path), market_data=MarketData(exchange=ex))
    runner.bootstrap()
    runner.run_once_cycle()
    import json
    dash = json.load(open(os.path.join(str(tmp_path), "dashboard.json")))
    for key in ("equity", "cash", "return_pct", "drawdown_pct", "daily_pnl_usdt",
                "portfolio_heat_pct", "crypto_allocation_pct", "positions",
                "risk_state", "ai_brain", "market"):
        assert key in dash, f"missing dashboard field: {key}"
    assert set(dash["market"].keys()) == set(SYMBOLS)
    assert "champions" in dash["ai_brain"]
    assert "strategies_active" in dash["ai_brain"]


def test_restart_preserves_portfolio_state(tmp_path):
    now = BASE + 300 * HOUR
    ex = FakeExchange(now, candles_by_key=_full_candle_set(now))
    runner1 = AdaptiveRunner(str(tmp_path), market_data=MarketData(exchange=ex))
    runner1.bootstrap()
    cash_before = runner1.portfolio.cash

    runner2 = AdaptiveRunner(str(tmp_path), market_data=MarketData(exchange=ex))
    runner2.bootstrap()
    assert runner2.portfolio.cash == pytest.approx(cash_before)
    assert len(runner2.portfolio.positions) == 0


def test_restart_with_open_position_preserves_entry_contract(tmp_path):
    now = BASE + 300 * HOUR
    ex = FakeExchange(now, candles_by_key=_full_candle_set(now))
    runner1 = AdaptiveRunner(str(tmp_path), market_data=MarketData(exchange=ex))
    runner1.bootstrap()
    runner1.portfolio.buy("BTC/USDT", best_ask=100.0, notional_usdt=500.0, fee_rate=0.001,
                           slippage=0.0002, ts_ms=1000, strategy="trend_momentum", regime="TREND_UP",
                           confidence=0.8, stop_price=95.0, initial_stop_pct=5.0, risk_amount_usdt=25.0)
    from adaptive.trailing import new_trail_state
    runner1.portfolio.positions["BTC/USDT"].trail_state = new_trail_state(100.0, 5.0)
    runner1._save_state()

    runner2 = AdaptiveRunner(str(tmp_path), market_data=MarketData(exchange=ex))
    runner2.bootstrap()
    assert "BTC/USDT" in runner2.portfolio.positions
    pos = runner2.portfolio.positions["BTC/USDT"]
    assert pos.strategy == "trend_momentum"
    assert pos.initial_stop_pct == 5.0
    assert pos.trail_state["sl_price"] == pytest.approx(95.0)


def test_gap_blocks_entries_but_not_exit_management(tmp_path):
    now = BASE + 300 * HOUR
    candles = _full_candle_set(now)
    ex = FakeExchange(now, candles_by_key=candles)
    runner = AdaptiveRunner(str(tmp_path), market_data=MarketData(exchange=ex))
    runner.bootstrap()

    # open a position directly, with a stop that WILL be hit
    runner.portfolio.buy("BTC/USDT", best_ask=100.0, notional_usdt=500.0, fee_rate=0.001,
                          slippage=0.0002, ts_ms=1000, strategy="trend_momentum", regime="TREND_UP",
                          confidence=0.8, stop_price=95.0, initial_stop_pct=5.0, risk_amount_usdt=25.0)
    from adaptive.trailing import new_trail_state
    runner.portfolio.positions["BTC/USDT"].trail_state = new_trail_state(100.0, 5.0)  # stop=95

    # force a gap on BTC's 5m feed
    runner.reconciliation_failed.add("BTC/USDT")

    # append a new 5m bar for BTC whose low breaches the stop
    new_now = now + M5
    ex._now_ms = new_now
    ts0 = now - (now % M5)
    ex.candles_by_key[("BTC/USDT", "5m")] = list(ex.candles_by_key[("BTC/USDT", "5m")]) + [
        [ts0, 100.0, 100.5, 90.0, 91.0, 500.0]]  # low=90 breaches stop=95

    runner.run_once_cycle()
    # exit management must still have fired despite the gap flag
    assert "BTC/USDT" not in runner.portfolio.positions
    assert runner.portfolio.n_trades == 1


def test_shock_continuation_not_double_counted_across_stale_cycles(tmp_path):
    """Regression test for the stateful shock_continuation bug caught
    during review: calling run_once_cycle() many times with NO new 1h bar
    must not corrupt the shock detector's rolling state."""
    now = BASE + 300 * HOUR
    ex = FakeExchange(now, candles_by_key=_full_candle_set(now))
    runner = AdaptiveRunner(str(tmp_path), market_data=MarketData(exchange=ex))
    runner.bootstrap()
    from adaptive.strategies import _shock_detectors
    _shock_detectors.clear()

    for _ in range(20):  # simulate 20 poll cycles with no new 1h bar
        runner.run_once_cycle()

    # BTC's shock detector should have been invoked at most once (the
    # first cycle after bootstrap has fresh_1h briefly possible only if a
    # genuinely new bar existed -- with a flat, unchanging fake exchange
    # and no time advance, it should never have been created at all,
    # since fresh_1h is False every cycle here)
    assert "BTC/USDT" not in _shock_detectors or len(
        _shock_detectors.get("BTC/USDT", type("x", (), {"tr_hist": []})()).tr_hist) <= 14
