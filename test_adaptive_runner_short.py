"""
Deterministic tests for the SIMULATED SHORT / MARGIN book's integration
into adaptive/runner.py -- exact mirror of test_adaptive_runner.py's
long-side coverage, for the short-side wiring added per explicit user
request ("I want long and short positions"). Same fake-ccxt-exchange
pattern, no network calls. See adaptive/short_portfolio.py's own module
docstring for the safety/structural-isolation rationale.
"""
import json
import os

import pytest

from adaptive.market_data import MarketData, SYMBOLS, TIMEFRAMES, TF_MS
from adaptive.runner import AdaptiveRunner

HOUR = TF_MS["1h"]
M15 = TF_MS["15m"]
M5 = TF_MS["5m"]
BASE = 1_800_000_000_000

# Reuse the exact fake-exchange + candle-set helpers from the long-side
# test module rather than duplicating them.
from test_adaptive_runner import FakeExchange, _full_candle_set  # noqa: E402


def _fresh_down_trend_bar(ex, symbol, now, tf_key="1h"):
    """Mirror of test_adaptive_runner.py's _fresh_dip_bar, but a sharp
    DOWN move on the 1h chart -- triggers trend_momentum's exit_long
    signal with a strongly positive bearish_research_strength (the one
    signal path score_short_opportunity() consumes). Also fills the 5m
    series up to the same new `now` so market_quality's staleness check
    doesn't reject the symbol."""
    tf_ms = TF_MS[tf_key]
    new_now = now + tf_ms
    ex._now_ms = new_now
    series = list(ex.candles_by_key[(symbol, tf_key)])
    last_close = series[-1][4]
    down_close = last_close * 0.95  # -5%, mirrors the +5% used for the long-side equivalent test
    new_bar_ts = series[-1][0] + tf_ms
    series.append([new_bar_ts, last_close, last_close * 1.001, down_close * 0.999, down_close, 500.0])
    ex.candles_by_key[(symbol, tf_key)] = series

    m5_ms = TF_MS["5m"]
    m5_series = list(ex.candles_by_key[(symbol, "5m")])
    next_ts = m5_series[-1][0] + m5_ms
    while next_ts + m5_ms <= new_now:
        m5_series.append([next_ts, down_close, down_close * 1.001, down_close * 0.999, down_close, 100.0])
        next_ts += m5_ms
    ex.candles_by_key[(symbol, "5m")] = m5_series

    ex.tickers[symbol] = {"bid": down_close * 0.9998, "ask": down_close * 1.0002}
    return new_now


def _btc_down_trend_bar(ex, now, tf_key="1h"):
    return _fresh_down_trend_bar(ex, "BTC/USDT", now, tf_key)


def test_bootstrap_flat_start_short_book_also_flat(tmp_path):
    now = BASE + 300 * HOUR
    ex = FakeExchange(now, candles_by_key=_full_candle_set(now))
    runner = AdaptiveRunner(str(tmp_path), market_data=MarketData(exchange=ex))
    runner.bootstrap()
    assert len(runner.short_portfolio.positions) == 0
    assert runner.short_portfolio.cash == runner.short_portfolio.start_capital
    # the two books must never be conflated
    assert runner.short_portfolio is not runner.portfolio
    assert runner.short_state_path != runner.state_path
    assert runner.short_trades_csv_path != runner.trades_csv_path


def test_fresh_system_can_take_its_first_short_via_exploration(tmp_path):
    """Mirror of test_fresh_system_can_take_its_first_trade_via_exploration
    for the short book: a strong, fresh bearish signal (trend_momentum's
    exit_long with bearish_research_strength near 1.0) opens a genuine
    paper SHORT via the exploration path on a cold (n=0) stats cell."""
    now = BASE + 300 * HOUR
    ex = FakeExchange(now, candles_by_key=_full_candle_set(now))
    runner = AdaptiveRunner(str(tmp_path), market_data=MarketData(exchange=ex))
    runner.bootstrap()
    assert len(runner.short_portfolio.positions) == 0

    _btc_down_trend_bar(ex, now)
    runner.run_once_cycle()

    assert "BTC/USDT" in runner.short_portfolio.positions
    pos = runner.short_portfolio.positions["BTC/USDT"]
    assert pos.strategy == "trend_momentum"
    assert runner.short_exploration_symbols == {"BTC/USDT"}
    # exploration-sized (<=0.10% of ~2708 equity => a few dollars), not normal (0.50%)
    assert pos.risk_amount_usdt < 10.0
    # never touched the long book
    assert len(runner.portfolio.positions) == 0
    assert runner.portfolio.cash == runner.portfolio.start_capital


def test_short_stop_out_closes_position_and_credits_cash(tmp_path):
    now = BASE + 300 * HOUR
    ex = FakeExchange(now, candles_by_key=_full_candle_set(now))
    runner = AdaptiveRunner(str(tmp_path), market_data=MarketData(exchange=ex))
    runner.bootstrap()

    runner.short_portfolio.sell_to_open(
        "BTC/USDT", best_bid=100.0, notional_usdt=200.0, fee_rate=0.001, slippage=0.0002,
        ts_ms=now, strategy="trend_momentum", regime="TREND_DOWN", confidence=0.8,
        initial_stop_pct=5.0, risk_amount_usdt=10.0)
    from adaptive.trailing import new_short_trail_state
    runner.short_portfolio.positions["BTC/USDT"].trail_state = new_short_trail_state(100.0, 5.0)  # stop=105
    cash_after_open = runner.short_portfolio.cash

    # append a new 5m bar for BTC whose high breaches the stop (105)
    new_now = now + M5
    ex._now_ms = new_now
    ts0 = now - (now % M5)
    ex.candles_by_key[("BTC/USDT", "5m")] = list(ex.candles_by_key[("BTC/USDT", "5m")]) + [
        [ts0, 100.0, 110.0, 99.5, 108.0, 500.0]]  # high=110 breaches stop=105

    runner.run_once_cycle()
    assert "BTC/USDT" not in runner.short_portfolio.positions
    assert runner.short_portfolio.n_trades == 1
    assert runner.short_portfolio.cash > cash_after_open  # margin + any residual credited back
    # never touched the long book
    assert runner.portfolio.n_trades == 0


def test_short_liquidation_forces_close_even_without_stop(tmp_path):
    """A short whose stop is far away (or somehow never checked) must
    still be force-closed once price rises past the simulated liquidation
    threshold -- mirrors what a real margin account would do."""
    now = BASE + 300 * HOUR
    ex = FakeExchange(now, candles_by_key=_full_candle_set(now))
    runner = AdaptiveRunner(str(tmp_path), market_data=MarketData(exchange=ex))
    runner.bootstrap()

    # wide stop (50%) so liquidation (leverage=2x -> ~40% adverse move) fires first
    runner.short_portfolio.sell_to_open(
        "BTC/USDT", best_bid=100.0, notional_usdt=200.0, fee_rate=0.001, slippage=0.0002,
        ts_ms=now, strategy="trend_momentum", regime="TREND_DOWN", confidence=0.8,
        initial_stop_pct=50.0, risk_amount_usdt=10.0)
    from adaptive.trailing import new_short_trail_state
    pos = runner.short_portfolio.positions["BTC/USDT"]
    pos.trail_state = new_short_trail_state(100.0, 50.0)  # stop=150, far away
    liq_price = pos.liquidation_price
    assert liq_price < 150.0  # liquidation is the tighter constraint here

    new_now = now + M5
    ex._now_ms = new_now
    ts0 = now - (now % M5)
    breach = liq_price + 1.0
    ex.candles_by_key[("BTC/USDT", "5m")] = list(ex.candles_by_key[("BTC/USDT", "5m")]) + [
        [ts0, 100.0, breach, 99.5, breach - 0.5, 500.0]]
    ex.tickers["BTC/USDT"] = {"bid": breach * 0.998, "ask": breach * 1.002}

    runner.run_once_cycle()
    assert "BTC/USDT" not in runner.short_portfolio.positions
    lines = [json.loads(l) for l in open(runner.short_decisions_path)]
    exits = [l for l in lines if l.get("action") == "short_exited" and l.get("symbol") == "BTC/USDT"]
    assert len(exits) == 1
    assert exits[0]["exit_reason"] == "liquidation"


def test_bullish_signal_covers_an_existing_short(tmp_path):
    """Mirror of the long book's exit_long-closes-a-long behavior: a
    bullish ("long") signal reading covers an existing short position in
    that symbol, via _exit_short_signal_management."""
    now = BASE + 300 * HOUR
    ex = FakeExchange(now, candles_by_key=_full_candle_set(now))
    runner = AdaptiveRunner(str(tmp_path), market_data=MarketData(exchange=ex))
    runner.bootstrap()
    prices = {"BTC/USDT": 100.0}

    runner.short_portfolio.sell_to_open(
        "BTC/USDT", best_bid=100.0, notional_usdt=200.0, fee_rate=0.001, slippage=0.0002,
        ts_ms=now, strategy="trend_momentum", regime="TREND_DOWN", confidence=0.8,
        initial_stop_pct=5.0, risk_amount_usdt=10.0)

    runner._exit_short_signal_management("BTC/USDT", "long", prices)
    assert "BTC/USDT" not in runner.short_portfolio.positions
    assert runner.short_portfolio.n_trades == 1

    lines = [json.loads(l) for l in open(runner.short_decisions_path)]
    exits = [l for l in lines if l.get("action") == "short_exited"]
    assert len(exits) == 1
    assert "exit signal" in exits[0]["exit_reason"]


def test_exploration_slot_frees_up_after_short_position_closes(tmp_path):
    now = BASE + 300 * HOUR
    ex = FakeExchange(now, candles_by_key=_full_candle_set(now))
    runner = AdaptiveRunner(str(tmp_path), market_data=MarketData(exchange=ex))
    runner.bootstrap()
    _btc_down_trend_bar(ex, now)
    runner.run_once_cycle()
    assert runner.short_exploration_symbols == {"BTC/USDT"}

    rec = runner.short_portfolio.buy_to_close("BTC/USDT", best_ask=90.0, fee_rate=0.001,
                                                slippage=0.0002, ts_ms=1000, exit_reason="test")
    runner._on_short_trade_closed(rec)
    assert runner.short_exploration_symbols == set()


def test_short_exploration_symbols_persist_across_restart(tmp_path):
    now = BASE + 300 * HOUR
    ex = FakeExchange(now, candles_by_key=_full_candle_set(now))
    runner1 = AdaptiveRunner(str(tmp_path), market_data=MarketData(exchange=ex))
    runner1.bootstrap()
    _btc_down_trend_bar(ex, now)
    runner1.run_once_cycle()
    assert runner1.short_exploration_symbols == {"BTC/USDT"}

    runner2 = AdaptiveRunner(str(tmp_path), market_data=MarketData(exchange=ex))
    runner2.bootstrap()
    assert runner2.short_exploration_symbols == {"BTC/USDT"}


def test_restart_preserves_short_portfolio_state(tmp_path):
    now = BASE + 300 * HOUR
    ex = FakeExchange(now, candles_by_key=_full_candle_set(now))
    runner1 = AdaptiveRunner(str(tmp_path), market_data=MarketData(exchange=ex))
    runner1.bootstrap()
    cash_before = runner1.short_portfolio.cash

    runner2 = AdaptiveRunner(str(tmp_path), market_data=MarketData(exchange=ex))
    runner2.bootstrap()
    assert runner2.short_portfolio.cash == pytest.approx(cash_before)
    assert len(runner2.short_portfolio.positions) == 0


def test_restart_with_open_short_preserves_entry_contract(tmp_path):
    now = BASE + 300 * HOUR
    ex = FakeExchange(now, candles_by_key=_full_candle_set(now))
    runner1 = AdaptiveRunner(str(tmp_path), market_data=MarketData(exchange=ex))
    runner1.bootstrap()
    runner1.short_portfolio.sell_to_open(
        "BTC/USDT", best_bid=100.0, notional_usdt=200.0, fee_rate=0.001, slippage=0.0002,
        ts_ms=now, strategy="trend_momentum", regime="TREND_DOWN", confidence=0.8,
        initial_stop_pct=5.0, risk_amount_usdt=10.0)
    from adaptive.trailing import new_short_trail_state
    runner1.short_portfolio.positions["BTC/USDT"].trail_state = new_short_trail_state(100.0, 5.0)
    runner1._save_short_state()

    runner2 = AdaptiveRunner(str(tmp_path), market_data=MarketData(exchange=ex))
    runner2.bootstrap()
    assert "BTC/USDT" in runner2.short_portfolio.positions
    pos = runner2.short_portfolio.positions["BTC/USDT"]
    assert pos.strategy == "trend_momentum"
    assert pos.initial_stop_pct == 5.0
    assert pos.leverage == pytest.approx(runner2.short_portfolio.leverage)
    assert pos.trail_state["sl_price"] == pytest.approx(105.0)


def test_gap_does_not_block_short_stop_management(tmp_path):
    """Proves the gap flag never blocks _manage_short_position's own
    stop/liquidation check (that method runs unconditionally, ahead of
    the reconciliation_failed skip used for NEW entries). Does NOT
    exercise _exit_short_signal_management (a bullish signal covering a
    short) -- that path sits inside the per-symbol loop behind
    `self.portfolio.held_qty(sym) > 0 or sym in self.reconciliation_failed:
    continue`, so it IS gated by a gap on this symbol, same as the
    long-book's pre-existing (and out-of-scope-to-change-here) exit_long
    signal path."""
    now = BASE + 300 * HOUR
    candles = _full_candle_set(now)
    ex = FakeExchange(now, candles_by_key=candles)
    runner = AdaptiveRunner(str(tmp_path), market_data=MarketData(exchange=ex))
    runner.bootstrap()

    runner.short_portfolio.sell_to_open(
        "BTC/USDT", best_bid=100.0, notional_usdt=200.0, fee_rate=0.001, slippage=0.0002,
        ts_ms=now, strategy="trend_momentum", regime="TREND_DOWN", confidence=0.8,
        initial_stop_pct=5.0, risk_amount_usdt=10.0)
    from adaptive.trailing import new_short_trail_state
    runner.short_portfolio.positions["BTC/USDT"].trail_state = new_short_trail_state(100.0, 5.0)  # stop=105

    runner.reconciliation_failed.add("BTC/USDT")

    new_now = now + M5
    ex._now_ms = new_now
    ts0 = now - (now % M5)
    ex.candles_by_key[("BTC/USDT", "5m")] = list(ex.candles_by_key[("BTC/USDT", "5m")]) + [
        [ts0, 100.0, 110.0, 99.5, 108.0, 500.0]]  # high=110 breaches stop=105

    runner.run_once_cycle()
    assert "BTC/USDT" not in runner.short_portfolio.positions
    assert runner.short_portfolio.n_trades == 1


def test_snapshot_short_book_never_summed_into_long_equity(tmp_path):
    now = BASE + 300 * HOUR
    ex = FakeExchange(now, candles_by_key=_full_candle_set(now))
    runner = AdaptiveRunner(str(tmp_path), market_data=MarketData(exchange=ex))
    runner.bootstrap()
    prices = {s: 100.0 for s in SYMBOLS}
    snap = runner.snapshot(prices)
    assert "short_book" in snap
    assert snap["equity"] == pytest.approx(runner.portfolio.start_capital)  # long-only, unaffected
    assert snap["short_book"]["equity"] == pytest.approx(runner.short_portfolio.start_capital)
    assert snap["short_book"]["leverage"] == runner.short_portfolio.leverage
    for key in ("cash", "equity", "return_pct", "positions", "risk_state", "daily_pnl_usdt"):
        assert key in snap["short_book"]


def test_dashboard_json_includes_short_book_section(tmp_path):
    now = BASE + 300 * HOUR
    ex = FakeExchange(now, candles_by_key=_full_candle_set(now))
    runner = AdaptiveRunner(str(tmp_path), market_data=MarketData(exchange=ex))
    runner.bootstrap()
    runner.run_once_cycle()
    dash = json.load(open(os.path.join(str(tmp_path), "dashboard.json")))
    assert "short_book" in dash
    assert "positions" in dash["short_book"]


def test_short_trades_csv_created_on_bootstrap(tmp_path):
    now = BASE + 300 * HOUR
    ex = FakeExchange(now, candles_by_key=_full_candle_set(now))
    runner = AdaptiveRunner(str(tmp_path), market_data=MarketData(exchange=ex))
    runner.bootstrap()
    assert os.path.exists(runner.short_trades_csv_path)
    with open(runner.short_trades_csv_path) as f:
        header = f.readline().strip().split(",")
    assert "leverage" in header
    assert "borrow_cost" in header


def test_many_cycles_with_no_signal_never_crash_short_wiring(tmp_path):
    """Broad smoke test: run many cycles on flat, unchanging data and
    confirm the short-side wiring never raises and never fabricates a
    position out of nothing."""
    now = BASE + 300 * HOUR
    ex = FakeExchange(now, candles_by_key=_full_candle_set(now))
    runner = AdaptiveRunner(str(tmp_path), market_data=MarketData(exchange=ex))
    runner.bootstrap()
    for _ in range(10):
        runner.run_once_cycle()
    assert len(runner.short_portfolio.positions) == 0
    assert len(runner.portfolio.positions) == 0
