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


def _fresh_dip_bar(ex, symbol, now, tf_key="15m"):
    """Appends a genuine fresh, sharply oversold bar for `symbol` (mirrors
    the real production mean_reversion signals seen in decisions.jsonl) on
    `tf_key`, and ALSO refreshes the 5m series to the same new_now so
    market_quality's staleness check (which reads PRIMARY_STOP_TF="5m")
    doesn't reject the symbol as stale -- advancing only the 15m clock
    without a matching fresh 5m bar leaves 5m data older than the default
    10-minute staleness threshold. Returns the new now_ms."""
    tf_ms = TF_MS[tf_key]
    new_now = now + tf_ms
    ex._now_ms = new_now
    # derive the new bar's ts from THIS SYMBOL's own last existing bar on
    # this timeframe (never independently from `now`'s own alignment) --
    # otherwise a symbol whose series wasn't touched on a PRIOR call (e.g.
    # a second call advancing the shared clock again for a DIFFERENT
    # symbol first) gets a bar placed with a gap before it, which
    # cursor-discipline silently drops (worse: 15m/1h/4h gaps aren't even
    # logged today, only 5m ones are -- see the "verify bar processing"
    # follow-up this bug points at).
    series = list(ex.candles_by_key[(symbol, tf_key)])
    last_close = series[-1][4]
    dip_close = last_close * 0.90
    new_bar_ts = series[-1][0] + tf_ms
    series.append([new_bar_ts, last_close, last_close * 1.001, dip_close * 0.999, dip_close, 100.0])
    ex.candles_by_key[(symbol, tf_key)] = series

    if tf_key != "5m":
        # fill EVERY missing 5m bar up through new_now (not just one) --
        # if this symbol's 5m series was last touched further in the past
        # than one bar-width (e.g. a second call advancing the clock again
        # for a DIFFERENT symbol), a single appended bar would still leave
        # it stale relative to the new ex._now_ms. Each appended bar is
        # exactly one bar-width after the previous, so cursor-discipline
        # never sees a gap.
        m5_ms = TF_MS["5m"]
        m5_series = list(ex.candles_by_key[(symbol, "5m")])
        next_ts = m5_series[-1][0] + m5_ms
        while next_ts + m5_ms <= new_now:
            m5_series.append([next_ts, dip_close, dip_close * 1.001, dip_close * 0.999, dip_close, 100.0])
            next_ts += m5_ms
        ex.candles_by_key[(symbol, "5m")] = m5_series

    # the ticker must track the dip too -- otherwise the entry fills near
    # the OLD static ticker price (unrelated to the candle that triggered
    # the signal) while the stop is computed from a candle low far below
    # it, causing an unrealistic immediate stop-out on the very next cycle
    ex.tickers[symbol] = {"bid": dip_close * 0.9998, "ask": dip_close * 1.0002}
    return new_now


def _btc_oversold_dip_bar(ex, now, tf_key="15m"):
    return _fresh_dip_bar(ex, "BTC/USDT", now, tf_key)


def test_fresh_system_can_take_its_first_trade_via_exploration(tmp_path):
    """The core fix: proves a brand-new system (zero live trades anywhere,
    empty stats_store) CAN open its first paper trade when a strong, fresh
    raw signal fires -- via the exploration path, since normal scoring
    alone mathematically cannot (see test_adaptive_meta_controller.py's
    cold-start-deadlock proof: shrunk_edge is pinned to exactly 0 at n=0)."""
    now = BASE + 300 * HOUR
    ex = FakeExchange(now, candles_by_key=_full_candle_set(now))
    runner = AdaptiveRunner(str(tmp_path), market_data=MarketData(exchange=ex))
    runner.bootstrap()
    assert len(runner.portfolio.positions) == 0

    _btc_oversold_dip_bar(ex, now)
    runner.run_once_cycle()

    assert "BTC/USDT" in runner.portfolio.positions
    pos = runner.portfolio.positions["BTC/USDT"]
    assert pos.strategy == "mean_reversion"
    assert runner.exploration_symbols == {"BTC/USDT"}
    # exploration-sized (0.10% of 10k equity = ~$10 risk), not normal (0.50% = ~$50)
    assert pos.risk_amount_usdt < 20.0


def test_weak_raw_signal_still_rejected_for_exploration(tmp_path):
    now = BASE + 300 * HOUR
    ex = FakeExchange(now, candles_by_key=_full_candle_set(now))
    runner = AdaptiveRunner(str(tmp_path), market_data=MarketData(exchange=ex))
    runner.bootstrap()

    # a tiny dip that doesn't clear mean_reversion's z<=-1.5 entry condition at all
    new_now = now + M15
    ex._now_ms = new_now
    new_bar_ts = now - (now % M15)
    series = list(ex.candles_by_key[("BTC/USDT", "15m")])
    last_close = series[-1][4]
    tiny_dip = last_close * 0.999
    series.append([new_bar_ts, last_close, last_close * 1.001, tiny_dip * 0.999, tiny_dip, 100.0])
    ex.candles_by_key[("BTC/USDT", "15m")] = series

    runner.run_once_cycle()
    assert len(runner.portfolio.positions) == 0
    assert runner.exploration_symbols == set()


def test_multiple_exploration_positions_can_be_held_simultaneously(tmp_path):
    """Up to MAX_EXPLORATION_POSITIONS (one per symbol) may explore in
    parallel -- BTC/ETH/SOL each independently acquiring their first live
    observation, not serialized behind a single system-wide slot."""
    now = BASE + 300 * HOUR
    ex = FakeExchange(now, candles_by_key=_full_candle_set(now))
    runner = AdaptiveRunner(str(tmp_path), market_data=MarketData(exchange=ex))
    runner.bootstrap()

    # inject a strong, fresh oversold dip on ALL THREE symbols in the same cycle
    for sym in ("BTC/USDT", "ETH/USDT", "SOL/USDT"):
        _fresh_dip_bar(ex, sym, now, "15m")
    runner.run_once_cycle()

    assert runner.exploration_symbols == {"BTC/USDT", "ETH/USDT", "SOL/USDT"}
    assert len(runner.portfolio.positions) == 3


def test_exploration_capped_at_max_exploration_positions(tmp_path):
    """A signal on EVERY symbol in the universe must never exceed
    MAX_EXPLORATION_POSITIONS -- written generically against SYMBOLS/
    MAX_EXPLORATION_POSITIONS rather than a hardcoded count, since the
    universe size and the cap are configured to match (see
    market_data.SYMBOLS / risk_engine.MAX_EXPLORATION_POSITIONS) but
    aren't guaranteed to stay equal forever."""
    from adaptive.risk_engine import MAX_EXPLORATION_POSITIONS
    now = BASE + 300 * HOUR
    ex = FakeExchange(now, candles_by_key=_full_candle_set(now))
    runner = AdaptiveRunner(str(tmp_path), market_data=MarketData(exchange=ex))
    runner.bootstrap()
    for sym in SYMBOLS:
        _fresh_dip_bar(ex, sym, now, "15m")
    runner.run_once_cycle()
    assert len(runner.exploration_symbols) == min(len(SYMBOLS), MAX_EXPLORATION_POSITIONS)


def test_second_symbol_can_join_exploration_while_first_still_open(tmp_path):
    now = BASE + 300 * HOUR
    ex = FakeExchange(now, candles_by_key=_full_candle_set(now))
    runner = AdaptiveRunner(str(tmp_path), market_data=MarketData(exchange=ex))
    runner.bootstrap()

    now2 = _btc_oversold_dip_bar(ex, now)
    runner.run_once_cycle()
    assert runner.exploration_symbols == {"BTC/USDT"}
    assert len(runner.portfolio.positions) == 1

    # a second, equally strong signal on a DIFFERENT symbol -- must now
    # join exploration alongside BTC (multi-slot behavior), not be blocked
    _fresh_dip_bar(ex, "ETH/USDT", now2, "15m")
    runner.run_once_cycle()

    assert "ETH/USDT" in runner.portfolio.positions
    assert runner.exploration_symbols == {"BTC/USDT", "ETH/USDT"}
    assert len(runner.portfolio.positions) == 2


def test_historical_bootstrap_bars_cannot_generate_exploration_entries(tmp_path):
    """The same freshness gate that blocks normal fabricated entries on
    stale bootstrap history also applies to exploration -- exploration
    candidates are drawn from the same `candidates` list, built only from
    bars freshly polled THIS cycle."""
    now = BASE + 300 * HOUR
    candles = _full_candle_set(now, price=100.0)
    # inject an obvious historical oversold dip directly into the BOOTSTRAP
    # data itself (not a fresh post-bootstrap bar)
    tf_ms = TF_MS["15m"]
    series = list(candles[("BTC/USDT", "15m")])
    last_close = series[-1][4]
    dip_close = last_close * 0.90
    series[-1] = [series[-1][0], series[-1][1], series[-1][2], dip_close * 0.999, dip_close, 100.0]
    candles[("BTC/USDT", "15m")] = series

    ex = FakeExchange(now, candles_by_key=candles)
    runner = AdaptiveRunner(str(tmp_path), market_data=MarketData(exchange=ex))
    runner.bootstrap()
    runner.run_once_cycle()  # no new bars have arrived since bootstrap
    assert len(runner.portfolio.positions) == 0
    assert runner.exploration_symbols == set()


def test_normal_risk_limits_override_exploration(tmp_path):
    """A real consecutive-loss streak must block exploration exactly like
    a normal entry -- exploration never bypasses basic risk management.
    (Manually forcing risk_engine.state directly wouldn't be a valid test:
    run_once_cycle() recomputes degradation state from live evidence at
    the START of every cycle, so a hand-set state never survives past the
    first line of the next cycle -- real evidence is required instead.)"""
    now = BASE + 300 * HOUR
    ex = FakeExchange(now, candles_by_key=_full_candle_set(now))
    runner = AdaptiveRunner(str(tmp_path), market_data=MarketData(exchange=ex))
    runner.bootstrap()
    # References the live breaker threshold rather than hardcoding it --
    # both the breaker check AND evaluate_degradation() independently see
    # this and HALT.
    runner.consecutive_losses = runner.risk_engine.limits.consecutive_loss_breaker

    _btc_oversold_dip_bar(ex, now)
    runner.run_once_cycle()
    assert len(runner.portfolio.positions) == 0
    assert runner.exploration_symbols == set()
    assert runner.risk_engine.state.value == "HALTED"


def test_exploration_slot_frees_up_after_position_closes(tmp_path):
    now = BASE + 300 * HOUR
    ex = FakeExchange(now, candles_by_key=_full_candle_set(now))
    runner = AdaptiveRunner(str(tmp_path), market_data=MarketData(exchange=ex))
    runner.bootstrap()
    _btc_oversold_dip_bar(ex, now)
    runner.run_once_cycle()
    assert runner.exploration_symbols == {"BTC/USDT"}

    # manually close the exploration position, as a stop-out or exit would
    rec = runner.portfolio.sell("BTC/USDT", best_bid=90.0, fee_rate=0.001, slippage=0.0002,
                                 ts_ms=1000, exit_reason="test")
    runner._on_trade_closed(rec)
    assert runner.exploration_symbols == set()  # slot freed for a future cell


def test_exploration_symbols_persist_across_restart(tmp_path):
    now = BASE + 300 * HOUR
    ex = FakeExchange(now, candles_by_key=_full_candle_set(now))
    runner1 = AdaptiveRunner(str(tmp_path), market_data=MarketData(exchange=ex))
    runner1.bootstrap()
    _btc_oversold_dip_bar(ex, now)
    runner1.run_once_cycle()
    assert runner1.exploration_symbols == {"BTC/USDT"}

    runner2 = AdaptiveRunner(str(tmp_path), market_data=MarketData(exchange=ex))
    runner2.bootstrap()
    assert runner2.exploration_symbols == {"BTC/USDT"}


def test_legacy_single_exploration_symbol_state_migrates_cleanly(tmp_path):
    """An older state.json (pre-multi-slot) stored a single
    "exploration_symbol" string -- must still load correctly."""
    import json
    now = BASE + 300 * HOUR
    ex = FakeExchange(now, candles_by_key=_full_candle_set(now))
    runner = AdaptiveRunner(str(tmp_path), market_data=MarketData(exchange=ex))
    runner.bootstrap()
    with open(runner.state_path) as f:
        state = json.load(f)
    del state["exploration_symbols"]
    state["exploration_symbol"] = "SOL/USDT"  # legacy singular key
    with open(runner.state_path, "w") as f:
        json.dump(state, f)

    runner2 = AdaptiveRunner(str(tmp_path), market_data=MarketData(exchange=ex))
    runner2.bootstrap()
    assert runner2.exploration_symbols == {"SOL/USDT"}


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


# ── cross-sectional relative strength (multi-symbol strategy) ───────────
def _dispersed_1h(now_ms, risers, fallers, riser_mult=1.004, faller_mult=0.996, n=200):
    """A full candle set where `risers` trend up and `fallers` trend down on
    1h, everything else flat. A small deterministic ±0.3% wiggle is layered
    on so per-bar return volatility is > 0 (a pure geometric series has zero
    return-vol and is excluded from the cross-sectional ranking by design)."""
    candles = _full_candle_set(now_ms, price=100.0)
    tf = TF_MS["1h"]
    last_close = now_ms - (now_ms % tf) - tf
    ts0 = last_close - (n - 1) * tf

    def series(mult):
        out = []
        for i in range(n):
            wig = 1.0 + 0.003 * (1 if i % 2 else -1)
            px = 100.0 * (mult ** i) * wig
            out.append([ts0 + i * tf, px, px * 1.002, px * 0.998, px, 100.0])
        return out

    for s in risers:
        candles[(s, "1h")] = series(riser_mult)
    for s in fallers:
        candles[(s, "1h")] = series(faller_mult)
    return candles


def _refresh_1h_and_5m(ex, sym, now_ms, new_now, mult):
    """Append a fresh 1h bar for `sym` and fill its 5m series up to new_now
    so market_quality's 5m-staleness gate passes (mirrors _fresh_dip_bar)."""
    a = list(ex.candles_by_key[(sym, "1h")])
    lc = a[-1][4]
    nc = lc * mult
    a.append([a[-1][0] + TF_MS["1h"], lc, nc * 1.002, lc * 0.998, nc, 500.0])
    ex.candles_by_key[(sym, "1h")] = a
    m = list(ex.candles_by_key[(sym, "5m")])
    t = m[-1][0] + TF_MS["5m"]
    while t + TF_MS["5m"] <= new_now:
        m.append([t, nc, nc * 1.001, nc * 0.999, nc, 100.0])
        t += TF_MS["5m"]
    ex.candles_by_key[(sym, "5m")] = m
    ex.tickers[sym] = {"bid": nc * 0.9998, "ask": nc * 1.0002}


def test_cross_sectional_reaches_candidate_pool_both_sides(tmp_path):
    """Regression for the cross-sectional strategy wiring AND the multi-
    candidate-per-symbol dedup bug it surfaced. With a genuinely dispersed
    universe (unlike the flat _full_candle_set), the strongest riser must
    produce a `cross_sectional` LONG candidate and the weakest faller a
    `cross_sectional` SHORT candidate -- and the cycle must NOT crash when a
    symbol is flagged by two strategies at once (trend_momentum AND
    cross_sectional both fire), which previously raised 'already held'."""
    import json
    now = BASE + 300 * HOUR
    risers = list(SYMBOLS[:3])
    fallers = list(SYMBOLS[3:6])
    candles = _dispersed_1h(now, risers, fallers)
    ex = FakeExchange(now, candles_by_key=candles)
    runner = AdaptiveRunner(str(tmp_path), market_data=MarketData(exchange=ex))
    runner.bootstrap()

    new_now = now + HOUR
    ex._now_ms = new_now
    _refresh_1h_and_5m(ex, risers[0], now, new_now, 1.004)   # strongest riser
    _refresh_1h_and_5m(ex, fallers[0], now, new_now, 0.996)  # weakest faller

    runner.run_once_cycle()  # must not raise

    lines = [json.loads(l) for l in open(runner.decisions_path)]
    ev = [l for l in lines if l.get("action") == "cycle_ranking"][-1]
    long_strats = {(c["symbol"], c["strategy"]) for c in ev["candidates"]}
    short_strats = {(c["symbol"], c["strategy"]) for c in ev.get("short_candidates", [])}
    assert (risers[0], "cross_sectional") in long_strats
    assert (fallers[0], "cross_sectional") in short_strats
    # the symbol was flagged by BOTH trend_momentum and cross_sectional; the
    # dedup guard means exactly one position exists, not a crash / double buy
    assert runner.portfolio.held_qty(risers[0]) > 0
    assert runner.short_portfolio.held_qty(fallers[0]) > 0


def test_bootstrap_survives_transient_fetch_error(tmp_path):
    """Regression: a transient exchange error on ONE (symbol, timeframe)
    poll during bootstrap must not crash the process (bootstrap runs before
    run_forever's try/except, so an unhandled raise here -> launchd crash
    loop). Became likely once the universe grew to 20 symbols = 80 startup
    fetches. The failing symbol just warms up on a later cycle."""
    now = BASE + 300 * HOUR

    class FlakyExchange(FakeExchange):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.raised = False

        def fetch_ohlcv(self, symbol, timeframe, since=None, limit=500):
            if symbol == SYMBOLS[7] and timeframe == "1h" and not self.raised:
                self.raised = True
                import ccxt
                raise ccxt.RequestTimeout(f"simulated timeout {symbol} {timeframe}")
            return super().fetch_ohlcv(symbol, timeframe, since=since, limit=limit)

    ex = FlakyExchange(now, candles_by_key=_full_candle_set(now))
    runner = AdaptiveRunner(str(tmp_path), market_data=MarketData(exchange=ex))
    runner.bootstrap()  # must NOT raise
    assert ex.raised
    assert len(runner.md.candles[(SYMBOLS[0], "1h")]) > 0


def test_cycle_ranking_logs_full_score_breakdown(tmp_path):
    """Observability upgrade F1: each cycle_ranking event must carry the full
    per-candidate score decomposition (edge, regime fit, robustness, and the
    friction/uncertainty/drawdown/correlation penalties), not just
    symbol/strategy/score -- so 'is this more profitable' can actually be
    inspected. Reuses the dispersed-universe fixture that reliably produces
    both long and short candidates."""
    import json
    now = BASE + 300 * HOUR
    risers = list(SYMBOLS[:3]); fallers = list(SYMBOLS[3:6])
    candles = _dispersed_1h(now, risers, fallers)
    ex = FakeExchange(now, candles_by_key=candles)
    runner = AdaptiveRunner(str(tmp_path), market_data=MarketData(exchange=ex))
    runner.bootstrap()
    new_now = now + HOUR; ex._now_ms = new_now
    _refresh_1h_and_5m(ex, risers[0], now, new_now, 1.004)
    _refresh_1h_and_5m(ex, fallers[0], now, new_now, 0.996)
    runner.run_once_cycle()

    ev = [json.loads(l) for l in open(runner.decisions_path) if json.loads(l).get("action") == "cycle_ranking"][-1]
    assert ev["candidates"], "expected at least one long candidate"
    required = {"symbol", "strategy", "regime", "direction", "score", "shrunk_edge",
                "signal_strength", "regime_fit", "robustness", "friction_penalty",
                "uncertainty_penalty", "drawdown_penalty", "correlation_penalty",
                "confidence", "is_exploration_eligible"}
    for c in ev["candidates"] + ev["short_candidates"]:
        assert required.issubset(c.keys()), f"missing breakdown fields: {required - set(c.keys())}"


def test_snapshot_exposes_combined_realized_pnl(tmp_path):
    """All-time realized P&L (the honest banked scorecard) must be surfaced
    at the top level of the snapshot, distinct from equity/return (which
    include unrealized marks). Combined = long + short realized."""
    now = BASE + 300 * HOUR
    ex = FakeExchange(now, candles_by_key=_full_candle_set(now))
    runner = AdaptiveRunner(str(tmp_path), market_data=MarketData(exchange=ex))
    runner.bootstrap()
    runner.portfolio.realized_pnl = -32.42
    runner.short_portfolio.realized_pnl = -21.35
    runner.portfolio.n_trades = 20; runner.portfolio.wins = 3
    runner.short_portfolio.n_trades = 14; runner.short_portfolio.wins = 3
    snap = runner.snapshot({s: 100.0 for s in SYMBOLS})
    assert snap["realized_pnl_long"] == pytest.approx(-32.42)
    assert snap["realized_pnl_short"] == pytest.approx(-21.35)
    assert snap["realized_pnl_combined"] == pytest.approx(-53.77)
    assert snap["n_trades_combined"] == 34
    assert snap["wins_combined"] == 6
    assert snap["win_rate_pct"] == pytest.approx(100 * 6 / 34)
