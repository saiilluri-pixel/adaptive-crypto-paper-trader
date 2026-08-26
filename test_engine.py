"""
Deterministic tests for engine.py: fee/PnL accounting (item 3), stop-fill
price correctness (item 2), risk-based sizing (item 4), and risk guards
(item 5) of the merge. No network calls.
"""
import pytest

import config
from engine import PaperEngine


def _fresh_engine(**overrides):
    eng = PaperEngine(log_fn=lambda m: None, log_trades=False)
    for k, v in overrides.items():
        setattr(eng, k, v)
    return eng


# ── Fee / PnL reconciliation (item 3) ──────────────────────────────

def test_realized_reconciles_with_equity_after_one_round_trip():
    eng = _fresh_engine()
    eng.enter("long", 100.0, "BUY", 1)
    assert eng.pos is not None
    eng.close(110.0, "test exit")
    assert eng.pos is None
    assert eng.equity == pytest.approx(eng.cash)
    assert eng.realized == pytest.approx(eng.equity - config.START_CAPITAL, abs=1e-6)


def test_csv_fees_and_net_pnl_are_consistent():
    eng = _fresh_engine()
    eng.enter("long", 100.0, "BUY", 1)
    entry_fee = eng.pos.entry_fee
    entry_price = eng.pos.entry_price
    qty = eng.pos.qty
    eng.close(110.0, "test exit")
    fill = eng._fill(110.0, "long", opening=False)
    gross = (fill - entry_price) * qty
    exit_fee = fill * qty * config.TAKER_FEE
    expected_net = gross - entry_fee - exit_fee   # what the "fees" column implies
    assert eng.realized == pytest.approx(expected_net, abs=1e-6)


def test_win_loss_uses_true_after_fee_net():
    # gross is barely positive but round-trip fees flip it net-negative
    eng = _fresh_engine()
    eng.enter("long", 100.0, "BUY", 1)
    tiny_up = eng.pos.entry_price * 1.00001
    eng.close(tiny_up, "test exit")
    assert eng.wins == 0
    assert eng.realized < 0


def test_multiple_round_trips_realized_still_reconciles():
    eng = _fresh_engine()
    for entry, exit_ in [(100, 105), (105, 95), (95, 130)]:
        eng.enter("long", entry, "BUY", 1)
        eng.close(exit_, "test exit")
    assert eng.equity == pytest.approx(eng.cash)
    assert eng.realized == pytest.approx(eng.equity - config.START_CAPITAL, abs=1e-3)


def test_short_round_trip_realized_reconciles():
    eng = _fresh_engine()
    eng.enter("short", 100.0, "SELL", -1)
    eng.close(80.0, "test exit")
    assert eng.equity == pytest.approx(eng.cash)
    assert eng.realized == pytest.approx(eng.equity - config.START_CAPITAL, abs=1e-6)


# ── Stop-fill correctness: check_stop_bar (OHLC, historical/restart/live
#    bar-boundary reconciliation) -- items 1, 2, 3, 4, 5, 6 ───────────────

def test_check_stop_bar_long_ordinary_touch_uses_stop_not_low():
    """Item 1: LONG ordinary intrabar touch -> reference = stop, not low."""
    eng = _fresh_engine()
    eng.enter("long", 100.0, "BUY", -1)
    entry_price, qty, entry_fee = eng.pos.entry_price, eng.pos.qty, eng.pos.entry_fee
    stop = eng.pos.tsl.sl_price
    open_, high, low = stop + 5, stop + 7, stop - 5   # open>stop, low<=stop

    eng.check_stop_bar(open_, high, low)

    assert eng.pos is None
    expected_fill = eng._fill(stop, "long", opening=False)
    expected_gross = (expected_fill - entry_price) * qty
    expected_net = expected_gross - entry_fee - (expected_fill * qty * config.TAKER_FEE)
    assert eng.realized == pytest.approx(expected_net, abs=1e-6)


def test_check_stop_bar_long_gap_uses_open_not_low():
    """Item 2: LONG gap -> reference = open, not low."""
    eng = _fresh_engine()
    eng.enter("long", 100.0, "BUY", -1)
    entry_price, qty, entry_fee = eng.pos.entry_price, eng.pos.qty, eng.pos.entry_fee
    stop = eng.pos.tsl.sl_price
    open_, high, low = stop - 5, stop - 3, stop - 10   # open already <= stop

    eng.check_stop_bar(open_, high, low)

    assert eng.pos is None
    expected_fill = eng._fill(open_, "long", opening=False)
    expected_gross = (expected_fill - entry_price) * qty
    expected_net = expected_gross - entry_fee - (expected_fill * qty * config.TAKER_FEE)
    assert eng.realized == pytest.approx(expected_net, abs=1e-6)


def test_check_stop_bar_short_ordinary_touch_uses_stop_not_high():
    """Item 3: SHORT ordinary touch mirror -> reference = stop, not high."""
    eng = _fresh_engine()
    eng.enter("short", 100.0, "SELL", 1)
    entry_price, qty, entry_fee = eng.pos.entry_price, eng.pos.qty, eng.pos.entry_fee
    stop = eng.pos.tsl.sl_price
    open_, high, low = stop - 5, stop + 5, stop - 7   # open<stop, high>=stop

    eng.check_stop_bar(open_, high, low)

    assert eng.pos is None
    expected_fill = eng._fill(stop, "short", opening=False)
    expected_gross = (entry_price - expected_fill) * qty
    expected_net = expected_gross - entry_fee - (expected_fill * qty * config.TAKER_FEE)
    assert eng.realized == pytest.approx(expected_net, abs=1e-6)


def test_check_stop_bar_short_gap_uses_open_not_high():
    """Item 4: SHORT gap mirror -> reference = open, not high."""
    eng = _fresh_engine()
    eng.enter("short", 100.0, "SELL", 1)
    entry_price, qty, entry_fee = eng.pos.entry_price, eng.pos.qty, eng.pos.entry_fee
    stop = eng.pos.tsl.sl_price
    open_, high, low = stop + 5, stop + 10, stop + 3   # open already >= stop

    eng.check_stop_bar(open_, high, low)

    assert eng.pos is None
    expected_fill = eng._fill(open_, "short", opening=False)
    expected_gross = (entry_price - expected_fill) * qty
    expected_net = expected_gross - entry_fee - (expected_fill * qty * config.TAKER_FEE)
    assert eng.realized == pytest.approx(expected_net, abs=1e-6)


def test_check_stop_bar_no_touch_stays_open():
    """Item 5: candle that never touches the stop -> position stays open."""
    eng = _fresh_engine()
    eng.enter("long", 100.0, "BUY", -1)
    stop = eng.pos.tsl.sl_price
    eng.check_stop_bar(stop + 10, stop + 12, stop + 5)   # entirely above stop
    assert eng.pos is not None


def test_check_stop_bar_does_not_itself_ratchet_the_stop():
    """Purity check supporting item 6: check_stop_bar must be a pure read of
    the CURRENT stop -- it never calls tsl.update() itself."""
    eng = _fresh_engine()
    eng.enter("long", 100.0, "BUY", -1)
    stop_before = eng.pos.tsl.sl_price
    entry_before = eng.pos.tsl.updated_entry
    sl_pct_before = eng.pos.tsl.sl_pct
    eng.check_stop_bar(stop_before + 10, stop_before + 12, stop_before + 5)   # no hit
    assert eng.pos is not None
    assert eng.pos.tsl.sl_price == stop_before
    assert eng.pos.tsl.updated_entry == entry_before
    assert eng.pos.tsl.sl_pct == sl_pct_before


def test_check_stop_bar_uses_stop_as_it_stood_before_this_bar():
    """Item 6: stop must be checked BEFORE the same bar's ratchet update.
    A bar whose low touches the OLD stop must close using that OLD stop as
    reference, even though this same bar's close is favorable enough that
    ratcheting FIRST would have raised the stop -- proving the check is not
    contaminated by this bar's own ratchet (which only runs afterward, in
    manage_on_bar, and only if the position is still open)."""
    eng = _fresh_engine()
    eng.enter("long", 100.0, "BUY", -1)
    old_stop = eng.pos.tsl.sl_price
    entry_price, qty, entry_fee = eng.pos.entry_price, eng.pos.qty, eng.pos.entry_fee
    open_, high, low = old_stop + 10, old_stop + 12, old_stop - 1   # wick below old stop, closes favorably

    eng.check_stop_bar(open_, high, low)   # correct order: called before manage_on_bar

    assert eng.pos is None
    expected_fill = eng._fill(old_stop, "long", opening=False)
    expected_gross = (expected_fill - entry_price) * qty
    expected_net = expected_gross - entry_fee - (expected_fill * qty * config.TAKER_FEE)
    assert eng.realized == pytest.approx(expected_net, abs=1e-6)


# ── check_stop_intrabar (live, item 10) ─────────────────────────────

def test_check_stop_intrabar_fills_at_observed_tick_not_theoretical_stop():
    eng = _fresh_engine()
    eng.enter("long", 100.0, "BUY", 1)
    entry_price = eng.pos.entry_price
    qty = eng.pos.qty
    entry_fee = eng.pos.entry_fee
    stop_before = eng.pos.tsl.sl_price
    gapped_tick = stop_before - 5.0

    eng.check_stop_intrabar(gapped_tick)

    assert eng.pos is None
    expected_fill = eng._fill(gapped_tick, "long", opening=False)
    expected_gross = (expected_fill - entry_price) * qty
    expected_net = expected_gross - entry_fee - (expected_fill * qty * config.TAKER_FEE)
    assert eng.realized == pytest.approx(expected_net, abs=1e-6)


# ── backtest/autotune call order (item 11) ──────────────────────────

def test_backtest_and_autotune_call_check_stop_bar_before_manage_on_bar():
    import inspect
    import backtest
    import autotune
    for mod, func in ((backtest, backtest.main), (autotune, autotune.run_replay)):
        src = inspect.getsource(func)
        stop_bar_idx = src.index("check_stop_bar(")     # actual call, not a comment mention
        manage_idx = src.index(".manage_on_bar(")
        assert stop_bar_idx < manage_idx, \
            f"{mod.__name__} must call check_stop_bar before manage_on_bar"


# ── Risk-based sizing (item 4) ───────────────────────────────────────

def test_risk_based_sizing_smaller_than_full_equity():
    eng = _fresh_engine()
    eng.enter("long", 100.0, "BUY", 1)
    notional = eng.pos.qty * eng.pos.entry_price
    assert notional < config.START_CAPITAL     # must be well under 100% equity
    expected_notional = min(
        config.START_CAPITAL * (config.RISK_PCT_PER_TRADE / 100.0) / (eng.init_stop / 100.0),
        config.START_CAPITAL * config.POSITION_PCT)
    assert notional == pytest.approx(expected_notional, rel=1e-6)


def test_zero_stop_distance_blocks_entry():
    eng = _fresh_engine()
    eng.init_stop = 0.0
    eng.enter("long", 100.0, "BUY", 1)
    assert eng.pos is None


def test_notional_capped_at_position_pct_even_with_high_risk_pct():
    eng = _fresh_engine()
    orig = config.RISK_PCT_PER_TRADE
    config.RISK_PCT_PER_TRADE = 1000.0     # deliberately absurd, to force the cap
    try:
        eng.enter("long", 100.0, "BUY", 1)
        notional = eng.pos.qty * eng.pos.entry_price
        assert notional <= config.START_CAPITAL * config.POSITION_PCT * 1.0001
    finally:
        config.RISK_PCT_PER_TRADE = orig


# ── Risk guards (item 5) ──────────────────────────────────────────────

def test_max_drawdown_blocks_new_entries():
    eng = _fresh_engine()
    eng.peak_equity = 10_000.0
    eng.equity = 10_000.0 * (1 - config.MAX_DRAWDOWN_PCT / 100.0 - 0.01)
    eng.enter("long", 100.0, "BUY", 1)
    assert eng.pos is None


def test_daily_loss_limit_blocks_new_entries():
    eng = _fresh_engine()
    eng._roll_day_if_needed()
    eng.daily_start_equity = 10_000.0
    eng.equity = 10_000.0 * (1 - config.DAILY_LOSS_LIMIT_PCT / 100.0 - 0.01)
    eng.peak_equity = eng.equity           # isolate from the drawdown guard
    eng.enter("long", 100.0, "BUY", 1)
    assert eng.pos is None


def test_consecutive_loss_breaker_blocks_new_entries():
    eng = _fresh_engine()
    eng.consecutive_losses = config.CONSECUTIVE_LOSS_LIMIT
    eng.enter("long", 100.0, "BUY", 1)
    assert eng.pos is None


def test_win_resets_consecutive_losses():
    eng = _fresh_engine()
    eng.consecutive_losses = config.CONSECUTIVE_LOSS_LIMIT - 1
    eng.enter("long", 100.0, "BUY", 1)
    assert eng.pos is not None
    eng.close(200.0, "big win")
    assert eng.consecutive_losses == 0


def test_losing_close_increments_consecutive_losses():
    eng = _fresh_engine()
    eng.enter("long", 100.0, "BUY", 1)
    eng.close(99.0, "loss")
    assert eng.consecutive_losses == 1


def test_risk_guards_never_block_managing_an_open_position():
    eng = _fresh_engine()
    eng.enter("long", 100.0, "BUY", -1)
    assert eng.pos is not None
    # trip every guard after entry
    eng.peak_equity = eng.equity * 10
    eng.daily_start_equity = eng.equity * 10
    eng.consecutive_losses = config.CONSECUTIVE_LOSS_LIMIT
    stop = eng.pos.tsl.sl_price
    eng.check_stop_intrabar(stop - 1.0)    # exit management must still work while guards are active
    assert eng.pos is None
