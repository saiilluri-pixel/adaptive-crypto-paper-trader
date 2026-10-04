"""Deterministic tests for adaptive/portfolio.py's Spot-legal invariants."""
import pytest

from adaptive.portfolio import Portfolio, SharedCash
from adaptive.short_portfolio import ShortPortfolio

FEE = 0.001
SLIP = 0.0002


def _entry_meta(strategy="trend", regime="TREND_UP", confidence=0.7, stop_price=95.0,
                 initial_stop_pct=5.0, risk_amount_usdt=50.0):
    return dict(strategy=strategy, regime=regime, confidence=confidence, stop_price=stop_price,
                initial_stop_pct=initial_stop_pct, risk_amount_usdt=risk_amount_usdt)


def test_shared_10k_account_starts_flat():
    p = Portfolio(10000.0)
    prices = {"BTC/USDT": 60000.0, "ETH/USDT": 3000.0, "SOL/USDT": 100.0}
    assert p.equity(prices) == 10000.0
    assert len(p.positions) == 0


def test_buy_uses_ask_then_adverse_slippage_on_top():
    p = Portfolio(10000.0)
    pos = p.buy("BTC/USDT", best_ask=100.0, notional_usdt=1000.0, fee_rate=FEE, slippage=SLIP,
                ts_ms=1000, **_entry_meta())
    expected_fill = 100.0 * (1 + SLIP)
    assert pos.entry_price == pytest.approx(expected_fill)
    assert pos.entry_ref_price == 100.0
    assert pos.qty == pytest.approx(1000.0 / expected_fill)


def test_sell_uses_bid_then_adverse_slippage_on_top():
    p = Portfolio(10000.0)
    p.buy("BTC/USDT", best_ask=100.0, notional_usdt=1000.0, fee_rate=FEE, slippage=SLIP,
          ts_ms=1000, **_entry_meta())
    rec = p.sell("BTC/USDT", best_bid=110.0, fee_rate=FEE, slippage=SLIP, ts_ms=2000, exit_reason="test")
    expected_fill = 110.0 * (1 - SLIP)
    assert rec["exit_price"] == pytest.approx(expected_fill)


def test_fees_correct_on_both_sides():
    p = Portfolio(10000.0)
    p.buy("BTC/USDT", best_ask=100.0, notional_usdt=1000.0, fee_rate=FEE, slippage=SLIP,
          ts_ms=1000, **_entry_meta())
    pos = p.positions["BTC/USDT"]
    assert pos.entry_fee == pytest.approx(1000.0 * FEE)
    rec = p.sell("BTC/USDT", best_bid=100.0, fee_rate=FEE, slippage=SLIP, ts_ms=2000, exit_reason="test")
    assert rec["exit_fee"] == pytest.approx(rec["gross_proceeds"] * FEE)


def test_cannot_short_without_inventory():
    p = Portfolio(10000.0)
    with pytest.raises(ValueError):
        p.sell("BTC/USDT", best_bid=100.0, fee_rate=FEE, slippage=SLIP, ts_ms=1000, exit_reason="x")


def test_sell_never_exceeds_owned_qty():
    """sell() always closes the FULL position -- by construction it can
    never sell more than owned. Verify qty sold == qty held, no partial
    over-sell path exists."""
    p = Portfolio(10000.0)
    p.buy("BTC/USDT", best_ask=100.0, notional_usdt=1000.0, fee_rate=FEE, slippage=SLIP,
          ts_ms=1000, **_entry_meta())
    held = p.held_qty("BTC/USDT")
    rec = p.sell("BTC/USDT", best_bid=100.0, fee_rate=FEE, slippage=SLIP, ts_ms=2000, exit_reason="x")
    assert rec["qty"] == pytest.approx(held)
    assert p.held_qty("BTC/USDT") == 0.0


def test_no_leverage_cash_cannot_go_negative_on_buy():
    p = Portfolio(1000.0)
    with pytest.raises(ValueError):
        p.buy("BTC/USDT", best_ask=100.0, notional_usdt=2000.0, fee_rate=FEE, slippage=SLIP,
              ts_ms=1000, **_entry_meta())
    assert p.cash == 1000.0  # unchanged -- the raise happens before any mutation


def test_at_most_one_position_per_symbol():
    p = Portfolio(10000.0)
    p.buy("BTC/USDT", best_ask=100.0, notional_usdt=1000.0, fee_rate=FEE, slippage=SLIP,
          ts_ms=1000, **_entry_meta())
    with pytest.raises(ValueError):
        p.buy("BTC/USDT", best_ask=100.0, notional_usdt=500.0, fee_rate=FEE, slippage=SLIP,
              ts_ms=1500, **_entry_meta())


def test_zero_one_two_three_simultaneous_positions():
    p = Portfolio(10000.0)
    prices = {"BTC/USDT": 100.0, "ETH/USDT": 100.0, "SOL/USDT": 100.0}
    assert len(p.positions) == 0
    p.buy("BTC/USDT", best_ask=100.0, notional_usdt=1000.0, fee_rate=FEE, slippage=SLIP,
          ts_ms=1000, **_entry_meta())
    assert len(p.positions) == 1
    p.buy("ETH/USDT", best_ask=100.0, notional_usdt=1000.0, fee_rate=FEE, slippage=SLIP,
          ts_ms=1000, **_entry_meta())
    assert len(p.positions) == 2
    p.buy("SOL/USDT", best_ask=100.0, notional_usdt=1000.0, fee_rate=FEE, slippage=SLIP,
          ts_ms=1000, **_entry_meta())
    assert len(p.positions) == 3
    assert p.equity(prices) < 10000.0  # fees/slippage make it strictly less, never more


def test_equity_marks_all_inventory_to_live_prices():
    p = Portfolio(10000.0)
    p.buy("BTC/USDT", best_ask=100.0, notional_usdt=1000.0, fee_rate=0.0, slippage=0.0,
          ts_ms=1000, **_entry_meta())
    prices_up = {"BTC/USDT": 200.0, "ETH/USDT": 1.0, "SOL/USDT": 1.0}
    eq = p.equity(prices_up)
    assert eq == pytest.approx(p.cash + p.held_qty("BTC/USDT") * 200.0)
    assert eq > 10000.0  # BTC doubled, no fees in this test


def test_realized_pnl_reconciles_with_cash_delta_across_round_trip():
    p = Portfolio(10000.0)
    cash_before = p.cash
    p.buy("BTC/USDT", best_ask=100.0, notional_usdt=1000.0, fee_rate=FEE, slippage=SLIP,
          ts_ms=1000, **_entry_meta())
    p.sell("BTC/USDT", best_bid=110.0, fee_rate=FEE, slippage=SLIP, ts_ms=2000, exit_reason="x")
    assert p.cash == pytest.approx(cash_before + p.realized_pnl)


def test_snapshot_reports_drawdown_and_allocation():
    p = Portfolio(10000.0)
    p.buy("BTC/USDT", best_ask=100.0, notional_usdt=4000.0, fee_rate=FEE, slippage=SLIP,
          ts_ms=1000, **_entry_meta())
    prices = {"BTC/USDT": 100.0, "ETH/USDT": 100.0, "SOL/USDT": 100.0}
    p.update_peak_equity(prices)
    snap = p.snapshot(prices)
    assert "drawdown_pct" in snap
    assert "crypto_allocation_pct" in snap
    assert snap["crypto_allocation_pct"] > 0
    assert "BTC/USDT" in snap["positions"]


# ── SharedCash: the money-pooling mechanism behind "no money split
# between short and long trades... first come first serve" ─────────────
def test_plain_float_construction_gets_its_own_private_pool():
    """Backward-compat: passing a float (as every pre-existing test and
    call site does) must behave exactly as before -- this Portfolio owns
    its own cash, invisible to anything else."""
    p1 = Portfolio(10000.0)
    p2 = Portfolio(10000.0)
    p1.buy("BTC/USDT", best_ask=100.0, notional_usdt=1000.0, fee_rate=FEE, slippage=SLIP,
           ts_ms=1000, **_entry_meta())
    assert p1.cash < 10000.0
    assert p2.cash == 10000.0  # untouched -- separate pools


def test_shared_pool_is_visible_to_both_portfolio_and_short_portfolio():
    pool = SharedCash(2000.0)
    long_book = Portfolio(pool)
    short_book = ShortPortfolio(pool, leverage=2.0)
    assert long_book.cash == 2000.0
    assert short_book.cash == 2000.0

    long_book.buy("BTC/USDT", best_ask=100.0, notional_usdt=500.0, fee_rate=FEE, slippage=SLIP,
                  ts_ms=1000, **_entry_meta())
    # the SHORT book must immediately see the reduced balance -- same pool
    assert short_book.cash == pytest.approx(long_book.cash)
    assert short_book.cash < 2000.0


def test_short_book_spending_is_visible_to_long_book():
    pool = SharedCash(2000.0)
    long_book = Portfolio(pool)
    short_book = ShortPortfolio(pool, leverage=2.0)
    short_book.sell_to_open("BTC/USDT", best_bid=100.0, notional_usdt=500.0, fee_rate=FEE,
                             slippage=SLIP, ts_ms=1000, strategy="trend_momentum", regime="TREND_DOWN",
                             confidence=0.8, initial_stop_pct=5.0, risk_amount_usdt=10.0)
    assert long_book.cash == pytest.approx(short_book.cash)
    assert long_book.cash < 2000.0


def test_first_come_first_serve_second_book_sees_reduced_room():
    """The essence of the redesign: whichever book spends FIRST in a
    cycle leaves genuinely less for the other -- no artificial per-book
    ceiling blocks one side from using capital the other isn't touching."""
    pool = SharedCash(1000.0)
    long_book = Portfolio(pool)
    short_book = ShortPortfolio(pool, leverage=2.0)
    # long spends almost everything
    long_book.buy("BTC/USDT", best_ask=100.0, notional_usdt=950.0, fee_rate=FEE, slippage=SLIP,
                  ts_ms=1000, **_entry_meta())
    remaining = short_book.cash
    assert remaining < 60.0  # only the leftover sliver is available now
    with pytest.raises(ValueError):
        short_book.sell_to_open("ETH/USDT", best_bid=100.0, notional_usdt=500.0, fee_rate=FEE,
                                 slippage=SLIP, ts_ms=1000, strategy="trend_momentum", regime="TREND_DOWN",
                                 confidence=0.8, initial_stop_pct=5.0, risk_amount_usdt=10.0)
