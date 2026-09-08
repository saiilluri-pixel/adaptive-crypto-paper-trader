"""Deterministic tests for adaptive/short_portfolio.py -- SIMULATED margin
short, paper only. Verifies Spot-separation invariants and margin mechanics."""
import pytest

from adaptive.short_portfolio import ShortPortfolio, MAINTENANCE_BUFFER

FEE = 0.001
SLIP = 0.0002


def _meta(strategy="mean_reversion", regime="RANGE", confidence=0.7, initial_stop_pct=5.0,
          risk_amount_usdt=50.0):
    return dict(strategy=strategy, regime=regime, confidence=confidence,
                initial_stop_pct=initial_stop_pct, risk_amount_usdt=risk_amount_usdt)


def test_starts_flat():
    p = ShortPortfolio(2708.40)
    assert p.cash == 2708.40
    assert len(p.positions) == 0


def test_sell_to_open_uses_bid_then_adverse_slippage_down():
    p = ShortPortfolio(10000.0)
    pos = p.sell_to_open("BTC/USDT", best_bid=100.0, notional_usdt=1000.0, fee_rate=FEE,
                          slippage=SLIP, ts_ms=1000, leverage=2.0, **_meta())
    expected_fill = 100.0 * (1 - SLIP)  # worse (lower) than the raw bid
    assert pos.entry_price == pytest.approx(expected_fill)
    assert pos.entry_ref_price == 100.0
    assert pos.qty == pytest.approx(1000.0 / expected_fill)


def test_only_margin_reserved_not_full_notional():
    p = ShortPortfolio(10000.0)
    cash_before = p.cash
    pos = p.sell_to_open("BTC/USDT", best_bid=100.0, notional_usdt=1000.0, fee_rate=0.0,
                          slippage=0.0, ts_ms=1000, leverage=2.0, **_meta())
    # only notional/leverage (=500) reserved, not the full 1000 notional
    assert cash_before - p.cash == pytest.approx(500.0)
    assert pos.margin_reserved == pytest.approx(500.0)


def test_buy_to_close_uses_ask_then_adverse_slippage_up():
    p = ShortPortfolio(10000.0)
    p.sell_to_open("BTC/USDT", best_bid=100.0, notional_usdt=1000.0, fee_rate=FEE, slippage=SLIP,
                   ts_ms=1000, leverage=2.0, **_meta())
    rec = p.buy_to_close("BTC/USDT", best_ask=90.0, fee_rate=FEE, slippage=SLIP,
                          ts_ms=2000, exit_reason="test")
    expected_fill = 90.0 * (1 + SLIP)  # worse (higher) than the raw ask
    assert rec["exit_price"] == pytest.approx(expected_fill)
    assert rec["net_pnl"] > 0  # price fell -- short profits


def test_short_profits_when_price_falls_loses_when_price_rises():
    p = ShortPortfolio(10000.0)
    p.sell_to_open("BTC/USDT", best_bid=100.0, notional_usdt=1000.0, fee_rate=0.0, slippage=0.0,
                   ts_ms=1000, leverage=2.0, **_meta())
    rec = p.buy_to_close("BTC/USDT", best_ask=80.0, fee_rate=0.0, slippage=0.0,
                          ts_ms=2000, exit_reason="test")
    assert rec["net_pnl"] > 0

    p2 = ShortPortfolio(10000.0)
    p2.sell_to_open("BTC/USDT", best_bid=100.0, notional_usdt=1000.0, fee_rate=0.0, slippage=0.0,
                    ts_ms=1000, leverage=2.0, **_meta())
    rec2 = p2.buy_to_close("BTC/USDT", best_ask=110.0, fee_rate=0.0, slippage=0.0,
                            ts_ms=2000, exit_reason="test")
    assert rec2["net_pnl"] < 0


def test_cannot_short_same_symbol_twice():
    p = ShortPortfolio(10000.0)
    p.sell_to_open("BTC/USDT", best_bid=100.0, notional_usdt=1000.0, fee_rate=FEE, slippage=SLIP,
                   ts_ms=1000, leverage=2.0, **_meta())
    with pytest.raises(ValueError):
        p.sell_to_open("BTC/USDT", best_bid=100.0, notional_usdt=500.0, fee_rate=FEE, slippage=SLIP,
                        ts_ms=1500, leverage=2.0, **_meta())


def test_cannot_close_without_open_position():
    p = ShortPortfolio(10000.0)
    with pytest.raises(ValueError):
        p.buy_to_close("BTC/USDT", best_ask=100.0, fee_rate=FEE, slippage=SLIP,
                        ts_ms=1000, exit_reason="x")


def test_insufficient_margin_rejected_cash_unchanged():
    p = ShortPortfolio(100.0)  # tiny account
    cash_before = p.cash
    with pytest.raises(ValueError):
        p.sell_to_open("BTC/USDT", best_bid=100.0, notional_usdt=10000.0, fee_rate=FEE,
                       slippage=SLIP, ts_ms=1000, leverage=2.0, **_meta())
    assert p.cash == cash_before  # unchanged -- raise happens before any mutation


def test_cash_never_goes_negative_across_a_round_trip():
    p = ShortPortfolio(10000.0)
    cash_before = p.cash
    p.sell_to_open("BTC/USDT", best_bid=100.0, notional_usdt=1000.0, fee_rate=FEE, slippage=SLIP,
                   ts_ms=1000, leverage=2.0, **_meta())
    assert p.cash >= 0
    p.buy_to_close("BTC/USDT", best_ask=100.0, fee_rate=FEE, slippage=SLIP, ts_ms=2000, exit_reason="x")
    assert p.cash >= 0
    assert p.cash == pytest.approx(cash_before + p.realized_pnl)


def test_liquidation_price_above_entry_for_a_short():
    p = ShortPortfolio(10000.0)
    pos = p.sell_to_open("BTC/USDT", best_bid=100.0, notional_usdt=1000.0, fee_rate=0.0, slippage=0.0,
                          ts_ms=1000, leverage=2.0, **_meta())
    assert pos.liquidation_price > pos.entry_price
    # at 2x leverage, full-margin-loss move is 50% -- liquidation should
    # trigger before that, scaled by MAINTENANCE_BUFFER
    expected_move = (1.0 / 2.0) * MAINTENANCE_BUFFER
    assert pos.liquidation_price == pytest.approx(pos.entry_price * (1 + expected_move))


def test_check_liquidation_true_when_price_above_liquidation_price():
    p = ShortPortfolio(10000.0)
    pos = p.sell_to_open("BTC/USDT", best_bid=100.0, notional_usdt=1000.0, fee_rate=0.0, slippage=0.0,
                          ts_ms=1000, leverage=2.0, **_meta())
    assert p.check_liquidation("BTC/USDT", pos.liquidation_price - 0.01) is False
    assert p.check_liquidation("BTC/USDT", pos.liquidation_price + 0.01) is True


def test_higher_leverage_means_closer_liquidation_price():
    p = ShortPortfolio(10000.0)
    pos_low_lev = p.sell_to_open("BTC/USDT", best_bid=100.0, notional_usdt=500.0, fee_rate=0.0,
                                  slippage=0.0, ts_ms=1000, leverage=2.0, **_meta())
    p2 = ShortPortfolio(10000.0)
    pos_high_lev = p2.sell_to_open("BTC/USDT", best_bid=100.0, notional_usdt=500.0, fee_rate=0.0,
                                    slippage=0.0, ts_ms=1000, leverage=5.0, **_meta())
    assert pos_high_lev.liquidation_price < pos_low_lev.liquidation_price


def test_borrow_cost_accrues_with_holding_time():
    p = ShortPortfolio(10000.0)
    p.sell_to_open("BTC/USDT", best_bid=100.0, notional_usdt=1000.0, fee_rate=0.0, slippage=0.0,
                   ts_ms=0, leverage=2.0, **_meta())
    one_day_ms = 86_400_000
    rec = p.buy_to_close("BTC/USDT", best_ask=100.0, fee_rate=0.0, slippage=0.0,
                          ts_ms=one_day_ms, exit_reason="x")
    assert rec["borrow_cost"] > 0
    assert rec["net_pnl"] == pytest.approx(-rec["borrow_cost"])  # flat price, only cost is borrow


def test_no_leverage_beyond_configured_default_unless_overridden():
    p = ShortPortfolio(10000.0, leverage=3.0)
    pos = p.sell_to_open("BTC/USDT", best_bid=100.0, notional_usdt=900.0, fee_rate=0.0, slippage=0.0,
                          ts_ms=1000, **_meta())  # no explicit leverage override
    assert pos.leverage == 3.0
    assert pos.margin_reserved == pytest.approx(300.0)


def test_equity_reflects_unrealized_pnl_not_notional_exposure():
    p = ShortPortfolio(10000.0)
    p.sell_to_open("BTC/USDT", best_bid=100.0, notional_usdt=1000.0, fee_rate=0.0, slippage=0.0,
                   ts_ms=1000, leverage=2.0, **_meta())
    prices_favorable = {"BTC/USDT": 50.0}  # price halved -- short way in profit
    eq = p.equity(prices_favorable)
    assert eq > p.start_capital  # equity should be UP, not treating the short notional as a loss


def test_snapshot_reports_leverage_and_liquidation_price():
    p = ShortPortfolio(10000.0)
    p.sell_to_open("BTC/USDT", best_bid=100.0, notional_usdt=1000.0, fee_rate=FEE, slippage=SLIP,
                   ts_ms=1000, leverage=2.0, **_meta())
    snap = p.snapshot({"BTC/USDT": 100.0})
    assert "BTC/USDT" in snap["positions"]
    assert snap["positions"]["BTC/USDT"]["leverage"] == 2.0
    assert "liquidation_price" in snap["positions"]["BTC/USDT"]
