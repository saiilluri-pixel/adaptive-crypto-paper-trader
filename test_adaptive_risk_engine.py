"""Deterministic tests for adaptive/risk_engine.py."""
import pytest

from adaptive.risk_engine import RiskEngine, RiskLimits, DegradationState


def _engine(**overrides):
    return RiskEngine(RiskLimits(**overrides))


def test_risk_budget_sizing_basic():
    eng = _engine()
    r = eng.size_entry(equity=10000, cash=10000, current_portfolio_heat_usdt=0,
                        current_crypto_value_usdt=0, stop_distance_frac=0.05, confidence=0.8)
    assert r.approved
    expected_risk = 10000 * 0.005  # 0.5% of equity
    assert r.risk_amount_usdt == pytest.approx(expected_risk)
    assert r.notional_usdt == pytest.approx(expected_risk / 0.05)
    assert r.binding_constraint == "risk_budget"


def test_stop_sizing_scales_inversely_with_stop_distance():
    eng = _engine()
    tight = eng.size_entry(equity=10000, cash=10000, current_portfolio_heat_usdt=0,
                            current_crypto_value_usdt=0, stop_distance_frac=0.02, confidence=0.8)
    wide = eng.size_entry(equity=10000, cash=10000, current_portfolio_heat_usdt=0,
                           current_crypto_value_usdt=0, stop_distance_frac=0.10, confidence=0.8)
    assert tight.notional_usdt > wide.notional_usdt  # tighter stop -> larger notional for same risk


def test_portfolio_heat_blocks_excess_entries():
    eng = _engine(max_portfolio_heat_pct=1.5, risk_per_trade_pct=0.5)
    # heat budget = 150 USDT on a 10k account; prior positions already used
    # 120 USDT of that -- only 30 USDT remains, strictly less than a
    # standard 50 USDT (0.5%) risk_budget trade would want
    r = eng.size_entry(equity=10000, cash=10000, current_portfolio_heat_usdt=120.0,
                        current_crypto_value_usdt=2000, stop_distance_frac=0.05, confidence=0.8)
    assert r.approved
    assert r.binding_constraint == "portfolio_heat"
    assert r.risk_amount_usdt == pytest.approx(30.0)


def test_portfolio_heat_fully_exhausted_rejects_new_entry():
    eng = _engine(max_portfolio_heat_pct=1.5, risk_per_trade_pct=0.5)
    r = eng.size_entry(equity=10000, cash=10000, current_portfolio_heat_usdt=150.0,
                        current_crypto_value_usdt=2000, stop_distance_frac=0.05, confidence=0.8)
    assert not r.approved
    assert r.binding_constraint == "portfolio_heat"


def test_symbol_allocation_cap_binds():
    eng = _engine(max_symbol_allocation_pct=5.0)  # tiny cap vs a huge risk budget
    r = eng.size_entry(equity=10000, cash=10000, current_portfolio_heat_usdt=0,
                        current_crypto_value_usdt=0, stop_distance_frac=0.001, confidence=0.8)
    assert r.approved
    assert r.binding_constraint == "symbol_allocation"
    assert r.notional_usdt == pytest.approx(500.0)


def test_total_exposure_cap_binds():
    eng = _engine(max_total_crypto_allocation_pct=90.0)
    # cash=2000 keeps the 10%-reserve cap well above 100 (2000-1000=1000)
    # so total_exposure (10000*0.9-8900=100) is unambiguously the tightest
    r = eng.size_entry(equity=10000, cash=2000, current_portfolio_heat_usdt=0,
                        current_crypto_value_usdt=8900, stop_distance_frac=0.001, confidence=0.8)
    assert r.approved
    assert r.binding_constraint == "total_exposure"
    assert r.notional_usdt == pytest.approx(100.0)


def test_cash_reserve_cap_binds():
    eng = _engine(min_cash_reserve_pct=10.0)
    r = eng.size_entry(equity=10000, cash=1050, current_portfolio_heat_usdt=0,
                        current_crypto_value_usdt=8950, stop_distance_frac=0.001, confidence=0.8)
    assert r.approved
    assert r.binding_constraint == "cash_reserve"
    assert r.notional_usdt == pytest.approx(50.0)


def test_invalid_stop_rejected():
    eng = _engine()
    r = eng.size_entry(equity=10000, cash=10000, current_portfolio_heat_usdt=0,
                        current_crypto_value_usdt=0, stop_distance_frac=0.0, confidence=0.8)
    assert not r.approved
    assert r.binding_constraint == "invalid_stop"


def test_daily_loss_breaker():
    eng = _engine(daily_loss_limit_pct=2.0)
    reason = eng.check_breakers(equity=9700, peak_equity=10000, daily_start_equity=10000,
                                 consecutive_losses=0)
    assert reason is not None
    assert "daily loss" in reason


def test_max_drawdown_breaker():
    eng = _engine(max_drawdown_limit_pct=10.0)
    reason = eng.check_breakers(equity=8900, peak_equity=10000, daily_start_equity=8900,
                                 consecutive_losses=0)
    assert reason is not None
    assert "drawdown" in reason


def test_consecutive_loss_breaker():
    eng = _engine(consecutive_loss_breaker=5)
    reason = eng.check_breakers(equity=10000, peak_equity=10000, daily_start_equity=10000,
                                 consecutive_losses=5)
    assert reason is not None
    assert "consecutive losses" in reason


def test_breakers_clear_when_within_limits():
    eng = _engine()
    reason = eng.check_breakers(equity=10000, peak_equity=10000, daily_start_equity=10000,
                                 consecutive_losses=0)
    assert reason is None


def test_degradation_normal_by_default_with_insufficient_sample():
    eng = _engine()
    state = eng.evaluate_degradation(drawdown_pct=1.0, loss_streak=0,
                                      recent_expectancy=None, recent_pf=None)
    assert state == DegradationState.NORMAL


def test_degradation_defensive_on_negative_expectancy():
    eng = _engine()
    state = eng.evaluate_degradation(drawdown_pct=1.0, loss_streak=0,
                                      recent_expectancy=-5.0, recent_pf=0.9)
    assert state == DegradationState.DEFENSIVE


def test_degradation_halted_on_drawdown_limit():
    eng = _engine(max_drawdown_limit_pct=10.0)
    state = eng.evaluate_degradation(drawdown_pct=10.5, loss_streak=0,
                                      recent_expectancy=5.0, recent_pf=1.5)
    assert state == DegradationState.HALTED


def test_degradation_halted_on_loss_streak():
    eng = _engine(consecutive_loss_breaker=5)
    state = eng.evaluate_degradation(drawdown_pct=0.0, loss_streak=5,
                                      recent_expectancy=5.0, recent_pf=1.5)
    assert state == DegradationState.HALTED


def test_halted_state_blocks_all_new_entries_regardless_of_confidence():
    eng = _engine()
    eng.set_state(DegradationState.HALTED, "test")
    r = eng.size_entry(equity=10000, cash=10000, current_portfolio_heat_usdt=0,
                        current_crypto_value_usdt=0, stop_distance_frac=0.05, confidence=1.0)
    assert not r.approved


def test_exploration_risk_capped_at_max_pct_for_full_strength_signal():
    from adaptive.risk_engine import EXPLORATION_RISK_MAX_PCT
    assert EXPLORATION_RISK_MAX_PCT == pytest.approx(0.15)
    eng = _engine()
    r = eng.size_exploration_entry(equity=10000, cash=10000, current_portfolio_heat_usdt=0,
                                    current_crypto_value_usdt=0, stop_distance_frac=0.05, confidence=0.8,
                                    signal_strength=1.0)
    assert r.approved
    assert r.risk_amount_usdt == pytest.approx(10000 * 0.0015)  # 0.15% of equity at max strength
    assert r.binding_constraint.startswith("exploration:")


def test_exploration_risk_scales_with_signal_strength():
    from adaptive.meta_controller import MIN_EXPLORATION_SIGNAL_STRENGTH
    eng = _engine()
    weak = eng.size_exploration_entry(equity=10000, cash=10000, current_portfolio_heat_usdt=0,
                                       current_crypto_value_usdt=0, stop_distance_frac=0.05, confidence=0.8,
                                       signal_strength=MIN_EXPLORATION_SIGNAL_STRENGTH)  # right at the floor
    strong = eng.size_exploration_entry(equity=10000, cash=10000, current_portfolio_heat_usdt=0,
                                         current_crypto_value_usdt=0, stop_distance_frac=0.05, confidence=0.8,
                                         signal_strength=1.0)
    assert weak.risk_amount_usdt == pytest.approx(10000 * 0.0008)  # EXPLORATION_RISK_MIN_PCT
    assert strong.risk_amount_usdt == pytest.approx(10000 * 0.0015)  # EXPLORATION_RISK_MAX_PCT
    assert weak.risk_amount_usdt < strong.risk_amount_usdt


def test_exploration_risk_smaller_than_normal_at_max_strength():
    eng = _engine()
    normal = eng.size_entry(equity=10000, cash=10000, current_portfolio_heat_usdt=0,
                             current_crypto_value_usdt=0, stop_distance_frac=0.05, confidence=0.8)
    exploration = eng.size_exploration_entry(equity=10000, cash=10000, current_portfolio_heat_usdt=0,
                                              current_crypto_value_usdt=0, stop_distance_frac=0.05, confidence=0.8,
                                              signal_strength=1.0)
    # 0.15% exploration vs 0.50% normal risk_per_trade_pct -- still meaningfully
    # smaller even after being raised, never as large as a normally-scored entry
    assert exploration.risk_amount_usdt == pytest.approx(normal.risk_amount_usdt * 0.15 / 0.50)
    assert exploration.risk_amount_usdt < normal.risk_amount_usdt


def test_exploration_still_respects_portfolio_heat_cap():
    eng = _engine(max_portfolio_heat_pct=1.5)
    # heat already fully used by normal-sized positions -- exploration must not bypass this
    r = eng.size_exploration_entry(equity=10000, cash=10000, current_portfolio_heat_usdt=150.0,
                                    current_crypto_value_usdt=2000, stop_distance_frac=0.05, confidence=0.8,
                                    signal_strength=1.0)
    assert not r.approved


def test_exploration_blocked_in_halted_state():
    """Normal risk guards still apply to exploration -- it never bypasses
    basic risk management, including the degradation state machine."""
    eng = _engine()
    eng.set_state(DegradationState.HALTED, "test")
    r = eng.size_exploration_entry(equity=10000, cash=10000, current_portfolio_heat_usdt=0,
                                    current_crypto_value_usdt=0, stop_distance_frac=0.05, confidence=1.0,
                                    signal_strength=1.0)
    assert not r.approved


def test_defensive_state_requires_high_confidence():
    eng = _engine()
    eng.set_state(DegradationState.DEFENSIVE, "test")
    low_conf = eng.size_entry(equity=10000, cash=10000, current_portfolio_heat_usdt=0,
                               current_crypto_value_usdt=0, stop_distance_frac=0.05, confidence=0.5)
    high_conf = eng.size_entry(equity=10000, cash=10000, current_portfolio_heat_usdt=0,
                                current_crypto_value_usdt=0, stop_distance_frac=0.05, confidence=0.9)
    assert not low_conf.approved
    assert high_conf.approved
    assert high_conf.risk_amount_usdt == pytest.approx(10000 * 0.005 * 0.25)  # DEFENSIVE = 25% risk
