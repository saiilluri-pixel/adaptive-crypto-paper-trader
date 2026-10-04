"""Deterministic tests for adaptive/risk_engine.py."""
import pytest

from adaptive.risk_engine import RiskEngine, RiskLimits, DegradationState


def _engine(**overrides):
    return RiskEngine(RiskLimits(**overrides))


def test_risk_budget_sizing_basic():
    # Pinned risk_per_trade_pct rather than relying on RiskLimits' deployed
    # default -- this test is about the sizing FORMULA (risk_amount =
    # equity * risk_pct/100), not "what is today's deployed config."
    eng = _engine(risk_per_trade_pct=0.50)
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


def test_degradation_tiers_by_profit_factor():
    """Aggressive-profile calibration: DEFENSIVE only for a decisively
    losing book (PF < 0.6); a mildly negative/breakeven book (0.6 <= PF <
    1.0) sits in CAUTION, not DEFENSIVE; PF >= 1.0 is NORMAL. Expectancy no
    longer independently forces DEFENSIVE."""
    eng = _engine()
    d = eng.evaluate_degradation(drawdown_pct=1.0, loss_streak=0,
                                  recent_expectancy=-5.0, recent_pf=0.5)
    assert d == DegradationState.DEFENSIVE          # PF 0.5 < 0.6
    c = eng.evaluate_degradation(drawdown_pct=1.0, loss_streak=0,
                                  recent_expectancy=-5.0, recent_pf=0.9)
    assert c == DegradationState.CAUTION            # mildly negative -> CAUTION, not DEFENSIVE
    n = eng.evaluate_degradation(drawdown_pct=1.0, loss_streak=0,
                                  recent_expectancy=-0.1, recent_pf=1.05)
    assert n == DegradationState.NORMAL             # PF >= 1.0 -> NORMAL even if expectancy slightly <0


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
    # References the live constant rather than hardcoding its value -- this
    # is a mechanism test ("is exploration capped at the configured max"),
    # not an assertion about what that max currently is.
    from adaptive.risk_engine import EXPLORATION_RISK_MAX_PCT
    eng = _engine()
    r = eng.size_exploration_entry(equity=10000, cash=10000, current_portfolio_heat_usdt=0,
                                    current_crypto_value_usdt=0, stop_distance_frac=0.05, confidence=0.8,
                                    signal_strength=1.0)
    assert r.approved
    assert r.risk_amount_usdt == pytest.approx(10000 * EXPLORATION_RISK_MAX_PCT / 100.0)
    assert r.binding_constraint.startswith("exploration:")


def test_exploration_risk_scales_with_signal_strength():
    from adaptive.meta_controller import MIN_EXPLORATION_SIGNAL_STRENGTH
    from adaptive.risk_engine import EXPLORATION_RISK_MIN_PCT, EXPLORATION_RISK_MAX_PCT
    eng = _engine()
    weak = eng.size_exploration_entry(equity=10000, cash=10000, current_portfolio_heat_usdt=0,
                                       current_crypto_value_usdt=0, stop_distance_frac=0.05, confidence=0.8,
                                       signal_strength=MIN_EXPLORATION_SIGNAL_STRENGTH)  # right at the floor
    strong = eng.size_exploration_entry(equity=10000, cash=10000, current_portfolio_heat_usdt=0,
                                         current_crypto_value_usdt=0, stop_distance_frac=0.05, confidence=0.8,
                                         signal_strength=1.0)
    assert weak.risk_amount_usdt == pytest.approx(10000 * EXPLORATION_RISK_MIN_PCT / 100.0)
    assert strong.risk_amount_usdt == pytest.approx(10000 * EXPLORATION_RISK_MAX_PCT / 100.0)
    assert weak.risk_amount_usdt < strong.risk_amount_usdt


def test_exploration_risk_smaller_than_normal_at_max_strength():
    # Pinned risk_per_trade_pct -- this test is about the RATIO relationship
    # (exploration always stays below normal sizing), not today's exact
    # deployed numbers.
    from adaptive.risk_engine import EXPLORATION_RISK_MAX_PCT
    eng = _engine(risk_per_trade_pct=0.50)
    normal = eng.size_entry(equity=10000, cash=10000, current_portfolio_heat_usdt=0,
                             current_crypto_value_usdt=0, stop_distance_frac=0.05, confidence=0.8)
    exploration = eng.size_exploration_entry(equity=10000, cash=10000, current_portfolio_heat_usdt=0,
                                              current_crypto_value_usdt=0, stop_distance_frac=0.05, confidence=0.8,
                                              signal_strength=1.0)
    assert exploration.risk_amount_usdt == pytest.approx(normal.risk_amount_usdt * EXPLORATION_RISK_MAX_PCT / 0.50)
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
    basic risk management, including the degradation state machine's HALTED
    hard-stop (risk multiplier 0)."""
    eng = _engine()
    eng.set_state(DegradationState.HALTED, "test")
    r = eng.size_exploration_entry(equity=10000, cash=10000, current_portfolio_heat_usdt=0,
                                    current_crypto_value_usdt=0, stop_distance_frac=0.05, confidence=1.0,
                                    signal_strength=1.0)
    assert not r.approved


def test_exploration_not_blocked_by_defensive_confidence_floor(monkeypatch):
    """Regression for the DEFENSIVE deadlock: an exploration entry (confidence
    ~0.0 by construction -- n=0 cell) must NOT be rejected by the DEFENSIVE
    confidence floor, or the data-gathering mechanism freezes exactly when
    the book is stuck in DEFENSIVE (observed live: 304 consecutive
    'confidence 0.00 below DEFENSIVE floor 0.45' rejections). A NORMAL entry
    at the same ~0 confidence IS still rejected -- the exemption is
    exploration-only."""
    import adaptive.risk_engine as rm
    monkeypatch.setitem(rm.STATE_MIN_CONFIDENCE, DegradationState.DEFENSIVE, 0.45)
    monkeypatch.setitem(rm.STATE_RISK_MULTIPLIER, DegradationState.DEFENSIVE, 0.60)
    eng = _engine()
    eng.set_state(DegradationState.DEFENSIVE, "test")
    # exploration at confidence 0.0 -> APPROVED (floor exempt), still tiny
    expl = eng.size_exploration_entry(equity=10000, cash=10000, current_portfolio_heat_usdt=0,
                                       current_crypto_value_usdt=0, stop_distance_frac=0.05, confidence=0.0,
                                       signal_strength=1.0)
    assert expl.approved
    assert expl.risk_amount_usdt > 0
    # a NORMAL entry at the same 0.0 confidence is still blocked by the floor
    norm = eng.size_entry(equity=10000, cash=10000, current_portfolio_heat_usdt=0,
                           current_crypto_value_usdt=0, stop_distance_frac=0.05, confidence=0.0)
    assert not norm.approved
    assert norm.binding_constraint == "degradation_state_confidence"


def test_defensive_state_requires_high_confidence(monkeypatch):
    # Monkeypatched to a known floor/multiplier rather than deriving both
    # the test inputs AND the assertion from the live module dict -- doing
    # that would make the test tautological (it'd "pass" for any floor,
    # including 0.0, since low_conf's confidence would track the floor
    # down with it). This pins the mechanism independent of today's
    # deployed DEFENSIVE values.
    import adaptive.risk_engine as risk_engine_mod
    monkeypatch.setitem(risk_engine_mod.STATE_MIN_CONFIDENCE, DegradationState.DEFENSIVE, 0.65)
    monkeypatch.setitem(risk_engine_mod.STATE_RISK_MULTIPLIER, DegradationState.DEFENSIVE, 0.25)
    eng = _engine(risk_per_trade_pct=0.50)
    eng.set_state(DegradationState.DEFENSIVE, "test")
    low_conf = eng.size_entry(equity=10000, cash=10000, current_portfolio_heat_usdt=0,
                               current_crypto_value_usdt=0, stop_distance_frac=0.05, confidence=0.5)
    high_conf = eng.size_entry(equity=10000, cash=10000, current_portfolio_heat_usdt=0,
                                current_crypto_value_usdt=0, stop_distance_frac=0.05, confidence=0.9)
    assert not low_conf.approved
    assert high_conf.approved
    assert high_conf.risk_amount_usdt == pytest.approx(10000 * 0.005 * 0.25)
