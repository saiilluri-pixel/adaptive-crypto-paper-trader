"""Deterministic tests for adaptive/stats_store.py and adaptive/meta_controller.py."""
import pytest

from adaptive.stats_store import StatsStore
from adaptive.strategies import StrategySignal
from adaptive.meta_controller import (
    score_opportunity, rank_and_select, rolling_correlation_matrix,
    MIN_QUALITY_SCORE,
)


def _signal(strategy="trend_momentum", strength=0.8, stop_pct=2.0, expected_rr=2.0,
            regime_compat=("TREND_UP",)):
    return StrategySignal(strategy=strategy, direction="long", strength=strength,
                           expected_rr=expected_rr, stop_pct=stop_pct, exit_plan="x",
                           regime_compatibility=list(regime_compat))


# ── stats_store shrinkage ────────────────────────────────────────────
def test_shrinkage_pulls_small_sample_toward_prior():
    store = StatsStore(shrinkage_k=20.0, global_prior_mean=0.0)
    store.record_trade("BTC/USDT", "trend_momentum", "TREND_UP", 10.0)  # one huge winner
    shrunk = store.shrunk_expectancy("BTC/USDT", "trend_momentum", "TREND_UP")
    assert 0 < shrunk < 10.0  # pulled well below the raw 10% mean
    assert shrunk == pytest.approx((1 / 21) * 10.0)


def test_shrinkage_converges_to_raw_mean_with_large_sample():
    store = StatsStore(shrinkage_k=20.0, global_prior_mean=0.0)
    for _ in range(500):
        store.record_trade("BTC/USDT", "trend_momentum", "TREND_UP", 1.0)
    shrunk = store.shrunk_expectancy("BTC/USDT", "trend_momentum", "TREND_UP")
    assert shrunk == pytest.approx(1.0, abs=0.05)


def test_uncertainty_decreases_with_sample_size():
    store = StatsStore(shrinkage_k=20.0)
    u0 = store.uncertainty("BTC/USDT", "trend_momentum", "TREND_UP")
    for _ in range(50):
        store.record_trade("BTC/USDT", "trend_momentum", "TREND_UP", 1.0)
    u1 = store.uncertainty("BTC/USDT", "trend_momentum", "TREND_UP")
    assert u1 < u0
    assert u0 == pytest.approx(1.0)  # n=0 -> full uncertainty


def test_stats_store_round_trip_serialization():
    store = StatsStore()
    store.record_trade("BTC/USDT", "trend_momentum", "TREND_UP", 2.5)
    store.record_trade("BTC/USDT", "trend_momentum", "TREND_UP", -1.0)
    d = store.to_dict()
    restored = StatsStore.from_dict(d)
    cell = restored.get("BTC/USDT", "trend_momentum", "TREND_UP")
    assert cell.trades == [2.5, -1.0]
    assert cell.wins == 1 and cell.losses == 1


# ── opportunity scoring ──────────────────────────────────────────────
def test_zero_edge_scores_non_positive_regardless_of_strength():
    store = StatsStore(global_prior_mean=0.0)  # no trades recorded -> shrunk_edge == prior == 0
    sig = _signal(strength=1.0)
    opp = score_opportunity(symbol="BTC/USDT", signal=sig, regime="TREND_UP", stats_store=store,
                             friction_penalty_pct=0.0, already_selected=[], correlation_matrix={})
    assert opp.score <= 0  # shrunk_edge=0 multiplicatively zeroes the positive term


def test_positive_edge_scores_higher_with_better_regime_fit():
    store = StatsStore()
    for _ in range(30):
        store.record_trade("BTC/USDT", "trend_momentum", "TREND_UP", 1.0)
    sig_fit = _signal(regime_compat=("TREND_UP",))
    sig_no_fit = _signal(regime_compat=("RANGE",))
    opp_fit = score_opportunity(symbol="BTC/USDT", signal=sig_fit, regime="TREND_UP", stats_store=store,
                                 friction_penalty_pct=0.05, already_selected=[], correlation_matrix={})
    opp_no_fit = score_opportunity(symbol="BTC/USDT", signal=sig_no_fit, regime="TREND_UP", stats_store=store,
                                    friction_penalty_pct=0.05, already_selected=[], correlation_matrix={})
    assert opp_fit.score > opp_no_fit.score


def test_friction_penalty_reduces_score():
    store = StatsStore()
    for _ in range(30):
        store.record_trade("BTC/USDT", "trend_momentum", "TREND_UP", 1.0)
    sig = _signal()
    low_friction = score_opportunity(symbol="BTC/USDT", signal=sig, regime="TREND_UP", stats_store=store,
                                      friction_penalty_pct=0.01, already_selected=[], correlation_matrix={})
    high_friction = score_opportunity(symbol="BTC/USDT", signal=sig, regime="TREND_UP", stats_store=store,
                                       friction_penalty_pct=5.0, already_selected=[], correlation_matrix={})
    assert low_friction.score > high_friction.score


# ── ranking / 0-1-2-3 positions ──────────────────────────────────────
def _good_opportunity(symbol, score):
    return score_opportunity(
        symbol=symbol, signal=_signal(), regime="TREND_UP",
        stats_store=_store_with_edge(symbol, score),
        friction_penalty_pct=0.02, already_selected=[], correlation_matrix={})


def _store_with_edge(symbol, target_score_hint):
    store = StatsStore()
    n = 40
    for _ in range(n):
        store.record_trade(symbol, "trend_momentum", "TREND_UP", target_score_hint)
    return store


def test_no_candidates_selects_zero_positions():
    selected = rank_and_select([], correlation_matrix={}, max_positions=3)
    assert selected == []


def test_below_threshold_candidate_never_selected():
    store = StatsStore()  # zero edge everywhere
    sig = _signal()
    opp = score_opportunity(symbol="BTC/USDT", signal=sig, regime="TREND_UP", stats_store=store,
                             friction_penalty_pct=0.02, already_selected=[], correlation_matrix={})
    assert opp.score < MIN_QUALITY_SCORE
    selected = rank_and_select([opp], correlation_matrix={}, max_positions=3)
    assert selected == []


def test_one_qualifying_candidate_selects_one_position():
    opp = _good_opportunity("BTC/USDT", 3.0)
    selected = rank_and_select([opp], correlation_matrix={}, max_positions=3)
    assert len(selected) == 1


def test_three_independent_qualifiers_select_three_positions():
    opps = [_good_opportunity(s, 3.0) for s in ("BTC/USDT", "ETH/USDT", "SOL/USDT")]
    selected = rank_and_select(opps, correlation_matrix={}, max_positions=3)
    assert len(selected) == 3


def test_correlated_signals_reduce_allocation():
    """Two highly-correlated symbols both signaling long: the SECOND one's
    penalty must grow once the first is selected, relative to an
    uncorrelated third symbol -- proving correlation reduces (not just
    caps) the effective ranking of redundant exposure."""
    opps = [_good_opportunity(s, 3.0) for s in ("BTC/USDT", "ETH/USDT", "SOL/USDT")]
    corr = {("BTC/USDT", "ETH/USDT"): 0.95, ("BTC/USDT", "SOL/USDT"): 0.1, ("ETH/USDT", "SOL/USDT"): 0.1}
    selected = rank_and_select(opps, correlation_matrix=corr, max_positions=3)
    by_symbol = {o.symbol: o for o in selected}
    # ETH (highly correlated with BTC, selected first since all start equal)
    # must carry a larger correlation penalty than SOL (low correlation)
    assert by_symbol["ETH/USDT"].correlation_penalty > by_symbol["SOL/USDT"].correlation_penalty


def test_max_positions_cap_respected_even_with_more_qualifiers():
    opps = [_good_opportunity(s, 3.0) for s in ("BTC/USDT", "ETH/USDT", "SOL/USDT")]
    selected = rank_and_select(opps, correlation_matrix={}, max_positions=2)
    assert len(selected) == 2


# ── correlation matrix computation ───────────────────────────────────
def test_correlation_matrix_detects_strong_positive_correlation():
    returns = {"A": [1, 2, -1, 3, -2, 4, 1, 2, -1, 3] * 3,
               "B": [1, 2, -1, 3, -2, 4, 1, 2, -1, 3] * 3}
    m = rolling_correlation_matrix(returns)
    assert m[("A", "B")] == pytest.approx(1.0, abs=0.01)


def test_correlation_matrix_low_for_uncorrelated_series():
    returns = {"A": [1, -1] * 15, "B": [1, 1, -1, -1] * 8}
    m = rolling_correlation_matrix(returns)
    assert abs(m[("A", "B")]) < 0.5


def test_correlation_matrix_insufficient_sample_defaults_to_zero():
    returns = {"A": [1, 2, 3], "B": [1, 2, 3]}
    m = rolling_correlation_matrix(returns)
    assert m[("A", "B")] == 0.0


# ── cold-start deadlock: exploration eligibility ─────────────────────
def test_cold_start_deadlock_confirmed_zero_trades_never_clears_threshold():
    """CAN A BRAND-NEW SYSTEM OPEN ITS FIRST TRADE via normal scoring
    alone? NO -- proven here with MAXIMUM possible signal strength/regime
    fit/robustness: shrunk_edge is pinned to exactly the global prior
    (0.0) at n=0, zeroing the score formula's only positive term, while
    friction_penalty and uncertainty_penalty stay strictly positive."""
    store = StatsStore()  # fresh, global_prior_mean=0.0, exactly as runner.py constructs it
    sig = _signal(strength=1.0, regime_compat=("TREND_UP",))  # best possible raw signal
    opp = score_opportunity(symbol="BTC/USDT", signal=sig, regime="TREND_UP", stats_store=store,
                             friction_penalty_pct=0.01, already_selected=[], correlation_matrix={})
    assert opp.shrunk_edge == 0.0
    assert opp.score < MIN_QUALITY_SCORE
    assert opp.score < 0  # not just "below threshold" -- strictly negative


def test_exploration_eligible_when_zero_trades_and_strong_signal():
    from adaptive.meta_controller import MIN_EXPLORATION_SIGNAL_STRENGTH
    store = StatsStore()
    sig = _signal(strength=MIN_EXPLORATION_SIGNAL_STRENGTH)
    opp = score_opportunity(symbol="BTC/USDT", signal=sig, regime="TREND_UP", stats_store=store,
                             friction_penalty_pct=0.01, already_selected=[], correlation_matrix={})
    assert opp.is_exploration_eligible is True


def test_exploration_not_eligible_for_weak_signal():
    """Weak raw signal must still be rejected even at n=0 -- exploration
    is not a blanket bypass, it has its own quality floor."""
    from adaptive.meta_controller import MIN_EXPLORATION_SIGNAL_STRENGTH
    store = StatsStore()
    sig = _signal(strength=MIN_EXPLORATION_SIGNAL_STRENGTH - 0.05)
    opp = score_opportunity(symbol="BTC/USDT", signal=sig, regime="TREND_UP", stats_store=store,
                             friction_penalty_pct=0.01, already_selected=[], correlation_matrix={})
    assert opp.is_exploration_eligible is False


def test_exploration_not_eligible_once_a_trade_exists():
    """Once n>=1 for a cell, shrinkage takes over -- exploration is a
    zero-trades-only escape hatch, not a permanent alternate path."""
    store = StatsStore()
    store.record_trade("BTC/USDT", "trend_momentum", "TREND_UP", 1.0)
    sig = _signal(strength=1.0)
    opp = score_opportunity(symbol="BTC/USDT", signal=sig, regime="TREND_UP", stats_store=store,
                             friction_penalty_pct=0.01, already_selected=[], correlation_matrix={})
    assert opp.is_exploration_eligible is False
