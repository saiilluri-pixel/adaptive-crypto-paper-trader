"""
The "AI"/meta-controller (adaptive spec section 8). NOT an LLM, NOT
per-trade machine learning -- a documented, deterministic, quantitative
scoring formula over rolling per (symbol, strategy, regime) statistics
(adaptive/stats_store.py, empirical-Bayes shrunk) applied to the live
StrategySignal candidates from adaptive/strategies.py. Per the spec's
explicit instruction: use quantitative evidence, and document the formula.

FORMULA
=======
    opportunity_score =
          shrunk_edge            -- empirical-Bayes-shrunk expectancy (%) for this
                                     (symbol, strategy, regime) cell
        * signal_strength        -- this signal's own strength, 0..1
        * regime_fit             -- 1.0 if the strategy is running in a regime it declares
                                     compatibility with, else REGIME_PARTIAL_CREDIT
        * robustness              -- profit-factor-derived multiplier, blended toward a
                                     neutral 1.0 as sample size shrinks (never a bonus/
                                     penalty from evidence that isn't there yet)
        - friction_penalty        -- modeled round-trip cost (2x fee + 2x slippage + typical
                                     spread), a %-return drag subtracted directly
        - uncertainty_penalty     -- shrinkage weight (k/(n+k)) x a fixed coefficient
        - drawdown_penalty        -- this cell's own worst historical drawdown x a fixed
                                     coefficient
        - correlation_penalty     -- see rank_and_select() below

All penalty terms are expressed in the SAME %-return units as shrunk_edge
so they can be subtracted directly, rather than combined by opaque
un-normalized multiplication. shrunk_edge anchors the formula
multiplicatively -- a candidate with zero or negative shrunk edge cannot
score positively no matter how strong the raw signal or how good the
regime fit, which is the intended behavior: unproven edge should not drive
sizing decisions on strength alone.

A candidate must clear MIN_QUALITY_SCORE to be entered at all -- this is
what lets the system hold 0 positions when nothing qualifies (section 3).

RANKING / 0-1-2-3 POSITIONS
============================
rank_and_select() is a greedy sequential pick: take the best-scoring
candidate, then RE-SCORE the remaining candidates' correlation_penalty
against the growing selected set (so a second BTC-correlated long is
penalized relative to what's already chosen, not evaluated in isolation),
repeat until max_positions is reached or nothing left clears
MIN_QUALITY_SCORE. This is what makes "hold the strongest one, or two, or
all three" an emergent property of scoring plus correlation, rather than a
hard-coded target count.
"""
import itertools
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import numpy as np

MIN_QUALITY_SCORE = 0.05
REGIME_PARTIAL_CREDIT = 0.4
UNCERTAINTY_PENALTY_COEF = 0.5
DRAWDOWN_PENALTY_COEF = 0.10
CORRELATION_PENALTY_COEF = 0.6


@dataclass
class Opportunity:
    symbol: str
    strategy: str
    regime: str
    direction: str
    score: float
    shrunk_edge: float
    signal_strength: float
    regime_fit: float
    robustness: float
    friction_penalty: float
    uncertainty_penalty: float
    drawdown_penalty: float
    correlation_penalty: float
    stop_pct: float
    expected_rr: float
    confidence: float  # 1 - uncertainty; consumed by risk_engine's DEFENSIVE-state confidence gate


def robustness_multiplier(cell) -> float:
    """Blends a PF-derived multiplier toward neutral (1.0) as sample size
    shrinks -- 'not yet proven' must never score as either better or worse
    than average, only 'unknown'."""
    pf, n = cell.profit_factor, cell.n
    if pf is None or n < 5:
        return 1.0
    pf_component = float(np.clip(pf / 1.5, 0.5, 1.5))
    shrink = n / (n + 20.0)
    return 1.0 * (1 - shrink) + pf_component * shrink


def regime_fit_score(strategy_regime_compat: List[str], current_regime: str) -> float:
    return 1.0 if current_regime in strategy_regime_compat else REGIME_PARTIAL_CREDIT


def friction_pct(fee_rate: float, slippage: float, typical_spread_pct: float) -> float:
    return (fee_rate + slippage) * 2 * 100 + typical_spread_pct


def correlation_penalty(symbol: str, direction: str, already_selected: List[Tuple[str, str]],
                         correlation_matrix: Dict[Tuple[str, str], float]) -> float:
    if not already_selected:
        return 0.0
    total = 0.0
    for sel_symbol, sel_direction in already_selected:
        if sel_direction != direction or sel_symbol == symbol:
            continue
        corr = correlation_matrix.get((symbol, sel_symbol))
        if corr is None:
            corr = correlation_matrix.get((sel_symbol, symbol), 0.0)
        total += max(0.0, corr)
    return CORRELATION_PENALTY_COEF * total


def score_opportunity(*, symbol: str, signal, regime: str, stats_store,
                       friction_penalty_pct: float, already_selected: List[Tuple[str, str]],
                       correlation_matrix: Dict[Tuple[str, str], float]) -> Opportunity:
    cell = stats_store.get(symbol, signal.strategy, regime)
    shrunk_edge = stats_store.shrunk_expectancy(symbol, signal.strategy, regime)
    uncertainty = stats_store.uncertainty(symbol, signal.strategy, regime)
    rfit = regime_fit_score(signal.regime_compatibility, regime)
    robust = robustness_multiplier(cell)
    unc_pen = uncertainty * UNCERTAINTY_PENALTY_COEF
    dd_pen = cell.max_dd_pct * DRAWDOWN_PENALTY_COEF
    corr_pen = correlation_penalty(symbol, signal.direction, already_selected, correlation_matrix)

    score = (shrunk_edge * signal.strength * rfit * robust
             - friction_penalty_pct - unc_pen - dd_pen - corr_pen)

    return Opportunity(symbol=symbol, strategy=signal.strategy, regime=regime,
                        direction=signal.direction, score=score, shrunk_edge=shrunk_edge,
                        signal_strength=signal.strength, regime_fit=rfit, robustness=robust,
                        friction_penalty=friction_penalty_pct, uncertainty_penalty=unc_pen,
                        drawdown_penalty=dd_pen, correlation_penalty=corr_pen,
                        stop_pct=signal.stop_pct, expected_rr=signal.expected_rr,
                        confidence=1.0 - uncertainty)


def rank_and_select(candidates: List[Opportunity],
                     correlation_matrix: Dict[Tuple[str, str], float],
                     max_positions: int = 3) -> List[Opportunity]:
    remaining = sorted([c for c in candidates if c.direction == "long"],
                        key=lambda o: o.score, reverse=True)
    selected: List[Opportunity] = []
    selected_pairs: List[Tuple[str, str]] = []
    while remaining and len(selected) < max_positions:
        best = remaining[0]
        if best.score < MIN_QUALITY_SCORE:
            break
        selected.append(best)
        selected_pairs.append((best.symbol, best.direction))
        remaining = remaining[1:]
        rescored = []
        for o in remaining:
            new_corr_pen = correlation_penalty(o.symbol, o.direction, selected_pairs, correlation_matrix)
            extra = max(0.0, new_corr_pen - o.correlation_penalty)
            rescored.append(replace(o, score=o.score - extra, correlation_penalty=new_corr_pen))
        remaining = sorted(rescored, key=lambda o: o.score, reverse=True)
    return selected


def rolling_correlation_matrix(returns_by_symbol: Dict[str, List[float]]) -> Dict[Tuple[str, str], float]:
    """Rolling Pearson correlation of recent bar-to-bar returns between
    each symbol pair. Symmetric; only (a, b) with a < b (lexicographic) is
    stored -- callers/consumers must check both orderings, which
    correlation_penalty() above already does."""
    symbols = sorted(returns_by_symbol.keys())
    out: Dict[Tuple[str, str], float] = {}
    for a, b in itertools.combinations(symbols, 2):
        ra, rb = returns_by_symbol[a], returns_by_symbol[b]
        n = min(len(ra), len(rb))
        if n < 10:
            out[(a, b)] = 0.0
            continue
        ra_arr, rb_arr = np.array(ra[-n:]), np.array(rb[-n:])
        if ra_arr.std() == 0 or rb_arr.std() == 0:
            out[(a, b)] = 0.0
            continue
        out[(a, b)] = float(np.corrcoef(ra_arr, rb_arr)[0, 1])
    return out
