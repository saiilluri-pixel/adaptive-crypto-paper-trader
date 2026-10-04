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

# Paper-exploration eligibility (cold-start-deadlock fix): a brand-new
# (symbol, strategy, regime) cell has shrunk_expectancy pinned to EXACTLY
# the global prior (0.0 by default) until it has at least one live trade,
# since (n/(n+k))*raw_mean vanishes at n=0 regardless of raw_mean. That
# zeroes the score formula's positive multiplicative term while every
# penalty term stays strictly positive (friction alone is always > 0), so
# NO signal -- however strong -- can ever clear MIN_QUALITY_SCORE for a
# cell with zero trades. Confirmed both mathematically and against real
# production decisions.jsonl history: 6 genuine mean_reversion BUY signals
# fired live across BTC/ETH/SOL and every one scored exactly the same
# deterministic floor (shrunk_edge=0 forces the same negative value
# regardless of which symbol or how strong that specific signal was),
# never reaching the ranked `selected` list. is_exploration_eligible()
# below is the escape hatch: a signal strong enough and a cell with truly
# zero history may take ONE small, tightly-risk-capped position (see
# risk_engine.size_exploration_entry, EXPLORATION_RISK_PER_TRADE_PCT) to
# acquire that first live observation -- once n>=1, shrinkage naturally
# takes over and normal scoring governs every subsequent decision for
# that cell, including whether to ever trade it again.
#
# Lowered 0.5 -> 0.35 in response to real trade-frequency feedback after 9
# days live, then 0.35 -> 0.25 per explicit user request for aggressive
# trading -- admits weaker signals into the exploration path. Paired with
# confidence-scaled exploration risk sizing (risk_engine
# .EXPLORATION_RISK_MIN_PCT/MAX_PCT, also raised for the same request) so
# admitting weaker signals doesn't mean risking the same amount on them as
# strong ones.
MIN_EXPLORATION_SIGNAL_STRENGTH = 0.25


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
    is_exploration_eligible: bool = False  # cell.n == 0 and signal.strength cleared the exploration floor


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
    exploration_eligible = cell.n == 0 and signal.strength >= MIN_EXPLORATION_SIGNAL_STRENGTH

    return Opportunity(symbol=symbol, strategy=signal.strategy, regime=regime,
                        direction=signal.direction, score=score, shrunk_edge=shrunk_edge,
                        signal_strength=signal.strength, regime_fit=rfit, robustness=robust,
                        friction_penalty=friction_penalty_pct, uncertainty_penalty=unc_pen,
                        drawdown_penalty=dd_pen, correlation_penalty=corr_pen,
                        stop_pct=signal.stop_pct, expected_rr=signal.expected_rr,
                        confidence=1.0 - uncertainty, is_exploration_eligible=exploration_eligible)


def score_short_opportunity(*, symbol: str, signal, regime: str, stats_store,
                             friction_penalty_pct: float, already_selected: List[Tuple[str, str]],
                             correlation_matrix: Dict[Tuple[str, str], float]) -> Optional[Opportunity]:
    """SIMULATED SHORT scoring (adaptive/short_portfolio.py) -- structurally
    identical to score_opportunity(), just fed by the bearish side of a
    signal. A StrategySignal only carries bearish_research_strength when
    direction=="exit_long" (a bearish reading); this is the one place in
    the codebase that treats that number as an ACTIVE short-entry signal
    rather than research-only context, since every strategy already
    computes it causally and it was explicitly built to support this. Uses
    a SEPARATE stats_store (short-side empirical-Bayes evidence is
    genuinely different from the long side's) but the SAME regime_fit/
    robustness/penalty machinery as the long side. Returns None if this
    signal has no bearish reading to score."""
    if signal.direction != "exit_long" or signal.bearish_research_strength <= 0:
        return None
    cell = stats_store.get(symbol, signal.strategy, regime)
    shrunk_edge = stats_store.shrunk_expectancy(symbol, signal.strategy, regime)
    uncertainty = stats_store.uncertainty(symbol, signal.strategy, regime)
    rfit = regime_fit_score(signal.regime_compatibility, regime)
    robust = robustness_multiplier(cell)
    unc_pen = uncertainty * UNCERTAINTY_PENALTY_COEF
    dd_pen = cell.max_dd_pct * DRAWDOWN_PENALTY_COEF
    corr_pen = correlation_penalty(symbol, "short", already_selected, correlation_matrix)

    strength = signal.bearish_research_strength
    score = (shrunk_edge * strength * rfit * robust
             - friction_penalty_pct - unc_pen - dd_pen - corr_pen)
    exploration_eligible = cell.n == 0 and strength >= MIN_EXPLORATION_SIGNAL_STRENGTH

    return Opportunity(symbol=symbol, strategy=signal.strategy, regime=regime,
                        direction="short", score=score, shrunk_edge=shrunk_edge,
                        signal_strength=strength, regime_fit=rfit, robustness=robust,
                        friction_penalty=friction_penalty_pct, uncertainty_penalty=unc_pen,
                        drawdown_penalty=dd_pen, correlation_penalty=corr_pen,
                        stop_pct=signal.stop_pct if signal.stop_pct > 0 else 2.0,
                        expected_rr=signal.expected_rr if signal.expected_rr > 0 else 1.5,
                        confidence=1.0 - uncertainty, is_exploration_eligible=exploration_eligible)


def rank_and_select(candidates: List[Opportunity],
                     correlation_matrix: Dict[Tuple[str, str], float],
                     max_positions: int = 3, direction: str = "long") -> List[Opportunity]:
    """direction: "long" (default, Spot) or "short" (SIMULATED margin,
    adaptive/short_portfolio.py) -- the two are always ranked/selected
    independently via separate calls, never mixed into one list, since
    they draw on separate portfolios/risk budgets/capital pools."""
    remaining = sorted([c for c in candidates if c.direction == direction],
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
