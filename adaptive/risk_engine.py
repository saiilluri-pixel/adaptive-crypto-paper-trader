"""
Portfolio-level risk policy layer, sitting above adaptive/portfolio.py's
hard Spot-legal invariants (portfolio.py refuses to violate cash/inventory
reality if asked anyway; this module is what should normally prevent ever
asking). Pure-function style: callers pass in the numbers it needs (equity,
cash, current heat, current crypto exposure) rather than this module
reaching into Portfolio directly, so it stays independently testable.

Position sizing (adaptive spec section 4): risk_amount = equity *
RISK_PER_TRADE_PCT/100, scaled by the current degradation state's risk
multiplier; notional = risk_amount / stop_distance_frac; then capped by
whichever of {available cash minus the reserve floor, this symbol's max
allocation, remaining portfolio-heat budget, remaining total-crypto-exposure
budget} is tightest. These are ceilings, never allocation targets -- the
system is never required to spend up to any of them.

Degradation state machine (section 15): NORMAL/CAUTION/DEFENSIVE/HALTED,
driven by rolling live-performance evidence, reduces risk or blocks new
entries BEFORE any blind reparameterization is even considered. State never
affects managing an existing position's exit -- only new-entry eligibility
and sizing.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class DegradationState(Enum):
    NORMAL = "NORMAL"
    CAUTION = "CAUTION"
    DEFENSIVE = "DEFENSIVE"
    HALTED = "HALTED"


# CAUTION/DEFENSIVE multipliers raised (0.50->0.75, 0.25->0.60) per explicit
# user request for more aggressive trading -- both books were observed
# sitting in DEFENSIVE/CAUTION off small, noisy early sample sizes (10-20
# trades), which was throttling position size to a quarter of normal right
# when "aggressive" was asked for. HALTED stays 0.0 -- that state exists to
# fully stop new entries after a real breaker trip (max drawdown/daily
# loss/loss streak, see RiskLimits below), not to be softened.
STATE_RISK_MULTIPLIER = {
    DegradationState.NORMAL: 1.00,
    DegradationState.CAUTION: 0.75,
    DegradationState.DEFENSIVE: 0.60,
    DegradationState.HALTED: 0.0,
}
# "only highest-confidence entries" in DEFENSIVE; HALTED's floor is
# unreachable (>1.0) so no confidence value can pass it -- redundant with
# the risk multiplier being 0, but explicit rather than relying on that
# alone in case a caller ever forgets to check `approved`. DEFENSIVE floor
# lowered 0.65->0.45 alongside the multiplier change above, same reason.
STATE_MIN_CONFIDENCE = {
    DegradationState.NORMAL: 0.0,
    DegradationState.CAUTION: 0.0,
    DegradationState.DEFENSIVE: 0.45,
    DegradationState.HALTED: 1.01,
}

# Paper-exploration sizing (cold-start-deadlock fix): smaller than the
# normal 0.50% risk_per_trade_pct. Capped at MAX_EXPLORATION_POSITIONS
# concurrent exploration positions system-wide -- see size_exploration_entry()
# and meta_controller.is_exploration_eligible(). Raised from 3 to 5 (one
# per symbol, matching MAX_POSITIONS in runner.py, now 5 symbols) so each
# of BTC/ETH/SOL/BNB/XRP can independently acquire its first live
# observation in parallel rather than serially -- Portfolio.buy()'s own
# one-position-per-symbol rule is still the hard ceiling on how many of
# those 5 slots can ever be occupied at once. NOTE: this same constant
# gates BOTH books independently (adaptive/runner.py tracks
# exploration_symbols and short_exploration_symbols separately), so the
# true system-wide ceiling on concurrent exploration positions is now 10
# (5 long + 5 short), not 5.
EXPLORATION_RISK_PER_TRADE_PCT = 0.10  # kept as the historical/documented ceiling value
MAX_EXPLORATION_POSITIONS = 5

# Confidence-scaled exploration risk: a signal right at the eligibility
# floor (MIN_EXPLORATION_SIGNAL_STRENGTH in meta_controller.py) risks the
# MIN fraction; a maximum-strength (1.0) signal risks the MAX fraction.
# Raised 0.08/0.15 -> 0.20/0.40 per explicit user request for aggressive
# trading -- a further deliberate, disclosed increase in per-trade
# exploration risk, not just trade count. Still below the normal (now
# aggressive) 1.75% risk_per_trade_pct: exploration positions remain
# smaller than normally-scored ones even after this increase.
EXPLORATION_RISK_MIN_PCT = 0.20
EXPLORATION_RISK_MAX_PCT = 0.40


@dataclass
class RiskLimits:
    # Aggressive-trading tier per explicit user request (2026-09-17),
    # roughly 3-4x the prior conservative defaults:
    #   risk_per_trade_pct        0.50  -> 1.75
    #   max_portfolio_heat_pct    1.50  -> 9.00  (sized so ~5 concurrent
    #                                              full-risk positions,
    #                                              MAX_POSITIONS in
    #                                              runner.py, don't
    #                                              themselves become the
    #                                              binding constraint --
    #                                              5 * 1.75% ~= 8.75%)
    #   max_symbol_allocation_pct 40.0  -> 55.0
    #   max_total_crypto_allocation_pct 90.0 -> 95.0
    #   daily_loss_limit_pct      2.0   -> 6.0   (breaker loosened too,
    #                                              per explicit request)
    #   max_drawdown_limit_pct    10.0  -> 25.0  (same)
    #   consecutive_loss_breaker  5     -> 8     (same)
    #   min_cash_reserve_pct      10.0  -> 5.0   (allow more of the pool
    #                                              to be deployed)
    risk_per_trade_pct: float = 1.75
    max_portfolio_heat_pct: float = 9.00
    max_symbol_allocation_pct: float = 55.0
    max_total_crypto_allocation_pct: float = 95.0
    daily_loss_limit_pct: float = 6.0
    max_drawdown_limit_pct: float = 25.0
    consecutive_loss_breaker: int = 8
    min_cash_reserve_pct: float = 5.0


@dataclass
class SizingResult:
    approved: bool
    notional_usdt: float
    risk_amount_usdt: float
    binding_constraint: str
    reason: str = ""


class RiskEngine:
    def __init__(self, limits: Optional[RiskLimits] = None):
        self.limits = limits or RiskLimits()
        self.state = DegradationState.NORMAL
        self.state_reason = ""

    # ── position sizing ──────────────────────────────────────────────
    def size_entry(self, *, equity: float, cash: float, current_portfolio_heat_usdt: float,
                    current_crypto_value_usdt: float, stop_distance_frac: float,
                    confidence: float, risk_pct_override: Optional[float] = None,
                    enforce_confidence_floor: bool = True) -> SizingResult:
        """risk_pct_override: used only by size_exploration_entry() below to
        substitute the exploration risk fraction for the normal
        risk_per_trade_pct.

        enforce_confidence_floor: normally True. size_exploration_entry()
        passes False, because an EXPLORATION entry is by construction a
        first-observation trade on a never-traded (n=0) cell, whose
        confidence (= 1 - uncertainty) is ~0.0 -- so the degradation-state
        confidence floor (meant to restrict CONVICTION-SCORED normal
        entries during a drawdown) would reject EVERY exploration entry the
        moment the book enters DEFENSIVE/CAUTION, silently freezing the
        data-gathering mechanism exactly when the book is stuck and most
        needs it (observed live: 304 consecutive exploration rejections,
        'confidence 0.00 below DEFENSIVE floor 0.45', book pinned in
        DEFENSIVE). Exploration still respects the HALTED hard-stop (risk
        multiplier 0 -> not approved, checked below) and the state's risk
        multiplier (0.60 in DEFENSIVE), so it stays tiny -- it just isn't
        gated by a conviction threshold it can never meet by design."""
        limits = self.limits
        mult = STATE_RISK_MULTIPLIER[self.state]
        if mult <= 0:
            return SizingResult(False, 0.0, 0.0, "degradation_state",
                                 f"state={self.state.value} blocks new entries")
        if enforce_confidence_floor and confidence < STATE_MIN_CONFIDENCE[self.state]:
            return SizingResult(False, 0.0, 0.0, "degradation_state_confidence",
                                 f"confidence {confidence:.2f} below {self.state.value} "
                                 f"floor {STATE_MIN_CONFIDENCE[self.state]:.2f}")
        if stop_distance_frac <= 0:
            return SizingResult(False, 0.0, 0.0, "invalid_stop", "stop_distance_frac must be positive")
        if equity <= 0:
            return SizingResult(False, 0.0, 0.0, "invalid_equity", "equity must be positive")

        risk_pct = risk_pct_override if risk_pct_override is not None else limits.risk_per_trade_pct
        risk_amount = equity * (risk_pct / 100.0) * mult
        notional_from_risk = risk_amount / stop_distance_frac

        caps = {"risk_budget": notional_from_risk}
        # min_cash_reserve is an approximation (equity includes crypto
        # already held, so this doesn't perfectly solve for post-trade
        # cash) -- conservative and simple, adequate for a paper system.
        caps["cash_reserve"] = max(0.0, cash - equity * (limits.min_cash_reserve_pct / 100.0))
        caps["symbol_allocation"] = equity * (limits.max_symbol_allocation_pct / 100.0)
        remaining_heat_usdt = max(0.0, equity * (limits.max_portfolio_heat_pct / 100.0)
                                   - current_portfolio_heat_usdt)
        caps["portfolio_heat"] = remaining_heat_usdt / stop_distance_frac
        remaining_exposure_usdt = max(0.0, equity * (limits.max_total_crypto_allocation_pct / 100.0)
                                       - current_crypto_value_usdt)
        caps["total_exposure"] = remaining_exposure_usdt

        binding = min(caps, key=caps.get)
        notional = caps[binding]
        if notional <= 1e-9:
            return SizingResult(False, 0.0, 0.0, binding, f"{binding} cap is zero/negative")
        actual_risk = notional * stop_distance_frac
        return SizingResult(True, notional, actual_risk, binding)

    def size_exploration_entry(self, *, equity: float, cash: float, current_portfolio_heat_usdt: float,
                                current_crypto_value_usdt: float, stop_distance_frac: float,
                                confidence: float, signal_strength: float) -> SizingResult:
        """Tightly-bounded PAPER EXPLORATION sizing (cold-start-deadlock
        fix -- see meta_controller.is_exploration_eligible() for the
        eligibility gate this is paired with). Otherwise identical to
        size_entry(): same degradation-state gate, same confidence floor,
        same four caps (cash reserve, symbol allocation, portfolio heat,
        total exposure). Exploration never bypasses basic risk management;
        it only accepts a smaller, deliberately-priced position to acquire
        the FIRST live observation for a (symbol, strategy, regime) cell
        that empirical-Bayes shrinkage would otherwise keep at exactly
        zero expected edge forever, since a cell with zero trades can
        never produce a nonzero shrunk_expectancy for shrinkage to update
        from.

        risk_pct is CONFIDENCE-SCALED between EXPLORATION_RISK_MIN_PCT (at
        signal_strength == the eligibility floor) and EXPLORATION_RISK_MAX_PCT
        (at signal_strength == 1.0) -- not a flat rate -- so that admitting
        weaker signals (a lower eligibility floor) doesn't also mean
        risking the same amount on them as on strong ones."""
        from adaptive.meta_controller import MIN_EXPLORATION_SIGNAL_STRENGTH
        floor = MIN_EXPLORATION_SIGNAL_STRENGTH
        span = max(1.0 - floor, 1e-9)
        frac = min(max((signal_strength - floor) / span, 0.0), 1.0)
        risk_pct = EXPLORATION_RISK_MIN_PCT + (EXPLORATION_RISK_MAX_PCT - EXPLORATION_RISK_MIN_PCT) * frac

        result = self.size_entry(equity=equity, cash=cash,
                                  current_portfolio_heat_usdt=current_portfolio_heat_usdt,
                                  current_crypto_value_usdt=current_crypto_value_usdt,
                                  stop_distance_frac=stop_distance_frac, confidence=confidence,
                                  risk_pct_override=risk_pct, enforce_confidence_floor=False)
        if result.approved:
            result.binding_constraint = "exploration:" + result.binding_constraint
        return result

    # ── portfolio-level breakers (mirror PaperEngine._risk_guard_blocked) ──
    def check_breakers(self, *, equity: float, peak_equity: float, daily_start_equity: float,
                        consecutive_losses: int) -> Optional[str]:
        """Returns a block reason if new entries should be halted, else
        None. Never affects managing an already-open position's exit."""
        limits = self.limits
        if peak_equity > 0:
            dd = (peak_equity - equity) / peak_equity * 100.0
            if dd >= limits.max_drawdown_limit_pct:
                return f"max drawdown {dd:.2f}% >= {limits.max_drawdown_limit_pct:.2f}%"
        if daily_start_equity > 0:
            daily_loss = (daily_start_equity - equity) / daily_start_equity * 100.0
            if daily_loss >= limits.daily_loss_limit_pct:
                return f"daily loss {daily_loss:.2f}% >= {limits.daily_loss_limit_pct:.2f}%"
        if consecutive_losses >= limits.consecutive_loss_breaker:
            return f"{consecutive_losses} consecutive losses >= {limits.consecutive_loss_breaker}"
        return None

    # ── degradation state machine ───────────────────────────────────
    def evaluate_degradation(self, *, drawdown_pct: float, loss_streak: int,
                              recent_expectancy: Optional[float],
                              recent_pf: Optional[float]) -> DegradationState:
        """Rolling-evidence-based state selection. Thresholds are
        deliberately conservative fixed defaults for a paper system, not
        fit to any data. Insufficient sample (None) never triggers a worse
        state -- it just means this signal is skipped, not treated as bad."""
        limits = self.limits
        if drawdown_pct >= limits.max_drawdown_limit_pct or loss_streak >= limits.consecutive_loss_breaker:
            return DegradationState.HALTED
        # Aggressive-profile recalibration (2026-09-20, per explicit user
        # request): the prior thresholds sent the book to full DEFENSIVE on
        # ANY negative recent expectancy (or PF < 0.8), which pinned it there
        # ~53% of cycles and, combined with the confidence floor, froze all
        # new entries. Now only a DECISIVELY losing book (profit factor
        # < 0.6 -- losses ~1.7x wins) goes DEFENSIVE; a mildly negative /
        # breakeven book sits in CAUTION (0.75x sizing) instead of DEFENSIVE
        # (0.60x), keeping more capital deployed. The hard breakers above
        # (drawdown, loss-streak -> HALTED) are unchanged -- this only moves
        # the softer degradation tiers. Tradeoff: the book will keep betting
        # through moderate drawdowns rather than throttling early. That is
        # what "aggressive" means and is the explicit, disclosed intent.
        if recent_pf is not None and recent_pf < 0.6:
            return DegradationState.DEFENSIVE
        if recent_pf is not None and recent_pf < 1.0:
            return DegradationState.CAUTION
        return DegradationState.NORMAL

    def set_state(self, state: DegradationState, reason: str = ""):
        self.state = state
        self.state_reason = reason
