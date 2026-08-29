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


STATE_RISK_MULTIPLIER = {
    DegradationState.NORMAL: 1.00,
    DegradationState.CAUTION: 0.50,
    DegradationState.DEFENSIVE: 0.25,
    DegradationState.HALTED: 0.0,
}
# "only highest-confidence entries" in DEFENSIVE; HALTED's floor is
# unreachable (>1.0) so no confidence value can pass it -- redundant with
# the risk multiplier being 0, but explicit rather than relying on that
# alone in case a caller ever forgets to check `approved`.
STATE_MIN_CONFIDENCE = {
    DegradationState.NORMAL: 0.0,
    DegradationState.CAUTION: 0.0,
    DegradationState.DEFENSIVE: 0.65,
    DegradationState.HALTED: 1.01,
}


@dataclass
class RiskLimits:
    risk_per_trade_pct: float = 0.50
    max_portfolio_heat_pct: float = 1.50
    max_symbol_allocation_pct: float = 40.0
    max_total_crypto_allocation_pct: float = 90.0
    daily_loss_limit_pct: float = 2.0
    max_drawdown_limit_pct: float = 10.0
    consecutive_loss_breaker: int = 5
    min_cash_reserve_pct: float = 10.0


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
                    confidence: float) -> SizingResult:
        limits = self.limits
        mult = STATE_RISK_MULTIPLIER[self.state]
        if mult <= 0:
            return SizingResult(False, 0.0, 0.0, "degradation_state",
                                 f"state={self.state.value} blocks new entries")
        if confidence < STATE_MIN_CONFIDENCE[self.state]:
            return SizingResult(False, 0.0, 0.0, "degradation_state_confidence",
                                 f"confidence {confidence:.2f} below {self.state.value} "
                                 f"floor {STATE_MIN_CONFIDENCE[self.state]:.2f}")
        if stop_distance_frac <= 0:
            return SizingResult(False, 0.0, 0.0, "invalid_stop", "stop_distance_frac must be positive")
        if equity <= 0:
            return SizingResult(False, 0.0, 0.0, "invalid_equity", "equity must be positive")

        risk_amount = equity * (limits.risk_per_trade_pct / 100.0) * mult
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
        if (recent_expectancy is not None and recent_expectancy < 0) or \
           (recent_pf is not None and recent_pf < 0.8):
            return DegradationState.DEFENSIVE
        if recent_pf is not None and recent_pf < 1.1:
            return DegradationState.CAUTION
        return DegradationState.NORMAL

    def set_state(self, state: DegradationState, reason: str = ""):
        self.state = state
        self.state_reason = reason
