"""
Hypothesis E: LIQUIDATION / VOLATILITY AFTERSHOCK

No proprietary liquidation feed available -- tested via observable candle
extremity instead: a bar whose true range, ATR-normalized, exceeds a
PRE-DECLARED threshold (not fit to performance) marks a "shock" bar.
Continuation and reversion are separate, mutually exclusive strategy
classes (never combined into one adaptive strategy), per the brief.
"""
from collections import deque

from research.strategies.base import ResearchStrategy


class _ShockBase(ResearchStrategy):
    def __init__(self, params):
        super().__init__(params)
        self.atr_n = int(params.get("ATR_N", 14))
        self.shock_mult = float(params.get("SHOCK_ATR_MULT", 3.0))   # pre-declared: 3x normal range
        self.tr_hist = deque(maxlen=self.atr_n)
        self.prev_close = None

    def _shock_direction(self, o, h, l, c):
        """Returns +1 (up shock), -1 (down shock), or None. Must be called
        exactly once per bar, in order (updates internal ATR state)."""
        tr = (h - l) if self.prev_close is None else max(
            h - l, abs(h - self.prev_close), abs(l - self.prev_close))
        direction = None
        if len(self.tr_hist) == self.atr_n:
            atr = sum(self.tr_hist) / len(self.tr_hist)
            if atr > 0 and tr > self.shock_mult * atr:
                direction = 1 if c > o else (-1 if c < o else None)
        self.tr_hist.append(tr)
        self.prev_close = c
        return direction


class ShockContinuationStrategy(_ShockBase):
    """After a shock bar, trade WITH its direction (momentum continuation)."""
    def on_bar(self, o, h, l, c, ts_ms, htf_trend=None, funding_rate=None):
        d = self._shock_direction(o, h, l, c)
        if d == 1:
            return "long"
        if d == -1:
            return "short"
        return None


class ShockReversionStrategy(_ShockBase):
    """After a shock bar, trade AGAINST its direction (mean reversion)."""
    def on_bar(self, o, h, l, c, ts_ms, htf_trend=None, funding_rate=None):
        d = self._shock_direction(o, h, l, c)
        if d == 1:
            return "short"
        if d == -1:
            return "long"
        return None
