"""
Family B: VOLATILITY BREAKOUT

Hypothesis: large crypto moves often follow volatility compression /
expansion -- a true-range spike relative to recent ATR, combined with a
level breakout, marks the start of a move (distinct from family A, which
fires on level breakout alone with no volatility-expansion requirement).

24/7 crypto has no "opening range" in the traditional sense, so "range" here
is a rolling N-bar high/low, not a session open.
"""
from collections import deque

from research.strategies.base import ResearchStrategy


class VolBreakoutStrategy(ResearchStrategy):
    def __init__(self, params):
        super().__init__(params)
        self.range_n = int(params.get("RANGE_N", 20))
        self.atr_n = int(params.get("ATR_N", 14))
        self.expansion_mult = float(params.get("ATR_EXPANSION_MULT", 1.5))
        self.highs = deque(maxlen=self.range_n)
        self.lows = deque(maxlen=self.range_n)
        self.tr_hist = deque(maxlen=self.atr_n)
        self.prev_close = None

    def on_bar(self, o, h, l, c, ts_ms, htf_trend=None, funding_rate=None):
        tr = (h - l) if self.prev_close is None else max(
            h - l, abs(h - self.prev_close), abs(l - self.prev_close))

        sig = None
        if len(self.tr_hist) == self.atr_n and len(self.highs) == self.range_n:
            atr = sum(self.tr_hist) / len(self.tr_hist)
            prior_high = max(self.highs)
            prior_low = min(self.lows)
            if tr > self.expansion_mult * atr:   # this bar itself is the expansion bar
                if c > prior_high and (htf_trend is None or htf_trend >= 0):
                    sig = "long"
                elif c < prior_low and (htf_trend is None or htf_trend <= 0):
                    sig = "short"

        self.tr_hist.append(tr)
        self.highs.append(h)
        self.lows.append(l)
        self.prev_close = c
        return sig
