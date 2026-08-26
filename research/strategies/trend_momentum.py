"""
Family A: TREND / MOMENTUM

Hypothesis: crypto has persistent directional moves and momentum after
meaningful breakouts.

Transparent, unstacked components: Donchian-style breakout of the prior
N-bar range, gated by a higher-timeframe trend filter (only take breakouts
in the direction of the HTF trend). Exit via the shared ATR-scaled trailing
stop (params["INIT_STOP_PCT"]).
"""
from collections import deque

from research.strategies.base import ResearchStrategy


class DonchianTrendStrategy(ResearchStrategy):
    def __init__(self, params):
        super().__init__(params)
        self.n = int(params.get("DONCHIAN_N", 20))
        self.highs = deque(maxlen=self.n)
        self.lows = deque(maxlen=self.n)

    def on_bar(self, o, h, l, c, ts_ms, htf_trend=None, funding_rate=None):
        sig = None
        if len(self.highs) == self.n:
            prior_high = max(self.highs)   # last N CLOSED bars, not including this one -- no lookahead
            prior_low = min(self.lows)
            if c > prior_high and (htf_trend is None or htf_trend >= 0):
                sig = "long"
            elif c < prior_low and (htf_trend is None or htf_trend <= 0):
                sig = "short"
        self.highs.append(h)
        self.lows.append(l)
        return sig
