"""
Family C: MEAN REVERSION

Hypothesis: short-horizon overextensions revert during non-trending regimes.

Volatility-normalized z-score deviation from a rolling mean, with a strict
higher-timeframe trend filter so the strategy does not repeatedly fade a
powerful trend (long only when HTF trend is not down; short only when HTF
trend is not up) -- deliberately the opposite gating direction from family
A, which requires trend alignment rather than trend absence.
"""
import statistics
from collections import deque

from research.strategies.base import ResearchStrategy


class ZScoreReversionStrategy(ResearchStrategy):
    def __init__(self, params):
        super().__init__(params)
        self.n = int(params.get("ZSCORE_N", 20))
        self.z_thresh = float(params.get("Z_THRESH", 2.0))
        self.closes = deque(maxlen=self.n)

    def on_bar(self, o, h, l, c, ts_ms, htf_trend=None, funding_rate=None):
        sig = None
        if len(self.closes) == self.n:
            mean = statistics.fmean(self.closes)
            std = statistics.pstdev(self.closes)
            if std > 0:
                z = (c - mean) / std
                if z < -self.z_thresh and (htf_trend is None or htf_trend >= 0):
                    sig = "long"     # oversold, HTF not in a strong downtrend
                elif z > self.z_thresh and (htf_trend is None or htf_trend <= 0):
                    sig = "short"    # overbought, HTF not in a strong uptrend
        self.closes.append(c)
        return sig
