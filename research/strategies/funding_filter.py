"""
Family D: FUNDING / BASIS-AWARE DIRECTIONAL FILTER

Tested as an INDEPENDENT hypothesis, not assumed to add edge, not combined
with another family initially, per the research brief: extreme perpetual
funding predicts a near-term reversal (contrarian) -- extreme positive
funding (longs crowded, paying shorts) -> short; extreme negative funding
(shorts crowded) -> long. Real Binance public funding-rate history (8h
Binance USDM perpetual cadence), forward-filled onto the chart timeframe --
never fabricated.
"""
from research.strategies.base import ResearchStrategy


class FundingContrarianStrategy(ResearchStrategy):
    """high_threshold/low_threshold: NOT symmetric -- observed Binance
    funding this year is capped near +0.0001 on the positive side for all
    three symbols but ranges much lower on the negative side (SOL to
    -0.003), so a single symmetric threshold would either never fire on the
    high side or almost always fire on the low side. Callers should pass
    each symbol's own distribution (e.g. p05/p95 of its funding history) --
    this is instrument-scale calibration, not fitting to returns."""
    def __init__(self, params):
        super().__init__(params)
        self.high_threshold = float(params.get("FUNDING_HIGH_THRESHOLD", 0.0001))
        self.low_threshold = float(params.get("FUNDING_LOW_THRESHOLD", -0.0001))
        self._last_bucket = None   # avoid re-firing every bar while funding stays extreme

    def on_bar(self, o, h, l, c, ts_ms, htf_trend=None, funding_rate=None):
        if funding_rate is None:
            return None
        bucket = "high" if funding_rate >= self.high_threshold else (
            "low" if funding_rate <= self.low_threshold else "mid")
        sig = None
        if bucket != self._last_bucket:
            if bucket == "high":
                sig = "short"     # fade crowded longs
            elif bucket == "low":
                sig = "long"      # fade crowded shorts
        self._last_bucket = bucket
        return sig
