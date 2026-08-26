"""
Hypothesis C.C: FUNDING PERSISTENCE

Opposite hypothesis to the V2/V3 contrarian filter: sustained one-sided
funding (K consecutive periods on the same side) reflects a real, ongoing
directional flow imbalance that is more likely to continue than reverse --
"follow the crowd that's been paying," rather than fade it.
"""
from research.strategies.base import ResearchStrategy


class FundingPersistenceStrategy(ResearchStrategy):
    def __init__(self, params):
        super().__init__(params)
        self.threshold = float(params.get("FUNDING_PERSIST_THRESHOLD", 0.00005))
        self.k = int(params.get("PERSIST_PERIODS", 3))
        self._streak_side = None
        self._streak_len = 0
        self._last_rate = None
        self._fired_this_streak = False

    def on_bar(self, o, h, l, c, ts_ms, htf_trend=None, funding_rate=None):
        if funding_rate is None:
            return None
        side = "high" if funding_rate > self.threshold else (
            "low" if funding_rate < -self.threshold else None)

        if funding_rate != self._last_rate:   # funding only updates every ~8h; count PRINTS not bars
            if side == self._streak_side and side is not None:
                self._streak_len += 1
            else:
                self._streak_side = side
                self._streak_len = 1 if side is not None else 0
                self._fired_this_streak = False
            self._last_rate = funding_rate

        sig = None
        if self._streak_len >= self.k and not self._fired_this_streak:
            sig = "long" if self._streak_side == "high" else "short"   # follow the crowded side
            self._fired_this_streak = True
        return sig
