"""
Rolling per (symbol, strategy, regime) live-paper trade statistics with
empirical-Bayes shrinkage toward a global prior mean, so a cell with 2
trades can't dominate ranking the way a cell with 200 trades would. This
is the evidence base meta_controller.py's opportunity scoring and
adaptation.py's champion/challenger promotion rules both read from.

Shrinkage: shrunk_expectancy = (n/(n+k))*sample_mean + (k/(n+k))*prior_mean,
a standard empirical-Bayes/James-Stein-flavored estimator. k is the
"weight of the prior" in equivalent trade count -- at n=0 the estimate is
entirely the prior; at n>>k it converges to the raw sample mean.
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

SHRINKAGE_K = 20.0


@dataclass
class CellStats:
    symbol: str
    strategy: str
    regime: str
    trades: List[float] = field(default_factory=list)  # net %-return per trade
    wins: int = 0
    losses: int = 0
    gross_win: float = 0.0
    gross_loss: float = 0.0
    max_dd_pct: float = 0.0
    _cum: float = field(default=0.0, repr=False)
    _peak: float = field(default=0.0, repr=False)

    def record(self, pnl_pct: float):
        self.trades.append(pnl_pct)
        if pnl_pct > 0:
            self.wins += 1
            self.gross_win += pnl_pct
        else:
            self.losses += 1
            self.gross_loss += abs(pnl_pct)
        self._cum += pnl_pct
        self._peak = max(self._peak, self._cum)
        self.max_dd_pct = max(self.max_dd_pct, self._peak - self._cum)

    @property
    def n(self) -> int:
        return len(self.trades)

    @property
    def raw_mean(self) -> float:
        return sum(self.trades) / self.n if self.n else 0.0

    @property
    def raw_std(self) -> Optional[float]:
        if self.n < 2:
            return None
        m = self.raw_mean
        var = sum((t - m) ** 2 for t in self.trades) / (self.n - 1)
        return math.sqrt(var)

    @property
    def profit_factor(self) -> Optional[float]:
        if self.gross_loss <= 0:
            return None if self.gross_win == 0 else float("inf")
        return self.gross_win / self.gross_loss

    @property
    def win_rate(self) -> Optional[float]:
        return self.wins / self.n if self.n else None


class StatsStore:
    def __init__(self, shrinkage_k: float = SHRINKAGE_K, global_prior_mean: float = 0.0):
        self.cells: Dict[Tuple[str, str, str], CellStats] = {}
        self.shrinkage_k = shrinkage_k
        self.global_prior_mean = global_prior_mean

    def get(self, symbol: str, strategy: str, regime: str) -> CellStats:
        key = (symbol, strategy, regime)
        if key not in self.cells:
            self.cells[key] = CellStats(symbol, strategy, regime)
        return self.cells[key]

    def record_trade(self, symbol: str, strategy: str, regime: str, pnl_pct: float):
        self.get(symbol, strategy, regime).record(pnl_pct)

    def shrunk_expectancy(self, symbol: str, strategy: str, regime: str) -> float:
        cell = self.get(symbol, strategy, regime)
        n, k = cell.n, self.shrinkage_k
        return (n / (n + k)) * cell.raw_mean + (k / (n + k)) * self.global_prior_mean

    def uncertainty(self, symbol: str, strategy: str, regime: str) -> float:
        """0 (fully confident, huge sample) .. 1 (no data yet)."""
        cell = self.get(symbol, strategy, regime)
        n, k = cell.n, self.shrinkage_k
        return k / (n + k)

    def to_dict(self) -> dict:
        return {f"{s}|{st}|{r}": {
            "trades": c.trades, "wins": c.wins, "losses": c.losses,
            "gross_win": c.gross_win, "gross_loss": c.gross_loss,
            "max_dd_pct": c.max_dd_pct, "_cum": c._cum, "_peak": c._peak,
        } for (s, st, r), c in self.cells.items()}

    @classmethod
    def from_dict(cls, d: dict, shrinkage_k: float = SHRINKAGE_K, global_prior_mean: float = 0.0):
        store = cls(shrinkage_k=shrinkage_k, global_prior_mean=global_prior_mean)
        for key, v in (d or {}).items():
            symbol, strategy, regime = key.split("|", 2)
            c = store.get(symbol, strategy, regime)
            c.trades = list(v.get("trades", []))
            c.wins = v.get("wins", 0)
            c.losses = v.get("losses", 0)
            c.gross_win = v.get("gross_win", 0.0)
            c.gross_loss = v.get("gross_loss", 0.0)
            c.max_dd_pct = v.get("max_dd_pct", 0.0)
            c._cum = v.get("_cum", 0.0)
            c._peak = v.get("_peak", 0.0)
        return store
