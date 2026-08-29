"""
Champion/challenger self-adaptation (adaptive spec section 13-14).

A champion is the currently-active parameter set for one strategy;
challengers are alternative parameter sets under evaluation by
adaptive/research_worker.py. shock_continuation deliberately has no
tunable params and is never a candidate here -- it reuses the already-
frozen, already-validated shock definition from the SOL forward-shadow
work, and reparameterizing an already-frozen definition would contradict
everything established about that strategy elsewhere in this codebase.

PROMOTION CRITERIA (ALL required -- never total return alone):
  - adequate sample size (>= MIN_PROMOTION_SAMPLE trades)
  - positive validation expectancy (net of the same friction model used
    everywhere else in this system -- trade pnl%% figures are assumed
    already cost-net by the caller)
  - profit factor strictly greater than the champion's
  - drawdown not materially worse (challenger max DD <= champion max DD *
    DD_TOLERANCE; a champion with zero recorded DD is treated as having no
    ceiling on this check, since "not materially worse than zero" is
    meaningless as a ratio)
  - improvement holds across multiple windows (>= MIN_WINDOWS_AGREEING_FRAC
    of evaluation windows must individually show positive expectancy)
  - not dependent on one extreme trade (removing the single best trade
    must not flip the challenger from profitable to unprofitable)

SAFETY RAILS (section 14):
  - minimum promotion interval per strategy (prevents noisy oscillation)
  - a promotion applies to NEW entries only -- existing Position objects
    (adaptive/portfolio.py) already carry their own entry-time stop/risk/
    exit contract fixed at creation; nothing here ever reaches into an
    open position
  - rollback restores the immediately-previous champion
  - every promotion AND rejection is appended to adaptation_log.jsonl
"""
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

MIN_PROMOTION_SAMPLE = 30
MIN_WINDOWS_AGREEING_FRAC = 0.6
DD_TOLERANCE = 1.2
MIN_PROMOTION_INTERVAL_SEC = 24 * 3600

NON_ADAPTIVE_STRATEGIES = ("shock_continuation",)  # frozen -- never reparameterized


@dataclass
class WindowResult:
    window_id: str
    trades: List[float] = field(default_factory=list)  # net %-return per trade, cost-inclusive

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def expectancy_pct(self) -> float:
        return sum(self.trades) / len(self.trades) if self.trades else 0.0

    @property
    def max_dd_pct(self) -> float:
        cum = peak = dd = 0.0
        for t in self.trades:
            cum += t
            peak = max(peak, cum)
            dd = max(dd, peak - cum)
        return dd


@dataclass
class EvaluationResult:
    strategy: str
    label: str
    params: dict
    windows: List[WindowResult] = field(default_factory=list)

    @property
    def all_trades(self) -> List[float]:
        return [t for w in self.windows for t in w.trades]

    @property
    def n_trades(self) -> int:
        return len(self.all_trades)

    @property
    def expectancy_pct(self) -> float:
        trades = self.all_trades
        return sum(trades) / len(trades) if trades else 0.0

    @property
    def profit_factor(self) -> Optional[float]:
        trades = self.all_trades
        wins = sum(t for t in trades if t > 0)
        losses = sum(abs(t) for t in trades if t <= 0)
        if losses <= 0:
            return None if wins == 0 else float("inf")
        return wins / losses

    @property
    def max_dd_pct(self) -> float:
        return max((w.max_dd_pct for w in self.windows), default=0.0)


def params_hash(params: dict) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]


def evaluate_promotion(champion: EvaluationResult, challenger: EvaluationResult) -> Tuple[bool, Dict[str, str]]:
    reasons: Dict[str, str] = {}

    if challenger.n_trades < MIN_PROMOTION_SAMPLE:
        reasons["sample_size"] = f"{challenger.n_trades} < required {MIN_PROMOTION_SAMPLE}"

    if challenger.expectancy_pct <= 0:
        reasons["expectancy"] = f"{challenger.expectancy_pct:.4f}% <= 0"

    champ_pf = champion.profit_factor
    chal_pf = challenger.profit_factor
    champ_pf_val = champ_pf if champ_pf is not None and champ_pf != float("inf") else 0.0
    chal_pf_val = chal_pf if chal_pf is not None else 0.0
    if chal_pf_val <= champ_pf_val:
        reasons["profit_factor"] = f"challenger PF {chal_pf_val:.3f} <= champion PF {champ_pf_val:.3f}"

    if champion.max_dd_pct > 0 and challenger.max_dd_pct > champion.max_dd_pct * DD_TOLERANCE:
        reasons["drawdown"] = (f"challenger max DD {challenger.max_dd_pct:.3f}% > "
                                f"champion {champion.max_dd_pct:.3f}% x {DD_TOLERANCE} tolerance")

    if challenger.windows:
        agreeing = sum(1 for w in challenger.windows if w.expectancy_pct > 0)
        frac = agreeing / len(challenger.windows)
        if frac < MIN_WINDOWS_AGREEING_FRAC:
            reasons["window_consistency"] = f"{frac:.2f} of windows agree < required {MIN_WINDOWS_AGREEING_FRAC}"
    else:
        reasons["window_consistency"] = "no evaluation windows supplied"

    trades = challenger.all_trades
    if trades:
        best = max(trades)
        rest = list(trades)
        rest.remove(best)
        exp_without_best = sum(rest) / len(rest) if rest else 0.0
        if exp_without_best <= 0:
            reasons["outlier_dependence"] = "profitable only due to a single best trade"

    return (len(reasons) == 0), reasons


class AdaptationEngine:
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.champions: Dict[str, dict] = {}
        self.champion_history: Dict[str, List[Optional[dict]]] = {}
        self.last_promotion_ts: Dict[str, float] = {}

    def get_champion_params(self, strategy: str, default_params: dict) -> dict:
        return self.champions.get(strategy, default_params)

    def consider_promotion(self, strategy: str, champion_eval: EvaluationResult,
                            challenger_eval: EvaluationResult, challenger_params: dict,
                            now_ts: Optional[float] = None,
                            min_interval: float = MIN_PROMOTION_INTERVAL_SEC) -> bool:
        if strategy in NON_ADAPTIVE_STRATEGIES:
            self._log(strategy, "rejected", {"reasons": {"frozen_strategy": "not eligible for adaptation"}})
            return False
        now_ts = now_ts if now_ts is not None else time.time()
        last = self.last_promotion_ts.get(strategy)  # None (not 0.0) means "never promoted" --
        # a 0.0 sentinel would incorrectly block a strategy's very first promotion whenever
        # now_ts is a small/test-relative value rather than a real epoch timestamp
        if last is not None and now_ts - last < min_interval:
            self._log(strategy, "rejected",
                       {"reasons": {"min_interval": f"{now_ts - last:.0f}s < {min_interval:.0f}s"}})
            return False

        ok, reasons = evaluate_promotion(champion_eval, challenger_eval)
        if not ok:
            self._log(strategy, "rejected", {"reasons": reasons,
                                               "challenger_hash": params_hash(challenger_params)})
            return False

        old = self.champions.get(strategy)
        self.champion_history.setdefault(strategy, []).append(old)
        self.champions[strategy] = challenger_params
        self.last_promotion_ts[strategy] = now_ts
        self._log(strategy, "promoted", {
            "old_params": old, "new_params": challenger_params,
            "old_hash": params_hash(old) if old else None,
            "new_hash": params_hash(challenger_params),
            "champion_metrics": {"n_trades": champion_eval.n_trades,
                                  "expectancy_pct": champion_eval.expectancy_pct,
                                  "profit_factor": champion_eval.profit_factor,
                                  "max_dd_pct": champion_eval.max_dd_pct},
            "challenger_metrics": {"n_trades": challenger_eval.n_trades,
                                    "expectancy_pct": challenger_eval.expectancy_pct,
                                    "profit_factor": challenger_eval.profit_factor,
                                    "max_dd_pct": challenger_eval.max_dd_pct},
        })
        return True

    def rollback(self, strategy: str) -> bool:
        hist = self.champion_history.get(strategy, [])
        if not hist:
            return False
        prev = hist.pop()
        current = self.champions.get(strategy)
        self.champions[strategy] = prev
        self._log(strategy, "rollback", {"restored_params": prev, "restored_from": current})
        return True

    def _log(self, strategy: str, action: str, detail: dict):
        record = {"ts_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "strategy": strategy, "action": action, **detail}
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def to_dict(self) -> dict:
        return {"champions": self.champions,
                "champion_history": self.champion_history,
                "last_promotion_ts": self.last_promotion_ts}

    @classmethod
    def from_dict(cls, d: dict, log_path: str) -> "AdaptationEngine":
        eng = cls(log_path)
        eng.champions = dict((d or {}).get("champions", {}))
        eng.champion_history = dict((d or {}).get("champion_history", {}))
        eng.last_promotion_ts = dict((d or {}).get("last_promotion_ts", {}))
        return eng
