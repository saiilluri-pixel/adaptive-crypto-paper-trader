"""
Phase 6, Stage 1: coarse hypothesis test.

Fixed/simple parameters, one run per (family, symbol) -- no grid search.
Reject obviously negative hypotheses quickly; only survivors proceed to
Stage 2 (parameter robustness) in research/stage2_robustness.py.

Timeframe: 15m (Phase 4's middle candidate -- less noisy than 5m, more
data-rich than 1h for a first pass). HTF context derived from 1h closes
(htf_bars maps a 15m-chart lookback to roughly a 1-day window).
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from research.data_cache import load_cached  # noqa: E402
from research.generic_runner import run_generic_backtest  # noqa: E402
from research.metrics import compute_metrics  # noqa: E402
from research.strategies.trend_momentum import DonchianTrendStrategy  # noqa: E402
from research.strategies.volatility_breakout import VolBreakoutStrategy  # noqa: E402
from research.strategies.mean_reversion import ZScoreReversionStrategy  # noqa: E402
from research.strategies.funding_filter import FundingContrarianStrategy  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TF = "15m"
HTF_BARS = 96   # 15m x 96 = 24h lookback for the trend-context filter


def _funding_thresholds(sym):
    f = pd.read_csv(os.path.join(HERE, "data", f"{sym.replace('/', '')}_funding.csv"))
    return float(f["funding_rate"].quantile(0.95)), float(f["funding_rate"].quantile(0.05)), f


FAMILIES = {
    "A_trend_donchian": (DonchianTrendStrategy, {"DONCHIAN_N": 20, "INIT_STOP_PCT": 3.0}),
    "B_vol_breakout": (VolBreakoutStrategy,
                       {"RANGE_N": 20, "ATR_N": 14, "ATR_EXPANSION_MULT": 1.5, "INIT_STOP_PCT": 3.0}),
    "C_mean_reversion": (ZScoreReversionStrategy, {"ZSCORE_N": 20, "Z_THRESH": 2.0, "INIT_STOP_PCT": 2.0}),
    "D_funding_contrarian": (FundingContrarianStrategy, {"INIT_STOP_PCT": 3.0}),  # thresholds set per-symbol below
}


def run_one(name, cls, params, sym, chart, funding_df=None):
    p = dict(params)
    htf_bars = HTF_BARS if name in ("A_trend_donchian", "B_vol_breakout", "C_mean_reversion") else None
    trades, eq, guards, blocked, intervals = run_generic_backtest(
        cls, p, chart, start_capital=config.START_CAPITAL, symbol=sym,
        htf_bars=htf_bars, funding_df=funding_df)
    m = compute_metrics(trades, eq, config.START_CAPITAL, guards, real_times=intervals)
    m["family"] = name
    m["symbol"] = sym
    m["params"] = p
    return m


def main():
    results = {}
    for name, (cls, params) in FAMILIES.items():
        results[name] = {}
        for sym in config.SYMBOLS:
            chart = load_cached(sym, TF)
            p = dict(params)
            funding_df = None
            if name == "D_funding_contrarian":
                hi, lo, funding_df = _funding_thresholds(sym)
                p["FUNDING_HIGH_THRESHOLD"] = hi
                p["FUNDING_LOW_THRESHOLD"] = lo
            m = run_one(name, cls, p, sym, chart, funding_df)
            results[name][sym] = m
            print(f"[{name}][{sym}] trades={m.get('total_trades')} "
                  f"ret={m.get('net_return_pct', 0):.2f}% pf={m.get('profit_factor')} "
                  f"sharpe={m.get('sharpe_ratio_annualized')}")

    os.makedirs(os.path.join(HERE, "reports"), exist_ok=True)
    with open(os.path.join(HERE, "reports", "stage1_coarse.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nwrote research/reports/stage1_coarse.json")


if __name__ == "__main__":
    main()
