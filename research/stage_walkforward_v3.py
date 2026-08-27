"""
V3 Stages 2-3: parameter robustness + rolling walk-forward, for generic_
runner strategies (E-continuation, C-persistence). Smaller windows than
Phase 2's Prime-Swing walk-forward (180d/45d/45d vs. 100d/25d/25d) to fit
more folds into the 2.75y dev period, targeting the requested 8+ OOS folds
where data permits -- disclosed if fewer fit. Same anti-leakage discipline:
selection on train+val only, min-trade eligibility enforced, test read only
once the winner is already fixed.

Also adds the statistical-uncertainty layer (Phase 11): bootstrap CI for
expectancy and total return, fraction of profitable folds, median/worst
fold, dispersion.
"""
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from research.data_cache import load_cached  # noqa: E402
from research.generic_runner import run_generic_backtest  # noqa: E402
from research.holdout import HOLDOUT_START_TS_MS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DAY_MS = 86_400_000
TRAIN_DAYS, VAL_DAYS, TEST_DAYS = 180, 45, 45
MIN_TRADES_TRAIN, MIN_TRADES_VAL = 10, 4
TARGET_FOLDS = 8


def fold_windows(data_start_ms, dev_end_ms, target_folds=TARGET_FOLDS):
    """Evenly spreads target_folds test-windows across the FULL available
    range [data_start_ms, dev_end_ms], rather than stepping forward by a
    fixed TEST_DAYS from the start and stopping once target_folds is hit --
    that approach silently clusters every fold in the earliest slice of a
    long history whenever step*target_folds << total_range (a real defect
    found and fixed during V3.1's pre-deployment leakage check: the
    "expanded 16-fold" walk-forward on 5 years of data actually only ever
    covered 2021-08 through 2022-10, not the full range, materially
    overstating SOL's apparent edge until corrected)."""
    span = (TRAIN_DAYS + VAL_DAYS + TEST_DAYS) * DAY_MS
    total_range = dev_end_ms - data_start_ms
    if total_range < span:
        return []
    max_start = dev_end_ms - span
    if target_folds == 1 or max_start <= data_start_ms:
        starts = [data_start_ms]
    else:
        step = (max_start - data_start_ms) / (target_folds - 1)
        starts = [int(data_start_ms + i * step) for i in range(target_folds)]
    folds = []
    for start in starts:
        train_start = start
        train_end = train_start + TRAIN_DAYS * DAY_MS
        val_end = train_end + VAL_DAYS * DAY_MS
        test_end = val_end + TEST_DAYS * DAY_MS
        folds.append((train_start, train_end, val_end, test_end))
    return folds


def window_ret_and_trades(equity_curve, trade_intervals, w_start, w_end):
    sub = [(t, e) for t, e in equity_curve if w_start <= t < w_end]
    if not sub:
        return None, 0
    ret = (sub[-1][1] / sub[0][1] - 1) * 100 if sub[0][1] else 0.0
    n = sum(1 for entry_ts, _ in trade_intervals if w_start <= entry_ts < w_end)
    return ret, n


def run_fold(cls, grid, sym, chart, fold, funding_df=None, extra_fixed=None):
    train_start, train_end, val_end, test_end = fold
    c = chart[(chart["ts"] >= train_start) & (chart["ts"] < test_end)].reset_index(drop=True)
    f = funding_df[(funding_df["ts"] >= train_start) & (funding_df["ts"] < test_end)].reset_index(drop=True) \
        if funding_df is not None else None
    if len(c) == 0:
        return None

    keys = list(grid.keys())
    combos = list(itertools.product(*grid.values()))
    candidates = []
    for combo in combos:
        params = dict(zip(keys, combo))
        if extra_fixed:
            params.update(extra_fixed)
        trades, eq, guards, blocked, intervals = run_generic_backtest(
            cls, params, c, start_capital=config.START_CAPITAL, symbol=sym, funding_df=f)
        train_ret, train_n = window_ret_and_trades(eq, intervals, train_start, train_end)
        val_ret, val_n = window_ret_and_trades(eq, intervals, train_end, val_end)
        test_ret, test_n = window_ret_and_trades(eq, intervals, val_end, test_end)
        candidates.append({"params": params, "train_ret": train_ret, "val_ret": val_ret,
                            "test_ret": test_ret, "train_trades": train_n, "val_trades": val_n,
                            "test_trades": test_n,
                            "score": (min(train_ret, val_ret) if train_ret is not None
                                      and val_ret is not None else None)})

    eligible = [cd for cd in candidates if cd["score"] is not None
                and cd["train_trades"] >= MIN_TRADES_TRAIN and cd["val_trades"] >= MIN_TRADES_VAL]
    if not eligible:
        return {"no_eligible_candidates": True, "n_candidates_total": len(candidates)}
    eligible.sort(key=lambda cd: cd["score"], reverse=True)
    chosen = eligible[0]
    return {"chosen_params": chosen["params"], "chosen_score": chosen["score"],
            "chosen_test_trades": chosen["test_trades"], "oos_test_ret_pct": chosen["test_ret"],
            "n_eligible": len(eligible), "n_total": len(candidates)}


def bootstrap_ci(values, n_boot=5000, ci=0.90, seed=0):
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    arr = np.array(values)
    means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    lo = float(np.percentile(means, (1 - ci) / 2 * 100))
    hi = float(np.percentile(means, (1 + ci) / 2 * 100))
    return {"mean": float(arr.mean()), "ci_lo": lo, "ci_hi": hi, "ci_level": ci}


def summarize(fold_results, label):
    rets = [f["oos_test_ret_pct"] for f in fold_results
            if f and not f.get("no_eligible_candidates") and f["oos_test_ret_pct"] is not None]
    n_no_elig = sum(1 for f in fold_results if f and f.get("no_eligible_candidates"))
    out = {
        "label": label, "n_folds": len(fold_results), "n_folds_no_eligible": n_no_elig,
        "n_folds_with_result": len(rets),
        "oos_returns_pct": rets,
        "mean_oos_ret_pct": float(np.mean(rets)) if rets else None,
        "median_oos_ret_pct": float(np.median(rets)) if rets else None,
        "worst_fold_pct": float(min(rets)) if rets else None,
        "best_fold_pct": float(max(rets)) if rets else None,
        "std_oos_ret_pct": float(np.std(rets)) if len(rets) > 1 else None,
        "fraction_folds_profitable": float(np.mean([r > 0 for r in rets])) if rets else None,
        "bootstrap_ci_mean_return": bootstrap_ci(rets) if len(rets) >= 3 else None,
    }
    return out
