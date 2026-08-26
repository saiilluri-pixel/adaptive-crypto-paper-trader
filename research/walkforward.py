"""
Phase 2, step 10: proper TRAIN -> VALIDATION -> TEST walk-forward.

Fixes the data-snooping defect found in the original audit of autotune.py
(score = min(train, test) let the "test" period influence selection). Here:

  - Parameter selection uses ONLY train+validation (score = min(train_ret,
    val_ret), same consistency-over-single-best philosophy as autotune.py,
    but validation is a genuinely separate window from test).
  - TEST is computed for every candidate (for efficiency -- one continuous
    run per candidate spans train->test) but is NEVER read, compared, or
    used in any way until AFTER the winning candidate is already chosen.
  - Only the CHOSEN candidate's test segment is ever reported. The walk-
    forward out-of-sample equity curve is built ONLY from these unseen test
    segments, concatenated across folds.

Uses autotune.py's existing GRID unmodified (not expanded). Folds are
spread across the cached year (early/mid/late) rather than every possible
rolling step, to keep runtime tractable -- flagged clearly as a scope
limitation in the report, not hidden.

Research only -- never touches symbol_params.json or the live bot.
"""
import itertools
import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from autotune import GRID  # noqa: E402  (existing grid, reused unmodified)
from research.data_cache import load_cached  # noqa: E402
from research.engine_runner import run_backtest  # noqa: E402
from research.metrics import compute_metrics  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DAY_MS = 86_400_000
TRAIN_DAYS, VAL_DAYS, TEST_DAYS = 100, 25, 25
MIN_TRADES_TRAIN, MIN_TRADES_VAL = 8, 3


def _fold_windows(data_start_ms, data_end_ms):
    """3 folds spread across the available year: early / mid / late.
    Scope-limited for runtime (162 combos x ~3s/run x 3 folds x 3 symbols
    is already ~45min); NOT every possible rolling step."""
    total_days = (data_end_ms - data_start_ms) / DAY_MS
    span = TRAIN_DAYS + VAL_DAYS + TEST_DAYS
    if total_days < span:
        raise ValueError(f"only {total_days:.0f}d cached, need >= {span}d for one fold")
    starts_days = [0,
                   max(0, (total_days - span) / 2),
                   max(0, total_days - span)]
    folds = []
    for d in starts_days:
        train_start = data_start_ms + int(d * DAY_MS)
        train_end = train_start + TRAIN_DAYS * DAY_MS
        val_end = train_end + VAL_DAYS * DAY_MS
        test_end = val_end + TEST_DAYS * DAY_MS
        if test_end > data_end_ms:
            test_end = data_end_ms
        folds.append((train_start, train_end, val_end, test_end))
    # dedupe (short caches may collapse early/mid/late to the same window)
    seen, uniq = set(), []
    for f in folds:
        if f not in seen:
            seen.add(f); uniq.append(f)
    return uniq


def _metrics_for_window(equity_curve, trade_intervals, w_start, w_end):
    """Returns (ret_pct, n_trades) for [w_start, w_end). Return is recomputed
    from equity deltas (robust against the trades-CSV wall-clock-time trap --
    see engine_runner docstring). Trade count uses trade_intervals (real,
    bar-time entry timestamps, from Position object identity -- correctly
    handles same-bar close+reopen) so the eligibility filter below is counting
    ACTUAL trades, not equity-curve bar samples (the bug this fixes: the
    original walk-forward run defined MIN_TRADES_TRAIN/VAL but never applied
    them, because it had no real per-window trade count to filter on)."""
    sub_curve = [(t, e) for t, e in equity_curve if w_start <= t < w_end]
    if not sub_curve:
        return None, 0
    start_eq = sub_curve[0][1]
    end_eq = sub_curve[-1][1]
    ret_pct = (end_eq / start_eq - 1) * 100 if start_eq else 0.0
    n_trades = sum(1 for entry_ts, _ in trade_intervals if w_start <= entry_ts < w_end)
    return ret_pct, n_trades


def run_fold(symbol, struct, chart, fold, grid_combos):
    train_start, train_end, val_end, test_end = fold
    s = struct[(struct["ts"] >= train_start) & (struct["ts"] < test_end)].reset_index(drop=True)
    c = chart[(chart["ts"] >= train_start) & (chart["ts"] < test_end)].reset_index(drop=True)
    if len(c) == 0:
        return None

    _base_params = {k: getattr(config, k) for k in
                     ("SWING_LEN", "NW_BANDWIDTH", "NW_MULT", "NW_WINDOW",
                      "ST_ATR_PERIOD", "ST_ATR_MULT", "ST_INIT_STOP")}
    candidates = []
    for combo in grid_combos:
        params = dict(_base_params)
        params.update(dict(zip(GRID.keys(), combo)))   # overlay all 5 GRID-tunable values
        trades, eq, guards, blocked, _pos, intervals = run_backtest(
            symbol, params, s, c, start_capital=config.START_CAPITAL)
        train_ret, train_n = _metrics_for_window(eq, intervals, train_start, train_end)
        val_ret, val_n = _metrics_for_window(eq, intervals, train_end, val_end)
        test_ret, test_n = _metrics_for_window(eq, intervals, val_end, test_end)
        candidates.append({
            "params": params, "train_ret": train_ret, "val_ret": val_ret, "test_ret": test_ret,
            "train_trades": train_n, "val_trades": val_n, "test_trades": test_n,
            "score": (min(train_ret, val_ret) if train_ret is not None and val_ret is not None else None),
        })

    # SELECTION: train+val ONLY, with the min-trade eligibility floor actually
    # applied this time (train_n/val_n are now real per-window trade counts).
    # test_ret exists on every candidate above (computed for efficiency, one
    # run per combo) but is not read here.
    eligible = [c for c in candidates if c["score"] is not None
                and c["train_trades"] >= MIN_TRADES_TRAIN
                and c["val_trades"] >= MIN_TRADES_VAL]
    if not eligible:
        return {"fold_window": {"train": [train_start, train_end], "val": [train_end, val_end],
                                 "test": [val_end, test_end]},
                "no_eligible_candidates": True,
                "n_candidates_total": len(candidates),
                "all_candidates_summary": [
                    {"params": c["params"], "train_ret": c["train_ret"], "val_ret": c["val_ret"],
                     "train_trades": c["train_trades"], "val_trades": c["val_trades"], "score": c["score"]}
                    for c in candidates]}
    eligible.sort(key=lambda c: c["score"], reverse=True)
    chosen = eligible[0]

    # NOW, and only now, report the chosen candidate's unseen test segment.
    params = chosen["params"]
    trades, eq, guards, blocked, _pos, _intervals = run_backtest(
        symbol, params, s, c, start_capital=config.START_CAPITAL)
    test_curve = [(t, e) for t, e in eq if val_end <= t < test_end]
    return {
        "fold_window": {"train": [train_start, train_end], "val": [train_end, val_end],
                         "test": [val_end, test_end]},
        "chosen_params": params,
        "chosen_train_ret": chosen["train_ret"], "chosen_val_ret": chosen["val_ret"],
        "chosen_train_trades": chosen["train_trades"], "chosen_val_trades": chosen["val_trades"],
        "chosen_test_trades": chosen["test_trades"],
        "chosen_score": chosen["score"],
        "n_eligible_candidates": len(eligible), "n_candidates_total": len(candidates),
        "min_trades_train": MIN_TRADES_TRAIN, "min_trades_val": MIN_TRADES_VAL,
        "oos_test_ret_pct": chosen["test_ret"],
        "oos_test_curve": test_curve,
        "all_candidates_summary": [
            {"params": c["params"], "train_ret": c["train_ret"], "val_ret": c["val_ret"],
             "train_trades": c["train_trades"], "val_trades": c["val_trades"], "score": c["score"]}
            for c in candidates],
    }


def main():
    combos = list(itertools.product(*GRID.values()))
    print(f"grid: {len(combos)} combos x {len(config.SYMBOLS)} symbols")

    results = {}
    for sym in config.SYMBOLS:
        struct = load_cached(sym, config.STRUCTURE_TF)
        chart = load_cached(sym, config.CHART_TF)
        folds = _fold_windows(int(chart["ts"].min()), int(chart["ts"].max()))
        print(f"[{sym}] {len(folds)} fold(s)")
        fold_results = []
        for i, fold in enumerate(folds):
            t0 = time.time()
            r = run_fold(sym, struct, chart, fold, combos)
            if r:
                fold_results.append(r)
                if r.get("no_eligible_candidates"):
                    print(f"  fold {i}: NO ELIGIBLE CANDIDATES "
                          f"(min_trades_train={MIN_TRADES_TRAIN}, min_trades_val={MIN_TRADES_VAL})  "
                          f"({time.time()-t0:.0f}s)")
                else:
                    print(f"  fold {i}: chosen score={r['chosen_score']:.2f} "
                          f"train_n={r['chosen_train_trades']} val_n={r['chosen_val_trades']} "
                          f"oos_test_ret={r['oos_test_ret_pct']:.2f}%  ({time.time()-t0:.0f}s)")
        results[sym] = fold_results

    # aggregate ONLY the unseen test segments into one walk-forward OOS curve per symbol
    oos_summary = {}
    for sym, folds in results.items():
        all_test_rets = [f["oos_test_ret_pct"] for f in folds
                          if not f.get("no_eligible_candidates") and f["oos_test_ret_pct"] is not None]
        n_no_eligible = sum(1 for f in folds if f.get("no_eligible_candidates"))
        oos_summary[sym] = {
            "n_folds": len(folds), "n_folds_no_eligible_candidates": n_no_eligible,
            "oos_test_returns_pct_per_fold": all_test_rets,
            "oos_mean_test_ret_pct": (sum(all_test_rets) / len(all_test_rets)) if all_test_rets else None,
            "oos_worst_fold_pct": min(all_test_rets) if all_test_rets else None,
            "oos_best_fold_pct": max(all_test_rets) if all_test_rets else None,
        }

    os.makedirs(os.path.join(HERE, "reports"), exist_ok=True)
    with open(os.path.join(HERE, "reports", "walkforward.json"), "w") as f:
        json.dump({"folds": results, "oos_summary": oos_summary}, f, indent=2, default=str)
    print(json.dumps(oos_summary, indent=2, default=str))
    print("\nwrote research/reports/walkforward.json")


if __name__ == "__main__":
    main()
