"""
Phase 6, Stage 3: walk-forward -- only robust (Stage 2) candidates reach
this. Same train->validation->test discipline as the Prime-Swing walk-
forward (research/walkforward.py): selection uses train+val only, test is
computed alongside every candidate for efficiency but never read until the
winner is already fixed, and MIN_TRADES eligibility is actually enforced
(the bug already fixed in walkforward.py).
"""
import itertools
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from research.data_cache import load_cached  # noqa: E402
from research.generic_runner import run_generic_backtest  # noqa: E402
from research.walkforward import _fold_windows  # noqa: E402
from research.stage2_robustness import GRIDS, SURVIVORS, HTF_BARS, TF  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DAY_MS = 86_400_000
TRAIN_DAYS, VAL_DAYS, TEST_DAYS = 100, 25, 25
MIN_TRADES_TRAIN, MIN_TRADES_VAL = 8, 3


def _window_ret_and_trades(equity_curve, trade_intervals, w_start, w_end):
    sub = [(t, e) for t, e in equity_curve if w_start <= t < w_end]
    if not sub:
        return None, 0
    ret = (sub[-1][1] / sub[0][1] - 1) * 100 if sub[0][1] else 0.0
    n = sum(1 for entry_ts, _ in trade_intervals if w_start <= entry_ts < w_end)
    return ret, n


def run_fold(cls, grid, sym, chart, fold):
    train_start, train_end, val_end, test_end = fold
    c = chart[(chart["ts"] >= train_start) & (chart["ts"] < test_end)].reset_index(drop=True)
    if len(c) == 0:
        return None

    keys = list(grid.keys())
    combos = list(itertools.product(*grid.values()))
    candidates = []
    for combo in combos:
        params = dict(zip(keys, combo))
        trades, eq, guards, blocked, intervals = run_generic_backtest(
            cls, params, c, start_capital=config.START_CAPITAL, symbol=sym, htf_bars=HTF_BARS)
        train_ret, train_n = _window_ret_and_trades(eq, intervals, train_start, train_end)
        val_ret, val_n = _window_ret_and_trades(eq, intervals, train_end, val_end)
        test_ret, test_n = _window_ret_and_trades(eq, intervals, val_end, test_end)
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
    return {
        "chosen_params": chosen["params"], "chosen_score": chosen["score"],
        "chosen_train_trades": chosen["train_trades"], "chosen_val_trades": chosen["val_trades"],
        "chosen_test_trades": chosen["test_trades"],
        "n_eligible": len(eligible), "n_total": len(candidates),
        "oos_test_ret_pct": chosen["test_ret"],
        "all_candidates": [{"params": cd["params"], "train_ret": cd["train_ret"],
                             "val_ret": cd["val_ret"], "score": cd["score"]} for cd in candidates],
    }


def main():
    out = {}
    for name, sym in SURVIVORS.items():
        cls, grid = GRIDS[name]["cls"], GRIDS[name]["grid"]
        chart = load_cached(sym, TF)
        folds = _fold_windows(int(chart["ts"].min()), int(chart["ts"].max()))
        print(f"[{name}][{sym}] {len(folds)} folds x "
              f"{len(list(itertools.product(*grid.values())))} combos")
        fold_results = []
        for i, fold in enumerate(folds):
            t0 = time.time()
            r = run_fold(cls, grid, sym, chart, fold)
            if r:
                fold_results.append(r)
                if r.get("no_eligible_candidates"):
                    print(f"  fold {i}: no eligible candidates  ({time.time()-t0:.0f}s)")
                else:
                    print(f"  fold {i}: score={r['chosen_score']:.2f} oos_test={r['oos_test_ret_pct']:.2f}%  "
                          f"({time.time()-t0:.0f}s)")
        rets = [f["oos_test_ret_pct"] for f in fold_results
                if not f.get("no_eligible_candidates") and f["oos_test_ret_pct"] is not None]
        out[f"{name}__{sym.replace('/', '')}"] = {
            "folds": fold_results,
            "oos_mean_ret_pct": (sum(rets) / len(rets)) if rets else None,
            "oos_returns": rets,
        }

    os.makedirs(os.path.join(HERE, "reports"), exist_ok=True)
    with open(os.path.join(HERE, "reports", "stage3_walkforward.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("\nwrote research/reports/stage3_walkforward.json")


if __name__ == "__main__":
    main()
