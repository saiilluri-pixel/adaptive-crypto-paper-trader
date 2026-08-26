"""
Phase 2, step 5 & 6: fixed-parameter baselines, no optimization.

Runs the CORRECTED engine (Strategy + PaperEngine, unmodified) once per
symbol over the full cached history, using:

  A) LEGACY_TUNED_BASELINE  -- symbol_params.json values (current live config)
  B) GLOBAL_DEFAULT_BASELINE -- config.py global defaults for all symbols

From each single continuous run we derive, without re-running:
  - full-period descriptive metrics
  - monthly buckets (from the same trade log)
  - a post-tuning forward-test slice (trades/equity from the verified
    symbol_params.json write date onward, re-based to that date's equity)

Writes research/reports/fixed_baseline.json. Never touches symbol_params.json,
config.py, or the live bot.
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from research.data_cache import load_cached  # noqa: E402
from research.engine_runner import run_backtest, real_trade_times  # noqa: E402
from research.metrics import compute_metrics  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
# verified in step 3: symbol_params.json was last written 2026-06-21 19:07 UTC
# (the code comment says "06-19"; that date is the EARLIER global-only tune,
# superseded by this per-symbol run -- see legacy_tuned_baseline.json)
FORWARD_TEST_START = pd.Timestamp("2026-06-22", tz="UTC")
_TUNABLE = ("SWING_LEN", "NW_BANDWIDTH", "NW_MULT", "NW_WINDOW",
            "ST_ATR_PERIOD", "ST_ATR_MULT", "ST_INIT_STOP")


def _global_default_params():
    return {k: getattr(config, k) for k in _TUNABLE}


def _slice_forward(trades, real_times, equity_curve, cutoff):
    cutoff_ms = int(cutoff.value // 1_000_000)
    keep_idx = [i for i, (en, _) in enumerate(real_times) if en >= cutoff_ms]
    fwd_trades = trades.iloc[keep_idx].reset_index(drop=True)
    fwd_real_times = [real_times[i] for i in keep_idx]
    idx = next((i for i, (ts, _) in enumerate(equity_curve) if ts >= cutoff_ms), None)
    if idx is None:
        return fwd_trades, fwd_real_times, [], None
    rebase_equity = equity_curve[idx][1]
    fwd_curve = equity_curve[idx:]
    return fwd_trades, fwd_real_times, fwd_curve, rebase_equity


def run_one(symbol, params, label, exit_on_flip=True, arm_flip=True):
    struct = load_cached(symbol, config.STRUCTURE_TF)
    chart = load_cached(symbol, config.CHART_TF)
    trades, equity_curve, guard_counts, blocked, pos_state, trade_intervals = run_backtest(
        symbol, params, struct, chart, exit_on_flip=exit_on_flip, arm_flip=arm_flip,
        start_capital=config.START_CAPITAL)
    rtimes = real_trade_times(trade_intervals=trade_intervals)

    full = compute_metrics(trades, equity_curve, config.START_CAPITAL, guard_counts, real_times=rtimes)
    full["entries_blocked_by_guard"] = blocked
    full["period_start"] = str(chart["dt"].iloc[0])
    full["period_end"] = str(chart["dt"].iloc[-1])

    fwd_trades, fwd_rtimes, fwd_curve, rebase_eq = _slice_forward(
        trades, rtimes, equity_curve, FORWARD_TEST_START)
    if fwd_curve and rebase_eq:
        forward = compute_metrics(fwd_trades, fwd_curve, rebase_eq, real_times=fwd_rtimes)
        forward["window"] = f"{FORWARD_TEST_START.date()} onward (post symbol_params.json tuning)"
    else:
        forward = {"note": "cached data does not extend past the forward-test cutoff"}

    return {
        "label": label, "symbol": symbol, "params": params,
        "full_period": full, "post_tuning_forward": forward,
    }


def main():
    results = {"legacy_tuned": {}, "global_default": {}}
    for sym in config.SYMBOLS:
        tuned_params = config.params_for(sym)
        print(f"[{sym}] LEGACY_TUNED_BASELINE …")
        results["legacy_tuned"][sym] = run_one(sym, tuned_params, "LEGACY_TUNED_BASELINE")
        print(f"[{sym}] GLOBAL_DEFAULT_BASELINE …")
        results["global_default"][sym] = run_one(sym, _global_default_params(), "GLOBAL_DEFAULT_BASELINE")

    os.makedirs(os.path.join(HERE, "reports"), exist_ok=True)
    out_path = os.path.join(HERE, "reports", "fixed_baseline.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
