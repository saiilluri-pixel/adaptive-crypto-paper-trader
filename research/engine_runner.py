"""
Thin wrapper that runs the PRODUCTION Strategy + PaperEngine (unmodified)
over an arbitrary slice of cached historical data, for research use.

Reuses the exact same call sequence as backtest.py (check_stop_bar before
manage_on_bar, same as live) -- no strategy/engine logic is reimplemented.
Trade records are captured via PaperEngine's own CSV-writing path, pointed
at an isolated tmp file so no live project file is ever touched.
"""
import csv
import os
import sys
import tempfile
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from strategy import Strategy  # noqa: E402
from engine import PaperEngine  # noqa: E402

TF_MS = {"5m": 300_000, "30m": 1_800_000}
_SIG_KEYS = ("sig_buy_prime", "sig_buy", "sig_sell_prime", "sig_sell")
_CSV_HEADER = ["entry_time", "exit_time", "side", "entry_px", "exit_px", "qty",
               "gross_pnl", "fees", "net_pnl", "entry_reason", "exit_reason", "equity_after"]


def _events(struct_df, chart_df):
    ev = []
    for _, r in struct_df.iterrows():
        ev.append((int(r.ts) + TF_MS[config.STRUCTURE_TF], 0, "S", r))
    for _, r in chart_df.iterrows():
        ev.append((int(r.ts) + TF_MS[config.CHART_TF], 1, "C", r))
    ev.sort(key=lambda e: (e[0], e[1]))
    return ev


def run_backtest(symbol, params, struct_df, chart_df, exit_on_flip=True, arm_flip=True,
                  start_capital=None):
    """Returns (trades_df, equity_curve[(ts_ms, equity)], guard_counts, blocked_entries)."""
    guard_counts = {"max_drawdown": 0, "daily_loss": 0, "consecutive_losses": 0}

    def log_fn(msg):
        if "risk guard ACTIVE:" in msg:
            if "max drawdown" in msg:
                guard_counts["max_drawdown"] += 1
            elif "daily loss" in msg:
                guard_counts["daily_loss"] += 1
            elif "consecutive losses" in msg:
                guard_counts["consecutive_losses"] += 1

    # Inject simulated-bar-time as the engine's clock for day-rollover risk-
    # guard bookkeeping (consecutive-loss/daily-loss resets) -- real wall-
    # clock time barely advances during a backtest, so without this a guard
    # trip early in the run would permanently block entries for the rest of
    # the simulated period. See engine.py PaperEngine.__init__ docstring.
    sim_clock = {"ts_ms": int(struct_df["ts"].iloc[0]) if len(struct_df) else 0}

    def now_fn():
        return datetime.fromtimestamp(sim_clock["ts_ms"] / 1000, tz=timezone.utc)

    strat = Strategy(params)
    eng = PaperEngine(log_fn=log_fn, log_trades=False, symbol=symbol, params=params,
                       exit_on_flip=exit_on_flip, arm_flip=arm_flip, now_fn=now_fn)
    if start_capital is not None:
        eng.cash = eng.equity = eng.peak_equity = eng.daily_start_equity = start_capital

    fd, tmp_path = tempfile.mkstemp(prefix="research_trades_", suffix=".csv")
    os.close(fd)
    with open(tmp_path, "w", newline="") as f:
        csv.writer(f).writerow(_CSV_HEADER)
    eng.csv_path = tmp_path
    eng.log_trades = True   # PaperEngine.close() now writes to our isolated tmp file only

    equity_curve = []
    position_state = []   # (ts_ms, "long"|"short"|None) per processed chart bar
    trade_intervals = []  # (entry_ts_ms, exit_ts_ms), in trades-DataFrame row order
    entry_ts_by_pos_id = {}
    blocked_entries = 0
    try:
        for _, _, kind, r in _events(struct_df, chart_df):
            sim_clock["ts_ms"] = int(r.ts)
            if kind == "S":
                strat.feed_struct_bar(r.open, r.high, r.low, r.close)
                continue
            pos_before_id = id(eng.pos) if eng.pos is not None else None
            n_trades_before = eng.n_trades
            if eng.pos is not None:
                eng.check_stop_bar(r.open, r.high, r.low)
            snap = strat.feed_chart_bar(r.open, r.high, r.low, r.close)
            if eng.pos is not None:
                eng.manage_on_bar(snap)
            if eng.pos is None and any(snap[k] for k in _SIG_KEYS):
                if eng._risk_guard_blocked() is not None:
                    blocked_entries += 1
            eng.on_signals(snap)

            # Trade-boundary tracking by Position OBJECT IDENTITY, not just
            # side value -- side alone can't disambiguate a same-bar close
            # followed immediately by a same-side re-entry (a real, existing
            # engine behavior), which would otherwise look like one unbroken
            # open interval and silently merge two trades into one.
            if eng.n_trades > n_trades_before and pos_before_id is not None:
                entry_ts = entry_ts_by_pos_id.pop(pos_before_id, int(r.ts))
                trade_intervals.append((entry_ts, int(r.ts)))
            if eng.pos is not None and id(eng.pos) not in entry_ts_by_pos_id:
                entry_ts_by_pos_id[id(eng.pos)] = int(r.ts)

            equity_curve.append((int(r.ts), eng.mark(r.close)))
            position_state.append((int(r.ts), eng.pos.side if eng.pos else None))
        trades = pd.read_csv(tmp_path)
    finally:
        os.remove(tmp_path)

    return trades, equity_curve, guard_counts, blocked_entries, position_state, trade_intervals


def real_trade_times(position_state=None, trade_intervals=None):
    """trades_*.csv timestamps entry_time/exit_time with backtest RUN-time
    (wall clock), not simulated bar time -- documented in README, harmless
    for live (where run-time IS real time) but useless for research holding-
    time metrics. Prefer the precise trade_intervals returned by
    run_backtest() (built via Position object identity, correctly handles
    same-bar close+reopen). position_state-based reconstruction is kept only
    as a fallback for callers that didn't capture trade_intervals; it merges
    same-bar/same-side close+reopen pairs into one interval."""
    if trade_intervals is not None:
        return trade_intervals
    intervals = []
    open_since = None
    for ts, side in position_state:
        if side is not None and open_since is None:
            open_since = ts
        elif side is None and open_since is not None:
            intervals.append((open_since, ts))
            open_since = None
    return intervals
