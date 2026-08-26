"""
Generic backtest runner for research/strategies/*.py candidates (Phase 2).

Drives the PRODUCTION PaperEngine directly (fees, slippage, risk-based
sizing, risk guards, corrected trailing-stop math, corrected OHLC-aware
stop-fill logic) via its generic enter()/check_stop_bar()/manage_on_bar()
methods -- bypassing on_signals()/Strategy, which speak Prime-Swing's
specific signal vocabulary and aren't needed here. Every research family
gets identical, already-validated execution; only entry signal generation
differs between candidates.

Exit model: Supertrend-flip exit is intentionally disabled (exit_on_flip=
False) for every research candidate -- each family expresses its own exit
via the entry signal's stop distance and the shared trailing-stop, keeping
exit execution identical and comparable across families ("execution
simplicity" per the V2 objective) rather than reimplementing a bespoke
exit per strategy.

KNOWN SIMPLIFICATION -- read before treating "ATR-based stop" results as
precise: PaperEngine.init_stop is one fixed percentage for the WHOLE
backtest run (set at construction), not recomputed per trade. A true
per-trade ATR-adaptive stop would require extending PaperEngine/Position,
which is out of scope here (a new feature, not a correctness fix, and V2's
instructions are explicit that the execution engine itself is not to keep
changing). Candidates that want an "ATR-based" stop instead compute ONE
representative stop percentage for the whole backtest (e.g. the median ATR%
over the period) via params["INIT_STOP_PCT"], and are evaluated on that
basis. This is disclosed in every report that uses it, not hidden.
"""
import csv
import os
import sys
import tempfile
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from engine import PaperEngine  # noqa: E402

_CSV_HEADER = ["entry_time", "exit_time", "side", "entry_px", "exit_px", "qty",
               "gross_pnl", "fees", "net_pnl", "entry_reason", "exit_reason", "equity_after"]


def compute_htf_trend(chart_df, htf_bars=48):
    """Simple higher-timeframe trend context: sign of the close vs. its
    htf_bars-bar-ago close (default 48 bars -- 4h on a 5m chart, 12h on a
    15m chart, 2 days on a 1h chart; callers on other timeframes should pass
    an htf_bars that maps to their own desired higher timeframe). +1 up,
    -1 down, 0 flat/insufficient history. Returned as a plain list aligned
    1:1 with chart_df rows -- purely descriptive context, not a signal."""
    close = chart_df["close"]
    ref = close.shift(htf_bars)
    diff = close - ref
    trend = diff.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    trend[ref.isna()] = 0
    return trend.tolist()


def align_funding(chart_df, funding_df):
    """Forward-fill the funding rate (real Binance 8h perpetual funding
    history) onto the chart timeframe -- each bar sees the most recently
    KNOWN rate as of that bar's timestamp, never a future one. Returns a
    plain list aligned 1:1 with chart_df rows; None before the first known
    funding print in range."""
    f = funding_df.sort_values("ts").reset_index(drop=True)
    out = []
    j = -1
    for ts in chart_df["ts"]:
        while j + 1 < len(f) and f["ts"].iloc[j + 1] <= ts:
            j += 1
        out.append(float(f["funding_rate"].iloc[j]) if j >= 0 else None)
    return out


def run_generic_backtest(strategy_cls, params, chart_df, start_capital,
                          symbol="RESEARCH", htf_bars=None, funding_df=None, bar_ms=None):
    """chart_df: single-timeframe OHLCV (the strategy's own bar TF). bar_ms:
    that timeframe's duration in ms (e.g. 300_000 for 5m, 3_600_000 for 1h)
    -- used only to compute each bar's CLOSE timestamp; derived from the
    data itself if not given. Returns (trades_df, equity_curve[(ts_ms,
    equity)], guard_counts, blocked_entries, trade_intervals[(entry_ts_ms,
    exit_ts_ms)])."""
    init_stop_pct = params.get("INIT_STOP_PCT", 3.0)
    if bar_ms is None:
        bar_ms = int(chart_df["ts"].iloc[1] - chart_df["ts"].iloc[0]) if len(chart_df) > 1 else 300_000
    guard_counts = {"max_drawdown": 0, "daily_loss": 0, "consecutive_losses": 0}

    def log_fn(msg):
        if "risk guard ACTIVE:" in msg:
            if "max drawdown" in msg:
                guard_counts["max_drawdown"] += 1
            elif "daily loss" in msg:
                guard_counts["daily_loss"] += 1
            elif "consecutive losses" in msg:
                guard_counts["consecutive_losses"] += 1

    sim_clock = {"ts_ms": int(chart_df["ts"].iloc[0]) if len(chart_df) else 0}

    def now_fn():
        return datetime.fromtimestamp(sim_clock["ts_ms"] / 1000, tz=timezone.utc)

    strat = strategy_cls(params)
    eng = PaperEngine(log_fn=log_fn, log_trades=False, symbol=symbol,
                       params={"ST_INIT_STOP": init_stop_pct},
                       exit_on_flip=False, arm_flip=False, now_fn=now_fn)
    eng.cash = eng.equity = eng.peak_equity = eng.daily_start_equity = start_capital

    fd, tmp_path = tempfile.mkstemp(prefix="research_trades_", suffix=".csv")
    os.close(fd)
    with open(tmp_path, "w", newline="") as f:
        csv.writer(f).writerow(_CSV_HEADER)
    eng.csv_path = tmp_path
    eng.log_trades = True

    htf_trend = compute_htf_trend(chart_df, htf_bars) if htf_bars else [None] * len(chart_df)
    funding = align_funding(chart_df, funding_df) if funding_df is not None else [None] * len(chart_df)

    equity_curve = []
    trade_intervals = []
    entry_ts_by_pos_id = {}
    blocked_entries = 0
    try:
        for i, r in enumerate(chart_df.itertuples(index=False)):
            ts_ms = int(r.ts) + bar_ms   # bar CLOSE timestamp (matches engine_runner convention)
            sim_clock["ts_ms"] = int(r.ts)
            pos_before_id = id(eng.pos) if eng.pos is not None else None
            n_trades_before = eng.n_trades

            if eng.pos is not None:
                eng.check_stop_bar(r.open, r.high, r.low)
            if eng.pos is not None:
                eng.manage_on_bar({"close": r.close, "st_dir": 0})   # ratchet only (exit_on_flip=False)

            sig = strat.on_bar(r.open, r.high, r.low, r.close, ts_ms, htf_trend[i], funding[i])
            if eng.pos is None and sig in ("long", "short"):
                if eng._risk_guard_blocked() is not None:
                    blocked_entries += 1
                else:
                    eng.enter(sig, r.close, "signal", st_dir=0)

            if eng.n_trades > n_trades_before and pos_before_id is not None:
                entry_ts = entry_ts_by_pos_id.pop(pos_before_id, ts_ms)
                trade_intervals.append((entry_ts, ts_ms))
            if eng.pos is not None and id(eng.pos) not in entry_ts_by_pos_id:
                entry_ts_by_pos_id[id(eng.pos)] = ts_ms

            equity_curve.append((ts_ms, eng.mark(r.close)))
        trades = pd.read_csv(tmp_path)
    finally:
        os.remove(tmp_path)

    return trades, equity_curve, guard_counts, blocked_entries, trade_intervals
