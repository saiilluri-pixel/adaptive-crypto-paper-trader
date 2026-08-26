"""
V3.1 section B: event ledger for SHOCK_CONTINUATION_V3_FROZEN.

_LoggedShockContinuation is a pure diagnostic subclass -- identical signal
logic to the frozen ShockContinuationStrategy (same on_bar, same shock
condition, same direction rule), it only ADDS a side-channel log of shock
metrics (ATR-normalized size, volume multiple, wick/body shape) at the
moment each signal fires. No trading behavior is changed.
"""
import os
import sys
from collections import deque

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from research.data_cache import load_cached  # noqa: E402
from research.generic_runner import run_generic_backtest  # noqa: E402
from research.strategies.shock_aftershock import _ShockBase  # noqa: E402

FROZEN_PARAMS = {"ATR_N": 14, "SHOCK_ATR_MULT": 3.0, "INIT_STOP_PCT": 4.0}


class _LoggedShockContinuation(_ShockBase):
    def __init__(self, params):
        super().__init__(params)
        self.shock_log = []   # filled as signals fire; consumed by build_ledger()
        # volume isn't part of generic_runner's on_bar interface (the frozen
        # strategy never used it) -- volume_mult is joined in separately in
        # build_ledger() from the chart's own 'volume' column, purely for
        # diagnostics, never fed back into the signal itself.

    def on_bar(self, o, h, l, c, ts_ms, htf_trend=None, funding_rate=None):
        atr_before = (sum(self.tr_hist) / len(self.tr_hist)) if len(self.tr_hist) == self.atr_n else None
        d = self._shock_direction(o, h, l, c)
        if d is not None and atr_before is not None:
            tr = max(h - l, abs(h - self.prev_close) if self.prev_close is not None else h - l)
            body = abs(c - o)
            upper_wick = h - max(o, c)
            lower_wick = min(o, c) - l
            rng = h - l if h > l else 1e-9
            self.shock_log.append({
                "ts_ms": ts_ms, "direction": "long" if d == 1 else "short",
                "shock_bar_return_pct": (c / o - 1) * 100 if o else None,
                "atr_norm_shock": tr / atr_before if atr_before else None,
                "body_frac": body / rng, "upper_wick_frac": upper_wick / rng, "lower_wick_frac": lower_wick / rng,
                "open": o, "high": h, "low": l, "close": c,
            })
        if d == 1:
            return "long"
        if d == -1:
            return "short"
        return None


def build_ledger(sym, chart, end_ts_ms=None, start_ts_ms=None):
    """Runs the frozen strategy (via the logged subclass) and returns a
    per-EVENT ledger (one row per fired signal, whether or not it resulted
    in an entry -- entries only happen when flat, matching production;
    joined against the actual trade if one occurred) plus forward-return and
    MAE/MFE enrichment computed directly from chart_df."""
    c = chart
    if start_ts_ms is not None:
        c = c[c["ts"] >= start_ts_ms]
    if end_ts_ms is not None:
        c = c[c["ts"] < end_ts_ms]
    c = c.reset_index(drop=True)

    strat_holder = {}

    class _Wrapped(_LoggedShockContinuation):
        def __init__(self, params):
            super().__init__(params)
            strat_holder["strat"] = self

    trades, eq, guards, blocked, intervals = run_generic_backtest(
        _Wrapped, FROZEN_PARAMS, c, start_capital=config.START_CAPITAL, symbol=sym)
    shock_log = pd.DataFrame(strat_holder["strat"].shock_log)

    # volume multiple: joined separately from the chart's own volume column
    # (diagnostic only -- never fed into the signal itself)
    if len(shock_log) and "volume" in c.columns:
        vol_df = pd.DataFrame({"ts_ms": c["ts"] + 3_600_000, "volume": c["volume"]})
        vol_df["vol_avg_trailing"] = vol_df["volume"].rolling(FROZEN_PARAMS["ATR_N"]).mean().shift(1)
        shock_log = shock_log.merge(vol_df, on="ts_ms", how="left")
        shock_log["volume_mult"] = shock_log["volume"] / shock_log["vol_avg_trailing"]

    # forward returns + MAE/MFE computed directly from the OHLCV series
    close_by_ts = c.set_index(c["ts"] + 3_600_000)["close"]   # index by bar CLOSE ts (matches ts_ms convention)
    high_by_ts = c.set_index(c["ts"] + 3_600_000)["high"]
    low_by_ts = c.set_index(c["ts"] + 3_600_000)["low"]
    ordered_ts = list(close_by_ts.index)

    def forward_return(ts_ms, hours):
        if ts_ms not in close_by_ts.index:
            return None
        i = ordered_ts.index(ts_ms)
        j = i + hours
        if j >= len(ordered_ts):
            return None
        p0, p1 = close_by_ts.iloc[i], close_by_ts.iloc[j]
        return (p1 / p0 - 1) * 100 if p0 else None

    if len(shock_log):
        shock_log["fwd_1h_pct"] = shock_log["ts_ms"].apply(lambda t: forward_return(t, 1))
        shock_log["fwd_4h_pct"] = shock_log["ts_ms"].apply(lambda t: forward_return(t, 4))
        shock_log["fwd_12h_pct"] = shock_log["ts_ms"].apply(lambda t: forward_return(t, 12))
        shock_log["fwd_24h_pct"] = shock_log["ts_ms"].apply(lambda t: forward_return(t, 24))

    # join trades (only signals that actually resulted in an entry -- flat-only, one at a time)
    trade_rows = []
    for idx, (entry_ts, exit_ts) in enumerate(intervals):
        row = trades.iloc[idx]
        # MAE/MFE: scan bars strictly between entry and exit (inclusive of exit bar)
        window = c[(c["ts"] + 3_600_000 > entry_ts) & (c["ts"] + 3_600_000 <= exit_ts)]
        side = row["side"]
        entry_px = row["entry_px"]
        if len(window):
            if side == "long":
                mae = (window["low"].min() / entry_px - 1) * 100
                mfe = (window["high"].max() / entry_px - 1) * 100
            else:
                mae = (entry_px / window["high"].max() - 1) * 100
                mfe = (entry_px / window["low"].min() - 1) * 100
        else:
            mae = mfe = None
        trade_rows.append({
            "symbol": sym, "entry_ts_ms": entry_ts, "exit_ts_ms": exit_ts,
            "entry_dt": pd.to_datetime(entry_ts, unit="ms", utc=True),
            "side": side, "entry_px": entry_px, "exit_px": row["exit_px"],
            "holding_hours": (exit_ts - entry_ts) / 3_600_000,
            "gross_pnl": row["gross_pnl"], "fees": row["fees"], "net_pnl": row["net_pnl"],
            "exit_reason": row["exit_reason"], "mae_pct": mae, "mfe_pct": mfe,
        })
    trade_df = pd.DataFrame(trade_rows)

    if len(trade_df) and len(shock_log):
        # match each trade to its originating shock signal by nearest ts <= entry_ts
        shock_log_sorted = shock_log.sort_values("ts_ms")
        merged = pd.merge_asof(trade_df.sort_values("entry_ts_ms"), shock_log_sorted,
                                left_on="entry_ts_ms", right_on="ts_ms", direction="backward")
        return merged
    return trade_df
