"""
Metrics computation for research backtests (Phase 2, step 7).

Takes the trades DataFrame (same schema PaperEngine writes to trades_*.csv)
plus an equity curve (list of (ts_ms, equity)) and produces the full metric
set requested. No live files are touched -- pure computation on in-memory
data produced by research/engine_runner.py.
"""
import numpy as np
import pandas as pd

import config


def _drawdown_series(equity):
    eq = np.array(equity, dtype=float)
    peak = np.maximum.accumulate(eq)
    dd = np.where(peak > 0, (eq - peak) / peak * 100.0, 0.0)
    return dd


def _max_dd_duration(ts_ms, equity):
    """Bars from a new equity peak until equity recovers back to that peak
    (or end of data, if never recovered) -- the longest such stretch."""
    eq = np.array(equity, dtype=float)
    peak = -np.inf
    peak_idx = 0
    worst_len = 0
    worst_start = worst_end = None
    for i, v in enumerate(eq):
        if v >= peak:
            peak = v
            peak_idx = i
        else:
            length = i - peak_idx
            if length > worst_len:
                worst_len = length
                worst_start, worst_end = peak_idx, i
    if worst_start is None or len(ts_ms) < 2:
        return 0.0
    hours = (ts_ms[worst_end] - ts_ms[worst_start]) / 3_600_000
    return hours


def _streaks(is_win_seq):
    longest_w = longest_l = cur_w = cur_l = 0
    for w in is_win_seq:
        if w:
            cur_w += 1; cur_l = 0
        else:
            cur_l += 1; cur_w = 0
        longest_w = max(longest_w, cur_w)
        longest_l = max(longest_l, cur_l)
    return longest_w, longest_l


def _sharpe_sortino(bar_returns, bars_per_year):
    r = np.array(bar_returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 2 or r.std() == 0:
        return None, None
    sharpe = (r.mean() / r.std()) * np.sqrt(bars_per_year)
    downside = r[r < 0]
    sortino = (r.mean() / downside.std()) * np.sqrt(bars_per_year) if len(downside) > 1 and downside.std() > 0 else None
    return float(sharpe), (float(sortino) if sortino is not None else None)


def compute_metrics(trades, equity_curve, start_capital, guard_counts=None, real_times=None):
    """
    real_times: optional list of (entry_ts_ms, exit_ts_ms), same order as
        trades rows, from engine_runner.real_trade_times() -- trades_*.csv's
        own entry_time/exit_time are backtest RUN-time (wall clock), not
        simulated bar time, so holding-time/time-in-market need this instead.
    trades: DataFrame with columns entry_time, exit_time, side, entry_px,
        exit_px, qty, gross_pnl, fees, net_pnl, entry_reason, exit_reason,
        equity_after (matches engine.py's trades_*.csv schema exactly).
    equity_curve: list of (ts_ms, equity) sampled once per processed bar.
    """
    ts_ms = [t for t, _ in equity_curve]
    eq = [e for _, e in equity_curve]
    out = {}
    out["starting_capital"] = start_capital
    out["ending_equity"] = eq[-1] if eq else start_capital
    out["net_return_pct"] = (out["ending_equity"] / start_capital - 1) * 100

    if len(trades) == 0:
        out["total_trades"] = 0
        out["note"] = "no trades in this window -- most metrics below are not meaningful"
        dd = _drawdown_series(eq) if eq else np.array([0.0])
        out["max_drawdown_pct"] = float(dd.min()) if len(dd) else 0.0
        return out

    gross = trades["gross_pnl"].astype(float)
    fees = trades["fees"].astype(float)
    net = trades["net_pnl"].astype(float)
    wins = net > 0
    losses = net <= 0

    out["gross_pnl"] = float(gross.sum())
    out["total_fees"] = float(fees.sum())
    out["total_slippage_estimate"] = float((
        config.SLIPPAGE * (trades["entry_px"].astype(float) + trades["exit_px"].astype(float))
        * trades["qty"].astype(float)).sum())
    out["total_trades"] = int(len(trades))
    out["winning_trades"] = int(wins.sum())
    out["losing_trades"] = int(losses.sum())
    out["win_rate_pct"] = float(wins.mean() * 100)
    out["avg_winner"] = float(net[wins].mean()) if wins.any() else 0.0
    out["avg_loser"] = float(net[losses].mean()) if losses.any() else 0.0
    out["payoff_ratio"] = (abs(out["avg_winner"] / out["avg_loser"])
                            if out["avg_loser"] != 0 else None)
    out["expectancy_per_trade"] = float(net.mean())
    gross_win = net[wins].sum() if wins.any() else 0.0
    gross_loss = abs(net[losses].sum()) if losses.any() else 0.0
    out["profit_factor"] = float(gross_win / gross_loss) if gross_loss > 0 else None

    dd = _drawdown_series(eq)
    out["max_drawdown_pct"] = float(dd.min()) if len(dd) else 0.0
    out["max_drawdown_duration_hours"] = _max_dd_duration(ts_ms, eq) if len(eq) else 0.0

    bars_per_year = 365 * 24 * 60 / 5     # 5m bars/year, chart timeframe
    bar_rets = np.diff(eq) / np.array(eq[:-1]) if len(eq) > 1 else np.array([])
    sharpe, sortino = _sharpe_sortino(bar_rets, bars_per_year)
    out["sharpe_ratio_annualized"] = sharpe
    out["sortino_ratio_annualized"] = sortino

    lw, ll = _streaks(list(wins))
    out["longest_winning_streak"] = lw
    out["longest_losing_streak"] = ll

    is_long = trades["side"] == "long"
    out["long_trades"] = int(is_long.sum())
    out["short_trades"] = int((~is_long).sum())
    out["long_pnl"] = float(net[is_long].sum())
    out["short_pnl"] = float(net[~is_long].sum())

    if real_times is not None and len(real_times) == len(trades):
        hold_hours = pd.Series([(ex - en) / 3_600_000 for en, ex in real_times])
        exit_t = pd.Series(pd.to_datetime([ex for _, ex in real_times], unit="ms", utc=True))
    else:
        entry_t = pd.to_datetime(trades["entry_time"])
        exit_t = pd.to_datetime(trades["exit_time"])
        hold_hours = (exit_t - entry_t).dt.total_seconds() / 3600.0
    out["avg_holding_time_hours"] = float(hold_hours.mean())
    out["median_holding_time_hours"] = float(hold_hours.median())

    total_span_hours = (ts_ms[-1] - ts_ms[0]) / 3_600_000 if len(ts_ms) > 1 else None
    out["pct_time_in_market"] = (float(hold_hours.sum() / total_span_hours * 100)
                                  if total_span_hours else None)

    exit_t_naive = exit_t.dt.tz_localize(None) if exit_t.dt.tz is not None else exit_t
    monthly = net.groupby(exit_t_naive.dt.to_period("M")).sum()
    out["monthly_net_pnl"] = {str(k): float(v) for k, v in monthly.items()}

    if guard_counts:
        out["risk_guard_activations"] = guard_counts

    return out
