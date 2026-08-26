"""
Phase 2, step 8: simple, transparent, non-optimized regime classification.

Definitions (fixed thresholds, not fit to performance):
  - Daily closes, 30-day rolling return r30.
  - Trend regime:  r30 > +15%  -> "strong_uptrend"
                    r30 < -15% -> "strong_downtrend"
                    else       -> "sideways"
  - Volatility regime: 30-day rolling annualized realized vol vs. that
    symbol's OWN median vol over the full window -> "high_vol" / "low_vol".

PnL is attributed by BAR (equity delta between consecutive processed bars),
tagged by that bar's regime -- not by trade entry date. Trades don't map
cleanly to a single regime (a position can span a regime change, and
trades_*.csv's entry_time is backtest run-time, not bar time -- see
engine_runner.real_trade_times()), whereas the equity curve has genuine
bar timestamps for every bar, making bar-level attribution both simpler
and more robust.
"""
import numpy as np
import pandas as pd


def classify_regimes(chart_df):
    """chart_df: the 5m OHLCV DataFrame with a 'dt' column. Returns a daily
    DataFrame indexed by date with trend_regime / vol_regime columns."""
    daily = chart_df.set_index("dt")["close"].resample("1D").last().dropna()
    r30 = daily.pct_change(30) * 100

    trend = pd.Series(index=daily.index, dtype=object)
    trend[r30 > 15] = "strong_uptrend"
    trend[r30 < -15] = "strong_downtrend"
    trend[(r30 >= -15) & (r30 <= 15)] = "sideways"

    daily_ret = daily.pct_change()
    vol30 = daily_ret.rolling(30).std() * np.sqrt(365)
    vol_median = vol30.median()
    vol_regime = pd.Series(index=daily.index, dtype=object)
    vol_regime[vol30 > vol_median] = "high_vol"
    vol_regime[vol30 <= vol_median] = "low_vol"

    out = pd.DataFrame({"trend_regime": trend, "vol_regime": vol_regime})
    return out


def pnl_by_regime(equity_curve, regimes):
    """equity_curve: list of (ts_ms, equity), genuine bar timestamps.
    Returns {trend_regime: {...}, vol_regime: {...}} with net PnL (sum of
    bar-to-bar equity deltas) and bar counts per regime label."""
    if len(equity_curve) < 2:
        return {}
    eq = pd.DataFrame(equity_curve, columns=["ts_ms", "equity"])
    eq["dt"] = pd.to_datetime(eq["ts_ms"], unit="ms", utc=True).dt.floor("D")
    eq["delta"] = eq["equity"].diff()
    eq = eq.dropna(subset=["delta"])
    reg = regimes.reindex(pd.DatetimeIndex(eq["dt"]))
    eq["trend_regime"] = reg["trend_regime"].values
    eq["vol_regime"] = reg["vol_regime"].values

    out = {}
    for dim in ("trend_regime", "vol_regime"):
        g = eq.dropna(subset=[dim]).groupby(dim)["delta"]
        out[dim] = {k: {"net_pnl": float(v.sum()), "bars": int(v.count())}
                    for k, v in g}
    return out
