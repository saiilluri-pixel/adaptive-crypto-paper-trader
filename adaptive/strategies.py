"""
Five independent, causal strategy modules producing a common StrategySignal.

Spot-legal semantics (adaptive spec section 1 -- no naked shorting): a
bearish reading only ever produces direction="exit_long" (close an existing
position in this symbol, never open one) or None. A strategy's raw bearish
strength is additionally exposed via `bearish_research_strength` purely for
research/statistics visibility -- explicitly labeled NON-SPOT-EXECUTABLE,
never fed into sizing, entries, or official portfolio performance.

Shock Continuation (D) reuses research/strategies/shock_aftershock.py's
_ShockBase._shock_direction() verbatim -- the exact frozen, already-
validated shock definition, not a reimplementation. Trend Momentum,
Volatility Breakout, and Mean Reversion (A/B/C) are new, deliberately
simple, fully causal implementations against the adaptive regime feature
set; they are not ports of the legacy research/strategies/*.py modules
(those use a different single-timeframe interface built for the V2/V3
backtest harness). ATR Trailing Stop (E) is a from-scratch, faithful causal
port of a user-supplied Pine Script v5 indicator ("ATR Trailing Stoploss"
by ceyhun, MPL-2.0) -- the algorithm (an ATR-offset rolling-high trailing
line with close-crossover entries/exits) is reimplemented independently in
Python here, not a transliteration of the Pine source. Prime-Swing itself
is not used here and receives no special treatment, per the adaptive spec
-- and a second user-supplied script ("Prime Strategy Swing") was
deliberately NOT added as a 6th strategy: it is the same structure-break/
Nadaraya-Watson/Fibonacci-zone strategy family already running live in the
legacy com.btcpaper.bot (strategy.py), so duplicating it here would
double-count the same signal rather than add a genuinely independent one.
"""
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from research.strategies.shock_aftershock import _ShockBase  # noqa: E402
from adaptive.regime import compute_features  # noqa: E402


@dataclass
class StrategySignal:
    strategy: str
    direction: Optional[str]          # "long" | "exit_long" | None -- SPOT-EXECUTABLE only
    strength: float                   # 0..1, raw signal strength before any shrinkage/confidence weighting
    expected_rr: float                # expected reward:risk ratio
    stop_pct: float                   # suggested initial stop distance, percent of entry price
    exit_plan: str
    regime_compatibility: List[str]
    bearish_research_strength: float = 0.0   # NON-SPOT-EXECUTABLE, research/logging only


# ── A. Trend / momentum (primary timeframe: 1h) ─────────────────────────
# Tunable via `params` (champion/challenger adaptation target -- see
# adaptive/adaptation.py). shock_continuation deliberately has NO tunable
# params here: it reuses the already-frozen, already-validated shock
# definition and is excluded from reparameterization by design.
TREND_DEFAULT_PARAMS = {"z_threshold": 1.0}


def trend_momentum(df_1h: pd.DataFrame, params: Optional[dict] = None) -> StrategySignal:
    p = {**TREND_DEFAULT_PARAMS, **(params or {})}
    feat = compute_features(df_1h)
    if feat is None:
        return StrategySignal("trend_momentum", None, 0.0, 0.0, 0.0, "n/a", ["TREND_UP"])
    z = feat.trend_slope_z
    if z >= p["z_threshold"]:
        strength = float(np.clip(z / 3.0, 0.0, 1.0))
        return StrategySignal("trend_momentum", "long", strength, expected_rr=2.0,
                               stop_pct=max(1.5, feat.atr_pct_of_price * 250),
                               exit_plan="ATR trailing stop, ride the trend",
                               regime_compatibility=["TREND_UP"])
    if z <= -p["z_threshold"]:
        bearish = float(np.clip(-z / 3.0, 0.0, 1.0))
        return StrategySignal("trend_momentum", "exit_long", 0.0, 0.0, 0.0,
                               "exit on trend reversal", ["TREND_UP"],
                               bearish_research_strength=bearish)
    return StrategySignal("trend_momentum", None, 0.0, 0.0, 0.0, "n/a", ["TREND_UP"])


# ── B. Breakout / volatility expansion (primary timeframe: 15m) ────────
BREAKOUT_DEFAULT_PARAMS = {"n": 20, "volume_percentile": 0.70}


def volatility_breakout(df_15m: pd.DataFrame, params: Optional[dict] = None) -> StrategySignal:
    p = {**BREAKOUT_DEFAULT_PARAMS, **(params or {})}
    n = p["n"]
    if len(df_15m) < n + 2:
        return StrategySignal("volatility_breakout", None, 0.0, 0.0, 0.0, "n/a", ["HIGH_VOLATILITY", "TREND_UP"])
    feat = compute_features(df_15m)
    if feat is None:
        return StrategySignal("volatility_breakout", None, 0.0, 0.0, 0.0, "n/a", ["HIGH_VOLATILITY", "TREND_UP"])
    closes = df_15m["close"].values
    highs = df_15m["high"].values
    prior_high = float(np.max(highs[-n - 1:-1]))  # excludes the current bar -- causal
    last_close = float(closes[-1])
    vol_ok = feat.volume_percentile is None or feat.volume_percentile >= p["volume_percentile"]
    if last_close > prior_high and vol_ok:
        breakout_pct = (last_close / prior_high - 1)
        strength = float(np.clip(breakout_pct / (feat.atr_pct_of_price * 2 + 1e-9), 0.0, 1.0))
        return StrategySignal("volatility_breakout", "long", strength, expected_rr=1.5,
                               stop_pct=max(1.0, feat.atr_pct_of_price * 150),
                               exit_plan="breakeven after 1R, then ATR trail",
                               regime_compatibility=["HIGH_VOLATILITY", "TREND_UP"])
    prior_low = float(np.min(df_15m["low"].values[-n - 1:-1]))
    if last_close < prior_low:
        bearish = float(np.clip((prior_low / last_close - 1) / (feat.atr_pct_of_price * 2 + 1e-9), 0.0, 1.0))
        return StrategySignal("volatility_breakout", "exit_long", 0.0, 0.0, 0.0,
                               "exit on breakdown", ["HIGH_VOLATILITY", "TREND_UP"],
                               bearish_research_strength=bearish)
    return StrategySignal("volatility_breakout", None, 0.0, 0.0, 0.0, "n/a", ["HIGH_VOLATILITY", "TREND_UP"])


# ── C. Mean reversion (primary timeframe: 15m) ──────────────────────────
MEANREV_DEFAULT_PARAMS = {"n": 20, "z_entry": -1.5, "z_exit": 0.0}


def mean_reversion(df_15m: pd.DataFrame, params: Optional[dict] = None) -> StrategySignal:
    p = {**MEANREV_DEFAULT_PARAMS, **(params or {})}
    n = p["n"]
    if len(df_15m) < n + 1:
        return StrategySignal("mean_reversion", None, 0.0, 0.0, 0.0, "n/a", ["RANGE", "LOW_VOLATILITY"])
    feat = compute_features(df_15m)
    if feat is None:
        return StrategySignal("mean_reversion", None, 0.0, 0.0, 0.0, "n/a", ["RANGE", "LOW_VOLATILITY"])
    closes = df_15m["close"].tail(n)
    mean, std = float(closes.mean()), float(closes.std())
    if std <= 0:
        return StrategySignal("mean_reversion", None, 0.0, 0.0, 0.0, "n/a", ["RANGE", "LOW_VOLATILITY"])
    last = float(closes.iloc[-1])
    z = (last - mean) / std
    if z <= p["z_entry"]:
        strength = float(np.clip(-z / 3.0, 0.0, 1.0))
        return StrategySignal("mean_reversion", "long", strength, expected_rr=1.2,
                               stop_pct=max(1.0, feat.atr_pct_of_price * 200),
                               exit_plan="exit as price reverts to the rolling mean",
                               regime_compatibility=["RANGE", "LOW_VOLATILITY"])
    if z >= p["z_exit"]:
        return StrategySignal("mean_reversion", "exit_long", 0.0, 0.0, 0.0,
                               "reverted to mean -- take profit", ["RANGE", "LOW_VOLATILITY"])
    return StrategySignal("mean_reversion", None, 0.0, 0.0, 0.0, "n/a", ["RANGE", "LOW_VOLATILITY"])


# ── D. Shock continuation (primary timeframe: 1h) -- frozen, validated ─
_shock_detectors: Dict[str, _ShockBase] = {}


def _get_shock_detector(symbol: str) -> _ShockBase:
    if symbol not in _shock_detectors:
        _shock_detectors[symbol] = _ShockBase({"ATR_N": 14, "SHOCK_ATR_MULT": 3.0})
    return _shock_detectors[symbol]


def shock_continuation(symbol: str, df_1h: pd.DataFrame) -> StrategySignal:
    """Stateful: must be called with EVERY new closed 1h bar for `symbol`,
    in order, exactly once per bar -- mirrors the frozen shadow's own
    shadow_step() contract. Uses only the latest bar plus the detector's
    own rolling internal state (never re-scans df_1h from scratch)."""
    if len(df_1h) < 2:
        return StrategySignal("shock_continuation", None, 0.0, 0.0, 0.0, "n/a", ["SHOCK"])
    det = _get_shock_detector(symbol)
    row = df_1h.iloc[-1]
    d = det._shock_direction(float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"]))
    if d == 1:
        return StrategySignal("shock_continuation", "long", 1.0, expected_rr=1.5, stop_pct=4.0,
                               exit_plan="ATR trailing stop, ratchet-only (frozen definition)",
                               regime_compatibility=["SHOCK"])
    if d == -1:
        return StrategySignal("shock_continuation", "exit_long", 0.0, 0.0, 0.0,
                               "down-shock -- exit, never short", ["SHOCK"],
                               bearish_research_strength=1.0)
    return StrategySignal("shock_continuation", None, 0.0, 0.0, 0.0, "n/a", ["SHOCK"])


# ── E. ATR Trailing Stop (from-scratch causal port, tunable) ───────────
# Ported from the "ATR Trailing Stoploss" Pine Script v5 indicator by
# ceyhun (Mozilla Public License 2.0), supplied by the user. Algorithm,
# reimplemented independently here, not a line-by-line Pine translation:
#   1. True range + Wilder RMA-smoothed ATR (matches Pine's ta.atr() exactly
#      -- NOT the simple rolling-mean ATR convention used elsewhere in this
#      codebase for regime.py/shock_aftershock.py; this strategy alone uses
#      Wilder smoothing to stay faithful to its Pine source).
#   2. basis[i] = high[i] - mult * atr[i]
#   3. TS[i] = rolling max of basis over the trailing hhv_period bars
#      (Pine's ta.highest) -- or close[i] itself during the first
#      warmup_bars bars, exactly mirroring the source's `cum_1 < 16` guard.
#   4. long on a close-crosses-above-TS event; exit_long (never short) on
#      close-crosses-below-TS, mirroring Pine's ta.crossover/ta.crossunder
#      (both bars' close-vs-TS relationship, not just the current bar's).
# Runs on the 4h chart -- the one timeframe none of the other four
# strategies use, filling out the multi-timeframe coverage the adaptive
# spec originally called for (5m execution / 15m tactical / 1h regime /
# 4h context).
ATR_TS_DEFAULT_PARAMS = {"atr_period": 5, "hhv_period": 10, "mult": 2.5, "warmup_bars": 16}


def _wilder_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> np.ndarray:
    """Pine's ta.atr(period): RMA (Wilder) smoothing of true range, seeded
    with the simple mean of the first `period` true ranges. NaN before the
    seed point, matching Pine's na-until-warmed behavior."""
    n = len(highs)
    trs = np.empty(n)
    trs[0] = highs[0] - lows[0]
    for i in range(1, n):
        trs[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
    atr = np.full(n, np.nan)
    if n >= period:
        atr[period - 1] = trs[:period].mean()
        alpha = 1.0 / period
        for i in range(period, n):
            atr[i] = alpha * trs[i] + (1 - alpha) * atr[i - 1]
    return atr


def atr_trailing_stop(df_4h: pd.DataFrame, params: Optional[dict] = None) -> StrategySignal:
    p = {**ATR_TS_DEFAULT_PARAMS, **(params or {})}
    atr_n, hhv_n, mult, warmup = p["atr_period"], p["hhv_period"], p["mult"], p["warmup_bars"]
    min_bars = max(atr_n, hhv_n, warmup) + 2  # +2: need two consecutive valid TS values for a crossover
    if len(df_4h) < min_bars:
        return StrategySignal("atr_trailing_stop", None, 0.0, 0.0, 0.0, "n/a", ["TREND_UP", "HIGH_VOLATILITY"])

    highs = df_4h["high"].values.astype(float)
    lows = df_4h["low"].values.astype(float)
    closes = df_4h["close"].values.astype(float)
    n = len(df_4h)

    atr = _wilder_atr(highs, lows, closes, atr_n)
    basis = highs - mult * atr  # NaN wherever atr is NaN (still-warming)
    rolling_max = pd.Series(basis).rolling(hhv_n, min_periods=hhv_n).max().values  # causal: trailing window only

    ts = np.where(np.arange(n) < warmup - 1, closes, rolling_max)

    if n < 2 or np.isnan(ts[-1]) or np.isnan(ts[-2]):
        return StrategySignal("atr_trailing_stop", None, 0.0, 0.0, 0.0, "n/a", ["TREND_UP", "HIGH_VOLATILITY"])

    prev_close, prev_ts = closes[-2], ts[-2]
    last_close, last_ts = closes[-1], ts[-1]
    last_atr = atr[-1] if not np.isnan(atr[-1]) else 0.0

    buy = prev_close <= prev_ts and last_close > last_ts
    sell = prev_close >= prev_ts and last_close < last_ts

    if buy:
        strength = float(np.clip((last_close - last_ts) / last_atr, 0.0, 1.0)) if last_atr > 0 else 0.5
        atr_pct = (last_atr / last_close * 100) if last_close else 2.0
        return StrategySignal("atr_trailing_stop", "long", strength, expected_rr=1.8,
                               stop_pct=max(1.0, atr_pct * mult),
                               exit_plan="exit on the same ATR trailing-stop line crossing back under close",
                               regime_compatibility=["TREND_UP", "HIGH_VOLATILITY"])
    if sell:
        bearish = float(np.clip((last_ts - last_close) / last_atr, 0.0, 1.0)) if last_atr > 0 else 0.5
        return StrategySignal("atr_trailing_stop", "exit_long", 0.0, 0.0, 0.0,
                               "crossunder -- exit, never short", ["TREND_UP", "HIGH_VOLATILITY"],
                               bearish_research_strength=bearish)
    return StrategySignal("atr_trailing_stop", None, 0.0, 0.0, 0.0, "n/a", ["TREND_UP", "HIGH_VOLATILITY"])


STRATEGY_NAMES = ("trend_momentum", "volatility_breakout", "mean_reversion", "shock_continuation",
                   "atr_trailing_stop")
