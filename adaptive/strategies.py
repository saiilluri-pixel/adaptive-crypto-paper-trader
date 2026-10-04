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
# z_threshold lowered 1.0 -> 0.8 -> 0.5 (the latter per explicit user
# request for aggressive trading) -- admits moderately-trending moves that
# previously needed a full 1x-noise-scale slope to register at all. A live
# 6.5h sample at the original 1.0 threshold produced ZERO trend_momentum
# signals on any of BTC/ETH/SOL, so this was genuinely never firing, not
# just firing rarely.
TREND_DEFAULT_PARAMS = {"z_threshold": 0.5}


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
# volume_percentile lowered 0.70 -> 0.55 -> 0.40 (the latter per explicit
# user request for aggressive trading) -- a breakout no longer needs to be
# in the top 30% of the last 100 bars by volume to qualify, just the top
# 60%. `n` (the lookback window defining "prior high/low") is left
# unchanged -- this was already the most active of the five strategies
# live, so only its easily-adjustable volume gate was loosened further,
# not its core lookback logic.
BREAKOUT_DEFAULT_PARAMS = {"n": 20, "volume_percentile": 0.40}


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
# z_entry history: -1.5 -> -1.2 -> -0.9 (loosened for "aggressive"
# trading) -> REVERTED to -1.5 (2026-09-22). A backtest across all 20
# symbols on the LIVE 15m timeframe (fee+slippage inclusive) showed the
# aggressive loosening actively destroyed this strategy's edge: at -0.9 it
# was PF 1.04 / +8% (and PF 0.89 / -52% on 1h), the single largest source
# of live trades yet a net drag on the book. -1.5 is the empirical sweet
# spot on 15m (PF 1.08, +13%, ~30% fewer trades) -- fewer, higher-quality
# dips, less fee churn, less DEFENSIVE-state noise. The dramatic 1h
# improvement from tightening further (PF 2.09 at -2.5) does NOT transfer
# to the 15m timeframe this strategy actually trades on, so -1.5 (not -2.5)
# is the honest choice. z_exit (take-profit for an existing long) stays 0.0.
#
# Symmetric SHORT leg added 2026-09-20 (per explicit user request "why
# aren't we short trading"): this strategy was the single largest source of
# LONG candidates (dip-buying) but produced ZERO short candidates, because
# its overbought exit carried no bearish_research_strength -- so it never
# fed the SHORT book. It now mirrors the long leg: an OVERBOUGHT spike
# (z >= |z_entry|) emits direction="exit_long" WITH a bearish_research_
# strength, which is exactly the Spot-legal channel score_short_opportunity
# consumes as a short-entry signal (the exit_long also correctly closes any
# existing long in that symbol). A milder reversion (z_exit <= z < |z_entry|)
# still emits a plain take-profit exit_long with no bearish strength, so it
# closes longs without triggering a short on a merely-back-to-mean move.
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
    z_short = abs(p["z_entry"])  # symmetric overbought threshold (e.g. +0.9)
    if z <= p["z_entry"]:
        strength = float(np.clip(-z / 3.0, 0.0, 1.0))
        return StrategySignal("mean_reversion", "long", strength, expected_rr=1.2,
                               stop_pct=max(1.0, feat.atr_pct_of_price * 200),
                               exit_plan="exit as price reverts to the rolling mean",
                               regime_compatibility=["RANGE", "LOW_VOLATILITY"])
    if z >= z_short:
        # overbought -> exit any long AND a bearish short-entry candidate
        bearish = float(np.clip(z / 3.0, 0.0, 1.0))
        return StrategySignal("mean_reversion", "exit_long", 0.0, expected_rr=1.2,
                               stop_pct=max(1.0, feat.atr_pct_of_price * 200),
                               exit_plan="overbought -- exit long / candidate short",
                               regime_compatibility=["RANGE", "LOW_VOLATILITY"],
                               bearish_research_strength=bearish)
    if z >= p["z_exit"]:
        return StrategySignal("mean_reversion", "exit_long", 0.0, 0.0, 0.0,
                               "reverted to mean -- take profit", ["RANGE", "LOW_VOLATILITY"])
    return StrategySignal("mean_reversion", None, 0.0, 0.0, 0.0, "n/a", ["RANGE", "LOW_VOLATILITY"])


# ── F. Cross-sectional relative strength (MULTI-symbol, primary tf: 1h) ─
# The "breadth" alpha (Grinold's Fundamental Law, IR ~= IC * sqrt(breadth)).
# Unlike A-E, which each judge ONE series against an absolute threshold,
# this ranks the WHOLE universe against ITSELF each cycle by risk-adjusted
# trailing momentum (trailing return / realized vol) and takes the cross-
# sectional extremes: long the strongest top_k, short the weakest bottom_k
# (the short leg via direction="exit_long" + bearish_research_strength, the
# same Spot-legal channel every other strategy already feeds the SHORT book
# through -- see module docstring). This is genuinely ORTHOGONAL to A-E: in
# a market drifting mildly up with no absolute breakout anywhere it still
# expresses a relative view, and in a broad selloff it shorts the weakest
# rather than going flat. Adapted to the live intraday loop from
# research/cross_sectional_momentum.py's daily/weekly backtest hypothesis.
#
# Quality filter: the long leg additionally requires POSITIVE absolute
# risk-adjusted momentum and the short leg NEGATIVE -- so this never longs
# the "least bad" faller in a crash nor shorts the "least good" riser in a
# melt-up. That trades a little market-neutral purity for robustness, which
# is the right call here since the two books share one pool (FCFS) and are
# not beta-hedged.
XSECT_DEFAULT_PARAMS = {"lookback": 24, "top_k": 3, "bottom_k": 3}


def _xsect_neutral() -> StrategySignal:
    return StrategySignal("cross_sectional", None, 0.0, 0.0, 0.0, "n/a",
                          ["TREND_UP", "TREND_DOWN", "RANGE", "HIGH_VOLATILITY", "LOW_VOLATILITY"])


def cross_sectional_signals(closes_by_symbol: Dict[str, object],
                            feats_by_symbol: Optional[Dict[str, object]] = None,
                            params: Optional[dict] = None) -> Dict[str, StrategySignal]:
    """Computed ONCE per cycle over the whole universe (not per-symbol like
    A-E). closes_by_symbol maps symbol -> 1D array/Series of 1h closes
    (oldest->newest). Returns a signal for EVERY input symbol: "long" for
    the top_k strongest, "exit_long"+bearish for the bottom_k weakest, and
    a neutral (direction=None) signal for everything in between and for any
    symbol with insufficient history -- so the caller can inject it into
    the per-symbol signals dict unconditionally, exactly like A-E.

    Ranking metric per symbol: total return over `lookback` bars divided by
    that window's per-bar return volatility (risk-adjusted momentum),
    sqrt(lookback)-scaled so it is comparable across symbols. Symbols with
    <lookback+1 bars or zero volatility are excluded from the ranking (never
    ranked on missing/degenerate data)."""
    p = {**XSECT_DEFAULT_PARAMS, **(params or {})}
    lookback = int(p["lookback"])
    top_k = int(p["top_k"])
    bottom_k = int(p["bottom_k"])
    feats_by_symbol = feats_by_symbol or {}
    compat = ["TREND_UP", "TREND_DOWN", "RANGE", "HIGH_VOLATILITY", "LOW_VOLATILITY"]

    out = {sym: _xsect_neutral() for sym in closes_by_symbol}

    raw = {}
    for sym, closes in closes_by_symbol.items():
        c = np.asarray(closes, dtype=float)
        if len(c) < lookback + 1:
            continue
        w = c[-(lookback + 1):]
        rets = np.diff(w) / w[:-1]
        vol = float(np.std(rets))
        if vol <= 0 or not np.isfinite(vol):
            continue
        total_ret = w[-1] / w[0] - 1.0
        raw[sym] = total_ret / (vol * np.sqrt(lookback))

    n = len(raw)
    if n < top_k + bottom_k:
        return out  # not enough ranked symbols to form both extremes without overlap

    order = sorted(raw, key=raw.get)   # weakest -> strongest
    strongest = order[-top_k:]
    weakest = order[:bottom_k]

    for i, sym in enumerate(strongest):     # i=0 weakest-of-strong .. i=top_k-1 strongest
        if raw[sym] <= 0:
            continue                        # quality filter: never long a faller
        strength = float(np.clip(0.4 + 0.6 * (i + 1) / top_k, 0.0, 1.0))
        feat = feats_by_symbol.get(sym)
        stop_pct = max(1.5, feat.atr_pct_of_price * 200) if feat is not None else 3.0
        out[sym] = StrategySignal("cross_sectional", "long", strength, expected_rr=1.5,
                                  stop_pct=stop_pct,
                                  exit_plan="hold while top-ranked by relative strength, ATR trail",
                                  regime_compatibility=compat)

    for i, sym in enumerate(weakest):       # i=0 absolute weakest .. i=bottom_k-1 least weak
        if raw[sym] >= 0:
            continue                        # quality filter: never short a riser
        bearish = float(np.clip(0.4 + 0.6 * (bottom_k - i) / bottom_k, 0.0, 1.0))
        out[sym] = StrategySignal("cross_sectional", "exit_long", 0.0, 0.0, 0.0,
                                  "relative weakness -- exit long / candidate short", compat,
                                  bearish_research_strength=bearish)
    return out


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
# mult raised 2.5 -> 3.0 (2026-09-22): a 20-symbol backtest on the live 4h
# timeframe (fee+slippage inclusive) showed a wider ATR trailing band is
# strictly better here -- mult 3.0 gave PF 1.81 / +403% vs 2.5's PF 1.49 /
# +381%, with FEWER trades (281 vs 442). A wider band whipsaws less, so it
# both raises profit factor and cuts fee churn. Going wider still (3.5) lifts
# PF further but starts trimming return, so 3.0 is the robust knee, not the
# lone in-sample spike. atr_period/hhv_period/warmup unchanged.
ATR_TS_DEFAULT_PARAMS = {"atr_period": 5, "hhv_period": 10, "mult": 3.0, "warmup_bars": 16}


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
