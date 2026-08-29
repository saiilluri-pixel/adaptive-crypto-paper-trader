"""
Four independent, causal strategy modules producing a common StrategySignal.

Spot-legal semantics (adaptive spec section 1 -- no naked shorting): a
bearish reading only ever produces direction="exit_long" (close an existing
position in this symbol, never open one) or None. A strategy's raw bearish
strength is additionally exposed via `bearish_research_strength` purely for
research/statistics visibility -- explicitly labeled NON-SPOT-EXECUTABLE,
never fed into sizing, entries, or official portfolio performance.

Shock Continuation (D) reuses research/strategies/shock_aftershock.py's
_ShockBase._shock_direction() verbatim -- the exact frozen, already-
validated shock definition, not a reimplementation. The other three (A/B/C)
are new, deliberately simple, fully causal implementations against the
adaptive regime feature set; they are not ports of the legacy research/
strategies/*.py modules (those use a different single-timeframe interface
built for the V2/V3 backtest harness). Prime-Swing itself is not used here
and receives no special treatment, per the adaptive spec.
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


STRATEGY_NAMES = ("trend_momentum", "volatility_breakout", "mean_reversion", "shock_continuation")
