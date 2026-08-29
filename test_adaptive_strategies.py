"""Deterministic tests for adaptive/strategies.py's four causal strategy modules."""
import numpy as np
import pandas as pd
import pytest

from adaptive.strategies import (
    trend_momentum, volatility_breakout, mean_reversion, shock_continuation,
    _shock_detectors,
)

BASE = 1_800_000_000_000
HOUR = 3_600_000


def _df(closes, vol=None, wick_frac=0.001, ts_step=HOUR):
    n = len(closes)
    highs = [c * (1 + wick_frac) for c in closes]
    lows = [c * (1 - wick_frac) for c in closes]
    vols = vol if vol is not None else [100.0] * n
    return pd.DataFrame({
        "ts": [BASE + i * ts_step for i in range(n)],
        "open": closes, "high": highs, "low": lows, "close": closes, "volume": vols,
    })


# ── A. trend_momentum ────────────────────────────────────────────────
def test_trend_momentum_fires_long_on_strong_uptrend():
    n = 50
    closes = [100.0 * (1.002 ** i) for i in range(n)]
    sig = trend_momentum(_df(closes))
    assert sig.direction == "long"
    assert sig.strength > 0
    assert "TREND_UP" in sig.regime_compatibility


def test_trend_momentum_exits_on_downtrend():
    n = 50
    closes = [100.0 * (0.998 ** i) for i in range(n)]
    sig = trend_momentum(_df(closes))
    assert sig.direction == "exit_long"
    assert sig.bearish_research_strength > 0


def test_trend_momentum_params_override_changes_threshold():
    """Proves champion/challenger adaptation (adaptive/adaptation.py) can
    actually change live behavior -- a moderate uptrend that doesn't clear
    the default z_threshold=1.0 DOES clear a looser challenger threshold."""
    n = 50
    closes = [100.0 * (1.0003 ** i) for i in range(n)]  # gentle uptrend, below default z_threshold
    default_sig = trend_momentum(_df(closes))
    loose_sig = trend_momentum(_df(closes), params={"z_threshold": 0.1})
    assert default_sig.direction is None
    assert loose_sig.direction == "long"


def test_trend_momentum_flat_in_range():
    n = 50
    closes = [100.0 + (i % 3) * 0.01 for i in range(n)]
    sig = trend_momentum(_df(closes))
    assert sig.direction is None


# ── B. volatility_breakout ───────────────────────────────────────────
def test_breakout_fires_long_above_prior_high_with_volume():
    n = 30
    closes = [100.0] * (n - 1) + [105.0]  # sharp break above the 20-bar prior high
    vols = [100.0] * (n - 1) + [500.0]    # high relative volume on the breakout bar
    sig = volatility_breakout(_df(closes, vol=vols))
    assert sig.direction == "long"


def test_breakout_does_not_fire_without_volume_confirmation():
    n = 200  # long enough for volume_percentile to be computed
    closes = [100.0] * (n - 1) + [105.0]
    vols = [100.0] * (n - 1) + [50.0]  # LOW relative volume despite the price break
    sig = volatility_breakout(_df(closes, vol=vols))
    assert sig.direction is None


def test_breakout_exits_on_breakdown_below_prior_low():
    n = 30
    closes = [100.0] * (n - 1) + [95.0]
    sig = volatility_breakout(_df(closes))
    assert sig.direction == "exit_long"


# ── C. mean_reversion ─────────────────────────────────────────────────
def test_mean_reversion_fires_long_on_oversold_dip():
    n = 25
    closes = [100.0] * (n - 1) + [90.0]  # sharp dip well below the rolling mean
    sig = mean_reversion(_df(closes))
    assert sig.direction == "long"
    assert "RANGE" in sig.regime_compatibility


def test_mean_reversion_exits_at_or_above_mean():
    n = 25
    closes = [90.0] * (n - 1) + [100.0]  # jumped up to/above the rolling mean
    sig = mean_reversion(_df(closes))
    assert sig.direction == "exit_long"


def test_mean_reversion_flat_when_near_mean():
    n = 25
    closes = [100.0 + (i % 2) * 0.1 for i in range(n)]
    sig = mean_reversion(_df(closes))
    assert sig.direction is None


# ── D. shock_continuation (frozen definition) ──────────────────────────
def test_shock_continuation_fires_long_on_up_shock():
    _shock_detectors.clear()
    symbol = "TEST/USDT"
    df = _df([100.0] * 20)  # warm up ATR
    for i in range(20):
        shock_continuation(symbol, df.iloc[:i + 1])
    shock_row = pd.DataFrame([{
        "ts": BASE + 20 * HOUR, "open": 100.0, "high": 115.0, "low": 99.0, "close": 114.0, "volume": 500.0,
    }])
    df_with_shock = pd.concat([df, shock_row], ignore_index=True)
    sig = shock_continuation(symbol, df_with_shock)
    assert sig.direction == "long"
    assert sig.stop_pct == 4.0


def test_shock_continuation_exits_never_shorts_on_down_shock():
    _shock_detectors.clear()
    symbol = "TEST2/USDT"
    df = _df([100.0] * 20)
    for i in range(20):
        shock_continuation(symbol, df.iloc[:i + 1])
    shock_row = pd.DataFrame([{
        "ts": BASE + 20 * HOUR, "open": 100.0, "high": 101.0, "low": 85.0, "close": 86.0, "volume": 500.0,
    }])
    df_with_shock = pd.concat([df, shock_row], ignore_index=True)
    sig = shock_continuation(symbol, df_with_shock)
    assert sig.direction == "exit_long"  # never "short"
    assert sig.bearish_research_strength == 1.0  # visible for research, NOT spot-executable


def test_shock_continuation_is_stateful_per_symbol():
    _shock_detectors.clear()
    assert "A/USDT" not in _shock_detectors
    shock_continuation("A/USDT", _df([100.0, 101.0]))
    assert "A/USDT" in _shock_detectors
    assert "B/USDT" not in _shock_detectors
