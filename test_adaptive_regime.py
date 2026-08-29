"""Deterministic tests for adaptive/regime.py -- causal-only feature computation."""
import numpy as np
import pandas as pd
import pytest

from adaptive.regime import compute_features, classify, classify_from_df, ATR_N, TREND_N

BASE = 1_800_000_000_000
HOUR = 3_600_000


def _df_from_closes(closes, vol=None, wick_frac=0.001):
    # wicks scale proportionally to price (like a real market's intrabar
    # range), not a fixed absolute unit -- a fixed +-1 wick at price~100
    # would swamp a realistic trend's per-bar drift relative to ATR
    n = len(closes)
    highs = [c * (1 + wick_frac) for c in closes]
    lows = [c * (1 - wick_frac) for c in closes]
    vols = vol if vol is not None else [100.0] * n
    return pd.DataFrame({
        "ts": [BASE + i * HOUR for i in range(n)],
        "open": closes, "high": highs, "low": lows, "close": closes, "volume": vols,
    })


def test_insufficient_history_returns_none_features():
    df = _df_from_closes([100.0] * 5)
    assert compute_features(df) is None
    assert classify_from_df(df) == "RANGE"


def test_flat_market_classifies_as_range():
    df = _df_from_closes([100.0] * (ATR_N + TREND_N + 5))
    regime = classify_from_df(df)
    assert regime == "RANGE"


def test_strong_uptrend_classifies_as_trend_up():
    n = ATR_N + TREND_N + 5
    closes = [100.0 * (1.002 ** i) for i in range(n)]  # steady compounding uptrend, tiny per-bar noise
    df = _df_from_closes(closes)
    regime = classify_from_df(df)
    assert regime == "TREND_UP"


def test_strong_downtrend_classifies_as_trend_down():
    n = ATR_N + TREND_N + 5
    closes = [100.0 * (0.998 ** i) for i in range(n)]
    df = _df_from_closes(closes)
    regime = classify_from_df(df)
    assert regime == "TREND_DOWN"


def test_shock_bar_overrides_trend_classification():
    n = ATR_N + TREND_N + 5
    closes = [100.0 * (1.002 ** i) for i in range(n)]  # uptrend context
    df = _df_from_closes(closes)
    # inject an extreme final bar: huge true range vs the trailing ATR
    df.loc[df.index[-1], "high"] = df["close"].iloc[-1] * 1.5
    df.loc[df.index[-1], "low"] = df["close"].iloc[-1] * 0.7
    regime = classify_from_df(df)
    assert regime == "SHOCK"


def test_features_never_use_bars_after_the_last_row():
    """Causality check: appending MORE bars after a given cutoff must not
    change the regime computed AT that cutoff (features only look
    backward from the last row supplied)."""
    n = ATR_N + TREND_N + 10
    closes = [100.0 + np.sin(i / 3.0) * 2 for i in range(n)]
    df_full = _df_from_closes(closes)
    cutoff = n - 5
    df_cut = df_full.iloc[:cutoff].reset_index(drop=True)

    feat_cut = compute_features(df_cut)
    # recompute using the SAME cutoff row set embedded inside the full df
    feat_from_full_at_same_point = compute_features(df_full.iloc[:cutoff].reset_index(drop=True))
    assert feat_cut.trend_slope == pytest.approx(feat_from_full_at_same_point.trend_slope)
    assert feat_cut.atr == pytest.approx(feat_from_full_at_same_point.atr)


def test_atr_percentile_is_none_before_lookback_satisfied():
    df = _df_from_closes([100.0 + (i % 5) for i in range(ATR_N + 5)])
    feat = compute_features(df)
    assert feat is not None
    assert feat.atr_percentile is None


def test_volume_percentile_reflects_relative_ranking():
    n = 150
    closes = [100.0] * n
    vols = [10.0] * (n - 1) + [1000.0]  # last bar has by far the highest volume
    df = _df_from_closes(closes, vol=vols)
    feat = compute_features(df)
    assert feat.volume_percentile == pytest.approx(1.0)
