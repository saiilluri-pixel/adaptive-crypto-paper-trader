"""
Per-symbol market regime classification from causal features only -- never
reads a bar beyond the one just closed. Six labels: TREND_UP, TREND_DOWN,
RANGE, HIGH_VOLATILITY, LOW_VOLATILITY, SHOCK.

Deliberately simple and transparent over "more sophisticated": trend
strength uses a z-scored slope (trend move size relative to ATR-based
noise) rather than full Wilder ADX -- same idea (is the recent move large
relative to typical noise?), fewer moving parts, easier to audit. Documented
here rather than buried, per the adaptive spec's "transparent causal
features" requirement.

Priority order when multiple conditions could apply (checked top to
bottom, first match wins): SHOCK overrides everything else since it marks
a specific abnormal bar, not a persistent state; then volatility extremes;
then trend strength; RANGE is the default when nothing else fires.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

SHOCK_ATR_N = 14
SHOCK_ATR_MULT = 3.0          # identical threshold to the frozen SOL shock-continuation definition
ATR_N = 14
VOL_LOOKBACK = 100            # bars used for the ATR-percentile ranking
TREND_N = 20                  # bars used for the trend-slope measurement
HIGH_VOL_PERCENTILE = 0.80
LOW_VOL_PERCENTILE = 0.20
TREND_Z_THRESHOLD = 1.0        # slope must exceed 1x the trailing noise scale to count as trending

REGIME_LABELS = ("TREND_UP", "TREND_DOWN", "RANGE", "HIGH_VOLATILITY", "LOW_VOLATILITY", "SHOCK")


@dataclass
class RegimeFeatures:
    atr: float
    atr_pct_of_price: float
    atr_percentile: Optional[float]   # None until VOL_LOOKBACK bars are available
    trend_slope: float                # (close - close_N_ago) / close_N_ago
    trend_slope_z: float              # trend_slope normalized by trailing noise scale
    volume_percentile: Optional[float]
    is_shock_bar: bool
    tr_last: float


def _true_range(h, l, prev_close):
    if prev_close is None:
        return h - l
    return max(h - l, abs(h - prev_close), abs(l - prev_close))


def compute_features(df: pd.DataFrame) -> Optional[RegimeFeatures]:
    """df: closed-bar OHLCV, columns ts/open/high/low/close/volume, ascending.
    Returns None if there isn't enough history yet (ATR_N bars minimum)."""
    if len(df) < ATR_N + 1:
        return None

    highs, lows, closes = df["high"].values, df["low"].values, df["close"].values
    trs = []
    prev_close = None
    for h, l, c in zip(highs, lows, closes):
        trs.append(_true_range(h, l, prev_close))
        prev_close = c
    trs = np.array(trs)

    atr_series = pd.Series(trs).rolling(ATR_N).mean()
    atr = float(atr_series.iloc[-1])
    if atr <= 0 or np.isnan(atr):
        return None
    price = float(closes[-1])
    atr_pct_of_price = atr / price

    atr_pct_series = (atr_series / pd.Series(closes)).dropna()
    atr_percentile = None
    if len(atr_pct_series) >= VOL_LOOKBACK:
        window = atr_pct_series.tail(VOL_LOOKBACK)
        atr_percentile = float((window <= window.iloc[-1]).mean())

    if len(df) >= TREND_N + 1:
        ref_price = float(closes[-1 - TREND_N])
        trend_slope = (price - ref_price) / ref_price if ref_price else 0.0
    else:
        trend_slope = 0.0
    noise_scale = atr_pct_of_price * np.sqrt(TREND_N)
    trend_slope_z = trend_slope / noise_scale if noise_scale > 0 else 0.0

    volume_percentile = None
    if "volume" in df.columns and len(df) >= VOL_LOOKBACK:
        vol_window = df["volume"].tail(VOL_LOOKBACK)
        volume_percentile = float((vol_window <= vol_window.iloc[-1]).mean())

    tr_last = float(trs[-1])
    atr_before_last = float(atr_series.iloc[-2]) if len(atr_series) >= 2 and not np.isnan(atr_series.iloc[-2]) else atr
    is_shock_bar = atr_before_last > 0 and tr_last > SHOCK_ATR_MULT * atr_before_last

    return RegimeFeatures(atr=atr, atr_pct_of_price=atr_pct_of_price, atr_percentile=atr_percentile,
                           trend_slope=trend_slope, trend_slope_z=trend_slope_z,
                           volume_percentile=volume_percentile, is_shock_bar=is_shock_bar, tr_last=tr_last)


def classify(features: Optional[RegimeFeatures]) -> str:
    if features is None:
        return "RANGE"  # insufficient history -- treat conservatively, not as a special state
    if features.is_shock_bar:
        return "SHOCK"
    if features.atr_percentile is not None:
        if features.atr_percentile >= HIGH_VOL_PERCENTILE:
            return "HIGH_VOLATILITY"
        if features.atr_percentile <= LOW_VOL_PERCENTILE:
            return "LOW_VOLATILITY"
    if features.trend_slope_z >= TREND_Z_THRESHOLD:
        return "TREND_UP"
    if features.trend_slope_z <= -TREND_Z_THRESHOLD:
        return "TREND_DOWN"
    return "RANGE"


def classify_from_df(df: pd.DataFrame) -> str:
    return classify(compute_features(df))
