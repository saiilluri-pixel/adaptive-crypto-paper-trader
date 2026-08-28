"""
Diagnostic wrapper around the FROZEN ShockContinuationStrategy
(research/strategies/shock_aftershock.py). Delegates every bit of signal
math to the frozen class's own `_shock_direction()` -- this module contains
no reimplementation of the shock condition, direction rule, or ATR math.
It only adds a side-channel diagnostic computation (atr-normalized shock
size, candle shape) for EVERY bar once the rolling ATR window is warmed, so
events.csv can log the full continuous series, not only bars that crossed
the threshold. eligible == (a signal COULD have fired on this bar, i.e. the
shock condition was met); signal is None additionally on a doji, matching
the frozen definition exactly.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from research.strategies.shock_aftershock import ShockContinuationStrategy  # noqa: E402
from shadow.shock_continuation_sol_v3_1.frozen_spec import FROZEN_PARAMS  # noqa: E402


def build_strategy():
    return ShockContinuationStrategy(dict(FROZEN_PARAMS))


def shadow_step(strat, o, h, l, c):
    """Advances `strat` by exactly one bar. Must be called exactly once per
    bar, in chronological order (mutates strat.tr_hist/prev_close via the
    single authoritative call to _shock_direction below).

    Returns (signal, diag): signal is "long"/"short"/None, exactly what
    ShockContinuationStrategy.on_bar() would return for this bar. diag is
    None until the rolling ATR_N window is warmed (no meaningful
    atr_norm_shock before then -- matches the frozen detector's own gating),
    otherwise a dict of diagnostics for events.csv.
    """
    prev_close_before = strat.prev_close
    warmed_before = len(strat.tr_hist) == strat.atr_n
    atr_before = (sum(strat.tr_hist) / len(strat.tr_hist)) if warmed_before else None

    d = strat._shock_direction(o, h, l, c)  # authoritative -- mutates tr_hist + prev_close
    signal = "long" if d == 1 else ("short" if d == -1 else None)

    if not warmed_before or atr_before is None:
        return signal, None

    # Mirrors _ShockBase._shock_direction's own `tr` formula exactly (same
    # o,h,l,c and prev_close AS IT STOOD BEFORE this call) -- diagnostic
    # only, does not feed back into the signal.
    tr = (h - l) if prev_close_before is None else max(
        h - l, abs(h - prev_close_before), abs(l - prev_close_before))
    atr_norm = (tr / atr_before) if atr_before else None
    body = abs(c - o)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    rng = h - l if h > l else 1e-9
    eligible = bool(atr_norm is not None and atr_norm > strat.shock_mult)
    # Sanity tripwire: eligible must exactly agree with whether a non-doji
    # signal COULD have fired (signal is None on eligible dojis too, so this
    # only asserts the necessary direction, not the sufficient one).
    if signal is not None:
        assert eligible, "signal fired without eligible shock condition -- shadow_step drifted from _shock_direction"

    return signal, {
        "atr_norm_shock": atr_norm,
        "return_pct": (c / o - 1) * 100 if o else None,
        "range": rng,
        "body_frac": body / rng,
        "upper_wick_frac": upper_wick / rng,
        "lower_wick_frac": lower_wick / rng,
        "eligible": eligible,
    }
