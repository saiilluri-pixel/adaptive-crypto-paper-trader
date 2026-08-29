"""
Per-position trailing-profit exit contract (adaptive spec section 10).

Reuses strategy.py's TrailingStop unmodified -- the same Pine-fidelity,
ratchet-only stop math already validated and extensively tested elsewhere
in this codebase (test_trailing_stop.py). Long-only here: Spot has no
naked shorting. The stop can only move UP, never widens back down after
entry -- a guarantee TrailingStop.update() already provides; this module
does not reimplement that logic, only serializes it into a Position's
`trail_state` dict (adaptive/portfolio.py) so the entry-time contract
survives a restart without needing a live TrailingStop object to persist
across a process boundary.
"""
import os
import sys
from typing import Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from strategy import TrailingStop  # noqa: E402


def new_trail_state(entry_price: float, init_stop_pct: float) -> dict:
    ts = TrailingStop("long", entry_price, init_stop_pct)
    return _serialize(ts)


def _serialize(ts: TrailingStop) -> dict:
    return {"side": ts.side, "updated_entry": ts.updated_entry,
            "base_stop_pct": ts.base_stop_pct, "sl_pct": ts.sl_pct, "sl_price": ts.sl_price}


def _deserialize(state: dict) -> TrailingStop:
    ts = TrailingStop.__new__(TrailingStop)
    ts.side = state["side"]
    ts.updated_entry = state["updated_entry"]
    ts.base_stop_pct = state["base_stop_pct"]
    ts.sl_pct = state["sl_pct"]
    ts.sl_price = state["sl_price"]
    return ts


def update_trail(trail_state: dict, close_price: float) -> dict:
    """Ratchets the stop on a closed bar. Must be called AFTER
    check_stop_bar() for the SAME bar (uses the pre-ratchet stop)."""
    ts = _deserialize(trail_state)
    ts.update(close_price)
    return _serialize(ts)


def stop_price(trail_state: dict) -> float:
    return trail_state["sl_price"]


def check_stop_bar(trail_state: dict, open_: float, high: float, low: float) -> Optional[float]:
    """OHLC-aware stop check, mirrors engine.py's check_stop_bar exactly
    (gap vs. ordinary touch), long-only. Returns the fill REFERENCE price
    (before adverse slippage -- portfolio.sell() applies that) if the stop
    was hit this bar, else None. Uses the stop as it stood BEFORE this
    bar's ratchet -- callers must check this BEFORE calling update_trail
    for the same bar, never after."""
    stop = trail_state["sl_price"]
    if open_ <= stop:
        return open_  # gap-through: earliest price this bar actually traded at
    if low <= stop:
        return stop  # ordinary intrabar touch
    return None
