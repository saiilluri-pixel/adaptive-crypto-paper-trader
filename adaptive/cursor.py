"""
Gap-integrity by construction.

`advance()` is the ONLY function in this codebase allowed to move a
market-data cursor forward. It refuses to advance past a hole in the candle
sequence rather than silently skipping it.

This exists because of a defect found auditing the SOL shock-continuation
forward shadow (shadow/shock_continuation_sol_v3_1/run_shadow.py): its
run_once_cycle() fetched a bounded "most recent N closed candles" window on
every poll. When a ~12-hour network outage exceeded that window, one bar
(2026-08-28T12:00:00Z) fell outside it and was silently dropped -- never
processed, never logged, no reconciliation flag set, no error raised. Gap
detection existed only at bootstrap/restart, never in the steady-state
polling path.

Two structural fixes, both enforced here:
  1. Catch-up fetches must page FORWARD from the cursor (since=cursor+1ms),
     never request "the most recent N ending now" -- the latter is anchored
     to wall-clock time, not to what has already been processed, which is
     exactly the shape of the bug above.
  2. Every returned bar's timestamp must be EXACTLY one timeframe after the
     previous one (or after the cursor, for the first bar). Any other delta
     -- a hole, a duplicate, an out-of-order bar -- stops consumption at
     that point and reports a gap, rather than jumping past it.
"""
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import pandas as pd


@dataclass
class CursorResult:
    bars: pd.DataFrame            # new CLOSED bars safe to process, in order, contiguous
    new_cursor_ts: Optional[int]  # ts (open time) of the last bar in `bars`; None if empty
    gap_detected: bool
    gap_detail: Optional[str] = None
    is_cold_start: bool = False   # last_cursor_ts was None -- no prior cursor to violate


def advance(fetch_since_fn: Callable[[int, int], pd.DataFrame],
            last_cursor_ts: Optional[int], timeframe_ms: int,
            limit: int = 1000) -> CursorResult:
    """
    fetch_since_fn(since_ts_ms, limit) -> DataFrame of CLOSED candles only
    (columns: ts, open, high, low, close, volume; ts = candle OPEN time,
    ascending, already excludes any still-forming candle), starting at or
    after since_ts_ms. Callers pass a closure over their own Feed/ccxt call.

    Cold start (last_cursor_ts is None): there is no prior cursor to
    validate against, so the first-ever fetch cannot itself be "a gap" --
    returned as-is with gap_detected=False, is_cold_start=True. Callers
    must treat a cold-start batch as indicator-warmup only (no entries),
    exactly like the SOL shadow's bootstrap() phase -- this module does not
    make that policy decision, it only guarantees contiguity going forward.
    """
    if last_cursor_ts is None:
        since = None
        fetched = fetch_since_fn(0, limit)
    else:
        since = last_cursor_ts + 1
        fetched = fetch_since_fn(since, limit)

    if fetched is None or len(fetched) == 0:
        return CursorResult(bars=fetched if fetched is not None else pd.DataFrame(),
                             new_cursor_ts=None, gap_detected=False,
                             is_cold_start=(last_cursor_ts is None))

    fetched = fetched.reset_index(drop=True)  # NOT sorted -- an out-of-order response from
    # the exchange is itself a data-integrity anomaly (Binance REST klines are always
    # ascending in practice) and must be caught as a gap below, not silently corrected

    if last_cursor_ts is None:
        return CursorResult(bars=fetched, new_cursor_ts=int(fetched["ts"].iloc[-1]),
                             gap_detected=False, is_cold_start=True)

    expected = last_cursor_ts + timeframe_ms
    good_rows = []
    for _, row in fetched.iterrows():
        ts = int(row["ts"])
        if ts != expected:
            detail = (f"expected next bar at ts={expected}, got ts={ts} "
                      f"(delta={ts - expected}ms = {(ts - expected) / timeframe_ms:+.1f} bars) "
                      f"-- stopping before the gap, not skipping past it")
            good_df = pd.DataFrame(good_rows) if good_rows else fetched.iloc[0:0]
            return CursorResult(
                bars=good_df,
                new_cursor_ts=int(good_df["ts"].iloc[-1]) if len(good_df) else last_cursor_ts,
                gap_detected=True, gap_detail=detail)
        good_rows.append(row)
        expected += timeframe_ms

    good_df = pd.DataFrame(good_rows).reset_index(drop=True)
    return CursorResult(bars=good_df, new_cursor_ts=int(good_df["ts"].iloc[-1]),
                         gap_detected=False)
