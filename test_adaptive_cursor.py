"""Deterministic tests for adaptive/cursor.py's gap-integrity guarantee."""
import pandas as pd
import pytest

from adaptive.cursor import advance

HOUR = 3_600_000
BASE = 1_800_000_000_000


def _bars(ts_list):
    return pd.DataFrame({
        "ts": ts_list,
        "open": [100.0] * len(ts_list), "high": [101.0] * len(ts_list),
        "low": [99.0] * len(ts_list), "close": [100.0] * len(ts_list),
        "volume": [10.0] * len(ts_list),
    })


def test_cold_start_no_prior_cursor_returns_all_bars_no_gap():
    fetched = _bars([BASE, BASE + HOUR, BASE + 2 * HOUR])
    result = advance(lambda since, limit: fetched, last_cursor_ts=None, timeframe_ms=HOUR)
    assert result.is_cold_start is True
    assert result.gap_detected is False
    assert len(result.bars) == 3
    assert result.new_cursor_ts == BASE + 2 * HOUR


def test_contiguous_bars_advance_cleanly():
    cursor = BASE
    fetched = _bars([BASE + HOUR, BASE + 2 * HOUR, BASE + 3 * HOUR])
    result = advance(lambda since, limit: fetched, last_cursor_ts=cursor, timeframe_ms=HOUR)
    assert result.gap_detected is False
    assert len(result.bars) == 3
    assert result.new_cursor_ts == BASE + 3 * HOUR


def test_gap_of_three_bars_detected_entries_would_be_blocked():
    """The exact failure mode that ate 2026-08-28T12:00Z in the SOL shadow:
    cursor at T, the feed jumps straight to T+3*TF (two bars missing in
    between). Must be detected as a gap, not silently advanced past."""
    cursor = BASE
    fetched = _bars([BASE + 3 * HOUR])  # T+1 and T+2 missing entirely
    result = advance(lambda since, limit: fetched, last_cursor_ts=cursor, timeframe_ms=HOUR)
    assert result.gap_detected is True
    assert len(result.bars) == 0  # nothing safe to process -- the gap is BEFORE any of these
    assert result.new_cursor_ts == cursor  # cursor does not move past the hole
    assert "T+" not in result.gap_detail or "expected" in result.gap_detail


def test_gap_partway_through_a_batch_keeps_the_good_prefix():
    cursor = BASE
    fetched = _bars([BASE + HOUR, BASE + 2 * HOUR, BASE + 5 * HOUR])  # hole after +2h
    result = advance(lambda since, limit: fetched, last_cursor_ts=cursor, timeframe_ms=HOUR)
    assert result.gap_detected is True
    assert len(result.bars) == 2
    assert result.new_cursor_ts == BASE + 2 * HOUR


def test_duplicate_bar_is_treated_as_a_gap_not_silently_deduped():
    cursor = BASE
    fetched = _bars([BASE, BASE + HOUR])  # BASE duplicates the cursor's own bar
    result = advance(lambda since, limit: fetched, last_cursor_ts=cursor, timeframe_ms=HOUR)
    assert result.gap_detected is True
    assert len(result.bars) == 0


def test_out_of_order_bar_is_treated_as_a_gap():
    cursor = BASE
    fetched = _bars([BASE + 2 * HOUR, BASE + HOUR])  # reversed
    result = advance(lambda since, limit: fetched, last_cursor_ts=cursor, timeframe_ms=HOUR)
    assert result.gap_detected is True  # first row (BASE+2h) isn't the expected BASE+1h


def test_no_new_bars_is_not_a_gap():
    cursor = BASE
    result = advance(lambda since, limit: _bars([]), last_cursor_ts=cursor, timeframe_ms=HOUR)
    assert result.gap_detected is False
    assert len(result.bars) == 0
    assert result.new_cursor_ts is None


def test_fetch_uses_since_not_bounded_recent_n():
    """The fetch closure must be called with since=cursor+1ms -- proves
    catch-up pages forward from the cursor rather than requesting a
    wall-clock-anchored 'most recent N', the actual root cause of the
    original defect."""
    cursor = BASE
    captured = {}

    def fetch(since, limit):
        captured["since"] = since
        return _bars([BASE + HOUR])

    advance(fetch, last_cursor_ts=cursor, timeframe_ms=HOUR)
    assert captured["since"] == cursor + 1
