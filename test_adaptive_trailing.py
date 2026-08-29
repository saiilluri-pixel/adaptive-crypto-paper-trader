"""Deterministic tests for adaptive/trailing.py."""
import pytest

from adaptive.trailing import new_trail_state, update_trail, stop_price, check_stop_bar


def test_initial_stop_below_entry_for_long():
    ts = new_trail_state(entry_price=100.0, init_stop_pct=5.0)
    assert stop_price(ts) == pytest.approx(95.0)


def test_stop_ratchets_up_on_favorable_move():
    ts = new_trail_state(entry_price=100.0, init_stop_pct=5.0)
    initial = stop_price(ts)
    ts = update_trail(ts, close_price=110.0)  # +10% favorable
    assert stop_price(ts) > initial


def test_stop_never_moves_down_after_a_ratchet():
    ts = new_trail_state(entry_price=100.0, init_stop_pct=5.0)
    ts = update_trail(ts, close_price=120.0)
    high_water_stop = stop_price(ts)
    ts = update_trail(ts, close_price=105.0)  # price pulls back, still above entry
    assert stop_price(ts) == pytest.approx(high_water_stop)  # unchanged, never widens down


def test_monotonic_across_a_full_favorable_sequence():
    ts = new_trail_state(entry_price=100.0, init_stop_pct=5.0)
    stops = [stop_price(ts)]
    for price in [102, 105, 103, 110, 108, 115, 112, 120]:
        ts = update_trail(ts, close_price=float(price))
        stops.append(stop_price(ts))
    assert all(stops[i + 1] >= stops[i] for i in range(len(stops) - 1))


def test_check_stop_bar_ordinary_touch_uses_stop_not_low():
    ts = new_trail_state(entry_price=100.0, init_stop_pct=5.0)  # stop = 95
    fill_ref = check_stop_bar(ts, open_=97.0, high=98.0, low=90.0)
    assert fill_ref == pytest.approx(95.0)  # the stop itself, not the lower low


def test_check_stop_bar_gap_uses_open_not_stop():
    ts = new_trail_state(entry_price=100.0, init_stop_pct=5.0)  # stop = 95
    fill_ref = check_stop_bar(ts, open_=90.0, high=91.0, low=88.0)
    assert fill_ref == pytest.approx(90.0)  # gapped below the stop -- fills at the open


def test_check_stop_bar_no_touch_returns_none():
    ts = new_trail_state(entry_price=100.0, init_stop_pct=5.0)
    assert check_stop_bar(ts, open_=101.0, high=103.0, low=99.0) is None


def test_trail_state_round_trips_through_dict_serialization():
    ts = new_trail_state(entry_price=100.0, init_stop_pct=5.0)
    ts = update_trail(ts, close_price=110.0)
    assert set(ts.keys()) == {"side", "updated_entry", "base_stop_pct", "sl_pct", "sl_price"}
    ts2 = update_trail(dict(ts), close_price=115.0)  # simulate reload from JSON then continue
    assert stop_price(ts2) >= stop_price(ts)
