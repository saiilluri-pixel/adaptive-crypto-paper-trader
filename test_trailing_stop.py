"""
Deterministic tests for the Phase 1 TrailingStop Pine-fidelity fix.

Root bug: the original Pine v4 source declares `stopLossPercent` WITHOUT
`var`, so Pine resets it to the initial configured percentage on every bar.
The Python port persisted and accumulated it (`self.sl_pct += ...`) instead,
producing a stop that could ratchet past the price observation that
generated it. This suite pins the CORRECTED (fresh-per-call) behavior and
documents the old bug as a regression, without asserting it as acceptable.

No network calls; pure TrailingStop unit tests.
"""
import pytest

from strategy import TrailingStop


# ── LONG: worked example from the audit ──────────────────────────

def test_long_initial_stop():
    ts = TrailingStop("long", entry_price=100, init_stop=5)
    assert ts.sl_price == pytest.approx(95.0)
    assert ts.updated_entry == 100.0


def test_long_worked_sequence_matches_pine():
    ts = TrailingStop("long", 100, 5)
    ts.update(106)
    assert ts.sl_price == pytest.approx(100.00, abs=1e-9)
    ts.update(110)
    assert ts.sl_price == pytest.approx(104.00, abs=1e-9)
    ts.update(111)
    assert ts.sl_price == pytest.approx(104.76, abs=1e-6)


def test_long_stop_never_above_generating_close():
    ts = TrailingStop("long", 100, 5)
    for close in (106, 110, 111):
        ts.update(close)
        assert ts.sl_price <= close


def test_long_monotonic_ratchet():
    ts = TrailingStop("long", 100, 5)
    prev = ts.sl_price
    for close in (106, 110, 111):
        ts.update(close)
        assert ts.sl_price >= prev
        prev = ts.sl_price


def test_long_unfavorable_move_does_not_ratchet():
    ts = TrailingStop("long", 100, 5)
    ts.update(106)
    stop_before, entry_before = ts.sl_price, ts.updated_entry
    ts.update(99)          # below updated_entry -> pct negative, no ratchet
    assert ts.sl_price == stop_before
    assert ts.updated_entry == entry_before


def test_long_below_threshold_move_does_not_ratchet():
    ts = TrailingStop("long", 100, 5)
    ts.update(106)                 # updated_entry -> 100.0
    stop_before, entry_before = ts.sl_price, ts.updated_entry
    ts.update(100.5)               # pct = 0.5% <= 1%, no ratchet
    assert ts.sl_price == stop_before
    assert ts.updated_entry == entry_before


def test_long_repeated_calls_below_threshold_are_idempotent():
    ts = TrailingStop("long", 100, 5)
    ts.update(106)
    snap = (ts.sl_price, ts.updated_entry, ts.sl_pct)
    for _ in range(5):
        ts.update(100.4)           # always < 1% above updated_entry
        assert (ts.sl_price, ts.updated_entry, ts.sl_pct) == snap


# ── SHORT: mirror ─────────────────────────────────────────────────

def test_short_initial_stop():
    ts = TrailingStop("short", entry_price=100, init_stop=5)
    assert ts.sl_price == pytest.approx(105.0)
    assert ts.updated_entry == 100.0


def test_short_worked_sequence_matches_pine():
    ts = TrailingStop("short", 100, 5)
    ts.update(94)
    assert ts.sl_price == pytest.approx(100.00, abs=1e-9)
    ts.update(90)
    assert ts.sl_price == pytest.approx(96.00, abs=1e-9)
    ts.update(89)
    assert ts.sl_price == pytest.approx(94.76, abs=1e-6)


def test_short_stop_never_below_generating_close():
    ts = TrailingStop("short", 100, 5)
    for close in (94, 90, 89):
        ts.update(close)
        assert ts.sl_price >= close


def test_short_monotonic_ratchet():
    ts = TrailingStop("short", 100, 5)
    prev = ts.sl_price
    for close in (94, 90, 89):
        ts.update(close)
        assert ts.sl_price <= prev
        prev = ts.sl_price


def test_short_unfavorable_move_does_not_ratchet():
    ts = TrailingStop("short", 100, 5)
    ts.update(94)
    stop_before, entry_before = ts.sl_price, ts.updated_entry
    ts.update(101)          # above updated_entry -> pct positive, no ratchet
    assert ts.sl_price == stop_before
    assert ts.updated_entry == entry_before


def test_short_below_threshold_move_does_not_ratchet():
    ts = TrailingStop("short", 100, 5)
    ts.update(94)                  # updated_entry -> 100.0
    stop_before, entry_before = ts.sl_price, ts.updated_entry
    ts.update(99.6)                # pct = -0.4% (> -1%), no ratchet
    assert ts.sl_price == stop_before
    assert ts.updated_entry == entry_before


def test_short_repeated_calls_below_threshold_are_idempotent():
    ts = TrailingStop("short", 100, 5)
    ts.update(94)
    snap = (ts.sl_price, ts.updated_entry, ts.sl_pct)
    for _ in range(5):
        ts.update(99.7)
        assert (ts.sl_price, ts.updated_entry, ts.sl_pct) == snap


# ── Regression: the cumulative (pre-fix) bug, documented not encoded ──

def _buggy_update(sl_pct, updated_entry, sl_price, side, close):
    """Standalone re-implementation of the OLD, accumulating formula (as
    strategy.py used to compute it pre-fix). Kept only to prove what the bug
    produced -- not imported from strategy.py, never asserted as acceptable."""
    pct = (close - updated_entry) / updated_entry * 100.0
    if side == "long" and pct > 1:
        sl_pct += pct - 1.0
        new = updated_entry * (1 + sl_pct / 100.0)
        sl_price = max(sl_price, new)
        updated_entry = sl_price
    elif side == "short" and pct < -1:
        sl_pct += pct + 1.0
        new = updated_entry * (1 + sl_pct / 100.0)
        sl_price = min(sl_price, new)
        updated_entry = sl_price
    return sl_pct, updated_entry, sl_price


def test_regression_long_old_formula_overshot_but_new_formula_does_not():
    # OLD (buggy, cumulative) formula reproduces the audit's overshoot:
    sl_pct, updated_entry, sl_price = -5.0, 100.0, 95.0
    for close in (106, 110, 111):
        sl_pct, updated_entry, sl_price = _buggy_update(
            sl_pct, updated_entry, sl_price, "long", close)
    assert sl_price == pytest.approx(119.72, abs=1e-6)   # documents the historical bug
    assert sl_price > 111                                 # was structurally invalid

    # CORRECTED implementation, same input sequence -- this is what we assert as right:
    ts = TrailingStop("long", 100, 5)
    for close in (106, 110, 111):
        ts.update(close)
    assert ts.sl_price == pytest.approx(104.76, abs=1e-6)
    assert ts.sl_price < 111


def test_regression_short_old_formula_inverted_but_new_formula_does_not():
    sl_pct, updated_entry, sl_price = 5.0, 100.0, 105.0
    for close in (94, 90, 89):
        sl_pct, updated_entry, sl_price = _buggy_update(
            sl_pct, updated_entry, sl_price, "short", close)
    assert sl_price == pytest.approx(81.72, abs=1e-6)
    assert sl_price < 89

    ts = TrailingStop("short", 100, 5)
    for close in (94, 90, 89):
        ts.update(close)
    assert ts.sl_price == pytest.approx(94.76, abs=1e-6)
    assert ts.sl_price > 89


# ── State-serialization compatibility (run.py reads/writes tsl.sl_pct) ──

def test_sl_pct_attribute_still_present_for_run_py_compatibility():
    ts = TrailingStop("long", 100, 5)
    assert hasattr(ts, "sl_pct")
    ts.update(106)
    assert hasattr(ts, "sl_pct")


def test_loading_stale_buggy_sl_pct_does_not_corrupt_future_ratchets():
    """Simulates run.py's load_state(), which does:
        pos.tsl.sl_pct = p.get("sl_pct", pos.tsl.sl_pct)
    overwriting tsl.sl_pct with a stale, pre-fix (inflated) value from an
    old state.json, then continuing to trade. Confirms the corrected
    update() ignores it (never reads self.sl_pct for computation), so
    sl_price/updated_entry -- the load-bearing state -- are unaffected."""
    ts = TrailingStop("long", 100, 5)
    ts.update(106)
    ts.update(110)
    ts.sl_pct = 9999.0             # poison, as if loaded from a stale state.json
    ts.update(111)
    assert ts.sl_price == pytest.approx(104.76, abs=1e-6)   # unaffected by poisoned sl_pct


# ── Property tests across mixed favorable/unfavorable sequences ──

LONG_SEQUENCES = [
    [101, 103, 108, 107, 112],    # sub-threshold move, then ups with one pullback
    [100.5, 100.9, 100.2, 106],   # several sub-threshold moves then a real ratchet
    [95, 90, 105, 120],           # adverse moves first (no ratchet), then a rally
]

SHORT_SEQUENCES = [
    [99, 97, 92, 93, 88],
    [99.5, 99.1, 99.8, 90],
    [105, 110, 95, 80],
]


@pytest.mark.parametrize("closes", LONG_SEQUENCES)
def test_long_invariants_hold_across_sequences(closes):
    ts = TrailingStop("long", 100, 5)
    prev_stop = ts.sl_price
    for c in closes:
        ts.update(c)
        assert ts.sl_price >= prev_stop            # stop never decreases once ratcheted
        if ts.sl_price != prev_stop:
            # a freshly-ratcheted candidate never exceeds the close that produced it
            assert ts.sl_price <= c
        prev_stop = ts.sl_price


@pytest.mark.parametrize("closes", SHORT_SEQUENCES)
def test_short_invariants_hold_across_sequences(closes):
    ts = TrailingStop("short", 100, 5)
    prev_stop = ts.sl_price
    for c in closes:
        ts.update(c)
        assert ts.sl_price <= prev_stop            # stop never increases once ratcheted
        if ts.sl_price != prev_stop:
            assert ts.sl_price >= c
        prev_stop = ts.sl_price
