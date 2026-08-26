"""
Deterministic tests for restart correctness (item 6): missed-bar management
catch-up on restart, without ever fabricating an entry from a signal that
fired while offline, and state-file round-trip compatibility (entry_fee,
cursors). No network calls -- uses a stub Feed returning synthetic candles.
"""
import pandas as pd
import pytest

import config
from engine import Position
from run import SymbolGroup, VariantTrader

CHART_MS = 300_000
STRUCT_MS = 1_800_000
BASE_TS = 1_700_000_000_000


class StubFeed:
    """Feed-compatible stub serving synthetic OHLCV, no network."""
    def __init__(self, struct_df, chart_df, price):
        self._dfs = {config.STRUCTURE_TF: struct_df, config.CHART_TF: chart_df}
        self._price = price

    def closed_candles(self, symbol, timeframe, limit):
        return self._dfs[timeframe].tail(limit).reset_index(drop=True)

    def last_price(self, symbol):
        return self._price

    def retry(self, fn, *a, **k):
        return fn(*a, **k)


def _flat_df(n, price=100.0):
    ts = [BASE_TS + i * STRUCT_MS for i in range(n)]
    return pd.DataFrame({"ts": ts, "open": [price] * n, "high": [price] * n,
                          "low": [price] * n, "close": [price] * n})


def _chart_df(closes):
    n = len(closes)
    ts = [BASE_TS + i * CHART_MS for i in range(n)]
    return pd.DataFrame({"ts": ts, "open": closes, "high": closes,
                          "low": closes, "close": closes})


@pytest.fixture
def one_variant(monkeypatch):
    monkeypatch.setattr(config, "SYMBOLS", ["BTC/USDT"])
    monkeypatch.setattr(config, "VARIANTS",
                         {"A": {"label": "t", "exit_on_flip": True,
                                "arm_flip": True, "overrides": {}}})
    return "A"


def test_warmup_catches_up_a_missed_stop_hit(one_variant):
    struct = _flat_df(config.WARMUP_STRUCT)
    n = config.WARMUP_CHART
    # flat at 100 except a sharp drop in the final two ("missed") bars
    closes = [100.0] * (n - 2) + [90.0, 80.0]
    chart = _chart_df(closes)
    feed = StubFeed(struct, chart, price=80.0)
    g = SymbolGroup("BTC/USDT", feed, log_trades=False, log_fn=lambda m: None)
    vt = g.variants["A"]

    vt.eng.pos = Position("long", 100.0, 1.0, "2026-01-01 00:00:00", "BUY",
                           vt.params["ST_INIT_STOP"], entry_fee=0.0)
    vt.eng.pos.tsl.sl_price = 95.0     # resting stop above the missed-window drop
    resume_ts = int(chart.iloc[-3].ts)  # everything from the drop onward is "missed"

    g.warmup(resume_chart_ts=resume_ts)

    assert vt.eng.pos is None          # caught up: closed during warmup, not left stale-open
    assert vt.eng.n_trades == 1
    assert vt.eng.wins == 0            # closed for a loss well below the stop, as expected


def test_warmup_ratchets_stop_normally_when_no_crossing_occurs(one_variant):
    """Item 8: missed bars with no stop crossing -> the trailing stop still
    ratchets normally through catch-up (check_stop_bar finds no hit each
    bar, manage_on_bar then ratchets), instead of being left stale."""
    struct = _flat_df(config.WARMUP_STRUCT)
    n = config.WARMUP_CHART
    # favorable, non-crossing rise in the missed window (well above any stop)
    closes = [100.0] * (n - 2) + [108.0, 112.0]
    chart = _chart_df(closes)
    feed = StubFeed(struct, chart, price=112.0)
    g = SymbolGroup("BTC/USDT", feed, log_trades=False, log_fn=lambda m: None)
    vt = g.variants["A"]

    vt.eng.pos = Position("long", 100.0, 1.0, "2026-01-01 00:00:00", "BUY",
                           vt.params["ST_INIT_STOP"], entry_fee=0.0)
    vt.eng.pos.tsl.sl_price = 95.0
    stop_before = vt.eng.pos.tsl.sl_price
    resume_ts = int(chart.iloc[-3].ts)

    g.warmup(resume_chart_ts=resume_ts)

    assert vt.eng.pos is not None          # never crossed -> still open
    assert vt.eng.n_trades == 0
    assert vt.eng.pos.tsl.sl_price > stop_before   # ratcheted up through catch-up, not left stale


def test_warmup_never_fabricates_an_entry_from_missed_signals(one_variant):
    struct = _flat_df(config.WARMUP_STRUCT)
    n = config.WARMUP_CHART
    closes = [100.0] * (n - 2) + [200.0, 50.0]   # violent swings that would fire many signals
    chart = _chart_df(closes)
    feed = StubFeed(struct, chart, price=50.0)
    g = SymbolGroup("BTC/USDT", feed, log_trades=False, log_fn=lambda m: None)
    vt = g.variants["A"]
    # flat before the outage -- no position to restore
    resume_ts = int(chart.iloc[-3].ts)

    g.warmup(resume_chart_ts=resume_ts)

    assert vt.eng.pos is None
    assert vt.eng.n_trades == 0        # on_signals() is never called during warmup


def test_no_cursor_means_no_catch_up_replay(one_variant):
    """resume_chart_ts=None (fresh start / pre-upgrade state.json) must behave
    exactly as before: no catch-up management is attempted."""
    struct = _flat_df(config.WARMUP_STRUCT)
    n = config.WARMUP_CHART
    closes = [100.0] * (n - 2) + [90.0, 80.0]
    chart = _chart_df(closes)
    feed = StubFeed(struct, chart, price=80.0)
    g = SymbolGroup("BTC/USDT", feed, log_trades=False, log_fn=lambda m: None)
    vt = g.variants["A"]
    vt.eng.pos = Position("long", 100.0, 1.0, "2026-01-01 00:00:00", "BUY",
                           vt.params["ST_INIT_STOP"], entry_fee=0.0)
    vt.eng.pos.tsl.sl_price = 95.0

    g.warmup(resume_chart_ts=None)

    assert vt.eng.pos is not None      # untouched -- no catch-up attempted without a cursor
    assert vt.eng.n_trades == 0


# ── State round-trip compatibility (entry_fee, cursors) ────────────────

def test_entry_fee_round_trips_through_snapshot_and_load_state(one_variant):
    vcfg = config.VARIANTS["A"]
    vt = VariantTrader("BTC/USDT", "BTC", "A", vcfg, log_trades=False, log_fn=lambda m: None)
    vt.eng.enter("long", 100.0, "BUY", 1)
    snap = vt.snapshot(100.0)
    assert snap["position"]["entry_fee"] == pytest.approx(vt.eng.pos.entry_fee)
    assert vt.eng.pos.entry_fee > 0

    vt2 = VariantTrader("BTC/USDT", "BTC", "A", vcfg, log_trades=False, log_fn=lambda m: None)
    vt2.load_state({"cash": vt.eng.cash, "realized": 0.0, "n_trades": 0, "wins": 0,
                     "equity": vt.eng.equity, "position": snap["position"]})
    assert vt2.eng.pos.entry_fee == pytest.approx(vt.eng.pos.entry_fee)


def test_no_reentry_fabricated_after_position_closes_mid_catchup(one_variant):
    """Item 9 (position-closes-mid-replay variant): once check_stop_bar closes
    the restored position partway through the missed window, subsequent
    missed bars -- even ones that would clearly fire an entry signal live --
    must NOT be allowed to open a new position, because on_signals() is never
    called anywhere in warmup()."""
    struct = _flat_df(config.WARMUP_STRUCT)
    n = config.WARMUP_CHART
    # drop below the stop early in the missed window, then wild swings after
    closes = [100.0] * (n - 4) + [80.0, 200.0, 50.0, 300.0]
    chart = _chart_df(closes)
    feed = StubFeed(struct, chart, price=300.0)
    g = SymbolGroup("BTC/USDT", feed, log_trades=False, log_fn=lambda m: None)
    vt = g.variants["A"]

    vt.eng.pos = Position("long", 100.0, 1.0, "2026-01-01 00:00:00", "BUY",
                           vt.params["ST_INIT_STOP"], entry_fee=0.0)
    vt.eng.pos.tsl.sl_price = 95.0
    resume_ts = int(chart.iloc[-5].ts)

    g.warmup(resume_chart_ts=resume_ts)

    assert vt.eng.pos is None       # closed on the drop to 80
    assert vt.eng.n_trades == 1     # exactly the one exit -- no fabricated re-entry despite the swings


def test_insufficient_restart_history_blocks_new_entries(one_variant):
    """Item 12: if the saved cursor predates the earliest bar the fetch can
    see, the gap cannot be fully reconciled for an open position -- must not
    silently resume as if nothing happened. reconciliation_failed is set and
    new entries are blocked (managing the existing position still works)."""
    struct = _flat_df(config.WARMUP_STRUCT)
    n = config.WARMUP_CHART
    chart = _chart_df([100.0] * n)
    feed = StubFeed(struct, chart, price=100.0)
    g = SymbolGroup("BTC/USDT", feed, log_trades=False, log_fn=lambda m: None)
    vt = g.variants["A"]

    vt.eng.pos = Position("long", 100.0, 1.0, "2026-01-01 00:00:00", "BUY",
                           vt.params["ST_INIT_STOP"], entry_fee=0.0)
    vt.eng.pos.tsl.sl_price = 95.0
    resume_ts = int(chart.iloc[0].ts) - 10 * CHART_MS   # cursor predates the fetched window

    g.warmup(resume_chart_ts=resume_ts)

    assert vt.eng.pos is not None                  # position untouched -- still managed, not discarded
    assert vt.eng.reconciliation_failed is True
    assert vt.eng.enter("long", 100.0, "BUY", 1) is None
    assert vt.eng.pos.entry_price == 100.0          # confirms enter() was actually refused (no re-entry)

    # once the tainted position closes, the flag clears and entries resume
    vt.eng.close(200.0, "test exit")
    assert vt.eng.reconciliation_failed is False


def test_reconciliation_flag_and_flat_variant_are_unaffected_by_gap(one_variant):
    """A gap in the cursor for a FLAT variant (no open position) is not a
    safety concern -- nothing was silently missed for a position that never
    existed -- so it must not spuriously flag or block trading."""
    struct = _flat_df(config.WARMUP_STRUCT)
    n = config.WARMUP_CHART
    chart = _chart_df([100.0] * n)
    feed = StubFeed(struct, chart, price=100.0)
    g = SymbolGroup("BTC/USDT", feed, log_trades=False, log_fn=lambda m: None)
    vt = g.variants["A"]
    resume_ts = int(chart.iloc[0].ts) - 10 * CHART_MS

    g.warmup(resume_chart_ts=resume_ts)

    assert vt.eng.reconciliation_failed is False


def test_loading_position_without_entry_fee_field_defaults_to_zero(one_variant):
    """A state.json written before this upgrade has no entry_fee key -- must
    load without crashing and default to 0.0, not corrupt sl_price/updated_entry."""
    vcfg = config.VARIANTS["A"]
    vt = VariantTrader("BTC/USDT", "BTC", "A", vcfg, log_trades=False, log_fn=lambda m: None)
    legacy_position = {"side": "long", "qty": 1.0, "entry": 100.0, "stop": 95.0,
                        "reason": "BUY", "entry_time": "2026-01-01 00:00:00",
                        "flip_armed": False, "sl_pct": -5.0, "updated_entry": 100.0}
    vt.load_state({"cash": 10_000.0, "realized": 0.0, "n_trades": 0, "wins": 0,
                    "equity": 10_000.0, "position": legacy_position})
    assert vt.eng.pos.entry_fee == 0.0
    assert vt.eng.pos.tsl.sl_price == 95.0
    assert vt.eng.pos.tsl.updated_entry == 100.0
