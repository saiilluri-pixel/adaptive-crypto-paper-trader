"""
Deterministic tests for the SOL/USDT Shock Continuation V3.1 forward shadow
(shadow/shock_continuation_sol_v3_1/). No network calls -- uses a stub Feed
serving synthetic 1h OHLCV. Every test that constructs a ShadowRunner passes
an explicit tmp_path as runtime_dir, so nothing here can write into the real
shadow runtime directory or the legacy bot's root-level files.
"""
import os

import pandas as pd
import pytest

import config
from feed import Feed
from shadow.shock_continuation_sol_v3_1 import frozen_spec
from shadow.shock_continuation_sol_v3_1.frozen_spec import FROZEN_PARAMS, TIMEFRAME_MS
from shadow.shock_continuation_sol_v3_1.strategy_adapter import build_strategy, shadow_step
from shadow.shock_continuation_sol_v3_1.run_shadow import ShadowRunner

ROOT = os.path.dirname(os.path.abspath(__file__))
SHADOW_DIR = os.path.join(ROOT, "shadow", "shock_continuation_sol_v3_1")
BASE_TS = 1_700_000_000_000


# ── synthetic candle builders ───────────────────────────────────────────
def _calm_bar(ts, price=100.0, v=1000):
    return (ts, price, price + 1, price - 1, price, v)


def _calm_chart(n, ts0=BASE_TS, price=100.0):
    rows = [_calm_bar(ts0 + i * TIMEFRAME_MS, price) for i in range(n)]
    return pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])


def _long_shock_bar(ts, o=100.0, mult=1.13, v=5000):
    c = o * mult
    return (ts, o, c + 1, o - 1, c, v)


def _short_shock_bar(ts, o=100.0, mult=0.87, v=5000):
    c = o * mult
    return (ts, o, o + 1, c - 1, c, v)


def _df(rows):
    return pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])


class MutableStubFeed:
    """Feed-compatible stub; test appends rows to .df between calls to
    simulate new candles arriving, exactly like a real Feed over time."""
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def append(self, rows):
        self.df = pd.concat([self.df, _df(rows)], ignore_index=True)

    def closed_candles(self, symbol, timeframe, limit):
        return self.df.tail(limit).reset_index(drop=True)

    def retry(self, fn, *a, **k):
        return fn(*a, **k)


# ── 1) long / short / non-shock signal (strategy_adapter, no engine) ────
def test_long_shock_signal_fires():
    strat = build_strategy()
    for i in range(FROZEN_PARAMS["ATR_N"] + 2):
        o, h, l, c, v = 100.0, 101.0, 99.0, 100.0, 1000
        signal, diag = shadow_step(strat, o, h, l, c)
    # warmed up and calm throughout -- now inject a clean up-shock
    o = 100.0
    c = o * 1.13
    signal, diag = shadow_step(strat, o, c + 1, o - 1, c)
    assert signal == "long"
    assert diag is not None
    assert diag["eligible"] is True
    assert diag["atr_norm_shock"] > FROZEN_PARAMS["SHOCK_ATR_MULT"]


def test_short_shock_signal_fires():
    strat = build_strategy()
    for i in range(FROZEN_PARAMS["ATR_N"] + 2):
        signal, diag = shadow_step(strat, 100.0, 101.0, 99.0, 100.0)
    o = 100.0
    c = o * 0.87
    signal, diag = shadow_step(strat, o, o + 1, c - 1, c)
    assert signal == "short"
    assert diag["eligible"] is True


def test_non_shock_bar_produces_no_signal():
    strat = build_strategy()
    for i in range(FROZEN_PARAMS["ATR_N"] + 2):
        signal, diag = shadow_step(strat, 100.0, 101.0, 99.0, 100.0)
    assert signal is None
    assert diag is not None
    assert diag["eligible"] is False


def test_doji_shock_produces_no_signal_but_is_eligible():
    """A shock-magnitude bar with c == o must be logged as eligible (the
    shock condition was met) but must never produce a signal, per the
    frozen definition's explicit doji rule."""
    strat = build_strategy()
    for i in range(FROZEN_PARAMS["ATR_N"] + 2):
        signal, diag = shadow_step(strat, 100.0, 101.0, 99.0, 100.0)
    o = c = 100.0
    signal, diag = shadow_step(strat, o, 115.0, 90.0, c)  # huge range, c == o
    assert signal is None
    assert diag["eligible"] is True


# ── 2) frozen config loading + hash stability ───────────────────────────
def test_frozen_params_match_spec_document():
    assert FROZEN_PARAMS == {"ATR_N": 14, "SHOCK_ATR_MULT": 3.0, "INIT_STOP_PCT": 4.0}


def test_config_hash_is_stable_across_calls():
    assert frozen_spec.combined_hash() == frozen_spec.combined_hash()
    h1 = frozen_spec.hashes()
    h2 = frozen_spec.hashes()
    assert h1 == h2


def test_config_hash_changes_if_frozen_params_change():
    import shadow.shock_continuation_sol_v3_1.frozen_spec as fs
    original = dict(fs.FROZEN_PARAMS)
    before = fs.combined_hash()
    try:
        fs.FROZEN_PARAMS["SHOCK_ATR_MULT"] = 99.0
        after = fs.combined_hash()
        assert after != before
    finally:
        fs.FROZEN_PARAMS.clear()
        fs.FROZEN_PARAMS.update(original)
    assert fs.combined_hash() == before


# ── 3) causal / no-future-bar behavior ──────────────────────────────────
def test_strategy_state_depends_only_on_trailing_window_not_history_length():
    """Warming up over a short vs. a long history that share the same final
    ATR_N bars must converge to identical internal state -- proves state
    depends only on the trailing window, never on how much (or how little)
    history preceded it."""
    tail_prices = [100.0, 102.0, 101.0, 103.0, 99.0, 100.0, 104.0,
                   98.0, 101.0, 100.0, 102.0, 99.5, 100.5, 101.5]
    assert len(tail_prices) == FROZEN_PARAMS["ATR_N"]

    strat_short = build_strategy()
    for p in tail_prices:
        shadow_step(strat_short, p, p + 1, p - 1, p)

    strat_long = build_strategy()
    # ends at exactly 100.0 (== tail_prices[0]) so the transition into the
    # shared tail produces the same true-range chain in both runs
    noise_prefix = [90.0 + (i % 7) for i in range(49)] + [100.0]
    for p in noise_prefix + tail_prices:
        shadow_step(strat_long, p, p + 1, p - 1, p)

    assert list(strat_short.tr_hist) == pytest.approx(list(strat_long.tr_hist))
    assert strat_short.prev_close == pytest.approx(strat_long.prev_close)


def test_run_once_cycle_never_reprocesses_or_reorders_bars(tmp_path):
    chart = _calm_chart(FROZEN_PARAMS["ATR_N"] + 5)
    feed = MutableStubFeed(chart)
    runner = ShadowRunner(runtime_dir=str(tmp_path), feed=feed)
    runner.bootstrap()
    cursor_after_bootstrap = runner.last_bar_ts

    # no new bars yet -- a cycle must be a strict no-op
    runner.run_once_cycle()
    assert runner.last_bar_ts == cursor_after_bootstrap

    # a genuinely new bar advances the cursor exactly once
    new_ts = int(chart.iloc[-1].ts) + TIMEFRAME_MS
    feed.append([_calm_bar(new_ts)])
    runner.run_once_cycle()
    assert runner.last_bar_ts == new_ts

    # replaying (no new rows) again must not move the cursor backward or reprocess
    runner.run_once_cycle()
    assert runner.last_bar_ts == new_ts


# ── 4) entry timing / exit timing / fee+slippage accounting ────────────
def _bootstrap_calm_runner(tmp_path, n_calm=20):
    chart = _calm_chart(n_calm)
    feed = MutableStubFeed(chart)
    runner = ShadowRunner(runtime_dir=str(tmp_path), feed=feed)
    runner.write_config_snapshot()
    runner.bootstrap()
    return runner, feed, chart


def test_entry_fires_on_same_bar_as_signal_at_that_bars_close(tmp_path):
    runner, feed, chart = _bootstrap_calm_runner(tmp_path)
    assert runner.eng.pos is None  # nothing opened during warmup

    shock_ts = int(chart.iloc[-1].ts) + TIMEFRAME_MS
    o = 100.0
    c = o * 1.13
    feed.append([(shock_ts, o, c + 1, o - 1, c, 5000)])
    runner.run_once_cycle()

    assert runner.eng.pos is not None
    assert runner.eng.pos.side == "long"
    close_ts_ms = shock_ts + TIMEFRAME_MS
    assert runner.open_ctx["signal_ts_ms"] == close_ts_ms
    assert runner.open_ctx["entry_ts_ms"] == close_ts_ms
    expected_fill = c * (1 + config.SLIPPAGE)
    assert runner.eng.pos.entry_price == pytest.approx(expected_fill)


def test_exit_never_fires_on_the_entry_bar_itself_only_a_later_bar(tmp_path):
    runner, feed, chart = _bootstrap_calm_runner(tmp_path)
    shock_ts = int(chart.iloc[-1].ts) + TIMEFRAME_MS
    o = 100.0
    c = o * 1.13
    feed.append([(shock_ts, o, c + 1, o - 1, c, 5000)])
    runner.run_once_cycle()
    assert runner.eng.pos is not None
    stop_at_entry = runner.eng.pos.tsl.sl_price

    # a calm bar right after entry must not close the position
    calm_ts = shock_ts + TIMEFRAME_MS
    feed.append([_calm_bar(calm_ts, price=c)])
    runner.run_once_cycle()
    assert runner.eng.pos is not None

    # a bar whose low breaches the (unchanged) stop closes it on THIS bar,
    # never earlier
    drop_ts = calm_ts + TIMEFRAME_MS
    feed.append([(drop_ts, c, c + 0.5, stop_at_entry - 2.0, c - 1.0, 1000)])
    runner.run_once_cycle()
    assert runner.eng.pos is None

    trades = pd.read_csv(os.path.join(str(tmp_path), "trades.csv"))
    assert len(trades) == 1
    assert trades.iloc[0]["exit_reason"] == "trailing stop (bar)"
    assert trades.iloc[0]["exit_ref_px"] == pytest.approx(stop_at_entry)


def test_fee_and_slippage_accounting_reconciles_with_engine(tmp_path):
    runner, feed, chart = _bootstrap_calm_runner(tmp_path)
    shock_ts = int(chart.iloc[-1].ts) + TIMEFRAME_MS
    o = 100.0
    c = o * 1.13
    feed.append([(shock_ts, o, c + 1, o - 1, c, 5000)])
    runner.run_once_cycle()
    pos = runner.eng.pos
    notional = pos.qty * pos.entry_price
    assert pos.entry_fee == pytest.approx(notional * config.TAKER_FEE)

    stop = pos.tsl.sl_price
    drop_ts = shock_ts + TIMEFRAME_MS
    feed.append([(drop_ts, c, c + 0.5, stop - 2.0, c - 1.0, 1000)])
    runner.run_once_cycle()

    assert runner.ledger_mismatch is False  # mirrored close() math agreed with the engine's own
    trades = pd.read_csv(os.path.join(str(tmp_path), "trades.csv"))
    row = trades.iloc[0]
    assert row["net_pnl"] == pytest.approx(runner.eng.realized, abs=1e-6)
    exit_notional = row["qty"] * row["exit_fill_px"]
    assert row["exit_fee"] == pytest.approx(exit_notional * config.TAKER_FEE)
    assert row["mae_pct"] < 0  # the exit bar's low breached the stop -- adverse excursion recorded


# ── 5) restart: flat, open, no downtime entries, state round-trip ──────
def test_restart_flat_preserves_state_and_reprocesses_nothing(tmp_path):
    runner1, feed, chart = _bootstrap_calm_runner(tmp_path)
    equity_before = runner1.eng.equity
    cursor_before = runner1.last_bar_ts

    runner2 = ShadowRunner(runtime_dir=str(tmp_path), feed=feed)
    runner2.bootstrap()
    assert runner2.eng.equity == pytest.approx(equity_before)
    assert runner2.last_bar_ts == cursor_before
    assert runner2.eng.pos is None
    assert runner2.eng.n_trades == 0


def test_restart_open_position_catches_up_without_fabricating_new_entries(tmp_path):
    runner1, feed, chart = _bootstrap_calm_runner(tmp_path)
    shock_ts = int(chart.iloc[-1].ts) + TIMEFRAME_MS
    o = 100.0
    c = o * 1.13
    feed.append([(shock_ts, o, c + 1, o - 1, c, 5000)])
    runner1.run_once_cycle()
    assert runner1.eng.pos is not None
    stop = runner1.eng.pos.tsl.sl_price
    resume_ts = runner1.last_bar_ts

    # "while offline": a bar that breaches the stop, THEN a bar with an
    # opposite-direction shock that must NOT open a new position on replay
    drop_ts = shock_ts + TIMEFRAME_MS
    reversal_ts = drop_ts + TIMEFRAME_MS
    ro = c - 1.0
    rc = ro * 0.87
    feed.append([
        (drop_ts, c, c + 0.5, stop - 2.0, c - 1.0, 1000),
        (reversal_ts, ro, ro + 1, rc - 1, rc, 5000),
    ])

    runner2 = ShadowRunner(runtime_dir=str(tmp_path), feed=feed)
    runner2.bootstrap()

    assert runner2.eng.pos is None          # missed stop-out correctly caught up
    assert runner2.eng.n_trades == 1        # exactly the one real trade, nothing fabricated
    assert runner2.last_bar_ts == reversal_ts

    events = pd.read_csv(os.path.join(str(tmp_path), "events.csv"))
    reversal_row = events[events["market_ts_ms"] == reversal_ts + TIMEFRAME_MS]
    assert len(reversal_row) == 1
    assert reversal_row.iloc[0]["phase"] == "catchup"
    assert reversal_row.iloc[0]["entry_decision"] == "rejected_offline"
    assert reversal_row.iloc[0]["signal"] == "short"  # the shock WAS detected...
    # ...but never traded: still flat, still exactly one trade total
    assert runner2.eng.pos is None
    assert runner2.eng.n_trades == 1


def test_state_round_trip_preserves_full_position_and_strategy_state(tmp_path):
    runner1, feed, chart = _bootstrap_calm_runner(tmp_path)
    shock_ts = int(chart.iloc[-1].ts) + TIMEFRAME_MS
    o = 100.0
    c = o * 1.13
    feed.append([(shock_ts, o, c + 1, o - 1, c, 5000)])
    runner1.run_once_cycle()
    # one favorable bar to move MFE and give the ratchet something to do
    fav_ts = shock_ts + TIMEFRAME_MS
    feed.append([(fav_ts, c, c * 1.02, c * 0.999, c * 1.015, 1200)])
    runner1.run_once_cycle()
    assert runner1.eng.pos is not None

    p1 = runner1.eng.pos
    ctx1 = dict(runner1.open_ctx)
    tr_hist1 = list(runner1.strat.tr_hist)
    prev_close1 = runner1.strat.prev_close

    runner2 = ShadowRunner(runtime_dir=str(tmp_path), feed=feed)
    runner2.bootstrap()
    p2 = runner2.eng.pos

    assert p2 is not None
    assert p2.side == p1.side
    assert p2.entry_price == pytest.approx(p1.entry_price)
    assert p2.qty == pytest.approx(p1.qty)
    assert p2.entry_fee == pytest.approx(p1.entry_fee)
    assert p2.tsl.sl_price == pytest.approx(p1.tsl.sl_price)
    assert p2.tsl.sl_pct == pytest.approx(p1.tsl.sl_pct)
    assert p2.tsl.updated_entry == pytest.approx(p1.tsl.updated_entry)
    assert runner2.open_ctx["mfe_pct"] == pytest.approx(ctx1["mfe_pct"])
    assert runner2.open_ctx["mae_pct"] == pytest.approx(ctx1["mae_pct"])
    assert runner2.open_ctx["entry_ts_ms"] == ctx1["entry_ts_ms"]
    assert list(runner2.strat.tr_hist) == pytest.approx(tr_hist1)
    assert runner2.strat.prev_close == pytest.approx(prev_close1)


def test_reconciliation_failure_blocks_new_entries(tmp_path):
    runner1, feed, chart = _bootstrap_calm_runner(tmp_path)
    shock_ts = int(chart.iloc[-1].ts) + TIMEFRAME_MS
    o = 100.0
    c = o * 1.13
    feed.append([(shock_ts, o, c + 1, o - 1, c, 5000)])
    runner1.run_once_cycle()
    assert runner1.eng.pos is not None

    # simulate a feed that can no longer see far enough back -- only bars
    # strictly AFTER the saved cursor are available
    gap_ts = shock_ts + 10 * TIMEFRAME_MS
    truncated = _df([_calm_bar(gap_ts, price=c)])
    feed2 = MutableStubFeed(truncated)

    runner2 = ShadowRunner(runtime_dir=str(tmp_path), feed=feed2)
    runner2.bootstrap()

    assert runner2.eng.reconciliation_failed is True
    assert runner2.eng.pos is not None  # the existing position itself is untouched, not silently closed

    decision = runner2._enter_with_ledger("long", 100.0, gap_ts + TIMEFRAME_MS)
    assert decision.startswith("rejected_not_flat") or decision.startswith("rejected_risk_guard")


# ── 6) output-path isolation ────────────────────────────────────────────
def test_runner_writes_only_inside_its_own_runtime_dir(tmp_path):
    other = tmp_path / "not_the_runtime_dir"
    runtime = tmp_path / "runtime"
    other.mkdir()
    runner, feed, chart = _bootstrap_calm_runner(runtime)
    for name in ("state.json", "paper.log", "trades.csv", "equity.csv",
                 "events.csv", "config_snapshot.json"):
        assert os.path.exists(os.path.join(str(runtime), name))
    assert os.listdir(str(other)) == []


REAL_SHADOW_RUNTIME_FILES = (
    "state.json", "paper.log", "trades.csv", "equity.csv",
    "events.csv", "config_snapshot.json",
)


def test_tests_do_not_touch_the_real_shadow_runtime_directory():
    """The full test suite above only ever constructs ShadowRunner with a
    tmp_path runtime_dir -- the real shadow/shock_continuation_sol_v3_1/
    directory must contain no runtime output as a result of running tests."""
    for name in REAL_SHADOW_RUNTIME_FILES:
        assert not os.path.exists(os.path.join(SHADOW_DIR, name)), (
            f"{name} exists in the real shadow runtime dir -- a test wrote there")


def test_shadow_source_never_references_legacy_runtime_paths():
    banned = ["config.STATE_FILE", "config.TRADES_CSV", "config.LOG_FILE"]
    for fname in os.listdir(SHADOW_DIR):
        if fname.endswith(".py"):
            with open(os.path.join(SHADOW_DIR, fname)) as f:
                src = f.read()
            for b in banned:
                assert b not in src, f"{fname} references legacy runtime constant {b}"


# ── 7) no authenticated-order code path ─────────────────────────────────
def _source_without_docstrings_and_comments(path):
    """Live code only -- strips module/class/function docstrings and '#'
    comments via the AST, so this check proves no CALL SITE exists rather
    than fighting safety-explanation prose that happens to name what's absent."""
    import ast
    with open(path) as f:
        src = f.read()
    tree = ast.parse(src)
    docstring_lines = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                doc_node = node.body[0]
                docstring_lines.update(range(doc_node.lineno, doc_node.end_lineno + 1))
    kept = []
    for i, line in enumerate(src.splitlines(), start=1):
        if i in docstring_lines:
            continue
        code_part = line.split("#", 1)[0]
        kept.append(code_part)
    return "\n".join(kept)


def test_no_authenticated_order_code_path_in_shadow_or_feed_or_engine():
    banned = ["create_order", "createOrder", "cancel_order", "cancelOrder",
              "withdraw", "transfer(", "apiKey", "api_key", "private_post"]
    paths = [os.path.join(SHADOW_DIR, f) for f in os.listdir(SHADOW_DIR) if f.endswith(".py")]
    paths += [os.path.join(ROOT, "feed.py"), os.path.join(ROOT, "engine.py")]
    for path in paths:
        code = _source_without_docstrings_and_comments(path)
        for token in banned:
            assert token not in code, f"{path} contains banned authenticated-order token {token!r} in live code"


def test_feed_has_no_credentials_configured():
    f = Feed()
    assert not f.ex.apiKey
    assert not f.ex.secret
