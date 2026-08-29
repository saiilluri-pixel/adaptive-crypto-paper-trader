"""Deterministic tests for adaptive/research_worker.py."""
import numpy as np
import pandas as pd
import pytest

from adaptive.research_worker import (
    simulate_window, evaluate_challenger, generate_challenger_candidates, run_research_cycle,
    fold_windows, evaluate_challenger_walkforward, run_research_cycle_walkforward,
    DEFAULT_TRAIN_DAYS, DEFAULT_VAL_DAYS, DEFAULT_TEST_DAYS, DAY_MS,
)
from adaptive.adaptation import AdaptationEngine

BASE = 1_800_000_000_000
HOUR = 3_600_000


def _trend_df(n, rate=1.0003, wick_frac=0.001):
    closes = [100.0 * (rate ** i) for i in range(n)]
    highs = [c * (1 + wick_frac) for c in closes]
    lows = [c * (1 - wick_frac) for c in closes]
    return pd.DataFrame({
        "ts": [BASE + i * HOUR for i in range(n)],
        "open": closes, "high": highs, "low": lows, "close": closes, "volume": [100.0] * n,
    })


def test_simulate_window_produces_trades_on_a_clean_uptrend_then_reversal():
    up = _trend_df(60, rate=1.002)
    down = _trend_df(60, rate=0.997)
    down["ts"] = down["ts"] + up["ts"].iloc[-1] + HOUR
    down["open"] *= up["close"].iloc[-1] / 100.0
    down["high"] *= up["close"].iloc[-1] / 100.0
    down["low"] *= up["close"].iloc[-1] / 100.0
    down["close"] *= up["close"].iloc[-1] / 100.0
    df = pd.concat([up, down], ignore_index=True)
    trades = simulate_window("trend_momentum", df, {"z_threshold": 1.0}, fee_rate=0.001, slippage=0.0002)
    assert len(trades) >= 1


def test_simulate_window_applies_costs():
    """A strategy that enters and exits on essentially flat data should
    never show a positive trade purely from noise -- costs must show up as
    a net negative or zero-trade result."""
    flat = _trend_df(100, rate=1.0)
    trades = simulate_window("mean_reversion", flat, {"n": 20, "z_entry": -1.5, "z_exit": 0.0},
                              fee_rate=0.001, slippage=0.0002)
    assert all(t <= 0 for t in trades)  # flat market -- no real edge, costs dominate whatever fires


def test_evaluate_challenger_produces_matching_window_counts():
    df = _trend_df(1000, rate=1.0005)
    champ_eval, chal_eval = evaluate_challenger(
        "trend_momentum", df, {"z_threshold": 1.0}, {"z_threshold": 0.7},
        fee_rate=0.001, slippage=0.0002, n_windows=3, window_bars=300)
    assert len(champ_eval.windows) == 3
    assert len(chal_eval.windows) == 3


def test_evaluate_challenger_handles_insufficient_history_gracefully():
    df = _trend_df(50, rate=1.0005)  # far less than window_bars
    champ_eval, chal_eval = evaluate_challenger(
        "trend_momentum", df, {"z_threshold": 1.0}, {"z_threshold": 0.7},
        fee_rate=0.001, slippage=0.0002, n_windows=4, window_bars=300)
    assert len(champ_eval.windows) == 0
    assert len(chal_eval.windows) == 0


def test_generate_challenger_candidates_perturbs_around_champion():
    candidates = generate_challenger_candidates("trend_momentum", {"z_threshold": 1.0})
    assert len(candidates) == 2
    z_values = sorted(c["z_threshold"] for c in candidates)
    assert z_values[0] < 1.0 < z_values[1]


def test_generate_challenger_candidates_empty_for_unknown_strategy():
    assert generate_challenger_candidates("shock_continuation", {}) == []


def test_run_research_cycle_never_touches_a_frozen_strategy(tmp_path):
    df = _trend_df(1200, rate=1.0005)
    eng = AdaptationEngine(str(tmp_path / "log.jsonl"))
    results = run_research_cycle(
        historical_data_by_symbol={"BTC/USDT": df},
        adaptation_engine=eng,
        default_params={"trend_momentum": {"z_threshold": 1.0},
                         "volatility_breakout": {"n": 20, "volume_percentile": 0.7},
                         "mean_reversion": {"n": 20, "z_entry": -1.5, "z_exit": 0.0}},
        fee_rate=0.001, slippage=0.0002, now_ts=1000.0)
    strategies_touched = {r["strategy"] for r in results}
    assert "shock_continuation" not in strategies_touched


# ── fold_windows: regression test for the fold-window range bug ─────────
def test_fold_windows_span_the_full_intended_range_not_just_the_start():
    """Regression test for the exact defect found in
    research/stage_walkforward_v3.py during the SOL forward-shadow audit:
    an earlier version stepped forward from data_start by a fixed
    increment and stopped once target_folds was reached, silently
    clustering every fold in the earliest slice of a long history. This
    asserts the corrected fold_windows() actually spans close to the full
    [data_start, data_end] range."""
    data_start = 1_600_000_000_000
    span_days = DEFAULT_TRAIN_DAYS + DEFAULT_VAL_DAYS + DEFAULT_TEST_DAYS
    total_days = span_days * 20  # a much longer history than one fold needs
    data_end = data_start + total_days * DAY_MS

    folds = fold_windows(data_start, data_end, target_folds=6)
    assert len(folds) == 6

    first_test_end = folds[0]["test_end"]
    last_test_end = folds[-1]["test_end"]

    # the LAST fold's test_end must land within one fold-span of data_end --
    # NOT clustered near data_start the way the buggy version did
    one_span_ms = span_days * DAY_MS
    assert data_end - last_test_end < one_span_ms
    # the folds must be spread out, not all crammed into the first slice:
    # the gap between the first and last fold's test_end should be a large
    # fraction of the total available range
    assert (last_test_end - first_test_end) > 0.5 * (data_end - data_start - one_span_ms)


def test_fold_windows_single_fold_uses_data_start():
    data_start = 1_600_000_000_000
    span_days = DEFAULT_TRAIN_DAYS + DEFAULT_VAL_DAYS + DEFAULT_TEST_DAYS
    data_end = data_start + span_days * DAY_MS
    folds = fold_windows(data_start, data_end, target_folds=1)
    assert len(folds) == 1
    assert folds[0]["train_start"] == data_start


def test_fold_windows_insufficient_history_returns_empty():
    data_start = 1_600_000_000_000
    data_end = data_start + DAY_MS  # far too short for even one fold
    assert fold_windows(data_start, data_end) == []


def test_fold_windows_folds_are_causal_train_before_val_before_test():
    data_start = 1_600_000_000_000
    span_days = DEFAULT_TRAIN_DAYS + DEFAULT_VAL_DAYS + DEFAULT_TEST_DAYS
    data_end = data_start + span_days * DAY_MS * 10
    for fold in fold_windows(data_start, data_end, target_folds=4):
        assert fold["train_start"] < fold["train_end"] <= fold["val_end"] <= fold["test_end"]


# ── TRAIN -> VALIDATION -> TEST walk-forward evaluation ──────────────────
def test_walkforward_eligibility_filter_applied_before_val_test():
    """A window with too few TRAIN trades must be excluded entirely --
    never contributes validation or test windows."""
    df = _trend_df(2000, rate=1.0)  # perfectly flat -- no strategy ever trades
    data_start, data_end = int(df["ts"].iloc[0]), int(df["ts"].iloc[-1])
    folds = fold_windows(data_start, data_end, train_days=5, val_days=2, test_days=2, target_folds=3)
    champ_val, chal_val, champ_test, chal_test, n_eligible = evaluate_challenger_walkforward(
        "mean_reversion", df, {"n": 20, "z_entry": -1.5, "z_exit": 0.0},
        {"n": 20, "z_entry": -1.2, "z_exit": 0.0}, fee_rate=0.001, slippage=0.0002, folds=folds)
    assert n_eligible == 0
    assert len(chal_val.windows) == 0


def test_walkforward_never_uses_test_result_to_select_challenger():
    """Structural proof: consider_promotion's decision (ok/reasons) from
    evaluate_promotion must be identical whether or not a test evaluation
    is supplied, AS LONG AS the test result doesn't itself veto (i.e. the
    validation-based accept/reject reasons are computed independently of
    the test set)."""
    from adaptive.adaptation import evaluate_promotion, WindowResult, EvaluationResult
    champ = EvaluationResult("trend_momentum", "champion", {}, [WindowResult("w0", [0.1] * 40)])
    chal = EvaluationResult("trend_momentum", "challenger", {}, [WindowResult("w0", [1.0] * 40)])
    ok_no_test, reasons_no_test = evaluate_promotion(champ, chal, challenger_test=None)
    good_test = EvaluationResult("trend_momentum", "challenger_test", {}, [WindowResult("t0", [1.0] * 10)])
    ok_with_good_test, reasons_with_good_test = evaluate_promotion(champ, chal, challenger_test=good_test)
    # identical validation-based reasons regardless of the (non-vetoing) test result
    assert ok_no_test == ok_with_good_test
    assert {k: v for k, v in reasons_no_test.items() if k != "test_expectancy"} == \
           {k: v for k, v in reasons_with_good_test.items() if k != "test_expectancy"}


def test_walkforward_test_expectancy_can_veto_an_otherwise_passing_challenger():
    from adaptive.adaptation import evaluate_promotion, WindowResult, EvaluationResult
    champ = EvaluationResult("trend_momentum", "champion", {}, [WindowResult("w0", [0.1] * 40)])
    chal = EvaluationResult("trend_momentum", "challenger", {}, [WindowResult("w0", [1.0] * 40)])
    bad_test = EvaluationResult("trend_momentum", "challenger_test", {}, [WindowResult("t0", [-5.0] * 10)])
    ok, reasons = evaluate_promotion(champ, chal, challenger_test=bad_test)
    assert not ok
    assert "test_expectancy" in reasons


def test_run_research_cycle_walkforward_never_touches_frozen_strategy(tmp_path):
    df = _trend_df(3000, rate=1.0005)
    eng = AdaptationEngine(str(tmp_path / "log.jsonl"))
    results = run_research_cycle_walkforward(
        historical_data_by_symbol={"BTC/USDT": df}, adaptation_engine=eng,
        default_params={"trend_momentum": {"z_threshold": 1.0},
                         "volatility_breakout": {"n": 20, "volume_percentile": 0.7},
                         "mean_reversion": {"n": 20, "z_entry": -1.5, "z_exit": 0.0}},
        fee_rate=0.001, slippage=0.0002, now_ts=1000.0,
        train_days=5, val_days=2, test_days=2, target_folds=3)
    assert "shock_continuation" not in {r["strategy"] for r in results}


def test_run_research_cycle_writes_to_adaptation_log_only_on_activity(tmp_path):
    log_path = tmp_path / "log.jsonl"
    df = _trend_df(1200, rate=1.0005)
    eng = AdaptationEngine(str(log_path))
    run_research_cycle(
        historical_data_by_symbol={"BTC/USDT": df},
        adaptation_engine=eng,
        default_params={"trend_momentum": {"z_threshold": 1.0},
                         "volatility_breakout": {"n": 20, "volume_percentile": 0.7},
                         "mean_reversion": {"n": 20, "z_entry": -1.5, "z_exit": 0.0}},
        fee_rate=0.001, slippage=0.0002, now_ts=1000.0)
    assert log_path.exists()  # at least one rejection/promotion was logged
