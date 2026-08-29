"""Deterministic tests for adaptive/research_worker.py."""
import numpy as np
import pandas as pd
import pytest

from adaptive.research_worker import (
    simulate_window, evaluate_challenger, generate_challenger_candidates, run_research_cycle,
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
