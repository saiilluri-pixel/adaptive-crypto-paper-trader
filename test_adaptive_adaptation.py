"""Deterministic tests for adaptive/adaptation.py's champion/challenger rules."""
import json
import os

import pytest

from adaptive.adaptation import (
    AdaptationEngine, EvaluationResult, WindowResult, evaluate_promotion,
    MIN_PROMOTION_SAMPLE,
)


def _eval(strategy, label, params, window_trades_list):
    windows = [WindowResult(window_id=f"w{i}", trades=list(t)) for i, t in enumerate(window_trades_list)]
    return EvaluationResult(strategy=strategy, label=label, params=params, windows=windows)


def _strong_challenger():
    # 40 trades across 4 windows, all positive, PF well above 1, no
    # dependence on a single outlier
    return _eval("trend_momentum", "challenger", {"z_threshold": 0.8},
                 [[1.0] * 10, [1.0] * 10, [1.0] * 10, [1.0] * 10])


def _weak_champion():
    return _eval("trend_momentum", "champion", {"z_threshold": 1.0},
                 [[0.2, -0.1] * 10])


def test_challenger_cannot_promote_with_insufficient_sample():
    champ = _weak_champion()
    chal = _eval("trend_momentum", "challenger", {"z_threshold": 0.8}, [[1.0] * 5])  # only 5 trades
    ok, reasons = evaluate_promotion(champ, chal)
    assert not ok
    assert "sample_size" in reasons


def test_weak_challenger_rejected_negative_expectancy():
    champ = _weak_champion()
    chal = _eval("trend_momentum", "challenger", {"z_threshold": 0.8},
                 [[-1.0] * 10, [-1.0] * 10, [-1.0] * 10, [-1.0] * 10])
    ok, reasons = evaluate_promotion(champ, chal)
    assert not ok
    assert "expectancy" in reasons


def test_weak_challenger_rejected_lower_profit_factor():
    champ = _eval("trend_momentum", "champion", {"z_threshold": 1.0},
                  [[5.0, -0.1] * 10])  # PF = 50/1 = 50.0
    # challenger: profitable, adequate sample, but a clearly LOWER (finite) PF than champion
    chal = _eval("trend_momentum", "challenger", {"z_threshold": 0.8},
                 [[1.0, -0.9] * 10, [1.0, -0.9] * 10])  # PF = 20/18 ~= 1.11
    ok, reasons = evaluate_promotion(champ, chal)
    assert not ok
    assert "profit_factor" in reasons


def test_stronger_validated_challenger_promoted(tmp_path):
    log = str(tmp_path / "adaptation_log.jsonl")
    eng = AdaptationEngine(log)
    champ = _weak_champion()
    chal = _strong_challenger()
    promoted = eng.consider_promotion("trend_momentum", champ, chal, {"z_threshold": 0.8}, now_ts=1000.0)
    assert promoted is True
    assert eng.champions["trend_momentum"] == {"z_threshold": 0.8}
    lines = open(log).readlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["action"] == "promoted"


def test_outlier_dependent_challenger_rejected():
    champ = _weak_champion()
    # 30 trades, all zero except one giant winner -- removing it flips profitable->flat/negative
    trades = [0.0] * 29 + [50.0]
    chal = _eval("trend_momentum", "challenger", {"z_threshold": 0.8},
                 [trades[:15], trades[15:]])
    ok, reasons = evaluate_promotion(champ, chal)
    assert not ok
    assert "outlier_dependence" in reasons


def test_drawdown_materially_worse_rejected():
    # champion has a small but nonzero max DD (0.1 per cycle) so the ratio
    # check is meaningful -- an all-winners champion (max DD == 0) would
    # skip this check entirely by design (see adaptation.py's docstring)
    champ = _eval("trend_momentum", "champion", {"z_threshold": 1.0},
                  [[0.5, -0.1] * 20])  # PF = 10/2 = 5.0, max DD ~0.1
    # challenger: higher PF (would otherwise pass) but a brutal drawdown far exceeding tolerance
    chal_trades = [[8.0, -20.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0]] * 4
    chal = _eval("trend_momentum", "challenger", {"z_threshold": 0.8}, chal_trades)
    ok, reasons = evaluate_promotion(champ, chal)
    assert not ok
    assert "drawdown" in reasons


def test_window_inconsistency_rejected():
    champ = _weak_champion()
    # profitable overall, but only 1 of 4 windows individually positive
    chal = _eval("trend_momentum", "challenger", {"z_threshold": 0.8},
                 [[100.0], [-1.0] * 10, [-1.0] * 10, [-1.0] * 9])
    ok, reasons = evaluate_promotion(champ, chal)
    assert not ok
    assert "window_consistency" in reasons


def test_frozen_shock_continuation_never_adapts(tmp_path):
    eng = AdaptationEngine(str(tmp_path / "log.jsonl"))
    champ = _eval("shock_continuation", "champion", {}, [[1.0] * 10])
    chal = _eval("shock_continuation", "challenger", {"some_param": 1}, [[2.0] * 40])
    promoted = eng.consider_promotion("shock_continuation", champ, chal, {"some_param": 1}, now_ts=1000.0)
    assert promoted is False
    assert "shock_continuation" not in eng.champions


def test_minimum_promotion_interval_enforced(tmp_path):
    eng = AdaptationEngine(str(tmp_path / "log.jsonl"))
    champ = _weak_champion()
    chal = _strong_challenger()
    assert eng.consider_promotion("trend_momentum", champ, chal, {"z_threshold": 0.8},
                                   now_ts=1000.0, min_interval=3600) is True
    # a second strong challenger arrives 10 seconds later -- must be blocked by interval
    chal2 = _strong_challenger()
    assert eng.consider_promotion("trend_momentum", champ, chal2, {"z_threshold": 0.7},
                                   now_ts=1010.0, min_interval=3600) is False
    assert eng.champions["trend_momentum"] == {"z_threshold": 0.8}  # unchanged


def test_rollback_restores_previous_champion(tmp_path):
    eng = AdaptationEngine(str(tmp_path / "log.jsonl"))
    champ = _weak_champion()
    chal = _strong_challenger()
    eng.consider_promotion("trend_momentum", champ, chal, {"z_threshold": 0.8}, now_ts=1000.0)
    assert eng.champions["trend_momentum"] == {"z_threshold": 0.8}
    ok = eng.rollback("trend_momentum")
    assert ok is True
    assert eng.champions["trend_momentum"] is None  # restored to the pre-promotion state (no champion yet)


def test_rollback_with_no_history_returns_false(tmp_path):
    eng = AdaptationEngine(str(tmp_path / "log.jsonl"))
    assert eng.rollback("trend_momentum") is False


def test_parameter_version_persisted_round_trip(tmp_path):
    log = str(tmp_path / "log.jsonl")
    eng = AdaptationEngine(log)
    champ = _weak_champion()
    chal = _strong_challenger()
    eng.consider_promotion("trend_momentum", champ, chal, {"z_threshold": 0.8}, now_ts=1000.0)
    d = eng.to_dict()
    restored = AdaptationEngine.from_dict(d, log)
    assert restored.champions == eng.champions
    assert restored.last_promotion_ts == eng.last_promotion_ts


def test_get_champion_params_falls_back_to_default():
    eng = AdaptationEngine("unused.jsonl")
    default = {"z_threshold": 1.0}
    assert eng.get_champion_params("trend_momentum", default) == default
