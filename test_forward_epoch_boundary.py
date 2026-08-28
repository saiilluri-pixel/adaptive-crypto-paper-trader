"""
Deterministic tests for the SOL Shock Continuation V3.1 forward-epoch
boundary guard (shadow/shock_continuation_sol_v3_1/forward_filter.py).

All tests use synthetic in-memory DataFrames or tmp_path CSVs -- never the
real shadow runtime files (state.json, events.csv, trades.csv are never
written to by anything here).
"""
import os

import pandas as pd
import pytest

from shadow.shock_continuation_sol_v3_1 import forward_filter as ff

BOUNDARY = 1787878800000  # 2026-08-28T01:00:00Z, pinned in FORWARD_EPOCH_BOUNDARY.json
HOUR = 3_600_000

HERE = os.path.dirname(os.path.abspath(__file__))
SHADOW_DIR = os.path.join(HERE, "shadow", "shock_continuation_sol_v3_1")


def _event(ts_ms, phase, eligible=True):
    return {"market_ts_ms": ts_ms, "market_dt_utc": "", "local_observation_dt_utc": "",
            "phase": phase, "price": 100.0, "signal": "long" if eligible else "",
            "direction_label": "long" if eligible else "", "atr_norm_shock": 4.0,
            "volume_mult": 1.0, "return_pct": 1.0, "range": 2.0, "body_frac": 0.5,
            "upper_wick_frac": 0.1, "lower_wick_frac": 0.1, "eligible": eligible,
            "position_state_before": "flat", "entry_decision": "entered" if eligible else "no_signal",
            "rejection_reason": ""}


def _trade(signal_ts_ms, phase="live"):
    return {"signal_ts_ms": signal_ts_ms, "signal_dt_utc": "", "entry_ts_ms": signal_ts_ms,
            "entry_dt_utc": "", "exit_ts_ms": signal_ts_ms + HOUR, "exit_dt_utc": "",
            "side": "long", "entry_ref_px": 100.0, "entry_fill_px": 100.0,
            "exit_ref_px": 101.0, "exit_fill_px": 101.0, "qty": 1.0,
            "risk_amount_usdt": 100.0, "entry_fee": 0.1, "exit_fee": 0.1,
            "slippage_cost_usdt": 0.1, "gross_pnl": 1.0, "net_pnl": 0.8,
            "mfe_pct": 1.0, "mae_pct": -0.5, "holding_hours": 1.0,
            "exit_reason": "trailing stop (bar)", "equity_after": 10000.8, "phase": phase,
            "local_observation_entry_dt": "", "local_observation_exit_dt": ""}


# 1) warmup event before boundary -> excluded
def test_warmup_event_before_boundary_excluded():
    row = _event(BOUNDARY - HOUR, "warmup", eligible=True)
    assert ff.is_forward_event(row, BOUNDARY) is False


# 2) warmup eligible=True -> still excluded
def test_warmup_eligible_event_still_excluded():
    row = _event(BOUNDARY + HOUR, "warmup", eligible=True)  # even with a market_ts AFTER boundary
    assert ff.is_forward_event(row, BOUNDARY) is False  # phase alone is disqualifying


# 3) live event at boundary (exactly) -> included
def test_live_event_at_boundary_included():
    row = _event(BOUNDARY, "live", eligible=True)
    assert ff.is_forward_event(row, BOUNDARY) is True


# 4) live event after boundary -> included
def test_live_event_after_boundary_included():
    row = _event(BOUNDARY + HOUR, "live", eligible=True)
    assert ff.is_forward_event(row, BOUNDARY) is True


# 5) catchup event after boundary -> included
def test_catchup_event_after_boundary_included():
    row = _event(BOUNDARY + 2 * HOUR, "catchup", eligible=True)
    assert ff.is_forward_event(row, BOUNDARY) is True


# 6) catchup event before boundary -> excluded
def test_catchup_event_before_boundary_excluded():
    """A restart replaying bars from BEFORE the epoch even started would tag
    them catchup too (e.g. if bootstrap had a resume_ts) -- phase=catchup
    alone must not be treated as sufficient; the timestamp still gates it."""
    row = _event(BOUNDARY - HOUR, "catchup", eligible=True)
    assert ff.is_forward_event(row, BOUNDARY) is False


# 7) trade signaled after boundary, exited during catchup -> included
def test_trade_signaled_after_boundary_exited_during_catchup_included():
    row = _trade(BOUNDARY + HOUR, phase="catchup")  # entered live, restart happened, exit replayed as catchup
    assert ff.is_forward_trade(row, BOUNDARY) is True


# 8) trade signaled before boundary -> excluded
def test_trade_signaled_before_boundary_excluded():
    row = _trade(BOUNDARY - HOUR, phase="live")
    assert ff.is_forward_trade(row, BOUNDARY) is False


# 9) clustering cannot include warmup events
def test_clustering_cannot_include_warmup_events():
    events = pd.DataFrame([
        _event(BOUNDARY - HOUR, "warmup", eligible=True),        # warmup, adjacent in time to the forward one
        _event(BOUNDARY + HOUR, "live", eligible=True),          # genuine forward shock
        _event(BOUNDARY + 50 * HOUR, "live", eligible=True),     # a second, well-separated forward shock
    ])
    clustered = ff.cluster_forward_events(events, BOUNDARY)
    assert len(clustered) == 2  # only the two forward rows ever entered clustering
    assert (clustered["phase"] != "warmup").all()
    assert clustered["cluster_id"].nunique() == 2  # far apart -> two independent clusters, not merged with warmup


def test_clustering_chains_forward_events_within_window_correctly():
    events = pd.DataFrame([
        _event(BOUNDARY - HOUR, "warmup", eligible=True),   # excluded before clustering even runs
        _event(BOUNDARY, "live", eligible=True),
        _event(BOUNDARY + HOUR, "live", eligible=True),     # within 24h of the previous -> same cluster
    ])
    clustered = ff.cluster_forward_events(events, BOUNDARY)
    assert len(clustered) == 2
    assert clustered["cluster_id"].nunique() == 1


def _source_without_docstrings_and_comments(path):
    """Live code only -- strips docstrings/comments via the AST, so this
    check proves no CALL SITE or dict-key access exists, rather than
    fighting safety-explanation prose that happens to name what's absent."""
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
        kept.append(line.split("#", 1)[0])
    return "\n".join(kept)


# 10) runtime epoch_start_iso cannot accidentally substitute for the market boundary
def test_forward_filter_never_reads_state_json_or_epoch_start_iso():
    code = _source_without_docstrings_and_comments(os.path.join(SHADOW_DIR, "forward_filter.py"))
    assert "state.json" not in code
    assert "epoch_start_iso" not in code


def test_forward_filter_boundary_is_not_process_construction_time(tmp_path):
    """Even if a state.json with a DIFFERENT (wrong) epoch_start_iso sits
    right next to events.csv/trades.csv, compute_counts must still use only
    the pinned FORWARD_EPOCH_BOUNDARY.json value, never state.json."""
    misleading_state = tmp_path / "state.json"
    misleading_state.write_text('{"epoch_start_iso": "2020-01-01T00:00:00Z"}')

    events = pd.DataFrame([
        _event(BOUNDARY - HOUR, "warmup", eligible=True),
        _event(BOUNDARY, "live", eligible=True),
    ])
    events.to_csv(tmp_path / "events.csv", index=False)
    pd.DataFrame([_trade(BOUNDARY)]).to_csv(tmp_path / "trades.csv", index=False)

    counts = ff.compute_counts(str(tmp_path), boundary_ts_ms=BOUNDARY)
    assert counts["warmup_eligible_events"] == 1
    assert counts["forward_eligible_events"] == 1
    assert counts["forward_trades"] == 1


# ── boundary metadata + read-only contract ──────────────────────────────
def test_boundary_file_matches_verified_values():
    b = ff.load_boundary()
    assert b["forward_epoch_start_ts_ms"] == BOUNDARY
    assert b["forward_epoch_start_iso"] == "2026-08-28T01:00:00Z"
    assert b["exclude_phases"] == ["warmup"]


def test_compute_counts_against_real_shadow_runtime_is_read_only(tmp_path):
    """Copies the real files into tmp_path first -- proves compute_counts
    doesn't need write access and definitely never mutates its input."""
    import shutil
    for name in ("events.csv", "trades.csv"):
        src = os.path.join(SHADOW_DIR, name)
        if os.path.exists(src):
            shutil.copy(src, tmp_path / name)
            os.chmod(tmp_path / name, 0o444)  # read-only permissions -- a write would raise
    ff.compute_counts(str(tmp_path))  # must not raise despite read-only files
