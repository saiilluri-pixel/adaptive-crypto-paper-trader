"""
Single source of truth for the FROZEN SOL/USDT Shock Continuation forward
shadow spec (research/SHOCK_CONTINUATION_V3_FROZEN.md), and for the hashes
that prove it hasn't drifted since the pre-launch snapshot.

Nothing in this file may change during a forward-validation epoch. If any of
the three hashed inputs (frozen-spec markdown, strategy implementation,
frozen params) changes, `hashes()` changes too -- the runner checks this on
every restart and refuses to silently continue on a mismatch (see
run_shadow.py `_check_config_drift`), per rule 9: a parameter/logic change
ends the current forward epoch rather than being silently absorbed into it.
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

FROZEN_SPEC_MD = os.path.join(ROOT, "research", "SHOCK_CONTINUATION_V3_FROZEN.md")
STRATEGY_IMPL = os.path.join(ROOT, "research", "strategies", "shock_aftershock.py")

SYMBOL = "SOL/USDT"
TIMEFRAME = "1h"
TIMEFRAME_MS = 3_600_000

# Exactly FROZEN_PARAMS from research/event_ledger.py, retyped here (not
# imported) so this module has zero import-time dependency on the research/
# package -- the shadow runtime must keep working even if research/ code is
# later changed for unrelated (non-frozen) hypothesis work. The hash below
# is what actually proves these still match; if research/event_ledger.py's
# FROZEN_PARAMS ever diverges from this, the two hashes documented in
# research/SHOCK_CONTINUATION_V3_FROZEN.md and here will disagree and that
# is the intended tripwire, not a bug.
FROZEN_PARAMS = {"ATR_N": 14, "SHOCK_ATR_MULT": 3.0, "INIT_STOP_PCT": 4.0}

# Execution/risk assumptions carried unmodified from the shared config module
# (config.TAKER_FEE, config.SLIPPAGE, config.RISK_PCT_PER_TRADE,
# config.POSITION_PCT, config.MAX_DRAWDOWN_PCT, config.DAILY_LOSS_LIMIT_PCT,
# config.CONSECUTIVE_LOSS_LIMIT) -- never overridden by the shadow. Recorded
# here as a snapshot for the hash/audit trail, read live from `config` at
# runtime (not from this frozen copy) so a drift in the shared module is
# caught by _check_config_drift() rather than silently ignored.
import config as _config  # noqa: E402

FROZEN_EXECUTION_ASSUMPTIONS = {
    "exit_on_flip": False,
    "arm_flip": False,
    "entry_execution": "shock bar's own close, via PaperEngine._fill() adverse slippage",
    "stop_check_granularity": "closed-bar OHLC only (check_stop_bar), no intrabar tick checks "
                               "-- matches research/generic_runner.py, which produced the "
                               "validated walk-forward numbers this forward test is following up on",
    "taker_fee": _config.TAKER_FEE,
    "slippage": _config.SLIPPAGE,
    "risk_pct_per_trade": _config.RISK_PCT_PER_TRADE,
    "position_pct": _config.POSITION_PCT,
    "max_drawdown_pct": _config.MAX_DRAWDOWN_PCT,
    "daily_loss_limit_pct": _config.DAILY_LOSS_LIMIT_PCT,
    "consecutive_loss_limit": _config.CONSECUTIVE_LOSS_LIMIT,
    "start_capital": _config.START_CAPITAL,
}


def _sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _sha256_obj(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def hashes():
    """Returns the dict of hashes that define this forward epoch's identity."""
    return {
        "frozen_spec_md_sha256": _sha256_file(FROZEN_SPEC_MD),
        "strategy_impl_sha256": _sha256_file(STRATEGY_IMPL),
        "frozen_params_sha256": _sha256_obj(FROZEN_PARAMS),
        "execution_assumptions_sha256": _sha256_obj(FROZEN_EXECUTION_ASSUMPTIONS),
    }


def combined_hash():
    """Single hash over all four component hashes -- what config_snapshot.json
    and the runner's drift check actually compare."""
    return _sha256_obj(hashes())


if __name__ == "__main__":
    h = hashes()
    for k, v in h.items():
        print(f"{k}: {v}")
    print(f"combined_hash: {combined_hash()}")
