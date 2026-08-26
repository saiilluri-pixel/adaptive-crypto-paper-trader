"""
Configuration for the BTC paper-trading harness.

Fused strategy:
  ENTRY  = Prime Strategy Swing  (BUY/BUY PRIME -> long, SELL/SELL PRIME -> short)
  EXIT   = Supertrend ATR with Trailing Stop Loss (TSL hit OR Supertrend flip)

Everything is simulated against Binance public live prices. No API keys are used.
"""

# ── Market / data ────────────────────────────────────────────────
EXCHANGE        = "binance"      # ccxt id; public market data only (no auth)
SYMBOL          = "BTC/USDT"     # primary symbol used by backtest.py / autotune.py
SYMBOLS         = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]  # live multi-symbol paper trading
CHART_TF        = "5m"           # timeframe signals + supertrend run on
STRUCTURE_TF    = "30m"          # timeframe swings + BOS/CHoCH run on (Prime default)

WARMUP_CHART    = 600            # closed 5m bars to preload (NW filter needs ~500)
WARMUP_STRUCT   = 320            # closed 30m bars to preload (swings/BOS history)
POLL_SECONDS    = 10             # how often the live loop polls price/candles

# ── Account / sim ────────────────────────────────────────────────
START_CAPITAL   = 10_000.0       # USDT
MARKET_TYPE     = "futures"      # long + short allowed (simulated USDT-M perp)
TAKER_FEE       = 0.0004         # 0.04% futures taker, charged per fill (entry & exit)
SLIPPAGE        = 0.0002         # 0.02% adverse slippage applied to every fill

# ── Position sizing (risk-based, not fixed-notional) ──────────────
# Notional is sized so a full stop-out (at this position's ST_INIT_STOP %)
# loses approximately RISK_PCT_PER_TRADE % of current equity, never more
# than POSITION_PCT of equity (POSITION_PCT is now a leverage ceiling --
# 1.00 = never more than 1x equity notional, no leverage modeled).
RISK_PCT_PER_TRADE = 1.0         # % of equity risked per trade (stop-distance based)
POSITION_PCT       = 1.00        # hard cap on notional as a fraction of equity (no leverage)

# ── Risk controls (block NEW entries only; never blocks managing an
#    already-open position's exit) ─────────────────────────────────
MAX_DRAWDOWN_PCT      = 20.0     # halt new entries while equity is this % below its all-time peak
DAILY_LOSS_LIMIT_PCT  = 5.0      # halt new entries for the rest of the UTC day past this realized loss
CONSECUTIVE_LOSS_LIMIT = 5       # halt new entries after this many losing trades in a row (resets on a win or new UTC day)

# ── Prime Strategy Swing params (from the Pine source) ───────────
SWING_LEN       = 5              # pivot sensitivity (i_swingLen)
NW_BANDWIDTH    = 4.0            # nwH
NW_MULT         = 2.0            # nwMult  (auto-tuned 2026-06-19; source default 3.0)
NW_WINDOW       = 500            # bars used by the Nadaraya-Watson estimator

# Session gates from the Pine code are DROPPED (24/7 crypto, no daily cap),
# per the chosen configuration.

# ── Supertrend ATR exit params (from the Pine source) ────────────
ST_ATR_PERIOD   = 14            # barsBack  (auto-tuned; source default 1 — too whippy on 5m)
ST_ATR_MULT     = 4.0           # multplierFactor  (auto-tuned; source default 3.0)
ST_INIT_STOP    = 5.0           # initialStopLossPercent (%)  (auto-tuned; source default 3.0)

# Exit mode. EXIT_ON_FLIP=False -> trailing-stop-only (pure trail-profit): the
# Supertrend ATR trailing stop is the SOLE exit; the Supertrend direction-flip
# exit is disabled, so winners run until profit is given back into the stop.
EXIT_ON_FLIP = True

# Arm the "Supertrend flipped against position" exit only AFTER entry (only used
# when EXIT_ON_FLIP is True), so an entry against the current Supertrend
# direction is not closed on the same bar. The trailing stop is always active.
ARM_FLIP_AFTER_ENTRY = True

# ── A/B variants (run side-by-side for comparison) ───────────────
# Each variant trades every SYMBOL as its own set of paper accounts, on the same
# live prices. "overrides" replace the tuned per-symbol params (entry params stay
# the same so the comparison isolates EXIT aggressiveness).
VARIANTS = {
    "A": {"label": "Profit-tuned (slow exits)",
          "exit_on_flip": True, "arm_flip": True, "overrides": {}},
    "B": {"label": "More trades (tight stop, no arm-delay)",
          "exit_on_flip": True, "arm_flip": False,
          "overrides": {"ST_INIT_STOP": 2.0, "ST_ATR_PERIOD": 3}},
}

# ── Per-symbol params ────────────────────────────────────────────
# Tunable params; the live runner asks for these per symbol. The values above
# are the global defaults (BTC-tuned). Per-symbol overrides live in
# symbol_params.json (written by autotune.py --all) and win when present.
import json as _json
import os as _os

_TUNABLE = ("SWING_LEN", "NW_BANDWIDTH", "NW_MULT", "NW_WINDOW",
            "ST_ATR_PERIOD", "ST_ATR_MULT", "ST_INIT_STOP")
SYMBOL_PARAMS_FILE = "symbol_params.json"


def params_for(symbol):
    """Global defaults, overlaid with this symbol's tuned params if available."""
    p = {k: globals()[k] for k in _TUNABLE}
    try:
        path = _os.path.join(_os.path.dirname(__file__), SYMBOL_PARAMS_FILE)
        with open(path) as f:
            overrides = _json.load(f)
        if symbol in overrides:
            p.update(overrides[symbol])
    except Exception:
        pass
    return p


# ── Output ───────────────────────────────────────────────────────
STATE_FILE      = "state.json"
TRADES_CSV      = "trades.csv"
LOG_FILE        = "paper.log"
