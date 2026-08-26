# SHOCK_CONTINUATION_V3_FROZEN

Recorded before any V3.1 diagnostic work, per the V3.1 brief's Phase 1. These parameters are not to be altered during V3.1's diagnostic phase. Any new idea discovered during diagnostics is a separate `SHOCK_CONTINUATION_V4_HYPOTHESIS`, evaluated later on new unseen data.

**Signal definition** (`research/strategies/shock_aftershock.py::ShockContinuationStrategy`):
- True range `tr` computed each bar (Wilder-style: `max(h-l, |h-prev_close|, |l-prev_close|)`, or `h-l` on the very first bar).
- Rolling ATR = simple mean of the trailing `ATR_N` true ranges (excludes the current bar).
- **Shock condition:** `tr > SHOCK_ATR_MULT × ATR`.
- **Direction:** shock bar's own candle direction — `c > o` → up-shock, `c < o` → down-shock. A doji (`c == o`) produces no signal.
- **No volume threshold in the frozen definition** — V3's Stage 1-3 work did not use volume at all. (V3.1 section K tests volume as a *diagnostic*, not a change to this frozen definition.)

**Direction rule:** continuation — trade **with** the shock direction (up-shock → long, down-shock → short). (The mirror `ShockReversionStrategy` — trade against — was tested as a separate hypothesis in V3 and is not part of this frozen definition.)

**Selected parameters** (modal walk-forward winner across BTC/ETH/SOL, V3 Stage 2-3):
- `ATR_N = 14`
- `SHOCK_ATR_MULT = 3.0`
- `INIT_STOP_PCT = 4.0`

**Timeframe:** 1h chart, no higher-timeframe filter (`htf_bars=None` for this candidate — the V3 Stage-1 run for E did not condition on HTF trend).

**Holding/exit rule:** the shared, already-validated `PaperEngine` trailing-stop mechanism (`TrailingStop` + `check_stop_bar`/`manage_on_bar`, ratchet-only — Supertrend-flip exit disabled, `exit_on_flip=False`). Initial stop = `INIT_STOP_PCT` from entry price; ratchets favorably per the corrected Phase-1 Pine-fidelity trailing-stop math. No fixed holding period, no time-based exit — position is held until the trailing stop is hit.

**Cost assumptions:** `config.TAKER_FEE = 0.0004` (0.04%), `config.SLIPPAGE = 0.0002` (0.02%), applied per fill exactly as in live/production `PaperEngine._fill()`. (V3.1 section L stress-tests these at 1.5x/2x/3x as a *diagnostic*, not a change to the frozen base assumption.)

**Risk sizing:** `PaperEngine`'s production risk-based sizing — notional sized so a full stop-out at `INIT_STOP_PCT` loses approximately `config.RISK_PCT_PER_TRADE` (1.0%) of current equity, capped at `config.POSITION_PCT` (1.0, i.e. no leverage). (V3.1 section M tests fixed-notional and volatility-normalized sizing as *diagnostics* against this same signal, not a change to the frozen definition.)

**Execution/entry price:** entry filled at the shock bar's own close (`eng.enter(sig, r.close, ...)`), through `PaperEngine._fill()`'s adverse-slippage model — i.e., the position is opened at the same bar whose shock condition just fired, same convention as every other strategy in this research program (signal confirmed at close, executed at that close).

**V3 walk-forward status at freeze time:** 8/8 folds with a result for all three symbols, mean OOS return +0.86%/+0.62%/+1.58% (BTC/ETH/SOL), 50-62.5% of folds profitable, 90% bootstrap CI on mean OOS return included zero for all three symbols. Final (now-spent, per-symbol) V3 holdout: BTC -3.57% (PF 0.51), ETH +2.34%, SOL +8.99%. Status at handoff to V3.1: **VALIDATION**.
