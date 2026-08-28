"""
SOL/USDT Shock Continuation V3.1 -- forward shadow paper trader.

Status: VALIDATION -- INCONCLUSIVE (see research/FORWARD_VALIDATION_PLAN.md).
This is NOT a validated edge and NOT a deployment. Binance PUBLIC market data
only (ccxt fetch_ohlcv/fetch_ticker) -- no API keys, no authenticated
endpoints, no create_order/cancel_order/withdraw/transfer anywhere in this
file or anything it imports. NO REAL ORDERS CAN BE SENT.

Completely isolated from the legacy com.btcpaper.bot: separate runtime
directory (this directory), separate state/logs/CSVs, separate launchd
service (com.btcpaper.shocksol). Reuses the production PaperEngine
(fees/slippage/risk-sizing/risk-guards/trailing-stop math) unmodified --
only ADDS a side-channel logging layer (wrapping the bound eng.close on this
one instance, and driving eng.enter()/check_stop_bar()/manage_on_bar()
directly instead of on_signals(), exactly as research/generic_runner.py
already does for every other research candidate).

Execution model matches research/generic_runner.py exactly (the methodology
that produced the walk-forward numbers this forward test follows up on):
stop checks and ratchet happen on CLOSED-BAR OHLC only (check_stop_bar,
manage_on_bar) -- there is no intrabar tick-based stop check
(check_stop_intrabar), and eng.mark() is called exactly once per processed
bar using that bar's own close, never a live ticker price. Both choices are
deliberate parity decisions, not omissions -- introducing intrabar checking
here would make forward results incomparable to the historical evidence.

    python3 run_shadow.py                  # run live forever
    python3 run_shadow.py --bootstrap-only  # warm up / catch up, print state, exit
"""
import argparse
import os
import sys
import time
from collections import deque
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

import config  # noqa: E402
from engine import PaperEngine, Position  # noqa: E402
from feed import Feed  # noqa: E402

from shadow.shock_continuation_sol_v3_1 import frozen_spec  # noqa: E402
from shadow.shock_continuation_sol_v3_1.frozen_spec import FROZEN_PARAMS, SYMBOL, TIMEFRAME, TIMEFRAME_MS  # noqa: E402
from shadow.shock_continuation_sol_v3_1.strategy_adapter import build_strategy, shadow_step  # noqa: E402
from shadow.shock_continuation_sol_v3_1.ledger import Ledger, iso, now_iso  # noqa: E402

POLL_SECONDS = 60          # 1h bars -- polling every 60s is ample headroom
BOOTSTRAP_FETCH_LIMIT = 500  # ~20 days of 1h candles; beyond this a restart
                              # gap is a genuine gap-integrity failure, handled explicitly


class ShadowRunner:
    def __init__(self, runtime_dir, feed=None, now_fn=None):
        self.ledger = Ledger(runtime_dir)
        self.feed = feed or Feed()
        self._now_fn = now_fn or (lambda: datetime.now(timezone.utc))

        self.strat = build_strategy()
        self.eng = PaperEngine(
            log_fn=self.ledger.log, log_trades=False, symbol=SYMBOL,
            params={"ST_INIT_STOP": FROZEN_PARAMS["INIT_STOP_PCT"]},
            exit_on_flip=False, arm_flip=False, now_fn=self._now_fn)
        self._wrap_close_for_ledger()

        self.last_bar_ts = None
        self.open_ctx = None
        self.ledger_mismatch = False
        self.vol_hist = deque(maxlen=FROZEN_PARAMS["ATR_N"])
        self.current_phase = "warmup"
        self._current_bar_ts_ms = None
        self.epoch_start_iso = now_iso()

    # ── logging-only instrumentation of THIS engine instance ───────────
    def _wrap_close_for_ledger(self):
        """Wraps eng.close on this one PaperEngine instance to also write a
        full trade-ledger row. Does not change what close() does -- the
        original method still runs, unmodified, and is the sole source of
        truth for eng.cash/realized/equity. This wrapper only mirrors
        close()'s own gross/fee/net formulas (read directly from engine.py)
        to build the ledger row, and cross-checks the mirror against the
        engine's own actual realized-PnL delta every single time (see the
        mismatch handling below) so a future engine.py change that this
        mirror fails to track is caught loudly rather than silently logging
        wrong numbers."""
        eng = self.eng
        orig_close = eng.close

        def logged_close(ref_price, exit_reason):
            p = eng.pos
            if p is None:
                return orig_close(ref_price, exit_reason)

            fill = eng._fill(ref_price, p.side, opening=False)
            gross = (fill - p.entry_price) * p.qty if p.side == "long" \
                else (p.entry_price - fill) * p.qty
            exit_fee = (fill * p.qty) * config.TAKER_FEE
            mirrored_net = gross - p.entry_fee - exit_fee
            realized_before = eng.realized
            ctx = self.open_ctx
            exit_ts_ms = self._current_bar_ts_ms
            phase = self.current_phase

            orig_close(ref_price, exit_reason)  # authoritative state mutation

            actual_net = eng.realized - realized_before
            net = actual_net
            if abs(actual_net - mirrored_net) > 1e-6:
                self.ledger.log(
                    f"!! LEDGER RECONCILE MISMATCH: mirrored net={mirrored_net:.6f} "
                    f"actual net={actual_net:.6f} -- logging the actual figure, "
                    f"flagging ledger_mismatch, new entries blocked until resolved")
                self.ledger_mismatch = True

            self.ledger.write_trade({
                "signal_ts_ms": (ctx or {}).get("signal_ts_ms", ""),
                "signal_dt_utc": iso(ctx["signal_ts_ms"]) if ctx and ctx.get("signal_ts_ms") else "",
                "entry_ts_ms": (ctx or {}).get("entry_ts_ms", ""),
                "entry_dt_utc": iso(ctx["entry_ts_ms"]) if ctx and ctx.get("entry_ts_ms") else "",
                "exit_ts_ms": exit_ts_ms, "exit_dt_utc": iso(exit_ts_ms) if exit_ts_ms else "",
                "side": p.side, "entry_ref_px": (ctx or {}).get("entry_ref_px", ""),
                "entry_fill_px": p.entry_price, "exit_ref_px": ref_price, "exit_fill_px": fill,
                "qty": p.qty, "risk_amount_usdt": (ctx or {}).get("risk_amount", ""),
                "entry_fee": p.entry_fee, "exit_fee": exit_fee,
                "slippage_cost_usdt": (ctx or {}).get("entry_slippage_cost", 0.0) + abs(fill - ref_price) * p.qty,
                "gross_pnl": gross, "net_pnl": net,
                "mfe_pct": (ctx or {}).get("mfe_pct", ""), "mae_pct": (ctx or {}).get("mae_pct", ""),
                "holding_hours": ((exit_ts_ms - ctx["entry_ts_ms"]) / 3_600_000)
                                  if ctx and ctx.get("entry_ts_ms") and exit_ts_ms else "",
                "exit_reason": exit_reason, "equity_after": eng.equity, "phase": phase,
                "local_observation_entry_dt": (ctx or {}).get("local_observation_entry_dt", ""),
                "local_observation_exit_dt": now_iso(),
            })
            self.open_ctx = None

        eng.close = logged_close

    def _enter_with_ledger(self, side, ref_price, signal_ts_ms):
        eng = self.eng
        if eng.pos is not None:
            return "rejected_not_flat"
        if self.ledger_mismatch:
            return "rejected_ledger_mismatch"
        guard_reason = eng._risk_guard_blocked()
        if guard_reason is not None:
            return f"rejected_risk_guard:{guard_reason}"
        equity_before = eng.equity
        eng.enter(side, ref_price, "shock continuation", st_dir=0)
        if eng.pos is None:
            return "rejected_sizing_invalid"
        p = eng.pos
        self.open_ctx = {
            "signal_ts_ms": signal_ts_ms, "entry_ts_ms": signal_ts_ms,
            "entry_ref_px": ref_price,
            "risk_amount": equity_before * (config.RISK_PCT_PER_TRADE / 100.0),
            "entry_slippage_cost": abs(p.entry_price - ref_price) * p.qty,
            "mfe_pct": 0.0, "mae_pct": 0.0,
            "local_observation_entry_dt": now_iso(),
        }
        return "entered"

    def _update_open_trade_extremes(self, h, l):
        if self.open_ctx is None or self.eng.pos is None:
            return
        p = self.eng.pos
        if p.side == "long":
            mfe = (h / p.entry_price - 1) * 100
            mae = (l / p.entry_price - 1) * 100
        else:
            mfe = (p.entry_price / l - 1) * 100
            mae = (p.entry_price / h - 1) * 100
        self.open_ctx["mfe_pct"] = max(self.open_ctx["mfe_pct"], mfe)
        self.open_ctx["mae_pct"] = min(self.open_ctx["mae_pct"], mae)

    # ── one bar, causal, used for warmup/catchup/live alike ─────────────
    def _process_bar(self, o, h, l, c, v, bar_open_ts_ms, phase):
        self.current_phase = phase
        close_ts_ms = bar_open_ts_ms + TIMEFRAME_MS
        self._current_bar_ts_ms = close_ts_ms
        eng = self.eng

        # extremes + stop check use the PRE-ratchet stop and this bar's full
        # OHLC, exactly the check_stop_bar contract -- must run before any
        # ratchet update or new entry from this same bar
        if eng.pos is not None:
            self._update_open_trade_extremes(h, l)
            eng.check_stop_bar(o, h, l)
        if eng.pos is not None:
            eng.manage_on_bar({"close": c, "st_dir": 0})

        vol_avg = (sum(self.vol_hist) / len(self.vol_hist)) if len(self.vol_hist) == self.vol_hist.maxlen else None
        signal, diag = shadow_step(self.strat, o, h, l, c)

        position_state_before = "open" if eng.pos is not None else "flat"
        entry_decision, rejection_reason = "no_signal", ""
        if diag is not None and diag.get("eligible") and signal is None:
            entry_decision = "doji_no_signal"

        if signal in ("long", "short"):
            if phase != "live":
                entry_decision, rejection_reason = "rejected_offline", "no_fabricated_entry_while_offline"
            else:
                entry_decision = self._enter_with_ledger(signal, c, close_ts_ms)
                if entry_decision != "entered":
                    rejection_reason = entry_decision

        eng.mark(c)  # closed-bar mark only, exactly once per processed bar

        if diag is not None:
            volume_mult = (v / vol_avg) if vol_avg else None
            self.ledger.write_event({
                "market_ts_ms": close_ts_ms, "market_dt_utc": iso(close_ts_ms),
                "local_observation_dt_utc": now_iso(), "phase": phase, "price": c,
                "signal": signal or "", "direction_label": signal or "",
                "atr_norm_shock": diag["atr_norm_shock"], "volume_mult": volume_mult,
                "return_pct": diag["return_pct"], "range": diag["range"],
                "body_frac": diag["body_frac"], "upper_wick_frac": diag["upper_wick_frac"],
                "lower_wick_frac": diag["lower_wick_frac"], "eligible": diag["eligible"],
                "position_state_before": position_state_before,
                "entry_decision": entry_decision, "rejection_reason": rejection_reason,
            })

        self.vol_hist.append(v)
        self.ledger.write_equity(close_ts_ms, eng.equity)

    # ── state persistence ────────────────────────────────────────────
    def _save_state(self):
        eng = self.eng
        pos = None
        if eng.pos is not None:
            p = eng.pos
            pos = {"side": p.side, "entry": p.entry_price, "qty": p.qty,
                   "entry_time": p.entry_time, "reason": p.reason,
                   "stop": p.tsl.sl_price, "sl_pct": p.tsl.sl_pct,
                   "updated_entry": p.tsl.updated_entry, "entry_fee": p.entry_fee}
        state = {
            "combined_hash": frozen_spec.combined_hash(),
            "last_bar_ts": self.last_bar_ts,
            "cash": eng.cash, "equity": eng.equity, "realized": eng.realized,
            "n_trades": eng.n_trades, "wins": eng.wins,
            "peak_equity": eng.peak_equity, "consecutive_losses": eng.consecutive_losses,
            "trading_day": eng.trading_day, "daily_start_equity": eng.daily_start_equity,
            "reconciliation_failed": eng.reconciliation_failed,
            "ledger_mismatch": self.ledger_mismatch,
            "position": pos, "open_ctx": self.open_ctx,
            "strat_tr_hist": list(self.strat.tr_hist), "strat_prev_close": self.strat.prev_close,
            "vol_hist": list(self.vol_hist),
            "epoch_start_iso": self.epoch_start_iso,
            "saved_at_iso": now_iso(),
        }
        self.ledger.write_state(state)

    def _restore_state(self, state):
        eng = self.eng
        eng.cash = state.get("cash", eng.cash)
        eng.equity = state.get("equity", eng.cash)
        eng.realized = state.get("realized", 0.0)
        eng.n_trades = state.get("n_trades", 0)
        eng.wins = state.get("wins", 0)
        eng.peak_equity = state.get("peak_equity", eng.equity)
        eng.consecutive_losses = state.get("consecutive_losses", 0)
        eng.trading_day = state.get("trading_day")
        eng.daily_start_equity = state.get("daily_start_equity", eng.equity)
        eng.reconciliation_failed = state.get("reconciliation_failed", False)
        self.ledger_mismatch = state.get("ledger_mismatch", False)
        p = state.get("position")
        if p:
            pos = Position(p["side"], p["entry"], p["qty"], p["entry_time"], p["reason"],
                            FROZEN_PARAMS["INIT_STOP_PCT"], entry_fee=p.get("entry_fee", 0.0))
            pos.tsl.sl_price = p["stop"]
            pos.tsl.sl_pct = p.get("sl_pct", pos.tsl.sl_pct)
            pos.tsl.updated_entry = p.get("updated_entry", p["entry"])
            eng.pos = pos
        self.open_ctx = state.get("open_ctx")
        self.strat.tr_hist = deque(state.get("strat_tr_hist") or [], maxlen=self.strat.atr_n)
        self.strat.prev_close = state.get("strat_prev_close")
        self.vol_hist = deque(state.get("vol_hist") or [], maxlen=FROZEN_PARAMS["ATR_N"])
        self.epoch_start_iso = state.get("epoch_start_iso") or now_iso()

    def write_config_snapshot(self):
        self.ledger.write_config_snapshot({
            "symbol": SYMBOL, "timeframe": TIMEFRAME, "frozen_params": FROZEN_PARAMS,
            "execution_assumptions": frozen_spec.FROZEN_EXECUTION_ASSUMPTIONS,
            "hashes": frozen_spec.hashes(), "combined_hash": frozen_spec.combined_hash(),
            "written_at_iso": now_iso(),
        })

    # ── bootstrap (cold start OR restart catch-up) ──────────────────────
    def bootstrap(self):
        state = self.ledger.load_state()
        chash = frozen_spec.combined_hash()
        if state is not None:
            stored_hash = state.get("combined_hash")
            if stored_hash is not None and stored_hash != chash:
                raise RuntimeError(
                    f"CONFIG DRIFT DETECTED: stored combined_hash={stored_hash} != "
                    f"current={chash}. Per the forward-validation plan's rule 9, a "
                    f"parameter/logic change ends the current forward epoch -- refusing "
                    f"to start. Archive this runtime directory's contents as the closed "
                    f"epoch before starting a new one.")
            self._restore_state(state)
            resume_ts = state.get("last_bar_ts")
            phase = "catchup"
        else:
            resume_ts = None
            phase = "warmup"

        chart = self.feed.retry(self.feed.closed_candles, SYMBOL, TIMEFRAME, BOOTSTRAP_FETCH_LIMIT)
        if len(chart) == 0:
            raise RuntimeError("Feed returned zero closed candles on bootstrap")

        earliest_ts = int(chart.iloc[0].ts)
        if resume_ts is not None and resume_ts < earliest_ts and self.eng.pos is not None:
            self.eng.reconciliation_failed = True
            self.ledger.log(
                f"!! GAP INTEGRITY FAILURE: saved cursor {resume_ts} predates earliest "
                f"available history {earliest_ts} ({(earliest_ts - resume_ts) // 3_600_000}h "
                f"gap) -- open position's missed bars cannot be fully reconciled; new "
                f"entries blocked until this position closes.")

        bars = chart if resume_ts is None else chart[chart["ts"] > resume_ts]
        n = 0
        for _, r in bars.iterrows():
            self._process_bar(r.open, r.high, r.low, r.close, r.volume, int(r.ts), phase)
            n += 1

        new_cursor = int(chart.iloc[-1].ts)
        assert resume_ts is None or new_cursor >= resume_ts, "cursor regression detected on bootstrap"
        self.last_bar_ts = new_cursor
        self.ledger.log(
            f"bootstrap complete: phase={phase} bars_replayed={n} last_bar_ts={self.last_bar_ts} "
            f"({iso(self.last_bar_ts + TIMEFRAME_MS)}) pos={'open' if self.eng.pos else 'flat'} "
            f"equity={self.eng.equity:.2f} reconciliation_failed={self.eng.reconciliation_failed} "
            f"ledger_mismatch={self.ledger_mismatch}")
        self._save_state()

    # ── live loop ────────────────────────────────────────────────────
    def run_once_cycle(self):
        chart = self.feed.retry(self.feed.closed_candles, SYMBOL, TIMEFRAME, 10)
        new = chart[chart["ts"] > self.last_bar_ts].sort_values("ts")
        for _, r in new.iterrows():
            ts = int(r.ts)
            assert self.last_bar_ts is None or ts > self.last_bar_ts, "cursor monotonicity violated"
            self._process_bar(r.open, r.high, r.low, r.close, r.volume, ts, "live")
            self.last_bar_ts = ts
            self._save_state()

    def run_forever(self):
        self.ledger.log(
            "── SOL SHOCK CONTINUATION FORWARD SHADOW — LIVE — "
            "status VALIDATION-INCONCLUSIVE, not a validated edge — public data only ──")
        while True:
            try:
                self.run_once_cycle()
                time.sleep(POLL_SECONDS)
            except KeyboardInterrupt:
                return
            except Exception as e:
                self.ledger.log(f"[loop] {type(e).__name__}: {str(e)[:200]} — continuing")
                time.sleep(POLL_SECONDS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap-only", action="store_true")
    args = ap.parse_args()
    runner = ShadowRunner(runtime_dir=HERE)
    runner.write_config_snapshot()
    runner.bootstrap()
    if args.bootstrap_only:
        runner.ledger.log("bootstrap-only: exiting.")
        return
    runner._save_state()
    runner.run_forever()


if __name__ == "__main__":
    main()
