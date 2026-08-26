"""
Simulated futures portfolio (long + short). 100% sim — no exchange orders.

Fills are modeled at the reference price plus adverse slippage, with a taker
fee charged on both entry and exit. One position at a time (enter only when
flat); exits are driven by the Supertrend trailing stop / flip.
"""
import csv
import os
from datetime import datetime, timezone

import config
from strategy import TrailingStop


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


class Position:
    def __init__(self, side, entry_price, qty, entry_time, reason, init_stop, entry_fee=0.0):
        self.side = side                  # "long" | "short"
        self.entry_price = entry_price
        self.qty = qty                    # BTC
        self.entry_time = entry_time
        self.reason = reason
        self.entry_fee = entry_fee        # USDT, charged at enter() -- kept so close() can
                                           # compute TRUE net P&L (both fees), not just exit fee
        self.tsl = TrailingStop(side, entry_price, init_stop)
        self.flip_armed = False           # Supertrend-flip exit arms after trend confirms

    def unrealized(self, price):
        if self.side == "long":
            return (price - self.entry_price) * self.qty
        return (self.entry_price - price) * self.qty


class PaperEngine:
    def __init__(self, log_fn=print, log_trades=True, symbol=None, params=None,
                 exit_on_flip=None, arm_flip=None, file_tag=None):
        self.cash = config.START_CAPITAL
        self.equity = config.START_CAPITAL
        self.pos = None
        self.realized = 0.0
        self.n_trades = 0
        self.wins = 0
        self.log = log_fn
        self.symbol = symbol
        self.file_tag = file_tag              # distinguishes A/B variant trade logs
        self.init_stop = (params or {}).get("ST_INIT_STOP", config.ST_INIT_STOP)
        self.exit_on_flip = config.EXIT_ON_FLIP if exit_on_flip is None else exit_on_flip
        self.arm_flip = config.ARM_FLIP_AFTER_ENTRY if arm_flip is None else arm_flip
        self.log_trades = log_trades          # False during optimization (no CSV writes)
        self.csv_path = None

        # ── risk-guard state (blocks NEW entries only; never blocks managing
        #    an already-open position) ──
        self.peak_equity = self.equity
        self.consecutive_losses = 0
        self.trading_day = None               # UTC date string, set on first check/close
        self.daily_start_equity = self.equity
        self._last_guard_reason = None        # avoids re-logging the same halt every poll

        # Set when a restart's saved cursor predates the historical data
        # available to reconcile a restored open position -- we can't prove
        # every missed bar was checked, so new entries are blocked until this
        # specific position closes (see run.py SymbolGroup.warmup()).
        self.reconciliation_failed = False

        if log_trades:
            self._init_csv()

    def _init_csv(self):
        if self.file_tag:
            fname = f"trades_{self.file_tag}.csv"
        elif self.symbol:
            fname = f"trades_{self.symbol.replace('/', '')}.csv"
        else:
            fname = config.TRADES_CSV
        self.csv_path = os.path.join(os.path.dirname(__file__), fname)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow([
                    "entry_time", "exit_time", "side", "entry_px", "exit_px",
                    "qty", "gross_pnl", "fees", "net_pnl", "entry_reason",
                    "exit_reason", "equity_after",
                ])

    # ── fills ────────────────────────────────────────────────────
    def _fill(self, ref_price, side, opening):
        """Apply slippage in the adverse direction for the action."""
        adverse_up = (side == "long" and opening) or (side == "short" and not opening)
        slip = config.SLIPPAGE if adverse_up else -config.SLIPPAGE
        return ref_price * (1 + slip)

    def enter(self, side, ref_price, reason, st_dir):
        if self.pos is not None:
            return
        if self._risk_guard_blocked():
            return
        fill = self._fill(ref_price, side, opening=True)

        # Risk-based sizing: size so a full stop-out (at this position's
        # ST_INIT_STOP %) loses approximately RISK_PCT_PER_TRADE % of
        # current equity, capped at POSITION_PCT of equity (no leverage).
        stop_distance_frac = self.init_stop / 100.0
        risk_amount = self.equity * (config.RISK_PCT_PER_TRADE / 100.0)
        if stop_distance_frac <= 0 or risk_amount <= 0:
            return                                     # invalid sizing input -- skip entry
        notional = min(risk_amount / stop_distance_frac, self.equity * config.POSITION_PCT)
        qty = notional / fill
        if qty <= 0 or notional <= 0:
            return                                     # defensive: never size a zero/negative trade

        fee = notional * config.TAKER_FEE
        self.cash -= fee
        self.pos = Position(side, fill, qty, _now(), reason, self.init_stop, entry_fee=fee)
        # arm flip-exit immediately if entry already aligns with supertrend trend
        fav = 1 if side == "long" else -1
        if st_dir == fav:
            self.pos.flip_armed = True
        self.log(f"  >>> ENTER {side.upper():5s} {qty:.5f} @ {fill:,.2f}  "
                 f"({reason})  notional={notional:,.2f}  fee={fee:.2f}  "
                 f"stop={self.pos.tsl.sl_price:,.2f}")

    def close(self, ref_price, exit_reason):
        if self.pos is None:
            return
        p = self.pos
        fill = self._fill(ref_price, p.side, opening=False)
        gross = (fill - p.entry_price) * p.qty if p.side == "long" \
            else (p.entry_price - fill) * p.qty
        exit_fee = (fill * p.qty) * config.TAKER_FEE
        cash_delta = gross - exit_fee          # entry fee already deducted from cash at entry
        net = gross - p.entry_fee - exit_fee   # TRUE net P&L (both fees) -- reconciles with equity
        self.cash += cash_delta
        self.realized += net
        self.equity = self.cash
        self.n_trades += 1
        self._roll_day_if_needed()
        if net > 0:
            self.wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
        self.peak_equity = max(self.peak_equity, self.equity)
        if self.reconciliation_failed:
            self.log("  !! gap-integrity reconciliation resolved (position closed) -- entries resume")
            self.reconciliation_failed = False
        self.log(f"  <<< EXIT  {p.side.upper():5s} @ {fill:,.2f}  "
                 f"({exit_reason})  net={net:+,.2f}  equity={self.equity:,.2f}")
        if self.log_trades:
            with open(self.csv_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    p.entry_time, _now(), p.side, f"{p.entry_price:.2f}", f"{fill:.2f}",
                    f"{p.qty:.6f}", f"{gross:.2f}", f"{p.entry_fee + exit_fee:.2f}",
                    f"{net:.2f}", p.reason, exit_reason, f"{self.equity:.2f}",
                ])
        self.pos = None

    # ── risk guards (entries only) ────────────────────────────────
    def _roll_day_if_needed(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.trading_day is None:
            # first check on a freshly constructed engine -- initialize only,
            # do not treat this as a rollover (would wipe a restored streak)
            self.trading_day = today
            self.daily_start_equity = self.equity
            return
        if self.trading_day != today:
            self.trading_day = today
            self.daily_start_equity = self.equity
            self.consecutive_losses = 0

    def _risk_guard_blocked(self):
        """Returns a reason string if new entries should be blocked, else None.
        Never affects managing an already-open position's exit."""
        self._roll_day_if_needed()
        reason = None
        if self.reconciliation_failed:
            reason = "gap-integrity reconciliation failed -- restart cursor predated available history"
        if reason is None and self.peak_equity > 0:
            dd = (self.peak_equity - self.equity) / self.peak_equity * 100.0
            if dd >= config.MAX_DRAWDOWN_PCT:
                reason = f"max drawdown {dd:.1f}% >= {config.MAX_DRAWDOWN_PCT:.1f}%"
        if reason is None and self.daily_start_equity > 0:
            daily_loss = (self.daily_start_equity - self.equity) / self.daily_start_equity * 100.0
            if daily_loss >= config.DAILY_LOSS_LIMIT_PCT:
                reason = f"daily loss {daily_loss:.1f}% >= {config.DAILY_LOSS_LIMIT_PCT:.1f}%"
        if reason is None and self.consecutive_losses >= config.CONSECUTIVE_LOSS_LIMIT:
            reason = f"{self.consecutive_losses} consecutive losses"
        if reason != self._last_guard_reason:
            if reason:
                self.log(f"  !! risk guard ACTIVE: {reason} -- new entries blocked")
            elif self._last_guard_reason:
                self.log("  !! risk guard cleared -- entries resumed")
            self._last_guard_reason = reason
        return reason

    # ── OHLC-aware stop check (closed bar, using the PRE-ratchet stop) ──
    def check_stop_bar(self, open_, high, low):
        """Test the trailing stop as it existed BEFORE this bar's ratchet
        update, against the bar's full OHLC range. Distinguishes a gap-through
        (reference = the bar's open -- the earliest price this bar actually
        traded at) from an ordinary intrabar touch (reference = the stop
        itself). The candle extreme (low for a long, high for a short) proves
        the stop was crossed at SOME point in the bar; it does not prove
        execution occurred at that extreme, so it is never used as the fill
        reference -- only to detect that a touch happened. Must be called
        BEFORE manage_on_bar() so the ratchet hasn't moved the stop yet."""
        if self.pos is None:
            return
        stop = self.pos.tsl.sl_price
        if self.pos.side == "long":
            if open_ <= stop:
                self.close(open_, "trailing stop (gap)")
            elif low <= stop:
                self.close(stop, "trailing stop (bar)")
        else:
            if open_ >= stop:
                self.close(open_, "trailing stop (gap)")
            elif high >= stop:
                self.close(stop, "trailing stop (bar)")

    # ── per-bar management (closed CHART_TF bar) ─────────────────
    def manage_on_bar(self, snap):
        """Ratchet the trailing stop and check the Supertrend flip exit on a
        closed bar. Does NOT itself check the trailing stop against this
        bar's price -- that is check_stop_bar()'s job, using the PRE-ratchet
        stop, called before this. (A close-based check here would test the
        stop this same call just ratcheted against the same close that
        produced it -- provably never triggers for a valid ratchet, and would
        be redundant with check_stop_bar's OHLC check for a non-ratcheted bar,
        which is a strict superset since low <= close <= high always.)"""
        if self.pos is None:
            return
        p = self.pos
        p.tsl.update(snap["close"])             # trail the profit-stop up

        # optional Supertrend direction-flip exit (off in trailing-stop-only mode)
        if self.exit_on_flip:
            fav = 1 if p.side == "long" else -1
            st_dir = snap["st_dir"]
            if self.arm_flip:
                if not p.flip_armed and st_dir == fav:
                    p.flip_armed = True
                flip_exit = p.flip_armed and st_dir == -fav
            else:
                flip_exit = st_dir == -fav
            if flip_exit:
                self.close(snap["close"], "supertrend flip")

    def check_stop_intrabar(self, price):
        """Check the trailing stop against a live observed tick price."""
        if self.pos is None:
            return
        if self.pos.tsl.hit(price):
            # Fill at the actually observed tick price, never the theoretical
            # stop -- e.g. a gap-through must record the real (worse) price.
            self.close(price, "trailing stop (intrabar)")

    # ── signals -> entries (only when flat) ──────────────────────
    def on_signals(self, snap):
        if self.pos is not None:
            return
        if self._risk_guard_blocked():
            return
        if snap["sig_buy_prime"]:
            self.enter("long", snap["close"], "BUY PRIME", snap["st_dir"])
        elif snap["sig_buy"]:
            self.enter("long", snap["close"], "BUY", snap["st_dir"])
        elif snap["sig_sell_prime"]:
            self.enter("short", snap["close"], "SELL PRIME", snap["st_dir"])
        elif snap["sig_sell"]:
            self.enter("short", snap["close"], "SELL", snap["st_dir"])

    def mark(self, price):
        self.equity = self.cash + (self.pos.unrealized(price) if self.pos else 0.0)
        self.peak_equity = max(self.peak_equity, self.equity)
        return self.equity
