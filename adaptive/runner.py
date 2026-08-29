"""
Main live loop: BTC/ETH/SOL Spot paper trading with ONE shared portfolio.
Wires together market_data (REST-primary, gap-integrity-checked candles),
regime classification, the four strategy modules, the meta-controller's
opportunity scoring/ranking, the risk engine's sizing/breakers/degradation
state, and trailing-stop exit management. Champion/challenger adaptation
runs as a separate, decoupled research cycle (adaptive/research_worker.py)
-- never called synchronously inside this decision loop, per spec section 12.

Fill convention (section 11): stop/exit DETECTION uses OHLC (so a fast
intrabar touch between polls is never missed, mirroring engine.py's
check_stop_bar), but the actual SELL always executes against the CURRENT
live best bid at the moment of detection, not a historical bar price --
this is what section 11 explicitly requires ("SELL reference = current
best bid") while still catching stops that happened between polls.

Restart/recovery (section 18): persists portfolio, positions (each with
its own entry-time trail_state contract, per section 14 -- adaptation
NEVER reaches into an open position), peak equity, daily PnL state, loss
streak, champion versions, rolling stats, market cursors, and adaptation
state. Gap-integrity failures (adaptive/cursor.py, via market_data.py)
block NEW entries for the affected symbol only -- exits and stop
management stay active regardless.
"""
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from adaptive.market_data import MarketData, SYMBOLS, TIMEFRAMES, TF_MS  # noqa: E402
from adaptive.portfolio import Portfolio  # noqa: E402
from adaptive.risk_engine import RiskEngine, RiskLimits, DegradationState  # noqa: E402
from adaptive.regime import classify_from_df  # noqa: E402
from adaptive.strategies import (  # noqa: E402
    trend_momentum, volatility_breakout, mean_reversion, shock_continuation,
    TREND_DEFAULT_PARAMS, BREAKOUT_DEFAULT_PARAMS, MEANREV_DEFAULT_PARAMS,
)
from adaptive.stats_store import StatsStore  # noqa: E402
from adaptive.adaptation import AdaptationEngine  # noqa: E402
from adaptive.meta_controller import (  # noqa: E402
    score_opportunity, rank_and_select, rolling_correlation_matrix, friction_pct,
)
from adaptive.trailing import new_trail_state, update_trail, check_stop_bar  # noqa: E402

START_CAPITAL = 10_000.0
FEE_RATE = 0.001       # 0.1% Binance Spot taker, configurable
SLIPPAGE = 0.0002
TYPICAL_SPREAD_PCT = 0.02
PRIMARY_STOP_TF = "5m"
REGIME_TF = "1h"
MAX_POSITIONS = 3
POLL_SECONDS = 30
RETURNS_WINDOW_BARS = 100

DEFAULT_PARAMS = {
    "trend_momentum": dict(TREND_DEFAULT_PARAMS),
    "volatility_breakout": dict(BREAKOUT_DEFAULT_PARAMS),
    "mean_reversion": dict(MEANREV_DEFAULT_PARAMS),
}


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def now_ms():
    return int(time.time() * 1000)


class AdaptiveRunner:
    def __init__(self, runtime_dir: str, market_data: Optional[MarketData] = None):
        self.runtime_dir = runtime_dir
        os.makedirs(runtime_dir, exist_ok=True)
        self.md = market_data if market_data is not None else MarketData()
        self.portfolio = Portfolio(START_CAPITAL)
        self.risk_engine = RiskEngine(RiskLimits())
        self.stats_store = StatsStore()
        self.adaptation = AdaptationEngine(os.path.join(runtime_dir, "adaptation_log.jsonl"))
        self.daily_start_equity = START_CAPITAL
        self.trading_day: Optional[str] = None
        self.consecutive_losses = 0
        self.reconciliation_failed = set()
        self.state_path = os.path.join(runtime_dir, "state.json")
        self.trades_csv_path = os.path.join(runtime_dir, "trades.csv")
        self.decisions_path = os.path.join(runtime_dir, "decisions.jsonl")
        self.log_path = os.path.join(runtime_dir, "paper.log")
        self._init_trades_csv()

    # ── logging ──────────────────────────────────────────────────────
    def log(self, msg: str):
        line = f"{now_iso()}  {msg}"
        print(line, flush=True)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")

    def _log_decision(self, symbol: str, action: str, detail: dict):
        record = {"ts_iso": now_iso(), "symbol": symbol, "action": action, **detail}
        with open(self.decisions_path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def _init_trades_csv(self):
        if not os.path.exists(self.trades_csv_path):
            import csv
            with open(self.trades_csv_path, "w", newline="") as f:
                csv.writer(f).writerow([
                    "symbol", "strategy", "regime", "entry_ts_ms", "exit_ts_ms",
                    "entry_price", "exit_price", "qty", "entry_fee", "exit_fee",
                    "net_pnl", "exit_reason", "equity_after",
                ])

    def _write_trade_csv(self, record: dict):
        import csv
        with open(self.trades_csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                record["symbol"], record["strategy"], record["regime"],
                record["entry_ts_ms"], record["exit_ts_ms"], record["entry_price"],
                record["exit_price"], record["qty"], record["entry_fee"], record["exit_fee"],
                record["net_pnl"], record["exit_reason"], record["cash_after"],
            ])

    # ── daily rollover (mirrors engine.py's _roll_day_if_needed) ────
    def _roll_day_if_needed(self, equity: float):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.trading_day is None:
            self.trading_day = today
            self.daily_start_equity = equity
            return
        if self.trading_day != today:
            self.trading_day = today
            self.daily_start_equity = equity
            self.consecutive_losses = 0

    # ── position management (exits are NEVER blocked) ────────────────
    def _manage_position(self, symbol: str, prices: Dict[str, float]):
        pos = self.portfolio.positions.get(symbol)
        if pos is None:
            return
        df = self.md.candles[(symbol, PRIMARY_STOP_TF)]
        if len(df) == 0:
            return
        last = df.iloc[-1]
        o, h, l, c = float(last["open"]), float(last["high"]), float(last["low"]), float(last["close"])
        prices[symbol] = c

        hit = check_stop_bar(pos.trail_state, o, h, l) is not None
        if hit:
            try:
                bid, _ = self.md.best_bid_ask(symbol)
            except Exception as e:
                self.log(f"!! could not fetch live bid for {symbol} stop exit: {type(e).__name__} -- using bar close")
                bid = c
            rec = self.portfolio.sell(symbol, best_bid=bid, fee_rate=FEE_RATE, slippage=SLIPPAGE,
                                        ts_ms=now_ms(), exit_reason="trailing stop")
            self._on_trade_closed(rec)
            return

        pos.trail_state = update_trail(pos.trail_state, c)
        pos.stop_price = pos.trail_state["sl_price"]

    def _exit_signal_management(self, symbol: str, signal_direction: str, prices: Dict[str, float]):
        pos = self.portfolio.positions.get(symbol)
        if pos is None or signal_direction != "exit_long":
            return
        try:
            bid, _ = self.md.best_bid_ask(symbol)
        except Exception:
            bid = prices.get(symbol, pos.entry_price)
        rec = self.portfolio.sell(symbol, best_bid=bid, fee_rate=FEE_RATE, slippage=SLIPPAGE,
                                    ts_ms=now_ms(), exit_reason=f"{pos.strategy} exit signal")
        self._on_trade_closed(rec)

    def _on_trade_closed(self, rec: dict):
        pnl_pct = (rec["net_pnl"] / (rec["qty"] * rec["entry_price"])) * 100 if rec["qty"] * rec["entry_price"] else 0.0
        self.stats_store.record_trade(rec["symbol"], rec["strategy"], rec["regime"], pnl_pct)
        self.consecutive_losses = 0 if rec["net_pnl"] > 0 else self.consecutive_losses + 1
        self._write_trade_csv(rec)
        self._log_decision(rec["symbol"], "exited", rec)
        self.log(f"  <<< SELL {rec['symbol']} @ {rec['exit_price']:.4f} ({rec['exit_reason']}) "
                 f"net={rec['net_pnl']:+.2f}")

    # ── bootstrap / restart ─────────────────────────────────────────
    def bootstrap(self):
        state = self._load_state()
        if state:
            self._restore(state)
        for sym in SYMBOLS:
            for tf in TIMEFRAMES:
                result = self.md.poll(sym, tf)
                if result.gap_detected and tf == PRIMARY_STOP_TF and self.portfolio.held_qty(sym) > 0:
                    self.reconciliation_failed.add(sym)
                    self.log(f"!! GAP INTEGRITY FAILURE {sym}/{tf}: {result.gap_detail} "
                             f"-- new entries blocked for {sym} until reconciled")
        self.log(f"bootstrap complete: positions={list(self.portfolio.positions.keys())} "
                 f"cash={self.portfolio.cash:.2f} reconciliation_failed={self.reconciliation_failed}")
        self._save_state()

    def _load_state(self) -> Optional[dict]:
        if not os.path.exists(self.state_path):
            return None
        try:
            with open(self.state_path) as f:
                return json.load(f)
        except Exception:
            return None

    def _restore(self, state: dict):
        from adaptive.portfolio import Position
        p = state.get("portfolio", {})
        self.portfolio.cash = p.get("cash", self.portfolio.cash)
        self.portfolio.realized_pnl = p.get("realized_pnl", 0.0)
        self.portfolio.n_trades = p.get("n_trades", 0)
        self.portfolio.wins = p.get("wins", 0)
        self.portfolio.peak_equity = p.get("peak_equity", self.portfolio.cash)
        for sym, pd_ in (p.get("positions") or {}).items():
            pos = Position(symbol=sym, qty=pd_["qty"], entry_price=pd_["entry_price"],
                            entry_ref_price=pd_["entry_ref_price"], entry_ts_ms=pd_["entry_ts_ms"],
                            entry_fee=pd_["entry_fee"], strategy=pd_["strategy"], regime=pd_["regime"],
                            confidence=pd_["confidence"], stop_price=pd_["stop_price"],
                            initial_stop_pct=pd_["initial_stop_pct"], risk_amount_usdt=pd_["risk_amount_usdt"],
                            trail_state=pd_["trail_state"])
            self.portfolio.positions[sym] = pos
        self.daily_start_equity = state.get("daily_start_equity", self.portfolio.cash)
        self.trading_day = state.get("trading_day")
        self.consecutive_losses = state.get("consecutive_losses", 0)
        self.stats_store = StatsStore.from_dict(state.get("stats_store"))
        self.adaptation = AdaptationEngine.from_dict(state.get("adaptation"), self.adaptation.log_path)
        self.risk_engine.state = DegradationState(state.get("risk_state", "NORMAL"))

    def _save_state(self):
        state = {
            "saved_at_iso": now_iso(),
            "portfolio": {
                "cash": self.portfolio.cash, "realized_pnl": self.portfolio.realized_pnl,
                "n_trades": self.portfolio.n_trades, "wins": self.portfolio.wins,
                "peak_equity": self.portfolio.peak_equity,
                "positions": {sym: {
                    "qty": pos.qty, "entry_price": pos.entry_price, "entry_ref_price": pos.entry_ref_price,
                    "entry_ts_ms": pos.entry_ts_ms, "entry_fee": pos.entry_fee, "strategy": pos.strategy,
                    "regime": pos.regime, "confidence": pos.confidence, "stop_price": pos.stop_price,
                    "initial_stop_pct": pos.initial_stop_pct, "risk_amount_usdt": pos.risk_amount_usdt,
                    "trail_state": pos.trail_state,
                } for sym, pos in self.portfolio.positions.items()},
            },
            "daily_start_equity": self.daily_start_equity, "trading_day": self.trading_day,
            "consecutive_losses": self.consecutive_losses,
            "reconciliation_failed": sorted(self.reconciliation_failed),
            "stats_store": self.stats_store.to_dict(),
            "adaptation": self.adaptation.to_dict(),
            "risk_state": self.risk_engine.state.value,
            "market_cursors": {f"{s}|{tf}": ts for (s, tf), ts in self.md.cursors.items()},
        }
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2, default=str)
        os.replace(tmp, self.state_path)

    def snapshot(self, prices: Dict[str, float]) -> dict:
        """Section 20 dashboard/status: PORTFOLIO, POSITIONS, AI BRAIN, MARKET."""
        snap = self.portfolio.snapshot(prices)
        equity = snap["equity"]
        daily_pnl = equity - self.daily_start_equity
        heat_usdt = sum(p.risk_amount_usdt for p in self.portfolio.positions.values())
        snap["daily_pnl_usdt"] = daily_pnl
        snap["daily_pnl_pct"] = (daily_pnl / self.daily_start_equity * 100) if self.daily_start_equity else 0.0
        snap["portfolio_heat_usdt"] = heat_usdt
        snap["portfolio_heat_pct"] = (heat_usdt / equity * 100) if equity else 0.0
        snap["risk_state"] = self.risk_engine.state.value
        snap["risk_state_reason"] = self.risk_engine.state_reason
        snap["consecutive_losses"] = self.consecutive_losses
        snap["reconciliation_failed"] = sorted(self.reconciliation_failed)

        snap["ai_brain"] = {
            "champions": self.adaptation.champions,
            "strategies_active": list(DEFAULT_PARAMS.keys()) + ["shock_continuation"],
            "risk_state": self.risk_engine.state.value,
            "risk_state_reason": self.risk_engine.state_reason,
            "last_promotions": self.adaptation.last_promotion_ts,
        }

        market = {}
        for sym in SYMBOLS:
            df5 = self.md.candles[(sym, PRIMARY_STOP_TF)]
            entry = {"last_price": prices.get(sym)}
            if len(df5):
                entry["last_closed_bar_ts"] = int(df5["ts"].iloc[-1])
                entry["candle_age_sec"] = (now_ms() - (int(df5["ts"].iloc[-1]) + TF_MS[PRIMARY_STOP_TF])) / 1000.0
            try:
                bid, ask = self.md.best_bid_ask(sym)
                entry["bid"], entry["ask"] = bid, ask
                entry["spread_pct"] = ((ask - bid) / ((ask + bid) / 2) * 100) if bid and ask else None
            except Exception:
                entry["bid"] = entry["ask"] = entry["spread_pct"] = None
            entry["gap_flag"] = self.md.gap_flags.get((sym, PRIMARY_STOP_TF), False)
            entry["reconciliation_failed"] = sym in self.reconciliation_failed
            market[sym] = entry
        snap["market"] = market
        return snap

    def write_dashboard(self, prices: Dict[str, float]):
        path = os.path.join(self.runtime_dir, "dashboard.json")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"written_at_iso": now_iso(), **self.snapshot(prices)}, f, indent=2, default=str)
        os.replace(tmp, path)

    # ── one decision cycle ───────────────────────────────────────────
    def run_once_cycle(self):
        poll_results = self.md.poll_all()
        # A strategy may only enter on a bar that was FRESHLY polled this
        # cycle -- never on history that was already sitting in
        # market_data.candles before this cycle ran (e.g. loaded wholesale
        # during bootstrap()). Without this gate, the first cycle after a
        # cold start would evaluate signals against whatever the latest
        # historical bar happened to be and could fabricate an entry from
        # an event that occurred before the process ever started --
        # exactly the failure mode the SOL shadow's warmup/live phase
        # separation exists to prevent (see run_shadow.py's docstring).
        fresh = {key: len(result.bars) > 0 for key, result in poll_results.items()}
        for (sym, tf), result in poll_results.items():
            if result.gap_detected and tf == PRIMARY_STOP_TF:
                self.reconciliation_failed.add(sym)
                self.log(f"!! GAP DETECTED {sym}/{tf}: {result.gap_detail} -- new entries blocked for {sym}")

        prices: Dict[str, float] = {}
        for sym in SYMBOLS:
            self._manage_position(sym, prices)
        for sym in SYMBOLS:
            if sym not in prices:
                df = self.md.candles[(sym, PRIMARY_STOP_TF)]
                if len(df):
                    prices[sym] = float(df.iloc[-1]["close"])

        if len(prices) < len(SYMBOLS):
            self._save_state()
            self.write_dashboard(prices)
            return  # not enough data yet to safely evaluate entries this cycle

        equity = self.portfolio.update_peak_equity(prices)
        self._roll_day_if_needed(equity)

        drawdown_pct = ((self.portfolio.peak_equity - equity) / self.portfolio.peak_equity * 100
                         if self.portfolio.peak_equity else 0.0)
        recent_pf, recent_exp = self._recent_aggregate_stats()
        new_state = self.risk_engine.evaluate_degradation(
            drawdown_pct=drawdown_pct, loss_streak=self.consecutive_losses,
            recent_expectancy=recent_exp, recent_pf=recent_pf)
        if new_state != self.risk_engine.state:
            self.log(f"  risk state {self.risk_engine.state.value} -> {new_state.value}")
            self.risk_engine.set_state(new_state, "degradation evaluation")

        breaker_reason = self.risk_engine.check_breakers(
            equity=equity, peak_equity=self.portfolio.peak_equity,
            daily_start_equity=self.daily_start_equity, consecutive_losses=self.consecutive_losses)

        candidates = []
        corr_matrix = self._correlation_matrix()
        for sym in SYMBOLS:
            if self.portfolio.held_qty(sym) > 0 or sym in self.reconciliation_failed:
                continue
            ok, reasons = self.md.market_quality(sym, primary_tf=PRIMARY_STOP_TF)
            if not ok:
                self._log_decision(sym, "skipped_market_quality", {"reasons": reasons})
                continue
            df_1h = self.md.candles[(sym, "1h")]
            df_15m = self.md.candles[(sym, "15m")]
            if len(df_1h) == 0 or len(df_15m) == 0:
                continue
            regime = classify_from_df(df_1h)
            fresh_1h = fresh.get((sym, "1h"), False)
            fresh_15m = fresh.get((sym, "15m"), False)
            signals = {
                "trend_momentum": (trend_momentum(df_1h, self.adaptation.get_champion_params(
                    "trend_momentum", DEFAULT_PARAMS["trend_momentum"])), fresh_1h),
                "volatility_breakout": (volatility_breakout(df_15m, self.adaptation.get_champion_params(
                    "volatility_breakout", DEFAULT_PARAMS["volatility_breakout"])), fresh_15m),
                "mean_reversion": (mean_reversion(df_15m, self.adaptation.get_champion_params(
                    "mean_reversion", DEFAULT_PARAMS["mean_reversion"])), fresh_15m),
            }
            if fresh_1h:
                # shock_continuation is STATEFUL (mutates its detector's
                # rolling tr_hist/prev_close on every call) -- it must be
                # called AT MOST ONCE per new 1h bar, never redundantly on
                # an unchanged bar between polls, or its internal ATR
                # history gets corrupted by re-consuming the same bar
                # dozens of times before the next one closes.
                signals["shock_continuation"] = (shock_continuation(sym, df_1h), True)
            for strat_name, (sig, is_fresh) in signals.items():
                if sig.direction == "long":
                    if not is_fresh:
                        continue  # see the `fresh` comment above -- never enter on stale/historical bars
                    fpen = friction_pct(FEE_RATE, SLIPPAGE, TYPICAL_SPREAD_PCT)
                    opp = score_opportunity(symbol=sym, signal=sig, regime=regime, stats_store=self.stats_store,
                                             friction_penalty_pct=fpen, already_selected=[],
                                             correlation_matrix=corr_matrix)
                    candidates.append(opp)
                elif sig.direction == "exit_long":
                    # exits are never gated by freshness -- recomputing the
                    # same exit decision on unchanged data is redundant but
                    # harmless, unlike entries, and exits must never be blocked
                    self._exit_signal_management(sym, "exit_long", prices)

        selected = rank_and_select(candidates, corr_matrix, max_positions=MAX_POSITIONS)
        self._log_decision("*", "cycle_ranking", {
            "candidates": [{"symbol": o.symbol, "strategy": o.strategy, "score": o.score} for o in candidates],
            "selected": [o.symbol for o in selected], "risk_state": self.risk_engine.state.value,
        })

        if breaker_reason:
            self._log_decision("*", "entries_blocked_portfolio_breaker", {"reason": breaker_reason})
        else:
            for opp in selected:
                if opp.symbol in self.reconciliation_failed:
                    continue
                sizing = self.risk_engine.size_entry(
                    equity=equity, cash=self.portfolio.cash,
                    current_portfolio_heat_usdt=sum(p.risk_amount_usdt for p in self.portfolio.positions.values()),
                    current_crypto_value_usdt=self.portfolio.crypto_value(prices),
                    stop_distance_frac=opp.stop_pct / 100.0, confidence=opp.confidence)
                if not sizing.approved:
                    self._log_decision(opp.symbol, "rejected_sizing",
                                        {"reason": sizing.reason, "binding": sizing.binding_constraint})
                    continue
                try:
                    bid, ask = self.md.best_bid_ask(opp.symbol)
                except Exception as e:
                    self._log_decision(opp.symbol, "rejected_price_fetch_failed", {"error": str(e)})
                    continue
                pos = self.portfolio.buy(
                    opp.symbol, best_ask=ask, notional_usdt=sizing.notional_usdt,
                    fee_rate=FEE_RATE, slippage=SLIPPAGE, ts_ms=now_ms(),
                    strategy=opp.strategy, regime=opp.regime, confidence=opp.confidence,
                    stop_price=ask * (1 - opp.stop_pct / 100.0), initial_stop_pct=opp.stop_pct,
                    risk_amount_usdt=sizing.risk_amount_usdt)
                pos.trail_state = new_trail_state(pos.entry_price, opp.stop_pct)
                pos.stop_price = pos.trail_state["sl_price"]
                self._log_decision(opp.symbol, "entered", {
                    "strategy": opp.strategy, "regime": opp.regime, "score": opp.score,
                    "notional": sizing.notional_usdt, "risk": sizing.risk_amount_usdt,
                    "correlation_penalty": opp.correlation_penalty, "fees_slippage_est": opp.friction_penalty,
                })
                self.log(f"  >>> BUY {opp.symbol} {pos.qty:.6f} @ {pos.entry_price:.4f} "
                         f"({opp.strategy}/{opp.regime}) risk={sizing.risk_amount_usdt:.2f}")

        self._save_state()
        self.write_dashboard(prices)

    def _recent_aggregate_stats(self):
        all_trades = [t for c in self.stats_store.cells.values() for t in c.trades[-20:]]
        if len(all_trades) < 10:
            return None, None
        wins = sum(t for t in all_trades if t > 0)
        losses = sum(abs(t) for t in all_trades if t <= 0)
        pf = (wins / losses) if losses > 0 else (float("inf") if wins > 0 else None)
        exp = sum(all_trades) / len(all_trades)
        return pf, exp

    def _correlation_matrix(self):
        returns_by_symbol = {}
        for sym in SYMBOLS:
            df = self.md.candles[(sym, PRIMARY_STOP_TF)]
            if len(df) >= 11:
                closes = df["close"].tail(RETURNS_WINDOW_BARS + 1).values
                returns_by_symbol[sym] = list((closes[1:] - closes[:-1]) / closes[:-1])
        if len(returns_by_symbol) < 2:
            return {}
        return rolling_correlation_matrix(returns_by_symbol)

    def run_forever(self):
        self.log("── ADAPTIVE SPOT PAPER TRADER — BTC/ETH/SOL — LIVE ──")
        while True:
            try:
                self.run_once_cycle()
                time.sleep(POLL_SECONDS)
            except KeyboardInterrupt:
                return
            except Exception as e:
                self.log(f"[loop] {type(e).__name__}: {str(e)[:200]} -- continuing")
                time.sleep(POLL_SECONDS)


def main():
    runtime_dir = os.path.join(ROOT, "adaptive_runtime")
    runner = AdaptiveRunner(runtime_dir)
    for sym in SYMBOLS:
        runner.md.verify_spot_market(sym)
    runner.bootstrap()
    runner.run_forever()


if __name__ == "__main__":
    main()
