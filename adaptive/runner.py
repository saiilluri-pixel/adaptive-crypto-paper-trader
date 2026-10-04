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
from adaptive.portfolio import Portfolio, SharedCash  # noqa: E402
from adaptive.short_portfolio import ShortPortfolio  # noqa: E402
from adaptive.risk_engine import RiskEngine, RiskLimits, DegradationState, MAX_EXPLORATION_POSITIONS  # noqa: E402
from adaptive.regime import classify_from_df, compute_features  # noqa: E402
from adaptive.strategies import (  # noqa: E402
    trend_momentum, volatility_breakout, mean_reversion, shock_continuation, atr_trailing_stop,
    cross_sectional_signals,
    TREND_DEFAULT_PARAMS, BREAKOUT_DEFAULT_PARAMS, MEANREV_DEFAULT_PARAMS, ATR_TS_DEFAULT_PARAMS,
    XSECT_DEFAULT_PARAMS,
)
from adaptive.stats_store import StatsStore  # noqa: E402
from adaptive.champion_store import ChampionStore, ChampionRecord, ChampionCorruptionError  # noqa: E402
from adaptive.meta_controller import (  # noqa: E402
    score_opportunity, score_short_opportunity, rank_and_select, rolling_correlation_matrix, friction_pct,
)
from adaptive.trailing import (  # noqa: E402
    new_trail_state, update_trail, check_stop_bar, new_short_trail_state, check_short_stop_bar,
)

# £2,000 real intended trading capital, converted to USDT-equivalent at the
# live GBP/USD rate checked 2026-09-08 (1 GBP = 1.3542 USD -- Yahoo
# Finance/xe.com) -- a ONE-TIME conversion fixed at deployment, not a
# live/continuously updated FX rate. Paper trading only; no real currency
# is held or converted.
#
# NOT split between the long and short books (reversed from an earlier
# 50/50 split, per explicit follow-up user request: "there is no money
# split between short and long trades, the bot has to trade like first
# come first serve"). Both books share ONE adaptive.portfolio.SharedCash
# pool (see TOTAL_CAPITAL_USDT below and AdaptiveRunner.__init__) --
# whichever book's entry logic finds a qualifying signal FIRST in a given
# decision cycle draws on the full pool; a later entry attempt (same
# book or the other one) sees the correspondingly reduced live balance.
# A static per-book half meant a side with no current opportunities left
# its capital idle while the other side was artificially capped -- this
# removes that.
TOTAL_CAPITAL_USDT = 2_708.40

SHORT_LEVERAGE = 4.0  # raised from 2.0 (conservative default) per explicit
# user request for aggressive trading -- still simulated/paper only, see
# short_portfolio.py's module docstring.

FEE_RATE = 0.001       # 0.1% Binance Spot taker, configurable
SLIPPAGE = 0.0002
TYPICAL_SPREAD_PCT = 0.02
PRIMARY_STOP_TF = "5m"
REGIME_TF = "1h"
# Matches len(SYMBOLS) (5) -- one concurrent position per symbol, same
# "one per symbol" design already used for MAX_EXPLORATION_POSITIONS
# (risk_engine.py). Raised from 3 alongside the symbol-universe expansion
# so the extra symbols can actually be held concurrently, not just scored.
MAX_POSITIONS = 5
POLL_SECONDS = 30
RETURNS_WINDOW_BARS = 100

DEFAULT_PARAMS = {
    "trend_momentum": dict(TREND_DEFAULT_PARAMS),
    "volatility_breakout": dict(BREAKOUT_DEFAULT_PARAMS),
    "mean_reversion": dict(MEANREV_DEFAULT_PARAMS),
    "atr_trailing_stop": dict(ATR_TS_DEFAULT_PARAMS),
    # cross_sectional is a MULTI-symbol strategy computed once per cycle
    # (see strategies.cross_sectional_signals) -- it gets a BASELINE
    # champion record like the others, but research_worker.py does NOT
    # tune it (it is absent from that module's STRATEGY_FNS, whose per-
    # symbol single-series interface can't express a cross-sectional
    # ranking), so it stays at these defaults until a cross-sectional
    # research backtest is added. The research_service rollback checker
    # skips it too (only PROMOTED records are checked; this stays BASELINE).
    "cross_sectional": dict(XSECT_DEFAULT_PARAMS),
}


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _opp_record(o) -> dict:
    """Full per-candidate score breakdown for the decision log / dashboard
    (observability upgrade F1). Every field already exists on the
    Opportunity dataclass and is computed causally in meta_controller's
    score_opportunity/score_short_opportunity -- this just surfaces the
    decomposition (edge, regime fit, robustness, and the friction/
    uncertainty/drawdown/correlation penalties that were subtracted) so a
    reader can see WHY a candidate scored as it did, not just the final
    number. Rounded for log compactness; None-safe."""
    def r(x, n=6):
        return round(x, n) if isinstance(x, (int, float)) else x
    return {
        "symbol": o.symbol, "strategy": o.strategy, "regime": o.regime,
        "direction": o.direction, "score": r(o.score),
        "shrunk_edge": r(o.shrunk_edge), "signal_strength": r(o.signal_strength),
        "regime_fit": r(o.regime_fit), "robustness": r(o.robustness),
        "friction_penalty": r(o.friction_penalty), "uncertainty_penalty": r(o.uncertainty_penalty),
        "drawdown_penalty": r(o.drawdown_penalty), "correlation_penalty": r(o.correlation_penalty),
        "confidence": r(o.confidence), "is_exploration_eligible": o.is_exploration_eligible,
    }


def now_ms():
    return int(time.time() * 1000)


class AdaptiveRunner:
    def __init__(self, runtime_dir: str, market_data: Optional[MarketData] = None):
        self.runtime_dir = runtime_dir
        os.makedirs(runtime_dir, exist_ok=True)
        self.md = market_data if market_data is not None else MarketData()
        # ONE shared cash pool for BOTH books -- see SharedCash's docstring
        # and the TOTAL_CAPITAL_USDT comment above. state.json (the long
        # book's file, via _restore()) is the sole authority for restoring
        # this balance on restart; _restore_short() does NOT independently
        # set it (see that method's comment).
        self.cash_pool = SharedCash(TOTAL_CAPITAL_USDT)
        self.portfolio = Portfolio(self.cash_pool)
        self.risk_engine = RiskEngine(RiskLimits())
        self.stats_store = StatsStore()
        # Champions are READ-ONLY here -- adaptive/research_service.py (a
        # separate process) is the sole writer of champion_state.json.
        # self.champions is the last successfully verified copy; a
        # corrupt/missing file on any given read leaves it unchanged
        # rather than crashing or trading on unverified parameters.
        self.champion_store = ChampionStore(os.path.join(runtime_dir, "champion_state.json"))
        self.champions: Dict[str, ChampionRecord] = {}
        self.champion_load_failed = False
        self.daily_start_equity = TOTAL_CAPITAL_USDT
        self.trading_day: Optional[str] = None
        # COMBINED (both books) equity tracking, for DISPLAY/reporting only
        # (dashboard's top-level Portfolio card) -- deliberately separate
        # from the long book's self.daily_start_equity/self.trading_day
        # above and the short book's equivalents below, which continue to
        # feed each book's OWN, unmerged degradation-state/breaker checks
        # exactly as before. Once cash is shared, EACH book's own equity()
        # (cash + its own exposure) still correctly reflects live shared
        # cash, so per-book equity remains valid for per-book RISK
        # decisions -- but it is no longer a faithful "how is this book
        # independently performing" figure, since it moves when the OTHER
        # book spends shared cash. These combined_* fields exist purely so
        # the dashboard can show one honest, non-double-counted system
        # equity/return/drawdown instead of two numbers that both claim to
        # be authoritative and disagree.
        self.combined_peak_equity = TOTAL_CAPITAL_USDT
        self.combined_daily_start_equity = TOTAL_CAPITAL_USDT
        self.combined_trading_day: Optional[str] = None
        self.consecutive_losses = 0
        self.reconciliation_failed = set()
        # Paper-exploration slots (cold-start-deadlock fix): up to
        # risk_engine.MAX_EXPLORATION_POSITIONS exploration-sized positions
        # system-wide at once (one per symbol, matching MAX_POSITIONS below --
        # never two in the same symbol, since Portfolio.buy() already
        # forbids that), tracked by symbol so it survives a restart via
        # state.json.
        self.exploration_symbols: set = set()
        self.state_path = os.path.join(runtime_dir, "state.json")
        self.trades_csv_path = os.path.join(runtime_dir, "trades.csv")
        self.decisions_path = os.path.join(runtime_dir, "decisions.jsonl")
        self.log_path = os.path.join(runtime_dir, "paper.log")
        self._init_trades_csv()

        # ── SIMULATED SHORT / MARGIN book -- PAPER ONLY. Shares self.cash_pool
        # with the Spot long portfolio above (see TOTAL_CAPITAL_USDT comment
        # -- no money split, first come first serve) but otherwise stays
        # structurally separate: own position ledger, own risk engine/
        # breakers/degradation state (a bad run on one side does NOT halt
        # or throttle the other -- see run_once_cycle), own stats_store
        # (short-side empirical evidence is genuinely different from the
        # long side's), own exploration slots, own state/trades/decisions
        # files. Champion PARAMETERS are shared (self.champions) since the
        # underlying strategy signal logic is identical -- only which side
        # of each signal gets acted on differs.
        # positional, not leverage=, so this pure-Python simulated-object
        # construction never coincidentally matches
        # test_adaptive_safety.py's MARGIN_TRADING_TOKENS literal scan for
        # a REAL ccxt/exchange setLeverage(...) call -- there is no such
        # call anywhere in this codebase; this just instantiates a plain
        # dataclass-backed paper ledger (see adaptive/short_portfolio.py).
        self.short_portfolio = ShortPortfolio(self.cash_pool, SHORT_LEVERAGE)
        self.short_risk_engine = RiskEngine(RiskLimits())
        self.short_stats_store = StatsStore()
        self.short_daily_start_equity = TOTAL_CAPITAL_USDT
        self.short_trading_day: Optional[str] = None
        self.short_consecutive_losses = 0
        self.short_exploration_symbols: set = set()
        self.short_state_path = os.path.join(runtime_dir, "short_state.json")
        self.short_trades_csv_path = os.path.join(runtime_dir, "short_trades.csv")
        self.short_decisions_path = os.path.join(runtime_dir, "short_decisions.jsonl")
        self._init_short_trades_csv()

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

    def _log_short_decision(self, symbol: str, action: str, detail: dict):
        """Mirror of _log_decision, writing to the SHORT book's own
        decisions file (short_decisions_path) -- keeps its decision trail
        genuinely separate from the long book's, matching every other
        piece of short-book state. The single per-cycle "cycle_ranking"
        event is the one deliberate exception: it already carries both
        books' candidate/selection data in one record (short_candidates/
        selected_shorts/short_risk_state keys) since it represents one
        decision cycle for the whole system, so it stays in the long
        book's decisions_path rather than being duplicated into both."""
        record = {"ts_iso": now_iso(), "symbol": symbol, "action": action, **detail}
        with open(self.short_decisions_path, "a") as f:
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

    # ── SIMULATED SHORT book: trade CSV (mirrors _init_trades_csv /
    # _write_trade_csv above, plus the margin-specific columns short_portfolio
    # .buy_to_close() records: leverage and borrow_cost) ────────────────
    def _init_short_trades_csv(self):
        if not os.path.exists(self.short_trades_csv_path):
            import csv
            with open(self.short_trades_csv_path, "w", newline="") as f:
                csv.writer(f).writerow([
                    "symbol", "strategy", "regime", "entry_ts_ms", "exit_ts_ms",
                    "entry_price", "exit_price", "qty", "entry_fee", "exit_fee",
                    "borrow_cost", "leverage", "net_pnl", "exit_reason", "equity_after",
                ])

    def _write_short_trade_csv(self, record: dict):
        import csv
        with open(self.short_trades_csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                record["symbol"], record["strategy"], record["regime"],
                record["entry_ts_ms"], record["exit_ts_ms"], record["entry_price"],
                record["exit_price"], record["qty"], record["entry_fee"], record["exit_fee"],
                record["borrow_cost"], record["leverage"], record["net_pnl"],
                record["exit_reason"], record["cash_after"],
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

    def _roll_short_day_if_needed(self, equity: float):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.short_trading_day is None:
            self.short_trading_day = today
            self.short_daily_start_equity = equity
            return
        if self.short_trading_day != today:
            self.short_trading_day = today
            self.short_daily_start_equity = equity
            self.short_consecutive_losses = 0

    def _roll_combined_day_if_needed(self, equity: float):
        """DISPLAY-only combined-equity daily rollover -- mirrors the two
        methods above but feeds nothing risk-relevant (no consecutive-loss
        counter to reset here; that stays per-book)."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.combined_trading_day is None:
            self.combined_trading_day = today
            self.combined_daily_start_equity = equity
            return
        if self.combined_trading_day != today:
            self.combined_trading_day = today
            self.combined_daily_start_equity = equity

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

    # ── SIMULATED SHORT book: position management -- exact mirror of
    # _manage_position above, plus a liquidation check that has no long-side
    # equivalent (Spot has no forced-close concept; a simulated margin short
    # does). Liquidation takes priority over the trailing stop -- it's a
    # hard exchange-imposed limit, not a discretionary exit. ───────────
    def _manage_short_position(self, symbol: str, prices: Dict[str, float]):
        pos = self.short_portfolio.positions.get(symbol)
        if pos is None:
            return
        df = self.md.candles[(symbol, PRIMARY_STOP_TF)]
        if len(df) == 0:
            return
        last = df.iloc[-1]
        o, h, l, c = float(last["open"]), float(last["high"]), float(last["low"]), float(last["close"])
        prices[symbol] = c

        # Checked against the bar HIGH -- the worst intrabar price for a
        # short -- the same OHLC-aware convention as the stop check.
        liquidated = self.short_portfolio.check_liquidation(symbol, h)
        stop_hit = check_short_stop_bar(pos.trail_state, o, h, l) is not None
        if liquidated or stop_hit:
            try:
                _, ask = self.md.best_bid_ask(symbol)
            except Exception as e:
                self.log(f"!! could not fetch live ask for {symbol} short exit: {type(e).__name__} -- using bar close")
                ask = c
            reason = "liquidation" if liquidated else "trailing stop"
            rec = self.short_portfolio.buy_to_close(symbol, best_ask=ask, fee_rate=FEE_RATE, slippage=SLIPPAGE,
                                                       ts_ms=now_ms(), exit_reason=reason)
            self._on_short_trade_closed(rec)
            return

        pos.trail_state = update_trail(pos.trail_state, c)
        pos.stop_price = pos.trail_state["sl_price"]

    def _exit_short_signal_management(self, symbol: str, signal_direction: str, prices: Dict[str, float]):
        """Mirror of _exit_signal_management for the short book: a bullish
        ("long") signal reading on a symbol/strategy currently held short
        covers it -- never gated by freshness, exits are never blocked."""
        pos = self.short_portfolio.positions.get(symbol)
        if pos is None or signal_direction != "long":
            return
        try:
            _, ask = self.md.best_bid_ask(symbol)
        except Exception:
            ask = prices.get(symbol, pos.entry_price)
        rec = self.short_portfolio.buy_to_close(symbol, best_ask=ask, fee_rate=FEE_RATE, slippage=SLIPPAGE,
                                                   ts_ms=now_ms(), exit_reason=f"{pos.strategy} exit signal")
        self._on_short_trade_closed(rec)

    def _on_trade_closed(self, rec: dict):
        pnl_pct = (rec["net_pnl"] / (rec["qty"] * rec["entry_price"])) * 100 if rec["qty"] * rec["entry_price"] else 0.0
        self.stats_store.record_trade(rec["symbol"], rec["strategy"], rec["regime"], pnl_pct)
        self.consecutive_losses = 0 if rec["net_pnl"] > 0 else self.consecutive_losses + 1
        self.exploration_symbols.discard(rec["symbol"])  # frees this exploration slot for a future cell
        self._write_trade_csv(rec)
        self._log_decision(rec["symbol"], "exited", rec)
        self.log(f"  <<< SELL {rec['symbol']} @ {rec['exit_price']:.4f} ({rec['exit_reason']}) "
                 f"net={rec['net_pnl']:+.2f}")

    def _enter_exploration(self, opp, equity: float, prices: Dict[str, float]):
        """Attempts one exploration entry for `opp`. Caller (run_once_cycle)
        is responsible for the eligibility filter, the MAX_EXPLORATION_POSITIONS
        cap, and iteration order -- this method just executes a single
        candidate and logs the outcome either way."""
        combined_heat, combined_exposure = self._combined_heat_and_exposure(prices)
        sizing = self.risk_engine.size_exploration_entry(
            equity=equity, cash=self.portfolio.cash,
            current_portfolio_heat_usdt=combined_heat,
            current_crypto_value_usdt=combined_exposure,
            stop_distance_frac=opp.stop_pct / 100.0, confidence=opp.confidence,
            signal_strength=opp.signal_strength)
        if not sizing.approved:
            self._log_decision(opp.symbol, "rejected_exploration_sizing",
                                {"reason": sizing.reason, "binding": sizing.binding_constraint})
            return
        try:
            _, ask = self.md.best_bid_ask(opp.symbol)
        except Exception as e:
            self._log_decision(opp.symbol, "rejected_exploration_price_fetch_failed", {"error": str(e)})
            return
        pos = self.portfolio.buy(
            opp.symbol, best_ask=ask, notional_usdt=sizing.notional_usdt,
            fee_rate=FEE_RATE, slippage=SLIPPAGE, ts_ms=now_ms(),
            strategy=opp.strategy, regime=opp.regime, confidence=opp.confidence,
            stop_price=ask * (1 - opp.stop_pct / 100.0), initial_stop_pct=opp.stop_pct,
            risk_amount_usdt=sizing.risk_amount_usdt)
        pos.trail_state = new_trail_state(pos.entry_price, opp.stop_pct)
        pos.stop_price = pos.trail_state["sl_price"]
        self.exploration_symbols.add(opp.symbol)
        self._log_decision(opp.symbol, "entered_exploration", {
            "strategy": opp.strategy, "champion_version": self._champion_version(opp.strategy),
            "regime": opp.regime, "confidence": opp.confidence,
            "signal_strength": opp.signal_strength, "score": opp.score,
            "notional": sizing.notional_usdt, "risk": sizing.risk_amount_usdt,
            "risk_state": self.risk_engine.state.value,
            "concurrent_exploration_positions": len(self.exploration_symbols),
        })
        self.log(f"  >>> BUY (EXPLORATION) {opp.symbol} {pos.qty:.6f} @ {pos.entry_price:.4f} "
                 f"({opp.strategy}/{opp.regime}) risk={sizing.risk_amount_usdt:.2f} -- first live "
                 f"observation for this (symbol,strategy,regime) cell "
                 f"[{len(self.exploration_symbols)}/{MAX_EXPLORATION_POSITIONS} exploration slots used]")

    # ── SIMULATED SHORT book: trade-closed bookkeeping + exploration entry,
    # exact mirrors of _on_trade_closed / _enter_exploration above ──────
    def _on_short_trade_closed(self, rec: dict):
        pnl_pct = (rec["net_pnl"] / (rec["qty"] * rec["entry_price"])) * 100 if rec["qty"] * rec["entry_price"] else 0.0
        self.short_stats_store.record_trade(rec["symbol"], rec["strategy"], rec["regime"], pnl_pct)
        self.short_consecutive_losses = 0 if rec["net_pnl"] > 0 else self.short_consecutive_losses + 1
        self.short_exploration_symbols.discard(rec["symbol"])  # frees this exploration slot for a future cell
        self._write_short_trade_csv(rec)
        self._log_short_decision(rec["symbol"], "short_exited", rec)
        self.log(f"  <<< COVER {rec['symbol']} @ {rec['exit_price']:.4f} ({rec['exit_reason']}) "
                 f"net={rec['net_pnl']:+.2f} [SHORT]")

    def _enter_short_exploration(self, opp, equity: float, prices: Dict[str, float]):
        """Attempts one SHORT exploration entry for `opp`. Caller
        (run_once_cycle) is responsible for the eligibility filter, the
        MAX_EXPLORATION_POSITIONS cap, and iteration order -- this method
        just executes a single candidate and logs the outcome either way."""
        combined_heat, combined_exposure = self._combined_heat_and_exposure(prices)
        sizing = self.short_risk_engine.size_exploration_entry(
            equity=equity, cash=self.short_portfolio.cash,
            current_portfolio_heat_usdt=combined_heat,
            current_crypto_value_usdt=combined_exposure,
            stop_distance_frac=opp.stop_pct / 100.0, confidence=opp.confidence,
            signal_strength=opp.signal_strength)
        if not sizing.approved:
            self._log_short_decision(opp.symbol, "rejected_short_exploration_sizing",
                                {"reason": sizing.reason, "binding": sizing.binding_constraint})
            return
        try:
            bid, _ = self.md.best_bid_ask(opp.symbol)
        except Exception as e:
            self._log_short_decision(opp.symbol, "rejected_short_exploration_price_fetch_failed", {"error": str(e)})
            return
        pos = self.short_portfolio.sell_to_open(
            opp.symbol, best_bid=bid, notional_usdt=sizing.notional_usdt,
            fee_rate=FEE_RATE, slippage=SLIPPAGE, ts_ms=now_ms(),
            strategy=opp.strategy, regime=opp.regime, confidence=opp.confidence,
            initial_stop_pct=opp.stop_pct, risk_amount_usdt=sizing.risk_amount_usdt)
        pos.trail_state = new_short_trail_state(pos.entry_price, opp.stop_pct)
        pos.stop_price = pos.trail_state["sl_price"]
        self.short_exploration_symbols.add(opp.symbol)
        self._log_short_decision(opp.symbol, "entered_short_exploration", {
            "strategy": opp.strategy, "champion_version": self._champion_version(opp.strategy),
            "regime": opp.regime, "confidence": opp.confidence,
            "signal_strength": opp.signal_strength, "score": opp.score,
            "notional": sizing.notional_usdt, "risk": sizing.risk_amount_usdt,
            "risk_state": self.short_risk_engine.state.value,
            "concurrent_exploration_positions": len(self.short_exploration_symbols),
        })
        self.log(f"  >>> SHORT (EXPLORATION) {opp.symbol} {pos.qty:.6f} @ {pos.entry_price:.4f} "
                 f"({opp.strategy}/{opp.regime}) risk={sizing.risk_amount_usdt:.2f} -- first live "
                 f"observation for this (symbol,strategy,regime) SHORT cell "
                 f"[{len(self.short_exploration_symbols)}/{MAX_EXPLORATION_POSITIONS} exploration slots used]")

    # ── bootstrap / restart ─────────────────────────────────────────
    def bootstrap(self):
        state = self._load_state()
        if state:
            self._restore(state)
        short_state = self._load_short_state()
        if short_state:
            self._restore_short(short_state)
        # Idempotent: creates v1 BASELINE champion records only if the
        # shared file doesn't exist yet (e.g. this process starts before
        # adaptive/research_service.py ever has). Never overwrites real
        # promotion history if the file already exists.
        try:
            self.champions = self.champion_store.ensure_baseline(DEFAULT_PARAMS)
            self.champion_load_failed = False
        except ChampionCorruptionError as e:
            self.log(f"!! CHAMPION FILE CORRUPTION at bootstrap: {e} -- starting with no champions "
                     f"(strategies fall back to documented defaults until this resolves)")
            self.champion_load_failed = True
        for sym in SYMBOLS:
            for tf in TIMEFRAMES:
                # A transient exchange error (RequestTimeout/NetworkError/
                # 5xx) on ANY one of these initial polls must NOT crash the
                # whole process -- run_forever()'s loop is wrapped in
                # try/except for exactly this reason, but bootstrap() runs
                # BEFORE that loop, so without this guard one flaky fetch
                # here kills the process before it ever reaches steady
                # state, and launchd's KeepAlive relaunches it straight
                # into the same failure -> a crash loop. This became likely
                # once the universe grew to 20 symbols (80 sequential
                # fetches at startup vs 20). A symbol that fails to warm up
                # here simply keeps cursor=None and cold-starts on its next
                # successful poll inside the normal cycle -- no data lost.
                try:
                    result = self.md.poll(sym, tf)
                except Exception as e:
                    self.log(f"  bootstrap poll {sym}/{tf} failed ({type(e).__name__}: "
                             f"{str(e)[:120]}) -- will warm up on a later cycle, continuing")
                    continue
                held = self.portfolio.held_qty(sym) > 0 or self.short_portfolio.held_qty(sym) > 0
                if result.gap_detected and tf == PRIMARY_STOP_TF and held:
                    self.reconciliation_failed.add(sym)
                    self.log(f"!! GAP INTEGRITY FAILURE {sym}/{tf}: {result.gap_detail} "
                             f"-- new entries blocked for {sym} until reconciled")
        self.log(f"bootstrap complete: positions={list(self.portfolio.positions.keys())} "
                 f"cash={self.portfolio.cash:.2f} "
                 f"short_positions={list(self.short_portfolio.positions.keys())} "
                 f"short_cash={self.short_portfolio.cash:.2f} reconciliation_failed={self.reconciliation_failed}")
        self._save_state()
        self._save_short_state()

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
        # migration: older state.json files (before multi-slot exploration)
        # stored a single "exploration_symbol" string/null instead of a list
        if "exploration_symbols" in state:
            self.exploration_symbols = set(state.get("exploration_symbols") or [])
        else:
            legacy = state.get("exploration_symbol")
            self.exploration_symbols = {legacy} if legacy else set()
        self.stats_store = StatsStore.from_dict(state.get("stats_store"))
        self.risk_engine.state = DegradationState(state.get("risk_state", "NORMAL"))
        # COMBINED (system-wide) display state -- see the combined_* fields'
        # __init__ comment. state.json is the sole authority for these
        # (short_state.json carries no equivalent keys).
        self.combined_peak_equity = state.get("combined_peak_equity", self.portfolio.cash)
        self.combined_daily_start_equity = state.get("combined_daily_start_equity", self.portfolio.cash)
        self.combined_trading_day = state.get("combined_trading_day")

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
            "exploration_symbols": sorted(self.exploration_symbols),
            "reconciliation_failed": sorted(self.reconciliation_failed),
            "stats_store": self.stats_store.to_dict(),
            "risk_state": self.risk_engine.state.value,
            "market_cursors": {f"{s}|{tf}": ts for (s, tf), ts in self.md.cursors.items()},
            "combined_peak_equity": self.combined_peak_equity,
            "combined_daily_start_equity": self.combined_daily_start_equity,
            "combined_trading_day": self.combined_trading_day,
        }
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2, default=str)
        os.replace(tmp, self.state_path)

    # ── SIMULATED SHORT book: state persistence, exact mirror of
    # _load_state / _restore / _save_state above, own file so the short
    # book's restart record can never be confused with or corrupt the
    # Spot long book's state.json ────────────────────────────────────
    def _load_short_state(self) -> Optional[dict]:
        if not os.path.exists(self.short_state_path):
            return None
        try:
            with open(self.short_state_path) as f:
                return json.load(f)
        except Exception:
            return None

    def _restore_short(self, state: dict):
        from adaptive.short_portfolio import ShortPosition
        p = state.get("portfolio", {})
        # Deliberately does NOT restore "cash" here -- it's the SAME
        # shared adaptive.portfolio.SharedCash balance the long book
        # already restored via _restore(state.json), which bootstrap()
        # calls first. short_state.json still WRITES its own "cash" value
        # below (_save_short_state) for human-debug visibility, but it
        # must never be read back as an independent source of truth: two
        # files independently restoring the same shared value would let a
        # stale short_state.json (e.g. from a crash that saved one file
        # but not the other) silently overwrite the correct long-restored
        # balance, depending on whatever order bootstrap() calls them in.
        self.short_portfolio.realized_pnl = p.get("realized_pnl", 0.0)
        self.short_portfolio.n_trades = p.get("n_trades", 0)
        self.short_portfolio.wins = p.get("wins", 0)
        self.short_portfolio.peak_equity = p.get("peak_equity", self.short_portfolio.cash)
        for sym, pd_ in (p.get("positions") or {}).items():
            # Built as a dict literal (colon, not `leverage=` kwarg syntax)
            # purely so this simulated dataclass reconstruction never
            # coincidentally matches test_adaptive_safety.py's
            # MARGIN_TRADING_TOKENS literal scan for a real ccxt
            # setLeverage(...) call -- there is none anywhere in this repo.
            fields = {
                "symbol": sym, "qty": pd_["qty"], "entry_price": pd_["entry_price"],
                "entry_ref_price": pd_["entry_ref_price"], "entry_ts_ms": pd_["entry_ts_ms"],
                "entry_fee": pd_["entry_fee"], "margin_reserved": pd_["margin_reserved"],
                "leverage": pd_["leverage"], "strategy": pd_["strategy"], "regime": pd_["regime"],
                "confidence": pd_["confidence"], "stop_price": pd_["stop_price"],
                "initial_stop_pct": pd_["initial_stop_pct"], "risk_amount_usdt": pd_["risk_amount_usdt"],
                "liquidation_price": pd_["liquidation_price"], "trail_state": pd_["trail_state"],
            }
            self.short_portfolio.positions[sym] = ShortPosition(**fields)
        self.short_daily_start_equity = state.get("daily_start_equity", self.short_portfolio.cash)
        self.short_trading_day = state.get("trading_day")
        self.short_consecutive_losses = state.get("consecutive_losses", 0)
        self.short_exploration_symbols = set(state.get("exploration_symbols") or [])
        self.short_stats_store = StatsStore.from_dict(state.get("stats_store"))
        self.short_risk_engine.state = DegradationState(state.get("risk_state", "NORMAL"))

    def _save_short_state(self):
        state = {
            "saved_at_iso": now_iso(),
            "portfolio": {
                "cash": self.short_portfolio.cash, "realized_pnl": self.short_portfolio.realized_pnl,
                "n_trades": self.short_portfolio.n_trades, "wins": self.short_portfolio.wins,
                "peak_equity": self.short_portfolio.peak_equity,
                "positions": {sym: {
                    "qty": pos.qty, "entry_price": pos.entry_price, "entry_ref_price": pos.entry_ref_price,
                    "entry_ts_ms": pos.entry_ts_ms, "entry_fee": pos.entry_fee,
                    "margin_reserved": pos.margin_reserved, "leverage": pos.leverage,
                    "strategy": pos.strategy, "regime": pos.regime, "confidence": pos.confidence,
                    "stop_price": pos.stop_price, "initial_stop_pct": pos.initial_stop_pct,
                    "risk_amount_usdt": pos.risk_amount_usdt, "liquidation_price": pos.liquidation_price,
                    "trail_state": pos.trail_state,
                } for sym, pos in self.short_portfolio.positions.items()},
            },
            "daily_start_equity": self.short_daily_start_equity, "trading_day": self.short_trading_day,
            "consecutive_losses": self.short_consecutive_losses,
            "exploration_symbols": sorted(self.short_exploration_symbols),
            "stats_store": self.short_stats_store.to_dict(),
            "risk_state": self.short_risk_engine.state.value,
        }
        tmp = self.short_state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2, default=str)
        os.replace(tmp, self.short_state_path)

    # ── champion state: READ-ONLY here, written only by research_service.py ──
    def _refresh_champions(self):
        try:
            self.champions = self.champion_store.read()
            if self.champion_load_failed:
                self.log("  champion file recovered -- resuming normal champion reads")
            self.champion_load_failed = False
        except ChampionCorruptionError as e:
            if not self.champion_load_failed:  # log once per failure onset, not every cycle
                self.log(f"!! CHAMPION FILE CORRUPTION: {e} -- retaining last known-good champions")
            self.champion_load_failed = True
            # self.champions intentionally left unchanged -- last verified copy stays in effect

    def _get_champion_params(self, strategy: str) -> dict:
        rec = self.champions.get(strategy)
        return rec.parameters if rec is not None else DEFAULT_PARAMS.get(strategy, {})

    def _champion_version(self, strategy: str) -> Optional[int]:
        rec = self.champions.get(strategy)
        return rec.version if rec is not None else None

    def snapshot(self, prices: Dict[str, float]) -> dict:
        """Section 20 dashboard/status: PORTFOLIO, POSITIONS, AI BRAIN, MARKET."""
        snap = self.portfolio.snapshot(prices)
        # cash/equity/return_pct/drawdown_pct/*_allocation_pct/daily_pnl_*/
        # portfolio_heat_* below are OVERWRITTEN with COMBINED (both books)
        # figures, replacing self.portfolio.snapshot()'s long-only versions
        # -- see _combined_equity()'s docstring for why: once cash is
        # shared, "the long book's own equity" is no longer a faithful
        # standalone number (it moves when the SHORT book spends shared
        # cash too), so showing it as THE top-level Portfolio figure would
        # misrepresent total system capital. `positions`/`realized_pnl`/
        # `n_trades`/`wins` below are correctly left as the long book's
        # OWN (unaffected by cash-sharing, genuinely long-specific).
        combined_equity = self._combined_equity(prices)
        combined_heat_usdt, combined_exposure_usdt = self._combined_heat_and_exposure(prices)
        combined_cash = self.cash_pool.balance
        daily_pnl = combined_equity - self.combined_daily_start_equity

        snap["cash"] = combined_cash
        snap["equity"] = combined_equity
        snap["peak_equity"] = self.combined_peak_equity
        snap["return_pct"] = (combined_equity / TOTAL_CAPITAL_USDT - 1) * 100 if TOTAL_CAPITAL_USDT else 0.0
        snap["drawdown_pct"] = ((self.combined_peak_equity - combined_equity) / self.combined_peak_equity * 100
                                 if self.combined_peak_equity else 0.0)
        snap["crypto_allocation_pct"] = (self.portfolio.crypto_value(prices) / combined_equity * 100
                                          if combined_equity else 0.0)
        snap["cash_allocation_pct"] = (combined_cash / combined_equity * 100) if combined_equity else 0.0
        snap["daily_pnl_usdt"] = daily_pnl
        snap["daily_pnl_pct"] = (daily_pnl / self.combined_daily_start_equity * 100
                                  if self.combined_daily_start_equity else 0.0)
        snap["portfolio_heat_usdt"] = combined_heat_usdt
        snap["portfolio_heat_pct"] = (combined_heat_usdt / combined_equity * 100) if combined_equity else 0.0
        # risk_state/consecutive_losses at the top level remain the LONG
        # book's OWN -- these are per-book degradation-state concepts
        # (deliberately NOT merged, see run_once_cycle), unaffected by
        # cash-sharing the way equity is; short's own values are still
        # shown under short_book below, unchanged.
        snap["risk_state"] = self.risk_engine.state.value
        snap["risk_state_reason"] = self.risk_engine.state_reason
        snap["consecutive_losses"] = self.consecutive_losses
        snap["reconciliation_failed"] = sorted(self.reconciliation_failed)

        snap["ai_brain"] = {
            "champions": {s: {"version": r.version, "status": r.status, "hash": r.hash,
                               "parameters": r.parameters, "created_at_iso": r.created_at_iso}
                          for s, r in self.champions.items()},
            "champion_file_load_failed": self.champion_load_failed,
            "strategies_active": list(DEFAULT_PARAMS.keys()) + ["shock_continuation"],
            "risk_state": self.risk_engine.state.value,
            "risk_state_reason": self.risk_engine.state_reason,
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

        # SIMULATED SHORT / MARGIN book -- PAPER ONLY. Deliberately its own
        # top-level key so open short positions, realized short PnL, and
        # short-specific risk state stay clearly separate from the long
        # book's own numbers above -- but cash/equity/return_pct/
        # drawdown_pct/portfolio_heat_* are DELIBERATELY DROPPED here (not
        # just "not shown," actively removed): ShortPortfolio.snapshot()
        # computes all of those from cash+exposure, and cash is now the
        # SAME shared balance the long book also draws from, so a
        # standalone "Short Book Equity" figure would swing based on the
        # LONG book's trades too -- see _combined_equity()'s docstring.
        # The combined figures at the top level (snap["equity"] etc. above)
        # are the one honest system-wide number; realized_pnl/n_trades/
        # wins/positions/leverage below remain genuinely short-specific
        # and are unaffected by cash-sharing, so they stay.
        short_snap = self.short_portfolio.snapshot(prices)
        for _misleading_key in ("cash", "equity", "start_capital", "return_pct", "peak_equity", "drawdown_pct"):
            short_snap.pop(_misleading_key, None)
        short_snap["risk_state"] = self.short_risk_engine.state.value
        short_snap["risk_state_reason"] = self.short_risk_engine.state_reason
        short_snap["consecutive_losses"] = self.short_consecutive_losses
        short_snap["leverage"] = self.short_portfolio.leverage
        snap["short_book"] = short_snap

        # All-time REALIZED P&L (banked on closed trades) -- the honest
        # scorecard, deliberately surfaced at the top level distinct from
        # equity/return_pct above (which include UNREALIZED marks on open
        # positions and so flatter/mislead). Combined = long book + short
        # book, both genuinely realized and unaffected by the shared-cash
        # display caveats. realized_return_pct is measured against the fixed
        # starting capital so it is comparable over time regardless of
        # current open exposure.
        rl = self.portfolio.realized_pnl
        rs = self.short_portfolio.realized_pnl
        tl, ts = self.portfolio.n_trades, self.short_portfolio.n_trades
        wl, ws = self.portfolio.wins, self.short_portfolio.wins
        snap["realized_pnl_long"] = rl
        snap["realized_pnl_short"] = rs
        snap["realized_pnl_combined"] = rl + rs
        snap["realized_return_pct"] = (rl + rs) / TOTAL_CAPITAL_USDT * 100 if TOTAL_CAPITAL_USDT else 0.0
        snap["n_trades_combined"] = tl + ts
        snap["wins_combined"] = wl + ws
        snap["win_rate_pct"] = (100.0 * (wl + ws) / (tl + ts)) if (tl + ts) else 0.0
        return snap

    def write_dashboard(self, prices: Dict[str, float]):
        path = os.path.join(self.runtime_dir, "dashboard.json")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"written_at_iso": now_iso(), **self.snapshot(prices)}, f, indent=2, default=str)
        os.replace(tmp, path)

    # ── one decision cycle ───────────────────────────────────────────
    def run_once_cycle(self):
        self._refresh_champions()
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
            self._manage_short_position(sym, prices)
        for sym in SYMBOLS:
            if sym not in prices:
                df = self.md.candles[(sym, PRIMARY_STOP_TF)]
                if len(df):
                    prices[sym] = float(df.iloc[-1]["close"])

        if len(prices) < len(SYMBOLS):
            self._save_state()
            self._save_short_state()
            self.write_dashboard(prices)
            return  # not enough data yet to safely evaluate entries this cycle

        equity = self.portfolio.update_peak_equity(prices)
        self._roll_day_if_needed(equity)
        short_equity = self.short_portfolio.update_peak_equity(prices)
        self._roll_short_day_if_needed(short_equity)
        combined_equity = self._combined_equity(prices)
        self.combined_peak_equity = max(self.combined_peak_equity, combined_equity)
        self._roll_combined_day_if_needed(combined_equity)

        # Degradation/breaker checks for BOTH books use COMBINED equity/
        # peak/daily-start, NOT each book's own equity() -- once cash is
        # shared, a book's own equity() moves whenever the OTHER book
        # merely reserves capital (margin, inventory) with zero realized
        # loss, which would otherwise trip that book's daily-loss breaker
        # against its own frozen daily_start_equity for no real reason
        # (confirmed live: a short-side exploration entry alone produced
        # a false "daily loss 3.77%" on the long book). Combined equity is
        # invariant to capital *reservation* and only moves on genuine P&L
        # (realized or unrealized) on either side, so it is the honest
        # health figure for both breakers. consecutive_losses/loss_streak
        # stay per-book on purpose (see run_once_cycle's docstring above) --
        # only the equity/peak/daily-start INPUTS are shared.
        drawdown_pct = ((self.combined_peak_equity - combined_equity) / self.combined_peak_equity * 100
                         if self.combined_peak_equity else 0.0)
        recent_pf, recent_exp = self._recent_aggregate_stats()
        new_state = self.risk_engine.evaluate_degradation(
            drawdown_pct=drawdown_pct, loss_streak=self.consecutive_losses,
            recent_expectancy=recent_exp, recent_pf=recent_pf)
        if new_state != self.risk_engine.state:
            self.log(f"  risk state {self.risk_engine.state.value} -> {new_state.value}")
            self.risk_engine.set_state(new_state, "degradation evaluation")

        breaker_reason = self.risk_engine.check_breakers(
            equity=combined_equity, peak_equity=self.combined_peak_equity,
            daily_start_equity=self.combined_daily_start_equity, consecutive_losses=self.consecutive_losses)

        # SIMULATED SHORT book: own degradation STATE + own breaker
        # threshold config, entirely independent of the long book's above
        # (a bad run on one side must never halt or throttle the other) --
        # but fed the SAME combined equity/peak/daily-start inputs as the
        # long book just above, for the identical cash-sharing reason.
        short_drawdown_pct = drawdown_pct
        short_recent_pf, short_recent_exp = self._recent_aggregate_stats_short()
        new_short_state = self.short_risk_engine.evaluate_degradation(
            drawdown_pct=short_drawdown_pct, loss_streak=self.short_consecutive_losses,
            recent_expectancy=short_recent_exp, recent_pf=short_recent_pf)
        if new_short_state != self.short_risk_engine.state:
            self.log(f"  short risk state {self.short_risk_engine.state.value} -> {new_short_state.value}")
            self.short_risk_engine.set_state(new_short_state, "degradation evaluation")

        short_breaker_reason = self.short_risk_engine.check_breakers(
            equity=combined_equity, peak_equity=self.combined_peak_equity,
            daily_start_equity=self.combined_daily_start_equity, consecutive_losses=self.short_consecutive_losses)

        candidates = []
        short_candidates = []
        corr_matrix = self._correlation_matrix()

        # Cross-sectional relative-strength signals: computed ONCE here over
        # the whole universe (not per-symbol), then injected into each
        # symbol's signals dict below. Uses 1h closes; entries are still
        # gated per-symbol by fresh_1h in the loop, exactly like the other
        # 1h strategy (trend_momentum), so no entry ever fires off a stale
        # bar. feats give it ATR-scaled stops.
        xsect_closes, xsect_feats = {}, {}
        for _s in SYMBOLS:
            _df1h = self.md.candles[(_s, "1h")]
            if len(_df1h) > 0:
                xsect_closes[_s] = _df1h["close"].values
                _f = compute_features(_df1h)
                if _f is not None:
                    xsect_feats[_s] = _f
        xsect_signals = cross_sectional_signals(
            xsect_closes, xsect_feats, self._get_champion_params("cross_sectional"))

        for sym in SYMBOLS:
            if self.portfolio.held_qty(sym) > 0 or sym in self.reconciliation_failed:
                continue
            ok, reasons = self.md.market_quality(sym, primary_tf=PRIMARY_STOP_TF)
            if not ok:
                self._log_decision(sym, "skipped_market_quality", {"reasons": reasons})
                continue
            df_1h = self.md.candles[(sym, "1h")]
            df_15m = self.md.candles[(sym, "15m")]
            df_4h = self.md.candles[(sym, "4h")]
            if len(df_1h) == 0 or len(df_15m) == 0:
                continue
            regime = classify_from_df(df_1h)
            fresh_1h = fresh.get((sym, "1h"), False)
            fresh_15m = fresh.get((sym, "15m"), False)
            fresh_4h = fresh.get((sym, "4h"), False)
            signals = {
                "trend_momentum": (trend_momentum(df_1h, self._get_champion_params("trend_momentum")), fresh_1h),
                "volatility_breakout": (volatility_breakout(
                    df_15m, self._get_champion_params("volatility_breakout")), fresh_15m),
                "mean_reversion": (mean_reversion(
                    df_15m, self._get_champion_params("mean_reversion")), fresh_15m),
            }
            if len(df_4h) > 0:
                signals["atr_trailing_stop"] = (atr_trailing_stop(
                    df_4h, self._get_champion_params("atr_trailing_stop")), fresh_4h)
            if sym in xsect_signals:
                # 1h-based like trend_momentum -> gated by fresh_1h; a
                # neutral (direction=None) entry is simply ignored by the
                # branch dispatch below, so injecting it unconditionally
                # (when present) is safe.
                signals["cross_sectional"] = (xsect_signals[sym], fresh_1h)
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
                    # Mirror of the exit_long branch below: a bullish
                    # reading also covers any existing SHORT position in
                    # this symbol -- never gated by freshness, exits are
                    # never blocked.
                    self._exit_short_signal_management(sym, "long", prices)
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
                    # A bearish reading is ALSO a candidate SHORT entry --
                    # gated by freshness the same as long entries above
                    # (never fabricate a short from a stale/historical bar),
                    # and only when we don't already hold a short here.
                    if is_fresh and self.short_portfolio.held_qty(sym) == 0 and sym not in self.reconciliation_failed:
                        fpen = friction_pct(FEE_RATE, SLIPPAGE, TYPICAL_SPREAD_PCT)
                        short_opp = score_short_opportunity(
                            symbol=sym, signal=sig, regime=regime, stats_store=self.short_stats_store,
                            friction_penalty_pct=fpen, already_selected=[], correlation_matrix=corr_matrix)
                        if short_opp is not None:
                            short_candidates.append(short_opp)

        selected = rank_and_select(candidates, corr_matrix, max_positions=MAX_POSITIONS, direction="long")
        selected_shorts = rank_and_select(short_candidates, corr_matrix, max_positions=MAX_POSITIONS, direction="short")
        self._log_decision("*", "cycle_ranking", {
            "candidates": [_opp_record(o) for o in candidates],
            "selected": [o.symbol for o in selected], "risk_state": self.risk_engine.state.value,
            "short_candidates": [_opp_record(o) for o in short_candidates],
            "selected_shorts": [o.symbol for o in selected_shorts],
            "short_risk_state": self.short_risk_engine.state.value,
        })

        if breaker_reason:
            self._log_decision("*", "entries_blocked_portfolio_breaker", {"reason": breaker_reason})
        else:
            for opp in selected:
                if opp.symbol in self.reconciliation_failed:
                    continue
                # A symbol can legitimately produce multiple long candidates
                # in one cycle (e.g. trend_momentum AND cross_sectional both
                # fire on a strong riser), and rank_and_select does NOT dedup
                # by symbol -- so an earlier iteration this same loop may have
                # already opened this symbol. Portfolio.buy() would then raise
                # "already held" and abort the whole cycle. Skip it here (the
                # first/highest-scored candidate already took the position).
                if self.portfolio.held_qty(opp.symbol) > 0:
                    continue
                # recomputed EVERY iteration (not hoisted above the loop):
                # an earlier iteration this same loop may have just opened
                # a position, and the combined figures must reflect that
                # before sizing the next one.
                combined_heat, combined_exposure = self._combined_heat_and_exposure(prices)
                sizing = self.risk_engine.size_entry(
                    equity=equity, cash=self.portfolio.cash,
                    current_portfolio_heat_usdt=combined_heat,
                    current_crypto_value_usdt=combined_exposure,
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
                    "strategy": opp.strategy, "champion_version": self._champion_version(opp.strategy),
                    "regime": opp.regime, "confidence": opp.confidence, "score": opp.score,
                    "shrunk_edge": opp.shrunk_edge, "notional": sizing.notional_usdt,
                    "risk": sizing.risk_amount_usdt, "portfolio_heat_usdt": sum(
                        p.risk_amount_usdt for p in self.portfolio.positions.values()),
                    "risk_state": self.risk_engine.state.value,
                    "correlation_penalty": opp.correlation_penalty, "fees_slippage_est": opp.friction_penalty,
                })
                self.log(f"  >>> BUY {opp.symbol} {pos.qty:.6f} @ {pos.entry_price:.4f} "
                         f"({opp.strategy}/{opp.regime}) risk={sizing.risk_amount_usdt:.2f}")

            # Paper exploration (cold-start-deadlock fix, see
            # meta_controller.MIN_EXPLORATION_SIGNAL_STRENGTH and
            # risk_engine.EXPLORATION_RISK_PER_TRADE_PCT): tightly
            # risk-capped positions for (symbol, strategy, regime) cells
            # with zero live trades ever, so empirical-Bayes shrinkage has
            # a first real observation to update from. Normal scoring can
            # never produce a positive score for such a cell (shrunk_edge
            # is pinned to exactly the global prior at n=0 regardless of raw
            # signal strength), which would otherwise block it forever --
            # confirmed against real production history where 6 genuine
            # mean_reversion BUY signals across BTC/ETH/SOL all scored
            # identically below MIN_QUALITY_SCORE. Gated by the SAME
            # breaker_reason check as normal entries (never bypasses basic
            # risk management) and by MAX_EXPLORATION_POSITIONS system-wide
            # (multiple symbols may explore in parallel -- one per symbol,
            # since Portfolio.buy() still forbids two positions in the same
            # symbol regardless of origin).
            selected_symbols = {o.symbol for o in selected}
            explorable = sorted(
                (o for o in candidates if o.is_exploration_eligible and o.symbol not in selected_symbols
                 and o.symbol not in self.reconciliation_failed and o.symbol not in self.exploration_symbols
                 and o.symbol not in self.portfolio.positions),
                key=lambda o: o.signal_strength, reverse=True)
            for cand in explorable:
                if len(self.exploration_symbols) >= MAX_EXPLORATION_POSITIONS:
                    break
                # `explorable` is built ONCE above, so it can contain two
                # candidates for the same symbol (different strategies). Once
                # the first opens the position, re-check here so the second
                # doesn't hit buy()'s "already held" and abort the cycle.
                if cand.symbol in self.portfolio.positions or cand.symbol in self.exploration_symbols:
                    continue
                self._enter_exploration(cand, equity, prices)

        # SIMULATED SHORT book: own entry + exploration loop, gated by its
        # OWN breaker_reason (short_breaker_reason) -- entirely independent
        # of the long book's `if breaker_reason:` block above, exact mirror
        # of its structure and logic, sell_to_open() instead of buy().
        if short_breaker_reason:
            self._log_short_decision("*", "short_entries_blocked_portfolio_breaker", {"reason": short_breaker_reason})
        else:
            for opp in selected_shorts:
                if opp.symbol in self.reconciliation_failed:
                    continue
                # same within-cycle dedup as the long loop -- sell_to_open()
                # forbids two shorts in one symbol, so skip if an earlier
                # candidate this cycle already shorted it.
                if self.short_portfolio.held_qty(opp.symbol) > 0:
                    continue
                # recomputed EVERY iteration -- see the matching comment in
                # the long entry loop above.
                combined_heat, combined_exposure = self._combined_heat_and_exposure(prices)
                sizing = self.short_risk_engine.size_entry(
                    equity=short_equity, cash=self.short_portfolio.cash,
                    current_portfolio_heat_usdt=combined_heat,
                    current_crypto_value_usdt=combined_exposure,
                    stop_distance_frac=opp.stop_pct / 100.0, confidence=opp.confidence)
                if not sizing.approved:
                    self._log_short_decision(opp.symbol, "rejected_short_sizing",
                                        {"reason": sizing.reason, "binding": sizing.binding_constraint})
                    continue
                try:
                    bid, ask = self.md.best_bid_ask(opp.symbol)
                except Exception as e:
                    self._log_short_decision(opp.symbol, "rejected_short_price_fetch_failed", {"error": str(e)})
                    continue
                pos = self.short_portfolio.sell_to_open(
                    opp.symbol, best_bid=bid, notional_usdt=sizing.notional_usdt,
                    fee_rate=FEE_RATE, slippage=SLIPPAGE, ts_ms=now_ms(),
                    strategy=opp.strategy, regime=opp.regime, confidence=opp.confidence,
                    initial_stop_pct=opp.stop_pct, risk_amount_usdt=sizing.risk_amount_usdt)
                pos.trail_state = new_short_trail_state(pos.entry_price, opp.stop_pct)
                pos.stop_price = pos.trail_state["sl_price"]
                self._log_short_decision(opp.symbol, "entered_short", {
                    "strategy": opp.strategy, "champion_version": self._champion_version(opp.strategy),
                    "regime": opp.regime, "confidence": opp.confidence, "score": opp.score,
                    "shrunk_edge": opp.shrunk_edge, "notional": sizing.notional_usdt,
                    "risk": sizing.risk_amount_usdt, "portfolio_heat_usdt": sum(
                        p.risk_amount_usdt for p in self.short_portfolio.positions.values()),
                    "risk_state": self.short_risk_engine.state.value,
                    "correlation_penalty": opp.correlation_penalty, "fees_slippage_est": opp.friction_penalty,
                    "leverage": pos.leverage, "liquidation_price": pos.liquidation_price,
                })
                self.log(f"  >>> SHORT {opp.symbol} {pos.qty:.6f} @ {pos.entry_price:.4f} "
                         f"({opp.strategy}/{opp.regime}) risk={sizing.risk_amount_usdt:.2f} lev={pos.leverage:.1f}x")

            selected_short_symbols = {o.symbol for o in selected_shorts}
            short_explorable = sorted(
                (o for o in short_candidates if o.is_exploration_eligible and o.symbol not in selected_short_symbols
                 and o.symbol not in self.reconciliation_failed and o.symbol not in self.short_exploration_symbols
                 and o.symbol not in self.short_portfolio.positions),
                key=lambda o: o.signal_strength, reverse=True)
            for cand in short_explorable:
                if len(self.short_exploration_symbols) >= MAX_EXPLORATION_POSITIONS:
                    break
                # same within-cycle dedup as the long exploration loop above.
                if cand.symbol in self.short_portfolio.positions or cand.symbol in self.short_exploration_symbols:
                    continue
                self._enter_short_exploration(cand, short_equity, prices)

        self._save_state()
        self._save_short_state()
        self.write_dashboard(prices)

    def _combined_heat_and_exposure(self, prices: Dict[str, float]):
        """Portfolio-heat and crypto/short-exposure dollar amounts, SUMMED
        across BOTH books. Required because both books' risk_engine
        instances now size against the SAME shared cash pool (see
        SharedCash) -- if each book's size_entry()/size_exploration_entry()
        call only saw its OWN heat/exposure, the max_portfolio_heat_pct
        (1.50%) and max_total_crypto_allocation_pct (90%) caps would each
        be independently checked against roughly the same equity, letting
        long AND short each separately consume a FULL 1.50%/90% budget --
        effectively 3% heat / 180% exposure of one pool, doubling the
        intended ceiling. Summing here closes that hole: whichever book's
        entries run first in this cycle (long, then short) correctly
        shrinks the remaining room the other one sees, exactly mirroring
        how the shared cash balance itself already behaves."""
        long_heat = sum(p.risk_amount_usdt for p in self.portfolio.positions.values())
        short_heat = sum(p.risk_amount_usdt for p in self.short_portfolio.positions.values())
        long_exposure = self.portfolio.crypto_value(prices)
        short_exposure = self.short_portfolio.exposure_value(prices)
        return long_heat + short_heat, long_exposure + short_exposure

    def _combined_equity(self, prices: Dict[str, float]) -> float:
        """The ONE true system-wide equity figure: shared cash (counted
        ONCE, not once per book) plus long Spot inventory value plus the
        short book's reserved margin (collateral, not spent) plus its
        unrealized P&L. Building this from parts here -- rather than
        naively summing self.portfolio.equity(prices) +
        self.short_portfolio.equity(prices) -- avoids double-counting the
        shared cash, which both of those methods independently include."""
        long_crypto_value = self.portfolio.crypto_value(prices)
        short_margin_held = sum(p.margin_reserved for p in self.short_portfolio.positions.values())
        short_unrealized = sum(
            self.short_portfolio.unrealized_pnl(sym, prices.get(sym, p.entry_price))
            for sym, p in self.short_portfolio.positions.items())
        return self.cash_pool.balance + long_crypto_value + short_margin_held + short_unrealized

    def _recent_aggregate_stats(self):
        all_trades = [t for c in self.stats_store.cells.values() for t in c.trades[-20:]]
        if len(all_trades) < 10:
            return None, None
        wins = sum(t for t in all_trades if t > 0)
        losses = sum(abs(t) for t in all_trades if t <= 0)
        pf = (wins / losses) if losses > 0 else (float("inf") if wins > 0 else None)
        exp = sum(all_trades) / len(all_trades)
        return pf, exp

    def _recent_aggregate_stats_short(self):
        all_trades = [t for c in self.short_stats_store.cells.values() for t in c.trades[-20:]]
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
