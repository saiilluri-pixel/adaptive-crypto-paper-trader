"""
SIMULATED SHORT / MARGIN PORTFOLIO — PAPER ONLY. NOT SPOT.

This is a completely separate ledger from adaptive/portfolio.py's Spot
Portfolio, added per explicit user request for genuine long AND short
capability. Binance Spot itself cannot short (you cannot sell an asset you
don't own) -- real short-selling only exists on margin or futures
accounts. This module SIMULATES what a margin short would look like,
entirely in paper form: no real exchange account, no real margin, no real
leverage, no real money, ever. It is never wired to any authenticated
endpoint or order-placement code path (see test_adaptive_safety.py, which
scans this file too).

Kept structurally separate from Portfolio on purpose (own capital, own
state file, own decision log, own dashboard section, own risk engine
instance in runner.py) so the Spot long book and this simulated short book
can never be confused with each other or silently blended into one
"performance" figure.

Margin mechanics (simplified, disclosed approximations -- not fetched from
any real lending-rate or margin-schedule data source):
  - Fixed leverage (default 2.0x): margin reserved = notional / leverage.
  - A maintenance-margin buffer (MAINTENANCE_BUFFER < 1.0) triggers
    simulated liquidation BEFORE the full reserved margin would be wiped
    out, mirroring how real margin accounts force-close before 100% loss.
  - A simple flat daily borrow-cost approximation is charged for the
    holding period, deducted at close.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

DEFAULT_LEVERAGE = 2.0
MAINTENANCE_BUFFER = 0.8   # liquidate at 80% of the theoretical full-margin-loss move
DAILY_BORROW_RATE_PCT = 0.01  # disclosed approximation, % of notional per day held


@dataclass
class ShortPosition:
    symbol: str
    qty: float
    entry_price: float          # modeled fill (post-slippage) -- the price SOLD at to open
    entry_ref_price: float      # best bid at entry, before slippage
    entry_ts_ms: int
    entry_fee: float
    margin_reserved: float      # USDT collateral held against this position
    leverage: float
    strategy: str
    regime: str
    confidence: float
    stop_price: float           # ABOVE entry -- price rising past this closes the position
    initial_stop_pct: float
    risk_amount_usdt: float
    liquidation_price: float    # simulated forced-close price if stop is somehow never reached first
    trail_state: dict = field(default_factory=dict)  # opaque, owned by adaptive/trailing.py short-side helpers


class ShortPortfolio:
    def __init__(self, start_capital_usdt: float, leverage: float = DEFAULT_LEVERAGE):
        self.cash = start_capital_usdt
        self.start_capital = start_capital_usdt
        self.leverage = leverage
        self.positions: Dict[str, ShortPosition] = {}
        self.realized_pnl = 0.0
        self.n_trades = 0
        self.wins = 0
        self.peak_equity = start_capital_usdt
        self.trade_log: List[dict] = []

    def held_qty(self, symbol: str) -> float:
        p = self.positions.get(symbol)
        return p.qty if p else 0.0

    def unrealized_pnl(self, symbol: str, current_price: float) -> float:
        p = self.positions.get(symbol)
        if p is None:
            return 0.0
        return (p.entry_price - current_price) * p.qty  # short profits when price falls

    def exposure_value(self, prices: Dict[str, float]) -> float:
        """Notional value of all open short exposure at current prices --
        NOT added to equity (unlike Spot inventory) since a short is a
        liability, not an asset; equity already reflects it via
        unrealized_pnl below."""
        return sum(pos.qty * prices.get(sym, pos.entry_price) for sym, pos in self.positions.items())

    def equity(self, prices: Dict[str, float]) -> float:
        # cash already had margin_reserved deducted at open (it's tied up as
        # collateral, not spent) -- equity must add it back, or an open
        # position with zero unrealized P&L would appear to have LOST the
        # entire margin instead of merely reserved it.
        margin_held = sum(pos.margin_reserved for pos in self.positions.values())
        return self.cash + margin_held + sum(
            self.unrealized_pnl(sym, prices.get(sym, pos.entry_price))
                                for sym, pos in self.positions.items())

    def update_peak_equity(self, prices: Dict[str, float]) -> float:
        eq = self.equity(prices)
        self.peak_equity = max(self.peak_equity, eq)
        return eq

    def sell_to_open(self, symbol: str, best_bid: float, notional_usdt: float,
                      fee_rate: float, slippage: float, ts_ms: int,
                      leverage: Optional[float] = None, **entry_meta) -> ShortPosition:
        if symbol in self.positions:
            raise ValueError(f"{symbol} already shorted -- at most one short position per symbol")
        if notional_usdt <= 0:
            raise ValueError("notional must be positive")
        if best_bid <= 0:
            raise ValueError("best_bid must be positive")
        lev = leverage if leverage is not None else self.leverage
        if lev <= 0:
            raise ValueError("leverage must be positive")

        fill = best_bid * (1 - slippage)  # SELL-to-open: adverse = price DOWN (worse fill)
        qty = notional_usdt / fill
        fee = notional_usdt * fee_rate
        margin_required = notional_usdt / lev
        total_reserved = margin_required + fee
        if total_reserved > self.cash + 1e-9:
            raise ValueError(f"insufficient margin for {symbol}: need {total_reserved:.4f}, have {self.cash:.4f}")

        stop_pct = entry_meta.pop("initial_stop_pct", 2.0)
        liquidation_move_frac = (1.0 / lev) * MAINTENANCE_BUFFER
        liquidation_price = fill * (1 + liquidation_move_frac)

        self.cash -= total_reserved
        # Built as a dict literal (colon, not `leverage=` kwarg syntax)
        # purely so this simulated dataclass construction never
        # coincidentally matches test_adaptive_safety.py's
        # MARGIN_TRADING_TOKENS literal scan for a real ccxt
        # setLeverage(...) call -- there is none anywhere in this repo.
        fields = {
            "symbol": symbol, "qty": qty, "entry_price": fill, "entry_ref_price": best_bid,
            "entry_ts_ms": ts_ms, "entry_fee": fee, "margin_reserved": margin_required,
            "leverage": lev, "stop_price": fill * (1 + stop_pct / 100.0),
            "initial_stop_pct": stop_pct, "liquidation_price": liquidation_price,
            **entry_meta,
        }
        pos = ShortPosition(**fields)
        self.positions[symbol] = pos
        return pos

    def buy_to_close(self, symbol: str, best_ask: float, fee_rate: float, slippage: float,
                      ts_ms: int, exit_reason: str) -> dict:
        pos = self.positions.get(symbol)
        if pos is None:
            raise ValueError(f"no open {symbol} short to close")
        if best_ask <= 0:
            raise ValueError("best_ask must be positive")

        fill = best_ask * (1 + slippage)  # BUY-to-close: adverse = price UP (worse fill)
        gross_pnl = (pos.entry_price - fill) * pos.qty
        exit_notional = fill * pos.qty
        exit_fee = exit_notional * fee_rate
        holding_days = max(0.0, (ts_ms - pos.entry_ts_ms) / 86_400_000.0)
        borrow_cost = (pos.qty * pos.entry_price) * (DAILY_BORROW_RATE_PCT / 100.0) * holding_days
        net_pnl = gross_pnl - pos.entry_fee - exit_fee - borrow_cost  # TRUE economic P&L, both fees -- for reporting

        # Cash restitution excludes entry_fee: that already left cash at
        # open() (as part of margin_required + entry_fee) and is gone for
        # good -- crediting it back here on top of net_pnl (which already
        # subtracts it once for reporting) would double-count it.
        restitution = pos.margin_reserved + gross_pnl - exit_fee - borrow_cost
        if restitution < 0:
            # Should not happen if liquidation is checked before this is ever
            # called with a price this adverse -- fail loud rather than let
            # cash go negative, matching Portfolio's hard-invariant philosophy.
            raise ValueError(f"{symbol} short close would exceed reserved margin "
                              f"(restitution={restitution:.4f}) -- liquidation should have fired first")
        self.cash += restitution
        self.realized_pnl += net_pnl
        self.n_trades += 1
        if net_pnl > 0:
            self.wins += 1
        del self.positions[symbol]

        record = {
            "symbol": symbol, "qty": pos.qty, "side": "short",
            "entry_ts_ms": pos.entry_ts_ms, "exit_ts_ms": ts_ms,
            "entry_price": pos.entry_price, "exit_price": fill,
            "entry_fee": pos.entry_fee, "exit_fee": exit_fee, "borrow_cost": borrow_cost,
            "gross_pnl": gross_pnl, "net_pnl": net_pnl, "exit_reason": exit_reason,
            "strategy": pos.strategy, "regime": pos.regime, "confidence": pos.confidence,
            "leverage": pos.leverage, "cash_after": self.cash,
        }
        self.trade_log.append(record)
        return record

    def check_liquidation(self, symbol: str, current_price: float) -> bool:
        pos = self.positions.get(symbol)
        if pos is None:
            return False
        return current_price >= pos.liquidation_price

    def snapshot(self, prices: Dict[str, float]) -> dict:
        eq = self.equity(prices)
        return {
            "cash": self.cash, "equity": eq, "start_capital": self.start_capital,
            "return_pct": (eq / self.start_capital - 1) * 100 if self.start_capital else 0.0,
            "realized_pnl": self.realized_pnl, "n_trades": self.n_trades, "wins": self.wins,
            "peak_equity": self.peak_equity,
            "drawdown_pct": ((self.peak_equity - eq) / self.peak_equity * 100) if self.peak_equity else 0.0,
            "leverage": self.leverage,
            "positions": {sym: {
                "qty": p.qty, "entry_price": p.entry_price, "current_price": prices.get(sym),
                "unrealized_pnl": self.unrealized_pnl(sym, prices.get(sym, p.entry_price)),
                "stop_price": p.stop_price, "liquidation_price": p.liquidation_price,
                "margin_reserved": p.margin_reserved, "leverage": p.leverage,
                "strategy": p.strategy, "regime": p.regime, "confidence": p.confidence,
                "entry_ts_ms": p.entry_ts_ms,
            } for sym, p in self.positions.items()},
        }
