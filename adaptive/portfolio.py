"""
Single shared Spot paper portfolio: USDT cash + BTC/ETH/SOL inventory.

Spot-legal only -- no shorting, no leverage, no margin. At most one open
position per symbol (a position is always closed in full on sell(); no
partial position scaling, matching the existing PaperEngine convention
elsewhere in this codebase). Cash can never go negative and a sell can
never exceed owned quantity -- both are hard invariants enforced by
raising, not silently rejecting: the risk engine is the policy layer that
decides whether an action should be attempted at all; this module is the
mechanism layer that refuses to violate Spot reality if asked to.

Fill convention (audited against real Binance bid/ask spread in
shadow/shock_continuation_sol_v3_1/diagnostics/):
  BUY  reference = best ASK, adverse slippage applied ON TOP (price moves up)
  SELL reference = best BID, adverse slippage applied ON TOP (price moves down)
This is stricter than applying slippage to a naive mid/close price -- the
live audit found modeled slippage (0.02%) already exceeds the typical
observed spread (~0.0096%) even against mid, so bid/ask-then-slippage is
more conservative still.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Unused within this module (Portfolio is symbol-agnostic) -- kept only as
# documentation, mirrored from adaptive/market_data.py's SYMBOLS, the
# actual source of truth for the live trading universe.
SYMBOLS = ("BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT")


@dataclass
class Position:
    symbol: str
    qty: float
    entry_price: float          # modeled fill price (post-slippage), NOT the reference price
    entry_ref_price: float      # best ask at entry, before slippage
    entry_ts_ms: int
    entry_fee: float
    strategy: str
    regime: str
    confidence: float
    stop_price: float
    initial_stop_pct: float
    risk_amount_usdt: float
    # Entry-time contract (per adaptive spec section 14): a promoted
    # champion must NEVER reach into an open position and change these.
    # trail_state is opaque here, owned/mutated only by adaptive/trailing.py.
    trail_state: dict = field(default_factory=dict)


class Portfolio:
    def __init__(self, start_capital_usdt: float):
        self.cash = start_capital_usdt
        self.start_capital = start_capital_usdt
        self.positions: Dict[str, Position] = {}
        self.realized_pnl = 0.0
        self.n_trades = 0
        self.wins = 0
        self.peak_equity = start_capital_usdt
        self.trade_log: List[dict] = []

    def held_qty(self, symbol: str) -> float:
        p = self.positions.get(symbol)
        return p.qty if p else 0.0

    def crypto_value(self, prices: Dict[str, float]) -> float:
        return sum(pos.qty * prices[sym] for sym, pos in self.positions.items())

    def equity(self, prices: Dict[str, float]) -> float:
        return self.cash + self.crypto_value(prices)

    def update_peak_equity(self, prices: Dict[str, float]) -> float:
        eq = self.equity(prices)
        self.peak_equity = max(self.peak_equity, eq)
        return eq

    def can_afford(self, notional_usdt: float) -> bool:
        return notional_usdt > 0 and notional_usdt <= self.cash

    def buy(self, symbol: str, best_ask: float, notional_usdt: float,
            fee_rate: float, slippage: float, ts_ms: int, **entry_meta) -> Position:
        if symbol in self.positions:
            raise ValueError(f"{symbol} already held -- Spot allows at most one position per symbol")
        if notional_usdt <= 0:
            raise ValueError("notional must be positive")
        if best_ask <= 0:
            raise ValueError("best_ask must be positive")
        fill = best_ask * (1 + slippage)  # BUY: adverse = price UP
        fee = notional_usdt * fee_rate
        total_cost = notional_usdt + fee
        if total_cost > self.cash + 1e-9:
            raise ValueError(f"insufficient cash for {symbol}: need {total_cost:.4f}, have {self.cash:.4f}")
        qty = notional_usdt / fill
        self.cash -= total_cost
        pos = Position(symbol=symbol, qty=qty, entry_price=fill, entry_ref_price=best_ask,
                       entry_ts_ms=ts_ms, entry_fee=fee, **entry_meta)
        self.positions[symbol] = pos
        return pos

    def sell(self, symbol: str, best_bid: float, fee_rate: float, slippage: float,
             ts_ms: int, exit_reason: str) -> dict:
        pos = self.positions.get(symbol)
        if pos is None:
            raise ValueError(f"no open {symbol} position to sell")
        if best_bid <= 0:
            raise ValueError("best_bid must be positive")
        qty = pos.qty  # always closes the FULL position -- see module docstring
        fill = best_bid * (1 - slippage)  # SELL: adverse = price DOWN
        gross_proceeds = qty * fill
        exit_fee = gross_proceeds * fee_rate
        net_proceeds = gross_proceeds - exit_fee
        if net_proceeds < 0:
            raise ValueError(f"sell of {symbol} would make cash negative: net_proceeds={net_proceeds:.4f}")
        cost_basis = qty * pos.entry_price + pos.entry_fee
        net_pnl = net_proceeds - cost_basis
        self.cash += net_proceeds
        self.realized_pnl += net_pnl
        self.n_trades += 1
        if net_pnl > 0:
            self.wins += 1
        del self.positions[symbol]
        record = {
            "symbol": symbol, "qty": qty, "side": "long",
            "entry_ts_ms": pos.entry_ts_ms, "exit_ts_ms": ts_ms,
            "entry_price": pos.entry_price, "exit_price": fill,
            "entry_fee": pos.entry_fee, "exit_fee": exit_fee,
            "gross_proceeds": gross_proceeds, "net_pnl": net_pnl,
            "exit_reason": exit_reason, "strategy": pos.strategy,
            "regime": pos.regime, "confidence": pos.confidence,
            "cash_after": self.cash,
        }
        self.trade_log.append(record)
        return record

    def snapshot(self, prices: Dict[str, float]) -> dict:
        eq = self.equity(prices)
        crypto_val = self.crypto_value(prices)
        return {
            "cash": self.cash, "equity": eq, "start_capital": self.start_capital,
            "return_pct": (eq / self.start_capital - 1) * 100 if self.start_capital else 0.0,
            "realized_pnl": self.realized_pnl, "n_trades": self.n_trades, "wins": self.wins,
            "peak_equity": self.peak_equity,
            "drawdown_pct": ((self.peak_equity - eq) / self.peak_equity * 100) if self.peak_equity else 0.0,
            "crypto_allocation_pct": (crypto_val / eq * 100) if eq else 0.0,
            "cash_allocation_pct": (self.cash / eq * 100) if eq else 0.0,
            "positions": {sym: {
                "qty": p.qty, "entry_price": p.entry_price, "current_price": prices.get(sym),
                "unrealized_pnl": p.qty * (prices.get(sym, p.entry_price) - p.entry_price),
                "stop_price": p.stop_price, "strategy": p.strategy, "regime": p.regime,
                "confidence": p.confidence, "entry_ts_ms": p.entry_ts_ms,
            } for sym, p in self.positions.items()},
        }
