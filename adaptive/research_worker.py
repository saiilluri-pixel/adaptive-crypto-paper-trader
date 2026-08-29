"""
Separate background research process (adaptive spec section 12): never
called synchronously from the live runner's decision loop -- invoked as
its own periodic job on a slow cadence (e.g. once per several hours), read
its own historical data independently, and hands results to
adaptive/adaptation.py's AdaptationEngine for a promotion decision.

Walk-forward-style challenger evaluation: splits recent historical candles
into several non-overlapping windows and runs a strategy from
adaptive/strategies.py -- the SAME functions the live system calls, not a
separate reimplementation -- causally across each window with a candidate
parameter set, producing the EvaluationResult/WindowResult objects
adaptation.py's promotion rules consume.

Scoping note: for RELATIVE comparison of parameter sets (is this candidate
better than the current champion?), trades are simulated signal-to-signal
(enter on "long", exit on the next "exit_long") rather than replaying the
exact live trailing-stop mechanics from adaptive/trailing.py -- adequate
for ranking parameter sets against each other, not a precise replica of
live fills. Fees and slippage are still applied, so a candidate that only
wins before costs cannot be promoted. shock_continuation is never a
research-worker target (see adaptive/adaptation.py's NON_ADAPTIVE_STRATEGIES).
"""
import os
import sys
from typing import Dict, List, Optional, Tuple

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from adaptive.strategies import trend_momentum, volatility_breakout, mean_reversion  # noqa: E402
from adaptive.adaptation import (  # noqa: E402
    WindowResult, EvaluationResult, AdaptationEngine, NON_ADAPTIVE_STRATEGIES,
)

STRATEGY_FNS = {
    "trend_momentum": trend_momentum,
    "volatility_breakout": volatility_breakout,
    "mean_reversion": mean_reversion,
}

DEFAULT_N_WINDOWS = 4
DEFAULT_WINDOW_BARS = 300
DEFAULT_MIN_WARMUP_BARS = 30


def simulate_window(strategy_name: str, df: pd.DataFrame, params: dict,
                     fee_rate: float, slippage: float,
                     min_warmup_bars: int = DEFAULT_MIN_WARMUP_BARS) -> List[float]:
    """Signal-to-signal simulation over one window. Returns a list of net
    %-return-of-entry-price trades, cost-inclusive."""
    fn = STRATEGY_FNS[strategy_name]
    trades: List[float] = []
    open_entry: Optional[Tuple[float, float]] = None  # (entry_fill_price, entry_fee)
    for i in range(min_warmup_bars, len(df)):
        window_df = df.iloc[:i + 1]
        sig = fn(window_df, params=params)
        price = float(window_df["close"].iloc[-1])
        if open_entry is None and sig.direction == "long":
            fill = price * (1 + slippage)
            fee = fill * fee_rate
            open_entry = (fill, fee)
        elif open_entry is not None and sig.direction == "exit_long":
            entry_fill, entry_fee = open_entry
            exit_fill = price * (1 - slippage)
            exit_fee = exit_fill * fee_rate
            net = (exit_fill - exit_fee - entry_fill - entry_fee) / entry_fill * 100
            trades.append(net)
            open_entry = None
    return trades


def evaluate_challenger(strategy_name: str, historical_df: pd.DataFrame,
                         champion_params: dict, challenger_params: dict,
                         fee_rate: float, slippage: float,
                         n_windows: int = DEFAULT_N_WINDOWS,
                         window_bars: int = DEFAULT_WINDOW_BARS
                         ) -> Tuple[EvaluationResult, EvaluationResult]:
    total = len(historical_df)
    usable = min(total, n_windows * window_bars)
    start = total - usable
    champ_windows, chal_windows = [], []
    for w in range(n_windows):
        s, e = start + w * window_bars, start + (w + 1) * window_bars
        if e > total:
            break
        wdf = historical_df.iloc[s:e].reset_index(drop=True)
        champ_windows.append(WindowResult(f"w{w}", simulate_window(
            strategy_name, wdf, champion_params, fee_rate, slippage)))
        chal_windows.append(WindowResult(f"w{w}", simulate_window(
            strategy_name, wdf, challenger_params, fee_rate, slippage)))
    champ_eval = EvaluationResult(strategy_name, "champion", champion_params, champ_windows)
    chal_eval = EvaluationResult(strategy_name, "challenger", challenger_params, chal_windows)
    return champ_eval, chal_eval


def generate_challenger_candidates(strategy_name: str, base_params: dict) -> List[dict]:
    """Small, deliberately conservative neighborhood search around the
    current champion -- perturbs one parameter at a time by +-30%, never a
    wholesale reparameterization. Not randomly optimized every cycle (per
    spec section 12): this is called at most once per research cycle."""
    if strategy_name == "trend_momentum":
        base = base_params.get("z_threshold", 1.0)
        return [{**base_params, "z_threshold": base * 0.7},
                {**base_params, "z_threshold": base * 1.3}]
    if strategy_name == "volatility_breakout":
        n = base_params.get("n", 20)
        return [{**base_params, "n": max(5, int(n * 0.7))},
                {**base_params, "n": int(n * 1.3)}]
    if strategy_name == "mean_reversion":
        z = base_params.get("z_entry", -1.5)
        return [{**base_params, "z_entry": z * 0.7},
                {**base_params, "z_entry": z * 1.3}]
    return []


def run_research_cycle(*, historical_data_by_symbol: Dict[str, pd.DataFrame],
                        adaptation_engine: AdaptationEngine, default_params: Dict[str, dict],
                        fee_rate: float, slippage: float, now_ts: Optional[float] = None) -> List[dict]:
    """One research cycle across every adaptive strategy x every symbol
    with historical data supplied. Returns a list of {strategy, symbol,
    promoted, reasons} summaries. Never touches live portfolio/positions --
    purely reads historical data and writes to adaptation_engine's own
    champion registry + adaptation_log.jsonl."""
    results = []
    for strategy_name in STRATEGY_FNS:
        if strategy_name in NON_ADAPTIVE_STRATEGIES:
            continue
        champion_params = adaptation_engine.get_champion_params(
            strategy_name, default_params.get(strategy_name, {}))
        for symbol, df in historical_data_by_symbol.items():
            if len(df) < DEFAULT_WINDOW_BARS:
                continue
            for challenger_params in generate_challenger_candidates(strategy_name, champion_params):
                champ_eval, chal_eval = evaluate_challenger(
                    strategy_name, df, champion_params, challenger_params, fee_rate, slippage)
                promoted = adaptation_engine.consider_promotion(
                    strategy_name, champ_eval, chal_eval, challenger_params, now_ts=now_ts)
                results.append({"strategy": strategy_name, "symbol": symbol,
                                 "challenger_params": challenger_params, "promoted": promoted})
                if promoted:
                    champion_params = challenger_params  # subsequent candidates in this cycle
                    # compare against the newly-promoted champion, not the stale one
    return results
