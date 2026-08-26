"""
Phase 2, steps 8 & 9: regime breakdown + portfolio/correlation view.

Reruns the LEGACY_TUNED_BASELINE (symbol_params.json, as currently live)
once per symbol to get raw trades/equity/position-state, then:
  - tags trades by simple trend/vol regime and sums PnL per regime
  - builds a normalized combined BTC+ETH+SOL portfolio equity curve
  - reports simultaneous-direction exposure (correlation risk)

Research only -- does not touch symbol_params.json or the live bot.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from research.data_cache import load_cached  # noqa: E402
from research.engine_runner import run_backtest  # noqa: E402
from research.regime import classify_regimes, pnl_by_regime  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    per_symbol = {}
    for sym in config.SYMBOLS:
        struct = load_cached(sym, config.STRUCTURE_TF)
        chart = load_cached(sym, config.CHART_TF)
        params = config.params_for(sym)
        trades, equity_curve, guards, blocked, pos_state, trade_intervals = run_backtest(
            sym, params, struct, chart, start_capital=config.START_CAPITAL)
        regimes = classify_regimes(chart)
        breakdown = pnl_by_regime(equity_curve, regimes)
        per_symbol[sym] = {
            "trades": trades, "equity_curve": equity_curve,
            "position_state": pos_state, "regime_breakdown": breakdown,
        }
        print(f"[{sym}] {len(trades)} trades, {len(equity_curve)} bars attributed by regime")

    # ── regime report ──
    regime_report = {sym: d["regime_breakdown"] for sym, d in per_symbol.items()}

    # ── portfolio view: normalize each symbol's equity curve to % return,
    #    align on shared timestamps, average for a simple equal-weight
    #    combined curve (NOT a shared-margin simulator -- purely descriptive) ──
    dfs = []
    for sym, d in per_symbol.items():
        s = pd.DataFrame(d["equity_curve"], columns=["ts", "equity"]).set_index("ts")
        s["ret_pct"] = (s["equity"] / config.START_CAPITAL - 1) * 100
        dfs.append(s["ret_pct"].rename(sym))
    combined = pd.concat(dfs, axis=1).sort_index().ffill().dropna()
    combined["portfolio_avg_ret_pct"] = combined.mean(axis=1)

    peak = combined["portfolio_avg_ret_pct"].cummax()
    port_dd = (combined["portfolio_avg_ret_pct"] - peak)
    combined_max_dd = float(port_dd.min())

    # ── simultaneous-direction exposure ──
    pos_dfs = []
    for sym, d in per_symbol.items():
        p = pd.DataFrame(d["position_state"], columns=["ts", "side"]).set_index("ts")
        pos_dfs.append(p["side"].rename(sym))
    pos_combined = pd.concat(pos_dfs, axis=1).sort_index().ffill()
    all_long = int(((pos_combined == "long").all(axis=1)).sum())
    all_short = int(((pos_combined == "short").all(axis=1)).sum())
    any_open = pos_combined.notna().any(axis=1)
    two_plus_same = 0
    for _, row in pos_combined[any_open].iterrows():
        sides = [v for v in row.values if v is not None and not pd.isna(v)]
        if len(sides) >= 2 and len(set(sides)) == 1:
            two_plus_same += 1
    total_bars = len(pos_combined)

    portfolio_report = {
        "combined_return_pct_final": float(combined["portfolio_avg_ret_pct"].iloc[-1]),
        "combined_max_drawdown_pct": combined_max_dd,
        "bars_all_symbols_long_simultaneously": all_long,
        "bars_all_symbols_short_simultaneously": all_short,
        "bars_2plus_symbols_same_direction": two_plus_same,
        "total_bars": total_bars,
        "pct_bars_2plus_same_direction": float(two_plus_same / total_bars * 100) if total_bars else None,
        "note": "descriptive/normalized view only -- NOT a shared-margin production simulator",
    }

    os.makedirs(os.path.join(HERE, "reports"), exist_ok=True)
    with open(os.path.join(HERE, "reports", "regime.json"), "w") as f:
        json.dump(regime_report, f, indent=2, default=str)
    with open(os.path.join(HERE, "reports", "portfolio.json"), "w") as f:
        json.dump(portfolio_report, f, indent=2, default=str)
    print(json.dumps(portfolio_report, indent=2, default=str))
    print("\nwrote research/reports/regime.json and portfolio.json")


if __name__ == "__main__":
    main()
