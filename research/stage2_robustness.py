"""
Phase 6, Stage 2: robustness -- only Stage-1 survivors get a parameter grid.

Small grids (each family has 2-3 tunable knobs -- "avoid stacking many
indicators" applies to research scope too, not just the strategies
themselves). Full year, single fixed split (no train/val/test yet -- that's
Stage 3). Reports parameter-neighborhood stability the same way the Phase 2
Prime-Swing robustness check did: does the best combo sit in a region of
similarly-decent neighbors, or is it an isolated spike?
"""
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from research.data_cache import load_cached  # noqa: E402
from research.generic_runner import run_generic_backtest  # noqa: E402
from research.metrics import compute_metrics  # noqa: E402
from research.strategies.trend_momentum import DonchianTrendStrategy  # noqa: E402
from research.strategies.mean_reversion import ZScoreReversionStrategy  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TF = "15m"
HTF_BARS = 96

GRIDS = {
    "A_trend_donchian": {
        "cls": DonchianTrendStrategy,
        "grid": {"DONCHIAN_N": [10, 20, 30], "INIT_STOP_PCT": [2.0, 3.0, 5.0]},
    },
    "C_mean_reversion": {
        "cls": ZScoreReversionStrategy,
        "grid": {"ZSCORE_N": [10, 20, 30], "Z_THRESH": [1.5, 2.0, 2.5], "INIT_STOP_PCT": [2.0, 3.0]},
    },
}
SURVIVORS = {"A_trend_donchian": "ETH/USDT", "C_mean_reversion": "BTC/USDT"}


def run_grid(name, sym):
    spec = GRIDS[name]
    cls, grid = spec["cls"], spec["grid"]
    keys = list(grid.keys())
    combos = list(itertools.product(*grid.values()))
    chart = load_cached(sym, TF)

    results = []
    for combo in combos:
        params = dict(zip(keys, combo))
        trades, eq, guards, blocked, intervals = run_generic_backtest(
            cls, params, chart, start_capital=config.START_CAPITAL, symbol=sym, htf_bars=HTF_BARS)
        m = compute_metrics(trades, eq, config.START_CAPITAL, guards, real_times=intervals)
        results.append({"params": params, "ret": m.get("net_return_pct"),
                         "trades": m.get("total_trades", 0), "pf": m.get("profit_factor"),
                         "sharpe": m.get("sharpe_ratio_annualized")})

    results.sort(key=lambda r: (r["ret"] if r["ret"] is not None else -1e9), reverse=True)
    best = results[0]

    def neighbors(params):
        out = []
        for r in results:
            diffs = [k for k in keys if r["params"][k] != params[k]]
            if len(diffs) == 1:
                out.append(r)
        return out

    nbrs = neighbors(best["params"])
    nbr_rets = [n["ret"] for n in nbrs if n["ret"] is not None]
    return {
        "family": name, "symbol": sym, "n_combos": len(combos),
        "best": best,
        "neighbor_count": len(nbrs), "neighbor_mean_ret": (sum(nbr_rets) / len(nbr_rets)) if nbr_rets else None,
        "neighbor_min_ret": min(nbr_rets) if nbr_rets else None,
        "neighbor_positive_count": sum(1 for r in nbr_rets if r > 0),
        "isolated_spike": bool(nbr_rets and best["ret"] is not None and best["ret"] > 0
                                and (sum(nbr_rets) / len(nbr_rets)) < 0),
        "all_results": results,
    }


def main():
    out = {}
    for name, sym in SURVIVORS.items():
        print(f"[{name}][{sym}] running grid ({len(list(itertools.product(*GRIDS[name]['grid'].values())))} combos)…")
        r = run_grid(name, sym)
        out[f"{name}__{sym.replace('/', '')}"] = r
        print(f"  best: {r['best']['params']} ret={r['best']['ret']:.2f}% trades={r['best']['trades']} "
              f"pf={r['best']['pf']}")
        print(f"  neighbors: {r['neighbor_count']}, mean_ret={r['neighbor_mean_ret']}, "
              f"positive={r['neighbor_positive_count']}, isolated_spike={r['isolated_spike']}")

    os.makedirs(os.path.join(HERE, "reports"), exist_ok=True)
    with open(os.path.join(HERE, "reports", "stage2_robustness.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("\nwrote research/reports/stage2_robustness.json")


if __name__ == "__main__":
    main()
