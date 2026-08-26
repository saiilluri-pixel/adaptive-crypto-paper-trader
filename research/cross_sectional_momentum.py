"""
Hypothesis B: CROSS-SECTIONAL MOMENTUM (simplified)

At weekly rebalances, rank symbols ELIGIBLE ON THAT DATE (from
research/universe.py's point-in-time universe -- never using symbols that
only became liquid/listed later) by trailing 30-day risk-adjusted momentum
(return / volatility). Two variants: long-only top-K (simpler, tested per
the brief's "optionally long-only... as a separate hypothesis"), and a
reduced long-short (long top-K, short bottom-K, equal dollar-weighted --
not a fully risk-balanced market-neutral book, disclosed as a
simplification).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REBALANCE_DAYS = 7
LOOKBACK_DAYS = 30
TOP_K = 3


def _load_daily(sym):
    df = pd.read_csv(os.path.join(HERE, "data", f"{sym.replace('/', '')}_1d.csv"))
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    return df


def run_cross_sectional(symbols, eligible_days_by_symbol, mode="long_only",
                         start_capital=10_000.0, end_ts_ms=None, fee=None, slippage=None):
    fee = config.TAKER_FEE if fee is None else fee
    slippage = config.SLIPPAGE if slippage is None else slippage

    data = {s: _load_daily(s).set_index("ts") for s in symbols}
    eligible_set = {s: set(v) for s, v in eligible_days_by_symbol.items()}
    all_ts = sorted(set.union(*[set(d.index) for d in data.values()]))
    if end_ts_ms:
        all_ts = [t for t in all_ts if t < end_ts_ms]

    equity = start_capital
    equity_curve = []
    rebal_log = []
    holdings = {}   # symbol -> weight
    last_rebal_i = -REBALANCE_DAYS

    for i, ts in enumerate(all_ts):
        # mark-to-market
        if holdings:
            day_ret = 0.0
            for s, w in holdings.items():
                if ts in data[s].index and (i - 1) >= 0 and all_ts[i - 1] in data[s].index:
                    p0 = data[s].loc[all_ts[i - 1], "close"]
                    p1 = data[s].loc[ts, "close"]
                    day_ret += w * (p1 / p0 - 1)
            equity *= (1 + day_ret)

        if i - last_rebal_i >= REBALANCE_DAYS and i >= LOOKBACK_DAYS:
            eligible_today = [s for s in symbols if ts in eligible_set.get(s, set())
                               and ts in data[s].index]
            scored = []
            for s in eligible_today:
                idx = data[s].index.get_loc(ts)
                if idx < LOOKBACK_DAYS:
                    continue
                window = data[s]["close"].iloc[idx - LOOKBACK_DAYS:idx + 1]
                rets = window.pct_change().dropna()
                if rets.std() == 0 or len(rets) < LOOKBACK_DAYS * 0.8:
                    continue
                mom = (window.iloc[-1] / window.iloc[0] - 1) / rets.std()
                scored.append((s, mom))
            scored.sort(key=lambda x: -x[1])

            new_holdings = {}
            if len(scored) >= TOP_K:
                if mode == "long_only":
                    for s, _ in scored[:TOP_K]:
                        new_holdings[s] = 1.0 / TOP_K
                elif mode == "long_short" and len(scored) >= 2 * TOP_K:
                    for s, _ in scored[:TOP_K]:
                        new_holdings[s] = 0.5 / TOP_K
                    for s, _ in scored[-TOP_K:]:
                        new_holdings[s] = -0.5 / TOP_K

            # turnover cost: sum of |weight changes| * (fee+slippage), both directions
            all_syms = set(holdings) | set(new_holdings)
            turnover = sum(abs(new_holdings.get(s, 0) - holdings.get(s, 0)) for s in all_syms)
            cost = equity * turnover * (fee + slippage)
            equity -= cost
            rebal_log.append({"ts": int(ts), "n_eligible": len(eligible_today),
                               "n_scored": len(scored), "holdings": new_holdings, "turnover": turnover,
                               "cost": cost})
            holdings = new_holdings
            last_rebal_i = i

        equity_curve.append((int(ts), equity))

    return equity_curve, rebal_log


def main():
    universe = json.load(open(os.path.join(HERE, "reports", "universe.json")))
    symbols = list(universe["eligible_days_by_symbol"].keys())
    eligible = universe["eligible_days_by_symbol"]

    from research.holdout import HOLDOUT_START_TS_MS
    for mode in ("long_only", "long_short"):
        eq, log = run_cross_sectional(symbols, eligible, mode=mode, end_ts_ms=HOLDOUT_START_TS_MS)
        ret = (eq[-1][1] / 10_000.0 - 1) * 100
        avg_turnover = sum(r["turnover"] for r in log) / len(log) if log else 0
        print(f"{mode:12s} final_equity={eq[-1][1]:.2f} ret={ret:.2f}% rebalances={len(log)} "
              f"avg_turnover={avg_turnover:.2f}")


if __name__ == "__main__":
    main()
